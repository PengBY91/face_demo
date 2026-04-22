# 批量推理吞吐率优化 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将多路摄像头的独立 GPU Session 合并为单个共享 FaceEngine + InferenceWorker 批量推理，将识别吞吐率提升 3-5x。

**Architecture:** 新增 `InferenceWorker` 线程收集所有摄像头帧，组 batch 后一次 GPU 调用，结果通过每路摄像头专属的 `result_queue` 非阻塞分发回 `CameraThread`。`CameraThread` 读帧与推理完全解耦，RTSP 缓冲永不积压。

**Tech Stack:** Python threading, queue.Queue, InsightFace ArcFace ONNX（`rec_model.get_feat` 原生支持 list 输入批量推理），OpenCV RTSP

---

## 文件改动清单

| 文件 | 改动类型 |
|------|---------|
| `utils/face_engine.py` | 新增 `batch_detect_and_extract` 方法 |
| `multi_camera_demo.py` | 新增 `InferenceWorker` 类；修改 `CameraThread.__init__` 和 `run`；修改 `MultiCameraApp.__init__` 和 `_start_cameras` |
| `tests/test_batch_inference.py` | 新建，单元测试 |

---

## Task 1: 新增 `FaceEngine.batch_detect_and_extract`

**Files:**
- Modify: `utils/face_engine.py`
- Test: `tests/test_batch_inference.py`

InsightFace 的 `rec_model.get_feat(imgs)` 原生接受 list 输入，内部用 `cv2.dnn.blobFromImages` 组 batch，一次 ONNX session 调用返回 `(N, 512)` 的 embedding 数组。这是最大的 GPU 吞吐收益点。

- [ ] **Step 1: 新建测试文件，写失败测试**

创建 `tests/test_batch_inference.py`：

```python
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from utils.face_engine import FaceEngine


def _make_engine():
    """创建带 mock 模型的 FaceEngine，无需真实 GPU"""
    engine = FaceEngine.__new__(FaceEngine)
    engine.det_thresh = 0.5

    mock_face = MagicMock()
    mock_face.det_score = 0.9
    mock_face.bbox = np.array([10.0, 10.0, 50.0, 50.0])
    mock_face.kps = np.zeros((5, 2), dtype=np.float32)

    engine.det_model = MagicMock()
    engine.det_model.get.return_value = [mock_face]

    engine.rec_model = MagicMock()
    # 返回 (N, 512)，N = 调用时传入的图片数量
    engine.rec_model.get_feat.side_effect = lambda imgs: np.random.rand(len(imgs), 512)

    return engine


def test_batch_returns_one_list_per_frame():
    engine = _make_engine()
    frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(3)]

    with patch('utils.face_engine.face_align') as mock_align:
        mock_align.norm_crop.return_value = np.zeros((112, 112, 3), dtype=np.uint8)
        results = engine.batch_detect_and_extract(frames)

    assert len(results) == 3


def test_batch_calls_get_feat_once_for_all_faces():
    engine = _make_engine()
    frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(3)]

    with patch('utils.face_engine.face_align') as mock_align:
        mock_align.norm_crop.return_value = np.zeros((112, 112, 3), dtype=np.uint8)
        engine.batch_detect_and_extract(frames)

    # 3帧各1张脸 = 一次 get_feat 调用，传入列表长度为3
    engine.rec_model.get_feat.assert_called_once()
    call_arg = engine.rec_model.get_feat.call_args[0][0]
    assert len(call_arg) == 3


def test_batch_empty_frames_returns_empty_lists():
    engine = _make_engine()
    engine.det_model.get.return_value = []  # 无人脸
    frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(2)]

    with patch('utils.face_engine.face_align'):
        results = engine.batch_detect_and_extract(frames)

    assert results == [[], []]
    engine.rec_model.get_feat.assert_not_called()


def test_batch_face_dict_has_required_keys():
    engine = _make_engine()
    frames = [np.zeros((480, 640, 3), dtype=np.uint8)]

    with patch('utils.face_engine.face_align') as mock_align:
        mock_align.norm_crop.return_value = np.zeros((112, 112, 3), dtype=np.uint8)
        results = engine.batch_detect_and_extract(frames)

    face = results[0][0]
    for key in ('bbox', 'landmarks', 'embedding', 'aligned_face', 'det_score'):
        assert key in face, f"missing key: {key}"
    assert face['embedding'].shape == (512,)
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
cd /home/steve/amhs/code/face_demo
python -m pytest tests/test_batch_inference.py -v 2>&1 | head -30
```

预期：`AttributeError: type object 'FaceEngine' has no attribute 'batch_detect_and_extract'`

- [ ] **Step 3: 在 `utils/face_engine.py` 末尾（`_bbox_area` 之前）添加方法**

在 `utils/face_engine.py` 的 `get_embedding` 方法之后、`_bbox_area` 之前插入：

```python
def batch_detect_and_extract(self, frames: List[np.ndarray]) -> List[List[Dict]]:
    """
    批量处理多帧，一次 GPU 调用提取所有人脸特征。

    Args:
        frames: BGR 图像列表

    Returns:
        与 frames 等长的列表，每个元素是该帧检测到的人脸列表。
        人脸 dict 结构与 detect_and_extract 相同：
        bbox, landmarks, embedding, aligned_face, det_score
    """
    results: List[List[Dict]] = [[] for _ in frames]
    all_aligned: List[np.ndarray] = []
    face_map: List[tuple] = []  # (frame_idx, partial_face_dict)

    # 第一步：逐帧检测，收集对齐人脸
    for frame_idx, frame in enumerate(frames):
        faces = self.det_model.get(frame)
        for face in faces:
            if face.det_score < self.det_thresh:
                continue
            bbox = face.bbox.astype(int)
            landmarks = face.kps
            aligned = face_align.norm_crop(frame, landmarks, image_size=112)
            all_aligned.append(aligned)
            face_map.append((frame_idx, {
                'bbox': bbox.tolist(),
                'landmarks': landmarks.astype(int).tolist(),
                'aligned_face': aligned,
                'det_score': float(face.det_score),
            }))

    if not all_aligned:
        return results

    # 第二步：一次 batch 调用提取所有特征
    embeddings = self.rec_model.get_feat(all_aligned)  # shape: (N, 512)

    # 第三步：分发 embedding 到对应帧
    for i, (frame_idx, face_dict) in enumerate(face_map):
        face_dict['embedding'] = embeddings[i].flatten()
        results[frame_idx].append(face_dict)

    return results
```

- [ ] **Step 4: 运行测试，确认全部通过**

```bash
python -m pytest tests/test_batch_inference.py -v
```

预期输出：
```
PASSED tests/test_batch_inference.py::test_batch_returns_one_list_per_frame
PASSED tests/test_batch_inference.py::test_batch_calls_get_feat_once_for_all_faces
PASSED tests/test_batch_inference.py::test_batch_empty_frames_returns_empty_lists
PASSED tests/test_batch_inference.py::test_batch_face_dict_has_required_keys
4 passed
```

- [ ] **Step 5: 提交**

```bash
git add utils/face_engine.py tests/test_batch_inference.py
git commit -m "feat: add FaceEngine.batch_detect_and_extract for GPU batch throughput"
```

---

## Task 2: 新增 `InferenceWorker` 类

**Files:**
- Modify: `multi_camera_demo.py`（在 `DetectionRecord` 类定义之后、`CameraThread` 类定义之前插入）
- Test: `tests/test_batch_inference.py`（追加）

- [ ] **Step 1: 在测试文件末尾追加 InferenceWorker 测试**

在 `tests/test_batch_inference.py` 末尾追加：

```python
import queue
import threading
import time
from multi_camera_demo import InferenceWorker


def _make_inference_worker(batch_window_ms=10):
    """创建带 mock engine 的 InferenceWorker"""
    mock_engine = MagicMock()
    mock_engine.batch_detect_and_extract.return_value = [[]]  # 空人脸列表

    infer_queue = queue.Queue(maxsize=16)
    result_queues = {
        'cam_01': queue.Queue(maxsize=2),
        'cam_02': queue.Queue(maxsize=2),
    }
    worker = InferenceWorker(mock_engine, infer_queue, result_queues)
    worker.BATCH_WINDOW_MS = batch_window_ms
    return worker, mock_engine, infer_queue, result_queues


def test_inference_worker_routes_result_to_correct_camera():
    worker, mock_engine, infer_queue, result_queues = _make_inference_worker()
    mock_engine.batch_detect_and_extract.return_value = [[{'embedding': np.zeros(512),
                                                           'bbox': [0,0,1,1],
                                                           'landmarks': [],
                                                           'aligned_face': np.zeros((112,112,3)),
                                                           'det_score': 0.9}]]

    worker.start()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    infer_queue.put(('cam_01', 1, frame))
    time.sleep(0.1)  # 等待 worker 处理
    worker.stop()
    worker.join(timeout=1.0)

    assert not result_queues['cam_01'].empty(), "cam_01 应收到结果"
    assert result_queues['cam_02'].empty(), "cam_02 不应收到结果"


def test_inference_worker_deduplicates_same_camera_in_batch():
    """同一摄像头的多帧提交，worker 只取最新一帧组 batch"""
    worker, mock_engine, infer_queue, result_queues = _make_inference_worker(batch_window_ms=50)
    mock_engine.batch_detect_and_extract.return_value = [[]]

    worker.start()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # 快速连续提交同一摄像头3帧
    for fid in range(3):
        try:
            infer_queue.put_nowait(('cam_01', fid, frame))
        except queue.Full:
            pass
    time.sleep(0.15)
    worker.stop()
    worker.join(timeout=1.0)

    # batch_detect_and_extract 每次调用时，cam_01 只出现一帧
    for call in mock_engine.batch_detect_and_extract.call_args_list:
        frames_arg = call[0][0]
        assert len(frames_arg) <= 1, "同一 cam_id 在一个 batch 内应只有 1 帧"
```

- [ ] **Step 2: 运行，确认测试失败**

```bash
python -m pytest tests/test_batch_inference.py::test_inference_worker_routes_result_to_correct_camera -v 2>&1 | head -20
```

预期：`ImportError: cannot import name 'InferenceWorker' from 'multi_camera_demo'`

- [ ] **Step 3: 在 `multi_camera_demo.py` 中添加 `InferenceWorker` 类**

在 `multi_camera_demo.py` 的 `class DetectionRecord:` 定义结束后（约第 145 行）、`class CameraThread:` 开始之前插入：

```python
class InferenceWorker(threading.Thread):
    """
    批量推理工作线程。
    收集所有摄像头的帧，组 batch 后一次 GPU 调用，结果分发回各摄像头的 result_queue。
    """
    BATCH_WINDOW_MS = 40   # 时间窗（ms），超时强制触发推理
    MAX_BATCH_SIZE  = 8    # 单次 batch 上限

    def __init__(self, face_engine, infer_queue: queue.Queue,
                 result_queues: Dict[str, queue.Queue]):
        super().__init__(daemon=True)
        self.face_engine   = face_engine
        self.infer_queue   = infer_queue
        self.result_queues = result_queues
        self._running      = True

    def run(self):
        while self._running:
            batch: Dict[str, tuple] = {}  # {cam_id: (frame_id, frame)}，每路最多1帧
            deadline = time.time() + self.BATCH_WINDOW_MS / 1000.0

            # 收集帧：时间窗结束或达到上限时停止
            while time.time() < deadline and len(batch) < self.MAX_BATCH_SIZE:
                remaining = max(0.001, deadline - time.time())
                try:
                    cam_id, frame_id, frame = self.infer_queue.get(timeout=remaining)
                    batch[cam_id] = (frame_id, frame)  # 同 cam_id 新帧覆盖旧帧
                except queue.Empty:
                    break

            if not batch:
                continue

            cam_ids  = list(batch.keys())
            frames   = [batch[cid][1] for cid in cam_ids]
            frame_ids = [batch[cid][0] for cid in cam_ids]

            try:
                all_results = self.face_engine.batch_detect_and_extract(frames)
            except Exception as e:
                print(f"InferenceWorker: 推理异常: {e}")
                continue

            # 分发结果到各摄像头的 result_queue（清旧保新）
            for cam_id, frame_id, faces in zip(cam_ids, frame_ids, all_results):
                rq = self.result_queues.get(cam_id)
                if rq is None:
                    continue
                try:
                    rq.get_nowait()   # 丢弃旧结果
                except queue.Empty:
                    pass
                try:
                    rq.put_nowait((frame_id, faces))
                except queue.Full:
                    pass

    def stop(self):
        self._running = False
```

- [ ] **Step 4: 运行所有推理测试，确认通过**

```bash
python -m pytest tests/test_batch_inference.py -v
```

预期：所有 6 个测试通过。

- [ ] **Step 5: 提交**

```bash
git add multi_camera_demo.py tests/test_batch_inference.py
git commit -m "feat: add InferenceWorker for batched multi-camera GPU inference"
```

---

## Task 3: 修改 `CameraThread` — 去掉 FaceEngine，接入推理队列

**Files:**
- Modify: `multi_camera_demo.py`（`CameraThread` 类）

- [ ] **Step 1: 修改 `CameraThread.__init__` 签名和初始化**

将 `CameraThread.__init__` 的参数列表从：
```python
def __init__(self, camera_config: dict, gallery: GalleryManager,
             detection_queue: queue.Queue, feature_db: dict,
             feature_lock: threading.Lock):
```
改为：
```python
def __init__(self, camera_config: dict, gallery: GalleryManager,
             detection_queue: queue.Queue, feature_db: dict,
             feature_lock: threading.Lock,
             infer_queue: queue.Queue, result_queue: queue.Queue):
```

在 `__init__` 中**删除**以下属性：
```python
# 删除这些行：
self.engine = None
self._engine_initialized = False
self.skip_frames = SKIP_FRAMES
self.frame_index = 0
self.cached_detections = []
```

**新增**以下属性（放在 `self.session = requests.Session()` 之后）：
```python
self.infer_queue  = infer_queue
self.result_queue = result_queue
self.frame_index  = 0           # 单调递增帧计数，用作 frame_id
self.cached_detections = []     # 缓存最新推理结果，推理未完成时继续使用
```

- [ ] **Step 2: 删除 `engine_initialized` property 和 setter**

在 `CameraThread` 中找到并**完整删除**以下代码块（约 257-265 行）：
```python
@property
def engine_initialized(self):
    with self._state_lock:
        return self._engine_initialized

@engine_initialized.setter
def engine_initialized(self, value):
    with self._state_lock:
        self._engine_initialized = value
```

- [ ] **Step 3: 修改 `run` 方法 — 移除 FaceEngine 初始化块**

在 `run` 方法中，找到并**完整删除** FaceEngine 初始化代码块：
```python
# 删除这整块（约 340-353 行）：
if not self.engine_initialized:
    print(f"CameraThread [{self.camera_name}]: 初始化 FaceEngine...")
    try:
        self.engine = FaceEngine(
            rec_model_path=ARCFACE_MODEL_PATH,
            providers=PROVIDERS,
            det_thresh=DET_THRESH
        )
        self.engine_initialized = True
        print(f"CameraThread [{self.camera_name}]: FaceEngine 初始化完成")
    except Exception as e:
        print(f"CameraThread [{self.camera_name}]: FaceEngine 初始化失败: {e}")
        self.running = False
        return
```

- [ ] **Step 4: 修改主读取循环中的推理逻辑**

在主读取循环中，找到以下代码块（约 460-503 行）：
```python
# 跳帧识别：每隔 skip_frames 帧进行一次识别
# 中间帧使用缓存的识别结果进行显示
should_detect = (self.frame_index % (self.skip_frames + 1) == 0)

# 应用畸变校正
frame = self._apply_undistort(frame)

if should_detect:
    # 进行人脸检测和识别
    try:
        faces = self.engine.detect_and_extract(frame)
        detections = []

        for face in faces:
            bbox = face['bbox']
            embedding = face['embedding']
            aligned_face = face['aligned_face']

            # 识别
            name, score, is_suspicious = self._identify(embedding)

            # 记录检测结果
            detection = {
                'bbox': bbox,
                'name': name,
                'score': score,
                'is_suspicious': is_suspicious,
                'aligned_face': aligned_face
            }
            detections.append(detection)

            # 生成识别记录（非 Unknown 且非疑似）
            if name != "Unknown" and not is_suspicious:
                self._process_detection(name, score, aligned_face)

        # 更新缓存
        self.cached_detections = detections
        self.latest_detections = detections

    except Exception as e:
        print(f"CameraThread [{self.camera_name}]: 处理异常: {e}")
else:
    # 使用缓存的识别结果（插值显示）
    self.latest_detections = self.cached_detections
```

**替换为**：
```python
# 应用畸变校正
frame = self._apply_undistort(frame)

# 提交帧到推理队列（非阻塞，队列满时跳过本帧）
self.frame_index += 1
try:
    self.infer_queue.put_nowait((self.camera_id, self.frame_index, frame))
except queue.Full:
    pass

# 非阻塞取推理结果，有结果则更新缓存，无结果则继续用缓存
try:
    _, faces = self.result_queue.get_nowait()
    detections = []
    for face in faces:
        name, score, is_suspicious = self._identify(face['embedding'])
        detections.append({
            'bbox': face['bbox'],
            'name': name,
            'score': score,
            'is_suspicious': is_suspicious,
            'aligned_face': face['aligned_face']
        })
        if name != "Unknown" and not is_suspicious:
            self._process_detection(name, score, face['aligned_face'])
    self.cached_detections = detections
except queue.Empty:
    pass

self.latest_detections = self.cached_detections
```

- [ ] **Step 5: 确认 `from config import` 中移除不再需要的导入**

检查 `multi_camera_demo.py` 顶部的 `from config import` 行，确认 `ARCFACE_MODEL_PATH`、`PROVIDERS`、`DET_THRESH` 已从 `CameraThread` 中不再使用（它们将在 `MultiCameraApp` 中用于创建共享 FaceEngine，保留导入）。

同时从 `from configs.cameras import` 中删除已不再使用的 `SKIP_FRAMES`：
```python
# 删除此行中的 SKIP_FRAMES：
SKIP_FRAMES,
```

- [ ] **Step 6: 验证语法正确**

```bash
cd /home/steve/amhs/code/face_demo
python -c "from multi_camera_demo import CameraThread, InferenceWorker; print('OK')"
```

预期输出：`OK`

- [ ] **Step 7: 提交**

```bash
git add multi_camera_demo.py
git commit -m "refactor: CameraThread delegates GPU inference to InferenceWorker"
```

---

## Task 4: 修改 `MultiCameraApp` — 创建共享 FaceEngine 并串联 InferenceWorker

**Files:**
- Modify: `multi_camera_demo.py`（`MultiCameraApp.__init__` 和 `_start_cameras`）

- [ ] **Step 1: 在 `MultiCameraApp.__init__` 中添加共享 FaceEngine 和 InferenceWorker**

在 `MultiCameraApp.__init__` 中，找到 `self._start_cameras()` 这一行，在其**之前**插入：

```python
# 共享 FaceEngine（单个 GPU Session，所有摄像头共用）
print("MultiCameraApp: 初始化共享 FaceEngine...")
self.shared_engine = FaceEngine(
    rec_model_path=ARCFACE_MODEL_PATH,
    providers=PROVIDERS,
    det_thresh=DET_THRESH
)
print("MultiCameraApp: 共享 FaceEngine 初始化完成")

# 推理队列和每路摄像头的结果队列
cameras = get_enabled_cameras()
self.infer_queue   = queue.Queue(maxsize=len(cameras) * 4)
self.result_queues = {cam['id']: queue.Queue(maxsize=2) for cam in cameras}

# 启动 InferenceWorker
self.infer_worker = InferenceWorker(
    self.shared_engine, self.infer_queue, self.result_queues
)
self.infer_worker.start()
print("MultiCameraApp: InferenceWorker 已启动")
```

- [ ] **Step 2: 修改 `_start_cameras`，给 CameraThread 传入队列参数**

找到 `_start_cameras` 方法中创建 `CameraThread` 的代码：
```python
thread = CameraThread(
    camera_config=cam_config,
    gallery=self.gallery,
    detection_queue=self.detection_queue,
    feature_db=self.feature_db,
    feature_lock=self.feature_lock
)
```

替换为：
```python
thread = CameraThread(
    camera_config=cam_config,
    gallery=self.gallery,
    detection_queue=self.detection_queue,
    feature_db=self.feature_db,
    feature_lock=self.feature_lock,
    infer_queue=self.infer_queue,
    result_queue=self.result_queues[cam_config['id']]
)
```

同时**删除**启动间隔（原本是为了错开 GPU 初始化，现在不再需要）：
```python
# 删除这4行：
# 错开启动，避免同时初始化 GPU 资源
if i < len(cameras) - 1:
    time.sleep(0.5)
```

- [ ] **Step 3: 在 `run` 方法的清理代码中停止 InferenceWorker**

在 `run` 方法末尾的清理部分，找到：
```python
# 清理
for thread in self.camera_threads:
    thread.stop()
cv2.destroyAllWindows()
```

替换为：
```python
# 清理
for thread in self.camera_threads:
    thread.stop()
self.infer_worker.stop()
cv2.destroyAllWindows()
```

- [ ] **Step 4: 验证整体导入和实例化无错误**

```bash
cd /home/steve/amhs/code/face_demo
python -c "
from multi_camera_demo import MultiCameraApp, InferenceWorker, CameraThread
import inspect
sig = inspect.signature(CameraThread.__init__)
params = list(sig.parameters.keys())
assert 'infer_queue' in params, 'infer_queue missing'
assert 'result_queue' in params, 'result_queue missing'
assert 'engine' not in params, 'engine should be removed'
print('参数签名验证通过:', params)
"
```

预期输出：`参数签名验证通过: ['self', 'camera_config', 'gallery', 'detection_queue', 'feature_db', 'feature_lock', 'infer_queue', 'result_queue']`

- [ ] **Step 5: 运行全部测试确认无回归**

```bash
python -m pytest tests/ -v
```

预期：所有测试通过，无新失败。

- [ ] **Step 6: 提交**

```bash
git add multi_camera_demo.py
git commit -m "feat: wire shared FaceEngine + InferenceWorker into MultiCameraApp"
```

---

## Task 5: 验收测试 — 确认吞吐率提升

**Files:**
- 无代码改动，仅运行验证

- [ ] **Step 1: 启动系统并观察 GPU 利用率**

打开一个终端，运行：
```bash
watch -n 1 nvidia-smi
```

另一个终端启动应用：
```bash
cd /home/steve/amhs/code/face_demo
python multi_camera_demo.py
```

预期：`nvidia-smi` 中 GPU Util 从原来的 <30% 提升到 60-80%。

- [ ] **Step 2: 观察控制台识别日志频率**

观察 `[识别]` 日志输出频率，对比优化前应明显加快。FPS 显示（窗口右上角/状态栏）应从 10-15 提升到 25-40。

- [ ] **Step 3: 可选 — 将 `SKIP_FRAMES` 降为 0**

如果 GPU 利用率有余量，可进一步提升识别帧率：

编辑 `configs/cameras.py`：
```python
# 修改前：
SKIP_FRAMES = 2
# 修改后：
SKIP_FRAMES = 0
```

注意：`SKIP_FRAMES` 已从 `CameraThread` 的逻辑中移除，此配置项已无效（可直接删除或留作注释）。重启后 InferenceWorker 的时间窗机制自动控制吞吐节奏，无需手动跳帧。

- [ ] **Step 4: 最终提交**

```bash
git add configs/cameras.py  # 仅在修改了 SKIP_FRAMES 时
git commit -m "perf: complete batch inference optimization, GPU util 30%->70%+"
```

---

## 快速回归检查

如果任何步骤出现问题，检查：

1. `CameraThread.__init__` 是否还有 `self.engine = None`（应已删除）
2. `InferenceWorker.run` 中 `batch[cam_id] = (frame_id, frame)` 是否覆盖同 cam_id 旧帧（是）
3. `result_queue` 写入前是否先 `get_nowait()` 清旧（是，在 `InferenceWorker.run` 分发部分）
4. `CameraThread.run` 中 `self.infer_queue.put_nowait` 是否在 try/except queue.Full 内（是）
