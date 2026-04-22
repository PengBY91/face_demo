# 多路摄像头批量推理吞吐率优化设计

**日期**: 2026-04-22  
**目标**: 最大化每秒识别帧数，解决多个独立 ONNX Session 争占 GPU 导致利用率低的问题

---

## 背景与问题

当前架构中每路摄像头持有独立的 `FaceEngine` 实例（即独立的 ONNX Session），3 路摄像头在同一 GPU 上串行竞争，GPU 利用率 < 30%，每路识别帧率约 10-15 FPS。RTX 3090/4090 级别 GPU 的批量推理能力完全未被利用。

**根本原因**：ArcFace 特征提取（`rec_model.get_feat`）逐脸单独调用，3 路各 2 张脸 = 6 次独立 GPU kernel 调用，而 batch_size=6 只需 1 次调用，吞吐率差距 4-6x。

---

## 目标架构

### 整体数据流

```
CameraThread-1 ──[读帧, fire-and-forget]──> infer_queue ─┐
CameraThread-2 ──[读帧, fire-and-forget]──> infer_queue ─┤──> InferenceWorker ──[batch推理]──> 单个 FaceEngine
CameraThread-3 ──[读帧, fire-and-forget]──> infer_queue ─┘         │
                                                                    └──> result_queues[cam_id] ──> CameraThread（非阻塞取结果）
```

- `FaceEngine` 从 N 个减为 **1 个**
- `InferenceWorker` 用时间窗收集所有摄像头帧，组 batch 一次 GPU 调用
- `CameraThread` 读帧与推理**完全解耦**，互不阻塞

---

## 组件设计

### 1. FaceEngine 新增接口

**文件**: `utils/face_engine.py`

```python
def batch_detect_and_extract(self, frames: List[np.ndarray]) -> List[List[Dict]]:
    """
    批量处理多帧，返回每帧的人脸列表。
    关键优化：将所有帧检出的对齐人脸打包为单次 ArcFace batch 调用。
    """
```

内部流程：
1. 逐帧调用 `det_model.get(frame)` 做检测（RetinaFace 当前逐帧调用）
2. 收集所有帧中检出的对齐人脸（`face_align.norm_crop`），拼成 `(N, 3, 112, 112)` 的 batch
3. **一次** `rec_model.get_feat(aligned_batch)` 调用提取所有特征
4. 按帧索引将特征分发回对应帧的结果

### 2. InferenceWorker

**文件**: `multi_camera_demo.py`（新增类，约 80 行）

```python
class InferenceWorker(threading.Thread):
    BATCH_WINDOW_MS = 40      # 时间窗：最多等待 40ms 凑 batch（对应 25 FPS 节奏）
    MAX_BATCH_SIZE  = 8       # batch 上限，防止单次推理时延过大

    def __init__(self, face_engine, infer_queue, result_queues): ...
    def run(self): ...        # 主循环：收帧 → batch 推理 → 分发结果
```

**batch 收集策略**（双条件触发）：
- 条件 A：`infer_queue` 中积累帧数 ≥ 摄像头数量
- 条件 B：距上次推理已超过 `BATCH_WINDOW_MS` ms

每路摄像头在一个时间窗内最多取 1 帧（防止单路摄像头淹没 batch）。InferenceWorker 内部用 `dict {cam_id: (frame_id, frame)}` 暂存当前 batch，从 `infer_queue` 取帧时同一 `cam_id` 的新帧直接覆盖旧帧，不追加。

### 3. CameraThread 改动

**文件**: `multi_camera_demo.py`

**去掉**：
- `self.engine = FaceEngine(...)` 初始化
- `self.engine.detect_and_extract(frame)` 调用

**保留不动**：
- `_identify(embedding)` — 纯 CPU numpy 相似度计算，无需 GPU，继续在 CameraThread 中执行
- `_process_detection` / `_flush_to_server` — UI 记录和服务器上报，不变

**新增**：
- 构造时接收 `infer_queue`（共享）和 `result_queue`（本线程专属，`maxsize=2`）
- 提交帧（非阻塞，队列满则跳帧）：
  ```python
  try:
      infer_queue.put_nowait((self.camera_id, self.frame_index, frame))
  except queue.Full:
      pass
  ```
- 取推理结果（非阻塞），拿到 embedding 后继续调 `_identify`：
  ```python
  try:
      _, faces = self.result_queue.get_nowait()  # faces: List[Dict with embedding]
      detections = []
      for face in faces:
          name, score, is_suspicious = self._identify(face['embedding'])
          detections.append({**face, 'name': name, 'score': score, 'is_suspicious': is_suspicious})
          if name != "Unknown" and not is_suspicious:
              self._process_detection(name, score, face['aligned_face'])
      self.cached_detections = detections
  except queue.Empty:
      pass  # 推理未完成，继续用 cached_detections
  ```

**读帧循环绝不阻塞**：`cap.read()` → 提交 → 取结果（非阻塞） → `_identify`（CPU） → 渲染，全程不等待 GPU。

### 4. MultiCameraApp 改动

**文件**: `multi_camera_demo.py`

```python
# __init__ 中
self.shared_engine = FaceEngine(...)          # 单个共享实例
self.infer_queue   = queue.Queue(maxsize=N*4) # N = 摄像头数
self.result_queues = {cam_id: queue.Queue(maxsize=2) for cam_id in ...}
self.infer_worker  = InferenceWorker(self.shared_engine, self.infer_queue, self.result_queues)
self.infer_worker.start()

# CameraThread 不再接收 engine 参数，改为接收 infer_queue + result_queue
```

---

## 时间同步方案

**核心原则**：检测结果无需精确对应当前帧，只需保证**永不阻塞读帧**，并使用**最新可用结果**。

| 问题 | 解法 |
|------|------|
| 帧与结果 frame_id 不对应（帧已推进） | result_queue maxsize=2，始终取最新结果，旧结果自动被覆盖 |
| InferenceWorker 忙时 CameraThread 阻塞 | `put_nowait` + `get_nowait` 全程非阻塞，跳帧而非等待 |
| 某路摄像头掉帧导致 batch 永远凑不满 | 时间窗 40ms 超时强制触发推理 |
| result_queue 积压旧结果 | InferenceWorker 写入前先 `get_nowait()` 清空队列再 `put_nowait()`，保证始终是最新结果（`put_nowait` 本身在队满时丢新不丢旧，需主动清旧） |

---

## 预期收益

| 指标 | 当前 | 优化后（预估） |
|------|------|-------------|
| FaceEngine 实例数 | N（每路1个） | 1 |
| ArcFace GPU 调用次数/秒 | ~45次（3路×15FPS） | ~15次（batch=3，15次/秒） |
| 每路识别帧率 | 10-15 FPS | 25-40 FPS |
| GPU 利用率 | <30% | 60-80% |
| 显存占用 | ~3x单模型 | 1x单模型 |

---

## 改动文件总结

| 文件 | 改动 | 估计行数 |
|------|------|---------|
| `utils/face_engine.py` | 新增 `batch_detect_and_extract` | +60 行 |
| `multi_camera_demo.py` | 新增 `InferenceWorker` 类 | +80 行 |
| `multi_camera_demo.py` | 修改 `CameraThread`（去推理，加提交/取结果） | ~-40 / +30 行 |
| `multi_camera_demo.py` | 修改 `MultiCameraApp.__init__` | ~+15 行 |
| `configs/cameras.py` | `SKIP_FRAMES` 可降为 0（可选） | 1 行 |

**不改动**：`GalleryManager`、`HistoryManager`、`server.py`、全部 UI 渲染方法。
