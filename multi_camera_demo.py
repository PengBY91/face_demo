"""
多路摄像头人脸识别系统
Multi-Camera Face Recognition System

实现多路（3-4路）摄像头的实时人脸识别，采用"均等网格 + 侧边栏"的 UI 布局
Implements real-time face recognition for multiple cameras (3-4) with grid + sidebar UI layout

架构:
Architecture:
- 每路摄像头独立线程处理，拥有独立的 FaceEngine 实例
- 主线程负责 UI 渲染
- 共享 GalleryManager（只读，线程安全）
- 同时启动 Web 服务端（端口 8008），提供历史记录查询和人脸库管理功能
"""
import cv2
import numpy as np
import os
import threading
import time
import queue
import base64
import requests
import asyncio
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from config import (
    ARCFACE_MODEL_PATH, PROVIDERS,
    GALLERY_DIR, DET_THRESH,
    SIMILARITY_THRESHOLD, SUSPICIOUS_THRESHOLD, SYNC_INTERVAL,
    SERVER_HOST, SERVER_PORT, SAMPLER_FLUSH_INTERVAL,
    RTSP_BUFFER_SIZE
)
from configs.cameras import (
    # 摄像头配置
    CAMERAS, get_camera_url, get_enabled_cameras,
    # RTSP 连接配置
    RTSP_MAX_RETRIES, RTSP_RETRY_DELAY, RTSP_RECONNECT_RETRIES,
    RTSP_CONNECT_WAIT, RTSP_VERIFY_READS, RTSP_VERIFY_INTERVAL,
    RTSP_EXPECTED_FPS,
    # 采样和上报配置
    MAX_BEST_RESULTS, REQUEST_TIMEOUT, FLUSH_INTERVAL, SKIP_FRAMES,
    # 队列和显示配置
    DETECTION_QUEUE_SIZE, MAX_DISPLAY_RECORDS,
    ENGINE_INIT_TIMEOUT, ENGINE_INIT_CHECK_INTERVAL, CAMERA_START_DELAY,
    # UI 布局配置
    WINDOW_DEFAULT_WIDTH, WINDOW_DEFAULT_HEIGHT,
    WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT,
    VIDEO_RATIO, STATUS_BAR_HEIGHT, STATUS_PANEL_HEIGHT,
    THUMBNAIL_CARD_WIDTH, THUMBNAIL_LABEL_HEIGHT, THUMBNAIL_ASPECT_RATIO,
    MAX_THUMBNAIL_HEIGHT, THUMBNAIL_MARGIN,
    RECORD_CARD_HEIGHT, RECORD_CARD_MARGIN
)
from utils.face_engine import FaceEngine
from utils.gallery_manager import GalleryManager

# 全局字体缓存，避免每次调用都加载字体文件
_CACHED_FONTS = {}
_DEFAULT_FONT = None
_FONT_LOCK = threading.Lock()

def _get_font(font_size=20):
    """
    获取缓存的字体对象，避免重复加载
    """
    global _DEFAULT_FONT

    font_path = None
    try:
        font_path = "C:/Windows/Fonts/msyh.ttc"
        if not os.path.exists(font_path):
            font_path = "C:/Windows/Fonts/simsun.ttc"
        if not os.path.exists(font_path):
            font_path = None
    except:
        font_path = None

    if font_path is None:
        with _FONT_LOCK:
            if _DEFAULT_FONT is None:
                _DEFAULT_FONT = ImageFont.load_default()
        return _DEFAULT_FONT

    # 使用缓存
    key = (font_path, font_size)
    if key not in _CACHED_FONTS:
        with _FONT_LOCK:
            if key not in _CACHED_FONTS:  # double check
                _CACHED_FONTS[key] = ImageFont.truetype(font_path, font_size)
    return _CACHED_FONTS[key]

# RTSP 环境变量只设置一次
_RTSP_ENV_SET = False
_RTSP_ENV_LOCK = threading.Lock()

def _ensure_rtsp_env():
    """确保 RTSP 环境变量只设置一次"""
    global _RTSP_ENV_SET
    if not _RTSP_ENV_SET:
        with _RTSP_ENV_LOCK:
            if not _RTSP_ENV_SET:
                os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|udp'
                _RTSP_ENV_SET = True


def cv2_add_chinese_text(img, text, position, font_size=20, color=(255, 255, 255)):
    """
    在 OpenCV 图像上添加中文文本

    Args:
        img: OpenCV 图像 (numpy array)
        text: 要显示的文本
        position: 文本位置 (x, y)
        font_size: 字体大小
        color: 文字颜色 (B, G, R)

    Returns:
        添加文本后的图像
    """
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # 使用缓存的字体
    font = _get_font(font_size)

    color_rgb = (color[2], color[1], color[0])
    draw.text(position, text, font=font, fill=color_rgb)
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    return img_cv


class DetectionRecord:
    """识别记录数据类"""
    def __init__(self, camera_id: str, camera_name: str, name: str, score: float,
                 snapshot: np.ndarray, gallery_img: Optional[np.ndarray], timestamp: float):
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.name = name
        self.score = score
        self.snapshot = snapshot
        self.gallery_img = gallery_img
        self.timestamp = timestamp


class CameraThread(threading.Thread):
    """
    单路摄像头处理线程
    每个线程拥有独立的 FaceEngine 实例和 VideoCapture
    """

    # 类常量（从配置导入）
    MAX_BEST_RESULTS = MAX_BEST_RESULTS
    REQUEST_TIMEOUT = REQUEST_TIMEOUT

    def __init__(self, camera_config: dict, gallery: GalleryManager,
                 detection_queue: queue.Queue, feature_db: dict,
                 feature_lock: threading.Lock):
        """
        初始化摄像头线程

        Args:
            camera_config: 摄像头配置
            gallery: 人脸库管理器（共享，只读）
            detection_queue: 识别记录队列（共享）
            feature_db: 特征数据库字典 {'vectors': np.ndarray, 'names': List[str]}（共享）
            feature_lock: 特征数据读写锁
        """
        super().__init__(daemon=True)

        self.camera_config = camera_config
        self.camera_id = camera_config['id']
        self.camera_name = camera_config['name']
        self.gallery = gallery
        self.detection_queue = detection_queue
        self.feature_db = feature_db  # 使用字典容器，更新时所有线程都能看到
        self.feature_lock = feature_lock

        # 线程状态 - 使用锁保护
        self._running = True
        self._connected = False
        self._fps = 0.0
        self._state_lock = threading.Lock()

        self.latest_frame = None
        self.latest_detections = []
        self.frame_count = 0
        self.last_fps_time = time.time()

        # FaceEngine 延迟初始化（在 run 方法中进行）
        self.engine = None
        self._engine_initialized = False

        # 采样器：用于定期上报识别结果
        self.best_results = {}  # {name: {"score": score, "image": image}}
        self.best_results_lock = threading.Lock()
        self.last_flush_time = time.time()
        self.flush_interval = FLUSH_INTERVAL  # 批量上报间隔
        self.server_url = f"http://{SERVER_HOST.replace('0.0.0.0', '127.0.0.1')}:{SERVER_PORT}"

        # 使用 Session 复用 HTTP 连接
        self.session = requests.Session()

        # 跳帧识别配置
        self.skip_frames = SKIP_FRAMES
        self.frame_index = 0  # 帧计数器
        self.cached_detections = []  # 缓存的识别结果，用于插值显示

        # 畸变校正配置
        self.undistort_config = camera_config.get('undistort')
        self.undistort_mapx = None
        self.undistort_mapy = None
        self.undistort_roi = None

    @property
    def running(self):
        with self._state_lock:
            return self._running

    @running.setter
    def running(self, value):
        with self._state_lock:
            self._running = value

    @property
    def connected(self):
        with self._state_lock:
            return self._connected

    @connected.setter
    def connected(self, value):
        with self._state_lock:
            self._connected = value

    @property
    def fps(self):
        with self._state_lock:
            return self._fps

    @fps.setter
    def fps(self, value):
        with self._state_lock:
            self._fps = value

    @property
    def engine_initialized(self):
        with self._state_lock:
            return self._engine_initialized

    @engine_initialized.setter
    def engine_initialized(self, value):
        with self._state_lock:
            self._engine_initialized = value

    def _init_undistort(self, frame_width: int, frame_height: int):
        """
        初始化畸变校正映射表

        Args:
            frame_width: 帧宽度
            frame_height: 帧高度
        """
        if not self.undistort_config or not self.undistort_config.get('enabled'):
            return

        image_size = (frame_width, frame_height)

        # 获取或估算相机内参
        cm_config = self.undistort_config.get('camera_matrix')
        if cm_config:
            camera_matrix = np.array(cm_config).reshape(3, 3)
        else:
            # 自动估算：假设 fx=fy=width, cx,cy=中心
            fx = fy = frame_width
            cx, cy = frame_width / 2, frame_height / 2
            camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

        # 畸变系数
        dist_coeffs = np.array(self.undistort_config.get('dist_coeffs', [0, 0, 0, 0, 0]), dtype=np.float64)

        # 缩放参数
        alpha = self.undistort_config.get('alpha', 0.5)

        # 计算最优新相机矩阵
        newcameramtx, self.undistort_roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, image_size, alpha, image_size
        )

        # 预计算映射表（提高运行时性能）
        self.undistort_mapx, self.undistort_mapy = cv2.initUndistortRectifyMap(
            camera_matrix, dist_coeffs, None, newcameramtx, image_size, cv2.CV_16SC2
        )

        print(f"CameraThread [{self.camera_name}]: 畸变校正已启用")

    def _apply_undistort(self, frame: np.ndarray) -> np.ndarray:
        """
        应用畸变校正

        Args:
            frame: 原始帧

        Returns:
            校正后的帧
        """
        if self.undistort_mapx is None:
            return frame

        # 使用预计算的映射表进行快速校正
        dst = cv2.remap(frame, self.undistort_mapx, self.undistort_mapy, cv2.INTER_LINEAR)

        # 裁剪有效区域（去除黑边）
        if self.undistort_roi:
            x, y, w, h = self.undistort_roi
            dst = dst[y:y+h, x:x+w]

        return dst

    def run(self):
        """线程主循环"""
        rtsp_url = get_camera_url(self.camera_config)
        rtsp_display_url = rtsp_url.split('@')[-1]  # 隐藏密码的显示 URL

        # 确保 RTSP 环境变量只设置一次（线程安全）
        _ensure_rtsp_env()

        # 延迟初始化 FaceEngine（避免多线程同时初始化 GPU 资源冲突）
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

        # RTSP 连接重试机制
        max_retries = RTSP_MAX_RETRIES
        retry_delay = RTSP_RETRY_DELAY
        cap = None

        for retry in range(max_retries):
            if not self.running:
                return

            print(f"CameraThread [{self.camera_name}]: 正在连接 {rtsp_display_url}... (尝试 {retry + 1}/{max_retries})")

            try:
                # 使用 FFMPEG 后端，设置超时参数
                cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, RTSP_BUFFER_SIZE)
                cap.set(cv2.CAP_PROP_FPS, RTSP_EXPECTED_FPS)

                # 等待连接建立
                time.sleep(RTSP_CONNECT_WAIT)

                if cap.isOpened():
                    # 尝试读取多帧验证连接（有些摄像头需要预热）
                    for _ in range(RTSP_VERIFY_READS):
                        ret, test_frame = cap.read()
                        if ret and test_frame is not None:
                            print(f"CameraThread [{self.camera_name}]: 连接成功! 分辨率: {test_frame.shape[1]}x{test_frame.shape[0]}")
                            self.connected = True
                            break
                        time.sleep(RTSP_VERIFY_INTERVAL)

                    if self.connected:
                        # 初始化畸变校正
                        self._init_undistort(test_frame.shape[1], test_frame.shape[0])
                        break
                    else:
                        print(f"CameraThread [{self.camera_name}]: 无法读取视频帧")
                        cap.release()
                        cap = None
                else:
                    print(f"CameraThread [{self.camera_name}]: 无法打开视频流")
                    if cap:
                        cap.release()
                        cap = None

            except Exception as e:
                print(f"CameraThread [{self.camera_name}]: 连接异常: {e}")
                if cap:
                    cap.release()
                    cap = None

            if retry < max_retries - 1:
                print(f"CameraThread [{self.camera_name}]: {retry_delay}秒后重试...")
                time.sleep(retry_delay)

        if not self.connected:
            print(f"CameraThread [{self.camera_name}]: 达到最大重试次数，连接失败!")
            if cap:
                cap.release()
            self.running = False
            return

        # 主循环
        while self.running:
            ret, frame = cap.read()
            if not ret:
                print(f"CameraThread [{self.camera_name}]: 读取帧失败，尝试重连...")
                cap.release()
                self.connected = False

                # 重连逻辑
                reconnect_success = False
                for retry in range(RTSP_RECONNECT_RETRIES):
                    time.sleep(RTSP_CONNECT_WAIT)
                    print(f"CameraThread [{self.camera_name}]: 重连中... (尝试 {retry + 1}/{RTSP_RECONNECT_RETRIES})")
                    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, RTSP_BUFFER_SIZE)
                    time.sleep(RTSP_CONNECT_WAIT)
                    if cap.isOpened():
                        for _ in range(RTSP_VERIFY_READS):
                            ret, test_frame = cap.read()
                            if ret:
                                self.connected = True
                                reconnect_success = True
                                print(f"CameraThread [{self.camera_name}]: 重连成功!")
                                break
                            time.sleep(RTSP_VERIFY_INTERVAL)
                        if reconnect_success:
                            break
                    if cap:
                        cap.release()
                        cap = None

                if not reconnect_success:
                    print(f"CameraThread [{self.camera_name}]: 重连失败，停止线程")
                    break
                continue

            # 更新 FPS
            self.frame_count += 1
            self.frame_index += 1
            current_time = time.time()
            if current_time - self.last_fps_time >= 1.0:
                self.fps = self.frame_count / (current_time - self.last_fps_time)
                self.frame_count = 0
                self.last_fps_time = current_time

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

            # 始终更新最新帧（用于显示）
            self.latest_frame = frame.copy()

        cap.release()
        print(f"CameraThread [{self.camera_name}]: 已停止")

    def _identify(self, face_embedding: np.ndarray) -> Tuple[str, float, bool]:
        """
        识别单个人脸

        Returns:
            (name, score, is_suspicious)
        """
        if face_embedding is None:
            return "Unknown", 0.0, False

        # 快速复制数据，减少锁持有时间
        with self.feature_lock:
            feature_vectors = self.feature_db['vectors']
            feature_names = self.feature_db['names']

            if len(feature_vectors) == 0:
                return "Unknown", 0.0, False

            # 复制引用（numpy array 的视图，非常快）
            vectors_copy = feature_vectors
            names_copy = feature_names

        # 锁外计算相似度（耗时操作）
        sims = cosine_similarity([face_embedding], vectors_copy)[0]
        best_idx = np.argmax(sims)
        best_score = sims[best_idx]

        if best_score >= SIMILARITY_THRESHOLD:
            return names_copy[best_idx], best_score, False
        elif best_score >= SUSPICIOUS_THRESHOLD:
            return names_copy[best_idx], best_score, True

        return "Unknown", best_score, False

    def _process_detection(self, name: str, score: float, aligned_face: np.ndarray):
        """处理识别结果：生成 UI 记录和采样上报"""
        now = time.time()

        # 生成 UI 识别记录
        gallery_img = self.gallery.get_face_image(name)
        record = DetectionRecord(
            camera_id=self.camera_id,
            camera_name=self.camera_name,
            name=name,
            score=score,
            snapshot=aligned_face.copy(),
            gallery_img=gallery_img,
            timestamp=now
        )

        # 非阻塞方式放入队列
        try:
            self.detection_queue.put_nowait(record)
        except queue.Full:
            pass  # 队列满时静默丢弃，但记录日志
            # 可以考虑记录日志：print(f"队列已满，丢弃记录: {name}")

        # 采样上报：更新最佳结果（线程安全）
        with self.best_results_lock:
            # 限制最大缓存数量，防止内存无限增长
            if name not in self.best_results and len(self.best_results) >= self.MAX_BEST_RESULTS:
                pass  # 跳过新的未知人员
            elif name not in self.best_results or score > self.best_results[name]['score']:
                self.best_results[name] = {
                    "score": score,
                    "image": aligned_face.copy()
                }

            # 定期批量上报
            if now - self.last_flush_time >= self.flush_interval:
                self._flush_to_server()
                self.last_flush_time = now

    def _flush_to_server(self):
        """批量上报识别结果到服务器"""
        if not self.best_results:
            return

        records = []

        for name, data in self.best_results.items():
            try:
                _, img_encoded = cv2.imencode('.jpg', data['image'])
                img_base64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')

                records.append({
                    'name': name,
                    'confidence': float(data['score']),
                    'image_b64': img_base64,
                    'camera_id': self.camera_id,
                    'camera_name': self.camera_name
                })
            except Exception as e:
                print(f"CameraThread [{self.camera_name}]: 准备数据异常 {name}: {e}")

        if records:
            try:
                resp = self.session.post(
                    f"{self.server_url}/api/records_batch",
                    json=records,
                    timeout=self.REQUEST_TIMEOUT
                )
                if resp.status_code == 200:
                    print(f"CameraThread [{self.camera_name}]: 已上报 {len(records)} 条记录")
                else:
                    print(f"CameraThread [{self.camera_name}]: 上报失败: {resp.text}")
            except requests.exceptions.Timeout:
                print(f"CameraThread [{self.camera_name}]: 上报超时")
            except requests.exceptions.RequestException as e:
                print(f"CameraThread [{self.camera_name}]: 上报异常: {e}")

        # 清空最佳结果缓存
        self.best_results = {}

    def stop(self):
        """停止线程"""
        self.running = False


class MultiCameraApp:
    """
    多路摄像头应用主类
    负责管理摄像头线程和渲染 UI
    """

    def __init__(self):
        print("MultiCameraApp: 初始化...")

        # 人脸库管理器（共享）
        self.gallery = GalleryManager(GALLERY_DIR)

        # 特征数据库（使用字典容器，方便线程间共享更新）
        self.feature_db = {
            'vectors': np.empty((0, 512)),
            'names': []
        }
        self.feature_lock = threading.Lock()
        self.gallery_count = 0
        self._load_gallery()

        # 识别记录队列
        self.detection_queue = queue.Queue(maxsize=DETECTION_QUEUE_SIZE)

        # UI 显示的识别记录列表
        self.display_records = []
        self.max_display_records = MAX_DISPLAY_RECORDS

        # 摄像头线程列表
        self.camera_threads: List[CameraThread] = []
        self.selected_camera_index = 0  # 当前选中的摄像头索引
        self.thumbnail_scroll_offset = 0  # 缩略图滚动偏移量

        # 侧边栏状态（收起/展开）
        self.sidebar_collapsed = False
        self.sidebar_collapse_button_rect = None  # 收起按钮区域
        self.sidebar_expand_button_rect = None  # 展开按钮区域

        # 全屏状态
        self.fullscreen = False

        # 系统状态
        self.running = True

        # 启动摄像头线程
        self._start_cameras()

        # 启动人脸库同步线程
        self.sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
        self.sync_thread.start()

        print("MultiCameraApp: 初始化完成")

    def _load_gallery(self):
        """从 GalleryManager 加载特征"""
        print("MultiCameraApp: 加载人脸库...")
        names, embeddings = self.gallery.load_embeddings()

        with self.feature_lock:
            # 更新字典内容，所有引用该字典的线程都能看到更新
            self.feature_db['names'] = names
            self.feature_db['vectors'] = embeddings
            self.gallery_count = len(names)

        print(f"MultiCameraApp: 成功加载 {len(names)} 个人脸特征")

    def _sync_loop(self):
        """后台定期重新加载人脸库"""
        while True:
            time.sleep(SYNC_INTERVAL)
            print("MultiCameraApp: 同步人脸库...")
            self._load_gallery()

    def _start_cameras(self):
        """启动所有启用的摄像头线程"""
        cameras = get_enabled_cameras()
        print(f"MultiCameraApp: 启动 {len(cameras)} 路摄像头...")

        for i, cam_config in enumerate(cameras):
            thread = CameraThread(
                camera_config=cam_config,
                gallery=self.gallery,
                detection_queue=self.detection_queue,
                feature_db=self.feature_db,
                feature_lock=self.feature_lock
            )
            thread.start()
            self.camera_threads.append(thread)

            # 每个线程启动后等待一段时间，避免同时初始化 GPU 资源
            # 第一个线程启动后等待 FaceEngine 初始化完成
            if i < len(cameras) - 1:
                print(f"MultiCameraApp: 等待 {cam_config['name']} 初始化完成...")
                # 等待该线程的 FaceEngine 初始化完成
                max_wait_count = int(ENGINE_INIT_TIMEOUT / ENGINE_INIT_CHECK_INTERVAL)
                wait_count = 0
                while not thread.engine_initialized and wait_count < max_wait_count:
                    time.sleep(ENGINE_INIT_CHECK_INTERVAL)
                    wait_count += 1
                if thread.engine_initialized:
                    print(f"MultiCameraApp: {cam_config['name']} 初始化完成，启动下一个...")
                else:
                    print(f"MultiCameraApp: {cam_config['name']} 初始化超时，继续启动下一个...")
                # 额外等待确保资源释放
                time.sleep(CAMERA_START_DELAY)

    def _process_detection_queue(self):
        """处理识别记录队列"""
        try:
            while True:
                record = self.detection_queue.get_nowait()
                # 添加到显示列表头部
                self.display_records.insert(0, record)
                # 限制显示数量
                if len(self.display_records) > self.max_display_records:
                    self.display_records.pop()

                # 在控制台打印识别日志
                time_str = datetime.fromtimestamp(record.timestamp).strftime("%Y-%m-%d %H:%M:%S")
                print(f"[识别] {time_str} | 摄像头: {record.camera_name} | 人员: {record.name} | 置信度: {record.score:.2%}")
        except queue.Empty:
            pass

    def _render_video_area(self, canvas: np.ndarray, area_width: int, area_height: int):
        """
        渲染视频区域（上方缩略图 + 下方大画面）

        Args:
            canvas: 画布
            area_width: 视频区域宽度
            area_height: 视频区域高度
        """
        # 渲染缩略图行
        actual_thumbnail_height = self._render_thumbnails(canvas, area_width, MAX_THUMBNAIL_HEIGHT, THUMBNAIL_MARGIN)

        # 渲染选中的摄像头大画面
        main_y = actual_thumbnail_height + THUMBNAIL_MARGIN
        main_height = area_height - main_y
        self._render_main_camera(canvas, 0, main_y, area_width, main_height)

    def _render_thumbnails(self, canvas: np.ndarray, width: int, max_height: int, margin: int) -> int:
        """
        渲染缩略图行（卡片式，支持滑动）

        Args:
            canvas: 画布
            width: 区域宽度
            max_height: 最大高度
            margin: 间距

        Returns:
            实际使用的高度
        """
        num_cameras = len(self.camera_threads)
        if num_cameras == 0:
            return 0

        # 固定缩略图卡片尺寸
        card_width = THUMBNAIL_CARD_WIDTH
        label_height = THUMBNAIL_LABEL_HEIGHT
        aspect_ratio = THUMBNAIL_ASPECT_RATIO
        card_height = int(card_width / aspect_ratio) + label_height

        # 计算可显示的卡片数量
        available_width = width - margin * 2
        visible_count = available_width // (card_width + margin)

        # 限制滚动偏移量
        max_offset = max(0, num_cameras - visible_count)
        self.thumbnail_scroll_offset = max(0, min(self.thumbnail_scroll_offset, max_offset))

        # 保存缩略图位置信息（用于点击检测）
        self.thumbnail_rects = []

        start_index = self.thumbnail_scroll_offset
        end_index = min(start_index + visible_count, num_cameras)

        x_offset = margin
        for i in range(start_index, end_index):
            thread = self.camera_threads[i]

            # 缩略图背景
            is_selected = (i == self.selected_camera_index)
            bg_color = (60, 60, 100) if is_selected else (40, 40, 50)

            thumb_x = x_offset
            thumb_y = margin
            thumb_w = card_width
            thumb_h = card_height - label_height

            # 绘制背景（整个缩略图区域）
            cv2.rectangle(canvas, (thumb_x, thumb_y), (thumb_x + thumb_w, thumb_y + card_height), bg_color, -1)

            # 绘制边框（选中时高亮）
            border_color = (0, 255, 255) if is_selected else (80, 80, 80)
            border_width = 3 if is_selected else 1
            cv2.rectangle(canvas, (thumb_x, thumb_y), (thumb_x + thumb_w, thumb_y + card_height), border_color, border_width)

            # 渲染缩略图内容
            if thread.latest_frame is not None:
                try:
                    frame = thread.latest_frame.copy()
                    # 直接缩放到缩略图尺寸（16:9）
                    thumb_frame = cv2.resize(frame, (thumb_w, thumb_h))
                    canvas[thumb_y:thumb_y + thumb_h, thumb_x:thumb_x + thumb_w] = thumb_frame
                except Exception as e:
                    # 渲染错误时显示提示
                    cv2.rectangle(canvas, (thumb_x, thumb_y), (thumb_x + thumb_w, thumb_y + thumb_h), (40, 40, 50), -1)
                    cv2.putText(canvas, "Error", (thumb_x + thumb_w // 2 - 25, thumb_y + thumb_h // 2),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            else:
                # 无帧时显示状态提示（使用英文避免字体问题）
                text = "Connecting..." if thread.connected else "Offline"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                text_x = thumb_x + (thumb_w - text_size[0]) // 2
                text_y = thumb_y + thumb_h // 2
                cv2.putText(canvas, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            # 绘制名称标签
            label_y = thumb_y + thumb_h
            cv2.rectangle(canvas, (thumb_x, label_y), (thumb_x + thumb_w, thumb_y + card_height), (30, 30, 40), -1)
            # 使用摄像头索引显示（避免中文显示问题）
            cam_label = f"Cam {i + 1}: {thread.camera_id}"
            cv2.putText(canvas, cam_label, (thumb_x + 5, label_y + 16),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1)

            # 显示连接状态
            status_color = (0, 255, 0) if thread.connected else (0, 0, 255)
            cv2.circle(canvas, (thumb_x + thumb_w - 12, thumb_y + 12), 5, status_color, -1)

            # 保存位置信息
            self.thumbnail_rects.append({
                'index': i,
                'x': thumb_x,
                'y': thumb_y,
                'width': thumb_w,
                'height': card_height
            })

            x_offset += card_width + margin

        # 绘制左右滑动提示箭头
        if self.thumbnail_scroll_offset > 0:
            # 左箭头
            arrow_x = 15
            arrow_y = card_height // 2
            cv2.polylines(canvas, [
                np.array([[arrow_x + 15, arrow_y - 10],
                          [arrow_x, arrow_y],
                          [arrow_x + 15, arrow_y + 10]])
            ], False, (200, 200, 200), 2)

        if self.thumbnail_scroll_offset < max_offset:
            # 右箭头
            arrow_x = width - 15
            arrow_y = card_height // 2
            cv2.polylines(canvas, [
                np.array([[arrow_x - 15, arrow_y - 10],
                          [arrow_x, arrow_y],
                          [arrow_x - 15, arrow_y + 10]])
            ], False, (200, 200, 200), 2)

        # 绘制滚动指示器
        if num_cameras > visible_count:
            indicator_width = 100
            indicator_height = 4
            indicator_x = (width - indicator_width) // 2
            indicator_y = card_height + 5

            # 背景条
            cv2.rectangle(canvas, (indicator_x, indicator_y),
                         (indicator_x + indicator_width, indicator_y + indicator_height), (60, 60, 60), -1)

            # 当前进度
            progress_width = max(indicator_width // max(1, num_cameras), 10)
            progress_x = indicator_x + int((self.thumbnail_scroll_offset / max(1, max_offset)) * (indicator_width - progress_width))
            cv2.rectangle(canvas, (progress_x, indicator_y),
                         (progress_x + progress_width, indicator_y + indicator_height), (100, 150, 255), -1)

        return card_height + margin * 2 + 10  # 额外空间给滚动指示器

    def _render_main_camera(self, canvas: np.ndarray, px: int, py: int, width: int, height: int):
        """
        渲染选中的摄像头大画面

        Args:
            canvas: 画布
            px: 起始 X 坐标
            py: 起始 Y 坐标
            width: 宽度
            height: 高度
        """
        if len(self.camera_threads) == 0:
            # 无摄像头时显示占位符
            cv2.rectangle(canvas, (px, py), (px + width, py + height), (30, 30, 40), -1)
            canvas = cv2_add_chinese_text(canvas, "无摄像头", (px + width // 2 - 50, py + height // 2),
                                         font_size=24, color=(100, 100, 100))
            return

        thread = self.camera_threads[self.selected_camera_index]

        if thread.latest_frame is not None:
            frame = thread.latest_frame.copy()

            # 绘制检测框
            for det in thread.latest_detections:
                bbox = det['bbox']
                x1, y1, x2, y2 = bbox
                name = det['name']
                score = det['score']
                is_suspicious = det['is_suspicious']

                # 将检测框扩大到 1.5 倍
                bbox_width = x2 - x1
                bbox_height = y2 - y1
                cx = (x1 + x2) / 2  # 中心点
                cy = (y1 + y2) / 2

                # 扩展后的宽高
                expand_ratio = 1.5
                new_width = bbox_width * expand_ratio
                new_height = bbox_height * expand_ratio

                # 计算新的坐标，并确保不超出画面边界
                new_x1 = int(max(cx - new_width / 2, 0))
                new_y1 = int(max(cy - new_height / 2, 0))
                new_x2 = int(min(cx + new_width / 2, frame.shape[1] - 1))
                new_y2 = int(min(cy + new_height / 2, frame.shape[0] - 1))

                # 颜色：绿色-确认，黄色-疑似，红色-未知
                if name != "Unknown":
                    color = (0, 165, 255) if is_suspicious else (0, 255, 0)
                else:
                    color = (0, 0, 255)

                # 绘制更粗的检测框（线条宽度从 3 增加到 5）
                cv2.rectangle(frame, (new_x1, new_y1), (new_x2, new_y2), color, 5)

                display_name = f"疑似{name}" if is_suspicious else name
                text = f"{display_name} ({score:.2f})"
                frame = cv2_add_chinese_text(frame, text, (new_x1, max(new_y1 - 30, 10)),
                                            font_size=22, color=color)

            # 缩放到目标大小
            main_frame = cv2.resize(frame, (width, height))

            # 绘制摄像头名称标签
            label_bg_height = 35
            cv2.rectangle(main_frame, (0, 0), (width, label_bg_height), (0, 0, 0), -1)
            main_frame = cv2_add_chinese_text(main_frame, thread.camera_name, (10, 8),
                                             font_size=20, color=(255, 255, 255))

            # 显示 FPS 和连接状态
            status_text = f"FPS: {thread.fps:.1f}" if thread.connected else "离线"
            cv2.putText(main_frame, status_text, (width - 150, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            # 无画面时显示等待状态
            main_frame = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.rectangle(main_frame, (0, 0), (width, height), (50, 50, 50), -1)

            text = "连接中..." if thread.connected else "连接失败"
            main_frame = cv2_add_chinese_text(main_frame, text, (width // 2 - 50, height // 2),
                                             font_size=28, color=(200, 200, 200))
            main_frame = cv2_add_chinese_text(main_frame, thread.camera_name, (10, 10),
                                             font_size=20, color=(255, 255, 255))

        canvas[py:py + height, px:px + width] = main_frame

    def _render_status_panel(self, canvas: np.ndarray, width: int, height: int):
        """渲染系统状态面板（右侧边栏上方）"""
        # 背景
        cv2.rectangle(canvas, (0, 0), (width, height), (35, 35, 50), -1)

        # 标题栏
        title_height = 35
        cv2.rectangle(canvas, (0, 0), (width, title_height), (50, 50, 70), -1)
        canvas = cv2_add_chinese_text(canvas, "系统状态", (width // 2 - 45, 8), font_size=18, color=(255, 255, 255))

        # 折叠按钮（放在标题栏最右侧）
        btn_x = width - 20
        btn_y = title_height // 2
        btn_size = 10

        # 绘制折叠按钮背景
        cv2.rectangle(canvas, (btn_x - btn_size, btn_y - btn_size),
                     (btn_x + btn_size, btn_y + btn_size), (80, 80, 100), -1)

        # 绘制向右箭头（表示点击后侧边栏会收起到右边）
        arrow_points = np.array([
            [btn_x - 5, btn_y - 6],
            [btn_x + 5, btn_y],
            [btn_x - 5, btn_y + 6]
        ])
        cv2.fillPoly(canvas, [arrow_points], (200, 200, 220))

        # 保存折叠按钮区域（用于点击检测）
        self.sidebar_collapse_button_rect = {
            'x': btn_x - btn_size,
            'y': btn_y - btn_size,
            'width': btn_size * 2,
            'height': btn_size * 2
        }

        y_offset = title_height + 15
        line_height = 28

        # 人脸库统计
        canvas = cv2_add_chinese_text(canvas, f"人脸库: {self.gallery_count} 人",
                            (15, y_offset), font_size=16, color=(0, 255, 0))
        y_offset += line_height + 5

        # 分隔线
        cv2.line(canvas, (10, y_offset), (width - 10, y_offset), (60, 60, 80), 1)
        y_offset += 10

        # 摄像头状态标题
        canvas = cv2_add_chinese_text(canvas, "摄像头状态:", (15, y_offset), font_size=16, color=(200, 200, 200))
        y_offset += line_height

        for thread in self.camera_threads:
            status = "在线" if thread.connected else "离线"
            color = (0, 255, 0) if thread.connected else (0, 0, 255)
            fps_text = f" ({thread.fps:.0f} FPS)" if thread.connected else ""
            status_text = f"{thread.camera_name}: {status}{fps_text}"
            canvas = cv2_add_chinese_text(canvas, status_text, (20, y_offset), font_size=14, color=color)
            y_offset += line_height - 8

        # 当前时间
        y_offset += 5
        cv2.line(canvas, (10, y_offset), (width - 10, y_offset), (60, 60, 80), 1)
        y_offset += 10
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        canvas = cv2_add_chinese_text(canvas, current_time, (15, y_offset), font_size=14, color=(150, 150, 150))

        return canvas

    def _render_collapsed_sidebar(self, canvas: np.ndarray, sidebar_height: int):
        """
        渲染收起状态的侧边栏（只显示展开按钮条）

        Args:
            canvas: 画布
            sidebar_height: 侧边栏高度
        """
        # 整体背景（窄条）
        collapsed_width = 30
        cv2.rectangle(canvas, (0, 0), (collapsed_width, sidebar_height), (35, 35, 50), -1)

        # 展开按钮放在最顶端
        btn_x = collapsed_width // 2
        btn_y = 20  # 放在顶部
        btn_size = 12

        # 绘制展开按钮背景
        cv2.rectangle(canvas, (5, btn_y - btn_size), (collapsed_width - 5, btn_y + btn_size), (60, 60, 80), -1)

        # 绘制向左箭头（表示点击后侧边栏会从右边展开）
        arrow_points = np.array([
            [btn_x + 5, btn_y - 6],
            [btn_x - 5, btn_y],
            [btn_x + 5, btn_y + 6]
        ])
        cv2.fillPoly(canvas, [arrow_points], (200, 200, 220))

        # 保存展开按钮区域
        self.sidebar_expand_button_rect = {
            'x': 0,
            'y': btn_y - btn_size,
            'width': collapsed_width,
            'height': btn_size * 2
        }

        return canvas, collapsed_width

    def _render_sidebar(self, canvas: np.ndarray, sidebar_width: int, sidebar_height: int):
        """
        渲染侧边栏（上方系统状态 + 下方识别日志）

        Args:
            canvas: 画布
            sidebar_width: 侧边栏宽度
            sidebar_height: 侧边栏高度
        """
        # 整体背景
        cv2.rectangle(canvas, (0, 0), (sidebar_width, sidebar_height), (25, 25, 35), -1)

        # 渲染系统状态面板（上方，包含折叠按钮）
        status_canvas = canvas[:STATUS_PANEL_HEIGHT, :]
        status_canvas = self._render_status_panel(status_canvas, sidebar_width, STATUS_PANEL_HEIGHT)
        canvas[:STATUS_PANEL_HEIGHT, :] = status_canvas

        # 识别日志区域（下方）
        log_start_y = STATUS_PANEL_HEIGHT + 5

        # 识别日志标题栏
        title_height = 35
        cv2.rectangle(canvas, (0, log_start_y), (sidebar_width, log_start_y + title_height), (40, 40, 60), -1)
        canvas = cv2_add_chinese_text(canvas, "识别日志", (sidebar_width // 2 - 45, log_start_y + 8),
                            font_size=18, color=(255, 255, 255))

        # 渲染识别记录卡片
        y_offset = log_start_y + title_height + 10

        for record in self.display_records:
            if y_offset + RECORD_CARD_HEIGHT > sidebar_height:
                break

            self._render_record_card(canvas, 8, y_offset, sidebar_width - 16, RECORD_CARD_HEIGHT, record)
            y_offset += RECORD_CARD_HEIGHT + RECORD_CARD_MARGIN

        return canvas

    def _render_record_card(self, canvas: np.ndarray, x: int, y: int,
                           width: int, height: int, record: DetectionRecord):
        """
        渲染单条识别记录卡片

        显示信息：
        - 左侧：抓拍人脸头像
        - 中间：摄像头名称、人员姓名、置信度、完整日期时间
        - 右侧：底库照片（如有空间）
        """
        # 卡片背景
        cv2.rectangle(canvas, (x, y), (x + width, y + height), (40, 40, 55), -1)
        cv2.rectangle(canvas, (x, y), (x + width, y + height), (60, 60, 80), 1)

        # 计算尺寸
        margin = 8
        snapshot_size = min(height - margin * 2, 70)  # 抓拍图大小，最大 70

        # ===== 左侧：抓拍人脸头像 =====
        snapshot_x = x + margin
        snapshot_y = y + (height - snapshot_size) // 2  # 垂直居中
        try:
            snapshot = cv2.resize(record.snapshot, (snapshot_size, snapshot_size))
            canvas[snapshot_y:snapshot_y + snapshot_size, snapshot_x:snapshot_x + snapshot_size] = snapshot
            # 头像边框
            cv2.rectangle(canvas, (snapshot_x, snapshot_y),
                         (snapshot_x + snapshot_size, snapshot_y + snapshot_size), (100, 150, 255), 2)
        except:
            pass

        # ===== 右侧：底库照片（仅在宽度足够时显示） =====
        gallery_size = snapshot_size
        show_gallery = record.gallery_img is not None and width > 250  # 宽度大于 250 才显示底库照片
        gallery_x = x + width - gallery_size - margin if show_gallery else width

        if show_gallery:
            try:
                gallery_img = cv2.resize(record.gallery_img, (gallery_size, gallery_size))
                gallery_y = y + (height - gallery_size) // 2  # 垂直居中
                canvas[gallery_y:gallery_y + gallery_size, gallery_x:gallery_x + gallery_size] = gallery_img
                # 底库照片边框
                cv2.rectangle(canvas, (gallery_x, gallery_y),
                            (gallery_x + gallery_size, gallery_y + gallery_size), (0, 200, 100), 2)
            except:
                show_gallery = False
                gallery_x = width

        # ===== 中间：信息区域 =====
        info_x = snapshot_x + snapshot_size + 10
        info_max_x = gallery_x - margin if show_gallery else x + width - margin
        info_width = info_max_x - info_x

        # 只有宽度足够时才显示信息
        if info_width > 60:
            info_y = y + 8

            # 摄像头名称
            cam_text = f"[{record.camera_name}]"
            canvas = cv2_add_chinese_text(canvas, cam_text,
                                         (info_x, info_y), font_size=12, color=(100, 200, 255))
            info_y += 18

            # 人员姓名（高亮显示）
            canvas = cv2_add_chinese_text(canvas, record.name,
                                         (info_x, info_y), font_size=18, color=(0, 255, 100))
            info_y += 24

            # 置信度
            score_text = f"Score: {record.score:.2%}"
            cv2.putText(canvas, score_text, (info_x, info_y + 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            info_y += 18

            # 完整识别时间
            time_str = datetime.fromtimestamp(record.timestamp).strftime("%m-%d %H:%M:%S")
            cv2.putText(canvas, time_str, (info_x, info_y + 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    def _render_status_bar(self, canvas: np.ndarray, width: int, height: int):
        """渲染底部状态栏"""
        cv2.rectangle(canvas, (0, 0), (width, height), (30, 30, 40), -1)

        # 运行状态
        canvas = cv2_add_chinese_text(canvas, "运行中", (20, 8),
                                     font_size=16, color=(0, 255, 0))

        # FPS 信息
        fps_parts = []
        for thread in self.camera_threads:
            fps_parts.append(f"{thread.fps:.0f}")
        fps_text = f"FPS: {','.join(fps_parts)}"
        cv2.putText(canvas, fps_text, (100, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # 人脸库
        gallery_text = f"人脸库: {self.gallery_count}人"
        canvas = cv2_add_chinese_text(canvas, gallery_text, (width - 150, 8),
                                     font_size=16, color=(200, 200, 200))

        return canvas

    def run(self):
        """主循环"""
        print("MultiCameraApp: 启动主循环，按 Q 退出")
        print("MultiCameraApp: 点击缩略图切换摄像头，鼠标滚轮滑动缩略图")
        print("MultiCameraApp: 点击侧边栏收起/展开按钮可折叠侧边栏")
        print("MultiCameraApp: 按 F 键切换全屏模式，按 S 键切换侧边栏")

        # 初始化缩略图位置信息
        self.thumbnail_rects = []

        # 创建可调整大小的窗口
        # WINDOW_NORMAL: 允许调整窗口大小
        # 注意：OpenCV 窗口受操作系统限制，最大不超过显示器分辨率
        cv2.namedWindow('Multi-Camera Face Recognition', cv2.WINDOW_NORMAL)

        # 设置默认窗口大小
        cv2.resizeWindow('Multi-Camera Face Recognition', WINDOW_DEFAULT_WIDTH, WINDOW_DEFAULT_HEIGHT)

        # 设置鼠标回调
        cv2.setMouseCallback('Multi-Camera Face Recognition', self._on_mouse_click)

        while self.running:
            # 处理识别记录队列
            self._process_detection_queue()

            # 获取当前窗口大小（支持动态调整，无最大尺寸限制）
            window_width = max(cv2.getWindowImageRect('Multi-Camera Face Recognition')[2], WINDOW_MIN_WIDTH)
            window_height = max(cv2.getWindowImageRect('Multi-Camera Face Recognition')[3], WINDOW_MIN_HEIGHT)

            # 计算布局参数
            if self.sidebar_collapsed:
                # 侧边栏收起时，视频区域占据更多空间
                collapsed_width = 30
                video_width = window_width - collapsed_width
                sidebar_width = collapsed_width
            else:
                video_width = int(window_width * VIDEO_RATIO)
                sidebar_width = window_width - video_width
            main_height = window_height - STATUS_BAR_HEIGHT

            # 创建主画布
            canvas = np.zeros((window_height, window_width, 3), dtype=np.uint8)

            # 渲染视频区域（上方缩略图 + 下方大画面）
            self._render_video_area(canvas, video_width, main_height)

            # 渲染侧边栏
            if self.sidebar_collapsed:
                # 渲染收起状态的侧边栏
                sidebar_canvas = canvas[:main_height, video_width:].copy()
                sidebar_canvas, actual_width = self._render_collapsed_sidebar(sidebar_canvas, main_height)
                canvas[:main_height, video_width:] = sidebar_canvas
            else:
                # 渲染展开状态的侧边栏（右侧：系统状态 + 识别日志）
                if sidebar_width > 100 and main_height > 100:  # 确保侧边栏有足够空间
                    sidebar_canvas = canvas[:main_height, video_width:].copy()
                    sidebar_canvas = self._render_sidebar(sidebar_canvas, sidebar_width, main_height)
                    canvas[:main_height, video_width:] = sidebar_canvas

            # 渲染状态栏
            status_canvas = canvas[main_height:, :].copy()
            status_canvas = self._render_status_bar(status_canvas, window_width, STATUS_BAR_HEIGHT)
            canvas[main_height:, :] = status_canvas

            # 显示
            cv2.imshow('Multi-Camera Face Recognition', canvas)

            # 按键检测
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                self.running = False
            elif key == ord('s') or key == ord('S'):
                # S 键切换侧边栏
                self.sidebar_collapsed = not self.sidebar_collapsed
                print(f"MultiCameraApp: 侧边栏{'已收起' if self.sidebar_collapsed else '已展开'}")
            elif key == ord('f') or key == ord('F'):
                # F 键切换全屏模式
                self.fullscreen = not self.fullscreen
                if self.fullscreen:
                    cv2.setWindowProperty('Multi-Camera Face Recognition', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    print("MultiCameraApp: 已进入全屏模式")
                else:
                    cv2.setWindowProperty('Multi-Camera Face Recognition', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('Multi-Camera Face Recognition', WINDOW_DEFAULT_WIDTH, WINDOW_DEFAULT_HEIGHT)
                    print("MultiCameraApp: 已退出全屏模式")

        # 清理
        for thread in self.camera_threads:
            thread.stop()
        cv2.destroyAllWindows()
        print("MultiCameraApp: 已退出")

    def _on_mouse_click(self, event, x, y, flags, param):
        """鼠标事件回调，用于切换摄像头、滚动缩略图、收起/展开侧边栏"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # 获取当前窗口大小
            window_width = cv2.getWindowImageRect('Multi-Camera Face Recognition')[2]

            # 计算视频区域宽度
            if self.sidebar_collapsed:
                video_width = window_width - 30
            else:
                video_width = int(window_width * VIDEO_RATIO)

            # 检查是否点击了侧边栏区域
            if x >= video_width:
                if self.sidebar_collapsed:
                    # 点击收起状态的侧边栏，检查是否点击了展开按钮
                    if self.sidebar_expand_button_rect:
                        btn = self.sidebar_expand_button_rect
                        # 将窗口坐标转换为侧边栏内相对坐标
                        rel_x = x - video_width
                        if (btn['x'] <= rel_x <= btn['x'] + btn['width'] and
                            btn['y'] <= y <= btn['y'] + btn['height']):
                            self.sidebar_collapsed = False
                            print("MultiCameraApp: 侧边栏已展开")
                else:
                    # 检查是否点击了收起按钮
                    if self.sidebar_collapse_button_rect:
                        btn = self.sidebar_collapse_button_rect
                        # 将窗口坐标转换为侧边栏内相对坐标
                        rel_x = x - video_width
                        if (btn['x'] <= rel_x <= btn['x'] + btn['width'] and
                            btn['y'] <= y <= btn['y'] + btn['height']):
                            self.sidebar_collapsed = True
                            print("MultiCameraApp: 侧边栏已收起")
                return

            # 检查是否点击了缩略图
            for rect in self.thumbnail_rects:
                if (rect['x'] <= x <= rect['x'] + rect['width'] and
                    rect['y'] <= y <= rect['y'] + rect['height']):
                    if self.selected_camera_index != rect['index']:
                        self.selected_camera_index = rect['index']
                        camera_name = self.camera_threads[rect['index']].camera_name
                        print(f"MultiCameraApp: 切换到摄像头 [{camera_name}]")
                    break

        elif event == cv2.EVENT_MOUSEWHEEL:
            # 鼠标滚轮滚动缩略图
            num_cameras = len(self.camera_threads)
            if num_cameras > 0:
                # 计算可显示的卡片数量
                window_width = cv2.getWindowImageRect('Multi-Camera Face Recognition')[2]
                if self.sidebar_collapsed:
                    available_width = window_width - 30 - THUMBNAIL_MARGIN * 2
                else:
                    available_width = int(window_width * VIDEO_RATIO) - THUMBNAIL_MARGIN * 2
                visible_count = available_width // (THUMBNAIL_CARD_WIDTH + THUMBNAIL_MARGIN)
                max_offset = max(0, num_cameras - visible_count)

                # 滚动方向（兼容不同 OpenCV 版本）
                # flags > 0 表示向上滚动，flags < 0 表示向下滚动
                scroll_delta = flags if flags != 0 else 0
                # 检查高位符号（某些版本 getMouseWheelDelta 不可用）
                if hasattr(cv2, 'getMouseWheelDelta'):
                    scroll_delta = cv2.getMouseWheelDelta(flags)

                if scroll_delta > 0:
                    self.thumbnail_scroll_offset = max(0, self.thumbnail_scroll_offset - 1)
                else:
                    self.thumbnail_scroll_offset = min(max_offset, self.thumbnail_scroll_offset + 1)


def run_web_server(port=8008):
    """
    在后台线程中运行 Web 服务器
    复用 server.py 的 FastAPI 应用

    Args:
        port: Web 服务器端口
    """
    import uvicorn
    # 导入 server 模块的 app 对象，复用所有路由和初始化逻辑
    from server import app as web_app

    print(f"Web服务端: 启动在 http://{SERVER_HOST}:{port}")
    uvicorn.run(web_app, host=SERVER_HOST, port=port)


if __name__ == "__main__":
    # 启动 Web 服务器（在后台线程中）
    web_port = 8008
    web_thread = threading.Thread(target=run_web_server, args=(web_port,), daemon=True)
    web_thread.start()
    print(f"Web服务端已在后台启动，访问地址: http://127.0.0.1:{web_port}")

    # 等待 Web 服务器初始化
    time.sleep(2)

    # 启动多路摄像头应用
    app = MultiCameraApp()
    app.run()
