"""
多路摄像头配置
Multi-Camera Configuration

配置多路 RTSP 网络摄像头及相关参数
Configure multiple RTSP network cameras and related parameters
"""

# ============================================================
# 摄像头配置
# Camera Configuration
# ============================================================

# 3路 RTSP 网络摄像头配置
# 3 RTSP network cameras configuration
CAMERAS = [
    {
        "id": "cam_01",
        "name": "192.168.10.106",
        "enabled": True,
        "type": "rtsp",
        "host": "192.168.10.106",
        "port": 554,
        "username": "admin",
        "password": "hzsm123456",
        "stream_path": "/0-0",
        # 畸变校正参数（None 表示不校正）
        "undistort": None
    },
    {
        "id": "cam_02",
        "name": "192.168.10.107",
        "enabled": True,
        "type": "rtsp",
        "host": "192.168.10.107",
        "port": 554,
        "username": "admin",
        "password": "123456",
        "stream_path": "/h264/ch1/main/av_stream",
        # 畸变校正参数（桶形畸变）
        # k1, k2, k3: 径向畸变系数（负值校正桶形畸变）
        # p1, p2: 切向畸变系数
        # 使用 tools/calibrate_camera.py 标定获取精确参数
        "undistort": {
            "enabled": True,
            # 相机内参矩阵 [fx, 0, cx; 0, fy, cy; 0, 0, 1]
            # 如果为 None，将根据图像尺寸自动估算
            "camera_matrix": None,
            # 畸变系数 [k1, k2, p1, p2, k3]
            # 桶形畸变: k1, k2 通常为负值
            # 这些是初始估算值，建议使用标定工具获取精确值
            "dist_coeffs": [-0.2, 0.1, 0, 0, -0.05],
            # 缩放比例 (0-1)，用于裁剪黑边
            "alpha": 0.5
        }
    },
    {
        "id": "cam_03",
        "name": "192.168.10.108",
        "enabled": True,
        "type": "rtsp",
        "host": "192.168.10.108",
        "port": 554,
        "username": "admin",
        "password": "admin",
        "stream_path": "/livestream/12",
        "undistort": None
    }
]


# ============================================================
# RTSP 连接配置
# RTSP Connection Configuration
# ============================================================

# 连接重试次数
# Number of connection retries
RTSP_MAX_RETRIES = 5

# 重试间隔（秒）
# Retry delay in seconds
RTSP_RETRY_DELAY = 3

# 重连次数
# Number of reconnection attempts
RTSP_RECONNECT_RETRIES = 3

# 连接等待时间（秒）
# Connection wait time in seconds
RTSP_CONNECT_WAIT = 2

# 帧读取验证次数
# Number of frame reads for verification
RTSP_VERIFY_READS = 3

# 帧验证间隔（秒）
# Frame verification interval in seconds
RTSP_VERIFY_INTERVAL = 0.5

# 预期帧率
# Expected FPS
RTSP_EXPECTED_FPS = 25


# ============================================================
# 采样和上报配置
# Sampling and Reporting Configuration
# ============================================================

# 采样器最大缓存数量（每个人）
# Maximum cached results per person
MAX_BEST_RESULTS = 100

# HTTP 请求超时时间（秒）
# HTTP request timeout in seconds
REQUEST_TIMEOUT = 5

# 批量上报间隔（秒）
# Batch reporting interval in seconds
FLUSH_INTERVAL = 30

# 跳帧识别配置（每 N+1 帧识别 1 帧）
# Skip frames for detection (detect 1 frame every N+1 frames)
SKIP_FRAMES = 2


# ============================================================
# 队列和显示配置
# Queue and Display Configuration
# ============================================================

# 识别记录队列大小
# Detection record queue size
DETECTION_QUEUE_SIZE = 100

# 最大显示记录数
# Maximum display records
MAX_DISPLAY_RECORDS = 20

# FaceEngine 初始化等待超时（秒）
# FaceEngine initialization wait timeout in seconds
ENGINE_INIT_TIMEOUT = 15

# 引擎初始化检查间隔（秒）
# Engine initialization check interval in seconds
ENGINE_INIT_CHECK_INTERVAL = 0.5

# 摄像头启动间隔（秒）
# Delay between camera starts in seconds
CAMERA_START_DELAY = 2


# ============================================================
# UI 布局配置
# UI Layout Configuration
# ============================================================

# 默认窗口尺寸
# Default window size
WINDOW_DEFAULT_WIDTH = 1600
WINDOW_DEFAULT_HEIGHT = 900

# 最小窗口尺寸
# Minimum window size
WINDOW_MIN_WIDTH = 800
WINDOW_MIN_HEIGHT = 600

# 视频区占比
# Video area ratio
VIDEO_RATIO = 0.75

# 状态栏高度
# Status bar height
STATUS_BAR_HEIGHT = 35

# 状态面板高度
# Status panel height
STATUS_PANEL_HEIGHT = 220

# 缩略图卡片配置
# Thumbnail card configuration
THUMBNAIL_CARD_WIDTH = 180
THUMBNAIL_LABEL_HEIGHT = 22
THUMBNAIL_ASPECT_RATIO = 16 / 9

# 缩略图最大高度
# Maximum thumbnail height
MAX_THUMBNAIL_HEIGHT = 150

# 缩略图边距
# Thumbnail margin
THUMBNAIL_MARGIN = 5

# 识别记录卡片高度（增加以显示更多信息）
# Detection record card height
RECORD_CARD_HEIGHT = 100
RECORD_CARD_MARGIN = 5


# ============================================================
# 辅助函数
# Helper Functions
# ============================================================

def get_camera_url(camera: dict) -> str:
    """
    生成 RTSP URL

    Args:
        camera: 摄像头配置字典

    Returns:
        RTSP URL 字符串
    """
    return f"rtsp://{camera['username']}:{camera['password']}@{camera['host']}:{camera['port']}{camera['stream_path']}"


def get_enabled_cameras() -> list:
    """
    获取启用的摄像头列表

    Returns:
        启用的摄像头配置列表
    """
    return [cam for cam in CAMERAS if cam.get('enabled', False)]
