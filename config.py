"""
人脸识别系统配置文件
Face Recognition System Configuration
"""
import os

# 项目根目录
# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# 模型配置 (Model Configuration)
# ============================================================================

# 模型目录
# Models directory
MODELS_ROOT = os.path.join(PROJECT_ROOT, "models")

# ArcFace 识别模型路径 (用于特征提取)
# ArcFace recognition model path (for feature extraction)
ARCFACE_MODEL_PATH = os.path.join(MODELS_ROOT, "buffalo_l", "r100_glint.onnx")

# ONNX Runtime 执行提供者 (优先级顺序)
# 如果没有 GPU 或 CUDA 环境配置不正确，请将 'CPUExecutionProvider' 置于首位或仅保留它
# ONNX Runtime execution providers (priority order)
PROVIDERS = ['CUDAExecutionProvider', 'CPUExecutionProvider']


# ============================================================================
# 数据配置 (Data Configuration)
# ============================================================================

# 统一人脸库目录 (数据库存储)
# Unified face gallery directory (database storage)
GALLERY_DIR = "data/gallery"

# 数据库路径配置
# Database path configuration
GALLERY_DB_PATH = os.path.join(GALLERY_DIR, "gallery.db")
HISTORY_DB_PATH = os.path.join(GALLERY_DIR, "history.db")

# 旧数据目录 (用于迁移)
# Old data directory (for migration)
OLD_DATASET_DIR = "dataset"
OLD_GALLERY_DIR = "data/gallery"

# ============================================================================
# 检测配置 (Detection Configuration)
# ============================================================================

# 检测置信度阈值 (RetinaFace)
# Detection confidence threshold (RetinaFace)
DET_THRESH = 0.5

# ============================================================================
# 识别配置 (Recognition Configuration)
# ============================================================================

# 相似度阈值 (余弦相似度)
# Similarity threshold (cosine similarity)
SIMILARITY_THRESHOLD = 0.5

# ============================================================================
# 服务端配置 (Server Configuration)
# ============================================================================

# 服务器端口
# Server port
SERVER_PORT = 8008

# 服务器主机地址
# Server host address
SERVER_HOST = "0.0.0.0"

# ============================================================================
# Demo 配置 (Demo Configuration)
# ============================================================================

# 视频源类型: 'camera' (本地摄像头), 'rtsp' (网络摄像头), 'video' (视频文件)
# Video source type: 'camera' (local camera), 'rtsp' (network camera), 'video' (video file)
VIDEO_SOURCE_TYPE = "rtsp"

# 测试视频路径
# Test video path
VIDEO_PATH = "data/测试数据/测试视频1.avi"

# 本地摄像头设备ID (0 为默认摄像头)
# Local camera device ID (0 for default camera)
CAMERA_ID = 0

# ============================================================================
# RTSP 网络摄像头配置 (RTSP Network Camera Configuration)
# ============================================================================

# RTSP 摄像头 IP 地址
# RTSP camera IP address
RTSP_HOST = "192.168.10.106"

# RTSP 端口
# RTSP port
RTSP_PORT = 554

# RTSP 用户名
# RTSP username
RTSP_USERNAME = "admin"

# RTSP 密码
# RTSP password
RTSP_PASSWORD = "hzsm123456"

# RTSP 流路径 (根据摄像头型号可能需要调整，常见: /stream1, /h264, /Streaming/Channels/1)
# RTSP stream path (may need adjustment based on camera model)
RTSP_STREAM_PATH = "/0-0"

# 完整的 RTSP URL (自动生成)
# Full RTSP URL (auto-generated)
RTSP_URL = f"rtsp://{RTSP_USERNAME}:{RTSP_PASSWORD}@{RTSP_HOST}:{RTSP_PORT}{RTSP_STREAM_PATH}"

# RTSP 连接超时时间（毫秒）
# RTSP connection timeout (milliseconds)
RTSP_TIMEOUT = 10000

# RTSP 缓冲区大小
# RTSP buffer size
RTSP_BUFFER_SIZE = 1024

# 人脸库同步间隔 (秒)
# Gallery sync interval (seconds)
SYNC_INTERVAL = 300

# 采样器上报间隔（秒）
# Sampler flush interval (seconds)
SAMPLER_FLUSH_INTERVAL = 120

# 历史记录稀疏化策略 (距今秒数, 保留间隔秒数)
# History thinning tiers: (seconds_ago, keep_interval_seconds)
HISTORY_THINNING_TIERS = [
    (600,    10),    # 最近10分钟：每10秒
    (3600,   60),    # 最近1小时：每1分钟
    (86400,  300),   # 最近1天：每5分钟
    (None,   1200),  # 超过1天：每20分钟
]

# 稀疏化执行间隔（秒）
# Thinning execution interval (seconds)
THINNING_INTERVAL = 600

# ============================================================================
# 批量注册配置 (Batch Enrollment Configuration)
# ============================================================================

# 源图片目录
# Source image directory
SOURCE_DIR = "data/测试数据/人像资料"

# 显示注册可视化
# Show enrollment visualization
SHOW_VISUALIZATION = False

# 可视化显示延迟 (毫秒)
# Visualization display delay (milliseconds)
VISUALIZATION_DELAY = 200
