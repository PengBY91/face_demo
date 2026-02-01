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

# ArcFace 识别模型路径 (用于特征提取)
# ArcFace recognition model path (for feature extraction)
ARCFACE_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "buffalo_l", "r100_glint.onnx")

# ONNX Runtime 执行提供者 (优先级顺序)
# ONNX Runtime execution providers (priority order)
PROVIDERS = ['CUDAExecutionProvider', 'CPUExecutionProvider']

# 注意: RetinaFace 检测模型 (ResNet50 backbone) 会自动下载到 ~/.deepface/weights/
# Note: RetinaFace detection model (ResNet50 backbone) will be auto-downloaded to ~/.deepface/weights/

# ============================================================================
# 数据配置 (Data Configuration)
# ============================================================================

# 统一人脸库目录 (扁平化存储)
# Unified face gallery directory (flat storage)
GALLERY_DIR = "data/gallery"

# 元数据文件路径
# Metadata file path
METADATA_FILE = os.path.join(GALLERY_DIR, "metadata.json")

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

# 测试视频路径
# Test video path
VIDEO_PATH = "data/测试数据/测试视频1.mp4"

# 摄像头设备ID (0 为默认摄像头)
# Camera device ID (0 for default camera)
CAMERA_ID = 0

# 人脸库同步间隔 (秒)
# Gallery sync interval (seconds)
SYNC_INTERVAL = 300

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
