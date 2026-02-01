# 人脸识别系统 Face Recognition System

基于 RetinaFace 检测 + ArcFace 特征提取的人脸识别系统，支持 Web 管理界面和实时识别演示。

Face recognition system based on RetinaFace detection + ArcFace feature extraction, with web management interface and real-time recognition demo.

## 系统架构 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    人脸识别系统架构                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │  server.py  │    │   demo.py   │    │ tools/*.py  │     │
│  │  (Web管理)   │    │  (实时识别)  │    │  (批量工具)  │     │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘     │
│         │                  │                  │             │
│         └──────────────────┼──────────────────┘             │
│                            ▼                                │
│              ┌─────────────────────────┐                    │
│              │    utils/face_engine.py │                    │
│              │       (核心引擎)          │                    │
│              └────────────┬────────────┘                    │
│                           │                                 │
│         ┌─────────────────┼─────────────────┐              │
│         ▼                                   ▼              │
│  ┌─────────────────┐              ┌─────────────────┐      │
│  │   RetinaFace    │              │     ArcFace     │      │
│  │   (人脸检测)     │              │   (特征提取)     │      │
│  │   ResNet50      │              │   r100_glint    │      │
│  └─────────────────┘              └─────────────────┘      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 主要特性 Features

- **高精度检测**: RetinaFace (ResNet50 backbone) 人脸检测
- **强特征提取**: ArcFace (ResNet100@Glint360K) 512维特征向量
- **扁平化存储**: 所有图片和特征向量在同一目录，通过 metadata.json 索引
- **实时同步**: Demo 自动同步 Web 上传的人脸数据
- **模块化设计**: FaceEngine 和 GalleryManager 可复用
- **配置化**: 所有参数通过 config.py 统一管理

## 文件结构 File Structure

```
face_demo/
├── config.py                    # 配置文件
├── server.py                    # Web服务端
├── demo.py                      # 实时识别Demo
├── requirements.txt             # 依赖包
├── models/                      # 模型文件目录
│   └── buffalo_l/
│       └── r100_glint.onnx     # ArcFace 识别模型 (需手动下载)
├── utils/
│   ├── face_engine.py          # 人脸处理引擎
│   └── gallery_manager.py      # 人脸库管理
├── data/
│   └── gallery/                # 人脸库 (扁平化)
│       ├── metadata.json       # 元数据索引
│       ├── person1.jpg
│       ├── person1.npy
│       └── ...
└── templates/
    └── index.html              # Web界面
```

## 安装 Installation

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

依赖包说明:

- `opencv-python` - 图像处理
- `numpy` - 数值计算
- `scikit-learn` - 余弦相似度计算
- `onnxruntime` - 模型推理引擎
- `insightface` - ArcFace 特征提取
- `retina-face` - RetinaFace 人脸检测
- `fastapi`, `uvicorn` - Web 服务

### 2. 模型准备

#### RetinaFace 检测模型 (自动下载)

首次运行时会自动下载到 `~/.deepface/weights/retinaface.h5`，大小约 145MB。

#### ArcFace 识别模型 (需手动准备)

确保以下模型文件存在:

```
models/buffalo_l/r100_glint.onnx
```

可从 InsightFace Model Zoo 下载 `buffalo_l` 或 `antelopev2` 模型包。

### 3. 验证安装

```bash
python tools/download_models.py
```

### 4. GPU 支持 (可选)

```bash
# 安装 GPU 版本
pip install onnxruntime-gpu

# 修改 config.py，确保 CUDA 在前
PROVIDERS = ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

## 快速开始 Quick Start

### 1. 验证人脸检测

```bash
# 使用测试图片验证
python tools/detect_faces.py --image your_image.jpg --output result.jpg
```

### 2. 批量注册人脸

将人脸照片放入 `data/测试数据/人像资料/` 目录，文件名即为人名:

```
data/测试数据/人像资料/
├── 张三.jpg
├── 李四.jpg
└── 王五.png
```

运行注册:

```bash
python utils/enroll.py
```

### 3. 启动 Web 管理界面

```bash
python server.py
```

访问 http://localhost:8000 可以:

- 上传新人脸
- 查看已注册人脸
- 删除/重命名人脸

### 4. 运行实时识别 Demo

```bash
python demo.py
```

按 `Q` 退出

### 5. 测试实时同步

1. 保持 `demo.py` 运行
2. 在 Web 界面上传一张新人脸
3. 等待 5 秒
4. Demo 窗口会自动识别新上传的人脸

## 配置说明 Configuration

编辑 `config.py` 修改配置:

```python
# ===========================================
# 模型配置
# ===========================================

# ArcFace 识别模型路径
ARCFACE_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "buffalo_l", "r100_glint.onnx")

# ONNX Runtime 执行提供者 (优先级顺序)
# GPU 用户将 CUDAExecutionProvider 放在首位
PROVIDERS = ['CUDAExecutionProvider', 'CPUExecutionProvider']

# ===========================================
# 检测配置
# ===========================================

# 检测置信度阈值 (0.0-1.0)
# 值越大，检测越严格，漏检率增加
# 值越小，检测越宽松，误检率增加
DET_THRESH = 0.5

# ===========================================
# 识别配置
# ===========================================

# 相似度阈值 (余弦相似度, 0.0-1.0)
# 值越大，识别越严格
SIMILARITY_THRESHOLD = 0.5

# ===========================================
# 服务配置
# ===========================================

SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000

# ===========================================
# Demo 配置
# ===========================================

# 测试视频路径 (或使用摄像头)
VIDEO_PATH = "data/测试数据/测试视频1.mp4"
CAMERA_ID = 0

# 人脸库同步间隔 (秒)
SYNC_INTERVAL = 5
```

## API 文档 API Reference

### FaceEngine

```python
from utils.face_engine import FaceEngine
from config import ARCFACE_MODEL_PATH, PROVIDERS, DET_THRESH

# 初始化引擎
engine = FaceEngine(
    rec_model_path=ARCFACE_MODEL_PATH,  # ArcFace 模型路径
    providers=PROVIDERS,                 # ONNX 执行提供者
    det_thresh=DET_THRESH               # 检测阈值
)

# 检测并提取所有人脸
faces = engine.detect_and_extract(img)  # img: BGR numpy array
# 返回: [{'bbox': [x1,y1,x2,y2], 'landmarks': [...], 'embedding': array, 'det_score': float}, ...]

# 获取最大人脸 (用于注册)
face = engine.get_largest_face(img)
# 返回: 单个 face dict 或 None

# 从文件直接提取特征
embedding = engine.get_embedding("path/to/image.jpg")
# 返回: 512维 numpy array 或 None
```

### GalleryManager

```python
from utils.gallery_manager import GalleryManager

gallery = GalleryManager("data/gallery")

# 添加人脸
success = gallery.add_person(name, image_path, embedding)

# 删除人脸
success = gallery.delete_person(name)

# 重命名
success = gallery.rename_person(old_name, new_name)

# 加载所有特征 (用于识别)
names, embeddings = gallery.load_embeddings()
# names: list of str
# embeddings: numpy array (N, 512)

# 列出所有人脸
all_faces = gallery.list_all()
# 返回: {name: {'image': 'xxx.jpg', 'embedding': 'xxx.npy', 'created_at': '...'}, ...}
```

## 数据格式 Data Format

### metadata.json

```json
{
  "张三": {
    "image": "zhangsan_001.jpg",
    "embedding": "zhangsan_001.npy",
    "created_at": "2026-02-01 18:30:00"
  },
  "李四": {
    "image": "lisi_001.jpg",
    "embedding": "lisi_001.npy",
    "created_at": "2026-02-01 18:35:00"
  }
}
```

### 人脸数据结构 Face Data Structure

```python
face = {
    'bbox': [x1, y1, x2, y2],           # 边界框坐标
    'landmarks': [                       # 5个关键点
        [left_eye_x, left_eye_y],
        [right_eye_x, right_eye_y],
        [nose_x, nose_y],
        [mouth_left_x, mouth_left_y],
        [mouth_right_x, mouth_right_y]
    ],
    'embedding': np.array(...),         # 512维特征向量
    'det_score': 0.99                   # 检测置信度
}
```

## Web API

### 上传人脸

```bash
curl -X POST "http://localhost:8000/upload/" \
  -F "name=张三" \
  -F "file=@photo.jpg"
```

### 获取人脸列表

```bash
curl "http://localhost:8000/api/faces"
```

### 删除人脸

```bash
curl -X DELETE "http://localhost:8000/api/faces/张三"
```

### 重命名人脸

```bash
curl -X PUT "http://localhost:8000/api/faces/旧名字/新名字"
```

### 同步特征数据 (Demo 使用)

```bash
curl "http://localhost:8000/sync_data"
```

## 常见问题 FAQ

### Q: 首次运行很慢?

A: RetinaFace 模型首次使用时会自动下载 (~145MB)，请确保网络连接正常。

### Q: 如何使用 GPU?

A:

1. 安装 `pip install onnxruntime-gpu`
2. 确保 CUDA 和 cuDNN 正确安装
3. 在 `config.py` 中将 `CUDAExecutionProvider` 放在 `PROVIDERS` 列表首位

### Q: 检测不到人脸?

A:

- 确保图片分辨率足够 (建议至少 480x480)
- 尝试降低 `DET_THRESH` 阈值 (如 0.3)
- 确保人脸清晰、光线充足

### Q: 识别准确率不高?

A:

- 调整 `SIMILARITY_THRESHOLD`，值越大越严格
- 确保注册照片质量好，人脸清晰
- 尝试多角度注册同一个人

### Q: 旧的 embedding 能用吗?

A: 不建议。不同模型的 embedding 不兼容，建议使用迁移工具重新提取。

### Q: Demo 不显示新上传的人脸?

A: Demo 每 5 秒自动同步一次 (可在 config.py 中修改 `SYNC_INTERVAL`)，或重启 demo.py。

### Q: 模型文件在哪里?

A:

- RetinaFace: `~/.deepface/weights/retinaface.h5` (自动下载)
- ArcFace: `models/buffalo_l/r100_glint.onnx` (需手动准备)

## 性能优化 Performance Tips

1. **使用 GPU**: 大幅提升处理速度 (需要 onnxruntime-gpu)
2. **降低检测阈值**: 如果漏检，降低 `DET_THRESH`
3. **批量处理**: 注册时关闭可视化 `SHOW_VISUALIZATION = False`
4. **调整视频分辨率**: 降低输入视频分辨率可提升处理速度

## 技术栈 Tech Stack

| 组件     | 技术                | 说明                                   |
| -------- | ------------------- | -------------------------------------- |
| 人脸检测 | RetinaFace          | serengil/retinaface, ResNet50 backbone |
| 特征提取 | ArcFace             | InsightFace, ResNet100@Glint360K       |
| 推理引擎 | ONNX Runtime        | 支持 CPU/GPU                           |
| Web框架  | FastAPI             | 异步高性能                             |
| 图像处理 | OpenCV              | 图像读取、绘制                         |
| 科学计算 | NumPy, scikit-learn | 特征存储、相似度计算                   |

## 更新日志 Changelog

### v3.0.0 (2026-02-01)

- **重大更新**: 人脸检测从 InsightFace 切换到 RetinaFace (serengil)
- 使用 ResNet50 backbone 的 RetinaFace，检测精度更高
- 保留 ArcFace (r100_glint) 进行特征提取
- 更新所有配置和初始化代码
- 新增 `tools/detect_faces.py` 检测测试工具
- 更新 `tools/download_models.py` 模型验证工具

### v2.0.0 (2026-02-01)

- 重构为统一的 InsightFace 架构
- 扁平化数据存储结构
- 新增 FaceEngine 和 GalleryManager 模块
- 移除重复代码 (130+ 行)
- 新增数据迁移工具
- 完善配置系统

### v1.0.0

- 初始版本

## 许可证 License

MIT

## 贡献 Contributing

欢迎提交 Issue 和 Pull Request!
