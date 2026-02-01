"""
人脸处理引擎 - 基于 RetinaFace 检测 + ArcFace 特征提取
Face Processing Engine - Based on RetinaFace detection + ArcFace feature extraction

统一的人脸检测、对齐和特征提取接口
Unified interface for face detection, alignment, and feature extraction

检测使用 serengil/retinaface (ResNet50 backbone)
特征提取使用 InsightFace ArcFace 模型
"""
import numpy as np
import cv2
from retinaface import RetinaFace
from insightface.model_zoo import get_model
from insightface.utils import face_align
from typing import List, Dict, Optional, Tuple
import os


class FaceEngine:
    """
    人脸处理引擎
    使用 RetinaFace 进行人脸检测，使用 ArcFace 进行特征提取

    Face Processing Engine
    Uses RetinaFace for face detection, ArcFace for feature extraction
    """

    def __init__(self,
                 rec_model_path: str = None,
                 providers: List[str] = None,
                 det_thresh: float = 0.5):
        """
        初始化人脸引擎

        Args:
            rec_model_path: ArcFace 识别模型路径 (ONNX格式)
            providers: ONNX Runtime 执行提供者列表
            det_thresh: 检测置信度阈值
        """
        if providers is None:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        self.det_thresh = det_thresh
        self.providers = providers

        # 加载 RetinaFace 检测器 (使用 ResNet50 backbone - 大模型)
        # RetinaFace 会自动下载模型到 ~/.deepface/weights/
        print("FaceEngine: 正在初始化 RetinaFace 检测器 (ResNet50)...")
        # 预热一次检测，确保模型加载
        # RetinaFace 使用 build_model() 自动加载，首次调用时初始化

        # 加载 ArcFace 识别模型
        if rec_model_path is None:
            # 默认使用 buffalo_l 的识别模型
            rec_model_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "models", "buffalo_l", "r100_glint.onnx"
            )

        print(f"FaceEngine: 正在加载 ArcFace 模型 {rec_model_path}...")
        self.rec_model = get_model(rec_model_path, providers=providers)
        self.rec_model.prepare(ctx_id=0)
        print("FaceEngine: 模型加载完成")

    def detect_and_extract(self, img: np.ndarray) -> List[Dict]:
        """
        检测图片中的所有人脸并提取特征

        Args:
            img: 输入图片 (BGR格式)

        Returns:
            人脸列表，每个人脸包含:
            - bbox: [x1, y1, x2, y2] 边界框
            - landmarks: 5个关键点 [[x, y], ...]
            - embedding: 512维特征向量
            - det_score: 检测置信度

            List of faces, each containing:
            - bbox: [x1, y1, x2, y2] bounding box
            - landmarks: 5 facial landmarks [[x, y], ...]
            - embedding: 512-dim feature vector
            - det_score: detection confidence
        """
        # 使用 RetinaFace 检测人脸
        # RetinaFace 需要 RGB 格式，但实际测试 BGR 也可以正常工作
        resp = RetinaFace.detect_faces(img, threshold=self.det_thresh)

        results = []

        # 如果没有检测到人脸，返回空列表
        if not isinstance(resp, dict):
            return results

        for face_key in resp:
            face_data = resp[face_key]

            # 提取边界框 (RetinaFace 返回 [x1, y1, x2, y2] 格式)
            facial_area = face_data['facial_area']
            bbox = [facial_area[0], facial_area[1], facial_area[2], facial_area[3]]

            # 提取关键点 (转换为标准 5 点格式)
            # RetinaFace 返回: right_eye, left_eye, nose, mouth_right, mouth_left
            # 注意：RetinaFace 的 left/right 是从图像角度，需要转换为人脸角度
            lm = face_data['landmarks']
            landmarks = np.array([
                lm['right_eye'],     # 右边(图像视角左侧) -> 对应目标左眼
                lm['left_eye'],      # 左边(图像视角右侧) -> 对应目标右眼
                lm['nose'],          # 鼻子
                lm['mouth_right'],   # 右嘴角(图像视角左侧) -> 对应目标左嘴角
                lm['mouth_left']     # 左嘴角(图像视角右侧) -> 对应目标右嘴角
            ], dtype=np.float32)

            # 对齐人脸并提取特征 (使用 insightface 官方对齐函数)
            aligned_face = face_align.norm_crop(img, landmarks, image_size=112)
            embedding = self.rec_model.get_feat(aligned_face)

            face_dict = {
                'bbox': [int(x) for x in bbox],
                'landmarks': landmarks.astype(int).tolist(),
                'embedding': embedding.flatten(),
                'aligned_face': aligned_face,
                'det_score': float(face_data['score'])
            }
            results.append(face_dict)

        return results

    def get_largest_face(self, img: np.ndarray) -> Optional[Dict]:
        """
        获取图片中最大的人脸
        常用于注册场景，假设只有一个目标人脸

        Get the largest face in the image
        Commonly used for enrollment, assuming single target face

        Args:
            img: 输入图片 (BGR格式)

        Returns:
            最大人脸信息字典，如果没有检测到人脸则返回 None
            Largest face info dict, or None if no face detected
        """
        faces = self.detect_and_extract(img)

        if not faces:
            return None

        # 计算面积并返回最大的
        # Calculate area and return the largest
        largest_face = max(faces, key=lambda f: self._bbox_area(f['bbox']))
        return largest_face

    def get_embedding(self, img_path: str) -> Optional[np.ndarray]:
        """
        从图片文件中提取最大人脸的特征向量
        便捷方法，用于快速从文件获取特征

        Extract embedding of the largest face from image file
        Convenience method for quick feature extraction from file

        Args:
            img_path: 图片文件路径

        Returns:
            512维特征向量，如果失败则返回 None
            512-dim embedding vector, or None if failed
        """
        img = cv2.imread(img_path)
        if img is None:
            return None

        face = self.get_largest_face(img)
        if face is None:
            return None

        return face['embedding']

    @staticmethod
    def _bbox_area(bbox: List[int]) -> float:
        """计算边界框面积"""
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
