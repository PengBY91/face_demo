"""
人脸处理引擎 - 基于 InsightFace 检测 + ArcFace 特征提取
Face Processing Engine - Based on InsightFace detection + ArcFace feature extraction

统一的人脸检测、对齐和特征提取接口
Unified interface for face detection, alignment, and feature extraction

检测使用 InsightFace FaceAnalysis (buffalo_l model pack)
特征提取使用 InsightFace ArcFace 模型
"""
import numpy as np
import cv2
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
from insightface.utils import face_align
from typing import List, Dict, Optional, Tuple
import os
from utils.cv_utils import imread_unicode


class FaceEngine:
    """
    人脸处理引擎
    使用 InsightFace 进行人脸检测，使用 ArcFace 进行特征提取

    Face Processing Engine
    Uses InsightFace for face detection, ArcFace for feature extraction
    """

    def __init__(self,
                 rec_model_path: str = None,
                 providers: List[str] = None,
                 det_thresh: float = 0.5,
                 det_size: Tuple[int, int] = (640, 640)):
        """
        初始化人脸引擎

        Args:
            rec_model_path: ArcFace 识别模型路径 (ONNX格式)
            providers: ONNX Runtime 执行提供者列表
            det_thresh: 检测置信度阈值
            det_size: 检测输入尺寸
        """
        if providers is None:
            # 默认顺序：优先使用 CUDA，其次是 CPU
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        self.det_thresh = det_thresh
        self.providers = providers
        
        # 确定 ctx_id (-1 为 CPU, 0 为 GPU)
        ctx_id = 0 if 'CUDAExecutionProvider' in providers else -1

        # 加载 InsightFace 检测器 (使用 buffalo_l 模型包)
        print(f"FaceEngine: 正在初始化 InsightFace 检测器 (buffalo_l, ctx_id={ctx_id})...")
        
        # 自动定位 models 目录
        if rec_model_path:
            models_root = os.path.dirname(os.path.dirname(rec_model_path))
        else:
            # 默认：寻找当前文件所在目录的上级目录下的 models
            models_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
        
        # InsightFace 的 root 参数应该是 models 目录所在的父目录
        models_parent = os.path.dirname(models_root)
        
        try:
            self.det_model = FaceAnalysis(name='buffalo_l', root=models_parent, allowed_modules=['detection'])
            self.det_model.prepare(ctx_id=ctx_id, det_size=det_size)
        except Exception as e:
            print(f"FaceEngine: 初始化检测器失败: {e}")
            print(f"请检查模型路径是否存在: {os.path.join(models_root, 'buffalo_l')}")
            raise e

        # 加载 ArcFace 识别模型
        if rec_model_path is None:
            # 默认使用 buffalo_l 的识别模型
            rec_model_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "models", "buffalo_l", "r100_glint.onnx"
            )

        print(f"FaceEngine: 正在加载 ArcFace 模型 {rec_model_path}...")
        self.rec_model = get_model(rec_model_path, providers=providers)
        self.rec_model.prepare(ctx_id=ctx_id)
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
        # 使用 InsightFace 检测人脸
        faces = self.det_model.get(img)

        results = []

        for face in faces:
            # 过滤低置信度的人脸
            if face.det_score < self.det_thresh:
                continue

            # 提取边界框 (InsightFace 返回 [x1, y1, x2, y2] 格式)
            bbox = face.bbox.astype(int)
            
            # 提取关键点 (InsightFace kps 已经是 5 点格式: [left_eye, right_eye, nose, left_mouth, right_mouth])
            # 注意：InsightFace 的 kps 顺序与 face_align.norm_crop 是一致的
            landmarks = face.kps

            # 对齐人脸并提取特征 (使用 insightface 官方对齐函数)
            aligned_face = face_align.norm_crop(img, landmarks, image_size=112)
            embedding = self.rec_model.get_feat(aligned_face)

            face_dict = {
                'bbox': bbox.tolist(),
                'landmarks': landmarks.astype(int).tolist(),
                'embedding': embedding.flatten(),
                'aligned_face': aligned_face,
                'det_score': float(face.det_score)
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
        # 使用支持中文路径的读取方法
        img = imread_unicode(img_path)
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

