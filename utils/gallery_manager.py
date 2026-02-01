"""
人脸库管理模块
Gallery Management Module

管理扁平化的人脸数据存储，包括图片、特征向量和元数据
Manages flat-structured face data storage including images, embeddings, and metadata
"""
import os
import json
import numpy as np
import shutil
import cv2
from typing import Dict, List, Tuple, Optional
from datetime import datetime


class GalleryManager:
    """
    人脸库管理器

    数据结构:
    data/gallery/
    ├── metadata.json          # 元数据索引
    ├── person1.jpg            # 人脸图片
    ├── person1.npy            # 特征向量
    ├── person2.jpg
    ├── person2.npy
    └── ...

    Gallery Manager

    Data structure:
    data/gallery/
    ├── metadata.json          # Metadata index
    ├── person1.jpg            # Face images
    ├── person1.npy            # Embeddings
    ├── person2.jpg
    ├── person2.npy
    └── ...
    """

    def __init__(self, gallery_dir: str):
        """
        初始化人脸库管理器

        Args:
            gallery_dir: 人脸库目录路径
        """
        self.gallery_dir = gallery_dir
        self.metadata_file = os.path.join(gallery_dir, "metadata.json")

        # 确保目录存在
        os.makedirs(gallery_dir, exist_ok=True)

        # 加载元数据
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """
        加载元数据文件

        Returns:
            元数据字典
        """
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"GalleryManager: 警告 - 无法加载元数据: {e}")
                return {}
        return {}

    def _save_metadata(self):
        """保存元数据到文件"""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"GalleryManager: 错误 - 无法保存元数据: {e}")

    def add_person(self, name: str, face_img: np.ndarray, embedding: np.ndarray) -> bool:
        """
        添加人脸到库

        Args:
            name: 人名
            face_img: 人脸图片 (numpy array)
            embedding: 特征向量 (numpy array)

        Returns:
            是否成功添加
        """
        try:
            # 生成唯一文件名 (处理重名情况)
            base_name = self._sanitize_filename(name)
            counter = 1
            final_name = base_name

            while final_name in self.metadata:
                final_name = f"{base_name}_{counter:03d}"
                counter += 1

            # 文件路径
            image_filename = f"{final_name}.jpg"
            embedding_filename = f"{final_name}.npy"

            dest_image_path = os.path.join(self.gallery_dir, image_filename)
            dest_embedding_path = os.path.join(self.gallery_dir, embedding_filename)

            # 保存图片 (对齐后的人脸)
            cv2.imwrite(dest_image_path, face_img)

            # 保存特征向量
            np.save(dest_embedding_path, embedding)

            # 更新元数据
            self.metadata[name] = {
                "image": image_filename,
                "embedding": embedding_filename,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            self._save_metadata()

            print(f"GalleryManager: 已添加 {name} (文件: {final_name})")
            return True

        except Exception as e:
            print(f"GalleryManager: 添加 {name} 失败: {e}")
            return False

    def delete_person(self, name: str) -> bool:
        """
        从库中删除人脸

        Args:
            name: 人名

        Returns:
            是否成功删除
        """
        if name not in self.metadata:
            print(f"GalleryManager: {name} 不存在于库中")
            return False

        try:
            # 删除文件
            image_file = os.path.join(self.gallery_dir, self.metadata[name]["image"])
            embedding_file = os.path.join(self.gallery_dir, self.metadata[name]["embedding"])

            if os.path.exists(image_file):
                os.remove(image_file)
            if os.path.exists(embedding_file):
                os.remove(embedding_file)

            # 更新元数据
            del self.metadata[name]
            self._save_metadata()

            print(f"GalleryManager: 已删除 {name}")
            return True

        except Exception as e:
            print(f"GalleryManager: 删除 {name} 失败: {e}")
            return False

    def rename_person(self, old_name: str, new_name: str) -> bool:
        """
        重命名人脸

        Args:
            old_name: 旧名字
            new_name: 新名字

        Returns:
            是否成功重命名
        """
        if old_name not in self.metadata:
            print(f"GalleryManager: {old_name} 不存在于库中")
            return False

        if new_name in self.metadata:
            print(f"GalleryManager: {new_name} 已存在")
            return False

        try:
            # 更新元数据 (文件名不变，只改索引)
            self.metadata[new_name] = self.metadata.pop(old_name)
            self._save_metadata()

            print(f"GalleryManager: 已重命名 {old_name} -> {new_name}")
            return True

        except Exception as e:
            print(f"GalleryManager: 重命名失败: {e}")
            return False

    def list_all(self) -> Dict[str, Dict]:
        """
        列出所有人脸

        Returns:
            人脸元数据字典 {name: metadata}
        """
        return self.metadata.copy()

    def get_person(self, name: str) -> Optional[Dict]:
        """
        获取单个人脸信息

        Args:
            name: 人名

        Returns:
            人脸元数据，如果不存在则返回 None
        """
        return self.metadata.get(name)

    def get_image_path(self, name: str) -> Optional[str]:
        """
        获取人脸图片路径

        Args:
            name: 人名

        Returns:
            图片绝对路径，如果不存在则返回 None
        """
        if name not in self.metadata:
            return None

        image_file = self.metadata[name]["image"]
        return os.path.join(self.gallery_dir, image_file)

    def load_embeddings(self) -> Tuple[List[str], np.ndarray]:
        """
        加载所有人脸特征向量用于识别

        Returns:
            (names_list, embeddings_array)
            - names_list: 人名列表
            - embeddings_array: 特征向量数组 (N, 512)
        """
        names = []
        embeddings = []

        for name, info in self.metadata.items():
            embedding_file = os.path.join(self.gallery_dir, info["embedding"])

            try:
                embedding = np.load(embedding_file)
                embeddings.append(embedding)
                names.append(name)
            except Exception as e:
                print(f"GalleryManager: 加载 {name} 的特征失败: {e}")

        if len(embeddings) == 0:
            return [], np.empty((0, 512))

        embeddings_array = np.array(embeddings, dtype=np.float32)
        print(f"GalleryManager: 已加载 {len(names)} 个人脸特征")

        return names, embeddings_array

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """
        清理文件名，移除非法字符

        Args:
            name: 原始名字

        Returns:
            安全的文件名
        """
        # 替换非法字符为下划线
        invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
        safe_name = name
        for char in invalid_chars:
            safe_name = safe_name.replace(char, '_')
        return safe_name
