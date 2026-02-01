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
import sqlite3
import io
from typing import Dict, List, Tuple, Optional
from datetime import datetime


class GalleryManager:
    """
    人脸库管理器

    数据结构 (SQLite):
    data/gallery/
    ├── gallery.db             # SQLite 数据库 (包含元数据、图片和特征)
    └── (旧文件将通过迁移脚本清理)

    Gallery Manager

    Data structure (SQLite):
    data/gallery/
    ├── gallery.db             # SQLite database (contains metadata, images, and embeddings)
    └── (Old files will be cleaned via migration script)
    """

    def __init__(self, gallery_dir: str):
        """
        初始化人脸库管理器

        Args:
            gallery_dir: 人脸库目录路径
        """
        self.gallery_dir = gallery_dir
        self.db_path = os.path.join(gallery_dir, "gallery.db")

        # 确保目录存在
        os.makedirs(gallery_dir, exist_ok=True)

        # 初始化数据库
        self._init_db()

    def _init_db(self):
        """初始化 SQLite 数据库表架构"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS gallery (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    face_image BLOB,
                    embedding BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()

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
            # 将图片编码为 JPEG 字节流
            _, img_encoded = cv2.imencode('.jpg', face_img)
            img_blob = img_encoded.tobytes()

            # 将特征向量转换为字节流
            emb_blob = embedding.astype(np.float32).tobytes()

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # 使用 INSERT OR REPLACE 处理重名（覆盖）
                cursor.execute('''
                    INSERT OR REPLACE INTO gallery (name, face_image, embedding, created_at)
                    VALUES (?, ?, ?, ?)
                ''', (name, img_blob, emb_blob, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                conn.commit()

            print(f"GalleryManager: 已在数据库中保存 {name}")
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
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM gallery WHERE name = ?', (name,))
                if cursor.rowcount == 0:
                    print(f"GalleryManager: {name} 不存在于库中")
                    return False
                conn.commit()

            print(f"GalleryManager: 已从数据库删除 {name}")
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
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('UPDATE gallery SET name = ? WHERE name = ?', (new_name, old_name))
                if cursor.rowcount == 0:
                    print(f"GalleryManager: {old_name} 不存在于库中")
                    return False
                conn.commit()

            print(f"GalleryManager: 已重命名 {old_name} -> {new_name}")
            return True

        except sqlite3.IntegrityError:
            print(f"GalleryManager: {new_name} 已存在")
            return False
        except Exception as e:
            print(f"GalleryManager: 重命名失败: {e}")
            return False

    def list_all(self) -> Dict[str, Dict]:
        """
        列出所有人脸

        Returns:
            人脸元数据字典 {name: {created_at, ...}}
        """
        results = {}
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT name, created_at FROM gallery')
                for row in cursor.fetchall():
                    results[row[0]] = {"created_at": row[1]}
        except Exception as e:
            print(f"GalleryManager: 获取列表失败: {e}")
        return results

    def get_person(self, name: str) -> Optional[Dict]:
        """
        获取单个人脸信息

        Args:
            name: 人名

        Returns:
            人脸元数据，如果不存在则返回 None
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT created_at FROM gallery WHERE name = ?', (name,))
                row = cursor.fetchone()
                if row:
                    return {"created_at": row[0]}
        except Exception as e:
            print(f"GalleryManager: 获取信息失败: {e}")
        return None

    def get_face_image(self, name: str) -> Optional[np.ndarray]:
        """
        从数据库获取人脸图片

        Args:
            name: 人名

        Returns:
            人脸图片 (numpy array)，如果不存在则返回 None
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT face_image FROM gallery WHERE name = ?', (name,))
                row = cursor.fetchone()
                if row:
                    img_bytes = row[0]
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"GalleryManager: 获取图片失败: {e}")
        return None

    def get_image_path(self, name: str) -> Optional[str]:
        """
        (兼容性保留) 原本返回路径，现在改为返回虚拟 API 或继续从旧路径加载 (迁移后应弃用)
        Better: Web UI should call an API that returns image bytes.
        """
        # 暂时返回 None，后续可能需要 server.py 适配直接从数据库读取
        return None

    def load_embeddings(self) -> Tuple[List[str], np.ndarray]:
        """
        加载所有人脸特征向量用于识别

        Returns:
            (names_list, embeddings_array)
        """
        names = []
        embeddings = []

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT name, embedding FROM gallery')
                for row in cursor.fetchall():
                    name, emb_bytes = row
                    embedding = np.frombuffer(emb_bytes, dtype=np.float32)
                    embeddings.append(embedding)
                    names.append(name)
        except Exception as e:
            print(f"GalleryManager: 加载特征向量失败: {e}")

        if len(embeddings) == 0:
            return [], np.empty((0, 512))

        embeddings_array = np.array(embeddings, dtype=np.float32)
        print(f"GalleryManager: 已从数据库加载 {len(names)} 个人脸特征")

        return names, embeddings_array

    def find_duplicate(self, embedding: np.ndarray, threshold: float = 0.7) -> Optional[Tuple[str, float]]:
        """
        在库中寻找相似的人脸 (重复检测)

        Args:
            embedding: 待检测的特征向量
            threshold: 相似度阈值 (默认 0.7)

        Returns:
            如果找到，返回 (name, similarity)，否则返回 None
        """
        names, gallery_embs = self.load_embeddings()
        if len(names) == 0:
            return None

        # 计算余弦相似度
        # Cosine similarity for normalized embeddings is just dot product
        # Ensure embedding is normalized
        norm_emb = embedding / np.linalg.norm(embedding)
        # gallery_embs should already be normalized if they come from ArcFace
        norms = np.linalg.norm(gallery_embs, axis=1, keepdims=True)
        norm_gallery = gallery_embs / norms
        
        similarities = np.dot(norm_gallery, norm_emb)
        max_idx = np.argmax(similarities)
        max_sim = similarities[max_idx]

        if max_sim >= threshold:
            return names[max_idx], float(max_sim)
        
        return None

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
