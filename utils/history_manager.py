"""
识别历史管理模块
History Management Module

管理识别历史数据的存储、查询和抓拍图片
Manages storage, querying, and snapshot images for recognition history
"""
import os
import sqlite3
import cv2
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime

class HistoryManager:
    """
    识别历史管理器
    History Manager
    """

    def __init__(self, gallery_dir: str):
        """
        初始化历史管理器

        Args:
            gallery_dir: 存储目录
        """
        self.db_path = os.path.join(gallery_dir, "history.db")
        os.makedirs(gallery_dir, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """初始化 SQLite 数据库表架构"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # 创建表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS recognition_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_name TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    face_image BLOB,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            # 添加索引以优化查询
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_history_name ON recognition_history(person_name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_history_timestamp ON recognition_history(timestamp)')
            conn.commit()

    def add_history_record(self, name: str, confidence: float, face_img) -> bool:
        """
        添加历史记录
        Args:
            name: 姓名
            confidence: 置信度
            face_img: 可以是 np.ndarray (由 cv2 解码后的) 或者 bytes (原始 JPEG 字节)
        """
        try:
            if isinstance(face_img, np.ndarray):
                _, img_encoded = cv2.imencode('.jpg', face_img)
                img_blob = img_encoded.tobytes()
            else:
                img_blob = face_img  # 假设已经是 bytes
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO recognition_history (person_name, confidence, face_image, timestamp)
                    VALUES (?, ?, ?, ?)
                ''', (name, confidence, img_blob, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                conn.commit()
            return True
        except Exception as e:
            print(f"HistoryManager: 保存历史失败: {e}")
            return False

    def get_history(self, name: Optional[str] = None, start_time: Optional[str] = None, 
                    end_time: Optional[str] = None, limit: int = 100, offset: int = 0) -> List[Dict]:
        """查询历史记录"""
        query = "SELECT id, person_name, confidence, timestamp FROM recognition_history WHERE 1=1"
        params = []
        if name:
            query += " AND person_name LIKE ?"
            params.append(f"%{name}%")
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)

        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        results = []
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, tuple(params))
                for row in cursor.fetchall():
                    results.append({
                        "id": row[0],
                        "person_name": row[1],
                        "confidence": row[2],
                        "timestamp": row[3],
                        "image_url": f"/api/history_image/{row[0]}"
                    })
        except Exception as e:
            print(f"HistoryManager: 查询历史失败: {e}")
        return results

    def get_history_image(self, record_id: int) -> Optional[np.ndarray]:
        """获取特定历史记录的图片"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT face_image FROM recognition_history WHERE id = ?', (record_id,))
                row = cursor.fetchone()
                if row:
                    nparr = np.frombuffer(row[0], np.uint8)
                    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"HistoryManager: 获取图片失败: {e}")
        return None

    def execute_query(self, sql: str, params: tuple = ()) -> List[Dict]:
        """执行自定义 SQL 查询 (用于 AI 搜索)"""
        if not sql.strip().upper().startswith("SELECT"):
            raise ValueError("Only SELECT queries are allowed.")
        results = []
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(sql, params)
                for row in cursor.fetchall():
                    item = dict(row)
                    if "face_image" in item:
                        del item["face_image"]
                    
                    if "id" in item:
                        item["image_url"] = f"/api/history_image/{item['id']}"
                    results.append(item)
        except Exception as e:
            print(f"HistoryManager: 执行查询失败: {e}")
            raise e
        return results

    def cleanup_old_records(self, days: int = 30) -> int:
        """
        删除超过指定天数的历史记录
        Args:
            days: 保留的天数
        Returns:
            int: 删除的记录数量
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "DELETE FROM recognition_history WHERE timestamp < datetime('now', ?)",
                    (f'-{days} days',)
                )
                count = cursor.rowcount
                conn.commit()
                if count > 0:
                    cursor.execute("VACUUM") # 压缩数据库文件
                return count
        except Exception as e:
            print(f"HistoryManager: 清理历史失败: {e}")
            return 0
