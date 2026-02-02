"""
识别历史管理模块
History Management Module

管理识别历史数据的存储、查询和抓拍图片
Manages storage, querying, and snapshot images for recognition history
"""
import os
import sqlite3
import time
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from config import HISTORY_THINNING_TIERS

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

    def add_history_records_batch(self, records: List[Dict]) -> bool:
        """
        批量添加历史记录
        Args:
            records: 列表，每个元素包含 {name, confidence, image_bytes}
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                rows = []
                for rec in records:
                    name = rec['name']
                    confidence = rec['confidence']
                    face_img = rec['image'] # 应该是 bytes
                    
                    if isinstance(face_img, np.ndarray):
                        _, img_encoded = cv2.imencode('.jpg', face_img)
                        img_blob = img_encoded.tobytes()
                    else:
                        img_blob = face_img
                    
                    rows.append((name, confidence, img_blob, now_str))
                
                cursor.executemany('''
                    INSERT INTO recognition_history (person_name, confidence, face_image, timestamp)
                    VALUES (?, ?, ?, ?)
                ''', rows)
                conn.commit()
            return True
        except Exception as e:
            print(f"HistoryManager: 批量保存历史失败: {e}")
            return False

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
                    end_time: Optional[str] = None, limit: int = 100, offset: int = 0,
                    include_images: bool = False) -> List[Dict]:
        """查询历史记录"""
        import base64
        
        cols = "id, person_name, confidence, timestamp"
        if include_images:
            cols += ", face_image"
            
        query = f"SELECT {cols} FROM recognition_history WHERE 1=1"
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
                    item = {
                        "id": row[0],
                        "person_name": row[1],
                        "confidence": row[2],
                        "timestamp": row[3],
                        "image_url": f"/api/history_image/{row[0]}"
                    }
                    if include_images and len(row) > 4:
                        img_blob = row[4]
                        if img_blob:
                            base64_img = base64.b64encode(img_blob).decode('utf-8')
                            item["image_data"] = f"data:image/jpeg;base64,{base64_img}"
                    results.append(item)
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
                        img_blob = item["face_image"]
                        if img_blob:
                            import base64
                            base64_img = base64.b64encode(img_blob).decode('utf-8')
                            item["image_data"] = f"data:image/jpeg;base64,{base64_img}"
                        del item["face_image"]
                    
                    if "id" in item:
                        item["image_url"] = f"/api/history_image/{item['id']}"
                    results.append(item)
        except Exception as e:
            print(f"HistoryManager: 执行查询失败: {e}")
            raise e
        return results

    def thin_history(self, tiers: Optional[List[Tuple]] = None) -> int:
        """
        分层稀疏化历史记录。
        对每个时间区间，按 person_name 分组、按时间窗口分桶，
        每桶只保留置信度最高的一条，删除其余记录。

        Returns:
            int: 总共删除的记录数量
        """
        if tiers is None:
            tiers = HISTORY_THINNING_TIERS

        total_deleted = 0
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                now_ts = int(time.time())

                for i, (max_age, interval) in enumerate(tiers):
                    # 确定时间范围的 start/end (用 Unix 时间戳)
                    if max_age is not None:
                        end_ts = now_ts - (tiers[i - 1][0] if i > 0 else 0)
                        start_ts = now_ts - max_age
                    else:
                        # 最后一个 tier：从上一个 tier 的边界到最早
                        end_ts = now_ts - (tiers[i - 1][0] if i > 0 else 0)
                        start_ts = 0

                    start_dt = datetime.fromtimestamp(start_ts).strftime("%Y-%m-%d %H:%M:%S") if start_ts > 0 else "1970-01-01 00:00:00"
                    end_dt = datetime.fromtimestamp(end_ts).strftime("%Y-%m-%d %H:%M:%S")

                    # 用窗口函数找出每个桶中要保留的记录 id
                    delete_sql = """
                        DELETE FROM recognition_history
                        WHERE id NOT IN (
                            SELECT id FROM (
                                SELECT id, ROW_NUMBER() OVER (
                                    PARTITION BY person_name,
                                                 CAST(strftime('%%s', timestamp) AS INTEGER) / :interval
                                    ORDER BY confidence DESC
                                ) AS rn
                                FROM recognition_history
                                WHERE timestamp >= :start AND timestamp < :end
                            ) WHERE rn = 1
                        )
                        AND timestamp >= :start AND timestamp < :end
                    """
                    cursor.execute(delete_sql, {
                        "interval": interval,
                        "start": start_dt,
                        "end": end_dt,
                    })
                    deleted = cursor.rowcount
                    total_deleted += deleted

                conn.commit()
                if total_deleted > 0:
                    print(f"HistoryManager: 稀疏化完成，共删除 {total_deleted} 条记录")
        except Exception as e:
            print(f"HistoryManager: 稀疏化失败: {e}")

        return total_deleted

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
