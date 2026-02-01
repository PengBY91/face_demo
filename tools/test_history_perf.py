import sys
import os
import time
import numpy as np
import cv2
from datetime import datetime, timedelta

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.history_manager import HistoryManager

def test_performance():
    gallery_dir = "data/test_gallery"
    os.makedirs(gallery_dir, exist_ok=True)
    hm = HistoryManager(gallery_dir)
    
    print("--- 性能测试开始 ---")
    
    # 模拟一张测试图片
    test_img = np.zeros((112, 112, 3), dtype=np.uint8)
    cv2.putText(test_img, "Test", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    _, img_encoded = cv2.imencode('.jpg', test_img)
    img_bytes = img_encoded.tobytes()
    
    # 1. 批量插入测试
    count = 1000
    print(f"正在插入 {count} 条记录...")
    start_time = time.time()
    for i in range(count):
        name = f"Person_{i % 10}"
        hm.add_history_record(name, 0.9, img_bytes)
    end_time = time.time()
    print(f"插入 {count} 条记录耗时: {end_time - start_time:.2f}s (平均 {(end_time - start_time)/count*1000:.2f}ms/条)")
    
    # 2. 查询测试 (带过滤)
    print("\n进行带过滤的查询测试...")
    start_time = time.time()
    results = hm.get_history(name="Person_5", limit=50)
    end_time = time.time()
    print(f"查询 'Person_5' (限50条) 耗时: {(end_time - start_time)*1000:.2f}ms (结果数: {len(results)})")
    
    # 3. 清理测试
    print("\n进行清理测试 (模拟旧数据)...")
    # 手动插入一个很久以前的记录
    import sqlite3
    old_time = (datetime.now() - timedelta(days=40)).strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(hm.db_path) as conn:
        conn.execute(
            "INSERT INTO recognition_history (person_name, confidence, face_image, timestamp) VALUES (?, ?, ?, ?)",
            ("Old_Person", 0.5, img_bytes, old_time)
        )
    
    print("执行清理 (保留30天)...")
    deleted_count = hm.cleanup_old_records(days=30)
    print(f"成功清理记录数: {deleted_count}")
    
    # 验证旧记录是否已删除
    old_results = hm.get_history(name="Old_Person")
    if len(old_results) == 0:
        print("验证成功: 旧记录已被清除。")
    else:
        print("验证失败: 旧记录仍然存在。")

    print("\n--- 性能测试结束 ---")
    # 清理测试数据
    # if os.path.exists(hm.db_path):
    #     os.remove(hm.db_path)

if __name__ == "__main__":
    test_performance()
