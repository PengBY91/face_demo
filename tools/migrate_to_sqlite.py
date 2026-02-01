import os
import json
import numpy as np
import cv2
from utils.gallery_manager import GalleryManager
from config import GALLERY_DIR

def migrate():
    print(f"开始从 {GALLERY_DIR} 迁移到 SQLite...")
    
    # 加载旧元数据
    metadata_path = os.path.join(GALLERY_DIR, "metadata.json")
    if not os.path.exists(metadata_path):
        print("未发现旧的 metadata.json，无需迁移。")
        return

    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            old_metadata = json.load(f)
    except Exception as e:
        print(f"读取旧元数据失败: {e}")
        return

    # 初始化管理器 (会创建 sqlite 表)
    gallery = GalleryManager(GALLERY_DIR)
    
    count = 0
    for name, info in old_metadata.items():
        img_file = os.path.join(GALLERY_DIR, info["image"])
        emb_file = os.path.join(GALLERY_DIR, info["embedding"])
        
        if os.path.exists(img_file) and os.path.exists(emb_file):
            try:
                # 读取数据
                img = cv2.imread(img_file)
                embedding = np.load(emb_file)
                
                # 写入数据库
                if gallery.add_person(name, img, embedding):
                    print(f"已迁移: {name}")
                    count += 1
                else:
                    print(f"迁移失败: {name}")
            except Exception as e:
                print(f"处理 {name} 时出错: {e}")
        else:
            print(f"缺少文件，跳过: {name} (img: {os.path.exists(img_file)}, emb: {os.path.exists(emb_file)})")

    print(f"迁移完成！共成功迁移 {count} / {len(old_metadata)} 个数据。")
    print("提示: 现在你可以手动备份并删除目录下的旧 .jpg, .npy 和 metadata.json 文件。")

if __name__ == "__main__":
    migrate()
