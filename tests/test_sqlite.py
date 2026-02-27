import numpy as np
import cv2
import os
from utils.gallery_manager import GalleryManager
from config import GALLERY_DIR

def test_sqlite():
    print("开始测试 SQLite 功能...")
    gallery = GalleryManager(GALLERY_DIR)
    
    # 1. 测试加载
    names, embs = gallery.load_embeddings()
    print(f"当前库中人数: {len(names)}")
    if len(names) == 11:
        print("✅ 加载成功，数量匹配。")
    else:
        print(f"❌ 数量不匹配: 期望 11, 实际 {len(names)}")

    # 2. 测试获取图片
    if len(names) > 0:
        test_name = names[0]
        img = gallery.get_face_image(test_name)
        if img is not None and img.shape == (112, 112, 3):
            print(f"✅ 获取图片成功: {test_name}, 尺寸 {img.shape}")
        else:
            print(f"❌ 获取图片失败: {test_name}")

    # 3. 测试添加新记录
    new_name = "test_user_sql"
    mock_img = np.zeros((112, 112, 3), dtype=np.uint8)
    mock_emb = np.random.rand(512).astype(np.float32)
    
    success = gallery.add_person(new_name, mock_img, mock_emb)
    if success:
        print(f"✅ 添加新用户成功: {new_name}")
    else:
        print(f"❌ 添加新用户失败: {new_name}")

    # 4. 验证新用户是否存在
    names_new, _ = gallery.load_embeddings()
    if new_name in names_new:
        print(f"✅ 在库中找到新用户: {new_name}")
    else:
        print(f"❌ 未能找到新用户: {new_name}")

    # 5. 测试删除
    del_success = gallery.delete_person(new_name)
    if del_success:
        print(f"✅ 删除新用户成功: {new_name}")
    else:
        print(f"❌ 删除新用户失败: {new_name}")

    print("测试结束。")

if __name__ == "__main__":
    os.environ["PYTHONPATH"] = "."
    test_sqlite()
