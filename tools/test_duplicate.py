import numpy as np
import os
import cv2
from utils.gallery_manager import GalleryManager
from config import GALLERY_DIR

def test_duplicate_detection():
    print("开始测试重复检测功能...")
    gallery = GalleryManager(GALLERY_DIR)
    
    # 1. 获取现有用户的一个特征
    names, embs = gallery.load_embeddings()
    if len(names) == 0:
        print("❌ 库中无数据，无法测试。")
        return
    
    test_name = names[0]
    test_emb = embs[0]
    
    # 2. 测试完全相同的特征 (相似度应为 1.0)
    print(f"测试相同特征: {test_name}")
    dup = gallery.find_duplicate(test_emb, threshold=0.7)
    if dup:
        name, sim = dup
        print(f"✅ 成功检测到重复: {name}, 相似度: {sim:.4f}")
    else:
        print("❌ 未能检测到完全相同的特征")

    # 3. 测试带有轻微噪声的特征
    print("测试带噪声的特征...")
    noisy_emb = test_emb + np.random.normal(0, 0.01, test_emb.shape).astype(np.float32)
    dup = gallery.find_duplicate(noisy_emb, threshold=0.7)
    if dup:
        name, sim = dup
        print(f"✅ 成功检测到相似重复: {name}, 相似度: {sim:.4f}")
    else:
        print("❌ 未能检测到带噪声的相似特征")

    # 4. 测试完全不同的特征
    print("测试完全不同的特征...")
    diff_emb = np.random.rand(512).astype(np.float32)
    dup = gallery.find_duplicate(diff_emb, threshold=0.7)
    if dup is None:
        print("✅ 成功识别为非重复 (随机向量)")
    else:
        name, sim = dup
        print(f"❌ 误报重复: {name}, 相似度: {sim:.4f}")

    print("重复检测功能测试结束。")

if __name__ == "__main__":
    test_duplicate_detection()
