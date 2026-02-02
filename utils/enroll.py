import os
import cv2
import numpy as np

# 导入新模块
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    ARCFACE_MODEL_PATH, PROVIDERS,
    GALLERY_DIR, DET_THRESH,
    SOURCE_DIR, SHOW_VISUALIZATION, VISUALIZATION_DELAY
)
from utils.face_engine import FaceEngine
from utils.gallery_manager import GalleryManager
from utils.cv_utils import imread_unicode


def main():
    # 1. 初始化模型
    print("初始化 FaceEngine...")
    engine = FaceEngine(
        rec_model_path=ARCFACE_MODEL_PATH,
        providers=PROVIDERS,
        det_thresh=DET_THRESH
    )

    # 2. 初始化库管理器
    gallery = GalleryManager(GALLERY_DIR)

    # 3. 遍历源目录中的图片
    if not os.path.exists(SOURCE_DIR):
        print(f"错误: 源目录 {SOURCE_DIR} 不存在")
        return

    image_files = sorted([f for f in os.listdir(SOURCE_DIR)
                          if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    print(f"找到 {len(image_files)} 张图片待注册")

    for filename in image_files:
        name = os.path.splitext(filename)[0]
        source_path = os.path.join(SOURCE_DIR, filename)

        print(f"处理: {filename} -> {name}")

        # 读取图片 (支持中文路径)
        img = imread_unicode(source_path)
        if img is None:
            print(f"  警告: 无法读取图片 {filename}")
            continue

        # 检测并提取最大人脸
        face = engine.get_largest_face(img)
        if face is None:
            print(f"  警告: 未在 {filename} 中检测到人脸")
            continue

        # 获取特征向量和对齐后的图片
        embedding = face['embedding']
        aligned_face = face['aligned_face']

        # 添加到库
        success = gallery.add_person(name, aligned_face, embedding)
        if success:
            print(f"  ✓ 成功注册 {name}")

            # 可视化
            if SHOW_VISUALIZATION:
                # 绘制检测框
                bbox = face['bbox']
                x1, y1, x2, y2 = bbox
                vis_img = img.copy()
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis_img, f"Enrolled: {name}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

                cv2.imshow("Enrollment", vis_img)
                cv2.waitKey(VISUALIZATION_DELAY)
        else:
            print(f"  × 注册失败 {name}")

    if SHOW_VISUALIZATION:
        cv2.destroyAllWindows()

    print("\n注册流程完成")
    print(f"总计注册: {len(gallery.list_all())} 人")


if __name__ == "__main__":
    main()
