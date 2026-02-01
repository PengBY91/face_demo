import cv2
import numpy as np
import sys
import os

# 将项目根目录添加到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.face_engine import FaceEngine

def test_detection(image_path):
    # 初始化引擎
    # 使用 CPU 演示，防止 CUDA 环境有问题
    engine = FaceEngine(providers=['CPUExecutionProvider'])
    
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: 无法读取图片 {image_path}")
        # 如果找不到图片，尝试创建一个测试图
        print("尝试创建测试图...")
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        cv2.putText(img, "Test Image", (100, 320), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

    # 检测并提取
    print("正在检测...")
    faces = engine.detect_and_extract(img)
    
    print(f"检测到 {len(faces)} 张人脸")
    
    # 绘制结果
    draw_img = img.copy()
    for face in faces:
        box = face['bbox']
        score = face['det_score']
        
        # 绘制矩形框
        cv2.rectangle(draw_img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        
        # 绘制关键点
        for kp in face['landmarks']:
            cv2.circle(draw_img, (kp[0], kp[1]), 2, (0, 0, 255), -1)
            
        # 绘制分数
        cv2.putText(draw_img, f"{score:.2f}", (box[0], box[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        print(f"Face: score={score:.4f}, bbox={box}, embedding_shape={face['embedding'].shape}")

    # 保存结果
    output_path = "detection_result.jpg"
    cv2.imwrite(output_path, draw_img)
    print(f"结果已保存至 {output_path}")

if __name__ == "__main__":
    test_img = "t1.jpg"
    if len(sys.argv) > 1:
        test_img = sys.argv[1]
    
    test_detection(test_img)
