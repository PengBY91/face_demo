import os
import sys
import cv2
import numpy as np

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from config import ARCFACE_MODEL_PATH, PROVIDERS, DET_THRESH
from utils.face_engine import FaceEngine

def test_model_loading():
    print("--- 模型加载测试 ---")
    print(f"ArcFace 模型路径: {ARCFACE_MODEL_PATH}")
    
    try:
        # 实例化引擎
        engine = FaceEngine(
            rec_model_path=ARCFACE_MODEL_PATH,
            providers=PROVIDERS,
            det_thresh=DET_THRESH
        )
        print("FaceEngine 初始化成功！")
        
        # 检查检测器是否使用了正确的 root
        # 在 InsightFace 中，可以通过检查 det_model.model_dir (或者类似属性)
        # 但我们直接运行一次检测来验证
        
        test_img = np.zeros((640, 640, 3), dtype=np.uint8)
        faces = engine.detect_and_extract(test_img)
        print(f"检测运行成功，发现 {len(faces)} 张人脸（全黑图片预期为0）。")
        
        return True
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # 模拟禁用系统路径 (可选，但更严谨)
    # insightface 默认在 ~/.insightface 下寻找，如果我们配置正确，应该不会报错
    success = test_model_loading()
    sys.exit(0 if success else 1)
