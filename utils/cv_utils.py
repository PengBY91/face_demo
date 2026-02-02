import cv2
import numpy as np
import os

def imread_unicode(path: str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """
    读取包含中文字符（Unicode）的项目路径图片
    Read image from path containing Unicode (e.g. Chinese) characters on Windows.
    
    Args:
        path: 图片路径
        flags: cv2.imread 标志位
        
    Returns:
        numpy.ndarray: 图片数据，如果读取失败则返回 None
    """
    try:
        # 使用 numpy 读取字节流，避开 OpenCV 对 Windows 路径编码的处理问题
        # Use numpy to read byte stream, avoiding OpenCV's path encoding issues on Windows
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
    except Exception as e:
        print(f"imread_unicode 错误: {e}")
        return None

def imwrite_unicode(path: str, img: np.ndarray, params: list = None) -> bool:
    """
    保存图片到包含中文字符（Unicode）的路径
    Write image to path containing Unicode characters on Windows.
    """
    try:
        ext = os.path.splitext(path)[1]
        result, nparray = cv2.imencode(ext, img, params)
        if result:
            nparray.tofile(path)
            return True
        return False
    except Exception as e:
        print(f"imwrite_unicode 错误: {e}")
        return False
