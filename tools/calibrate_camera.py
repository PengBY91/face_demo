"""
摄像头畸变标定工具
Camera Distortion Calibration Tool

使用棋盘格标定板获取精确的相机内参和畸变系数
Use a chessboard calibration pattern to get accurate camera parameters

使用方法:
1. 打印棋盘格标定板（建议 9x6 内角点，每格 25mm）
2. 运行此脚本，按空格键采集标定图像（至少 10 张不同角度）
3. 按 ESC 完成标定并显示结果
4. 将输出的参数复制到 configs/cameras.py 中

Usage:
1. Print a chessboard pattern (recommended 9x6 inner corners, 25mm per square)
2. Run this script, press SPACE to capture calibration images (at least 10 from different angles)
3. Press ESC to finish calibration and show results
4. Copy the output parameters to configs/cameras.py
"""

import cv2
import numpy as np
import glob
import os
import sys
import json

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import RTSP_BUFFER_SIZE
from configs.cameras import CAMERAS, get_camera_url


# 标定板参数
CHESSBOARD_SIZE = (9, 6)  # 内角点数量 (列, 行)
SQUARE_SIZE = 25.0  # 棋盘格边长 (mm)，仅影响实际尺寸测量，不影响畸变系数


def get_camera_by_id(camera_id: str):
    """根据 ID 获取摄像头配置"""
    for cam in CAMERAS:
        if cam['id'] == camera_id:
            return cam
    return None


def capture_calibration_images(camera_config: dict, save_dir: str):
    """
    从摄像头采集标定图像

    Args:
        camera_config: 摄像头配置
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)

    rtsp_url = get_camera_url(camera_config)
    rtsp_display_url = rtsp_url.split('@')[-1]

    print(f"\n连接摄像头: {rtsp_display_url}")
    print("=" * 60)
    print("操作说明:")
    print("  [空格] - 采集当前帧")
    print("  [ESC]  - 完成采集，开始标定")
    print("  [Q]    - 取消退出")
    print("")
    print("采集技巧:")
    print("  1. 至少采集 10-20 张图像")
    print("  2. 标定板应覆盖图像的不同区域（左上、右下、中心等）")
    print("  3. 标定板应有不同倾斜角度（俯仰、左右、旋转）")
    print("  4. 标定板应填满画面的 20%-70%")
    print("=" * 60)

    # 设置 RTSP 环境变量
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|udp'

    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, RTSP_BUFFER_SIZE)

    if not cap.isOpened():
        print(f"错误: 无法连接摄像头")
        return False

    # 等待连接稳定
    cv2.waitKey(2000)

    captured_count = 0
    last_capture_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        display = frame.copy()

        # 检测棋盘格（预览）
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret_cb, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

        if ret_cb:
            cv2.drawChessboardCorners(display, CHESSBOARD_SIZE, corners, ret_cb)
            status = f"检测到棋盘格 - 按空格采集"
            color = (0, 255, 0)
        else:
            status = "未检测到棋盘格 - 调整标定板位置"
            color = (0, 0, 255)

        # 显示状态
        cv2.putText(display, f"已采集: {captured_count} 张", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(display, status, (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow('Calibration - Press SPACE to capture, ESC to finish', display)

        key = cv2.waitKey(1) & 0xFF

        current_time = cv2.getTickCount() / cv2.getTickFrequency()
        if key == 32 and ret_cb and (current_time - last_capture_time) > 0.5:  # 空格键
            # 保存图像
            filename = os.path.join(save_dir, f"calib_{captured_count:03d}.jpg")
            cv2.imwrite(filename, frame)
            print(f"已保存: {filename}")
            captured_count += 1
            last_capture_time = current_time
        elif key == 27:  # ESC
            break
        elif key == ord('q') or key == ord('Q'):
            cap.release()
            cv2.destroyAllWindows()
            return False

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n共采集 {captured_count} 张图像")
    return captured_count >= 10


def calibrate_from_images(image_dir: str):
    """
    从图像目录进行标定

    Args:
        image_dir: 图像目录路径

    Returns:
        (camera_matrix, dist_coeffs, rms_error) 或 None
    """
    # 准备标定板世界坐标
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    objpoints = []  # 3D 点
    imgpoints = []  # 2D 点
    image_size = None

    # 读取所有标定图像
    images = glob.glob(os.path.join(image_dir, "calib_*.jpg"))
    images.extend(glob.glob(os.path.join(image_dir, "calib_*.png")))

    if len(images) < 10:
        print(f"错误: 需要至少 10 张图像，当前只有 {len(images)} 张")
        return None

    print(f"\n处理 {len(images)} 张图像...")

    valid_count = 0
    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = gray.shape[::-1]

        # 查找棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

        if ret:
            objpoints.append(objp)

            # 亚像素精度优化
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            imgpoints.append(corners2)

            valid_count += 1
            print(f"  [{valid_count}] {os.path.basename(fname)} - OK")
        else:
            print(f"  [ ] {os.path.basename(fname)} - 未检测到棋盘格")

    if valid_count < 10:
        print(f"错误: 有效图像不足 10 张")
        return None

    print(f"\n使用 {valid_count} 张有效图像进行标定...")

    # 执行标定
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )

    if not ret:
        print("标定失败!")
        return None

    # 计算重投影误差
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i],
                                          camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    mean_error /= len(objpoints)

    print("\n" + "=" * 60)
    print("标定结果")
    print("=" * 60)
    print(f"图像尺寸: {image_size[0]} x {image_size[1]}")
    print(f"RMS 重投影误差: {mean_error:.4f} 像素 (越小越好，<0.5 为优秀)")
    print("")
    print("相机内参矩阵 (camera_matrix):")
    print(camera_matrix)
    print("")
    print("畸变系数 (dist_coeffs):")
    print(dist_coeffs.ravel())
    print("")
    print("=" * 60)
    print("Python 配置格式 (复制到 configs/cameras.py):")
    print("=" * 60)
    print(f'''
"undistort": {{
    "enabled": True,
    "camera_matrix": {list(camera_matrix.ravel().round(4))},
    "dist_coeffs": {list(dist_coeffs.ravel().round(6))},
    "alpha": 0.5
}}
''')

    return camera_matrix, dist_coeffs, mean_error


def test_undistortion(camera_config: dict, camera_matrix: np.ndarray, dist_coeffs: np.ndarray):
    """
    测试畸变校正效果

    Args:
        camera_config: 摄像头配置
        camera_matrix: 相机内参
        dist_coeffs: 畸变系数
    """
    rtsp_url = get_camera_url(camera_config)
    rtsp_display_url = rtsp_url.split('@')[-1]

    print(f"\n测试畸变校正: {rtsp_display_url}")
    print("按 ESC 退出")

    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|udp'

    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, RTSP_BUFFER_SIZE)

    if not cap.isOpened():
        print("无法连接摄像头")
        return

    cv2.waitKey(2000)

    # 获取图像尺寸
    ret, frame = cap.read()
    if not ret:
        print("无法读取帧")
        cap.release()
        return

    h, w = frame.shape[:2]
    image_size = (w, h)

    # 计算最优新相机矩阵
    alpha = 0.5  # 0=最大裁剪, 1=保留所有像素（有黑边）
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs,
                                                       image_size, alpha, image_size)

    # 计算映射表（只需计算一次，提高性能）
    mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None,
                                              newcameramtx, image_size, cv2.CV_16SC2)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # 应用畸变校正
        dst = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

        # 裁剪有效区域
        x, y, w_roi, h_roi = roi
        dst = dst[y:y+h_roi, x:x+w_roi]

        # 并排显示
        display = np.hstack([frame, cv2.resize(dst, (frame.shape[1], frame.shape[0]))])
        cv2.putText(display, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(display, "Corrected", (frame.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Distortion Correction Test', display)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def quick_estimate(camera_config: dict):
    """
    快速估算畸变参数（无需标定板）

    通过实时调整参数来近似校正畸变
    """
    rtsp_url = get_camera_url(camera_config)
    rtsp_display_url = rtsp_url.split('@')[-1]

    print(f"\n快速畸变参数估算: {rtsp_display_url}")
    print("=" * 60)
    print("操作说明:")
    print("  [1/2] - 调整 k1 (主要径向畸变)")
    print("  [3/4] - 调整 k2 (次要径向畸变)")
    print("  [5/6] - 调整 k3 (三阶径向畸变)")
    print("  [R]   - 重置参数")
    print("  [S]   - 保存参数")
    print("  [ESC] - 退出")
    print("=" * 60)

    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|udp'

    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, RTSP_BUFFER_SIZE)

    if not cap.isOpened():
        print("无法连接摄像头")
        return

    cv2.waitKey(2000)

    ret, frame = cap.read()
    if not ret:
        print("无法读取帧")
        cap.release()
        return

    h, w = frame.shape[:2]
    image_size = (w, h)

    # 初始畸变参数
    k1, k2, k3 = -0.2, 0.1, -0.05
    p1, p2 = 0, 0
    alpha = 0.5
    step = 0.01

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # 构建相机矩阵和畸变系数
        fx = fy = w
        cx, cy = w / 2, h / 2
        camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float64)

        # 计算最优新相机矩阵
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs,
                                                           image_size, alpha, image_size)

        # 校正
        dst = cv2.undistort(frame, camera_matrix, dist_coeffs, None, newcameramtx)

        # 显示参数
        display = np.hstack([frame, dst])
        cv2.putText(display, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(display, "Corrected", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 显示当前参数
        params_text = f"k1={k1:.3f}  k2={k2:.3f}  k3={k3:.3f}"
        cv2.putText(display, params_text, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow('Quick Estimation', display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('1'):
            k1 -= step
        elif key == ord('2'):
            k1 += step
        elif key == ord('3'):
            k2 -= step
        elif key == ord('4'):
            k2 += step
        elif key == ord('5'):
            k3 -= step
        elif key == ord('6'):
            k3 += step
        elif key == ord('r') or key == ord('R'):
            k1, k2, k3 = -0.2, 0.1, -0.05
        elif key == ord('s') or key == ord('S'):
            print("\n当前参数配置:")
            print(f'''
"undistort": {{
    "enabled": True,
    "camera_matrix": null,
    "dist_coeffs": [{k1:.6f}, {k2:.6f}, 0, 0, {k3:.6f}],
    "alpha": {alpha}
}}
''')

    cap.release()
    cv2.destroyAllWindows()


def main():
    print("=" * 60)
    print("摄像头畸变标定工具")
    print("Camera Distortion Calibration Tool")
    print("=" * 60)
    print("\n选择摄像头:")
    for i, cam in enumerate(CAMERAS):
        print(f"  [{i+1}] {cam['id']} - {cam['name']}")
    print(f"  [Q] 快速估算模式（无需标定板）")

    choice = input("\n请选择: ").strip()

    if choice.lower() == 'q':
        # 快速估算模式
        print("\n选择摄像头进行快速估算:")
        for i, cam in enumerate(CAMERAS):
            print(f"  [{i+1}] {cam['id']} - {cam['name']}")
        cam_choice = input("请选择: ").strip()
        try:
            idx = int(cam_choice) - 1
            if 0 <= idx < len(CAMERAS):
                quick_estimate(CAMERAS[idx])
        except:
            print("无效选择")
        return

    try:
        idx = int(choice) - 1
        if not (0 <= idx < len(CAMERAS)):
            print("无效选择")
            return
    except:
        print("无效选择")
        return

    camera_config = CAMERAS[idx]
    save_dir = os.path.join(os.path.dirname(__file__), '..', 'calibration_images', camera_config['id'])

    print("\n选择模式:")
    print("  [1] 采集标定图像并标定")
    print("  [2] 使用已有图像标定")
    print("  [3] 测试校正效果（需要已标定参数）")

    mode = input("请选择: ").strip()

    if mode == '1':
        # 采集并标定
        if capture_calibration_images(camera_config, save_dir):
            result = calibrate_from_images(save_dir)
            if result:
                camera_matrix, dist_coeffs, error = result
                test = input("\n是否测试校正效果? (y/n): ").strip().lower()
                if test == 'y':
                    test_undistortion(camera_config, camera_matrix, dist_coeffs)

    elif mode == '2':
        # 使用已有图像标定
        result = calibrate_from_images(save_dir)
        if result:
            camera_matrix, dist_coeffs, error = result
            test = input("\n是否测试校正效果? (y/n): ").strip().lower()
            if test == 'y':
                test_undistortion(camera_config, camera_matrix, dist_coeffs)

    elif mode == '3':
        # 测试已有参数
        undistort_config = camera_config.get('undistort')
        if undistort_config and undistort_config.get('enabled'):
            cm = np.array(undistort_config['camera_matrix']).reshape(3, 3) if undistort_config.get('camera_matrix') else None
            dc = np.array(undistort_config['dist_coeffs'])

            if cm is not None:
                test_undistortion(camera_config, cm, dc)
            else:
                print("警告: camera_matrix 为 None，无法测试")
                print("请先使用模式 1 或 2 进行标定")
        else:
            print("该摄像头未配置畸变校正参数")


if __name__ == "__main__":
    main()
