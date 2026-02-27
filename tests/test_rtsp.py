"""
RTSP 摄像头连接测试工具
用于测试不同的 RTSP 流路径，找到正确的连接方式
"""
import cv2
import sys
import os

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import RTSP_HOST, RTSP_PORT, RTSP_USERNAME, RTSP_PASSWORD

# 常见的 RTSP 流路径列表（不同品牌摄像头使用不同路径）
COMMON_PATHS = [
    "",  # 无路径
    "/",  # 根路径
    "/stream1",  # 通用路径
    "/stream0",
    "/live",
    "/live/0",
    "/live/1",
    "/h264",
    "/h264/ch1/main/av_stream",  # 海康威视
    "/h264/ch1/sub/av_stream",   # 海康威视子码流
    "/Streaming/Channels/1",      # 海康威视新版
    "/Streaming/Channels/101",    # 海康威视主码流
    "/Streaming/Channels/102",    # 海康威视子码流
    "/cam/realmonitor?channel=1&subtype=0",  # 大华主码流
    "/cam/realmonitor?channel=1&subtype=1",  # 大华子码流
    "/video1",
    "/video2",
    "/11",  # 某些摄像头
    "/12",
    "/mpeg4",
    "/mjpeg",
    "/axis-media/media.amp",  # Axis 摄像头
]

def test_rtsp_path(rtsp_url, timeout=5):
    """
    测试单个 RTSP URL 是否可以连接
    """
    print(f"\n正在测试: {rtsp_url}")

    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

    # 设置超时和缓冲
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout * 1000)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout * 1000)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("  ❌ 连接失败")
        cap.release()
        return False

    # 尝试读取一帧
    ret, frame = cap.read()
    cap.release()

    if ret and frame is not None:
        print(f"  ✅ 连接成功! 分辨率: {frame.shape[1]}x{frame.shape[0]}")
        return True
    else:
        print("  ⚠️  连接建立但无法读取画面")
        return False

def main():
    print("=" * 60)
    print("RTSP 摄像头连接测试工具")
    print("=" * 60)
    print(f"摄像头信息:")
    print(f"  IP地址: {RTSP_HOST}")
    print(f"  端口: {RTSP_PORT}")
    print(f"  用户名: {RTSP_USERNAME}")
    print(f"  密码: {'*' * len(RTSP_PASSWORD)}")
    print("=" * 60)

    successful_urls = []

    print(f"\n开始测试 {len(COMMON_PATHS)} 个常见路径...")
    print("(每个测试最多等待 5 秒)")

    for i, path in enumerate(COMMON_PATHS, 1):
        rtsp_url = f"rtsp://{RTSP_USERNAME}:{RTSP_PASSWORD}@{RTSP_HOST}:{RTSP_PORT}{path}"
        print(f"\n[{i}/{len(COMMON_PATHS)}]", end="")

        if test_rtsp_path(rtsp_url):
            successful_urls.append((path, rtsp_url))

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)

    if successful_urls:
        print(f"\n✅ 找到 {len(successful_urls)} 个可用的 RTSP 路径:\n")
        for i, (path, url) in enumerate(successful_urls, 1):
            print(f"{i}. 路径: {path if path else '(空)'}")
            print(f"   完整URL: {url}")
            print(f"   配置方式: RTSP_STREAM_PATH = \"{path}\"")
            print()

        print("请将上述路径配置到 config.py 中的 RTSP_STREAM_PATH")
    else:
        print("\n❌ 未找到可用的 RTSP 路径")
        print("\n可能的原因:")
        print("1. 网络连接问题 - 请确认摄像头和电脑在同一网络")
        print("2. 用户名或密码错误")
        print("3. 摄像头使用了非标准的流路径")
        print("4. 摄像头的 RTSP 功能未启用")
        print("5. 防火墙阻止了连接")
        print("\n建议:")
        print("- 在浏览器中访问摄像头的 Web 管理界面查看 RTSP 设置")
        print("- 使用 VLC 播放器测试 RTSP 连接")
        print("- 查看摄像头的用户手册获取正确的 RTSP URL 格式")

if __name__ == "__main__":
    main()
