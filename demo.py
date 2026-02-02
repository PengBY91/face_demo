import cv2
import numpy as np
import os
import threading
import time
import requests
import base64
from sklearn.metrics.pairwise import cosine_similarity

# 导入新模块
from config import (
    ARCFACE_MODEL_PATH, PROVIDERS,
    GALLERY_DIR, DET_THRESH,
    VIDEO_PATH, CAMERA_ID, USE_CAMERA,
    SIMILARITY_THRESHOLD, SYNC_INTERVAL,
    SERVER_HOST, SERVER_PORT
)
from utils.face_engine import FaceEngine
from utils.gallery_manager import GalleryManager


class HistorySampler:
    """
    10帧采样器：每隔10帧保存一个周期内识别出的每个人的最好结果（置信度最高）
    """
    def __init__(self, server_url="http://127.0.0.1:8008"):
        self.server_url = server_url
        self.frame_count = 0
        self.window_size = 10
        # 缓存当前窗口内的最佳结果 {name: {"score": score, "image": image}}
        self.best_results = {}

    def process_frame(self, detections):
        """
        detections: List[Dict] with 'name', 'score', 'aligned_face'
        """
        self.frame_count += 1
        
        for det in detections:
            name = det['name']
            if name == "Unknown":
                continue
            
            score = det['score']
            if name not in self.best_results or score > self.best_results[name]['score']:
                self.best_results[name] = {
                    "score": score,
                    "image": det['aligned_face'].copy()
                }
        
        # 周期结束，上报并清空
        if self.frame_count >= self.window_size:
            self._flush()
            self.frame_count = 0
            self.best_results = {}

    def _flush(self):
        if not self.best_results:
            return
            
        records = []
        for name, data in self.best_results.items():
            try:
                # 将图片转换为字节流并转为 base64
                _, img_encoded = cv2.imencode('.jpg', data['image'])
                img_base64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')
                
                records.append({
                    'name': name,
                    'confidence': float(data['score']),
                    'image_b64': img_base64
                })
            except Exception as e:
                print(f"采样器: 准备数据异常 {name}: {e}")

        if records:
            try:
                resp = requests.post(f"{self.server_url}/api/records_batch", json=records)
                if resp.status_code == 200:
                    print(f"采样器: 已批量保存 {len(records)} 条识别结果")
                else:
                    print(f"采样器: 批量保存失败: {resp.text}")
            except Exception as e:
                print(f"采样器: 批量上报异常: {e}")


class DemoClient:
    def __init__(self):
        # 1. 初始化人脸引擎
        print(f"客户端: 初始化 FaceEngine...")
        self.engine = FaceEngine(
            rec_model_path=ARCFACE_MODEL_PATH,
            providers=PROVIDERS,
            det_thresh=DET_THRESH
        )

        # 2. 初始化库管理器
        self.gallery = GalleryManager(GALLERY_DIR)

        # 3. 本地特征库和照片路径
        self.feature_db_vectors = np.empty((0, 512))
        self.feature_db_names = []

        # 4. 加载本地人脸库
        self.load_gallery()

        # 5. 初始化历史采样器
        self.sampler = HistorySampler(server_url=f"http://{SERVER_HOST.replace('0.0.0.0', '127.0.0.1')}:{SERVER_PORT}")

        # 6. 启动后台同步线程
        self.sync_thread = threading.Thread(target=self.sync_loop, daemon=True)
        self.sync_thread.start()

    def load_gallery(self):
        """从 GalleryManager 加载特征和照片路径"""
        print(f"客户端: 从 {GALLERY_DIR} 加载人脸库...")

        names, embeddings = self.gallery.load_embeddings()

        if len(names) > 0:
            # 特征库名字和向量
            self.feature_db_names = names
            self.feature_db_vectors = embeddings

            print(f"客户端: 成功加载 {len(names)} 个人脸特征")
        else:
            print("客户端: 未在库中找到有效的人脸特征")

    def sync_loop(self):
        """后台定期重新加载人脸库，实现与服务端同步"""
        while True:
            time.sleep(SYNC_INTERVAL)
            print("客户端: 同步人脸库...")
            self.load_gallery()

    def identify(self, face_embedding):
        """识别单个人脸"""
        if face_embedding is None or len(self.feature_db_vectors) == 0:
            return "Unknown", 0.0, None

        sims = cosine_similarity([face_embedding], self.feature_db_vectors)[0]
        best_idx = np.argmax(sims)
        best_score = sims[best_idx]

        if best_score > SIMILARITY_THRESHOLD:
            return self.feature_db_names[best_idx], best_score, None

        return "Unknown", best_score, None

    def run(self):
        if USE_CAMERA:
            print(f"客户端: 正在打开摄像头 (ID: {CAMERA_ID})...")
            cap = cv2.VideoCapture(CAMERA_ID)
            source_name = "Camera"
        else:
            print(f"客户端: 正在打开视频文件 {VIDEO_PATH}...")
            cap = cv2.VideoCapture(VIDEO_PATH)
            source_name = f"Video ({VIDEO_PATH})"

        if not cap.isOpened():
            print(f"客户端: 错误 - 无法打开 {source_name}")
            return

        print(f"客户端: 开始测试 ({source_name})，按 Q 退出")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            display_frame = frame.copy()

            # 使用 FaceEngine 检测和提取特征
            faces = self.engine.detect_and_extract(frame)

            for i, face in enumerate(faces):
                # 获取边界框和特征
                bbox = face['bbox']
                x1, y1, x2, y2 = bbox
                embedding = face['embedding']

                # 识别
                name, score, gallery_img_path = self.identify(embedding)

                # 绘制边界框和名字
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display_frame, f"{name} ({score:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                # 显示匹配的照片
                gallery_img = self.gallery.get_face_image(name)
                if gallery_img is not None:
                    gw = 120
                    gh = int(gallery_img.shape[0] * (gw / gallery_img.shape[1]))
                    gallery_img = cv2.resize(gallery_img, (gw, gh))
                    pos_x, pos_y = display_frame.shape[1] - gw - 10, 10 + i * (gh + 30)

                    if pos_y + gh < display_frame.shape[0]:
                        cv2.putText(display_frame, f"Match: {name}", (pos_x, pos_y - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        display_frame[pos_y:pos_y + gh, pos_x:pos_x + gw] = gallery_img
                        cv2.rectangle(display_frame, (pos_x, pos_y), (pos_x + gw, pos_y + gh), (0, 255, 0), 2)

                # 记录检测结果用于采样
                face['name'] = name
                face['score'] = score

            # 采样处理
            self.sampler.process_frame(faces)

            cv2.imshow('Face Recognition Demo', display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    client = DemoClient()
    client.run()
