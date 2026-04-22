import os
import threading
import time
import unittest
from unittest.mock import MagicMock, patch
import sys
import numpy as np

# Ensure the project root is on sys.path so multi_camera_demo can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Stub heavy dependencies that require GPU/models
for mod in [
    'cv2', 'onnxruntime', 'onnx',
    'insightface', 'insightface.app', 'insightface.model_zoo', 'insightface.utils',
    'insightface.utils.face_align',
    'sklearn', 'sklearn.metrics', 'sklearn.metrics.pairwise',
    'PIL', 'PIL.Image', 'PIL.ImageDraw', 'PIL.ImageFont',
    'utils.face_engine', 'utils.gallery_manager', 'utils.cv_utils',
]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

# cv2.imencode needs to return a real tuple
import cv2 as _cv2_mock
_cv2_mock.imencode = lambda fmt, img: (True, MagicMock(tobytes=lambda: b'fake_jpeg'))

from multi_camera_demo import CameraThread


def _make_thread():
    cam_cfg = {
        'id': 'cam_test', 'name': 'Test', 'type': 'rtsp',
        'host': '127.0.0.1', 'port': 554, 'username': '',
        'password': '', 'stream_path': '/test', 'undistort': None
    }
    gallery = MagicMock()
    gallery.get_face_image.return_value = None
    import queue as q_mod
    feature_db = {'vectors': np.empty((0, 512)), 'names': []}
    feature_lock = threading.Lock()
    det_queue = q_mod.Queue(maxsize=100)
    return CameraThread(cam_cfg, gallery, det_queue, feature_db, feature_lock)


class TestFlushThread(unittest.TestCase):
    def test_flush_does_not_block_detection_thread(self):
        """_process_detection 应在 50ms 内返回，即使 HTTP 请求耗时 2 秒"""
        t = _make_thread()

        slow_response = MagicMock()
        slow_response.status_code = 200

        def slow_post(*args, **kwargs):
            time.sleep(2)
            return slow_response

        t.session.post = slow_post
        t.best_results = {
            'Alice': {
                'score': 0.9,
                'image': np.zeros((112, 112, 3), dtype=np.uint8)
            }
        }
        t.last_flush_time = 0  # force flush trigger

        t._start_flush_thread()

        start = time.time()
        face_img = np.zeros((112, 112, 3), dtype=np.uint8)
        t._process_detection('Alice', 0.9, face_img)
        elapsed = time.time() - start

        self.assertLess(elapsed, 0.05,
                        f"_process_detection blocked for {elapsed:.3f}s — HTTP leak into detection thread")

        t._stop_flush_thread()

    def test_flush_thread_started_in_run_teardown(self):
        """_start_flush_thread 和 _stop_flush_thread 应正常工作"""
        t = _make_thread()
        t._start_flush_thread()
        self.assertIsNotNone(t._flush_thread_obj)
        self.assertTrue(t._flush_thread_obj.is_alive())
        t._stop_flush_thread()
        # After stop, thread should exit within timeout
        t._flush_thread_obj.join(timeout=3)
        self.assertFalse(t._flush_thread_obj.is_alive())


if __name__ == '__main__':
    unittest.main()
