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


def _make_thread(engine_init_semaphore=None):
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
    return CameraThread(cam_cfg, gallery, det_queue, feature_db, feature_lock,
                        engine_init_semaphore=engine_init_semaphore)


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


class TestFrameLock(unittest.TestCase):
    def test_get_latest_frame_returns_consistent_pair(self):
        """get_latest_frame() 应返回 (frame, detections) 的一致快照"""
        t = _make_thread()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[0, 0] = [1, 2, 3]  # mark to verify copy fidelity
        detections = [{'bbox': [10, 10, 50, 50], 'name': 'Alice',
                       'score': 0.9, 'is_suspicious': False, 'aligned_face': None}]
        t.update_latest_frame(frame, detections)

        f, d = t.get_latest_frame()
        self.assertIsNotNone(f)
        self.assertEqual(f.shape, (480, 640, 3))
        np.testing.assert_array_equal(f[0, 0], [1, 2, 3])
        self.assertEqual(len(d), 1)
        self.assertEqual(d[0]['name'], 'Alice')

    def test_get_latest_frame_returns_none_before_update(self):
        """未更新时 get_latest_frame() 应返回 (None, [])"""
        t = _make_thread()
        f, d = t.get_latest_frame()
        self.assertIsNone(f)
        self.assertEqual(d, [])

    def test_update_frame_is_independent_copy(self):
        """update_latest_frame 存储副本，外部修改不影响内部状态"""
        t = _make_thread()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        t.update_latest_frame(frame, [])
        frame[0, 0] = [99, 99, 99]  # mutate original after update
        f, _ = t.get_latest_frame()
        self.assertFalse(np.array_equal(f[0, 0], [99, 99, 99]),
                         "Internal frame should not be affected by external mutation")


class TestFaceEngineInitSerialization(unittest.TestCase):
    def test_semaphore_parameter_accepted(self):
        """CameraThread 应接受 engine_init_semaphore 参数"""
        sem = threading.Semaphore(1)
        t = _make_thread(engine_init_semaphore=sem)
        self.assertIs(t.engine_init_semaphore, sem)

    def test_default_semaphore_created_when_not_provided(self):
        """未提供 semaphore 时应自动创建 Semaphore(1)"""
        t = _make_thread()
        self.assertIsNotNone(t.engine_init_semaphore)
        self.assertIsInstance(t.engine_init_semaphore, type(threading.Semaphore(1)))


if __name__ == '__main__':
    unittest.main()
