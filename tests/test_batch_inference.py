import os
import sys
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

# Stub heavy GPU/model dependencies before importing face_engine
for mod in [
    'cv2', 'onnxruntime', 'onnx',
    'insightface', 'insightface.app', 'insightface.model_zoo',
    'insightface.model_zoo.model_zoo',
    'insightface.utils', 'insightface.utils.face_align',
    'utils.cv_utils',
]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

from utils.face_engine import FaceEngine


def _make_engine():
    """创建带 mock 模型的 FaceEngine，无需真实 GPU"""
    engine = FaceEngine.__new__(FaceEngine)
    engine.det_thresh = 0.5

    mock_face = MagicMock()
    mock_face.det_score = 0.9
    mock_face.bbox = np.array([10.0, 10.0, 50.0, 50.0])
    mock_face.kps = np.zeros((5, 2), dtype=np.float32)

    engine.det_model = MagicMock()
    engine.det_model.get.return_value = [mock_face]

    engine.rec_model = MagicMock()
    # 返回 (N, 512)，N = 调用时传入的图片数量
    engine.rec_model.get_feat.side_effect = lambda imgs: np.random.rand(len(imgs), 512)

    return engine


def test_batch_returns_one_list_per_frame():
    engine = _make_engine()
    frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(3)]

    with patch('utils.face_engine.face_align') as mock_align:
        mock_align.norm_crop.return_value = np.zeros((112, 112, 3), dtype=np.uint8)
        results = engine.batch_detect_and_extract(frames)

    assert len(results) == 3


def test_batch_calls_get_feat_once_for_all_faces():
    engine = _make_engine()
    frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(3)]

    with patch('utils.face_engine.face_align') as mock_align:
        mock_align.norm_crop.return_value = np.zeros((112, 112, 3), dtype=np.uint8)
        engine.batch_detect_and_extract(frames)

    # 3帧各1张脸 = 一次 get_feat 调用，传入列表长度为3
    engine.rec_model.get_feat.assert_called_once()
    call_arg = engine.rec_model.get_feat.call_args[0][0]
    assert len(call_arg) == 3


def test_batch_empty_frames_returns_empty_lists():
    engine = _make_engine()
    engine.det_model.get.return_value = []  # 无人脸
    frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(2)]

    with patch('utils.face_engine.face_align'):
        results = engine.batch_detect_and_extract(frames)

    assert results == [[], []]
    engine.rec_model.get_feat.assert_not_called()


def test_batch_face_dict_has_required_keys():
    engine = _make_engine()
    frames = [np.zeros((480, 640, 3), dtype=np.uint8)]

    with patch('utils.face_engine.face_align') as mock_align:
        mock_align.norm_crop.return_value = np.zeros((112, 112, 3), dtype=np.uint8)
        results = engine.batch_detect_and_extract(frames)

    face = results[0][0]
    for key in ('bbox', 'landmarks', 'embedding', 'aligned_face', 'det_score'):
        assert key in face, f"missing key: {key}"
    assert face['embedding'].shape == (512,)


def test_batch_embeddings_routed_to_correct_frame():
    engine = _make_engine()
    sentinel_0 = np.full(512, 0.1, dtype=np.float32)
    sentinel_1 = np.full(512, 0.9, dtype=np.float32)
    engine.rec_model.get_feat.side_effect = lambda imgs: np.stack([sentinel_0, sentinel_1])
    frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(2)]

    with patch('utils.face_engine.face_align') as mock_align:
        mock_align.norm_crop.return_value = np.zeros((112, 112, 3), dtype=np.uint8)
        results = engine.batch_detect_and_extract(frames)

    np.testing.assert_array_almost_equal(results[0][0]['embedding'], sentinel_0)
    np.testing.assert_array_almost_equal(results[1][0]['embedding'], sentinel_1)
