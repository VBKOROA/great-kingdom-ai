from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from great_kingdom_ai.features import BOARD_SIZE, FEATURE_CHANNELS
from great_kingdom_ai.onnx_quantization import (
    FeatureCalibrationDataReader,
    _calibration_features,
)

FEATURE_SHAPE = (FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE)


def test_feature_calibration_reader_batches_and_rewinds() -> None:
    features = np.arange(3 * np.prod(FEATURE_SHAPE), dtype=np.float32).reshape(3, *FEATURE_SHAPE)
    reader = FeatureCalibrationDataReader(features, batch_size=2)

    first = reader.get_next()
    second = reader.get_next()
    done = reader.get_next()
    reader.rewind()
    repeated = reader.get_next()

    assert first is not None
    assert second is not None
    assert done is None
    assert first["features"].shape == (2, *FEATURE_SHAPE)
    assert second["features"].shape == (1, *FEATURE_SHAPE)
    assert repeated is not None
    assert np.array_equal(repeated["features"], first["features"])


def test_calibration_features_can_load_npz_features(tmp_path: Path) -> None:
    features = np.zeros((5, *FEATURE_SHAPE), dtype=np.float32)
    features[:, 3, :, :] = 1.0
    path = tmp_path / "replay.npz"
    np.savez_compressed(path, features=features)

    selected, source = _calibration_features(
        calibration_features_path=path,
        sample_count=3,
        seed=7,
    )

    assert source == str(path)
    assert selected.shape == (3, *FEATURE_SHAPE)
    assert selected.dtype == np.float32


def test_synthetic_calibration_features_match_model_input_shape() -> None:
    features, source = _calibration_features(
        calibration_features_path=None,
        sample_count=4,
        seed=0,
    )

    assert source == "synthetic"
    assert features.shape == (4, *FEATURE_SHAPE)
    assert np.isfinite(features).all()
    assert features.min() >= 0.0
    assert features.max() <= 1.0


def test_calibration_features_reject_wrong_shape(tmp_path: Path) -> None:
    path = tmp_path / "bad.npy"
    np.save(path, np.zeros((2, 3), dtype=np.float32))

    with pytest.raises(ValueError, match="expected calibration features shape"):
        _calibration_features(calibration_features_path=path, sample_count=1, seed=0)
