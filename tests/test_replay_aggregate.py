from __future__ import annotations

import numpy as np
import pytest
from great_kingdom_ai.features import ACTION_SPACE, BOARD_SIZE, FEATURE_CHANNELS
from great_kingdom_ai.replay_aggregate import (
    aggregate_duplicate_replay,
    load_replay,
    save_replay,
)


def make_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.zeros((3, FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    features[0, 0, 0, 0] = 1.0
    features[1] = features[0]
    features[2, 0, 0, 1] = 1.0

    policies = np.zeros((3, ACTION_SPACE), dtype=np.float32)
    policies[0, 0] = 1.0
    policies[1, 1] = 1.0
    policies[2, 2] = 1.0
    values = np.asarray([1.0, -1.0, 1.0], dtype=np.float32)
    return features, policies, values


def test_aggregate_duplicate_replay_averages_targets() -> None:
    features, policies, values = make_arrays()

    replay = aggregate_duplicate_replay(
        features=features,
        policies=policies,
        values=values,
        capacity=10,
    )

    assert replay.capacity == 10
    assert replay.features.shape[0] == 2
    assert replay.counts.tolist() == [2, 1]
    assert replay.policies[0, 0] == pytest.approx(0.5)
    assert replay.policies[0, 1] == pytest.approx(0.5)
    assert replay.policies[0].sum() == pytest.approx(1.0)
    assert replay.values[0] == pytest.approx(0.0)
    assert replay.values[1] == pytest.approx(1.0)


def test_aggregate_duplicate_replay_save_and_load_round_trip(tmp_path) -> None:
    features, policies, values = make_arrays()
    replay = aggregate_duplicate_replay(features=features, policies=policies, values=values)
    output_path = tmp_path / "aggregated.npz"

    save_replay(output_path, replay)
    loaded_features, loaded_policies, loaded_values, capacity = load_replay(output_path)

    assert capacity == 2
    assert np.array_equal(loaded_features, replay.features)
    assert np.allclose(loaded_policies, replay.policies)
    assert np.allclose(loaded_values, replay.values)


def test_aggregate_duplicate_replay_rejects_shape_mismatch() -> None:
    features, policies, values = make_arrays()

    with pytest.raises(ValueError, match="expected values shape"):
        aggregate_duplicate_replay(
            features=features,
            policies=policies,
            values=values[:-1],
        )
