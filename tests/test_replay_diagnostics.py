from __future__ import annotations

import numpy as np
import pytest
from great_kingdom_ai.features import ACTION_SPACE, BOARD_SIZE, FEATURE_CHANNELS
from great_kingdom_ai.replay_diagnostics import summarize_replay_arrays


def make_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.zeros((3, FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    features[:, 4, :, :] = 1.0
    policies = np.zeros((3, ACTION_SPACE), dtype=np.float32)
    policies[0, 0] = 1.0
    policies[1, 1] = 0.5
    policies[1, 2] = 0.5
    policies[2, 81] = 1.0
    values = np.asarray([-1.0, 0.0, 1.0], dtype=np.float32)
    return features, policies, values


def test_summarize_replay_arrays_reports_policy_value_and_legal_stats() -> None:
    features, policies, values = make_arrays()

    summary = summarize_replay_arrays(
        features=features,
        policies=policies,
        values=values,
        capacity=10,
        conflict_samples=0,
    )

    assert summary["samples"] == 3
    assert summary["capacity"] == 10
    assert summary["policy"]["support"]["max"] == 2.0
    assert summary["policy"]["argmax_top"][0]["action"] == 0
    assert summary["value"]["negative_fraction"] == pytest.approx(1 / 3)
    assert summary["value"]["drawish_fraction"] == pytest.approx(1 / 3)
    assert summary["value"]["positive_fraction"] == pytest.approx(1 / 3)
    assert summary["legal"]["rows_with_illegal_target_mass"] == 0


def test_summarize_replay_arrays_detects_illegal_target_mass() -> None:
    features, policies, values = make_arrays()
    features[:, 4, :, :] = 0.0

    summary = summarize_replay_arrays(
        features=features,
        policies=policies,
        values=values,
        conflict_samples=0,
    )

    assert summary["legal"]["rows_with_illegal_target_mass"] == 2
    assert summary["legal"]["illegal_target_mass"]["max"] == 1.0


def test_summarize_replay_arrays_rejects_shape_mismatch() -> None:
    features, policies, values = make_arrays()

    with pytest.raises(ValueError, match="expected policies shape"):
        summarize_replay_arrays(
            features=features,
            policies=policies[:, :-1],
            values=values,
        )
