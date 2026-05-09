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


def test_summarize_replay_arrays_reports_aggregate_counts_and_weights() -> None:
    features, policies, values = make_arrays()

    summary = summarize_replay_arrays(
        features=features,
        policies=policies,
        values=values,
        counts=np.asarray([3, 1, 2], dtype=np.int64),
        sample_weights=np.asarray([1.7, 1.0, 1.4], dtype=np.float32),
        conflict_samples=0,
    )

    assert summary["aggregate_counts"]["duplicate_groups"] == 2
    assert summary["aggregate_counts"]["represented_raw_rows"] == 6
    assert summary["sample_weight"]["effective_weighted_rows"] == pytest.approx(4.1)


def test_summarize_replay_arrays_reports_move_count_duplicate_distribution() -> None:
    features, policies, values = make_arrays()
    for row, moves in enumerate([0, 10, 30]):
        remaining_total = 80 - moves
        features[row, 7, :, :] = (remaining_total / 2) / 40
        features[row, 8, :, :] = (remaining_total / 2) / 40

    summary = summarize_replay_arrays(
        features=features,
        policies=policies,
        values=values,
        counts=np.asarray([100, 4, 2], dtype=np.int64),
        sample_weights=np.asarray([10.0, 2.0, 1.4], dtype=np.float32),
        conflict_samples=0,
    )

    assert summary["move_count"]["unique_rows"]["max"] == 30.0
    assert summary["move_count"]["raw_rows"]["p50"] == 0.0
    assert summary["move_count"]["top_move_counts_by_raw_rows"][0]["move_count"] == 0
    assert summary["move_count"]["top_move_counts_by_raw_rows"][0]["raw_rows"] == pytest.approx(
        100.0
    )
    opening_bucket = summary["move_count"]["buckets"][0]
    assert opening_bucket["moves"] == "0-0"
    assert opening_bucket["raw_fraction"] == pytest.approx(100 / 106)


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


def test_summarize_replay_arrays_reports_target_vs_prior_diagnostics() -> None:
    features, policies, values = make_arrays()
    root_logits = np.full((3, ACTION_SPACE), -5.0, dtype=np.float32)
    root_logits[0, 0] = 5.0
    root_logits[1, 1] = 5.0
    root_logits[2, 0] = 5.0

    summary = summarize_replay_arrays(
        features=features,
        policies=policies,
        values=values,
        root_policy_logits=root_logits,
        conflict_samples=0,
    )

    diagnostics = summary["target_vs_prior"]
    assert diagnostics["available_rows"] == 3
    assert diagnostics["missing_rows"] == 0
    assert diagnostics["argmax_mismatch_count"] == 1
    assert diagnostics["argmax_mismatch_ratio"] == pytest.approx(1 / 3)
    assert diagnostics["kl_target_prior"]["mean"] > 0.0


def test_summarize_replay_arrays_counts_missing_root_logits() -> None:
    features, policies, values = make_arrays()
    root_logits = np.full((3, ACTION_SPACE), np.nan, dtype=np.float32)
    root_logits[0] = 0.0

    summary = summarize_replay_arrays(
        features=features,
        policies=policies,
        values=values,
        root_policy_logits=root_logits,
        conflict_samples=0,
    )

    assert summary["target_vs_prior"]["available_rows"] == 1
    assert summary["target_vs_prior"]["missing_rows"] == 2
