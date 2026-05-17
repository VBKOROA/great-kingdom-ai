from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

import numpy as np
import pytest
from great_kingdom_ai.features import ACTION_SPACE
from great_kingdom_ai.replay.schema import FEATURE_SHAPE

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "run_aggregate_weight_mode_comparison.py"
)
SPEC = importlib.util.spec_from_file_location("run_aggregate_weight_mode_comparison", SCRIPT_PATH)
assert SPEC is not None
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_write_reweighted_aggregate_replay_reuses_saved_counts(tmp_path: Path) -> None:
    source_path = tmp_path / "replay-aggregated.npz"
    output_path = tmp_path / "replay-count.npz"
    counts = np.asarray([1, 4, 9], dtype=np.int64)
    features = np.zeros((3, *FEATURE_SHAPE), dtype=np.float32)
    policies = np.zeros((3, ACTION_SPACE), dtype=np.float32)
    policies[:, 0] = 1.0
    values = np.asarray([0.0, 0.5, -0.5], dtype=np.float32)
    root_policy_logits = np.ones((3, ACTION_SPACE), dtype=np.float32)

    np.savez_compressed(
        source_path,
        capacity=np.asarray(99, dtype=np.int64),
        features=features,
        policies=policies,
        values=values,
        counts=counts,
        sample_weights=np.sqrt(counts.astype(np.float32)),
        root_policy_logits=root_policy_logits,
    )

    stats = module.write_reweighted_aggregate_replay(
        source_path=source_path,
        output_path=output_path,
        weight_mode="count",
        weight_cap=None,
        force=True,
    )

    with np.load(output_path) as data:
        assert np.array_equal(data["counts"], counts)
        assert np.array_equal(data["sample_weights"], counts.astype(np.float32))
        assert np.array_equal(data["features"], features)
        assert np.array_equal(data["policies"], policies)
        assert np.array_equal(data["values"], values)
        assert np.array_equal(data["root_policy_logits"], root_policy_logits)
        assert int(data["capacity"]) == 99

    assert stats["samples"] == 3
    assert stats["raw_rows_represented"] == 14
    assert stats["max_count"] == 9
    assert stats["max_weight"] == pytest.approx(9.0)


def test_write_reweighted_aggregate_replay_supports_none_mode(tmp_path: Path) -> None:
    source_path = tmp_path / "replay-aggregated.npz"
    output_path = tmp_path / "replay-none.npz"
    counts = np.asarray([1, 4, 9], dtype=np.int64)
    np.savez_compressed(
        source_path,
        capacity=np.asarray(3, dtype=np.int64),
        features=np.zeros((3, *FEATURE_SHAPE), dtype=np.float32),
        policies=np.zeros((3, ACTION_SPACE), dtype=np.float32),
        values=np.zeros((3,), dtype=np.float32),
        counts=counts,
        sample_weights=np.sqrt(counts.astype(np.float32)),
    )

    stats = module.write_reweighted_aggregate_replay(
        source_path=source_path,
        output_path=output_path,
        weight_mode="none",
        weight_cap=None,
        force=True,
    )

    with np.load(output_path) as data:
        assert np.array_equal(data["sample_weights"], np.ones((3,), dtype=np.float32))

    assert stats["max_weight"] == pytest.approx(1.0)
    assert stats["mean_weight"] == pytest.approx(1.0)


def test_load_training_replay_supports_recency_biased_sampling(tmp_path: Path) -> None:
    source_path = tmp_path / "replay-aggregated.npz"
    output_path = tmp_path / "replay-count.npz"
    counts = np.asarray([1, 4, 9], dtype=np.int64)
    features = np.zeros((3, *FEATURE_SHAPE), dtype=np.float32)
    for index in range(3):
        features[index, 0, 0, 0] = float(index)
    policies = np.zeros((3, ACTION_SPACE), dtype=np.float32)
    policies[:, 0] = 1.0
    values = np.asarray([0.0, 0.5, -0.5], dtype=np.float32)
    np.savez_compressed(
        source_path,
        capacity=np.asarray(3, dtype=np.int64),
        features=features,
        policies=policies,
        values=values,
        counts=counts,
        sample_weights=np.sqrt(counts.astype(np.float32)),
    )
    module.write_reweighted_aggregate_replay(
        source_path=source_path,
        output_path=output_path,
        weight_mode="count",
        weight_cap=None,
        force=True,
    )

    replay = module._load_training_replay(
        output_path,
        weight_mode="count",
        weight_cap=None,
    )
    batch = replay.sample_arrays(
        2,
        random.Random(0),
        recent_fraction=0.5,
        recent_window=2,
    )

    assert batch.features.shape == (2, *FEATURE_SHAPE)
    assert set(batch.sample_weights.tolist()).issubset({1.0, 4.0, 9.0})


def test_write_reweighted_aggregate_replay_requires_counts(tmp_path: Path) -> None:
    source_path = tmp_path / "replay-aggregated.npz"
    output_path = tmp_path / "replay-count.npz"
    np.savez_compressed(
        source_path,
        capacity=np.asarray(1, dtype=np.int64),
        features=np.zeros((1, *FEATURE_SHAPE), dtype=np.float32),
        policies=np.zeros((1, ACTION_SPACE), dtype=np.float32),
        values=np.zeros((1,), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="aggregate counts"):
        module.write_reweighted_aggregate_replay(
            source_path=source_path,
            output_path=output_path,
            weight_mode="count",
            weight_cap=None,
            force=True,
        )


def test_tail_loss_summary_averages_last_quarter() -> None:
    losses = [
        {"step": float(index), "total": float(index), "policy_kl": float(index * 2)}
        for index in range(1, 9)
    ]

    summary = module._tail_loss_summary(losses)

    assert summary["total"] == pytest.approx(7.5)
    assert summary["policy_kl"] == pytest.approx(15.0)


def test_parser_accepts_none_mode_with_sqrt_count_baseline() -> None:
    args = module.build_parser().parse_args(
        ["--weight-mode", "sqrt_count", "none", "--baseline-weight-mode", "sqrt_count"]
    )

    assert args.weight_mode == ["sqrt_count", "none"]
    assert args.baseline_weight_mode == "sqrt_count"
    assert module._arena_report_name("none", "sqrt_count") == "none-vs-sqrt_count-arena.json"
