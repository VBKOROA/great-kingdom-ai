from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / (
    "diagnose_async_update_pressure.py"
)
SPEC = importlib.util.spec_from_file_location("diagnose_async_update_pressure", SCRIPT_PATH)
assert SPEC is not None
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_estimate_sampling_pressure_reports_recent_overweight() -> None:
    pressure = module.estimate_sampling_pressure(
        replay_rows=512_000,
        batch_size=256,
        max_train_steps=128,
        recent_fraction=0.35,
        recent_window=65_536,
        train_reuse_factor=8.0,
        min_replay_transitions=8192,
        imported_transitions=8192,
    )

    assert pressure.train_steps == 128
    assert pressure.train_draws == 32_768
    assert pressure.recent_draws == pytest.approx(11_520.0)
    assert pressure.old_draws == pytest.approx(21_248.0)
    assert pressure.natural_recent_fraction == pytest.approx(0.128)
    assert pressure.effective_recent_fraction == pytest.approx(90 / 256)
    assert pressure.recent_fraction_overweight == pytest.approx((90 / 256) / 0.128)
    assert pressure.recent_row_overweight is not None
    assert pressure.recent_row_overweight > 3.0


def test_estimate_sampling_pressure_uniform_when_recent_fraction_disabled() -> None:
    pressure = module.estimate_sampling_pressure(
        replay_rows=100,
        batch_size=10,
        max_train_steps=5,
        recent_fraction=0.0,
        recent_window=20,
        train_reuse_factor=1.0,
        min_replay_transitions=0,
        imported_transitions=None,
    )

    assert pressure.recent_draws == pytest.approx(10.0)
    assert pressure.old_draws == pytest.approx(40.0)
    assert pressure.recent_row_overweight == pytest.approx(1.0)
    assert pressure.recent_fraction_overweight == pytest.approx(1.0)


def test_masked_log_softmax_and_policy_kl_ignore_illegal_actions() -> None:
    logits = np.asarray([[0.0, 1.0, 100.0]], dtype=np.float32)
    legal = np.asarray([[True, True, False]], dtype=np.bool_)
    policy = np.asarray([[0.25, 0.75, 0.0]], dtype=np.float32)

    log_probs = module.masked_log_softmax(logits, legal)
    kl = module.categorical_kl_from_log_probs(policy, log_probs)

    assert np.isneginf(log_probs[0, 2])
    assert np.isfinite(kl).all()
    assert kl[0] >= 0.0


def test_make_probe_indexes_splits_recent_and_old_ranges() -> None:
    indexes = module.make_probe_indexes(
        replay_rows=100,
        rows_per_split=8,
        recent_window=20,
        seed=7,
    )

    assert indexes["all"].shape == (8,)
    assert indexes["old"].shape == (8,)
    assert indexes["recent"].shape == (8,)
    assert indexes["old"].max() < 80
    assert indexes["recent"].min() >= 80
