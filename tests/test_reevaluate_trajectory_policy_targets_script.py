from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / (
    "reevaluate_trajectory_policy_targets.py"
)
SPEC = importlib.util.spec_from_file_location("reevaluate_trajectory_policy_targets", SCRIPT_PATH)
assert SPEC is not None
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_sample_indexes_is_seeded_and_caps_to_replay_size() -> None:
    first = module.sample_indexes(5, 99, random.Random(3))
    second = module.sample_indexes(5, 99, random.Random(3))

    assert first.shape == (5,)
    assert sorted(first.tolist()) == [0, 1, 2, 3, 4]
    assert first.tolist() == second.tolist()


def test_masked_softmax_excludes_illegal_actions() -> None:
    logits = np.asarray([[0.0, 1.0, 100.0]], dtype=np.float32)
    legal_masks = np.asarray([[True, True, False]], dtype=np.bool_)

    policy = module.masked_softmax(logits, legal_masks)

    assert policy[0, 2] == pytest.approx(0.0)
    assert policy.sum() == pytest.approx(1.0)
    assert policy[0, 1] > policy[0, 0]


def test_compare_policy_to_target_reports_mismatch_and_illegal_mass() -> None:
    target = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    prior = np.asarray([[0.8, 0.2, 0.0], [0.1, 0.3, 0.6]], dtype=np.float32)
    legal_masks = np.asarray([[True, True, False], [True, True, False]], dtype=np.bool_)

    summary = module.compare_policy_to_target(
        target=target,
        prior=prior,
        legal_masks=legal_masks,
        top_k=3,
    )

    assert summary["argmax_mismatch_ratio"] == pytest.approx(0.5)
    assert summary["argmax_mismatch_count"] == 1
    assert summary["illegal_mass"]["mean"] == pytest.approx(0.3)
    assert summary["policy_cross_entropy"]["mean"] > 0.0


def test_summarize_stored_root_prior_sample_handles_missing_rows() -> None:
    target = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    legal_masks = np.asarray([[True, True], [True, True]], dtype=np.bool_)
    replay = SimpleNamespace(
        root_policy_logits=np.asarray([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32),
        root_policy_logits_present=np.asarray([True, False], dtype=np.bool_),
    )

    summary = module.summarize_stored_root_prior_sample(
        replay=replay,
        indexes=np.asarray([0, 1], dtype=np.int64),
        target=target,
        legal_masks=legal_masks,
        top_k=2,
    )

    assert summary["available_rows"] == 1
    assert summary["missing_rows"] == 1
    assert summary["argmax_mismatch_count"] == 0
