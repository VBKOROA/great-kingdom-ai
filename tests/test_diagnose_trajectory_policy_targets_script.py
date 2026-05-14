from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / (
    "diagnose_trajectory_policy_targets.py"
)
SPEC = importlib.util.spec_from_file_location("diagnose_trajectory_policy_targets", SCRIPT_PATH)
assert SPEC is not None
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_masked_softmax_excludes_illegal_logits() -> None:
    logits = np.asarray([[0.0, 1.0, 100.0]], dtype=np.float32)
    legal = np.asarray([[True, True, False]], dtype=np.bool_)

    policy = module.masked_softmax(logits, legal)

    assert policy[0, 2] == pytest.approx(0.0)
    assert policy.sum() == pytest.approx(1.0)
    assert policy[0, 1] > policy[0, 0]


def test_categorical_kl_is_zero_for_matching_distributions() -> None:
    policy = np.asarray([[0.25, 0.75]], dtype=np.float32)

    kl = module.categorical_kl(policy, policy)

    assert kl.tolist() == pytest.approx([0.0])


def test_summarize_root_prior_reports_argmax_mismatch() -> None:
    target = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    logits = np.asarray([[2.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    legal = np.asarray([[True, True], [True, True]], dtype=np.bool_)
    available = np.asarray([True, True], dtype=np.bool_)

    summary = module.summarize_root_prior(
        target=target,
        root_logits=logits,
        legal_masks=legal,
        available=available,
        top_k=2,
    )

    assert summary["available_rows"] == 2
    assert summary["argmax_mismatch_ratio"] == pytest.approx(0.5)
    assert summary["max_probability"]["mean"] > 0.5
