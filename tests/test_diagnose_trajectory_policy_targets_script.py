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


class FakeReplay:
    def __init__(self) -> None:
        self.episode_offsets = np.asarray([0, 3], dtype=np.int64)
        self.turn_offsets = np.asarray([0, 3], dtype=np.int64)
        self.timesteps = np.asarray([0, 1, 2], dtype=np.int64)
        self.players = np.asarray([1, 2, 1], dtype=np.int64)
        self.turn_players = np.asarray([1, 2, 1], dtype=np.int64)
        self.turn_root_values = np.asarray([0.1, -0.4, 0.7], dtype=np.float32)
        self.episode_winners = np.asarray([1], dtype=np.int64)
        self.episode_count = 1

    def __len__(self) -> int:
        return 3


class FakeSummaryReplay(FakeReplay):
    def __init__(self) -> None:
        super().__init__()
        self.capacity = 8
        self.policy_targets = np.asarray(
            [[1.0, 0.0], [0.25, 0.75], [0.5, 0.5]],
            dtype=np.float32,
        )
        self.legal_masks = None
        self.features = None
        self.root_policy_logits = np.zeros((3, 2), dtype=np.float32)
        self.model_versions = np.asarray([1, 1, 2], dtype=np.int64)
        self.created_iterations = np.asarray([4, 4, 5], dtype=np.int64)


def test_summarize_trajectory_policy_targets_tolerates_missing_legal_masks() -> None:
    replay = FakeSummaryReplay()

    summary = module.summarize_trajectory_policy_targets(
        replay,
        value_config={
            "bootstrap_td_steps": 0,
            "gamma": 1.0,
            "value_bootstrap_source": "terminal",
        },
    )

    assert summary["policy_target"]["legal_mask_source"] == "missing"
    assert summary["policy_target"]["illegal_mass"] is None
    assert summary["root_prior"]["available_rows"] == 0
    assert "legal_masks or features" in summary["root_prior"]["missing_reason"]


def test_mcts_root_bootstrap_targets_match_training_dataset_logic() -> None:
    replay = FakeReplay()

    targets = module.training_value_targets(
        replay,
        bootstrap_td_steps=1,
        gamma=1.0,
        value_bootstrap_source="mcts_root",
    )

    assert targets.tolist() == pytest.approx([0.4, -0.7, 1.0])


def test_resolve_value_target_config_allows_cli_overrides() -> None:
    resolved = module.resolve_value_target_config(
        train_config_path=None,
        bootstrap_td_steps=2,
        gamma=0.5,
        value_bootstrap_source="terminal",
    )

    assert resolved == {
        "bootstrap_td_steps": 2,
        "gamma": 0.5,
        "value_bootstrap_source": "terminal",
    }
