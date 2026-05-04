from __future__ import annotations

import importlib
import importlib.util

import numpy as np
import pytest

_torch_spec = importlib.util.find_spec("torch")
pytestmark = pytest.mark.skipif(
    _torch_spec is None,
    reason="torch is not installed",
)
torch = importlib.import_module("torch") if _torch_spec is not None else None

from great_kingdom_ai.features import ACTION_SPACE, BOARD_SIZE, FEATURE_CHANNELS  # noqa: E402
from great_kingdom_ai.replay_buffer import ReplayBuffer, ReplaySample  # noqa: E402
from great_kingdom_ai.single_batch_overfit import run_single_batch_overfit  # noqa: E402
from great_kingdom_ai.train import TrainingConfig  # noqa: E402


def make_sample(index: int, value: float = 1.0) -> ReplaySample:
    features = np.zeros((FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    features[index % FEATURE_CHANNELS, index % BOARD_SIZE, (index * 3) % BOARD_SIZE] = 1.0
    features[4, :, :] = 1.0
    policy = np.zeros(ACTION_SPACE, dtype=np.float32)
    policy[index % ACTION_SPACE] = 1.0
    return ReplaySample(features=features, policy=policy, value=value)


def make_replay(size: int = 4) -> ReplayBuffer:
    replay = ReplayBuffer(capacity=size)
    for index in range(size):
        replay.push(make_sample(index, value=1.0 if index % 2 else -1.0))
    return replay


def test_single_batch_overfit_saves_checkpoint_and_logs_losses(tmp_path) -> None:
    checkpoint_path = tmp_path / "overfit.pt"
    config = TrainingConfig(
        batch_size=2,
        steps=2,
        learning_rate=1e-3,
        seed=7,
        model_preset="small",
        symmetry_augmentation=False,
    )

    summary = run_single_batch_overfit(
        make_replay(),
        config,
        checkpoint_path=checkpoint_path,
        log_every=1,
    )

    assert summary.start_step == 0
    assert summary.end_step == 2
    assert checkpoint_path.is_file()
    assert summary.checkpoint_path == checkpoint_path
    assert summary.losses[0]["step"] == 0.0
    assert summary.losses[-1]["step"] == 2.0
    assert summary.initial_loss["total"] > 0.0
    assert summary.final_loss["total"] > 0.0


def test_single_batch_overfit_requires_enough_replay_samples() -> None:
    config = TrainingConfig(batch_size=3, steps=1)

    with pytest.raises(ValueError, match="at least batch_size"):
        run_single_batch_overfit(make_replay(size=2), config)
