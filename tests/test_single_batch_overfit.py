from __future__ import annotations

import importlib
import importlib.util

import pytest

_torch_spec = importlib.util.find_spec("torch")
pytestmark = pytest.mark.skipif(
    _torch_spec is None,
    reason="torch is not installed",
)
torch = importlib.import_module("torch") if _torch_spec is not None else None

from _training_helpers import make_replay  # noqa: E402
from great_kingdom_ai.single_batch_overfit import run_single_batch_overfit  # noqa: E402
from great_kingdom_ai.training import TrainingConfig  # noqa: E402


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
