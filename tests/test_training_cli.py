from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_torch_spec = importlib.util.find_spec("torch")
pytestmark = pytest.mark.skipif(
    _torch_spec is None,
    reason="torch is not installed",
)

from _training_helpers import make_replay  # noqa: E402
from great_kingdom_ai.training.cli import build_parser, print_training_startup_config  # noqa: E402
from great_kingdom_ai.training.config import TrainingConfig  # noqa: E402


def test_train_parser_accepts_log_every_override() -> None:
    args = build_parser().parse_args(
        [
            "--replay",
            "replay.npz",
            "--checkpoint",
            "checkpoint.pt",
            "--log-every",
            "100",
        ]
    )

    assert args.log_every == 100


def test_train_parser_accepts_weight_bootstrap_checkpoint() -> None:
    args = build_parser().parse_args(
        [
            "--replay",
            "replay.npz",
            "--checkpoint",
            "checkpoint.pt",
            "--bootstrap-weights",
            "best.pt",
        ]
    )

    assert args.bootstrap_weights == Path("best.pt")


def test_train_parser_accepts_ema_decay() -> None:
    args = build_parser().parse_args(
        [
            "--replay",
            "replay.npz",
            "--checkpoint",
            "checkpoint.pt",
            "--ema-decay",
            "0.99",
        ]
    )

    assert args.ema_decay == pytest.approx(0.99)

def test_print_training_startup_config_outputs_effective_settings(tmp_path, capsys) -> None:
    replay = make_replay(size=3)
    config = TrainingConfig(
        batch_size=2,
        steps=5,
        device="cuda",
        symmetry_augmentation=False,
    )

    print_training_startup_config(
        config=config,
        replay=replay,
        replay_path=tmp_path / "replay.npz",
        checkpoint_path=tmp_path / "checkpoint.pt",
        resume_path=None,
    )

    output = capsys.readouterr().out
    assert '"event": "train_config"' in output
    assert '"batch_size": 2' in output
    assert '"device": "cuda"' in output
    assert '"symmetry_augmentation": false' in output
    assert '"samples": 3' in output
