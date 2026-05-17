"""Command line interface for trajectory replay training."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import NoReturn

from great_kingdom_ai.training.batch import ReplayDataset
from great_kingdom_ai.training.config import TrainingConfig, load_training_config
from great_kingdom_ai.training.loop import train_from_replay


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a Great Kingdom policy-value network")
    parser.add_argument("--replay", type=Path, required=True, help="Path to replay .npz file")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Output checkpoint path")
    parser.add_argument("--resume", type=Path, default=None, help="Checkpoint to resume from")
    parser.add_argument(
        "--bootstrap-weights",
        type=Path,
        default=None,
        help="Checkpoint to load model weights from without optimizer, scheduler, or step state",
    )
    parser.add_argument("--config", type=Path, default=None, help="JSON TrainingConfig override")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--model-preset",
        choices=[
            "small",
            "medium",
            "medium_plus",
            "strong",
            "large",
            "large_policy",
            "large_plus",
        ],
        default=None,
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=None,
        help="record loss every N training steps instead of only the final step",
    )
    parser.add_argument(
        "--no-symmetry-augmentation",
        action="store_true",
        help="disable random board symmetry augmentation during batch sampling",
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=None,
        help="enable exponential moving average weights with the given decay",
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> TrainingConfig:
    config = load_training_config(args.config) if args.config is not None else TrainingConfig()
    overrides = {
        "device": args.device,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "model_preset": args.model_preset,
        "symmetry_augmentation": False if args.no_symmetry_augmentation else None,
        "ema_decay": args.ema_decay,
    }
    data = asdict(config)
    data.update({key: value for key, value in overrides.items() if value is not None})
    return TrainingConfig(**data)


def _log_every_from_args(args: argparse.Namespace, config: TrainingConfig) -> int:
    log_every = args.log_every if args.log_every is not None else config.steps
    if log_every <= 0:
        raise ValueError("log_every must be positive")
    return max(1, log_every)


def print_training_startup_config(
    *,
    config: TrainingConfig,
    replay: ReplayDataset,
    replay_path: Path,
    checkpoint_path: Path,
    resume_path: Path | None,
    bootstrap_weights_path: Path | None = None,
) -> None:
    print(
        json.dumps(
            {
                "event": "train_config",
                "config": asdict(config),
                "replay": {
                    "path": str(replay_path),
                    "samples": len(replay),
                    "capacity": getattr(replay, "capacity", len(replay)),
                    "type": type(replay).__name__,
                },
                "checkpoint": str(checkpoint_path),
                "resume": str(resume_path) if resume_path is not None else None,
                "bootstrap_weights": (
                    str(bootstrap_weights_path) if bootstrap_weights_path is not None else None
                ),
            },
            sort_keys=True,
        )
    )


def _load_trajectory_replay(path: str | Path) -> ReplayDataset:
    from great_kingdom_ai.replay import TrajectoryReplayDataset, TrajectoryReplayStore

    return TrajectoryReplayDataset(TrajectoryReplayStore.load(path))


def main() -> NoReturn:
    args = build_parser().parse_args()
    config = _config_from_args(args)
    replay = _load_trajectory_replay(args.replay)
    print_training_startup_config(
        config=config,
        replay=replay,
        replay_path=args.replay,
        checkpoint_path=args.checkpoint,
        resume_path=args.resume,
        bootstrap_weights_path=args.bootstrap_weights,
    )
    summary = train_from_replay(
        replay,
        config,
        checkpoint_path=args.checkpoint,
        resume_path=args.resume,
        bootstrap_weights_path=args.bootstrap_weights,
        log_every=_log_every_from_args(args, config),
    )
    print(
        json.dumps(
            {
                "event": "train_summary",
                "start_step": summary.start_step,
                "end_step": summary.end_step,
                "checkpoint": str(summary.checkpoint_path) if summary.checkpoint_path else None,
                "losses": summary.losses,
            },
            sort_keys=True,
        )
    )
    raise SystemExit(0)


__all__ = [
    "build_parser",
    "main",
    "print_training_startup_config",
]
