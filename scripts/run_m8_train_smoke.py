"""Run a short M8 training smoke test with toy replay data.

This script is intended for the Runpod image documented in README:
runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import NoReturn

import numpy as np

DEFAULT_CONFIG = Path("configs/m8-train-smoke.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run M8 toy replay training smoke test")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--work-dir", type=Path, default=Path("data/m8-smoke"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--allow-cpu", action="store_true")
    return parser


def main() -> NoReturn:
    args = build_parser().parse_args()

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit("PyTorch is required for M8 train smoke test") from exc

    from great_kingdom_ai.features import ACTION_SPACE, BOARD_SIZE, FEATURE_CHANNELS
    from great_kingdom_ai.replay_buffer import ReplayBuffer, ReplaySample
    from great_kingdom_ai.train import TrainingConfig, load_training_config, train_from_replay

    config = load_training_config(args.config)
    device_name = args.device or config.device
    if device_name == "cuda" and not torch.cuda.is_available():
        if not args.allow_cpu:
            raise SystemExit("CUDA was requested but is not available")
        device_name = "cpu"
    config = TrainingConfig(**{**asdict(config), "device": device_name})

    args.work_dir.mkdir(parents=True, exist_ok=True)
    replay = ReplayBuffer(capacity=max(8, config.batch_size))
    for index in range(max(8, config.batch_size)):
        features = np.zeros((FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        features[index % FEATURE_CHANNELS, index % BOARD_SIZE, (index * 2) % BOARD_SIZE] = 1.0
        policy = np.zeros(ACTION_SPACE, dtype=np.float32)
        policy[index % ACTION_SPACE] = 1.0
        replay.push(
            ReplaySample(features=features, policy=policy, value=1.0 if index % 2 else -1.0)
        )

    replay_path = args.work_dir / "replay.npz"
    replay.save(replay_path)
    checkpoint_path = args.work_dir / "checkpoint.pt"
    summary = train_from_replay(replay, config, checkpoint_path=checkpoint_path, log_every=1)
    print(
        json.dumps(
            {
                "device": device_name,
                "start_step": summary.start_step,
                "end_step": summary.end_step,
                "checkpoint": str(summary.checkpoint_path),
                "replay": str(replay_path),
                "losses": summary.losses,
            },
            sort_keys=True,
        )
    )
    raise SystemExit(0)


if __name__ == "__main__":
    main()
