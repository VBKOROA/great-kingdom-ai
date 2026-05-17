"""Run a short M8 training smoke test with toy replay data.

This script is intended for the Runpod image documented in README:
runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import NoReturn

import numpy as np
from great_kingdom_ai.priority_sampling import legal_masks_from_features
from great_kingdom_ai.replay.sample import ReplaySample
from great_kingdom_ai.training.batch import TrainingArrays

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
    from great_kingdom_ai.training import TrainingConfig, load_training_config, train_from_replay

    config = load_training_config(args.config)
    device_name = args.device or config.device
    if device_name == "cuda" and not torch.cuda.is_available():
        if not args.allow_cpu:
            raise SystemExit("CUDA was requested but is not available")
        device_name = "cpu"
    config = TrainingConfig(**{**asdict(config), "device": device_name})

    args.work_dir.mkdir(parents=True, exist_ok=True)
    samples: list[ReplaySample] = []
    for index in range(max(8, config.batch_size)):
        features = np.zeros((FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        features[index % FEATURE_CHANNELS, index % BOARD_SIZE, (index * 2) % BOARD_SIZE] = 1.0
        features[4, :, :] = 1.0
        policy = np.zeros(ACTION_SPACE, dtype=np.float32)
        policy[index % ACTION_SPACE] = 1.0
        samples.append(
            ReplaySample(features=features, policy=policy, value=1.0 if index % 2 else -1.0)
        )

    replay = _ToyReplayDataset(samples)
    checkpoint_path = args.work_dir / "checkpoint.pt"
    summary = train_from_replay(replay, config, checkpoint_path=checkpoint_path, log_every=1)
    print(
        json.dumps(
            {
                "device": device_name,
                "start_step": summary.start_step,
                "end_step": summary.end_step,
                "checkpoint": str(summary.checkpoint_path),
                "replay_samples": len(replay),
                "losses": summary.losses,
            },
            sort_keys=True,
        )
    )
    raise SystemExit(0)


class _ToyReplayDataset:
    def __init__(self, samples: list[ReplaySample]) -> None:
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def sample(self, batch_size: int, rng: random.Random) -> list[ReplaySample]:
        return [self._samples[index] for index in rng.sample(range(len(self._samples)), batch_size)]

    def sample_arrays(
        self,
        batch_size: int,
        rng: random.Random,
        *,
        recent_fraction: float = 0.0,
        recent_window: int = 0,
    ) -> TrainingArrays:
        del recent_fraction, recent_window
        batch = self.sample(batch_size, rng)
        features = np.stack([sample.features for sample in batch], axis=0).astype(np.float32)
        return TrainingArrays(
            features=np.ascontiguousarray(features, dtype=np.float32),
            policies=np.ascontiguousarray(
                np.stack([sample.policy for sample in batch], axis=0),
                dtype=np.float32,
            ),
            values=np.ascontiguousarray(
                np.asarray([sample.value for sample in batch], dtype=np.float32)
            ),
            sample_weights=np.ones((batch_size,), dtype=np.float32),
            legal_masks=legal_masks_from_features(features),
        )


if __name__ == "__main__":
    main()
