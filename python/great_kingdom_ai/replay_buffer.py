"""Replay buffer storage for self-play training samples."""

from __future__ import annotations

import random
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from great_kingdom_ai.features import ACTION_SPACE, BOARD_SIZE, FEATURE_CHANNELS

FEATURE_SHAPE = (FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE)


@dataclass(frozen=True)
class ReplaySample:
    features: np.ndarray
    policy: np.ndarray
    value: float


class ReplayBuffer:
    """Fixed-capacity in-memory replay buffer with compact NumPy persistence."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._samples: deque[ReplaySample] = deque(maxlen=capacity)

    @property
    def capacity(self) -> int:
        return self._samples.maxlen or 0

    def __len__(self) -> int:
        return len(self._samples)

    def push(self, sample: ReplaySample) -> None:
        self._samples.append(_validated_sample(sample))

    def extend(self, samples: Sequence[ReplaySample]) -> None:
        for sample in samples:
            self.push(sample)

    def sample(self, batch_size: int, rng: random.Random) -> list[ReplaySample]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size > len(self._samples):
            raise ValueError("batch_size exceeds replay buffer size")
        indexes = rng.sample(range(len(self._samples)), batch_size)
        samples = list(self._samples)
        return [samples[index] for index in indexes]

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        samples = list(self._samples)
        if samples:
            features = np.stack([sample.features for sample in samples], axis=0).astype(np.float32)
            policies = np.stack([sample.policy for sample in samples], axis=0).astype(np.float32)
            values = np.asarray([sample.value for sample in samples], dtype=np.float32)
        else:
            features = np.empty((0, *FEATURE_SHAPE), dtype=np.float32)
            policies = np.empty((0, ACTION_SPACE), dtype=np.float32)
            values = np.empty((0,), dtype=np.float32)
        np.savez_compressed(
            destination,
            capacity=np.asarray(self.capacity, dtype=np.int64),
            features=features,
            policies=policies,
            values=values,
        )

    @classmethod
    def load(cls, path: str | Path) -> ReplayBuffer:
        with np.load(Path(path)) as data:
            capacity = int(data["capacity"])
            features = np.asarray(data["features"], dtype=np.float32)
            policies = np.asarray(data["policies"], dtype=np.float32)
            values = np.asarray(data["values"], dtype=np.float32)

        if features.shape[0] != policies.shape[0] or features.shape[0] != values.shape[0]:
            raise ValueError("replay buffer arrays have inconsistent lengths")

        buffer = cls(capacity)
        for index in range(features.shape[0]):
            buffer.push(
                ReplaySample(
                    features=features[index],
                    policy=policies[index],
                    value=float(values[index]),
                )
            )
        return buffer


def _validated_sample(sample: ReplaySample) -> ReplaySample:
    features = np.asarray(sample.features, dtype=np.float32)
    policy = np.asarray(sample.policy, dtype=np.float32)
    value = float(sample.value)

    if features.shape != FEATURE_SHAPE:
        raise ValueError(f"expected feature shape {FEATURE_SHAPE}, got {features.shape}")
    if policy.shape != (ACTION_SPACE,):
        raise ValueError(f"expected policy shape {(ACTION_SPACE,)}, got {policy.shape}")
    if not np.isclose(policy.sum(), 1.0):
        raise ValueError("policy target must sum to 1")
    if np.any(policy < 0.0):
        raise ValueError("policy target must be non-negative")
    if value < -1.0 or value > 1.0:
        raise ValueError("value target must be in [-1, 1]")

    return ReplaySample(features=features.copy(), policy=policy.copy(), value=value)
