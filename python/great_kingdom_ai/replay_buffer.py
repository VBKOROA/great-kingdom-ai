"""Replay buffer storage for self-play training samples."""

from __future__ import annotations

import random
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from great_kingdom_ai.features import ACTION_SPACE, BOARD_SIZE, FEATURE_CHANNELS

FEATURE_SHAPE = (FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE)


@dataclass(frozen=True)
class ReplaySample:
    features: np.ndarray
    policy: np.ndarray
    value: float
    root_policy_logits: np.ndarray | None = None
    sample_weight: float = 1.0


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
            sample_weights = np.asarray(
                [sample.sample_weight for sample in samples],
                dtype=np.float32,
            )
        else:
            features = np.empty((0, *FEATURE_SHAPE), dtype=np.float32)
            policies = np.empty((0, ACTION_SPACE), dtype=np.float32)
            values = np.empty((0,), dtype=np.float32)
            sample_weights = np.empty((0,), dtype=np.float32)
        payload: dict[str, np.ndarray] = {
            "capacity": np.asarray(self.capacity, dtype=np.int64),
            "features": features,
            "policies": policies,
            "values": values,
        }
        if sample_weights.size and not np.allclose(sample_weights, 1.0):
            payload["sample_weights"] = sample_weights
        root_policy_logits = _root_policy_logits_array(samples)
        if root_policy_logits is not None:
            payload["root_policy_logits"] = root_policy_logits
        np.savez_compressed(destination, **cast(dict[str, Any], payload))

    @classmethod
    def load(cls, path: str | Path) -> ReplayBuffer:
        with np.load(Path(path)) as data:
            capacity = int(data["capacity"])
            features = np.asarray(data["features"], dtype=np.float32)
            policies = np.asarray(data["policies"], dtype=np.float32)
            values = np.asarray(data["values"], dtype=np.float32)
            root_policy_logits = (
                np.asarray(data["root_policy_logits"], dtype=np.float32)
                if "root_policy_logits" in data
                else None
            )
            sample_weights = (
                np.asarray(data["sample_weights"], dtype=np.float32)
                if "sample_weights" in data
                else np.ones(values.shape, dtype=np.float32)
            )

        if features.shape[0] != policies.shape[0] or features.shape[0] != values.shape[0]:
            raise ValueError("replay buffer arrays have inconsistent lengths")
        if sample_weights.shape != values.shape:
            raise ValueError("replay buffer sample_weights shape must match values shape")
        if root_policy_logits is not None and root_policy_logits.shape != policies.shape:
            raise ValueError(
                "replay buffer root_policy_logits shape must match policies shape"
            )

        buffer = cls(capacity)
        for index in range(features.shape[0]):
            root_logits = (
                root_policy_logits[index]
                if root_policy_logits is not None
                and np.isfinite(root_policy_logits[index]).all()
                else None
            )
            buffer.push(
                ReplaySample(
                    features=features[index],
                    policy=policies[index],
                    value=float(values[index]),
                    root_policy_logits=root_logits,
                    sample_weight=float(sample_weights[index]),
                )
            )
        return buffer


def _validated_sample(sample: ReplaySample) -> ReplaySample:
    features = np.asarray(sample.features, dtype=np.float32)
    policy = np.asarray(sample.policy, dtype=np.float32)
    value = float(sample.value)
    sample_weight = float(sample.sample_weight)

    if features.shape != FEATURE_SHAPE:
        raise ValueError(f"expected feature shape {FEATURE_SHAPE}, got {features.shape}")
    if policy.shape != (ACTION_SPACE,):
        raise ValueError(f"expected policy shape {(ACTION_SPACE,)}, got {policy.shape}")
    root_policy_logits = (
        None
        if sample.root_policy_logits is None
        else np.asarray(sample.root_policy_logits, dtype=np.float32)
    )
    if root_policy_logits is not None:
        if root_policy_logits.shape != (ACTION_SPACE,):
            raise ValueError(
                f"expected root_policy_logits shape {(ACTION_SPACE,)}, "
                f"got {root_policy_logits.shape}"
            )
        if not np.isfinite(root_policy_logits).all():
            raise ValueError("root_policy_logits must be finite")
    if not np.isclose(policy.sum(), 1.0):
        raise ValueError("policy target must sum to 1")
    if np.any(policy < 0.0):
        raise ValueError("policy target must be non-negative")
    if value < -1.0 or value > 1.0:
        raise ValueError("value target must be in [-1, 1]")
    if not np.isfinite(sample_weight) or sample_weight <= 0.0:
        raise ValueError("sample_weight must be finite and positive")

    return ReplaySample(
        features=features.copy(),
        policy=policy.copy(),
        value=value,
        root_policy_logits=(
            None if root_policy_logits is None else root_policy_logits.copy()
        ),
        sample_weight=sample_weight,
    )


def _root_policy_logits_array(samples: list[ReplaySample]) -> np.ndarray | None:
    if not any(sample.root_policy_logits is not None for sample in samples):
        return None
    rows = np.full((len(samples), ACTION_SPACE), np.nan, dtype=np.float32)
    for index, sample in enumerate(samples):
        if sample.root_policy_logits is None:
            continue
        rows[index] = np.asarray(sample.root_policy_logits, dtype=np.float32)
    return rows
