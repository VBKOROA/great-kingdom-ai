from __future__ import annotations

import random
from dataclasses import replace

import numpy as np
from great_kingdom_ai.features import ACTION_SPACE, BOARD_SIZE, FEATURE_CHANNELS
from great_kingdom_ai.priority_sampling import (
    PrioritySamplingConfig,
    legal_masks_from_features,
    sample_priority_indexes,
)
from great_kingdom_ai.replay.sample import ReplaySample, validate_replay_sample
from great_kingdom_ai.training.batch import TrainingArrays


def make_sample(index: int, value: float = 1.0) -> ReplaySample:
    features = np.zeros((FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    features[index % FEATURE_CHANNELS, index % BOARD_SIZE, (index * 3) % BOARD_SIZE] = 1.0
    features[4, :, :] = 1.0
    policy = np.zeros(ACTION_SPACE, dtype=np.float32)
    policy[index % ACTION_SPACE] = 1.0
    return ReplaySample(features=features, policy=policy, value=value)


def make_weighted_sample(index: int, value: float, sample_weight: float) -> ReplaySample:
    sample = make_sample(index, value=value)
    return ReplaySample(
        features=sample.features,
        policy=sample.policy,
        value=sample.value,
        sample_weight=sample_weight,
    )


def make_replay(size: int = 6) -> InMemoryReplayDataset:
    buffer = InMemoryReplayDataset(capacity=size)
    for index in range(size):
        buffer.push(make_sample(index, value=1.0 if index % 2 else -1.0))
    return buffer


class InMemoryReplayDataset:
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = capacity
        self._samples: list[ReplaySample] = []

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        return len(self._samples)

    def push(self, sample: ReplaySample) -> None:
        if len(self._samples) == self._capacity:
            self._samples.pop(0)
        self._samples.append(validate_replay_sample(sample))

    def sample(self, batch_size: int, rng: random.Random) -> list[ReplaySample]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size > len(self._samples):
            raise ValueError("batch_size exceeds replay size")
        return [self._samples[index] for index in rng.sample(range(len(self._samples)), batch_size)]

    def sample_priority_biased(
        self,
        batch_size: int,
        rng: random.Random,
        *,
        priority_config: PrioritySamplingConfig,
        recent_fraction: float = 0.0,
        recent_window: int = 0,
    ) -> list[ReplaySample]:
        sampled = sample_priority_indexes(
            priorities=np.asarray(
                [sample.sample_weight for sample in self._samples],
                dtype=np.float32,
            )
            ** np.float32(priority_config.alpha),
            batch_size=batch_size,
            rng=rng,
            beta=priority_config.beta,
            recent_fraction=recent_fraction,
            recent_window=recent_window,
        )
        return [
            replace(
                self._samples[index],
                sample_weight=float(self._samples[index].sample_weight * importance_weight),
            )
            for index, importance_weight in zip(
                sampled.indexes,
                sampled.importance_weights,
                strict=True,
            )
        ]

    def sample_arrays(
        self,
        batch_size: int,
        rng: random.Random,
        *,
        recent_fraction: float = 0.0,
        recent_window: int = 0,
        priority_config: PrioritySamplingConfig | None = None,
    ) -> TrainingArrays:
        del recent_fraction, recent_window, priority_config
        selected = self.sample(batch_size, rng)
        features = np.stack([sample.features for sample in selected], axis=0).astype(np.float32)
        return TrainingArrays(
            features=np.ascontiguousarray(features, dtype=np.float32),
            policies=np.ascontiguousarray(
                np.stack([sample.policy for sample in selected], axis=0),
                dtype=np.float32,
            ),
            values=np.ascontiguousarray(
                np.asarray([sample.value for sample in selected], dtype=np.float32)
            ),
            sample_weights=np.ascontiguousarray(
                np.asarray([sample.sample_weight for sample in selected], dtype=np.float32)
            ),
            legal_masks=legal_masks_from_features(features),
        )


class RecencyReplay:
    def __init__(self) -> None:
        self.calls: list[tuple[int, float, int]] = []

    def __len__(self) -> int:
        return 4

    def sample(self, batch_size: int, rng: random.Random) -> list[ReplaySample]:
        del batch_size, rng
        raise AssertionError("uniform sample should not be used")

    def sample_recency_biased(
        self,
        batch_size: int,
        rng: random.Random,
        *,
        recent_fraction: float,
        recent_window: int,
    ) -> list[ReplaySample]:
        del rng
        self.calls.append((batch_size, recent_fraction, recent_window))
        return [make_sample(index) for index in range(batch_size)]


class ArrayReplay:
    def __init__(self) -> None:
        self.calls: list[tuple[int, float, int]] = []

    def __len__(self) -> int:
        return 4

    def sample(self, batch_size: int, rng: random.Random) -> list[ReplaySample]:
        del batch_size, rng
        raise AssertionError("sample should not be used when sample_arrays exists")

    def sample_arrays(
        self,
        batch_size: int,
        rng: random.Random,
        *,
        recent_fraction: float = 0.0,
        recent_window: int = 0,
    ) -> TrainingArrays:
        del rng
        self.calls.append((batch_size, recent_fraction, recent_window))
        samples = [
            make_sample(index, value=1.0 if index % 2 else -1.0)
            for index in range(batch_size)
        ]
        return TrainingArrays(
            features=np.stack([sample.features for sample in samples], axis=0).astype(np.float32),
            policies=np.stack([sample.policy for sample in samples], axis=0).astype(np.float32),
            values=np.asarray([sample.value for sample in samples], dtype=np.float32),
            sample_weights=np.ones((batch_size,), dtype=np.float32),
        )


class PriorityArrayReplay:
    def __init__(self) -> None:
        self.calls: list[tuple[int, float, int, bool, float, float]] = []

    def __len__(self) -> int:
        return 4

    def sample(self, batch_size: int, rng: random.Random) -> list[ReplaySample]:
        del batch_size, rng
        raise AssertionError("sample should not be used when sample_arrays exists")

    def sample_arrays(
        self,
        batch_size: int,
        rng: random.Random,
        *,
        recent_fraction: float = 0.0,
        recent_window: int = 0,
        priority_config: PrioritySamplingConfig | None = None,
    ) -> TrainingArrays:
        del rng
        assert priority_config is not None
        self.calls.append(
            (
                batch_size,
                recent_fraction,
                recent_window,
                priority_config.enabled,
                priority_config.alpha,
                priority_config.beta,
            )
        )
        samples = [make_sample(index) for index in range(batch_size)]
        return TrainingArrays(
            features=np.stack([sample.features for sample in samples], axis=0).astype(np.float32),
            policies=np.stack([sample.policy for sample in samples], axis=0).astype(np.float32),
            values=np.asarray([sample.value for sample in samples], dtype=np.float32),
            sample_weights=np.ones((batch_size,), dtype=np.float32),
        )
