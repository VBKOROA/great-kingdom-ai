"""Online aggregate replay buffer for duplicate self-play states."""

from __future__ import annotations

import hashlib
import random
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from great_kingdom_ai.features import ACTION_SPACE
from great_kingdom_ai.replay_aggregate import _sample_weights_from_counts
from great_kingdom_ai.replay_buffer import FEATURE_SHAPE, ReplaySample, _validated_sample


@dataclass
class _AggregateEntry:
    features: np.ndarray
    policy_sum: np.ndarray
    value_sum: float
    count: int
    root_policy_sum: np.ndarray
    root_policy_count: int


class OnlineAggregateReplayBuffer:
    """Fixed-capacity replay store that merges exact duplicate feature rows on push.

    Capacity is measured in unique aggregate rows. Duplicate pushes update the existing
    row and refresh its recency, so frequently revisited states are less likely to be
    evicted when the unique-row capacity is reached.
    """

    def __init__(
        self,
        capacity: int,
        *,
        sample_weight_mode: str = "sqrt_count",
        sample_weight_cap: float | None = 16.0,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = capacity
        self._sample_weight_mode = sample_weight_mode
        self._sample_weight_cap = sample_weight_cap
        self._entries: OrderedDict[str, _AggregateEntry] = OrderedDict()

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def raw_sample_count(self) -> int:
        return sum(entry.count for entry in self._entries.values())

    def __len__(self) -> int:
        return len(self._entries)

    def push(self, sample: ReplaySample) -> None:
        validated = _validated_sample(sample)
        digest = _feature_digest(validated.features)
        entry = self._entries.get(digest)
        if entry is None:
            if len(self._entries) >= self._capacity:
                self._entries.popitem(last=False)
            self._entries[digest] = _entry_from_sample(validated)
            return

        entry.policy_sum += validated.policy
        entry.value_sum += validated.value
        entry.count += 1
        if validated.root_policy_logits is not None:
            entry.root_policy_sum += validated.root_policy_logits
            entry.root_policy_count += 1
        self._entries.move_to_end(digest)

    def extend(self, samples: Sequence[ReplaySample]) -> None:
        for sample in samples:
            self.push(sample)

    def sample(self, batch_size: int, rng: random.Random) -> list[ReplaySample]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size > len(self._entries):
            raise ValueError("batch_size exceeds replay buffer size")
        entries = list(self._entries.values())
        indexes = rng.sample(range(len(entries)), batch_size)
        return [self._sample_from_entry(entries[index]) for index in indexes]

    def sample_recency_biased(
        self,
        batch_size: int,
        rng: random.Random,
        *,
        recent_fraction: float,
        recent_window: int,
    ) -> list[ReplaySample]:
        """Sample a batch with a fixed fraction drawn from recently updated rows."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size > len(self._entries):
            raise ValueError("batch_size exceeds replay buffer size")
        if not 0.0 <= recent_fraction <= 1.0:
            raise ValueError("recent_fraction must be in [0, 1]")
        if recent_window <= 0:
            raise ValueError("recent_window must be positive")

        entries = list(self._entries.values())
        recent_count = min(recent_window, len(entries))
        recent_indexes = list(range(len(entries) - recent_count, len(entries)))
        old_indexes = list(range(0, len(entries) - recent_count))

        target_recent = round(batch_size * recent_fraction)
        recent_take = min(target_recent, len(recent_indexes), batch_size)
        old_take = min(batch_size - recent_take, len(old_indexes))
        recent_take = min(batch_size - old_take, len(recent_indexes))
        old_take = batch_size - recent_take
        if old_take > len(old_indexes):
            old_take = len(old_indexes)
            recent_take = batch_size - old_take
        if recent_take > len(recent_indexes):
            raise ValueError("not enough replay rows to satisfy recency-biased sample")

        indexes = rng.sample(recent_indexes, recent_take)
        indexes.extend(rng.sample(old_indexes, old_take))
        rng.shuffle(indexes)
        return [self._sample_from_entry(entries[index]) for index in indexes]

    def save(self, path: str | Path, *, compressed: bool = True) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        _save_payload(destination, self.to_payload(), compressed=compressed)

    def save_atomic(self, path: str | Path, *, compressed: bool = True) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f"{destination.name}.tmp")
        _save_payload(temporary, self.to_payload(), compressed=compressed)
        temporary.replace(destination)

    def to_payload(self) -> dict[str, np.ndarray]:
        entries = list(self._entries.values())
        if entries:
            features = np.stack([entry.features for entry in entries], axis=0).astype(np.float32)
            counts = np.asarray([entry.count for entry in entries], dtype=np.int64)
            policies = np.stack(
                [_policy_from_entry(entry) for entry in entries],
                axis=0,
            ).astype(np.float32)
            values = np.asarray(
                [entry.value_sum / entry.count for entry in entries],
                dtype=np.float32,
            )
            sample_weights = _sample_weights_from_counts(
                counts,
                mode=self._sample_weight_mode,
                cap=self._sample_weight_cap,
            )
        else:
            features = np.empty((0, *FEATURE_SHAPE), dtype=np.float32)
            counts = np.empty((0,), dtype=np.int64)
            policies = np.empty((0, ACTION_SPACE), dtype=np.float32)
            values = np.empty((0,), dtype=np.float32)
            sample_weights = np.empty((0,), dtype=np.float32)

        payload: dict[str, np.ndarray] = {
            "capacity": np.asarray(self.capacity, dtype=np.int64),
            "features": features,
            "policies": policies,
            "values": values,
            "counts": counts,
            "sample_weights": sample_weights,
            "raw_sample_count": np.asarray(self.raw_sample_count, dtype=np.int64),
        }
        root_policy_logits, root_policy_counts = _root_policy_logits_arrays(entries)
        if root_policy_logits is not None:
            payload["root_policy_logits"] = root_policy_logits
            payload["root_policy_counts"] = root_policy_counts
        return payload

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        capacity: int | None = None,
        sample_weight_mode: str = "sqrt_count",
        sample_weight_cap: float | None = 16.0,
    ) -> OnlineAggregateReplayBuffer:
        with np.load(Path(path)) as data:
            loaded_capacity = int(data["capacity"])
            features = np.asarray(data["features"], dtype=np.float32)
            policies = np.asarray(data["policies"], dtype=np.float32)
            values = np.asarray(data["values"], dtype=np.float32)
            counts = (
                np.asarray(data["counts"], dtype=np.int64)
                if "counts" in data
                else np.ones(values.shape, dtype=np.int64)
            )
            root_policy_logits = (
                np.asarray(data["root_policy_logits"], dtype=np.float32)
                if "root_policy_logits" in data
                else None
            )
            root_policy_counts = (
                np.asarray(data["root_policy_counts"], dtype=np.int64)
                if "root_policy_counts" in data
                else None
            )

        _validate_loaded_arrays(features, policies, values, counts, root_policy_logits)
        buffer = cls(
            max(capacity if capacity is not None else loaded_capacity, features.shape[0]),
            sample_weight_mode=sample_weight_mode,
            sample_weight_cap=sample_weight_cap,
        )
        for index in range(features.shape[0]):
            count = int(counts[index])
            if count <= 0:
                raise ValueError("counts must be positive")
            root_logits = (
                root_policy_logits[index]
                if root_policy_logits is not None
                and np.isfinite(root_policy_logits[index]).all()
                else None
            )
            root_count = (
                int(root_policy_counts[index])
                if root_policy_counts is not None
                else (count if root_logits is not None else 0)
            )
            if root_count < 0 or root_count > count:
                raise ValueError("root_policy_counts must be in [0, count]")
            entry = _AggregateEntry(
                features=features[index].astype(np.float32, copy=True),
                policy_sum=policies[index].astype(np.float32, copy=True) * np.float32(count),
                value_sum=float(values[index]) * count,
                count=count,
                root_policy_sum=(
                    np.zeros(ACTION_SPACE, dtype=np.float32)
                    if root_logits is None
                    else root_logits.astype(np.float32, copy=True) * np.float32(root_count)
                ),
                root_policy_count=root_count,
            )
            buffer._entries[_feature_digest(entry.features)] = entry
        return buffer

    def _sample_from_entry(self, entry: _AggregateEntry) -> ReplaySample:
        counts = np.asarray([entry.count], dtype=np.int64)
        weights = _sample_weights_from_counts(
            counts,
            mode=self._sample_weight_mode,
            cap=self._sample_weight_cap,
        )
        root_policy_logits = (
            None
            if entry.root_policy_count == 0
            else (entry.root_policy_sum / np.float32(entry.root_policy_count)).astype(np.float32)
        )
        return ReplaySample(
            features=entry.features.copy(),
            policy=_policy_from_entry(entry),
            value=float(entry.value_sum / entry.count),
            root_policy_logits=root_policy_logits,
            sample_weight=float(weights[0]),
        )


def _entry_from_sample(sample: ReplaySample) -> _AggregateEntry:
    return _AggregateEntry(
        features=sample.features.copy(),
        policy_sum=sample.policy.copy(),
        value_sum=sample.value,
        count=1,
        root_policy_sum=(
            np.zeros(ACTION_SPACE, dtype=np.float32)
            if sample.root_policy_logits is None
            else sample.root_policy_logits.copy()
        ),
        root_policy_count=0 if sample.root_policy_logits is None else 1,
    )


def _feature_digest(features: np.ndarray) -> str:
    return hashlib.blake2b(features.tobytes(), digest_size=16).hexdigest()


def _policy_from_entry(entry: _AggregateEntry) -> np.ndarray:
    policy = (entry.policy_sum / np.float32(entry.count)).astype(np.float32)
    total = float(policy.sum())
    if total <= 0.0:
        raise ValueError("policy rows must have positive mass")
    return (policy / np.float32(total)).astype(np.float32)


def _root_policy_logits_arrays(
    entries: list[_AggregateEntry],
) -> tuple[np.ndarray | None, np.ndarray]:
    root_policy_counts = np.asarray(
        [entry.root_policy_count for entry in entries],
        dtype=np.int64,
    )
    if not np.any(root_policy_counts > 0):
        return None, root_policy_counts
    rows = np.full((len(entries), ACTION_SPACE), np.nan, dtype=np.float32)
    for index, entry in enumerate(entries):
        if entry.root_policy_count <= 0:
            continue
        rows[index] = entry.root_policy_sum / np.float32(entry.root_policy_count)
    return rows, root_policy_counts


def _save_payload(
    path: Path,
    payload: dict[str, np.ndarray],
    *,
    compressed: bool,
) -> None:
    save = np.savez_compressed if compressed else np.savez
    with path.open("wb") as file:
        save(file, **cast(dict[str, Any], payload))


def _validate_loaded_arrays(
    features: np.ndarray,
    policies: np.ndarray,
    values: np.ndarray,
    counts: np.ndarray,
    root_policy_logits: np.ndarray | None,
) -> None:
    if features.ndim != 4 or tuple(features.shape[1:]) != FEATURE_SHAPE:
        raise ValueError(f"expected features shape [N, {FEATURE_SHAPE}], got {features.shape}")
    if policies.shape != (features.shape[0], ACTION_SPACE):
        raise ValueError(
            f"expected policies shape {(features.shape[0], ACTION_SPACE)}, got {policies.shape}"
        )
    if values.shape != (features.shape[0],):
        raise ValueError(f"expected values shape {(features.shape[0],)}, got {values.shape}")
    if counts.shape != values.shape:
        raise ValueError("counts shape must match values shape")
    if root_policy_logits is not None and root_policy_logits.shape != policies.shape:
        raise ValueError("root_policy_logits shape must match policies shape")


__all__ = ["OnlineAggregateReplayBuffer"]
