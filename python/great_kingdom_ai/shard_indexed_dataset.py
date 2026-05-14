"""Training dataset view over shard-indexed trajectory replay."""

from __future__ import annotations

import random
from bisect import bisect_right
from collections import OrderedDict

import numpy as np

from great_kingdom_ai.priority_sampling import PrioritySamplingConfig, sample_priority_indexes
from great_kingdom_ai.replay_buffer import ReplaySample
from great_kingdom_ai.self_play_data import value_target_for_player
from great_kingdom_ai.shard_replay_index import ActiveShardSpan, ShardReplayIndex
from great_kingdom_ai.trajectory_dataset import TrajectoryArrayBatch
from great_kingdom_ai.trajectory_replay import TrajectoryReplayStore


class ShardIndexedTrajectoryDataset:
    """Trajectory replay dataset backed by indexed shard files."""

    def __init__(self, index: ShardReplayIndex, *, cache_shards: int = 16) -> None:
        if len(index) == 0:
            raise ValueError("shard-indexed replay must contain at least one transition")
        if cache_shards <= 0:
            raise ValueError("cache_shards must be positive")
        self._index = index
        self._spans = index.active_spans()
        self._span_starts = [span.start for span in self._spans]
        self._cache_shards = cache_shards
        self._cache: OrderedDict[str, _CachedShard] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0

    @property
    def capacity(self) -> int:
        return self._index.capacity

    @property
    def cache_info(self) -> dict[str, int]:
        return {
            "size": len(self._cache),
            "max_size": self._cache_shards,
            "hits": self._cache_hits,
            "misses": self._cache_misses,
        }

    def __len__(self) -> int:
        return len(self._index)

    def sample(self, batch_size: int, rng: random.Random) -> list[ReplaySample]:
        batch = self.sample_arrays(batch_size, rng)
        return [
            ReplaySample(
                features=batch.features[index],
                policy=batch.policies[index],
                value=float(batch.values[index]),
                sample_weight=float(batch.sample_weights[index]),
            )
            for index in range(batch_size)
        ]

    def sample_arrays(
        self,
        batch_size: int,
        rng: random.Random,
        *,
        recent_fraction: float = 0.0,
        recent_window: int = 0,
        priority_config: PrioritySamplingConfig | None = None,
    ) -> TrajectoryArrayBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size > len(self):
            raise ValueError("batch_size exceeds shard-indexed replay size")

        if priority_config is not None and priority_config.enabled:
            priorities = np.maximum(
                self._sample_weights() ** np.float32(priority_config.alpha),
                np.float32(1e-6),
            )
            sampled = sample_priority_indexes(
                priorities=priorities,
                batch_size=batch_size,
                rng=rng,
                beta=priority_config.beta,
                recent_fraction=recent_fraction,
                recent_window=recent_window,
            )
            indexes = sampled.indexes
            importance_weights = sampled.importance_weights
        else:
            sampled = sample_priority_indexes(
                priorities=np.ones((len(self),), dtype=np.float32),
                batch_size=batch_size,
                rng=rng,
                beta=0.0,
                recent_fraction=recent_fraction,
                recent_window=recent_window,
            )
            indexes = sampled.indexes
            importance_weights = np.ones((batch_size,), dtype=np.float32)

        index_array = np.asarray(indexes, dtype=np.int64)
        return self._gather(index_array, importance_weights)

    def _gather(
        self,
        indexes: np.ndarray,
        importance_weights: np.ndarray,
    ) -> TrajectoryArrayBatch:
        grouped: dict[int, list[tuple[int, int]]] = {}
        for output_row, global_row in enumerate(indexes.tolist()):
            span_index = self._span_index(int(global_row))
            span = self._spans[span_index]
            grouped.setdefault(span_index, []).append((output_row, int(global_row) - span.start))

        first_shard = self._load_span(self._spans[next(iter(grouped))])
        batch_size = indexes.shape[0]
        features = np.empty((batch_size, *first_shard.replay.features.shape[1:]), dtype=np.float32)
        policies = np.empty(
            (batch_size, first_shard.replay.policy_targets.shape[1]),
            dtype=np.float32,
        )
        values = np.empty((batch_size,), dtype=np.float32)
        sample_weights = np.empty((batch_size,), dtype=np.float32)
        legal_masks = np.empty(
            (batch_size, first_shard.replay.legal_masks.shape[1]),
            dtype=np.bool_,
        )

        for span_index, rows in grouped.items():
            cached = self._load_span(self._spans[span_index])
            local_rows = np.asarray([local for _, local in rows], dtype=np.int64)
            output_rows = np.asarray([output for output, _ in rows], dtype=np.int64)
            features[output_rows] = cached.replay.features[local_rows]
            policies[output_rows] = cached.replay.policy_targets[local_rows]
            values[output_rows] = cached.values[local_rows]
            sample_weights[output_rows] = cached.replay.sample_weights[local_rows]
            legal_masks[output_rows] = cached.replay.legal_masks[local_rows]

        return TrajectoryArrayBatch(
            indexes=indexes,
            features=np.ascontiguousarray(features, dtype=np.float32),
            policies=np.ascontiguousarray(policies, dtype=np.float32),
            values=np.ascontiguousarray(values, dtype=np.float32),
            sample_weights=np.ascontiguousarray(
                sample_weights * importance_weights,
                dtype=np.float32,
            ),
            legal_masks=np.ascontiguousarray(legal_masks, dtype=np.bool_),
        )

    def _sample_weights(self) -> np.ndarray:
        weights = np.empty((len(self),), dtype=np.float32)
        for span in self._spans:
            cached = self._load_span(span)
            weights[span.start : span.stop] = cached.replay.sample_weights.astype(
                np.float32,
                copy=False,
            )
        return weights

    def _span_index(self, global_row: int) -> int:
        if global_row < 0 or global_row >= len(self):
            raise IndexError("global row index is out of range")
        span_index = bisect_right(self._span_starts, global_row) - 1
        if span_index < 0:
            raise IndexError("global row index is out of range")
        return span_index

    def _load_span(self, span: ActiveShardSpan) -> _CachedShard:
        key = str(span.record.replay_path)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache_hits += 1
            self._cache.move_to_end(key)
            return cached

        replay = TrajectoryReplayStore.load(span.record.replay_path)
        replay.validate()
        if len(replay) != span.record.rows:
            raise ValueError(f"indexed shard row count mismatch: {span.record.shard_id}")
        cached = _CachedShard(replay=replay, values=_terminal_values(replay))
        self._cache[key] = cached
        self._cache_misses += 1
        if len(self._cache) > self._cache_shards:
            self._cache.popitem(last=False)
        return cached


class _CachedShard:
    def __init__(self, *, replay: TrajectoryReplayStore, values: np.ndarray) -> None:
        self.replay = replay
        self.values = values


def _terminal_values(replay: TrajectoryReplayStore) -> np.ndarray:
    episode_indexes = np.searchsorted(
        replay.episode_offsets,
        np.arange(len(replay), dtype=np.int64),
        side="right",
    ) - 1
    values = np.empty((len(replay),), dtype=np.float32)
    for row in range(len(replay)):
        values[row] = np.float32(
            value_target_for_player(
                player=int(replay.players[row]),
                winner=int(replay.episode_winners[int(episode_indexes[row])]),
            )
        )
    return values


__all__ = ["ShardIndexedTrajectoryDataset"]
