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

    def __init__(
        self,
        index: ShardReplayIndex,
        *,
        cache_shards: int = 16,
        sample_shards_per_batch: int | None = None,
    ) -> None:
        if len(index) == 0:
            raise ValueError("shard-indexed replay must contain at least one transition")
        if cache_shards <= 0:
            raise ValueError("cache_shards must be positive")
        if sample_shards_per_batch is not None and sample_shards_per_batch <= 0:
            raise ValueError("sample_shards_per_batch must be positive")
        self._index = index
        self._spans = index.active_spans()
        self._span_starts = [span.start for span in self._spans]
        self._cache_shards = cache_shards
        self._sample_shards_per_batch = sample_shards_per_batch
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
        elif self._sample_shards_per_batch is not None:
            indexes = self._sample_cache_local_indexes(
                batch_size,
                rng,
                recent_fraction=recent_fraction,
                recent_window=recent_window,
            )
            importance_weights = np.ones((batch_size,), dtype=np.float32)
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

    def _sample_cache_local_indexes(
        self,
        batch_size: int,
        rng: random.Random,
        *,
        recent_fraction: float,
        recent_window: int,
    ) -> list[int]:
        if recent_fraction <= 0.0:
            return self._sample_range_cache_local(0, len(self), batch_size, rng)
        if not 0.0 <= recent_fraction <= 1.0:
            raise ValueError("recent_fraction must be in [0, 1]")
        if recent_window <= 0:
            raise ValueError("recent_window must be positive when recency sampling is enabled")

        size = len(self)
        recent_count = min(recent_window, size)
        old_count = size - recent_count
        target_recent = round(batch_size * recent_fraction)
        recent_take = min(target_recent, recent_count, batch_size)
        old_take = min(batch_size - recent_take, old_count)
        recent_take = min(batch_size - old_take, recent_count)
        old_take = batch_size - recent_take
        if old_take > old_count:
            old_take = old_count
            recent_take = batch_size - old_take
        if recent_take > recent_count:
            raise ValueError("not enough rows to satisfy recency-biased sample")

        indexes: list[int] = []
        if recent_take > 0:
            indexes.extend(
                self._sample_range_cache_local(
                    size - recent_count,
                    size,
                    recent_take,
                    rng,
                )
            )
        if old_take > 0:
            indexes.extend(self._sample_range_cache_local(0, old_count, old_take, rng))
        rng.shuffle(indexes)
        return indexes

    def _sample_range_cache_local(
        self,
        start: int,
        stop: int,
        count: int,
        rng: random.Random,
    ) -> list[int]:
        if count <= 0:
            return []
        if start < 0 or stop > len(self) or start >= stop:
            raise ValueError("invalid sample range")
        candidates = self._overlapping_span_ranges(start, stop)
        total_rows = sum(candidate.rows for candidate in candidates)
        if count > total_rows:
            raise ValueError("cannot sample more rows than the selected range contains")

        selected: list[_SpanRowRange] = []
        remaining = candidates.copy()
        selected_rows = 0
        target_shards = min(self._sample_shards_per_batch or len(candidates), len(candidates))
        while remaining and (len(selected) < target_shards or selected_rows < count):
            choice_index = _weighted_choice_index([candidate.rows for candidate in remaining], rng)
            selected_candidate = remaining.pop(choice_index)
            selected.append(selected_candidate)
            selected_rows += selected_candidate.rows

        allocations = {candidate.span_index: 0 for candidate in selected}
        capacities = {candidate.span_index: candidate.rows for candidate in selected}
        by_span = {candidate.span_index: candidate for candidate in selected}
        for _ in range(count):
            available = [
                candidate
                for candidate in selected
                if allocations[candidate.span_index] < capacities[candidate.span_index]
            ]
            choice_index = _weighted_choice_index(
                [
                    capacities[candidate.span_index] - allocations[candidate.span_index]
                    for candidate in available
                ],
                rng,
            )
            chosen = available[choice_index]
            allocations[chosen.span_index] += 1

        indexes: list[int] = []
        for span_index, allocation in allocations.items():
            if allocation == 0:
                continue
            candidate = by_span[span_index]
            indexes.extend(rng.sample(range(candidate.start, candidate.stop), allocation))
        return indexes

    def _overlapping_span_ranges(self, start: int, stop: int) -> list[_SpanRowRange]:
        ranges: list[_SpanRowRange] = []
        for span_index, span in enumerate(self._spans):
            overlap_start = max(start, span.start)
            overlap_stop = min(stop, span.stop)
            if overlap_start < overlap_stop:
                ranges.append(
                    _SpanRowRange(
                        span_index=span_index,
                        start=overlap_start,
                        stop=overlap_stop,
                    )
                )
        return ranges

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


class _SpanRowRange:
    def __init__(self, *, span_index: int, start: int, stop: int) -> None:
        self.span_index = span_index
        self.start = start
        self.stop = stop

    @property
    def rows(self) -> int:
        return self.stop - self.start


def _weighted_choice_index(weights: list[int], rng: random.Random) -> int:
    total = sum(weights)
    if total <= 0:
        raise ValueError("weights must contain a positive value")
    threshold = rng.randrange(total)
    running = 0
    for index, weight in enumerate(weights):
        running += weight
        if threshold < running:
            return index
    return len(weights) - 1


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
