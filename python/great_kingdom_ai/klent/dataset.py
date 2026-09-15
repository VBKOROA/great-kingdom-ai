"""KLENT dataset view over stored self-play shards.

Unlike the Gumbel dataset, value targets are the stored lambda returns and no
terminal or n-step bootstrap target is computed. Missing or invalid KLENT
fields are rejected instead of silently substituted.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np

from great_kingdom_ai.klent.types import KlentConfig
from great_kingdom_ai.priority_sampling import PrioritySamplingConfig
from great_kingdom_ai.replay.dataset import TrajectoryArrayBatch, sample_replay_indexes
from great_kingdom_ai.replay.sample import ReplaySample
from great_kingdom_ai.replay.trajectory import TrajectoryReplayStore

if TYPE_CHECKING:
    from great_kingdom_ai.klent.shards import KlentShardMetadata


class KlentReplayDataset:
    """Strict KLENT training view over trajectory replay transitions."""

    def __init__(
        self,
        replay: TrajectoryReplayStore,
        *,
        config: KlentConfig,
        metadata: KlentShardMetadata | None = None,
    ) -> None:
        if len(replay) == 0:
            raise ValueError("KLENT replay must contain at least one transition")
        if replay.features is None or replay.legal_masks is None:
            raise ValueError("KLENT replay requires stored features and legal masks")
        if replay.lambda_returns is None or replay.lambda_returns_present is None:
            raise ValueError("KLENT replay requires stored lambda returns")
        if not bool(replay.lambda_returns_present.all()):
            raise ValueError("KLENT replay lambda returns must be present for every transition")
        if metadata is not None:
            if not metadata.matches_config(config):
                raise ValueError("KLENT shard config does not match the training config")

        self._replay = replay
        self._config = config
        self._values = np.ascontiguousarray(replay.lambda_returns, dtype=np.float32)
        self._validate_actions()
        self._validate_policy_targets()

    @property
    def capacity(self) -> int:
        return self._replay.capacity

    def __len__(self) -> int:
        return len(self._replay)

    def sample(self, batch_size: int, rng: random.Random) -> list[ReplaySample]:
        batch = self.sample_arrays(batch_size, rng)
        assert batch.actions is not None
        return [
            ReplaySample(
                features=batch.features[index],
                policy=batch.policies[index],
                value=float(batch.values[index]),
                sample_weight=float(batch.sample_weights[index]),
                action=int(batch.actions[index]),
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
        index_array, importance_weights = sample_replay_indexes(
            self._replay,
            batch_size,
            rng,
            recent_fraction=recent_fraction,
            recent_window=recent_window,
            priority_config=priority_config,
        )
        batch = self.arrays_for_indexes(index_array)
        return TrajectoryArrayBatch(
            indexes=batch.indexes,
            features=batch.features,
            policies=batch.policies,
            values=batch.values,
            sample_weights=np.ascontiguousarray(
                batch.sample_weights * importance_weights,
                dtype=np.float32,
            ),
            legal_masks=batch.legal_masks,
            actions=batch.actions,
        )

    def arrays_for_indexes(self, indexes: np.ndarray) -> TrajectoryArrayBatch:
        """Return replay arrays for explicit row indexes without resampling."""
        index_array = np.asarray(indexes, dtype=np.int64)
        if index_array.ndim != 1 or index_array.shape[0] == 0:
            raise ValueError("indexes must be a non-empty rank-1 array")
        if np.any(index_array < 0) or np.any(index_array >= len(self._replay)):
            raise IndexError("indexes are out of range")
        assert self._replay.features is not None
        assert self._replay.legal_masks is not None
        return TrajectoryArrayBatch(
            indexes=index_array,
            features=np.ascontiguousarray(self._replay.features[index_array], dtype=np.float32),
            policies=np.ascontiguousarray(
                self._replay.policy_targets[index_array],
                dtype=np.float32,
            ),
            values=np.ascontiguousarray(self._values[index_array], dtype=np.float32),
            sample_weights=np.ascontiguousarray(
                self._replay.sample_weights[index_array].astype(np.float32, copy=False),
                dtype=np.float32,
            ),
            legal_masks=np.ascontiguousarray(
                self._replay.legal_masks[index_array],
                dtype=np.bool_,
            ),
            actions=np.ascontiguousarray(self._replay.actions[index_array], dtype=np.int64),
        )

    def update_sampling_priorities(
        self,
        indexes: np.ndarray,
        priorities: np.ndarray,
        *,
        ema: float,
        epsilon: float,
        max_priority: float | None,
    ) -> None:
        self._replay.update_sampling_priorities(
            indexes,
            priorities,
            ema=ema,
            epsilon=epsilon,
            max_priority=max_priority,
        )

    def _validate_actions(self) -> None:
        assert self._replay.legal_masks is not None
        rows = np.arange(len(self._replay), dtype=np.int64)
        legal = self._replay.legal_masks[rows, self._replay.actions]
        if not bool(legal.all()):
            raise ValueError("KLENT replay contains illegal taken actions")

    def _validate_policy_targets(self) -> None:
        assert self._replay.legal_masks is not None
        illegal_mass = float(
            self._replay.policy_targets[
                np.logical_not(self._replay.legal_masks)
            ].sum()
        )
        if illegal_mass > 1e-5:
            raise ValueError("KLENT replay policy targets assign probability to illegal actions")


__all__ = ["KlentReplayDataset"]