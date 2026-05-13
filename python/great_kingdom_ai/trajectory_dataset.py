"""Training dataset view over v2 trajectory replay."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from great_kingdom_ai.priority_sampling import PrioritySamplingConfig, sample_priority_indexes
from great_kingdom_ai.replay_buffer import ReplaySample
from great_kingdom_ai.self_play_data import value_target_for_player
from great_kingdom_ai.trajectory_replay import TrajectoryReplayStore


@dataclass(frozen=True)
class TrajectoryArrayBatch:
    indexes: np.ndarray
    features: np.ndarray
    policies: np.ndarray
    values: np.ndarray
    sample_weights: np.ndarray
    legal_masks: np.ndarray


class TrajectoryReplayDataset:
    """Pure-Gumbel training view for trajectory replay.

    Policy targets are the self-play Gumbel targets stored in replay. Value
    targets are terminal outcomes from the sampled transition player's view.
    """

    def __init__(self, replay: TrajectoryReplayStore) -> None:
        if len(replay) == 0:
            raise ValueError("trajectory replay must contain at least one transition")
        self._replay = replay
        self._values = _terminal_values(replay)

    @property
    def capacity(self) -> int:
        return self._replay.capacity

    def __len__(self) -> int:
        return len(self._replay)

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
            raise ValueError("batch_size exceeds trajectory replay size")

        if priority_config is not None and priority_config.enabled:
            priorities = np.maximum(
                np.asarray(self._replay.sample_weights, dtype=np.float32),
                np.float32(1e-6),
            )
            sampled = sample_priority_indexes(
                priorities=priorities ** np.float32(priority_config.alpha),
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
        return TrajectoryArrayBatch(
            indexes=index_array,
            features=np.ascontiguousarray(self._replay.features[index_array], dtype=np.float32),
            policies=np.ascontiguousarray(
                self._replay.policy_targets[index_array],
                dtype=np.float32,
            ),
            values=np.ascontiguousarray(self._values[index_array], dtype=np.float32),
            sample_weights=np.ascontiguousarray(
                self._replay.sample_weights[index_array].astype(np.float32, copy=False)
                * importance_weights,
                dtype=np.float32,
            ),
            legal_masks=np.ascontiguousarray(
                self._replay.legal_masks[index_array],
                dtype=np.bool_,
            ),
        )


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


__all__ = ["TrajectoryArrayBatch", "TrajectoryReplayDataset"]
