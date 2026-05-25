"""Training dataset view over v2 trajectory replay."""

from __future__ import annotations

import random
from dataclasses import dataclass
import math

import numpy as np

from great_kingdom_ai.features import ACTION_SPACE, BOARD_SIZE, FEATURE_CHANNELS
from great_kingdom_ai.game_core import import_core, request_feature_rows_and_masks
from great_kingdom_ai.priority_sampling import PrioritySamplingConfig, sample_priority_indexes
from great_kingdom_ai.replay.sample import ReplaySample
from great_kingdom_ai.replay.trajectory import TrajectoryReplayStore
from great_kingdom_ai.self_play_data import value_target_for_player


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
    targets are terminal outcomes by default, or n-step MCTS root values when
    configured. Feature tensors are reconstructed from the episode action
    timeline when the replay shard omits stored transition features.
    """

    def __init__(
        self,
        replay: TrajectoryReplayStore,
        *,
        bootstrap_td_steps: int = 0,
        gamma: float = 1.0,
        value_bootstrap_source: str = "terminal",
    ) -> None:
        if len(replay) == 0:
            raise ValueError("trajectory replay must contain at least one transition")
        if bootstrap_td_steps < 0:
            raise ValueError("bootstrap_td_steps must be non-negative")
        if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be finite and in [0, 1]")
        if value_bootstrap_source not in {"terminal", "mcts_root"}:
            raise ValueError("value_bootstrap_source must be one of: terminal, mcts_root")
        self._replay = replay
        if value_bootstrap_source == "terminal" or bootstrap_td_steps == 0:
            self._values = _terminal_values(replay)
        else:
            self._values = _mcts_root_bootstrap_values(
                replay,
                td_steps=bootstrap_td_steps,
                gamma=gamma,
            )

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
                np.asarray(self._replay.sampling_priorities, dtype=np.float32),
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
        features, legal_masks = _features_and_masks_for_rows(self._replay, index_array)
        return TrajectoryArrayBatch(
            indexes=index_array,
            features=features,
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
            legal_masks=legal_masks,
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


def _mcts_root_bootstrap_values(
    replay: TrajectoryReplayStore,
    *,
    td_steps: int,
    gamma: float,
) -> np.ndarray:
    episode_indexes = np.searchsorted(
        replay.episode_offsets,
        np.arange(len(replay), dtype=np.int64),
        side="right",
    ) - 1
    values = np.empty((len(replay),), dtype=np.float32)
    for row in range(len(replay)):
        episode_index = int(episode_indexes[row])
        turn_start = int(replay.turn_offsets[episode_index])
        turn_end = int(replay.turn_offsets[episode_index + 1])
        local_turn = int(replay.timesteps[row])
        target_local_turn = local_turn + td_steps
        if target_local_turn >= turn_end - turn_start:
            values[row] = _terminal_value_for_row(replay, row, episode_index)
            continue
        target_row = turn_start + target_local_turn
        root_value = float(replay.turn_root_values[target_row])
        if not math.isfinite(root_value):
            raise ValueError("mcts_root bootstrap requires finite turn root values")
        if int(replay.turn_players[target_row]) != int(replay.players[row]):
            root_value = -root_value
        values[row] = np.float32((gamma**td_steps) * root_value)
    return values


def _features_and_masks_for_rows(
    replay: TrajectoryReplayStore,
    indexes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if replay.features is not None and replay.legal_masks is not None:
        return (
            np.ascontiguousarray(replay.features[indexes], dtype=np.float32),
            np.ascontiguousarray(replay.legal_masks[indexes], dtype=np.bool_),
        )
    return _reconstruct_features_and_masks(replay, indexes)


def _reconstruct_features_and_masks(
    replay: TrajectoryReplayStore,
    indexes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    core = import_core("trajectory replay feature reconstruction")
    episode_indexes = np.searchsorted(
        replay.episode_offsets,
        indexes,
        side="right",
    ) - 1
    histories: list[list[int]] = []
    for output_row, replay_row in enumerate(indexes):
        episode_index = int(episode_indexes[output_row])
        turn_start = int(replay.turn_offsets[episode_index])
        local_turn = int(replay.timesteps[int(replay_row)])
        histories.append(
            [
                int(action)
                for action in replay.turn_actions[turn_start : turn_start + local_turn]
            ]
        )
    batch_type = getattr(core, "GumbelSelfPlayBatch", None)
    if batch_type is None or not hasattr(batch_type, "from_action_histories"):
        return _reconstruct_features_and_masks_slow(core, replay, indexes, episode_indexes)
    batch = batch_type.from_action_histories(
        histories,
        simulations=1,
        max_considered_actions=1,
        c_visit=1.0,
        c_scale=1.0,
        seed=0,
        gumbel_scale=1.0,
        policy_target_temperature=1.0,
        policy_target_c_visit=1.0,
        policy_target_c_scale=1.0,
    )
    if list(batch.active_game_indexes()) != list(range(indexes.shape[0])):
        raise ValueError("reconstructed feature batch contains terminal states")
    request = batch.active_eval_request()
    feature_rows, mask_rows, row_count = request_feature_rows_and_masks(request)
    if row_count != indexes.shape[0]:
        raise ValueError("reconstructed feature row count mismatch")
    features = np.asarray(feature_rows, dtype=np.float32)
    if features.shape == (indexes.shape[0], FEATURE_CHANNELS * BOARD_SIZE * BOARD_SIZE):
        features = features.reshape(indexes.shape[0], FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    if features.shape != (indexes.shape[0], FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE):
        raise ValueError("reconstructed feature shape mismatch")
    legal_masks = np.asarray(mask_rows, dtype=np.bool_)
    if legal_masks.shape != (indexes.shape[0], ACTION_SPACE):
        raise ValueError("reconstructed legal mask shape mismatch")
    players = [int(player) for player in batch.current_players()]
    for output_row, replay_row in enumerate(indexes):
        if players[output_row] != int(replay.players[int(replay_row)]):
            raise ValueError("reconstructed player does not match replay row")
    return (
        np.ascontiguousarray(features, dtype=np.float32),
        np.ascontiguousarray(legal_masks, dtype=np.bool_),
    )


def _reconstruct_features_and_masks_slow(
    core: object,
    replay: TrajectoryReplayStore,
    indexes: np.ndarray,
    episode_indexes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    features = np.empty(
        (indexes.shape[0], FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE),
        dtype=np.float32,
    )
    legal_masks = np.empty((indexes.shape[0], ACTION_SPACE), dtype=np.bool_)
    expected_flat = FEATURE_CHANNELS * BOARD_SIZE * BOARD_SIZE
    game_state = getattr(core, "GameState")
    for output_row, replay_row in enumerate(indexes):
        episode_index = int(episode_indexes[output_row])
        turn_start = int(replay.turn_offsets[episode_index])
        local_turn = int(replay.timesteps[int(replay_row)])
        state = game_state()
        for action in replay.turn_actions[turn_start : turn_start + local_turn]:
            if state.is_terminal():
                raise ValueError("cannot reconstruct feature after terminal state")
            state.apply_action(int(action))
        flat = np.asarray(state.feature_planes(), dtype=np.float32)
        if flat.shape != (expected_flat,):
            raise ValueError("reconstructed feature shape mismatch")
        features[output_row] = flat.reshape(FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE)
        legal_masks[output_row] = np.asarray(state.legal_mask(), dtype=np.bool_)
        if int(state.current_player()) != int(replay.players[int(replay_row)]):
            raise ValueError("reconstructed player does not match replay row")
    return (
        np.ascontiguousarray(features, dtype=np.float32),
        np.ascontiguousarray(legal_masks, dtype=np.bool_),
    )


def _terminal_value_for_row(
    replay: TrajectoryReplayStore,
    row: int,
    episode_index: int,
) -> np.float32:
    return np.float32(
        value_target_for_player(
            player=int(replay.players[row]),
            winner=int(replay.episode_winners[episode_index]),
        )
    )


__all__ = ["TrajectoryArrayBatch", "TrajectoryReplayDataset"]
