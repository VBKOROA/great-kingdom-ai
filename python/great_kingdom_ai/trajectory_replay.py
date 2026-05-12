"""Trajectory replay storage and legacy replay compatibility views."""

from __future__ import annotations

import random
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np

from great_kingdom_ai.features import (
    ACTION_SPACE,
    BOARD_CELLS,
    BOARD_SIZE,
    FEATURE_CHANNELS,
    LEGAL_PLACE_FEATURE_CHANNEL,
    PASS_ACTION,
)
from great_kingdom_ai.replay_buffer import ReplayBuffer, ReplaySample
from great_kingdom_ai.trajectory_targets import (
    BootstrapValueTargetConfig,
    replay_sample_from_episode_transition,
    replay_sample_from_transition,
)

FEATURE_SHAPE = (FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE)


class MoveLike(Protocol):
    turn: int
    player: int
    action: int


class GameLogLike(Protocol):
    seed: int
    moves: Sequence[MoveLike]
    winner: int
    end_reason: int
    territory_scores: tuple[int, int]


@dataclass(frozen=True)
class TrajectoryTransition:
    episode_id: int
    timestep: int
    player: int
    features: np.ndarray
    legal_mask: np.ndarray
    action: int
    policy_target: np.ndarray
    root_policy_logits: np.ndarray | None = None
    root_value: float | None = None
    next_features: np.ndarray | None = None
    winner: int | None = None
    terminal: bool = False
    model_version: int = 0
    search_config_hash: str = ""
    created_iteration: int = 0
    sample_weight: float = 1.0


@dataclass(frozen=True)
class TrajectoryEpisode:
    episode_id: int
    seed: int
    transitions: tuple[TrajectoryTransition, ...]
    winner: int
    end_reason: int
    territory_scores: tuple[int, int]


class TrajectoryReplayBuffer:
    """Fixed-capacity in-memory trajectory replay buffer.

    Capacity is counted in transitions. When new episodes push the buffer over
    capacity, whole oldest episodes are evicted so episode boundaries remain intact.
    """

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = capacity
        self._episodes: deque[TrajectoryEpisode] = deque()
        self._transition_count = 0

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def episode_count(self) -> int:
        return len(self._episodes)

    @property
    def episodes(self) -> tuple[TrajectoryEpisode, ...]:
        return tuple(self._episodes)

    def __len__(self) -> int:
        return self._transition_count

    def push_episode(self, episode: TrajectoryEpisode) -> None:
        validated = _validated_episode(episode)
        episode_size = len(validated.transitions)
        if episode_size > self._capacity:
            raise ValueError("episode transition count exceeds replay capacity")

        self._episodes.append(validated)
        self._transition_count += episode_size
        while self._transition_count > self._capacity:
            removed = self._episodes.popleft()
            self._transition_count -= len(removed.transitions)

    def extend_episodes(self, episodes: Sequence[TrajectoryEpisode]) -> None:
        for episode in episodes:
            self.push_episode(episode)

    def sample(self, batch_size: int, rng: random.Random) -> list[TrajectoryTransition]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        transitions = self.transitions()
        if batch_size > len(transitions):
            raise ValueError("batch_size exceeds trajectory replay size")
        indexes = rng.sample(range(len(transitions)), batch_size)
        return [transitions[index] for index in indexes]

    def sample_replay_samples(
        self,
        batch_size: int,
        rng: random.Random,
        *,
        value_target_config: BootstrapValueTargetConfig | None = None,
    ) -> list[ReplaySample]:
        if value_target_config is None:
            return [
                replay_sample_from_transition(transition)
                for transition in self.sample(batch_size, rng)
            ]
        refs = self._sample_transition_refs(batch_size, rng)
        return [
            replay_sample_from_episode_transition(
                episode,
                transition_index,
                value_target_config=value_target_config,
            )
            for episode, transition_index in refs
        ]

    def transitions(self) -> list[TrajectoryTransition]:
        return [
            transition
            for episode in self._episodes
            for transition in episode.transitions
        ]

    def as_replay_buffer(
        self,
        *,
        capacity: int | None = None,
        value_target_config: BootstrapValueTargetConfig | None = None,
    ) -> ReplayBuffer:
        replay = ReplayBuffer(capacity=capacity or max(1, len(self)))
        if value_target_config is None:
            for transition in self.transitions():
                replay.push(replay_sample_from_transition(transition))
            return replay
        for episode in self._episodes:
            for transition_index in range(len(episode.transitions)):
                replay.push(
                    replay_sample_from_episode_transition(
                        episode,
                        transition_index,
                        value_target_config=value_target_config,
                    )
                )
        return replay

    def save(self, path: str | Path, *, compressed: bool = True) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = _episodes_to_payload(self._capacity, list(self._episodes))
        save = np.savez_compressed if compressed else np.savez
        save(destination, **cast(dict[str, Any], payload))

    def _sample_transition_refs(
        self,
        batch_size: int,
        rng: random.Random,
    ) -> list[tuple[TrajectoryEpisode, int]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        refs = [
            (episode, transition_index)
            for episode in self._episodes
            for transition_index in range(len(episode.transitions))
        ]
        if batch_size > len(refs):
            raise ValueError("batch_size exceeds trajectory replay size")
        indexes = rng.sample(range(len(refs)), batch_size)
        return [refs[index] for index in indexes]

    @classmethod
    def load(cls, path: str | Path) -> TrajectoryReplayBuffer:
        with np.load(Path(path)) as data:
            capacity = int(data["capacity"])
            episodes = _episodes_from_payload(data)

        buffer = cls(capacity)
        buffer.extend_episodes(episodes)
        return buffer


def trajectory_episode_from_self_play_result(
    log: GameLogLike,
    samples: Sequence[ReplaySample],
    *,
    episode_id: int,
    root_values: Sequence[float | None] | None = None,
    model_version: int = 0,
    search_config_hash: str = "",
    created_iteration: int = 0,
) -> TrajectoryEpisode:
    """Build a trajectory episode from a self-play game log and per-turn samples.

    This helper expects one replay sample per move. If playout-cap randomization
    filtered out some moves, callers should build transitions explicitly instead.
    """
    if len(log.moves) != len(samples):
        raise ValueError(
            "self-play log moves and replay samples must have the same length "
            "to build a full trajectory"
        )
    if root_values is not None and len(root_values) != len(samples):
        raise ValueError("root_values and replay samples must have the same length")
    transitions = []
    for index, (move, sample) in enumerate(zip(log.moves, samples, strict=True)):
        next_features = samples[index + 1].features if index + 1 < len(samples) else None
        transitions.append(
            TrajectoryTransition(
                episode_id=episode_id,
                timestep=int(move.turn),
                player=int(move.player),
                features=sample.features,
                legal_mask=legal_mask_from_features(sample.features),
                action=int(move.action),
                policy_target=sample.policy,
                root_policy_logits=sample.root_policy_logits,
                root_value=None if root_values is None else root_values[index],
                next_features=next_features,
                winner=int(log.winner),
                terminal=index == len(samples) - 1,
                model_version=model_version,
                search_config_hash=search_config_hash,
                created_iteration=created_iteration,
                sample_weight=sample.sample_weight,
            )
        )
    return TrajectoryEpisode(
        episode_id=episode_id,
        seed=int(log.seed),
        transitions=tuple(transitions),
        winner=int(log.winner),
        end_reason=int(log.end_reason),
        territory_scores=log.territory_scores,
    )


def legal_mask_from_features(features: np.ndarray) -> np.ndarray:
    feature_array = np.asarray(features, dtype=np.float32)
    if feature_array.shape != FEATURE_SHAPE:
        raise ValueError(f"expected feature shape {FEATURE_SHAPE}, got {feature_array.shape}")
    legal_place = feature_array[LEGAL_PLACE_FEATURE_CHANNEL].reshape(BOARD_CELLS) > 0.5
    legal_mask = np.zeros((ACTION_SPACE,), dtype=np.bool_)
    legal_mask[:BOARD_CELLS] = legal_place
    legal_mask[PASS_ACTION] = True
    return legal_mask


def _validated_episode(episode: TrajectoryEpisode) -> TrajectoryEpisode:
    if episode.episode_id < 0:
        raise ValueError("episode_id must be non-negative")
    if not episode.transitions:
        raise ValueError("episode must contain at least one transition")
    if episode.winner not in {1, 2}:
        raise ValueError("episode winner must be 1 or 2")
    if len(episode.territory_scores) != 2:
        raise ValueError("territory_scores must contain two values")

    transitions = tuple(_validated_transition(transition) for transition in episode.transitions)
    for index, transition in enumerate(transitions):
        if transition.episode_id != episode.episode_id:
            raise ValueError("transition episode_id must match episode")
        if transition.timestep != index:
            raise ValueError("transition timesteps must be contiguous from zero")
        if transition.winner != episode.winner:
            raise ValueError("transition winner must match episode winner")
        expected_terminal = index == len(transitions) - 1
        if transition.terminal is not expected_terminal:
            raise ValueError("only the final transition may be terminal")

    return TrajectoryEpisode(
        episode_id=episode.episode_id,
        seed=int(episode.seed),
        transitions=transitions,
        winner=int(episode.winner),
        end_reason=int(episode.end_reason),
        territory_scores=(int(episode.territory_scores[0]), int(episode.territory_scores[1])),
    )


def _validated_transition(transition: TrajectoryTransition) -> TrajectoryTransition:
    features = np.asarray(transition.features, dtype=np.float32)
    legal_mask = np.asarray(transition.legal_mask, dtype=np.bool_)
    policy = np.asarray(transition.policy_target, dtype=np.float32)
    root_policy_logits = _optional_vector(
        transition.root_policy_logits,
        shape=(ACTION_SPACE,),
        label="root_policy_logits",
    )
    next_features = _optional_features(transition.next_features, "next_features")
    root_value = None if transition.root_value is None else float(transition.root_value)
    sample_weight = float(transition.sample_weight)

    if transition.episode_id < 0:
        raise ValueError("episode_id must be non-negative")
    if transition.timestep < 0:
        raise ValueError("timestep must be non-negative")
    if transition.player not in {1, 2}:
        raise ValueError("player must be 1 or 2")
    if transition.action < 0 or transition.action >= ACTION_SPACE:
        raise ValueError(f"action must be in [0, {ACTION_SPACE})")
    if features.shape != FEATURE_SHAPE:
        raise ValueError(f"expected feature shape {FEATURE_SHAPE}, got {features.shape}")
    if legal_mask.shape != (ACTION_SPACE,):
        raise ValueError(f"expected legal_mask shape {(ACTION_SPACE,)}, got {legal_mask.shape}")
    if not legal_mask[transition.action]:
        raise ValueError("transition action must be legal")
    if policy.shape != (ACTION_SPACE,):
        raise ValueError(f"expected policy_target shape {(ACTION_SPACE,)}, got {policy.shape}")
    if np.any(policy < 0.0):
        raise ValueError("policy target must be non-negative")
    if not np.isclose(policy.sum(), 1.0):
        raise ValueError("policy target must sum to 1")
    if transition.winner is not None and transition.winner not in {1, 2}:
        raise ValueError("winner must be 1, 2, or None")
    if root_value is not None:
        if not np.isfinite(root_value) or root_value < -1.0 or root_value > 1.0:
            raise ValueError("root_value must be finite and in [-1, 1]")
    if transition.model_version < 0:
        raise ValueError("model_version must be non-negative")
    if transition.created_iteration < 0:
        raise ValueError("created_iteration must be non-negative")
    if not np.isfinite(sample_weight) or sample_weight <= 0.0:
        raise ValueError("sample_weight must be finite and positive")

    return TrajectoryTransition(
        episode_id=int(transition.episode_id),
        timestep=int(transition.timestep),
        player=int(transition.player),
        features=features.copy(),
        legal_mask=legal_mask.copy(),
        action=int(transition.action),
        policy_target=policy.copy(),
        root_policy_logits=(
            None if root_policy_logits is None else root_policy_logits.copy()
        ),
        root_value=root_value,
        next_features=None if next_features is None else next_features.copy(),
        winner=None if transition.winner is None else int(transition.winner),
        terminal=bool(transition.terminal),
        model_version=int(transition.model_version),
        search_config_hash=str(transition.search_config_hash),
        created_iteration=int(transition.created_iteration),
        sample_weight=sample_weight,
    )


def _optional_vector(
    value: np.ndarray | None,
    *,
    shape: tuple[int, ...],
    label: str,
) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape:
        raise ValueError(f"expected {label} shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must be finite")
    return array


def _optional_features(value: np.ndarray | None, label: str) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32)
    if array.shape != FEATURE_SHAPE:
        raise ValueError(f"expected {label} shape {FEATURE_SHAPE}, got {array.shape}")
    return array


def _episodes_to_payload(
    capacity: int,
    episodes: Sequence[TrajectoryEpisode],
) -> dict[str, np.ndarray]:
    transitions = [
        transition
        for episode in episodes
        for transition in episode.transitions
    ]
    offsets = [0]
    for episode in episodes:
        offsets.append(offsets[-1] + len(episode.transitions))

    payload: dict[str, np.ndarray] = {
        "capacity": np.asarray(capacity, dtype=np.int64),
        "episode_ids": np.asarray([episode.episode_id for episode in episodes], dtype=np.int64),
        "episode_seeds": np.asarray([episode.seed for episode in episodes], dtype=np.int64),
        "episode_winners": np.asarray([episode.winner for episode in episodes], dtype=np.int64),
        "episode_end_reasons": np.asarray(
            [episode.end_reason for episode in episodes],
            dtype=np.int64,
        ),
        "territory_scores": np.asarray(
            [episode.territory_scores for episode in episodes],
            dtype=np.int64,
        ).reshape(len(episodes), 2),
        "episode_offsets": np.asarray(offsets, dtype=np.int64),
        "timesteps": np.asarray(
            [transition.timestep for transition in transitions],
            dtype=np.int64,
        ),
        "players": np.asarray([transition.player for transition in transitions], dtype=np.int64),
        "actions": np.asarray([transition.action for transition in transitions], dtype=np.int64),
        "features": _stack_or_empty(
            [transition.features for transition in transitions],
            shape=FEATURE_SHAPE,
            dtype=np.float32,
        ),
        "legal_masks": _stack_or_empty(
            [transition.legal_mask for transition in transitions],
            shape=(ACTION_SPACE,),
            dtype=np.bool_,
        ),
        "policy_targets": _stack_or_empty(
            [transition.policy_target for transition in transitions],
            shape=(ACTION_SPACE,),
            dtype=np.float32,
        ),
        "winners": np.asarray(
            [-1 if transition.winner is None else transition.winner for transition in transitions],
            dtype=np.int64,
        ),
        "terminals": np.asarray(
            [transition.terminal for transition in transitions],
            dtype=np.bool_,
        ),
        "root_values": np.asarray(
            [
                np.nan if transition.root_value is None else transition.root_value
                for transition in transitions
            ],
            dtype=np.float32,
        ),
        "model_versions": np.asarray(
            [transition.model_version for transition in transitions],
            dtype=np.int64,
        ),
        "created_iterations": np.asarray(
            [transition.created_iteration for transition in transitions],
            dtype=np.int64,
        ),
        "sample_weights": np.asarray(
            [transition.sample_weight for transition in transitions],
            dtype=np.float32,
        ),
    }
    _add_search_config_hashes(
        payload,
        [transition.search_config_hash for transition in transitions],
    )
    _add_optional_2d(payload, "root_policy_logits", [
        transition.root_policy_logits for transition in transitions
    ], (ACTION_SPACE,))
    _add_optional_4d(payload, "next_features", [
        transition.next_features for transition in transitions
    ], FEATURE_SHAPE)
    return payload


def _episodes_from_payload(data: Any) -> list[TrajectoryEpisode]:
    offsets = np.asarray(data["episode_offsets"], dtype=np.int64)
    episode_ids = np.asarray(data["episode_ids"], dtype=np.int64)
    episode_seeds = np.asarray(data["episode_seeds"], dtype=np.int64)
    episode_winners = np.asarray(data["episode_winners"], dtype=np.int64)
    episode_end_reasons = np.asarray(data["episode_end_reasons"], dtype=np.int64)
    territory_scores = np.asarray(data["territory_scores"], dtype=np.int64)
    timesteps = np.asarray(data["timesteps"], dtype=np.int64)
    players = np.asarray(data["players"], dtype=np.int64)
    actions = np.asarray(data["actions"], dtype=np.int64)
    features = np.asarray(data["features"], dtype=np.float32)
    legal_masks = np.asarray(data["legal_masks"], dtype=np.bool_)
    policy_targets = np.asarray(data["policy_targets"], dtype=np.float32)
    winners = np.asarray(data["winners"], dtype=np.int64)
    terminals = np.asarray(data["terminals"], dtype=np.bool_)
    root_values = np.asarray(data["root_values"], dtype=np.float32)
    model_versions = np.asarray(data["model_versions"], dtype=np.int64)
    created_iterations = np.asarray(data["created_iterations"], dtype=np.int64)
    sample_weights = np.asarray(data["sample_weights"], dtype=np.float32)
    transition_count = features.shape[0]
    _validate_payload_lengths(data, transition_count)

    root_policy_logits, root_policy_present = _load_optional_array(
        data,
        key="root_policy_logits",
        shape=(transition_count, ACTION_SPACE),
    )
    next_features, next_features_present = _load_optional_array(
        data,
        key="next_features",
        shape=(transition_count, *FEATURE_SHAPE),
    )
    search_config_hashes = _load_search_config_hashes(data, transition_count)

    episodes: list[TrajectoryEpisode] = []
    for episode_index in range(len(offsets) - 1):
        start = int(offsets[episode_index])
        end = int(offsets[episode_index + 1])
        transitions = []
        for row in range(start, end):
            winner = int(winners[row])
            transitions.append(
                TrajectoryTransition(
                    episode_id=int(episode_ids[episode_index]),
                    timestep=int(timesteps[row]),
                    player=int(players[row]),
                    features=features[row],
                    legal_mask=legal_masks[row],
                    action=int(actions[row]),
                    policy_target=policy_targets[row],
                    root_policy_logits=(
                        root_policy_logits[row] if root_policy_present[row] else None
                    ),
                    root_value=_none_if_nan(float(root_values[row])),
                    next_features=next_features[row] if next_features_present[row] else None,
                    winner=None if winner < 0 else winner,
                    terminal=bool(terminals[row]),
                    model_version=int(model_versions[row]),
                    search_config_hash=search_config_hashes[row],
                    created_iteration=int(created_iterations[row]),
                    sample_weight=float(sample_weights[row]),
                )
            )
        territory_score_row = territory_scores[episode_index]
        episodes.append(
            TrajectoryEpisode(
                episode_id=int(episode_ids[episode_index]),
                seed=int(episode_seeds[episode_index]),
                transitions=tuple(transitions),
                winner=int(episode_winners[episode_index]),
                end_reason=int(episode_end_reasons[episode_index]),
                territory_scores=(int(territory_score_row[0]), int(territory_score_row[1])),
            )
        )
    return episodes


def _add_search_config_hashes(
    payload: dict[str, np.ndarray],
    hashes: Sequence[str],
) -> None:
    table: list[str] = []
    indexes: dict[str, int] = {}
    ids = np.empty((len(hashes),), dtype=np.int32)
    for row, value in enumerate(hashes):
        key = str(value)
        index = indexes.get(key)
        if index is None:
            index = len(table)
            indexes[key] = index
            table.append(key)
        ids[row] = index
    payload["search_config_hash_table"] = np.asarray(table, dtype=np.str_)
    payload["search_config_hash_ids"] = ids


def _load_search_config_hashes(data: Any, transition_count: int) -> list[str]:
    if "search_config_hash_ids" in data and "search_config_hash_table" in data:
        table = np.asarray(data["search_config_hash_table"], dtype=np.str_)
        ids = np.asarray(data["search_config_hash_ids"], dtype=np.int64)
        if ids.shape != (transition_count,):
            raise ValueError("trajectory replay search_config_hash_ids length mismatch")
        if np.any(ids < 0) or np.any(ids >= len(table)):
            raise ValueError("trajectory replay search_config_hash_ids contain invalid indexes")
        return [str(table[index]) for index in ids]
    return [str(value) for value in np.asarray(data["search_config_hashes"], dtype=np.str_)]


def _stack_or_empty(
    arrays: Sequence[np.ndarray],
    *,
    shape: tuple[int, ...],
    dtype: Any,
) -> np.ndarray:
    if not arrays:
        return np.empty((0, *shape), dtype=dtype)
    return np.stack(arrays, axis=0).astype(dtype)


def _add_optional_2d(
    payload: dict[str, np.ndarray],
    key: str,
    arrays: Sequence[np.ndarray | None],
    shape: tuple[int, ...],
) -> None:
    _add_optional_array(payload, key, arrays, shape)


def _add_optional_4d(
    payload: dict[str, np.ndarray],
    key: str,
    arrays: Sequence[np.ndarray | None],
    shape: tuple[int, ...],
) -> None:
    _add_optional_array(payload, key, arrays, shape)


def _add_optional_array(
    payload: dict[str, np.ndarray],
    key: str,
    arrays: Sequence[np.ndarray | None],
    shape: tuple[int, ...],
) -> None:
    if not any(array is not None for array in arrays):
        return
    values = np.full((len(arrays), *shape), np.nan, dtype=np.float32)
    present = np.zeros((len(arrays),), dtype=np.bool_)
    for index, array in enumerate(arrays):
        if array is None:
            continue
        values[index] = np.asarray(array, dtype=np.float32)
        present[index] = True
    payload[key] = values
    payload[f"{key}_present"] = present


def _load_optional_array(
    data: Any,
    *,
    key: str,
    shape: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    if key not in data:
        return np.empty(shape, dtype=np.float32), np.zeros((shape[0],), dtype=np.bool_)
    values = np.asarray(data[key], dtype=np.float32)
    present = np.asarray(data[f"{key}_present"], dtype=np.bool_)
    if values.shape != shape:
        raise ValueError(f"trajectory replay {key} shape must be {shape}")
    if present.shape != (shape[0],):
        raise ValueError(f"trajectory replay {key}_present length mismatch")
    return values, present


def _validate_payload_lengths(data: Any, transition_count: int) -> None:
    per_transition_keys = (
        "timesteps",
        "players",
        "actions",
        "legal_masks",
        "policy_targets",
        "winners",
        "terminals",
        "root_values",
        "model_versions",
        "created_iterations",
        "sample_weights",
    )
    for key in per_transition_keys:
        if np.asarray(data[key]).shape[0] != transition_count:
            raise ValueError(f"trajectory replay {key} length mismatch")
    offsets = np.asarray(data["episode_offsets"], dtype=np.int64)
    if offsets.size == 0 or int(offsets[0]) != 0 or int(offsets[-1]) != transition_count:
        raise ValueError("trajectory replay episode_offsets are inconsistent")
    if "search_config_hash_ids" in data:
        if np.asarray(data["search_config_hash_ids"]).shape[0] != transition_count:
            raise ValueError("trajectory replay search_config_hash_ids length mismatch")
        if "search_config_hash_table" not in data:
            raise ValueError("trajectory replay missing search_config_hash_table")
    elif np.asarray(data["search_config_hashes"]).shape[0] != transition_count:
        raise ValueError("trajectory replay search_config_hashes length mismatch")


def _none_if_nan(value: float) -> float | None:
    return None if np.isnan(value) else value
