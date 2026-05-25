"""Array-backed trajectory replay storage."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from great_kingdom_ai.features import (
    ACTION_SPACE,
    BOARD_CELLS,
    LEGAL_PLACE_FEATURE_CHANNEL,
    PASS_ACTION,
)
from great_kingdom_ai.replay.persistence import NpzSaveStats, save_npz_atomic
from great_kingdom_ai.replay.sample import ReplaySample
from great_kingdom_ai.replay.schema import (
    FEATURE_SHAPE,
    GameLogLike,
    TrajectoryEpisode,
    TrajectoryTransition,
)


@dataclass
class TrajectoryReplayStore:
    """Array-backed trajectory replay store for training-scale replay operations."""

    capacity: int
    episode_ids: np.ndarray
    episode_seeds: np.ndarray
    episode_winners: np.ndarray
    episode_end_reasons: np.ndarray
    territory_scores: np.ndarray
    episode_offsets: np.ndarray
    timesteps: np.ndarray
    players: np.ndarray
    actions: np.ndarray
    features: np.ndarray
    legal_masks: np.ndarray
    policy_targets: np.ndarray
    winners: np.ndarray
    terminals: np.ndarray
    root_values: np.ndarray
    model_versions: np.ndarray
    created_iterations: np.ndarray
    sample_weights: np.ndarray
    sampling_priorities: np.ndarray
    search_config_hash_table: np.ndarray
    search_config_hash_ids: np.ndarray
    root_policy_logits: np.ndarray | None = None
    root_policy_logits_present: np.ndarray | None = None
    next_features: np.ndarray | None = None
    next_features_present: np.ndarray | None = None

    @classmethod
    def empty(cls, capacity: int) -> TrajectoryReplayStore:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        return cls.from_episodes(capacity, ())

    @classmethod
    def from_episodes(
        cls,
        capacity: int,
        episodes: Sequence[TrajectoryEpisode],
    ) -> TrajectoryReplayStore:
        return cls.from_payload(_episodes_to_payload(capacity, episodes))

    @classmethod
    def load(cls, path: str | Path) -> TrajectoryReplayStore:
        with np.load(Path(path)) as data:
            return cls.from_payload(data)

    @classmethod
    def from_payload(cls, data: Any) -> TrajectoryReplayStore:
        capacity = int(data["capacity"])
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        features = np.asarray(data["features"], dtype=np.float32)
        transition_count = features.shape[0]
        _validate_payload_lengths(data, transition_count)
        root_policy_logits, root_policy_logits_present = _load_optional_array(
            data,
            key="root_policy_logits",
            shape=(transition_count, ACTION_SPACE),
        )
        next_features, next_features_present = _load_optional_array(
            data,
            key="next_features",
            shape=(transition_count, *FEATURE_SHAPE),
        )
        table, ids = _load_search_config_hash_encoding(data, transition_count)
        store = cls(
            capacity=capacity,
            episode_ids=np.asarray(data["episode_ids"], dtype=np.int64),
            episode_seeds=np.asarray(data["episode_seeds"], dtype=np.int64),
            episode_winners=np.asarray(data["episode_winners"], dtype=np.int64),
            episode_end_reasons=np.asarray(data["episode_end_reasons"], dtype=np.int64),
            territory_scores=np.asarray(data["territory_scores"], dtype=np.int64),
            episode_offsets=np.asarray(data["episode_offsets"], dtype=np.int64),
            timesteps=np.asarray(data["timesteps"], dtype=np.int64),
            players=np.asarray(data["players"], dtype=np.int64),
            actions=np.asarray(data["actions"], dtype=np.int64),
            features=features,
            legal_masks=np.asarray(data["legal_masks"], dtype=np.bool_),
            policy_targets=np.asarray(data["policy_targets"], dtype=np.float32),
            winners=np.asarray(data["winners"], dtype=np.int64),
            terminals=np.asarray(data["terminals"], dtype=np.bool_),
            root_values=np.asarray(data["root_values"], dtype=np.float32),
            model_versions=np.asarray(data["model_versions"], dtype=np.int64),
            created_iterations=np.asarray(data["created_iterations"], dtype=np.int64),
            sample_weights=np.asarray(data["sample_weights"], dtype=np.float32),
            sampling_priorities=_load_sampling_priorities(data, transition_count),
            search_config_hash_table=table,
            search_config_hash_ids=ids,
            root_policy_logits=root_policy_logits if root_policy_logits_present.any() else None,
            root_policy_logits_present=(
                root_policy_logits_present if root_policy_logits_present.any() else None
            ),
            next_features=next_features if next_features_present.any() else None,
            next_features_present=next_features_present if next_features_present.any() else None,
        )
        store.validate()
        return store

    @property
    def episode_count(self) -> int:
        return int(self.episode_ids.shape[0])

    @property
    def episodes(self) -> tuple[TrajectoryEpisode, ...]:
        return tuple(_episodes_from_store(self))

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def validate(self) -> None:
        if self.features.shape != (len(self), *FEATURE_SHAPE):
            raise ValueError("trajectory replay features shape mismatch")
        if self.legal_masks.shape != (len(self), ACTION_SPACE):
            raise ValueError("trajectory replay legal_masks shape mismatch")
        if self.policy_targets.shape != (len(self), ACTION_SPACE):
            raise ValueError("trajectory replay policy_targets shape mismatch")
        if self.territory_scores.shape != (self.episode_count, 2):
            raise ValueError("trajectory replay territory_scores shape mismatch")
        if self.episode_offsets.shape != (self.episode_count + 1,):
            raise ValueError("trajectory replay episode_offsets length mismatch")
        if self.episode_offsets.size == 0 or int(self.episode_offsets[0]) != 0:
            raise ValueError("trajectory replay episode_offsets must start at zero")
        if int(self.episode_offsets[-1]) != len(self):
            raise ValueError("trajectory replay episode_offsets must end at transition count")
        per_transition = (
            self.timesteps,
            self.players,
            self.actions,
            self.winners,
            self.terminals,
            self.root_values,
            self.model_versions,
            self.created_iterations,
            self.sample_weights,
            self.sampling_priorities,
            self.search_config_hash_ids,
        )
        for array in per_transition:
            if array.shape != (len(self),):
                raise ValueError("trajectory replay per-transition array length mismatch")
        if self.root_policy_logits is not None:
            if self.root_policy_logits.shape != (len(self), ACTION_SPACE):
                raise ValueError("trajectory replay root_policy_logits shape mismatch")
            if self.root_policy_logits_present is None:
                raise ValueError("trajectory replay root_policy_logits_present missing")
        if self.next_features is not None:
            if self.next_features.shape != (len(self), *FEATURE_SHAPE):
                raise ValueError("trajectory replay next_features shape mismatch")
            if self.next_features_present is None:
                raise ValueError("trajectory replay next_features_present missing")
        if self.search_config_hash_ids.shape != (len(self),):
            raise ValueError("trajectory replay search_config_hash_ids length mismatch")
        if np.any(self.search_config_hash_ids < 0) or np.any(
            self.search_config_hash_ids >= len(self.search_config_hash_table)
        ):
            raise ValueError("trajectory replay search_config_hash_ids contain invalid indexes")
        if not np.isfinite(self.sampling_priorities).all() or np.any(
            self.sampling_priorities <= 0.0
        ):
            raise ValueError("trajectory replay sampling_priorities must be finite and positive")

    def current_max_sampling_priority(self) -> float:
        if len(self) == 0:
            return 1.0
        return float(np.max(self.sampling_priorities, initial=np.float32(1.0)))

    def update_sampling_priorities(
        self,
        indexes: np.ndarray,
        priorities: np.ndarray,
        *,
        ema: float,
        epsilon: float,
        max_priority: float | None,
    ) -> None:
        index_array = np.asarray(indexes, dtype=np.int64)
        priority_array = np.asarray(priorities, dtype=np.float32)
        if index_array.shape != priority_array.shape:
            raise ValueError("priority update indexes and priorities must have the same shape")
        if not 0.0 <= ema <= 1.0 or not np.isfinite(ema):
            raise ValueError("priority ema must be finite and in [0, 1]")
        if not np.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("priority epsilon must be finite and positive")
        if max_priority is not None and (
            not np.isfinite(max_priority) or max_priority <= epsilon
        ):
            raise ValueError("priority max_priority must be greater than epsilon")
        if np.any(index_array < 0) or np.any(index_array >= len(self)):
            raise IndexError("priority update index is out of range")
        if not np.isfinite(priority_array).all():
            raise ValueError("updated priorities must be finite")
        clipped = np.maximum(priority_array, np.float32(epsilon))
        if max_priority is not None:
            clipped = np.minimum(clipped, np.float32(max_priority))
        old = self.sampling_priorities[index_array].astype(np.float32, copy=False)
        self.sampling_priorities[index_array] = np.asarray(
            np.float32(ema) * old + np.float32(1.0 - ema) * clipped,
            dtype=np.float32,
        )

    def extend_episodes(self, episodes: Sequence[TrajectoryEpisode]) -> None:
        if not episodes:
            return
        incoming = TrajectoryReplayStore.from_episodes(self.capacity, episodes)
        self.extend_store(incoming)

    def extend_store(self, incoming: TrajectoryReplayStore) -> None:
        self.extend_stores((incoming,))

    def extend_stores(
        self,
        incoming_stores: Sequence[TrajectoryReplayStore],
        *,
        defer_capacity_eviction: bool = False,
    ) -> None:
        non_empty = tuple(store for store in incoming_stores if len(store) > 0)
        if not non_empty:
            return
        if not defer_capacity_eviction and any(len(store) > self.capacity for store in non_empty):
            raise ValueError("episode transition count exceeds replay capacity")
        combined = _concat_many_stores(self, non_empty)
        kept = combined if defer_capacity_eviction else _evict_to_capacity(combined)
        self.__dict__.update(kept.__dict__)

    def compact_to_capacity(self, capacity: int | None = None) -> None:
        target_capacity = self.capacity if capacity is None else int(capacity)
        if target_capacity <= 0:
            raise ValueError("capacity must be positive")
        payload = self.to_payload()
        payload["capacity"] = np.asarray(target_capacity, dtype=np.int64)
        kept = _evict_to_capacity(TrajectoryReplayStore.from_payload(payload))
        self.__dict__.update(kept.__dict__)

    def save(
        self,
        path: str | Path,
        *,
        compressed: bool = True,
        temp_dir: str | Path | None = None,
    ) -> NpzSaveStats:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_payload()
        temporary_dir = None if temp_dir is None else Path(temp_dir)
        return save_npz_atomic(
            destination,
            payload,
            compressed=compressed,
            temp_dir=temporary_dir,
        )

    def to_payload(self) -> dict[str, np.ndarray]:
        payload: dict[str, np.ndarray] = {
            "capacity": np.asarray(self.capacity, dtype=np.int64),
            "episode_ids": self.episode_ids,
            "episode_seeds": self.episode_seeds,
            "episode_winners": self.episode_winners,
            "episode_end_reasons": self.episode_end_reasons,
            "territory_scores": self.territory_scores,
            "episode_offsets": self.episode_offsets,
            "timesteps": self.timesteps,
            "players": self.players,
            "actions": self.actions,
            "features": self.features,
            "legal_masks": self.legal_masks,
            "policy_targets": self.policy_targets,
            "winners": self.winners,
            "terminals": self.terminals,
            "root_values": self.root_values,
            "model_versions": self.model_versions,
            "created_iterations": self.created_iterations,
            "sample_weights": self.sample_weights,
            "sampling_priorities": self.sampling_priorities,
            "search_config_hash_table": self.search_config_hash_table,
            "search_config_hash_ids": self.search_config_hash_ids,
        }
        if self.root_policy_logits is not None and self.root_policy_logits_present is not None:
            payload["root_policy_logits"] = self.root_policy_logits
            payload["root_policy_logits_present"] = self.root_policy_logits_present
        if self.next_features is not None and self.next_features_present is not None:
            payload["next_features"] = self.next_features
            payload["next_features_present"] = self.next_features_present
        return payload

    def transitions(self) -> list[TrajectoryTransition]:
        return [transition for episode in self.episodes for transition in episode.transitions]

    def transition_refs(
        self,
        row_indexes: Sequence[int],
    ) -> list[tuple[TrajectoryEpisode, int]]:
        if not row_indexes:
            return []
        search_config_hashes = _decode_search_config_hashes(
            self.search_config_hash_table,
            self.search_config_hash_ids,
        )
        episodes: dict[int, TrajectoryEpisode] = {}
        refs: list[tuple[TrajectoryEpisode, int]] = []
        for row_index in row_indexes:
            if row_index < 0 or row_index >= len(self):
                raise IndexError("row_index is out of range")
            episode_index = int(
                np.searchsorted(self.episode_offsets, row_index, side="right") - 1
            )
            episode = episodes.get(episode_index)
            if episode is None:
                episode = _episode_from_store(
                    self,
                    episode_index,
                    search_config_hashes=search_config_hashes,
                )
                episodes[episode_index] = episode
            transition_index = row_index - int(self.episode_offsets[episode_index])
            refs.append((episode, transition_index))
        return refs


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
        "sampling_priorities": np.ones((len(transitions),), dtype=np.float32),
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
    return _episodes_from_store(TrajectoryReplayStore.from_payload(data))


def _episodes_from_store(store: TrajectoryReplayStore) -> list[TrajectoryEpisode]:
    search_config_hashes = _decode_search_config_hashes(
        store.search_config_hash_table,
        store.search_config_hash_ids,
    )
    return [
        _episode_from_store(
            store,
            episode_index,
            search_config_hashes=search_config_hashes,
        )
        for episode_index in range(len(store.episode_offsets) - 1)
    ]


def _episode_from_store(
    store: TrajectoryReplayStore,
    episode_index: int,
    *,
    search_config_hashes: Sequence[str],
) -> TrajectoryEpisode:
    start = int(store.episode_offsets[episode_index])
    end = int(store.episode_offsets[episode_index + 1])
    transitions = []
    for row in range(start, end):
        winner = int(store.winners[row])
        transitions.append(
            TrajectoryTransition(
                episode_id=int(store.episode_ids[episode_index]),
                timestep=int(store.timesteps[row]),
                player=int(store.players[row]),
                features=store.features[row],
                legal_mask=store.legal_masks[row],
                action=int(store.actions[row]),
                policy_target=store.policy_targets[row],
                root_policy_logits=(
                    store.root_policy_logits[row]
                    if store.root_policy_logits is not None
                    and store.root_policy_logits_present is not None
                    and store.root_policy_logits_present[row]
                    else None
                ),
                root_value=_none_if_nan(float(store.root_values[row])),
                next_features=(
                    store.next_features[row]
                    if store.next_features is not None
                    and store.next_features_present is not None
                    and store.next_features_present[row]
                    else None
                ),
                winner=None if winner < 0 else winner,
                terminal=bool(store.terminals[row]),
                model_version=int(store.model_versions[row]),
                search_config_hash=search_config_hashes[row],
                created_iteration=int(store.created_iterations[row]),
                sample_weight=float(store.sample_weights[row]),
            )
        )
    territory_score_row = store.territory_scores[episode_index]
    return TrajectoryEpisode(
        episode_id=int(store.episode_ids[episode_index]),
        seed=int(store.episode_seeds[episode_index]),
        transitions=tuple(transitions),
        winner=int(store.episode_winners[episode_index]),
        end_reason=int(store.episode_end_reasons[episode_index]),
        territory_scores=(int(territory_score_row[0]), int(territory_score_row[1])),
    )


def _concat_stores(
    left: TrajectoryReplayStore,
    right: TrajectoryReplayStore,
) -> TrajectoryReplayStore:
    return _concat_many_stores(left, (right,))


def _concat_many_stores(
    left: TrajectoryReplayStore,
    rights: Sequence[TrajectoryReplayStore],
) -> TrajectoryReplayStore:
    stores = (left, *rights)
    max_priority = left.current_max_sampling_priority()
    payload: dict[str, np.ndarray] = {
        "capacity": np.asarray(left.capacity, dtype=np.int64),
        "episode_ids": np.concatenate([store.episode_ids for store in stores]),
        "episode_seeds": np.concatenate([store.episode_seeds for store in stores]),
        "episode_winners": np.concatenate([store.episode_winners for store in stores]),
        "episode_end_reasons": np.concatenate(
            [store.episode_end_reasons for store in stores]
        ),
        "territory_scores": np.concatenate([store.territory_scores for store in stores]),
        "episode_offsets": _concat_many_offsets([store.episode_offsets for store in stores]),
        "timesteps": np.concatenate([store.timesteps for store in stores]),
        "players": np.concatenate([store.players for store in stores]),
        "actions": np.concatenate([store.actions for store in stores]),
        "features": np.concatenate([store.features for store in stores]),
        "legal_masks": np.concatenate([store.legal_masks for store in stores]),
        "policy_targets": np.concatenate([store.policy_targets for store in stores]),
        "winners": np.concatenate([store.winners for store in stores]),
        "terminals": np.concatenate([store.terminals for store in stores]),
        "root_values": np.concatenate([store.root_values for store in stores]),
        "model_versions": np.concatenate([store.model_versions for store in stores]),
        "created_iterations": np.concatenate(
            [store.created_iterations for store in stores]
        ),
        "sample_weights": np.concatenate([store.sample_weights for store in stores]),
        "sampling_priorities": np.concatenate(
            [
                left.sampling_priorities,
                *[
                    np.full((len(store),), max_priority, dtype=np.float32)
                    for store in rights
                ],
            ]
        ),
    }
    _add_search_config_hashes(
        payload,
        [
            hash_value
            for store in stores
            for hash_value in _decode_search_config_hashes(
                store.search_config_hash_table,
                store.search_config_hash_ids,
            )
        ],
    )
    _add_optional_concat_many(payload, stores, "root_policy_logits", (ACTION_SPACE,))
    _add_optional_concat_many(payload, stores, "next_features", FEATURE_SHAPE)
    return TrajectoryReplayStore.from_payload(payload)


def _concat_offsets(left_offsets: np.ndarray, right_offsets: np.ndarray) -> np.ndarray:
    if right_offsets.shape == (1,):
        return left_offsets.copy()
    right_tail = right_offsets[1:] + int(left_offsets[-1])
    return np.concatenate([left_offsets, right_tail.astype(np.int64)])


def _concat_many_offsets(offsets: Sequence[np.ndarray]) -> np.ndarray:
    if not offsets:
        return np.asarray([0], dtype=np.int64)
    combined = offsets[0].astype(np.int64, copy=True)
    transition_count = int(combined[-1])
    for offset in offsets[1:]:
        if offset.shape == (1,):
            continue
        tail = offset[1:].astype(np.int64, copy=False) + transition_count
        combined = np.concatenate([combined, tail])
        transition_count = int(combined[-1])
    return combined


def _add_optional_concat(
    payload: dict[str, np.ndarray],
    left: TrajectoryReplayStore,
    right: TrajectoryReplayStore,
    key: str,
    shape: tuple[int, ...],
) -> None:
    _add_optional_concat_many(payload, (left, right), key, shape)


def _add_optional_concat_many(
    payload: dict[str, np.ndarray],
    stores: Sequence[TrajectoryReplayStore],
    key: str,
    shape: tuple[int, ...],
) -> None:
    total_rows = sum(len(store) for store in stores)
    if not any(getattr(store, key) is not None for store in stores):
        return
    values = np.full((total_rows, *shape), np.nan, dtype=np.float32)
    present = np.zeros((total_rows,), dtype=np.bool_)
    offset = 0
    for store in stores:
        store_rows = len(store)
        store_values = getattr(store, key)
        store_present = getattr(store, f"{key}_present")
        if store_values is not None and store_present is not None:
            values[offset : offset + store_rows] = store_values
            present[offset : offset + store_rows] = store_present
        offset += store_rows
    payload[key] = values
    payload[f"{key}_present"] = present


def _evict_to_capacity(store: TrajectoryReplayStore) -> TrajectoryReplayStore:
    if len(store) <= store.capacity:
        return store
    drop_episodes = 0
    while (
        drop_episodes + 1 < store.episode_offsets.shape[0]
        and len(store) - int(store.episode_offsets[drop_episodes]) > store.capacity
    ):
        drop_episodes += 1
    if drop_episodes == 0:
        return store
    transition_start = int(store.episode_offsets[drop_episodes])
    payload = store.to_payload()
    for key in (
        "episode_ids",
        "episode_seeds",
        "episode_winners",
        "episode_end_reasons",
        "territory_scores",
    ):
        payload[key] = payload[key][drop_episodes:]
    payload["episode_offsets"] = payload["episode_offsets"][drop_episodes:] - transition_start
    for key in (
        "timesteps",
        "players",
        "actions",
        "features",
        "legal_masks",
        "policy_targets",
        "winners",
        "terminals",
        "root_values",
        "model_versions",
        "created_iterations",
        "sample_weights",
        "sampling_priorities",
        "search_config_hash_ids",
        "root_policy_logits",
        "root_policy_logits_present",
        "next_features",
        "next_features_present",
    ):
        if key in payload:
            payload[key] = payload[key][transition_start:]
    return TrajectoryReplayStore.from_payload(payload)


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
    table, ids = _load_search_config_hash_encoding(data, transition_count)
    return _decode_search_config_hashes(table, ids)


def _load_sampling_priorities(data: Any, transition_count: int) -> np.ndarray:
    if "sampling_priorities" not in data:
        return np.ones((transition_count,), dtype=np.float32)
    priorities = np.asarray(data["sampling_priorities"], dtype=np.float32)
    if priorities.shape != (transition_count,):
        raise ValueError("trajectory replay sampling_priorities length mismatch")
    return priorities


def _load_search_config_hash_encoding(
    data: Any,
    transition_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    if "search_config_hash_ids" in data and "search_config_hash_table" in data:
        encoded_table = np.asarray(data["search_config_hash_table"], dtype=np.str_)
        ids = np.asarray(data["search_config_hash_ids"], dtype=np.int64)
        if ids.shape != (transition_count,):
            raise ValueError("trajectory replay search_config_hash_ids length mismatch")
        if np.any(ids < 0) or np.any(ids >= len(encoded_table)):
            raise ValueError("trajectory replay search_config_hash_ids contain invalid indexes")
        return encoded_table, ids
    hashes = np.asarray(data["search_config_hashes"], dtype=np.str_)
    hash_table: list[str] = []
    indexes: dict[str, int] = {}
    ids = np.empty((transition_count,), dtype=np.int64)
    for row, value in enumerate(hashes):
        key = str(value)
        index = indexes.get(key)
        if index is None:
            index = len(hash_table)
            indexes[key] = index
            hash_table.append(key)
        ids[row] = index
    return np.asarray(hash_table, dtype=np.str_), ids


def _decode_search_config_hashes(table: np.ndarray, ids: np.ndarray) -> list[str]:
    return [str(table[index]) for index in ids]


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
) -> tuple[np.ndarray | None, np.ndarray]:
    if key not in data:
        return None, np.zeros((shape[0],), dtype=np.bool_)
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
    if (
        "sampling_priorities" in data
        and np.asarray(data["sampling_priorities"]).shape[0] != transition_count
    ):
        raise ValueError("trajectory replay sampling_priorities length mismatch")
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
