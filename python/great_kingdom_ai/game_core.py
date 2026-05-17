"""Shared Python boundary helpers for the Rust rules/search core."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol, cast

import numpy as np

from great_kingdom_ai.features import ACTION_SPACE, BOARD_SIZE, FEATURE_CHANNELS


class SelfPlayState(Protocol):
    def current_player(self) -> int: ...

    def legal_actions(self) -> list[int]: ...

    def apply_action(self, action_index: int) -> int | None: ...

    def is_terminal(self) -> bool: ...

    def winner(self) -> int | None: ...

    def end_reason(self) -> int | None: ...

    def territory_scores(self) -> tuple[int, int]: ...

    def feature_planes(self) -> list[float]: ...

    def legal_mask(self) -> list[bool]: ...


class SearchResultLike(Protocol):
    def selected_action(self) -> int | None: ...

    def visit_counts(self) -> list[int]: ...


def import_core(context: str) -> Any:
    try:
        import great_kingdom_core as core  # type: ignore[import-untyped]
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"great_kingdom_core is not installed. Build it with maturin before {context}."
        ) from exc
    return core


def create_core_game_state() -> SelfPlayState:
    return cast(SelfPlayState, import_core("self-play").GameState())


def create_core_gumbel_search(config: Any, *, seed_offset: int = 0) -> Any:
    core = import_core("Gumbel search")
    return core.GumbelSearch(
        simulations=config.gumbel_simulations,
        max_considered_actions=config.gumbel_max_considered_actions,
        c_visit=config.gumbel_c_visit,
        c_scale=config.gumbel_c_scale,
        seed=config.gumbel_seed + seed_offset,
        gumbel_scale=config.gumbel_scale,
        policy_target_temperature=config.policy_target_temperature,
        policy_target_c_visit=config.policy_target_c_visit,
        policy_target_c_scale=config.policy_target_c_scale,
    )


def create_core_self_play_batch(
    config: Any,
    *,
    game_count: int,
    seed_offset: int = 0,
) -> Any:
    core = import_core("self-play")
    return core.GumbelSelfPlayBatch(
        game_count=game_count,
        simulations=config.gumbel_simulations,
        max_considered_actions=config.gumbel_max_considered_actions,
        c_visit=config.gumbel_c_visit,
        c_scale=config.gumbel_c_scale,
        seed=config.gumbel_seed + seed_offset,
        gumbel_scale=config.gumbel_scale,
        policy_target_temperature=config.policy_target_temperature,
        policy_target_c_visit=config.policy_target_c_visit,
        policy_target_c_scale=config.policy_target_c_scale,
    )


def can_create_core_self_play_batch() -> bool:
    try:
        import great_kingdom_core as core
    except ModuleNotFoundError:
        return False
    return hasattr(core, "GumbelSelfPlayBatch")


def create_core_arena_batch(
    config: Any,
    *,
    game_count: int,
    seed_start: int,
    game_index_start: int = 0,
) -> Any:
    core = import_core("arena evaluation")
    if not hasattr(core, "GumbelArenaBatch"):
        raise RuntimeError(
            "great_kingdom_core.GumbelArenaBatch is not available. "
            "Rebuild the Rust extension before batched arena evaluation."
        )
    return core.GumbelArenaBatch(
        game_count=game_count,
        seed_start=seed_start,
        game_index_start=game_index_start,
        simulations=config.gumbel_simulations,
        max_considered_actions=config.gumbel_max_considered_actions,
        c_visit=config.gumbel_c_visit,
        c_scale=config.gumbel_c_scale,
        seed=config.gumbel_seed,
        gumbel_scale=config.gumbel_scale,
        policy_target_temperature=config.policy_target_temperature,
        policy_target_c_visit=config.policy_target_c_visit,
        policy_target_c_scale=config.policy_target_c_scale,
        paired_seeds=config.paired_seeds,
    )


def flat_features_for_replay(feature_planes: Sequence[float]) -> np.ndarray:
    features = np.asarray(feature_planes, dtype=np.float32)
    expected = FEATURE_CHANNELS * BOARD_SIZE * BOARD_SIZE
    if features.shape != (expected,):
        raise ValueError(f"expected flat feature shape {(expected,)}, got {features.shape}")
    return cast(np.ndarray, features.reshape(FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE))


def as_int_list(values: Sequence[int] | bytes) -> list[int]:
    return [int(value) for value in values]


def request_feature_rows_and_masks(request: Any) -> tuple[Any, Any, int]:
    if (
        hasattr(request, "len")
        and hasattr(request, "feature_plane_bytes")
        and hasattr(request, "legal_mask_bytes")
    ):
        row_count = int(request.len())
        feature_rows = np.frombuffer(request.feature_plane_bytes(), dtype=np.float32).reshape(
            row_count,
            FEATURE_CHANNELS,
            BOARD_SIZE,
            BOARD_SIZE,
        )
        legal_masks = np.frombuffer(request.legal_mask_bytes(), dtype=np.bool_).reshape(
            row_count,
            ACTION_SPACE,
        )
        return feature_rows, legal_masks, row_count

    feature_rows = request.feature_planes()
    legal_masks = request.legal_masks()
    return feature_rows, legal_masks, len(feature_rows)


def active_legal_masks(batch: Any) -> list[list[bool]]:
    if hasattr(batch, "active_legal_masks"):
        return [[bool(value) for value in row] for row in batch.active_legal_masks()]
    request = batch.active_eval_request()
    _feature_rows, masks, _row_count = request_feature_rows_and_masks(request)
    return [[bool(value) for value in row] for row in masks]


def request_game_indexes(request: Any, *, expected_len: int) -> list[int]:
    if not hasattr(request, "game_indexes"):
        raise RuntimeError("arena eval request did not include game index metadata")
    game_indexes = as_int_list(request.game_indexes())
    if len(game_indexes) != expected_len:
        raise RuntimeError("arena eval request did not include game index metadata")
    return game_indexes


def request_current_players(request: Any, *, expected_len: int) -> list[int]:
    if not hasattr(request, "current_players"):
        raise RuntimeError("arena eval request did not include current player metadata")
    players = as_int_list(request.current_players())
    if len(players) != expected_len:
        raise RuntimeError("arena eval request current player metadata length mismatch")
    return players


def request_states(request: Any) -> list[Any]:
    class RequestState:
        def __init__(self, features: list[float], mask: list[bool], player: int | None) -> None:
            self._features = features
            self._mask = mask
            self._player = player

        def feature_planes(self) -> list[float]:
            return self._features

        def legal_mask(self) -> list[bool]:
            return self._mask

        def current_player(self) -> int:
            if self._player is None:
                raise RuntimeError("eval request did not include current player metadata")
            return self._player

    features = request.feature_planes()
    players = (
        [int(player) for player in request.current_players()]
        if hasattr(request, "current_players")
        else [None] * len(features)
    )
    return [
        RequestState(feature_planes, mask, player)
        for feature_planes, mask, player in zip(
            features,
            request.legal_masks(),
            players,
            strict=True,
        )
    ]


def evaluate_core_batch_priors(
    prior_provider: Callable[[Sequence[SelfPlayState]], Sequence[Sequence[float]]],
    request: Any,
) -> list[list[float]]:
    states = request_states(request)
    return [
        [float(value) for value in row]
        for row in prior_provider(cast(Sequence[SelfPlayState], states))
    ]


def evaluate_core_batch_policy_values(
    evaluator_provider: Callable[
        [Sequence[SelfPlayState]],
        tuple[Sequence[Sequence[float]], Sequence[float]],
    ],
    request: Any,
) -> tuple[list[list[float]], list[float]]:
    states = request_states(request)
    policies, values = evaluator_provider(cast(Sequence[SelfPlayState], states))
    return (
        [[float(value) for value in row] for row in policies],
        [float(value) for value in values],
    )


def deterministic_action(
    result: SearchResultLike,
    scores: Sequence[float],
    legal_actions: Sequence[int],
) -> int:
    selected = result.selected_action()
    legal_set = set(legal_actions)
    if selected in legal_set:
        return int(selected)

    visits = result.visit_counts()
    if len(visits) != len(scores):
        raise ValueError("visit count and score lengths must match")
    if not legal_actions:
        raise ValueError("state has no legal actions")
    return max(legal_actions, key=lambda action: (visits[action], scores[action], -action))
