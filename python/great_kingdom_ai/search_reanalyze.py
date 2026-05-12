"""Rust-search policy target refresh for trajectory reanalyze snapshots."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

import numpy as np

from great_kingdom_ai.evaluator import evaluate_feature_batch_logits_values
from great_kingdom_ai.features import ACTION_SPACE, BOARD_SIZE, FEATURE_CHANNELS
from great_kingdom_ai.priority_sampling import PrioritySamplingConfig, priority_scores
from great_kingdom_ai.trajectory_replay import TrajectoryEpisode, TrajectoryTransition

SearchReanalyzeProgressCallback = Callable[[str, int, int, str], None]


class SearchReanalyzeModel(Protocol):
    def eval(self) -> Any: ...


@dataclass(frozen=True)
class SearchReanalyzeConfig:
    fraction: float = 0.0
    budget: int | None = None
    simulations: int = 32
    max_considered_actions: int = 16
    c_visit: float = 50.0
    c_scale: float = 1.0
    policy_target_c_visit: float = 5.0
    policy_target_c_scale: float = 0.25
    policy_target_temperature: float = 1.0
    leaf_batch_size: int = 8
    root_batch_size: int = 128
    seed: int = 0
    value_error_weight: float = 1.0
    policy_kl_weight: float = 1.0
    target_age_weight: float = 0.25
    opening_weight: float = 0.25
    opening_max_timestep: int = 40
    max_priority: float | None = 64.0

    @property
    def enabled(self) -> bool:
        return self.fraction > 0.0 or (self.budget is not None and self.budget > 0)

    def __post_init__(self) -> None:
        if not math.isfinite(self.fraction) or not 0.0 <= self.fraction <= 1.0:
            raise ValueError("search reanalyze fraction must be finite and in [0, 1]")
        if self.budget is not None and self.budget < 0:
            raise ValueError("search reanalyze budget must be non-negative")
        if self.simulations <= 0:
            raise ValueError("search reanalyze simulations must be positive")
        if self.max_considered_actions <= 0:
            raise ValueError("search reanalyze max_considered_actions must be positive")
        for label, value in (
            ("c_visit", self.c_visit),
            ("c_scale", self.c_scale),
            ("policy_target_c_visit", self.policy_target_c_visit),
            ("policy_target_c_scale", self.policy_target_c_scale),
            ("policy_target_temperature", self.policy_target_temperature),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"search reanalyze {label} must be finite and positive")
        if self.leaf_batch_size <= 0:
            raise ValueError("search reanalyze leaf_batch_size must be positive")
        if self.root_batch_size <= 0:
            raise ValueError("search reanalyze root_batch_size must be positive")
        for label, value in (
            ("value_error_weight", self.value_error_weight),
            ("policy_kl_weight", self.policy_kl_weight),
            ("target_age_weight", self.target_age_weight),
            ("opening_weight", self.opening_weight),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"search reanalyze {label} must be finite and non-negative")
        if self.opening_max_timestep < 0:
            raise ValueError("search reanalyze opening_max_timestep must be non-negative")
        if self.max_priority is not None:
            if not math.isfinite(self.max_priority) or self.max_priority <= 1.0:
                raise ValueError("search reanalyze max_priority must be greater than 1")


@dataclass(frozen=True)
class SearchReanalyzeResult:
    policies: np.ndarray
    search_reanalyzed: np.ndarray
    selected_indexes: tuple[int, ...]


@dataclass(frozen=True)
class _TransitionRef:
    episode: TrajectoryEpisode
    transition_index: int
    row_index: int


def refresh_policies_with_search(
    *,
    episodes: Sequence[TrajectoryEpisode],
    policies: np.ndarray,
    values: np.ndarray,
    policy_logits: np.ndarray,
    refreshed_values: np.ndarray,
    target_ages: np.ndarray,
    model: SearchReanalyzeModel,
    device: str,
    config: SearchReanalyzeConfig,
    progress_callback: SearchReanalyzeProgressCallback | None = None,
) -> SearchReanalyzeResult:
    policies = np.asarray(policies, dtype=np.float32)
    _report_progress(
        progress_callback,
        "search-select",
        0,
        1,
        f"scoring rows={policies.shape[0]}",
    )
    selected_indexes = select_search_reanalyze_indexes(
        episodes=episodes,
        policies=policies,
        values=values,
        policy_logits=policy_logits,
        refreshed_values=refreshed_values,
        target_ages=target_ages,
        config=config,
    )
    _report_progress(
        progress_callback,
        "search-select",
        1,
        1,
        f"selected={len(selected_indexes)}",
    )
    search_reanalyzed = np.zeros((policies.shape[0],), dtype=np.bool_)
    if not selected_indexes:
        return SearchReanalyzeResult(
            policies=policies.copy(),
            search_reanalyzed=search_reanalyzed,
            selected_indexes=(),
        )

    core = _import_core()
    refs = {ref.row_index: ref for ref in _transition_refs(episodes)}
    refreshed = policies.copy()
    evaluator = _leaf_evaluator(model, device)
    total_chunks = math.ceil(len(selected_indexes) / config.root_batch_size)
    _report_progress(
        progress_callback,
        "search",
        0,
        total_chunks,
        (
            f"selected={len(selected_indexes)}, sims={config.simulations}, "
            f"root_batch={config.root_batch_size}, leaf_batch={config.leaf_batch_size}"
        ),
    )
    for start in range(0, len(selected_indexes), config.root_batch_size):
        chunk_number = start // config.root_batch_size + 1
        chunk_indexes = selected_indexes[start : start + config.root_batch_size]
        _report_progress(
            progress_callback,
            "search",
            chunk_number - 1,
            total_chunks,
            f"chunk={chunk_number}/{total_chunks}, rows={len(chunk_indexes)}",
        )
        chunk_refs = [refs[row_index] for row_index in chunk_indexes]
        batch = _reconstruct_batch(core, chunk_refs, config=config)
        _validate_reconstructed_batch(batch, chunk_refs)
        results = batch.search_active_with_logits_and_evaluator(
            [
                policy_logits[row_index].astype(np.float32).tolist()
                for row_index in chunk_indexes
            ],
            evaluator,
            [float(refreshed_values[row_index]) for row_index in chunk_indexes],
            config.leaf_batch_size,
        )
        if len(results) != len(chunk_indexes):
            raise ValueError("batch search result length does not match selected rows")
        for row_index, result in zip(chunk_indexes, results, strict=True):
            if result is None:
                raise ValueError("batch search did not return a result for selected row")
            refreshed[row_index] = _policy_target_from_result(result)
            search_reanalyzed[row_index] = True
        _report_progress(
            progress_callback,
            "search",
            chunk_number,
            total_chunks,
            f"chunk={chunk_number}/{total_chunks}, refreshed={np.count_nonzero(search_reanalyzed)}",
        )

    return SearchReanalyzeResult(
        policies=refreshed,
        search_reanalyzed=search_reanalyzed,
        selected_indexes=selected_indexes,
    )


def select_search_reanalyze_indexes(
    *,
    episodes: Sequence[TrajectoryEpisode],
    policies: np.ndarray,
    values: np.ndarray,
    policy_logits: np.ndarray,
    refreshed_values: np.ndarray,
    target_ages: np.ndarray,
    config: SearchReanalyzeConfig,
) -> tuple[int, ...]:
    row_count = int(np.asarray(policies).shape[0])
    count = _search_budget(row_count, config)
    if count == 0:
        return ()
    refs = _transition_refs(episodes)
    if len(refs) != row_count:
        raise ValueError("episode transition count must match policy row count")
    scores = priority_scores(
        values=np.asarray(values, dtype=np.float32),
        value_predictions=np.asarray(refreshed_values, dtype=np.float32),
        policies=np.asarray(policies, dtype=np.float32),
        policy_logits=np.asarray(policy_logits, dtype=np.float32),
        legal_masks=np.stack(
            [ref.episode.transitions[ref.transition_index].legal_mask for ref in refs],
            axis=0,
        ).astype(np.bool_),
        target_ages=np.asarray(target_ages, dtype=np.int64),
        config=PrioritySamplingConfig(
            enabled=True,
            value_error_weight=config.value_error_weight,
            policy_kl_weight=config.policy_kl_weight,
            target_age_weight=config.target_age_weight,
            max_priority=config.max_priority,
        ),
    )
    if config.opening_weight > 0.0:
        timesteps = np.asarray(
            [ref.episode.transitions[ref.transition_index].timestep for ref in refs],
            dtype=np.int64,
        )
        opening = (timesteps <= config.opening_max_timestep).astype(np.float32)
        scores = scores + np.float32(config.opening_weight) * opening
    order = np.lexsort((np.arange(row_count), -scores))
    return tuple(sorted(int(index) for index in order[:count]))


def _search_budget(row_count: int, config: SearchReanalyzeConfig) -> int:
    if row_count <= 0:
        return 0
    fraction_count = int(math.ceil(row_count * config.fraction)) if config.fraction > 0.0 else 0
    if config.budget is None:
        count = fraction_count
    elif fraction_count == 0:
        count = config.budget
    else:
        count = min(fraction_count, config.budget)
    return min(row_count, max(0, count))


def _transition_refs(episodes: Sequence[TrajectoryEpisode]) -> list[_TransitionRef]:
    refs: list[_TransitionRef] = []
    row_index = 0
    for episode in episodes:
        for transition_index in range(len(episode.transitions)):
            refs.append(_TransitionRef(episode, transition_index, row_index))
            row_index += 1
    return refs


def _reconstruct_state(core: Any, episode: TrajectoryEpisode, transition_index: int) -> Any:
    state = core.GameState()
    for prior in episode.transitions[:transition_index]:
        if state.is_terminal():
            raise ValueError("cannot reconstruct a transition after terminal state")
        state.apply_action(int(prior.action))
    return state


def _reconstruct_batch(
    core: Any,
    refs: Sequence[_TransitionRef],
    *,
    config: SearchReanalyzeConfig,
) -> Any:
    batch = core.GumbelSelfPlayBatch(
        game_count=len(refs),
        simulations=config.simulations,
        max_considered_actions=config.max_considered_actions,
        c_visit=config.c_visit,
        c_scale=config.c_scale,
        seed=config.seed + min(ref.row_index for ref in refs),
        policy_target_temperature=config.policy_target_temperature,
        policy_target_c_visit=config.policy_target_c_visit,
        policy_target_c_scale=config.policy_target_c_scale,
    )
    max_depth = max(ref.transition_index for ref in refs)
    for depth in range(max_depth):
        batch.apply_actions(
            [
                int(ref.episode.transitions[depth].action)
                if depth < ref.transition_index
                else None
                for ref in refs
            ]
        )
    return batch


def _validate_reconstructed_batch(batch: Any, refs: Sequence[_TransitionRef]) -> None:
    active_indexes = list(batch.active_game_indexes())
    if active_indexes != list(range(len(refs))):
        raise ValueError("reanalyze batch reconstructed terminal or inactive states")
    request = batch.active_eval_request()
    feature_rows = request.feature_planes()
    if len(feature_rows) != len(refs):
        raise ValueError("reanalyze batch feature count does not match selected rows")
    expected = FEATURE_CHANNELS * BOARD_SIZE * BOARD_SIZE
    for ref, feature_row in zip(refs, feature_rows, strict=True):
        features = np.asarray(feature_row, dtype=np.float32)
        if features.shape != (expected,):
            raise ValueError(
                f"expected reconstructed feature shape {(expected,)}, got {features.shape}"
            )
        transition = ref.episode.transitions[ref.transition_index]
        features = features.reshape(FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE)
        if not np.allclose(features, transition.features, atol=1e-6):
            raise ValueError("reconstructed state features do not match trajectory transition")


def _validate_reconstructed_state(state: Any, transition: TrajectoryTransition) -> None:
    if int(state.current_player()) != int(transition.player):
        raise ValueError("reconstructed state player does not match trajectory transition")
    features = np.asarray(state.feature_planes(), dtype=np.float32)
    expected = FEATURE_CHANNELS * BOARD_SIZE * BOARD_SIZE
    if features.shape != (expected,):
        raise ValueError(
            f"expected reconstructed feature shape {(expected,)}, got {features.shape}"
        )
    features = features.reshape(FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    if not np.allclose(features, transition.features, atol=1e-6):
        raise ValueError("reconstructed state features do not match trajectory transition")


def _leaf_evaluator(model: SearchReanalyzeModel, device: str) -> Any:
    def evaluate(request: Any) -> tuple[list[list[float]], list[float]]:
        evaluation = evaluate_feature_batch_logits_values(
            cast(Any, model),
            request.feature_planes(),
            request.legal_masks(),
            device=device,
        )
        return evaluation.policy_logits.astype(np.float32).tolist(), [
            float(value) for value in evaluation.value
        ]

    return evaluate


def _create_search(core: Any, config: SearchReanalyzeConfig, *, seed_offset: int) -> Any:
    return core.GumbelSearch(
        simulations=config.simulations,
        max_considered_actions=config.max_considered_actions,
        c_visit=config.c_visit,
        c_scale=config.c_scale,
        seed=config.seed + seed_offset,
        policy_target_temperature=config.policy_target_temperature,
        policy_target_c_visit=config.policy_target_c_visit,
        policy_target_c_scale=config.policy_target_c_scale,
    )


def _policy_target_from_result(result: Any) -> np.ndarray:
    policy = np.asarray(result.policy_target(), dtype=np.float32)
    if policy.shape != (ACTION_SPACE,):
        raise ValueError(f"expected search policy target shape {(ACTION_SPACE,)}")
    if np.any(policy < 0.0) or not np.isclose(policy.sum(), 1.0):
        raise ValueError("search policy target must be normalized and non-negative")
    return policy


def _report_progress(
    progress_callback: SearchReanalyzeProgressCallback | None,
    stage: str,
    current: int,
    total: int,
    detail: str,
) -> None:
    if progress_callback is not None:
        progress_callback(stage, current, total, detail)


def _import_core() -> Any:
    try:
        import great_kingdom_core as core  # type: ignore[import-untyped]
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "great_kingdom_core is not installed. Build it with maturin before search reanalyze."
        ) from exc
    return core


__all__ = [
    "SearchReanalyzeConfig",
    "SearchReanalyzeProgressCallback",
    "SearchReanalyzeResult",
    "refresh_policies_with_search",
    "select_search_reanalyze_indexes",
]
