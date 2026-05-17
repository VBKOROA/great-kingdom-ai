"""Public bootstrap target services for trajectory reanalyze."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from great_kingdom_ai.replay import TrajectoryEpisode, TrajectoryReplayStore
from great_kingdom_ai.search_reanalyze import (
    SearchReanalyzeConfig,
    refresh_sampled_policies_with_search,
)
from great_kingdom_ai.self_play_data import value_target_for_player


def bootstrap_targets_from_refreshed_values(
    episodes: Sequence[TrajectoryEpisode],
    refreshed_values: np.ndarray,
    *,
    td_steps: int,
    gamma: float,
    model_version: int = 0,
    dynamic_horizon_enabled: bool = False,
    dynamic_horizon_tau: float = 0.3,
    dynamic_horizon_total_steps: int | None = None,
) -> np.ndarray:
    targets = np.empty((refreshed_values.shape[0],), dtype=np.float32)
    row_offset = 0
    for episode in episodes:
        terminal_index = len(episode.transitions) - 1
        for index, transition in enumerate(episode.transitions):
            row = row_offset + index
            effective_td_steps = effective_bootstrap_td_steps(
                td_steps=td_steps,
                model_version=model_version,
                created_iteration=transition.created_iteration,
                dynamic_horizon_enabled=dynamic_horizon_enabled,
                dynamic_horizon_tau=dynamic_horizon_tau,
                dynamic_horizon_total_steps=dynamic_horizon_total_steps,
            )
            target_index = index + effective_td_steps
            if (
                effective_td_steps == 0
                or transition.terminal
                or target_index >= terminal_index
            ):
                targets[row] = value_target_for_player(
                    player=transition.player,
                    winner=episode.winner,
                )
                continue

            bootstrap = float(refreshed_values[row_offset + target_index])
            bootstrap_transition = episode.transitions[target_index]
            if bootstrap_transition.player != transition.player:
                bootstrap = -bootstrap
            targets[row] = np.float32((gamma**effective_td_steps) * bootstrap)
        row_offset += len(episode.transitions)
    return targets


def mcts_root_bootstrap_values_from_store(
    replay: TrajectoryReplayStore,
    *,
    policy_logits: np.ndarray,
    refreshed_values: np.ndarray,
    model: Any,
    device: str,
    onnx_evaluator: Any | None,
    config: SearchReanalyzeConfig,
) -> np.ndarray:
    search_result = refresh_sampled_policies_with_search(
        transitions=replay.transition_refs(list(range(len(replay)))),
        policies=np.ascontiguousarray(replay.policy_targets, dtype=np.float32),
        policy_logits=policy_logits,
        refreshed_values=refreshed_values,
        model=model,
        device=device,
        onnx_evaluator=onnx_evaluator,
        config=config,
    )
    if search_result.root_values is None or not np.isfinite(search_result.root_values).all():
        raise RuntimeError("MCTS root bootstrap requires root values from search results")
    return np.ascontiguousarray(search_result.root_values, dtype=np.float32)


def bootstrap_targets_from_store(
    replay: TrajectoryReplayStore,
    refreshed_values: np.ndarray,
    *,
    td_steps: int,
    gamma: float,
    model_version: int = 0,
    dynamic_horizon_enabled: bool = False,
    dynamic_horizon_tau: float = 0.3,
    dynamic_horizon_total_steps: int | None = None,
) -> np.ndarray:
    targets = np.empty((refreshed_values.shape[0],), dtype=np.float32)
    for episode_index in range(replay.episode_count):
        start = int(replay.episode_offsets[episode_index])
        end = int(replay.episode_offsets[episode_index + 1])
        terminal_index = end - 1
        winner = int(replay.episode_winners[episode_index])
        for row in range(start, end):
            effective_td_steps = effective_bootstrap_td_steps(
                td_steps=td_steps,
                model_version=model_version,
                created_iteration=int(replay.created_iterations[row]),
                dynamic_horizon_enabled=dynamic_horizon_enabled,
                dynamic_horizon_tau=dynamic_horizon_tau,
                dynamic_horizon_total_steps=dynamic_horizon_total_steps,
            )
            target_index = row + effective_td_steps
            if (
                effective_td_steps == 0
                or bool(replay.terminals[row])
                or target_index >= terminal_index
            ):
                targets[row] = value_target_for_player(
                    player=int(replay.players[row]),
                    winner=winner,
                )
                continue

            bootstrap = float(refreshed_values[target_index])
            if int(replay.players[target_index]) != int(replay.players[row]):
                bootstrap = -bootstrap
            targets[row] = np.float32((gamma**effective_td_steps) * bootstrap)
    return targets


def effective_bootstrap_td_steps(
    *,
    td_steps: int,
    model_version: int,
    created_iteration: int,
    dynamic_horizon_enabled: bool,
    dynamic_horizon_tau: float,
    dynamic_horizon_total_steps: int | None,
) -> int:
    if td_steps <= 0 or not dynamic_horizon_enabled:
        return td_steps
    if dynamic_horizon_total_steps is None:
        raise ValueError("dynamic_horizon_total_steps is required")
    age = max(model_version - created_iteration, 0)
    shrink = math.floor(age / (dynamic_horizon_tau * dynamic_horizon_total_steps))
    return min(td_steps, max(1, td_steps - shrink))


__all__ = [
    "bootstrap_targets_from_refreshed_values",
    "bootstrap_targets_from_store",
    "effective_bootstrap_td_steps",
    "mcts_root_bootstrap_values_from_store",
]
