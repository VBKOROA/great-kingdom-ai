"""Python wrapper for Rust ONNX self-play generation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from great_kingdom_ai.features import ACTION_SPACE, BOARD_SIZE, FEATURE_CHANNELS
from great_kingdom_ai.replay import (
    TrajectoryEpisode,
    TrajectoryTransition,
    legal_mask_from_features,
)
from great_kingdom_ai.replay_buffer import ReplaySample
from great_kingdom_ai.self_play import GameLog, MoveLog, SelfPlayConfig
from great_kingdom_ai.self_play_data import value_target_for_player


@dataclass(frozen=True)
class RustOnnxSelfPlayConfig:
    onnx_model_path: Path
    output_dir: Path
    games: int = 2
    seed_start: int = 0
    model_version: int = 0
    created_iteration: int = 0
    onnx_device: str = "cpu"
    onnx_max_batch_size: int = 128
    rust_self_play_batch_size: int = 2
    self_play: SelfPlayConfig = SelfPlayConfig()


@dataclass(frozen=True)
class RustSelfPlayRunSummary:
    artifact_dir: Path
    games: int
    samples: int
    onnx_model_path: Path
    onnx_device: str
    replay_samples: tuple[ReplaySample, ...] = ()
    game_logs: tuple[GameLog, ...] = ()
    trajectory_episodes: tuple[TrajectoryEpisode, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_dir": str(self.artifact_dir),
            "games": self.games,
            "samples": self.samples,
            "onnx_model_path": str(self.onnx_model_path),
            "onnx_device": self.onnx_device,
        }


def run_rust_onnx_self_play(config: RustOnnxSelfPlayConfig) -> RustSelfPlayRunSummary:
    core = _import_core()
    evaluator = core.OnnxEvaluator(
        str(config.onnx_model_path),
        device=config.onnx_device,
        max_batch_size=config.onnx_max_batch_size,
    )
    logs: list[GameLog] = []
    samples: list[ReplaySample] = []
    episodes: list[TrajectoryEpisode] = []
    remaining = config.games
    seed = config.seed_start
    while remaining > 0:
        batch_size = min(config.rust_self_play_batch_size, remaining)
        batch_logs, batch_samples, batch_episodes = _run_one_batch(
            config,
            core=core,
            evaluator=evaluator,
            seed_start=seed,
            game_count=batch_size,
        )
        logs.extend(batch_logs)
        samples.extend(batch_samples)
        episodes.extend(batch_episodes)
        seed += batch_size
        remaining -= batch_size

    return RustSelfPlayRunSummary(
        artifact_dir=config.output_dir,
        games=len(logs),
        samples=len(samples),
        onnx_model_path=config.onnx_model_path,
        onnx_device=config.onnx_device,
        replay_samples=tuple(samples),
        game_logs=tuple(logs),
        trajectory_episodes=tuple(episodes),
    )


def _run_one_batch(
    config: RustOnnxSelfPlayConfig,
    *,
    core: Any,
    evaluator: Any,
    seed_start: int,
    game_count: int,
) -> tuple[list[GameLog], list[ReplaySample], list[TrajectoryEpisode]]:
    batch = core.GumbelSelfPlayBatch(
        game_count=game_count,
        simulations=config.self_play.gumbel_simulations,
        max_considered_actions=config.self_play.gumbel_max_considered_actions,
        c_visit=config.self_play.gumbel_c_visit,
        c_scale=config.self_play.gumbel_c_scale,
        seed=config.self_play.gumbel_seed + seed_start,
        gumbel_scale=config.self_play.gumbel_scale,
        policy_target_temperature=config.self_play.policy_target_temperature,
        policy_target_c_visit=config.self_play.policy_target_c_visit,
        policy_target_c_scale=config.self_play.policy_target_c_scale,
    )
    seeds = list(range(seed_start, seed_start + game_count))
    rngs = [random.Random(seed) for seed in seeds]
    moves: list[list[MoveLog]] = [[] for _ in seeds]
    pending: list[list[tuple[int, np.ndarray, np.ndarray, np.ndarray | None]]] = [
        [] for _ in seeds
    ]
    trajectory_rows: list[list[tuple[int, int, np.ndarray, int, np.ndarray, np.ndarray | None]]] = [
        [] for _ in seeds
    ]

    for turn in range(config.self_play.max_turns):
        active_indexes = [int(index) for index in batch.active_game_indexes()]
        if not active_indexes:
            break
        request = batch.active_eval_request()
        feature_rows = request.feature_planes()
        players = [int(player) for player in batch.current_players()]
        features_by_game = {
            game_index: _flat_features_for_replay(feature_row)
            for game_index, feature_row in zip(active_indexes, feature_rows, strict=True)
        }
        use_full_by_game: dict[int, bool] = {}
        simulation_budgets: list[int | None] = [None] * game_count
        for game_index in active_indexes:
            use_full = _use_full_search_turn(rngs[game_index], config.self_play)
            use_full_by_game[game_index] = use_full
            if config.self_play.playout_cap_randomization:
                simulation_budgets[game_index] = (
                    config.self_play.playout_cap_full_simulations
                    if use_full
                    else config.self_play.playout_cap_fast_simulations
                )
        if config.self_play.playout_cap_randomization:
            batch.set_simulations(simulation_budgets)
            batch.set_max_considered_actions(
                [
                    (
                        _playout_cap_max_considered_actions(
                            use_full_by_game[game_index],
                            config.self_play,
                        )
                        if game_index in use_full_by_game
                        else None
                    )
                    for game_index in range(game_count)
                ]
            )

        results, root_policy_logits_rows = _search_active_with_root_policy_logits(
            batch,
            evaluator=evaluator,
            request=request,
            leaf_batch_size=config.self_play.leaf_batch_size,
        )
        root_policy_logits_by_game = {
            game_index: np.asarray(row, dtype=np.float32)
            for game_index, row in zip(
                active_indexes,
                root_policy_logits_rows,
                strict=True,
            )
        }
        actions: list[int | None] = [None] * game_count
        for game_index in active_indexes:
            result = results[game_index]
            if result is None:
                continue
            policy = np.asarray(result.policy_target(), dtype=np.float32)
            action = int(result.selected_action())
            if use_full_by_game[game_index]:
                pending[game_index].append(
                    (
                        players[game_index],
                        features_by_game[game_index],
                        policy,
                        root_policy_logits_by_game[game_index],
                    )
                )
                trajectory_rows[game_index].append(
                    (
                        turn,
                        players[game_index],
                        features_by_game[game_index],
                        action,
                        policy,
                        root_policy_logits_by_game[game_index],
                    )
                )
            moves[game_index].append(
                MoveLog(turn=turn, player=players[game_index], action=action)
            )
            actions[game_index] = action
        batch.apply_actions(actions)
    else:
        unfinished = [
            seeds[index]
            for index, is_terminal in enumerate(batch.is_terminal())
            if not is_terminal
        ]
        if unfinished:
            raise RuntimeError(
                f"self-play exceeded max_turns={config.self_play.max_turns} "
                f"for seeds={unfinished}"
            )

    winners = batch.winners()
    end_reasons = batch.end_reasons()
    territory_scores = batch.territory_scores()
    logs: list[GameLog] = []
    samples: list[ReplaySample] = []
    episodes: list[TrajectoryEpisode] = []
    for game_index, seed in enumerate(seeds):
        winner = winners[game_index]
        end_reason = end_reasons[game_index]
        if winner is None or end_reason is None:
            raise RuntimeError("Rust ONNX self-play stopped before terminal outcome")
        log = GameLog(
            seed=seed,
            moves=moves[game_index],
            winner=int(winner),
            end_reason=int(end_reason),
            territory_scores=territory_scores[game_index],
        )
        logs.append(log)
        samples.extend(
            ReplaySample(
                features=features,
                policy=policy,
                value=value_target_for_player(player=player, winner=int(winner)),
                root_policy_logits=root_policy_logits,
            )
            for player, features, policy, root_policy_logits in pending[game_index]
        )
        transitions: list[TrajectoryTransition] = []
        rows = trajectory_rows[game_index]
        for index, (_turn, player, features, action, policy, root_policy_logits) in enumerate(rows):
            next_features = rows[index + 1][2] if index + 1 < len(rows) else None
            transitions.append(
                TrajectoryTransition(
                    episode_id=seed,
                    timestep=index,
                    player=player,
                    features=features,
                    legal_mask=legal_mask_from_features(features),
                    action=action,
                    policy_target=policy,
                    root_policy_logits=root_policy_logits,
                    root_value=None,
                    next_features=next_features,
                    winner=int(winner),
                    terminal=index == len(rows) - 1,
                    model_version=config.model_version,
                    search_config_hash="",
                    created_iteration=config.created_iteration,
                )
            )
        if transitions:
            episodes.append(
                TrajectoryEpisode(
                    episode_id=seed,
                    seed=seed,
                    transitions=tuple(transitions),
                    winner=int(winner),
                    end_reason=int(end_reason),
                    territory_scores=territory_scores[game_index],
                )
            )
    return logs, samples, episodes


def _search_active_with_root_policy_logits(
    batch: Any,
    *,
    evaluator: Any,
    request: Any,
    leaf_batch_size: int,
) -> tuple[list[Any | None], list[list[float]]]:
    if hasattr(batch, "search_active_with_onnx_evaluator_and_root_logits"):
        results, root_policy_logits = batch.search_active_with_onnx_evaluator_and_root_logits(
            evaluator,
            leaf_batch_size=leaf_batch_size,
        )
        return list(results), [list(row) for row in root_policy_logits]

    root_policy_logits, _root_values = evaluator.evaluate(request)
    rows = [list(row) for row in root_policy_logits]
    if any(len(row) != ACTION_SPACE for row in rows):
        raise ValueError("ONNX evaluator returned invalid root policy logits shape")
    results = batch.search_active_with_onnx_evaluator(
        evaluator,
        leaf_batch_size=leaf_batch_size,
    )
    return list(results), rows


def _flat_features_for_replay(feature_planes: list[float]) -> np.ndarray:
    features = np.asarray(feature_planes, dtype=np.float32)
    expected = FEATURE_CHANNELS * BOARD_SIZE * BOARD_SIZE
    if features.shape != (expected,):
        raise ValueError(f"expected flat feature shape {(expected,)}, got {features.shape}")
    return cast(np.ndarray, features.reshape(FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE))


def _use_full_search_turn(rng: random.Random, config: SelfPlayConfig) -> bool:
    if not config.playout_cap_randomization:
        return True
    return rng.random() < config.playout_cap_full_search_fraction


def _playout_cap_max_considered_actions(use_full_search: bool, config: SelfPlayConfig) -> int:
    if use_full_search:
        return (
            config.playout_cap_full_max_considered_actions
            if config.playout_cap_full_max_considered_actions is not None
            else config.gumbel_max_considered_actions
        )
    return (
        config.playout_cap_fast_max_considered_actions
        if config.playout_cap_fast_max_considered_actions is not None
        else config.gumbel_max_considered_actions
    )


def _import_core() -> Any:
    try:
        import great_kingdom_core as core  # type: ignore[import-untyped]
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "great_kingdom_core is not installed. Build it with maturin before Rust ONNX self-play."
        ) from exc
    return core


__all__ = [
    "RustOnnxSelfPlayConfig",
    "RustSelfPlayRunSummary",
    "run_rust_onnx_self_play",
]
