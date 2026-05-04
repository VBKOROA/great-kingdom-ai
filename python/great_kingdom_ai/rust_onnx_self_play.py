"""Python wrapper for Rust ONNX self-play artifact generation."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from great_kingdom_ai.features import BOARD_SIZE, FEATURE_CHANNELS
from great_kingdom_ai.replay_buffer import ReplaySample
from great_kingdom_ai.rust_onnx_replay import write_rust_self_play_artifacts
from great_kingdom_ai.self_play import GameLog, MoveLog, SelfPlayConfig
from great_kingdom_ai.self_play_data import value_target_for_player


@dataclass(frozen=True)
class RustOnnxSelfPlayConfig:
    onnx_model_path: Path
    output_dir: Path
    games: int = 2
    seed_start: int = 0
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
    remaining = config.games
    seed = config.seed_start
    while remaining > 0:
        batch_size = min(config.rust_self_play_batch_size, remaining)
        batch_logs, batch_samples = _run_one_batch(
            config,
            core=core,
            evaluator=evaluator,
            seed_start=seed,
            game_count=batch_size,
        )
        logs.extend(batch_logs)
        samples.extend(batch_samples)
        seed += batch_size
        remaining -= batch_size

    artifact_dir = write_rust_self_play_artifacts(
        output_dir=config.output_dir,
        samples=samples,
        logs=logs,
        manifest={
            "format_version": 1,
            "board_size": BOARD_SIZE,
            "feature_channels": FEATURE_CHANNELS,
            "action_space": 82,
            "model_path": str(config.onnx_model_path),
            "onnx_device": config.onnx_device,
            "gumbel_config": asdict(config.self_play),
            "seed_start": config.seed_start,
            "games": config.games,
        },
    )
    return RustSelfPlayRunSummary(
        artifact_dir=artifact_dir,
        games=len(logs),
        samples=len(samples),
        onnx_model_path=config.onnx_model_path,
        onnx_device=config.onnx_device,
    )


def _run_one_batch(
    config: RustOnnxSelfPlayConfig,
    *,
    core: Any,
    evaluator: Any,
    seed_start: int,
    game_count: int,
) -> tuple[list[GameLog], list[ReplaySample]]:
    batch = core.GumbelSelfPlayBatch(
        game_count=game_count,
        simulations=config.self_play.gumbel_simulations,
        max_considered_actions=config.self_play.gumbel_max_considered_actions,
        c_visit=config.self_play.gumbel_c_visit,
        c_scale=config.self_play.gumbel_c_scale,
        seed=config.self_play.gumbel_seed + seed_start,
    )
    seeds = list(range(seed_start, seed_start + game_count))
    rngs = [random.Random(seed) for seed in seeds]
    moves: list[list[MoveLog]] = [[] for _ in seeds]
    pending: list[list[tuple[int, np.ndarray, np.ndarray]]] = [[] for _ in seeds]

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

        results = batch.search_active_with_onnx_evaluator(
            evaluator,
            leaf_batch_size=config.self_play.leaf_batch_size,
        )
        actions: list[int | None] = [None] * game_count
        for game_index in active_indexes:
            result = results[game_index]
            if result is None:
                continue
            policy = np.asarray(result.policy_target(), dtype=np.float32)
            action = int(result.selected_action())
            if use_full_by_game[game_index]:
                pending[game_index].append(
                    (players[game_index], features_by_game[game_index], policy)
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
    for game_index, seed in enumerate(seeds):
        winner = winners[game_index]
        end_reason = end_reasons[game_index]
        if winner is None or end_reason is None:
            raise RuntimeError("Rust ONNX self-play stopped before terminal outcome")
        logs.append(
            GameLog(
                seed=seed,
                moves=moves[game_index],
                winner=int(winner),
                end_reason=int(end_reason),
                territory_scores=territory_scores[game_index],
            )
        )
        samples.extend(
            ReplaySample(
                features=features,
                policy=policy,
                value=value_target_for_player(player=player, winner=int(winner)),
            )
            for player, features, policy in pending[game_index]
        )
    return logs, samples


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
