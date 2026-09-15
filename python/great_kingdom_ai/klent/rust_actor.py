"""Rust zero-search KLENT actor runner.

The actor ONNX model is exported separately (3 outputs). Rust computes the
legal-masked ``pi'`` target and ``E_pi'[Q]`` per turn and samples actions;
Python reconstructs features, applies the shared lambda-return function, and
writes the strict KLENT shard.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from great_kingdom_ai.features import BOARD_SIZE, FEATURE_CHANNELS
from great_kingdom_ai.game_core import import_core
from great_kingdom_ai.klent.shards import klent_config_hash
from great_kingdom_ai.klent.targets import compute_lambda_returns
from great_kingdom_ai.klent.types import KlentConfig
from great_kingdom_ai.replay import (
    TrajectoryEpisode,
    TrajectoryReplayStore,
    TrajectoryTransition,
)
from great_kingdom_ai.self_play_data import value_target_for_player
from great_kingdom_ai.self_play_types import GameLog, MoveLog


@dataclass(frozen=True)
class RustKlentActorConfig:
    actor_onnx_path: Path
    output_dir: Path
    games: int
    seed_start: int = 0
    alpha: float = 0.03
    beta: float = 0.1
    lambda_param: float = 0.8825
    gamma: float = 1.0
    max_turns: int = 200
    onnx_device: str = "cpu"
    onnx_max_batch_size: int = 256
    rust_self_play_batch_size: int = 64
    model_version: int = 0
    created_iteration: int = 0
    episode_id_offset: int = 0

    def __post_init__(self) -> None:
        if self.games <= 0:
            raise ValueError("games must be positive")
        if self.seed_start < 0:
            raise ValueError("seed_start must be non-negative")
        if self.max_turns <= 0:
            raise ValueError("max_turns must be positive")
        if self.rust_self_play_batch_size <= 0:
            raise ValueError("rust_self_play_batch_size must be positive")
        if self.episode_id_offset < 0:
            raise ValueError("episode_id_offset must be non-negative")
        KlentConfig(
            alpha=self.alpha,
            beta=self.beta,
            lambda_param=self.lambda_param,
            gamma=self.gamma,
        )


@dataclass(frozen=True)
class RustKlentRunSummary:
    artifact_dir: Path
    games: int
    transitions: int
    actor_onnx_path: Path
    game_logs: tuple[GameLog, ...]
    trajectory_episodes: tuple[TrajectoryEpisode, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_dir": str(self.artifact_dir),
            "games": self.games,
            "transitions": self.transitions,
            "actor_onnx_path": str(self.actor_onnx_path),
        }


def run_rust_klent_zero_search(config: RustKlentActorConfig) -> RustKlentRunSummary:
    """Run batched zero-search self-play and write the shard artifacts."""
    summary = play_rust_klent_zero_search(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    replay = TrajectoryReplayStore.from_episodes(
        max(1, summary.transitions),
        summary.trajectory_episodes,
    )
    replay.save(config.output_dir / "trajectory-replay.npz", compressed=False)
    with (config.output_dir / "game_logs.json").open("w", encoding="utf-8") as file:
        json.dump([log.to_dict() for log in summary.game_logs], file, indent=2, sort_keys=True)
    return summary


def play_rust_klent_zero_search(config: RustKlentActorConfig) -> RustKlentRunSummary:
    """Run batched zero-search self-play without writing artifacts."""
    klent = KlentConfig(
        alpha=config.alpha,
        beta=config.beta,
        lambda_param=config.lambda_param,
        gamma=config.gamma,
    )
    core = import_core("KLENT zero-search actor")
    logs: list[GameLog] = []
    episodes: list[TrajectoryEpisode] = []
    remaining = config.games
    seed = config.seed_start
    game_offset = 0
    while remaining > 0:
        batch_size = min(config.rust_self_play_batch_size, remaining)
        actor = core.KlentZeroSearchBatch(
            str(config.actor_onnx_path),
            batch_size,
            seed,
            device=config.onnx_device,
            max_batch_size=config.onnx_max_batch_size,
            alpha=config.alpha,
            beta=config.beta,
            max_turns=config.max_turns,
        )
        actor.run()
        for game_index in range(batch_size):
            log, episode = _episode_from_actor(
                actor,
                game_index,
                core=core,
                seed=seed + game_index,
                episode_id=config.episode_id_offset + game_offset + game_index,
                config=config,
                klent=klent,
            )
            logs.append(log)
            episodes.append(episode)
        seed += batch_size
        game_offset += batch_size
        remaining -= batch_size

    transition_count = sum(len(episode.transitions) for episode in episodes)
    return RustKlentRunSummary(
        artifact_dir=config.output_dir,
        games=config.games,
        transitions=transition_count,
        actor_onnx_path=config.actor_onnx_path,
        game_logs=tuple(logs),
        trajectory_episodes=tuple(episodes),
    )


def _episode_from_actor(
    actor: Any,
    game_index: int,
    *,
    core: Any,
    seed: int,
    episode_id: int,
    config: RustKlentActorConfig,
    klent: KlentConfig,
) -> tuple[GameLog, TrajectoryEpisode]:
    winner = actor.winners()[game_index]
    end_reason = actor.end_reasons()[game_index]
    territory_scores = actor.territory_scores()[game_index]
    if winner is None or end_reason is None:
        raise RuntimeError("zero-search actor returned an unfinished game")

    actions = [int(action) for action in actor.turn_actions(game_index)]
    players = [int(player) for player in actor.turn_players(game_index)]
    policy_targets = [
        np.asarray(target, dtype=np.float32) for target in actor.policy_targets(game_index)
    ]
    state_values = [float(value) for value in actor.state_values(game_index)]
    if not actions:
        raise RuntimeError("zero-search actor returned a game without moves")
    if not (len(actions) == len(players) == len(policy_targets) == len(state_values)):
        raise RuntimeError("zero-search actor turn records are inconsistent")

    rewards = np.zeros((len(actions),), dtype=np.float32)
    rewards[-1] = value_target_for_player(player=players[-1], winner=int(winner))
    returns = compute_lambda_returns(
        rewards,
        np.asarray(state_values, dtype=np.float32),
        np.asarray(players, dtype=np.int64),
        lambda_param=klent.lambda_param,
        gamma=klent.gamma,
    )

    config_hash = klent_config_hash(klent)
    state = core.GameState()
    transitions: list[TrajectoryTransition] = []
    moves: list[MoveLog] = []
    for index, action in enumerate(actions):
        if state.is_terminal():
            raise RuntimeError("zero-search actor recorded a move after a terminal state")
        features = np.asarray(
            state.feature_planes(),
            dtype=np.float32,
        ).reshape(FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE)
        legal_mask = np.asarray(state.legal_mask(), dtype=np.bool_)
        if int(state.current_player()) != players[index]:
            raise RuntimeError("reconstructed player does not match the actor record")
        if not bool(legal_mask[action]):
            raise RuntimeError("zero-search actor returned an illegal action")
        transitions.append(
            TrajectoryTransition(
                episode_id=episode_id,
                timestep=index,
                player=players[index],
                features=features,
                legal_mask=legal_mask,
                action=action,
                policy_target=policy_targets[index],
                winner=int(winner),
                terminal=index == len(actions) - 1,
                model_version=config.model_version,
                search_config_hash=config_hash,
                created_iteration=config.created_iteration,
                lambda_return=float(returns[index]),
            )
        )
        moves.append(
            MoveLog(turn=index, player=players[index], action=action)
        )
        state.apply_action(action)

    episode = TrajectoryEpisode(
        episode_id=episode_id,
        seed=seed,
        transitions=tuple(transitions),
        winner=int(winner),
        end_reason=int(end_reason),
        territory_scores=(int(territory_scores[0]), int(territory_scores[1])),
    )
    log = GameLog(
        seed=seed,
        moves=moves,
        winner=int(winner),
        end_reason=int(end_reason),
        territory_scores=(int(territory_scores[0]), int(territory_scores[1])),
    )
    return log, episode


__all__ = [
    "RustKlentActorConfig",
    "RustKlentRunSummary",
    "play_rust_klent_zero_search",
    "run_rust_klent_zero_search",
]