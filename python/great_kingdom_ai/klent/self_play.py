"""KLENT zero-search self-play for CPU smoke testing.

Each turn performs a single network forward pass, builds the closed-form
target policy ``pi'``, samples an action from it, and records
``E_pi'[Q]`` for the lambda-return computation. No MCTS is used.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from great_kingdom_ai.features import (
    ACTION_SPACE,
    BOARD_SIZE,
    FEATURE_CHANNELS,
)
from great_kingdom_ai.klent._torch import _import_torch
from great_kingdom_ai.klent.shards import klent_config_hash
from great_kingdom_ai.klent.targets import compute_analytical_policy, compute_lambda_returns
from great_kingdom_ai.klent.types import KlentConfig, KlentPolicyValueModel
from great_kingdom_ai.replay import (
    TrajectoryEpisode,
    TrajectoryTransition,
    legal_mask_from_features,
)
from great_kingdom_ai.self_play_data import value_target_for_player
from great_kingdom_ai.self_play_types import GameLog, MoveLog, SelfPlayState


@dataclass(frozen=True)
class KlentSelfPlayConfig:
    max_turns: int = 200
    episode_id: int = 0
    model_version: int = 0
    created_iteration: int = 0
    sample_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.max_turns <= 0:
            raise ValueError("max_turns must be positive")
        if self.episode_id < 0:
            raise ValueError("episode_id must be non-negative")
        if self.model_version < 0:
            raise ValueError("model_version must be non-negative")
        if self.created_iteration < 0:
            raise ValueError("created_iteration must be non-negative")
        if not np.isfinite(self.sample_weight) or self.sample_weight <= 0.0:
            raise ValueError("sample_weight must be finite and positive")


def play_klent_game(
    model: KlentPolicyValueModel,
    *,
    seed: int,
    config: KlentConfig,
    self_play: KlentSelfPlayConfig | None = None,
    state: SelfPlayState | None = None,
) -> tuple[GameLog, TrajectoryEpisode]:
    """Play one zero-search KLENT game and return its log and trajectory episode."""
    torch = _import_torch()
    options = self_play if self_play is not None else KlentSelfPlayConfig()
    game_state = state if state is not None else _create_game_state()
    rng = random.Random(seed)
    device = next(model.parameters()).device
    config_hash = klent_config_hash(config)

    moves: list[MoveLog] = []
    features_rows: list[np.ndarray] = []
    legal_mask_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    state_values: list[float] = []

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            turn = 0
            while not game_state.is_terminal():
                if turn >= options.max_turns:
                    raise RuntimeError(
                        f"KLENT self-play exceeded max_turns={options.max_turns} "
                        "without a terminal outcome"
                    )
                features = np.asarray(
                    game_state.feature_planes(),
                    dtype=np.float32,
                ).reshape(FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE)
                legal_mask = legal_mask_from_features(features)
                feature_tensor = torch.from_numpy(features).unsqueeze(0).to(device=device)
                legal_tensor = torch.from_numpy(legal_mask).unsqueeze(0).to(device=device)

                policy_logits, q_values = model.forward_q(feature_tensor)
                target = compute_analytical_policy(
                    policy_logits,
                    q_values,
                    legal_tensor,
                    alpha=config.alpha,
                    beta=config.beta,
                )[0]
                q_row = q_values[0].to(dtype=torch.float32)
                action = int(
                    rng.choices(
                        range(ACTION_SPACE),
                        weights=target.detach().cpu().tolist(),
                        k=1,
                    )[0]
                )

                features_rows.append(features)
                legal_mask_rows.append(legal_mask)
                target_rows.append(target.detach().cpu().numpy().astype(np.float32))
                state_values.append(float((target * q_row).sum().detach().cpu()))
                moves.append(
                    MoveLog(turn=turn, player=game_state.current_player(), action=action)
                )
                game_state.apply_action(action)
                turn += 1
    finally:
        if was_training:
            model.train()

    winner = game_state.winner()
    end_reason = game_state.end_reason()
    if winner is None or end_reason is None:
        raise RuntimeError("KLENT self-play stopped before a terminal outcome")

    players = np.asarray([move.player for move in moves], dtype=np.int64)
    rewards = np.zeros((len(moves),), dtype=np.float32)
    rewards[-1] = value_target_for_player(player=int(players[-1]), winner=int(winner))
    returns = compute_lambda_returns(
        rewards,
        np.asarray(state_values, dtype=np.float32),
        players,
        lambda_param=config.lambda_param,
        gamma=config.gamma,
    )

    transitions = tuple(
        TrajectoryTransition(
            episode_id=options.episode_id,
            timestep=index,
            player=int(players[index]),
            features=features_rows[index],
            legal_mask=legal_mask_rows[index],
            action=int(moves[index].action),
            policy_target=target_rows[index],
            winner=int(winner),
            terminal=index == len(moves) - 1,
            model_version=options.model_version,
            search_config_hash=config_hash,
            created_iteration=options.created_iteration,
            sample_weight=options.sample_weight,
            lambda_return=float(returns[index]),
        )
        for index in range(len(moves))
    )
    episode = TrajectoryEpisode(
        episode_id=options.episode_id,
        seed=seed,
        transitions=transitions,
        winner=int(winner),
        end_reason=int(end_reason),
        territory_scores=game_state.territory_scores(),
    )
    log = GameLog(
        seed=seed,
        moves=moves,
        winner=int(winner),
        end_reason=int(end_reason),
        territory_scores=game_state.territory_scores(),
    )
    return log, episode


def play_klent_games(
    model: KlentPolicyValueModel,
    *,
    seeds: list[int],
    config: KlentConfig,
    model_version: int = 0,
    created_iteration: int = 0,
    max_turns: int = 200,
    first_episode_id: int = 0,
) -> list[tuple[GameLog, TrajectoryEpisode]]:
    return [
        play_klent_game(
            model,
            seed=seed,
            config=config,
            self_play=KlentSelfPlayConfig(
                max_turns=max_turns,
                episode_id=first_episode_id + index,
                model_version=model_version,
                created_iteration=created_iteration,
            ),
        )
        for index, seed in enumerate(seeds)
    ]


def _create_game_state() -> SelfPlayState:
    from great_kingdom_ai.game_core import create_core_game_state

    return create_core_game_state()


__all__ = [
    "KlentSelfPlayConfig",
    "play_klent_game",
    "play_klent_games",
]