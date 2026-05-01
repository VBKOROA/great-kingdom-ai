"""Random self-play orchestration backed by the Rust rules engine."""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any, NoReturn, Protocol, cast

import numpy as np

from great_kingdom_ai.features import BOARD_SIZE, FEATURE_CHANNELS
from great_kingdom_ai.replay_buffer import ReplaySample
from great_kingdom_ai.self_play_data import (
    apply_root_dirichlet_noise,
    policy_target_from_visit_counts,
    select_action_from_visit_counts,
    value_target_for_player,
)

PASS_ACTION = 81


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


class MctsResultLike(Protocol):
    def selected_action(self) -> int | None: ...

    def visit_counts(self) -> list[int]: ...


class MctsSearchLike(Protocol):
    def search(self, state: SelfPlayState) -> MctsResultLike: ...

    def search_with_priors(
        self,
        state: SelfPlayState,
        priors: list[float],
    ) -> MctsResultLike: ...


@dataclass(frozen=True)
class MoveLog:
    turn: int
    player: int
    action: int


@dataclass(frozen=True)
class GameLog:
    seed: int
    moves: list[MoveLog]
    winner: int
    end_reason: int
    territory_scores: tuple[int, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SmokeSummary:
    games: int
    total_moves: int
    max_moves: int
    blue_wins: int
    orange_wins: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class MctsSelfPlayConfig:
    max_turns: int = 200
    temperature_turns: int = 10
    sampling_temperature: float = 1.0
    root_noise: bool = True
    root_dirichlet_alpha: float = 0.3
    root_exploration_fraction: float = 0.25


def choose_random_legal_action(
    state: SelfPlayState,
    rng: random.Random,
    *,
    prefer_place: bool = False,
) -> int:
    """Choose one action from the Rust-provided legal action list."""
    legal_actions = state.legal_actions()
    if not legal_actions:
        raise ValueError("state has no legal actions")

    candidates: Sequence[int] = legal_actions
    if prefer_place:
        place_actions = [action for action in legal_actions if action != PASS_ACTION]
        if place_actions:
            candidates = place_actions

    return rng.choice(candidates)


def play_random_game(
    *,
    seed: int,
    state: SelfPlayState | None = None,
    max_turns: int = 200,
    prefer_place: bool = False,
) -> GameLog:
    game_state = state if state is not None else create_core_game_state()
    rng = random.Random(seed)
    moves: list[MoveLog] = []

    for turn in range(max_turns):
        if game_state.is_terminal():
            break

        action = choose_random_legal_action(game_state, rng, prefer_place=prefer_place)
        moves.append(MoveLog(turn=turn, player=game_state.current_player(), action=action))
        game_state.apply_action(action)
    else:
        raise RuntimeError(f"random self-play exceeded max_turns={max_turns}")

    winner = game_state.winner()
    end_reason = game_state.end_reason()
    if winner is None or end_reason is None:
        raise RuntimeError("random self-play stopped before terminal outcome")

    return GameLog(
        seed=seed,
        moves=moves,
        winner=winner,
        end_reason=end_reason,
        territory_scores=game_state.territory_scores(),
    )


def play_random_games(
    *,
    seeds: Sequence[int],
    max_turns: int = 200,
    prefer_place: bool = False,
) -> list[GameLog]:
    return [
        play_random_game(seed=seed, max_turns=max_turns, prefer_place=prefer_place)
        for seed in seeds
    ]


def play_mcts_game(
    *,
    seed: int,
    state: SelfPlayState | None = None,
    search: MctsSearchLike | None = None,
    config: MctsSelfPlayConfig | None = None,
    prior_provider: Callable[[SelfPlayState], Sequence[float]] | None = None,
) -> tuple[GameLog, list[ReplaySample]]:
    """Run one MCTS self-play game and return replay samples with final value targets."""
    config = config if config is not None else MctsSelfPlayConfig()
    game_state = state if state is not None else create_core_game_state()
    mcts = search if search is not None else create_core_mcts_search()
    rng = random.Random(seed)
    moves: list[MoveLog] = []
    pending_samples: list[tuple[int, np.ndarray, np.ndarray]] = []

    for turn in range(config.max_turns):
        if game_state.is_terminal():
            break

        player = game_state.current_player()
        features = _state_features_for_replay(game_state)
        root_priors = prior_provider(game_state) if prior_provider is not None else None
        result = _run_self_play_search(game_state, mcts, rng, config, root_priors=root_priors)
        visit_counts = result.visit_counts()
        policy = policy_target_from_visit_counts(visit_counts)
        temperature = (
            config.sampling_temperature if turn < config.temperature_turns else 0.0
        )
        action = select_action_from_visit_counts(
            visit_counts,
            rng,
            temperature=temperature,
        )

        pending_samples.append((player, features, policy))
        moves.append(MoveLog(turn=turn, player=player, action=action))
        game_state.apply_action(action)
    else:
        raise RuntimeError(f"MCTS self-play exceeded max_turns={config.max_turns}")

    winner = game_state.winner()
    end_reason = game_state.end_reason()
    if winner is None or end_reason is None:
        raise RuntimeError("MCTS self-play stopped before terminal outcome")

    samples = [
        ReplaySample(
            features=features,
            policy=policy,
            value=value_target_for_player(player=player, winner=winner),
        )
        for player, features, policy in pending_samples
    ]
    return (
        GameLog(
            seed=seed,
            moves=moves,
            winner=winner,
            end_reason=end_reason,
            territory_scores=game_state.territory_scores(),
        ),
        samples,
    )


def summarize_logs(logs: Sequence[GameLog]) -> SmokeSummary:
    move_counts = [len(log.moves) for log in logs]
    return SmokeSummary(
        games=len(logs),
        total_moves=sum(move_counts),
        max_moves=max(move_counts, default=0),
        blue_wins=sum(1 for log in logs if log.winner == 1),
        orange_wins=sum(1 for log in logs if log.winner == 2),
    )


def create_core_game_state() -> SelfPlayState:
    try:
        import great_kingdom_core as core  # type: ignore[import-untyped]
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "great_kingdom_core is not installed. Build it with maturin before self-play."
        ) from exc

    return cast(SelfPlayState, core.GameState())


def create_core_mcts_search(*, simulations: int = 50, c_puct: float = 1.5) -> MctsSearchLike:
    try:
        import great_kingdom_core as core
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "great_kingdom_core is not installed. Build it with maturin before self-play."
        ) from exc

    return cast(MctsSearchLike, core.MctsSearch(simulations=simulations, c_puct=c_puct))


def _run_self_play_search(
    state: SelfPlayState,
    search: MctsSearchLike,
    rng: random.Random,
    config: MctsSelfPlayConfig,
    *,
    root_priors: Sequence[float] | None = None,
) -> MctsResultLike:
    if root_priors is None and not config.root_noise:
        return search.search(state)

    priors = list(root_priors) if root_priors is not None else [0.0] * PASS_ACTION + [0.0]
    noisy_priors = apply_root_dirichlet_noise(
        priors,
        state.legal_mask(),
        rng,
        alpha=config.root_dirichlet_alpha,
        epsilon=config.root_exploration_fraction,
    )
    if config.root_noise:
        return search.search_with_priors(state, noisy_priors.tolist())
    return search.search_with_priors(state, priors)


def _state_features_for_replay(state: SelfPlayState) -> np.ndarray:
    features = np.asarray(state.feature_planes(), dtype=np.float32)
    expected = FEATURE_CHANNELS * BOARD_SIZE * BOARD_SIZE
    if features.shape != (expected,):
        raise ValueError(f"expected flat feature shape {(expected,)}, got {features.shape}")
    return features.reshape(FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="great-kingdom-random-self-play",
        description="Run random self-play games against the Rust rules engine.",
    )
    parser.add_argument("--games", type=int, default=10, help="number of games to run")
    parser.add_argument("--seed-start", type=int, default=0, help="first deterministic seed")
    parser.add_argument("--max-turns", type=int, default=200, help="guard per game")
    parser.add_argument(
        "--prefer-place",
        action="store_true",
        help="prefer place actions while any place action is legal",
    )
    parser.add_argument("--json", action="store_true", help="print JSON logs instead of summary")
    return parser


def main() -> NoReturn:
    args = build_parser().parse_args()
    if args.games < 0:
        raise SystemExit("--games must be non-negative")

    seeds = list(range(args.seed_start, args.seed_start + args.games))
    logs = play_random_games(
        seeds=seeds,
        max_turns=args.max_turns,
        prefer_place=args.prefer_place,
    )
    if args.json:
        print(json.dumps([log.to_dict() for log in logs], indent=2, sort_keys=True))
    else:
        print(json.dumps(summarize_logs(logs).to_dict(), sort_keys=True))

    raise SystemExit(0)


if __name__ == "__main__":
    main()
