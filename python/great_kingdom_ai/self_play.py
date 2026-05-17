"""Random and Gumbel self-play public API."""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Callable, Sequence
from typing import Any, NoReturn, cast

from great_kingdom_ai import game_core
from great_kingdom_ai.replay.sample import ReplaySample
from great_kingdom_ai.self_play_runner import play_self_play_game as _play_self_play_game
from great_kingdom_ai.self_play_runner import (
    play_self_play_games_batched as _play_self_play_games_batched,
)
from great_kingdom_ai.self_play_types import (
    PASS_ACTION,
    GameLog,
    MoveLog,
    SearchLike,
    SearchResultLike,
    SelfPlayBatchLike,
    SelfPlayConfig,
    SelfPlayState,
    SmokeSummary,
)


def choose_random_legal_action(
    state: SelfPlayState,
    rng: random.Random,
    *,
    prefer_place: bool = False,
) -> int:
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


def play_self_play_game(
    *,
    seed: int,
    state: SelfPlayState | None = None,
    search: SearchLike | None = None,
    config: SelfPlayConfig | None = None,
    prior_provider: Callable[[SelfPlayState], Sequence[float]] | None = None,
    evaluator_provider: Callable[
        [Sequence[SelfPlayState]],
        tuple[Sequence[Sequence[float]], Sequence[float]],
    ]
    | None = None,
) -> tuple[GameLog, list[ReplaySample]]:
    return _play_self_play_game(
        seed=seed,
        state=state,
        search=search,
        config=config,
        prior_provider=prior_provider,
        evaluator_provider=evaluator_provider,
        state_factory=create_core_game_state,
        search_engine_factory=lambda cfg, offset: create_core_search_engine(
            cfg,
            seed_offset=offset,
        ),
    )


def play_self_play_games_batched(
    *,
    seeds: Sequence[int],
    search_factory: Callable[[], SearchLike],
    config: SelfPlayConfig,
    prior_provider: Callable[[Sequence[SelfPlayState]], Sequence[Sequence[float]]]
    | None = None,
    evaluator_provider: Callable[
        [Sequence[SelfPlayState]],
        tuple[Sequence[Sequence[float]], Sequence[float]],
    ]
    | None = None,
    feature_batch_prior_provider: Callable[
        [Sequence[Sequence[float]], Sequence[Sequence[bool]]],
        Sequence[Sequence[float]],
    ]
    | None = None,
    request_evaluator_provider: Callable[[Any], tuple[Any, Any]] | None = None,
    state_factory: Callable[[], SelfPlayState] | None = None,
) -> list[tuple[GameLog, list[ReplaySample]]]:
    return _play_self_play_games_batched(
        seeds=seeds,
        search_factory=search_factory,
        config=config,
        prior_provider=prior_provider,
        evaluator_provider=evaluator_provider,
        feature_batch_prior_provider=feature_batch_prior_provider,
        request_evaluator_provider=request_evaluator_provider,
        state_factory=state_factory,
        default_state_factory=create_core_game_state,
        core_batch_factory=lambda cfg, count: create_core_self_play_batch(
            cfg,
            game_count=count,
        ),
        core_batch_available=_can_create_core_self_play_batch,
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
    return game_core.create_core_game_state()


def create_core_search_engine(
    config: SelfPlayConfig,
    *,
    seed_offset: int = 0,
) -> SearchLike:
    return cast(SearchLike, game_core.create_core_gumbel_search(config, seed_offset=seed_offset))


def create_core_self_play_batch(
    config: SelfPlayConfig,
    *,
    game_count: int,
    seed_offset: int = 0,
) -> SelfPlayBatchLike:
    return cast(
        SelfPlayBatchLike,
        game_core.create_core_self_play_batch(
            config,
            game_count=game_count,
            seed_offset=seed_offset,
        ),
    )


def _can_create_core_self_play_batch() -> bool:
    return game_core.can_create_core_self_play_batch()


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


__all__ = [
    "PASS_ACTION",
    "GameLog",
    "MoveLog",
    "SearchLike",
    "SearchResultLike",
    "SelfPlayBatchLike",
    "SelfPlayConfig",
    "SelfPlayState",
    "SmokeSummary",
    "choose_random_legal_action",
    "create_core_game_state",
    "create_core_search_engine",
    "create_core_self_play_batch",
    "play_random_game",
    "play_random_games",
    "play_self_play_game",
    "play_self_play_games_batched",
    "summarize_logs",
]
