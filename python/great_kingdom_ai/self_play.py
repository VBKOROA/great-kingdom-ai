"""Random self-play orchestration backed by the Rust rules engine."""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, NoReturn, Protocol, cast

PASS_ACTION = 81


class SelfPlayState(Protocol):
    def current_player(self) -> int: ...

    def legal_actions(self) -> list[int]: ...

    def apply_action(self, action_index: int) -> int | None: ...

    def is_terminal(self) -> bool: ...

    def winner(self) -> int | None: ...

    def end_reason(self) -> int | None: ...

    def territory_scores(self) -> tuple[int, int]: ...


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
