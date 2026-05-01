"""Random self-play orchestration backed by the Rust rules engine."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol, cast

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


def create_core_game_state() -> SelfPlayState:
    try:
        import great_kingdom_core as core  # type: ignore[import-untyped]
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "great_kingdom_core is not installed. Build it with maturin before self-play."
        ) from exc

    return cast(SelfPlayState, core.GameState())
