"""Manual self-play CLI for checking the Rust rules engine."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import NoReturn, Protocol, cast

BOARD_SIZE = 9
BOARD_CELLS = BOARD_SIZE * BOARD_SIZE
PASS_ACTION = BOARD_CELLS

CELL_LABELS = {
    0: ".",
    1: "B",
    2: "O",
    3: "N",
}
PLAYER_NAMES = {
    1: "Blue",
    2: "Orange",
}
END_REASON_NAMES = {
    1: "opponent castle destroyed",
    2: "own castle destroyed",
    3: "consecutive passes",
}


class CliExit(Exception):
    """Raised when the player asks to leave the CLI."""


class GameStateProtocol(Protocol):
    def board(self) -> list[int]: ...

    def current_player(self) -> int: ...

    def blue_used(self) -> int: ...

    def orange_used(self) -> int: ...

    def previous_pass(self) -> bool: ...

    def winner(self) -> int | None: ...

    def end_reason(self) -> int | None: ...

    def legal_actions(self) -> list[int]: ...

    def territory_scores(self) -> tuple[int, int]: ...

    def apply_action(self, action_index: int) -> int | None: ...

    def is_terminal(self) -> bool: ...


@dataclass(frozen=True)
class ParsedCommand:
    action: int | None = None
    show_help: bool = False
    show_legal: bool = False
    show_board: bool = False


def parse_command(raw: str) -> ParsedCommand:
    text = raw.strip().lower()
    if not text:
        raise ValueError("empty input")

    if text in {"q", "quit", "exit"}:
        raise CliExit
    if text in {"h", "help", "?"}:
        return ParsedCommand(show_help=True)
    if text in {"b", "board"}:
        return ParsedCommand(show_board=True)
    if text in {"l", "legal"}:
        return ParsedCommand(show_legal=True)
    if text in {"p", "pass"}:
        return ParsedCommand(action=PASS_ACTION)

    if text.startswith("i "):
        return ParsedCommand(action=parse_action_index(text.removeprefix("i ").strip()))

    coordinate = parse_coordinate(text)
    if coordinate is not None:
        row, col = coordinate
        return ParsedCommand(action=row * BOARD_SIZE + col)

    parts = text.replace(",", " ").split()
    if len(parts) == 2:
        row = parse_one_based_number(parts[0], "row")
        col = parse_one_based_number(parts[1], "column")
        return ParsedCommand(action=(row - 1) * BOARD_SIZE + (col - 1))

    return ParsedCommand(action=parse_action_index(text))


def parse_action_index(text: str) -> int:
    try:
        action = int(text)
    except ValueError as exc:
        raise ValueError(f"unknown command or coordinate: {text!r}") from exc

    if action < 0 or action > PASS_ACTION:
        raise ValueError(f"action index must be between 0 and {PASS_ACTION}")
    return action


def parse_action_sequence(raw: str) -> list[int]:
    actions: list[int] = []
    for token in raw.replace(",", " ").split():
        parsed = parse_command(token)
        if parsed.action is None:
            raise ValueError(f"replay token must be an action or coordinate: {token!r}")
        actions.append(parsed.action)
    if not actions:
        raise ValueError("replay action sequence is empty")
    return actions


def parse_coordinate(text: str) -> tuple[int, int] | None:
    if len(text) < 2 or len(text) > 3:
        return None

    col_text = text[0]
    row_text = text[1:]
    if col_text < "a" or col_text > "i" or not row_text.isdecimal():
        return None

    row = int(row_text)
    if row < 1 or row > BOARD_SIZE:
        raise ValueError("row must be between 1 and 9")
    return row - 1, ord(col_text) - ord("a")


def parse_one_based_number(text: str, name: str) -> int:
    if not text.isdecimal():
        raise ValueError(f"{name} must be a number between 1 and 9")
    value = int(text)
    if value < 1 or value > BOARD_SIZE:
        raise ValueError(f"{name} must be between 1 and 9")
    return value


def render_board(board: Iterable[int]) -> str:
    cells = list(board)
    if len(cells) != BOARD_CELLS:
        raise ValueError(f"board must have {BOARD_CELLS} cells")

    lines = ["    A B C D E F G H I"]
    for row in range(BOARD_SIZE):
        start = row * BOARD_SIZE
        labels = " ".join(CELL_LABELS.get(cell, "?") for cell in cells[start : start + BOARD_SIZE])
        lines.append(f"{row + 1:>2}  {labels}")
    return "\n".join(lines)


def format_legal_actions(actions: Iterable[int]) -> str:
    action_list = list(actions)
    place_actions = sorted(action for action in action_list if action != PASS_ACTION)
    coordinates = [index_to_coordinate(action) for action in place_actions]
    chunks = [" ".join(coordinates[index : index + 18]) for index in range(0, len(coordinates), 18)]
    if PASS_ACTION in action_list:
        chunks.append("PASS")
    return "\n".join(chunk for chunk in chunks if chunk)


def index_to_coordinate(action: int) -> str:
    if action == PASS_ACTION:
        return "PASS"
    if action < 0 or action >= BOARD_CELLS:
        raise ValueError(f"place action must be between 0 and {BOARD_CELLS - 1}")
    row = action // BOARD_SIZE
    col = action % BOARD_SIZE
    return f"{chr(ord('A') + col)}{row + 1}"


def help_text() -> str:
    return "\n".join(
        [
            "Commands:",
            "  A1..I9 or 'row col'  place a castle",
            "  p / pass             pass",
            "  l / legal            show legal moves",
            "  b / board            redraw board",
            "  i <0-81>             apply raw action index",
            "  q / quit             exit",
        ]
    )


def status_line(state: GameStateProtocol) -> str:
    current_player = PLAYER_NAMES.get(state.current_player(), f"Player {state.current_player()}")
    blue_territory, orange_territory = state.territory_scores()
    return (
        f"Turn: {current_player} | "
        f"Blue used: {state.blue_used()}/40 | "
        f"Orange used: {state.orange_used()}/40 | "
        f"Territory: Blue {blue_territory}, Orange {orange_territory} | "
        f"Previous pass: {state.previous_pass()}"
    )


def outcome_line(state: GameStateProtocol) -> str:
    winner = state.winner()
    reason = state.end_reason()
    winner_name = PLAYER_NAMES[winner] if winner is not None else "Unknown"
    reason_name = END_REASON_NAMES[reason] if reason is not None else "unknown reason"
    blue_territory, orange_territory = state.territory_scores()
    return (
        f"Game over: {winner_name} wins by {reason_name}. "
        f"Territory: Blue {blue_territory}, Orange {orange_territory}."
    )


def move_line(*, turn: int, player: int, action: int) -> str:
    player_name = PLAYER_NAMES.get(player, f"Player {player}")
    return f"Move {turn + 1:>2}: {player_name} {index_to_coordinate(action)} (action {action})"


def replay_actions(
    state: GameStateProtocol,
    actions: Iterable[int],
    *,
    pause: bool = False,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
) -> int:
    action_list = list(actions)
    print_fn(render_board(state.board()))
    print_fn(status_line(state))

    for turn, action in enumerate(action_list):
        if state.is_terminal():
            print_fn(f"Replay stopped: game already ended before move {turn + 1}.")
            return 1

        player = state.current_player()
        print_fn(move_line(turn=turn, player=player, action=action))
        try:
            state.apply_action(action)
        except ValueError as exc:
            print_fn(f"Illegal replay move: {exc}")
            return 1

        print_fn(render_board(state.board()))
        if state.is_terminal():
            print_fn(outcome_line(state))
            return 0
        print_fn(status_line(state))
        if pause and turn != len(action_list) - 1:
            input_fn("Press Enter for next move...")

    if not state.is_terminal():
        print_fn("Replay ended before terminal outcome.")
    return 0


def run_repl(
    *,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
) -> int:
    try:
        import great_kingdom_core as core  # type: ignore[import-untyped]
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "great_kingdom_core is not installed. Build it first with:\n"
            "  cd rust/great_kingdom_core\n"
            "  ../../.venv/bin/python -m maturin develop\n"
            "  cd ../.."
        ) from exc

    state = cast(GameStateProtocol, core.GameState())
    print_fn(render_board(state.board()))
    print_fn(status_line(state))
    print_fn(help_text())

    while not state.is_terminal():
        player = PLAYER_NAMES.get(state.current_player(), f"Player {state.current_player()}")
        try:
            parsed = parse_command(input_fn(f"{player}> "))
        except CliExit:
            print_fn("Exited.")
            return 0
        except ValueError as exc:
            print_fn(f"Input error: {exc}")
            continue

        if parsed.show_help:
            print_fn(help_text())
            continue
        if parsed.show_board:
            print_fn(render_board(state.board()))
            print_fn(status_line(state))
            continue
        if parsed.show_legal:
            print_fn(format_legal_actions(state.legal_actions()))
            continue

        if parsed.action is None:
            continue

        try:
            state.apply_action(parsed.action)
        except ValueError as exc:
            print_fn(f"Illegal move: {exc}")
            continue

        print_fn(render_board(state.board()))
        if state.is_terminal():
            print_fn(outcome_line(state))
        else:
            print_fn(status_line(state))

    return 0


def run_replay(
    actions: Iterable[int],
    *,
    pause: bool = False,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
) -> int:
    try:
        import great_kingdom_core as core  # type: ignore[import-untyped]
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "great_kingdom_core is not installed. Build it first with:\n"
            "  cd rust/great_kingdom_core\n"
            "  ../../.venv/bin/python -m maturin develop\n"
            "  cd ../.."
        ) from exc

    state = cast(GameStateProtocol, core.GameState())
    return replay_actions(state, actions, pause=pause, input_fn=input_fn, print_fn=print_fn)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="great-kingdom-play",
        description="Manual self-play CLI backed by the Rust Great Kingdom rules engine.",
    )
    parser.add_argument(
        "--replay-actions",
        help="replay comma/space separated action indexes or coordinates, e.g. '20,68,C3'",
    )
    parser.add_argument(
        "--pause",
        action="store_true",
        help="wait for Enter between replay moves",
    )
    return parser


def main() -> NoReturn:
    args = build_parser().parse_args()
    if args.replay_actions is not None:
        try:
            actions = parse_action_sequence(args.replay_actions)
        except ValueError as exc:
            raise SystemExit(f"Replay input error: {exc}") from exc
        raise SystemExit(run_replay(actions, pause=args.pause))
    raise SystemExit(run_repl())


if __name__ == "__main__":
    main()
