import pytest
from great_kingdom_ai.cli import (
    BOARD_CELLS,
    PASS_ACTION,
    CliExit,
    format_legal_actions,
    index_to_coordinate,
    outcome_line,
    parse_command,
    render_board,
    status_line,
)


class FakeState:
    def current_player(self) -> int:
        return 1

    def blue_used(self) -> int:
        return 4

    def orange_used(self) -> int:
        return 3

    def previous_pass(self) -> bool:
        return False

    def territory_scores(self) -> tuple[int, int]:
        return (5, 2)

    def winner(self) -> int | None:
        return 1

    def end_reason(self) -> int | None:
        return 3


def test_parse_coordinate_commands() -> None:
    assert parse_command("A1").action == 0
    assert parse_command("i9").action == BOARD_CELLS - 1
    assert parse_command("5 5").action == 40
    assert parse_command("5,5").action == 40


def test_parse_pass_and_raw_index() -> None:
    assert parse_command("pass").action == PASS_ACTION
    assert parse_command("p").action == PASS_ACTION
    assert parse_command("i 81").action == PASS_ACTION
    assert parse_command("0").action == 0


def test_parse_meta_commands() -> None:
    assert parse_command("help").show_help
    assert parse_command("legal").show_legal
    assert parse_command("board").show_board
    with pytest.raises(CliExit):
        parse_command("quit")


def test_parse_rejects_invalid_input() -> None:
    with pytest.raises(ValueError, match="row must be between 1 and 9"):
        parse_command("A10")
    with pytest.raises(ValueError, match="action index must be between 0 and 81"):
        parse_command("82")
    with pytest.raises(ValueError, match="unknown command"):
        parse_command("north")


def test_render_board_uses_expected_labels() -> None:
    board = [0] * BOARD_CELLS
    board[0] = 1
    board[1] = 2
    board[40] = 3

    rendered = render_board(board)

    assert "A B C D E F G H I" in rendered
    assert " 1  B O . . . . . . ." in rendered
    assert " 5  . . . . N . . . ." in rendered


def test_format_legal_actions_groups_coordinates_and_pass() -> None:
    assert index_to_coordinate(0) == "A1"
    assert index_to_coordinate(80) == "I9"
    assert format_legal_actions([0, 1, PASS_ACTION]) == "A1 B1\nPASS"


def test_status_line_shows_territory_scores() -> None:
    assert status_line(FakeState()) == (
        "Turn: Blue | Blue used: 4/40 | Orange used: 3/40 | "
        "Territory: Blue 5, Orange 2 | Previous pass: False"
    )


def test_outcome_line_shows_final_territory_scores() -> None:
    assert outcome_line(FakeState()) == (
        "Game over: Blue wins by consecutive passes. Territory: Blue 5, Orange 2."
    )
