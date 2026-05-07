import pytest
from great_kingdom_ai.cli import (
    BOARD_CELLS,
    PASS_ACTION,
    CliExit,
    format_legal_actions,
    index_to_coordinate,
    move_line,
    outcome_line,
    parse_action_sequence,
    parse_command,
    parse_player,
    play_against_model,
    render_board,
    replay_actions,
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


class FakeReplayState:
    def __init__(self) -> None:
        self._board = [0] * BOARD_CELLS
        self._current_player = 1
        self._winner: int | None = None
        self._end_reason: int | None = None
        self.applied: list[int] = []

    def board(self) -> list[int]:
        return list(self._board)

    def current_player(self) -> int:
        return self._current_player

    def blue_used(self) -> int:
        return sum(1 for cell in self._board if cell == 1)

    def orange_used(self) -> int:
        return sum(1 for cell in self._board if cell == 2)

    def previous_pass(self) -> bool:
        return False

    def territory_scores(self) -> tuple[int, int]:
        return (0, 0)

    def winner(self) -> int | None:
        return self._winner

    def end_reason(self) -> int | None:
        return self._end_reason

    def legal_actions(self) -> list[int]:
        return list(range(BOARD_CELLS + 1))

    def apply_action(self, action_index: int) -> int | None:
        self.applied.append(action_index)
        self._board[action_index] = self._current_player
        self._current_player = 2 if self._current_player == 1 else 1
        if len(self.applied) == 2:
            self._winner = 2
            self._end_reason = 1
        return self._winner

    def is_terminal(self) -> bool:
        return self._winner is not None


class FakeModelPlayState(FakeReplayState):
    def __init__(self, *, terminal_after: int = 2) -> None:
        super().__init__()
        self._terminal_after = terminal_after

    def feature_planes(self) -> list[float]:
        return [0.0] * (11 * 9 * 9)

    def legal_mask(self) -> list[bool]:
        return [True] * (BOARD_CELLS + 1)

    def apply_action(self, action_index: int) -> int | None:
        self.applied.append(action_index)
        if action_index < BOARD_CELLS:
            self._board[action_index] = self._current_player
        self._current_player = 2 if self._current_player == 1 else 1
        if len(self.applied) >= self._terminal_after:
            self._winner = 1
            self._end_reason = 3
        return self._winner


class FakeModelPlayer:
    def __init__(self, action: int) -> None:
        self.action = action
        self.seen_players: list[int] = []

    def select_action(self, state: FakeModelPlayState) -> int:
        self.seen_players.append(state.current_player())
        return self.action


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


def test_parse_action_sequence_accepts_indexes_and_coordinates() -> None:
    assert parse_action_sequence("20, F8 C3") == [20, 68, 20]


def test_parse_player_accepts_side_aliases() -> None:
    assert parse_player("blue") == 1
    assert parse_player("B") == 1
    assert parse_player("2") == 2
    assert parse_player("orange") == 2
    with pytest.raises(ValueError, match="player must be blue or orange"):
        parse_player("green")


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


def test_replay_actions_prints_each_move_and_board() -> None:
    state = FakeReplayState()
    lines: list[str] = []

    result = replay_actions(state, [20, 68], print_fn=lines.append)

    assert result == 0
    assert state.applied == [20, 68]
    assert move_line(turn=0, player=1, action=20) in lines
    assert move_line(turn=1, player=2, action=68) in lines
    assert lines[-1] == (
        "Game over: Orange wins by opponent castle destroyed. Territory: Blue 0, Orange 0."
    )


def test_play_against_model_applies_model_move_after_human_move() -> None:
    state = FakeModelPlayState()
    model_player = FakeModelPlayer(action=10)
    lines: list[str] = []

    result = play_against_model(
        state,
        model_player,
        human_player=1,
        input_fn=lambda prompt: "A1",
        print_fn=lines.append,
    )

    assert result == 0
    assert state.applied == [0, 10]
    assert model_player.seen_players == [2]
    assert move_line(turn=0, player=1, action=0) in lines
    assert move_line(turn=1, player=2, action=10) in lines
    assert "Orange model thinking..." in lines


def test_play_against_model_can_open_as_blue_model() -> None:
    state = FakeModelPlayState(terminal_after=1)
    model_player = FakeModelPlayer(action=20)
    lines: list[str] = []

    result = play_against_model(
        state,
        model_player,
        human_player=2,
        input_fn=lambda prompt: pytest.fail(f"unexpected prompt: {prompt}"),
        print_fn=lines.append,
    )

    assert result == 0
    assert state.applied == [20]
    assert model_player.seen_players == [1]
    assert move_line(turn=0, player=1, action=20) in lines
