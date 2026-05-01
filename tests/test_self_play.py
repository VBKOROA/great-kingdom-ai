import random

import pytest

from great_kingdom_ai.self_play import choose_random_legal_action, play_random_game


class LegalOnlyState:
    def __init__(self, legal_actions: list[int]) -> None:
        self._legal_actions = legal_actions

    def legal_actions(self) -> list[int]:
        return list(self._legal_actions)


class ScriptedTerminalState:
    def __init__(self) -> None:
        self.applied_actions: list[int] = []
        self._current_player = 1
        self._terminal = False

    def current_player(self) -> int:
        return self._current_player

    def legal_actions(self) -> list[int]:
        return [10]

    def apply_action(self, action_index: int) -> int | None:
        self.applied_actions.append(action_index)
        self._terminal = True
        return 1

    def is_terminal(self) -> bool:
        return self._terminal

    def winner(self) -> int | None:
        return 1 if self._terminal else None

    def end_reason(self) -> int | None:
        return 1 if self._terminal else None

    def territory_scores(self) -> tuple[int, int]:
        return (0, 0)


class NonTerminalState:
    def current_player(self) -> int:
        return 1

    def legal_actions(self) -> list[int]:
        return [81]

    def apply_action(self, action_index: int) -> int | None:
        return None

    def is_terminal(self) -> bool:
        return False

    def winner(self) -> int | None:
        return None

    def end_reason(self) -> int | None:
        return None

    def territory_scores(self) -> tuple[int, int]:
        return (0, 0)


def test_random_selector_only_returns_legal_actions() -> None:
    state = LegalOnlyState([2, 5, 81])
    rng = random.Random(7)

    chosen = [choose_random_legal_action(state, rng) for _ in range(20)]

    assert set(chosen) <= {2, 5, 81}


def test_random_selector_can_prefer_place_actions_for_smoke_depth() -> None:
    state = LegalOnlyState([4, 8, 81])
    rng = random.Random(11)

    chosen = [choose_random_legal_action(state, rng, prefer_place=True) for _ in range(20)]

    assert set(chosen) <= {4, 8}


def test_random_selector_keeps_pass_when_it_is_the_only_legal_action() -> None:
    state = LegalOnlyState([81])
    rng = random.Random(13)

    assert choose_random_legal_action(state, rng, prefer_place=True) == 81


def test_random_selector_rejects_states_without_legal_actions() -> None:
    state = LegalOnlyState([])
    rng = random.Random(17)

    with pytest.raises(ValueError, match="no legal actions"):
        choose_random_legal_action(state, rng)


def test_play_random_game_records_reproducible_log() -> None:
    state = ScriptedTerminalState()

    log = play_random_game(seed=23, state=state)

    assert state.applied_actions == [10]
    assert log.seed == 23
    assert log.moves[0].turn == 0
    assert log.moves[0].player == 1
    assert log.moves[0].action == 10
    assert log.winner == 1
    assert log.end_reason == 1
    assert log.territory_scores == (0, 0)
    assert log.to_dict()["moves"] == [{"turn": 0, "player": 1, "action": 10}]


def test_play_random_game_guard_rejects_non_terminating_games() -> None:
    state = NonTerminalState()

    with pytest.raises(RuntimeError, match="max_turns=1"):
        play_random_game(seed=29, state=state, max_turns=1)
