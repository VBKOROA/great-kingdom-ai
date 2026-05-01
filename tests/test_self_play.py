import random

import pytest

from great_kingdom_ai.self_play import choose_random_legal_action


class LegalOnlyState:
    def __init__(self, legal_actions: list[int]) -> None:
        self._legal_actions = legal_actions

    def legal_actions(self) -> list[int]:
        return list(self._legal_actions)


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
