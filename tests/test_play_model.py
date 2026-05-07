import pytest
from great_kingdom_ai.features import ACTION_SPACE
from great_kingdom_ai.play_model import ModelPlayConfig, select_search_action


class FakeSearchResult:
    def __init__(self, *, selected: int | None, visits: list[int]) -> None:
        self._selected = selected
        self._visits = visits

    def selected_action(self) -> int | None:
        return self._selected

    def visit_counts(self) -> list[int]:
        return self._visits


def test_model_play_config_rejects_invalid_search_settings() -> None:
    with pytest.raises(ValueError, match="gumbel_simulations must be positive"):
        ModelPlayConfig(gumbel_simulations=0)


def test_select_search_action_prefers_selected_legal_action() -> None:
    action = select_search_action(
        FakeSearchResult(selected=4, visits=[0] * ACTION_SPACE),
        [0.0] * ACTION_SPACE,
        [3, 4],
    )

    assert action == 4


def test_select_search_action_falls_back_to_visits_then_priors() -> None:
    visits = [0] * ACTION_SPACE
    visits[3] = 2
    visits[4] = 2
    priors = [0.0] * ACTION_SPACE
    priors[3] = 0.7
    priors[4] = 0.2

    action = select_search_action(
        FakeSearchResult(selected=None, visits=visits),
        priors,
        [3, 4],
    )

    assert action == 3
