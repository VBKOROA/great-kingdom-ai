import importlib.util

import pytest


@pytest.mark.skipif(
    importlib.util.find_spec("great_kingdom_core") is None,
    reason="great_kingdom_core extension is not installed",
)
def test_mcts_search_returns_legal_action_and_visit_counts() -> None:
    import great_kingdom_core as core  # type: ignore[import-untyped]

    state = core.GameState()
    search = core.MctsSearch(simulations=8, c_puct=1.5)

    result = search.search(state)

    assert result.selected_action() in state.legal_actions()
    assert len(result.visit_counts()) == core.action_space()
    assert sum(result.visit_counts()) == 8
    assert result.visit_counts()[40] == 0


@pytest.mark.skipif(
    importlib.util.find_spec("great_kingdom_core") is None,
    reason="great_kingdom_core extension is not installed",
)
def test_mcts_config_rejects_invalid_c_puct() -> None:
    import great_kingdom_core as core  # type: ignore[import-untyped]

    with pytest.raises(ValueError, match="c_puct"):
        core.MctsSearch(c_puct=-1.0)
