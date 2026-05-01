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


@pytest.mark.skipif(
    importlib.util.find_spec("great_kingdom_core") is None,
    reason="great_kingdom_core extension is not installed",
)
def test_mcts_search_with_evaluator_batches_leaf_requests() -> None:
    import great_kingdom_core as core  # type: ignore[import-untyped]

    state = core.GameState()
    search = core.MctsSearch(simulations=4, c_puct=1.5)
    batch_sizes: list[int] = []

    def evaluator(request):
        batch_sizes.append(request.len())
        policies = []
        values = []
        for mask in request.legal_masks():
            policy = [1.0 if is_legal else 0.0 for is_legal in mask]
            policy[40] = 100.0
            policies.append(policy)
            values.append(0.0)
        return policies, values

    result = search.search_with_evaluator(state, evaluator, leaf_batch_size=2)

    assert batch_sizes[0] == 1
    assert sum(result.visit_counts()) == 4
    assert result.selected_action() in state.legal_actions()
    assert result.visit_counts()[40] == 0


@pytest.mark.skipif(
    importlib.util.find_spec("great_kingdom_core") is None,
    reason="great_kingdom_core extension is not installed",
)
def test_mcts_search_with_evaluator_rejects_bad_policy_shape() -> None:
    import great_kingdom_core as core  # type: ignore[import-untyped]

    state = core.GameState()
    search = core.MctsSearch(simulations=1, c_puct=1.5)

    def evaluator(request):
        return [[1.0]], [0.0] * request.len()

    with pytest.raises(ValueError, match="policy row"):
        search.search_with_evaluator(state, evaluator)


@pytest.mark.skipif(
    importlib.util.find_spec("great_kingdom_core") is None,
    reason="great_kingdom_core extension is not installed",
)
def test_mcts_self_play_batch_advances_active_games_with_batched_priors() -> None:
    import great_kingdom_core as core  # type: ignore[import-untyped]

    batch = core.MctsSelfPlayBatch(game_count=2, simulations=2, c_puct=1.5)
    request = batch.active_eval_request()
    priors = [
        [1.0 if is_legal else 0.0 for is_legal in mask]
        for mask in request.legal_masks()
    ]

    results = batch.play_turns_with_priors(priors)

    assert batch.len() == 2
    assert batch.active_count() == 2
    assert len(results) == 2
    assert all(result is not None for result in results)
    assert list(batch.current_players()) == [2, 2]
    assert batch.winners() == [None, None]


@pytest.mark.skipif(
    importlib.util.find_spec("great_kingdom_core") is None,
    reason="great_kingdom_core extension is not installed",
)
def test_mcts_self_play_batch_rejects_prior_count_mismatch() -> None:
    import great_kingdom_core as core  # type: ignore[import-untyped]

    batch = core.MctsSelfPlayBatch(game_count=2, simulations=2, c_puct=1.5)

    with pytest.raises(ValueError, match="prior rows"):
        batch.play_turns_with_priors([[1.0] * core.action_space()])


@pytest.mark.skipif(
    importlib.util.find_spec("great_kingdom_core") is None,
    reason="great_kingdom_core extension is not installed",
)
def test_mcts_self_play_batch_combines_leaf_evaluator_requests_across_games() -> None:
    import great_kingdom_core as core  # type: ignore[import-untyped]

    batch = core.MctsSelfPlayBatch(game_count=2, simulations=2, c_puct=1.5)
    request = batch.active_eval_request()
    priors = [
        [1.0 if is_legal else 0.0 for is_legal in mask]
        for mask in request.legal_masks()
    ]
    evaluator_batch_sizes: list[int] = []

    def evaluator(eval_request):
        evaluator_batch_sizes.append(eval_request.len())
        policies = []
        values = []
        for mask in eval_request.legal_masks():
            policies.append([1.0 if is_legal else 0.0 for is_legal in mask])
            values.append(0.0)
        return policies, values

    results = batch.search_active_with_priors_and_evaluator(
        priors,
        evaluator,
        leaf_batch_size=8,
    )

    assert all(result is not None for result in results)
    assert max(evaluator_batch_sizes) >= 2
