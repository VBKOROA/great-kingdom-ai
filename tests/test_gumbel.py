import importlib.util

import pytest


@pytest.mark.skipif(
    importlib.util.find_spec("great_kingdom_core") is None,
    reason="great_kingdom_core extension is not installed",
)
def test_gumbel_search_constructor_exposes_config() -> None:
    import great_kingdom_core as core  # type: ignore[import-untyped]

    assert core.rayon_thread_count() >= 1

    search = core.GumbelSearch(
        simulations=32,
        max_considered_actions=8,
        c_visit=25.0,
        c_scale=1.5,
        seed=123,
    )

    assert search.simulations() == 32
    assert search.max_considered_actions() == 8
    assert search.c_visit() == 25.0
    assert search.c_scale() == 1.5
    assert search.seed() == 123


@pytest.mark.skipif(
    importlib.util.find_spec("great_kingdom_core") is None,
    reason="great_kingdom_core extension is not installed",
)
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"simulations": 0}, "simulations"),
        ({"max_considered_actions": 0}, "max_considered_actions"),
        ({"c_visit": 0.0}, "c_visit"),
        ({"c_scale": -1.0}, "c_scale"),
    ],
)
def test_gumbel_search_rejects_invalid_config(
    kwargs: dict[str, object],
    match: str,
) -> None:
    import great_kingdom_core as core  # type: ignore[import-untyped]

    with pytest.raises(ValueError, match=match):
        core.GumbelSearch(**kwargs)


@pytest.mark.skipif(
    importlib.util.find_spec("great_kingdom_core") is None,
    reason="great_kingdom_core extension is not installed",
)
def test_gumbel_result_shape_from_logits_skeleton() -> None:
    import great_kingdom_core as core  # type: ignore[import-untyped]

    state = core.GameState()
    search = core.GumbelSearch(simulations=4, max_considered_actions=2, seed=7)
    logits = [0.0] * core.action_space()
    logits[0] = 20.0

    result = search.search_with_logits(state, logits)

    assert result.selected_action() == 0
    assert len(result.policy_target()) == core.action_space()
    assert len(result.visit_counts()) == core.action_space()
    assert sum(result.policy_target()) == pytest.approx(1.0)
    assert result.policy_target()[40] == 0.0


@pytest.mark.skipif(
    importlib.util.find_spec("great_kingdom_core") is None,
    reason="great_kingdom_core extension is not installed",
)
def test_gumbel_self_play_batch_constructor_and_active_request() -> None:
    import great_kingdom_core as core  # type: ignore[import-untyped]

    batch = core.GumbelSelfPlayBatch(game_count=2, simulations=4, seed=99)

    assert batch.len() == 2
    assert batch.active_count() == 2
    assert batch.active_eval_request().len() == 2
    assert list(batch.current_players()) == [1, 1]


@pytest.mark.skipif(
    importlib.util.find_spec("great_kingdom_core") is None,
    reason="great_kingdom_core extension is not installed",
)
def test_gumbel_search_with_evaluator_batches_leaf_logits() -> None:
    import great_kingdom_core as core  # type: ignore[import-untyped]

    request_lengths: list[int] = []

    def evaluator(request: object) -> tuple[list[list[float]], list[float]]:
        request_len = request.len()  # type: ignore[attr-defined]
        request_lengths.append(request_len)
        rows = []
        for _ in range(request_len):
            logits = [-3.0] * core.action_space()
            logits[0] = 4.0
            logits[1] = 2.0
            rows.append(logits)
        return rows, [0.25] * request_len

    state = core.GameState()
    search = core.GumbelSearch(simulations=6, max_considered_actions=4, seed=7)
    root_logits = [0.0] * core.action_space()

    result = search.search_with_logits_and_evaluator(
        state,
        root_logits,
        evaluator,
        leaf_batch_size=4,
    )

    assert max(request_lengths) > 1
    assert sum(result.visit_counts()) == 6


@pytest.mark.skipif(
    importlib.util.find_spec("great_kingdom_core") is None,
    reason="great_kingdom_core extension is not installed",
)
def test_gumbel_self_play_batch_batches_active_game_leaf_eval() -> None:
    import great_kingdom_core as core  # type: ignore[import-untyped]

    request_lengths: list[int] = []

    def evaluator(request: object) -> tuple[list[list[float]], list[float]]:
        request_len = request.len()  # type: ignore[attr-defined]
        request_lengths.append(request_len)
        rows = []
        for _ in range(request_len):
            logits = [-3.0] * core.action_space()
            logits[0] = 4.0
            rows.append(logits)
        return rows, [0.0] * request_len

    batch = core.GumbelSelfPlayBatch(
        game_count=2,
        simulations=4,
        max_considered_actions=2,
        seed=7,
    )
    root_logits = [[0.0] * core.action_space() for _ in range(2)]

    results = batch.search_active_with_logits_and_evaluator(
        root_logits,
        evaluator,
        leaf_batch_size=4,
    )

    assert max(request_lengths) >= 2
    assert [sum(result.visit_counts()) for result in results if result is not None] == [4, 4]
