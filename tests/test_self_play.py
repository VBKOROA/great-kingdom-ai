import importlib.util
import random

import great_kingdom_ai.self_play as self_play_module
import pytest
from great_kingdom_ai.self_play import (
    MctsSelfPlayConfig,
    choose_random_legal_action,
    play_mcts_game,
    play_mcts_games_batched,
    play_random_game,
    play_random_games,
    summarize_logs,
)


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

    def feature_planes(self) -> list[float]:
        return [0.0] * (11 * 9 * 9)

    def legal_mask(self) -> list[bool]:
        mask = [False] * 82
        mask[10] = True
        return mask


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


class ScriptedMctsState:
    def __init__(self) -> None:
        self.applied_actions: list[int] = []
        self._current_player = 1
        self._terminal = False

    def current_player(self) -> int:
        return self._current_player

    def legal_actions(self) -> list[int]:
        return [1, 2, 81]

    def legal_mask(self) -> list[bool]:
        mask = [False] * 82
        for action in self.legal_actions():
            mask[action] = True
        return mask

    def feature_planes(self) -> list[float]:
        features = [0.0] * (11 * 9 * 9)
        features[0] = float(len(self.applied_actions) + 1)
        return features

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


class MultiTurnMctsState(ScriptedMctsState):
    def __init__(self, terminal_after: int) -> None:
        super().__init__()
        self._terminal_after = terminal_after

    def apply_action(self, action_index: int) -> int | None:
        self.applied_actions.append(action_index)
        self._terminal = len(self.applied_actions) >= self._terminal_after
        return 1 if self._terminal else None


class FakeMctsResult:
    def __init__(self, visits: list[int]) -> None:
        self._visits = visits

    def selected_action(self) -> int | None:
        return None

    def visit_counts(self) -> list[int]:
        return self._visits


class FakeGumbelResult(FakeMctsResult):
    def __init__(self, selected: int, policy: list[float]) -> None:
        visits = [0] * 82
        visits[selected] = 1
        super().__init__(visits)
        self._selected = selected
        self._policy = policy

    def selected_action(self) -> int | None:
        return self._selected

    def policy_target(self) -> list[float]:
        return self._policy


class FakeMctsSearch:
    def __init__(self, visits: list[int]) -> None:
        self.visits = visits
        self.noisy_priors: list[float] | None = None
        self.search_calls = 0

    def search(self, state: ScriptedMctsState) -> FakeMctsResult:
        self.search_calls += 1
        return FakeMctsResult(self.visits)

    def search_with_priors(
        self,
        state: ScriptedMctsState,
        priors: list[float],
    ) -> FakeMctsResult:
        self.noisy_priors = priors
        return FakeMctsResult(self.visits)

    def search_with_priors_and_evaluator(
        self,
        state: ScriptedMctsState,
        priors: list[float],
        evaluator,
        leaf_batch_size: int = 8,
    ) -> FakeMctsResult:
        self.noisy_priors = priors
        policies, values = evaluator(_SingleStateEvalRequest(state))
        assert len(policies) == 1
        assert len(values) == 1
        assert leaf_batch_size == 8
        return FakeMctsResult(self.visits)

    def set_simulations(self, simulations: int) -> None:
        del simulations


class FakeGumbelSearch(FakeMctsSearch):
    def __init__(self, selected: int, policy: list[float]) -> None:
        super().__init__([0] * 82)
        self._selected = selected
        self._policy = policy
        self.root_logits: list[float] | None = None

    def search_with_logits(
        self,
        state: ScriptedMctsState,
        policy_logits: list[float],
    ) -> FakeGumbelResult:
        del state
        self.root_logits = policy_logits
        return FakeGumbelResult(self._selected, self._policy)

    def search_with_logits_and_evaluator(
        self,
        state: ScriptedMctsState,
        policy_logits: list[float],
        evaluator,
        leaf_batch_size: int = 8,
    ) -> FakeGumbelResult:
        del leaf_batch_size
        self.root_logits = policy_logits
        policies, values = evaluator(_SingleStateEvalRequest(state))
        assert len(policies) == 1
        assert len(values) == 1
        return FakeGumbelResult(self._selected, self._policy)


class BudgetRecordingMctsSearch(FakeMctsSearch):
    def __init__(self, visits: list[int]) -> None:
        super().__init__(visits)
        self.simulation_budgets: list[int] = []

    def set_simulations(self, simulations: int) -> None:
        self.simulation_budgets.append(simulations)


class FakeCoreBatch:
    def __init__(self, game_count: int) -> None:
        self._game_count = game_count
        self._terminal = [False] * game_count
        self._winners: list[int | None] = [None] * game_count
        self._end_reasons: list[int | None] = [None] * game_count
        self.applied_actions: list[int | None] = []
        self.leaf_batch_sizes: list[int] = []

    def len(self) -> int:
        return self._game_count

    def active_count(self) -> int:
        return len(self.active_game_indexes())

    def active_game_indexes(self) -> list[int]:
        return [index for index, terminal in enumerate(self._terminal) if not terminal]

    def active_eval_request(self):
        active = self.active_game_indexes()

        class Request:
            def feature_planes(self) -> list[list[float]]:
                return [[float(index + 1)] + [0.0] * (11 * 9 * 9 - 1) for index in active]

            def legal_masks(self) -> list[list[bool]]:
                masks = []
                for _index in active:
                    mask = [False] * 82
                    mask[1] = True
                    masks.append(mask)
                return masks

        return Request()

    def current_players(self) -> list[int]:
        return [1] * self._game_count

    def is_terminal(self) -> list[bool]:
        return list(self._terminal)

    def winners(self) -> list[int | None]:
        return list(self._winners)

    def end_reasons(self) -> list[int | None]:
        return list(self._end_reasons)

    def territory_scores(self) -> list[tuple[int, int]]:
        return [(0, 0)] * self._game_count

    def search_active_with_priors(self, priors: list[list[float]]):
        assert len(priors) == self.active_count()
        visits = [0] * 82
        visits[1] = 1
        results = [None] * self._game_count
        for index in self.active_game_indexes():
            results[index] = FakeMctsResult(visits)
        return results

    def search_active_with_priors_and_evaluator(
        self,
        priors: list[list[float]],
        evaluator,
        leaf_batch_size: int = 8,
    ):
        self.leaf_batch_sizes.append(leaf_batch_size)
        policies, values = evaluator(self.active_eval_request())
        assert len(policies) == self.active_count()
        assert len(values) == self.active_count()
        return self.search_active_with_priors(priors)

    def search_active_with_logits(self, policy_logits: list[list[float]]):
        assert len(policy_logits) == self.active_count()
        policy = [0.0] * 82
        policy[1] = 1.0
        results = [None] * self._game_count
        for index in self.active_game_indexes():
            results[index] = FakeGumbelResult(1, policy)
        return results

    def search_active_with_logits_and_evaluator(
        self,
        policy_logits: list[list[float]],
        evaluator,
        leaf_batch_size: int = 8,
    ):
        self.leaf_batch_sizes.append(leaf_batch_size)
        policies, values = evaluator(self.active_eval_request())
        assert len(policies) == self.active_count()
        assert len(values) == self.active_count()
        return self.search_active_with_logits(policy_logits)

    def apply_actions(self, actions: list[int | None]) -> list[int | None]:
        self.applied_actions.extend(actions)
        for index, action in enumerate(actions):
            if action is not None:
                self._terminal[index] = True
                self._winners[index] = 1
                self._end_reasons[index] = 1
        return self._winners

    def set_simulations(self, simulations: list[int | None]) -> None:
        del simulations


class _SingleStateEvalRequest:
    def __init__(self, state: ScriptedMctsState) -> None:
        self._state = state

    def feature_planes(self) -> list[list[float]]:
        return [self._state.feature_planes()]

    def legal_masks(self) -> list[list[bool]]:
        return [self._state.legal_mask()]

    def current_players(self) -> list[int]:
        return [self._state.current_player()]


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


def test_play_mcts_game_records_policy_and_final_value_targets() -> None:
    visits = [0] * 82
    visits[1] = 2
    visits[2] = 8
    state = ScriptedMctsState()
    search = FakeMctsSearch(visits)

    log, samples = play_mcts_game(
        seed=41,
        state=state,
        search=search,
        config=MctsSelfPlayConfig(
            max_turns=5,
            temperature_turns=0,
            root_noise=False,
        ),
    )

    assert state.applied_actions == [2]
    assert log.moves[0].action == 2
    assert len(samples) == 1
    assert samples[0].features.shape == (11, 9, 9)
    assert samples[0].policy[1] == pytest.approx(0.2)
    assert samples[0].policy[2] == pytest.approx(0.8)
    assert samples[0].value == 1.0


def test_play_mcts_game_uses_gumbel_policy_target_and_selected_action() -> None:
    policy = [0.0] * 82
    policy[1] = 0.1
    policy[2] = 0.9
    logits = [-10.0] * 82
    logits[1] = 4.0
    logits[2] = 9.0
    state = ScriptedMctsState()
    search = FakeGumbelSearch(selected=1, policy=policy)

    log, samples = play_mcts_game(
        seed=41,
        state=state,
        search=search,
        config=MctsSelfPlayConfig(
            search_backend="gumbel",
            max_turns=5,
            temperature_turns=10,
            sampling_temperature=1.0,
            root_noise=True,
        ),
        prior_provider=lambda current_state: logits,
    )

    assert search.root_logits == logits
    assert state.applied_actions == [1]
    assert log.moves[0].action == 1
    assert len(samples) == 1
    assert samples[0].policy[1] == pytest.approx(0.1)
    assert samples[0].policy[2] == pytest.approx(0.9)


def test_mcts_self_play_config_samples_only_opening_turns_by_default() -> None:
    assert MctsSelfPlayConfig().temperature_turns == 10


def test_play_mcts_game_can_skip_fast_playout_cap_turns() -> None:
    visits = [0] * 82
    visits[1] = 1
    state = MultiTurnMctsState(terminal_after=3)
    search = BudgetRecordingMctsSearch(visits)

    log, samples = play_mcts_game(
        seed=1,
        state=state,
        search=search,
        config=MctsSelfPlayConfig(
            max_turns=5,
            temperature_turns=0,
            root_noise=False,
            playout_cap_randomization=True,
            playout_cap_full_search_fraction=0.01,
            playout_cap_full_simulations=100,
            playout_cap_fast_simulations=16,
        ),
    )

    assert len(log.moves) == 3
    assert samples == []
    assert search.simulation_budgets == [16, 16, 16]


def test_play_mcts_game_keeps_full_playout_cap_turns_as_samples() -> None:
    visits = [0] * 82
    visits[1] = 1
    state = MultiTurnMctsState(terminal_after=2)
    search = BudgetRecordingMctsSearch(visits)

    log, samples = play_mcts_game(
        seed=3,
        state=state,
        search=search,
        config=MctsSelfPlayConfig(
            max_turns=5,
            temperature_turns=0,
            root_noise=False,
            playout_cap_randomization=True,
            playout_cap_full_search_fraction=1.0,
            playout_cap_full_simulations=100,
            playout_cap_fast_simulations=16,
        ),
    )

    assert len(log.moves) == 2
    assert len(samples) == 2
    assert search.simulation_budgets == [100, 100]


def test_play_mcts_games_batched_evaluates_root_priors_together() -> None:
    visits = [0] * 82
    visits[1] = 1
    batch_sizes: list[int] = []

    def batch_prior_provider(states) -> list[list[float]]:
        batch_sizes.append(len(states))
        priors = [0.0] * 82
        priors[1] = 1.0
        return [priors for _ in states]

    results = play_mcts_games_batched(
        seeds=[11, 12],
        search_factory=lambda: FakeMctsSearch(visits),
        state_factory=lambda: MultiTurnMctsState(terminal_after=1),
        config=MctsSelfPlayConfig(
            max_turns=5,
            temperature_turns=0,
            root_noise=False,
        ),
        prior_provider=batch_prior_provider,
    )

    logs = [log for log, _samples in results]
    samples = [sample for _log, game_samples in results for sample in game_samples]
    assert [log.seed for log in logs] == [11, 12]
    assert [log.moves[0].action for log in logs] == [1, 1]
    assert len(samples) == 2
    assert batch_sizes == [2]


def test_play_mcts_games_batched_uses_core_batch_when_available(monkeypatch) -> None:
    fake_batch = FakeCoreBatch(game_count=2)
    batch_sizes: list[int] = []

    monkeypatch.setattr(self_play_module, "_can_create_core_self_play_batch", lambda *_args: True)
    monkeypatch.setattr(
        self_play_module,
        "create_core_mcts_self_play_batch",
        lambda **_kwargs: fake_batch,
    )

    def batch_prior_provider(states) -> list[list[float]]:
        batch_sizes.append(len(states))
        priors = [0.0] * 82
        priors[1] = 1.0
        return [priors for _state in states]

    results = play_mcts_games_batched(
        seeds=[21, 22],
        search_factory=lambda: FakeMctsSearch([0] * 82),
        config=MctsSelfPlayConfig(
            max_turns=5,
            temperature_turns=0,
            root_noise=False,
        ),
        prior_provider=batch_prior_provider,
    )

    logs = [log for log, _samples in results]
    samples = [sample for _log, game_samples in results for sample in game_samples]
    assert batch_sizes == [2]
    assert fake_batch.applied_actions == [1, 1]
    assert [log.seed for log in logs] == [21, 22]
    assert [log.moves[0].action for log in logs] == [1, 1]
    assert len(samples) == 2


def test_play_mcts_games_batched_passes_leaf_evaluator_to_core_batch(monkeypatch) -> None:
    fake_batch = FakeCoreBatch(game_count=2)
    evaluator_batch_sizes: list[int] = []

    monkeypatch.setattr(self_play_module, "_can_create_core_self_play_batch", lambda *_args: True)
    monkeypatch.setattr(
        self_play_module,
        "create_core_mcts_self_play_batch",
        lambda **_kwargs: fake_batch,
    )

    def batch_prior_provider(states) -> list[list[float]]:
        priors = [0.0] * 82
        priors[1] = 1.0
        return [priors for _state in states]

    def evaluator_provider(states) -> tuple[list[list[float]], list[float]]:
        evaluator_batch_sizes.append(len(states))
        policy = [0.0] * 82
        policy[1] = 1.0
        return [policy for _state in states], [0.25 for _state in states]

    play_mcts_games_batched(
        seeds=[31, 32],
        search_factory=lambda: FakeMctsSearch([0] * 82),
        config=MctsSelfPlayConfig(
            max_turns=5,
            temperature_turns=0,
            root_noise=False,
            leaf_batch_size=32,
        ),
        prior_provider=batch_prior_provider,
        evaluator_provider=evaluator_provider,
    )

    assert evaluator_batch_sizes == [2]
    assert fake_batch.leaf_batch_sizes == [32]


def test_play_mcts_games_batched_can_use_core_request_fast_path(monkeypatch) -> None:
    fake_batch = FakeCoreBatch(game_count=2)
    prior_batch_sizes: list[int] = []
    evaluator_batch_sizes: list[int] = []

    monkeypatch.setattr(self_play_module, "_can_create_core_self_play_batch", lambda *_args: True)
    monkeypatch.setattr(
        self_play_module,
        "create_core_mcts_self_play_batch",
        lambda **_kwargs: fake_batch,
    )

    def feature_batch_prior_provider(features, masks) -> list[list[float]]:
        assert len(features) == len(masks)
        prior_batch_sizes.append(len(features))
        priors = [0.0] * 82
        priors[1] = 1.0
        return [priors for _features in features]

    def request_evaluator_provider(request) -> tuple[list[list[float]], list[float]]:
        features = request.feature_planes()
        masks = request.legal_masks()
        assert len(features) == len(masks)
        evaluator_batch_sizes.append(len(features))
        policy = [0.0] * 82
        policy[1] = 1.0
        return [policy for _features in features], [0.25 for _features in features]

    play_mcts_games_batched(
        seeds=[41, 42],
        search_factory=lambda: FakeMctsSearch([0] * 82),
        config=MctsSelfPlayConfig(
            max_turns=5,
            temperature_turns=0,
            root_noise=False,
            leaf_batch_size=16,
        ),
        feature_batch_prior_provider=feature_batch_prior_provider,
        request_evaluator_provider=request_evaluator_provider,
    )

    assert prior_batch_sizes == [2]
    assert evaluator_batch_sizes == [2]
    assert fake_batch.leaf_batch_sizes == [16]


def test_play_mcts_games_batched_uses_gumbel_core_batch(monkeypatch) -> None:
    fake_batch = FakeCoreBatch(game_count=2)
    seen_backend: list[str] = []
    root_batch_sizes: list[int] = []

    def can_create(search_backend: str = "mcts") -> bool:
        seen_backend.append(search_backend)
        return search_backend == "gumbel"

    monkeypatch.setattr(self_play_module, "_can_create_core_self_play_batch", can_create)
    monkeypatch.setattr(
        self_play_module,
        "create_core_self_play_batch_backend",
        lambda _config, **_kwargs: fake_batch,
    )

    def feature_batch_prior_provider(features, masks) -> list[list[float]]:
        assert len(features) == len(masks)
        root_batch_sizes.append(len(features))
        logits = [-1.0] * 82
        logits[1] = 5.0
        return [logits for _features in features]

    results = play_mcts_games_batched(
        seeds=[51, 52],
        search_factory=lambda: FakeGumbelSearch(1, [0.0] * 82),
        config=MctsSelfPlayConfig(
            search_backend="gumbel",
            max_turns=5,
            temperature_turns=10,
            root_noise=True,
        ),
        feature_batch_prior_provider=feature_batch_prior_provider,
    )

    logs = [log for log, _samples in results]
    assert seen_backend == ["gumbel"]
    assert root_batch_sizes == [2]
    assert fake_batch.applied_actions == [1, 1]
    assert [log.moves[0].action for log in logs] == [1, 1]


def test_play_mcts_game_applies_root_noise_only_when_enabled() -> None:
    visits = [0] * 82
    visits[1] = 1
    state = ScriptedMctsState()
    search = FakeMctsSearch(visits)

    play_mcts_game(
        seed=43,
        state=state,
        search=search,
        config=MctsSelfPlayConfig(root_noise=True),
    )

    assert search.noisy_priors is not None
    assert sum(search.noisy_priors) == pytest.approx(1.0)
    assert search.noisy_priors[1] > 0.0
    assert search.noisy_priors[2] > 0.0
    assert search.noisy_priors[81] > 0.0
    assert search.search_calls == 0


def test_play_mcts_game_can_use_model_root_priors_without_noise() -> None:
    visits = [0] * 82
    visits[1] = 1
    state = ScriptedMctsState()
    search = FakeMctsSearch(visits)
    priors = [0.0] * 82
    priors[1] = 0.2
    priors[2] = 0.7
    priors[81] = 0.1

    play_mcts_game(
        seed=47,
        state=state,
        search=search,
        config=MctsSelfPlayConfig(root_noise=False),
        prior_provider=lambda current_state: priors,
    )

    assert search.noisy_priors == priors
    assert search.search_calls == 0


def test_play_mcts_game_passes_leaf_evaluator_when_available() -> None:
    visits = [0] * 82
    visits[1] = 1
    state = ScriptedMctsState()
    search = FakeMctsSearch(visits)
    seen_players: list[int] = []

    def evaluator_provider(states) -> tuple[list[list[float]], list[float]]:
        seen_players.extend(state.current_player() for state in states)
        policy = [0.0] * 82
        policy[1] = 1.0
        return [policy for _state in states], [0.5 for _state in states]

    play_mcts_game(
        seed=49,
        state=state,
        search=search,
        config=MctsSelfPlayConfig(root_noise=False),
        prior_provider=lambda current_state: [1.0 / 82.0] * 82,
        evaluator_provider=evaluator_provider,
    )

    assert seen_players == [1]
    assert search.noisy_priors == [1.0 / 82.0] * 82
    assert search.search_calls == 0


def test_play_random_game_guard_rejects_non_terminating_games() -> None:
    state = NonTerminalState()

    with pytest.raises(RuntimeError, match="max_turns=1"):
        play_random_game(seed=29, state=state, max_turns=1)


def test_summarize_logs_counts_games_moves_and_winners() -> None:
    logs = [
        play_random_game(seed=31, state=ScriptedTerminalState()),
        play_random_game(seed=37, state=ScriptedTerminalState()),
    ]

    summary = summarize_logs(logs)

    assert summary.games == 2
    assert summary.total_moves == 2
    assert summary.max_moves == 1
    assert summary.blue_wins == 2
    assert summary.orange_wins == 0
    assert summary.to_dict() == {
        "games": 2,
        "total_moves": 2,
        "max_moves": 1,
        "blue_wins": 2,
        "orange_wins": 0,
    }


@pytest.mark.skipif(
    importlib.util.find_spec("great_kingdom_core") is None,
    reason="great_kingdom_core extension is not installed",
)
def test_random_self_play_smoke_with_real_core() -> None:
    logs = play_random_games(seeds=list(range(5)), max_turns=200, prefer_place=True)

    assert len(logs) == 5
    assert all(log.moves for log in logs)
    assert all(len(log.moves) <= 200 for log in logs)
    assert all(log.winner in {1, 2} for log in logs)
    assert all(log.end_reason in {1, 2, 3} for log in logs)
