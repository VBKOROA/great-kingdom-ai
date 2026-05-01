import importlib.util
import random

import pytest
from great_kingdom_ai.self_play import (
    MctsSelfPlayConfig,
    choose_random_legal_action,
    play_mcts_game,
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


class FakeMctsResult:
    def __init__(self, visits: list[int]) -> None:
        self._visits = visits

    def selected_action(self) -> int | None:
        return None

    def visit_counts(self) -> list[int]:
        return self._visits


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


def test_mcts_self_play_config_samples_only_opening_turns_by_default() -> None:
    assert MctsSelfPlayConfig().temperature_turns == 10


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
