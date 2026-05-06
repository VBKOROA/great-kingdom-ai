from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import great_kingdom_ai.evaluate as evaluate_module
import numpy as np
import pytest
from great_kingdom_ai.evaluate import (
    ArenaConfig,
    ArenaGameResult,
    ArenaReport,
    create_core_arena_batch,
    evaluate_state_policy,
    play_arena_game,
    promote_candidate_if_needed,
    run_arena,
    run_arena_batched,
    save_arena_report,
    summarize_arena,
)
from great_kingdom_ai.features import ACTION_SPACE, BOARD_SIZE, FEATURE_CHANNELS
from great_kingdom_ai.self_play import MoveLog


@dataclass(frozen=True)
class FakeNetwork:
    preferred_action: int


@dataclass(frozen=True)
class FakeEvaluation:
    policy: np.ndarray
    policy_logits: np.ndarray
    value: np.ndarray


def test_arena_config_rejects_non_positive_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size must be positive"):
        ArenaConfig(batch_size=0)


class OneMoveState:
    def __init__(self) -> None:
        self.applied_actions: list[int] = []
        self._terminal = False
        self._winner: int | None = None

    def current_player(self) -> int:
        return 1

    def legal_actions(self) -> list[int]:
        return [2, 3, 81]

    def apply_action(self, action_index: int) -> int | None:
        self.applied_actions.append(action_index)
        self._terminal = True
        self._winner = 1 if action_index == 2 else 2
        return self._winner

    def is_terminal(self) -> bool:
        return self._terminal

    def winner(self) -> int | None:
        return self._winner

    def end_reason(self) -> int | None:
        return 1 if self._terminal else None

    def territory_scores(self) -> tuple[int, int]:
        return (0, 0)

    def feature_planes(self) -> list[float]:
        return [0.0] * (FEATURE_CHANNELS * BOARD_SIZE * BOARD_SIZE)

    def legal_mask(self) -> list[bool]:
        mask = [False] * ACTION_SPACE
        for action in self.legal_actions():
            mask[action] = True
        return mask


class TwoMoveState(OneMoveState):
    def current_player(self) -> int:
        return 1 if len(self.applied_actions) % 2 == 0 else 2

    def apply_action(self, action_index: int) -> int | None:
        self.applied_actions.append(action_index)
        if len(self.applied_actions) >= 2:
            self._terminal = True
            self._winner = 2 if action_index == 3 else 1
        return self._winner


class PriorSearchResult:
    def __init__(self, visits: list[int]) -> None:
        self._visits = visits

    def selected_action(self) -> int | None:
        return None

    def visit_counts(self) -> list[int]:
        return self._visits


class PriorSearch:
    def search_with_priors(self, state: OneMoveState, priors: list[float]) -> PriorSearchResult:
        action = max(state.legal_actions(), key=lambda legal_action: priors[legal_action])
        visits = [0] * ACTION_SPACE
        visits[action] = 1
        return PriorSearchResult(visits)

    def search_with_priors_and_evaluator(
        self,
        state: OneMoveState,
        priors: list[float],
        evaluator: Any,
        leaf_batch_size: int = 8,
    ) -> PriorSearchResult:
        del evaluator, leaf_batch_size
        return self.search_with_priors(state, priors)

    def search_with_logits_and_evaluator(
        self,
        state: OneMoveState,
        policy_logits: list[float],
        evaluator: Any,
        root_value: float,
        leaf_batch_size: int = 8,
    ) -> PriorSearchResult:
        del evaluator, root_value, leaf_batch_size
        return self.search_with_priors(state, policy_logits)


class LogitSearch:
    def __init__(self) -> None:
        self.root_logits: list[float] | None = None

    def search_with_logits_and_evaluator(
        self,
        state: OneMoveState,
        policy_logits: list[float],
        evaluator: Any,
        root_value: float,
        leaf_batch_size: int = 8,
    ) -> PriorSearchResult:
        del leaf_batch_size, root_value
        self.root_logits = policy_logits
        policies, values = evaluator(_ArenaEvalRequest(state))
        assert len(policies) == 1
        assert len(values) == 1
        action = max(state.legal_actions(), key=lambda legal_action: policy_logits[legal_action])
        visits = [0] * ACTION_SPACE
        visits[action] = 1
        return PriorSearchResult(visits)


class _ArenaEvalRequest:
    def __init__(self, state: OneMoveState) -> None:
        self._state = state

    def feature_planes(self) -> list[list[float]]:
        return [self._state.feature_planes()]

    def legal_masks(self) -> list[list[bool]]:
        return [self._state.legal_mask()]


class _ArenaBatchEvalRequest:
    def __init__(
        self,
        states: list[OneMoveState],
        game_indexes: list[int],
        *,
        include_game_indexes: bool = True,
    ) -> None:
        self._states = states
        self._game_indexes = game_indexes
        self._include_game_indexes = include_game_indexes

    def feature_planes(self) -> list[list[float]]:
        return [state.feature_planes() for state in self._states]

    def legal_masks(self) -> list[list[bool]]:
        return [state.legal_mask() for state in self._states]

    def current_players(self) -> list[int]:
        return [state.current_player() for state in self._states]

    def game_indexes(self) -> list[int]:
        return self._game_indexes if self._include_game_indexes else []


class FakeArenaBatch:
    missing_leaf_metadata = False

    def __init__(
        self,
        *,
        game_count: int,
        seed_start: int,
        game_index_start: int,
    ) -> None:
        del seed_start
        self.states = [OneMoveState() for _ in range(game_count)]
        self._candidate_players = [
            1 if (game_index_start + index) % 2 == 0 else 2
            for index in range(game_count)
        ]

    def len(self) -> int:
        return len(self.states)

    def active_game_indexes(self) -> list[int]:
        return [
            index for index, state in enumerate(self.states) if not state.is_terminal()
        ]

    def active_eval_request(self) -> _ArenaBatchEvalRequest:
        active = self.active_game_indexes()
        return _ArenaBatchEvalRequest([self.states[index] for index in active], active)

    def current_players(self) -> list[int]:
        return [state.current_player() for state in self.states]

    def candidate_players(self) -> list[int]:
        return self._candidate_players

    def search_active_with_logits_and_evaluator(
        self,
        policy_logits: list[list[float]],
        evaluator: Any,
        root_values: list[float],
        leaf_batch_size: int = 8,
    ) -> list[PriorSearchResult | None]:
        del leaf_batch_size
        active = self.active_game_indexes()
        assert len(policy_logits) == len(active)
        assert len(root_values) == len(active)
        evaluator(
            _ArenaBatchEvalRequest(
                [self.states[index] for index in active],
                active,
                include_game_indexes=not self.missing_leaf_metadata,
            )
        )
        results: list[PriorSearchResult | None] = [None] * self.len()
        for offset, game_index in enumerate(active):
            action = max(
                self.states[game_index].legal_actions(),
                key=lambda legal_action: policy_logits[offset][legal_action],
            )
            visits = [0] * ACTION_SPACE
            visits[action] = 1
            results[game_index] = PriorSearchResult(visits)
        return results

    def apply_actions(self, actions: list[int | None]) -> list[int | None]:
        for index, action in enumerate(actions):
            if action is not None:
                self.states[index].apply_action(action)
        return actions

    def is_terminal(self) -> list[bool]:
        return [state.is_terminal() for state in self.states]

    def winners(self) -> list[int | None]:
        return [state.winner() for state in self.states]

    def end_reasons(self) -> list[int | None]:
        return [state.end_reason() for state in self.states]

    def territory_scores(self) -> list[tuple[int, int]]:
        return [state.territory_scores() for state in self.states]


class UnevenFakeArenaBatch(FakeArenaBatch):
    def __init__(
        self,
        *,
        game_count: int,
        seed_start: int,
        game_index_start: int,
    ) -> None:
        super().__init__(
            game_count=game_count,
            seed_start=seed_start,
            game_index_start=game_index_start,
        )
        if game_count >= 2:
            self.states[1] = TwoMoveState()


def fake_evaluate_feature_batch(
    model: FakeNetwork,
    feature_planes: list[list[float]],
    legal_masks: list[list[bool]],
    *,
    device: Any = None,
) -> FakeEvaluation:
    del feature_planes, device
    policy = np.zeros((len(legal_masks), ACTION_SPACE), dtype=np.float32)
    for row, mask in enumerate(legal_masks):
        legal_actions = [action for action, is_legal in enumerate(mask) if is_legal]
        for action in legal_actions:
            policy[row, action] = 1.0
        policy[row, model.preferred_action] = 8.0
        policy[row] /= policy[row].sum()
    return FakeEvaluation(
        policy=policy,
        policy_logits=np.log(np.maximum(policy, 1.0e-6)).astype(np.float32),
        value=np.zeros((len(legal_masks),), dtype=np.float32),
    )


@pytest.fixture(autouse=True)
def patch_evaluator(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(evaluate_module, "evaluate_feature_batch", fake_evaluate_feature_batch)
    monkeypatch.setattr(
        evaluate_module,
        "evaluate_feature_batch_logits_values",
        fake_evaluate_feature_batch,
    )
    monkeypatch.setattr(
        evaluate_module,
        "evaluate_feature_arrays_logits_values",
        fake_evaluate_feature_batch,
    )


def test_evaluate_state_policy_masks_and_normalizes_legal_actions() -> None:
    state = OneMoveState()

    priors = evaluate_state_policy(FakeNetwork(2), state)

    assert len(priors) == ACTION_SPACE
    assert priors[0] == 0.0
    assert priors[2] > priors[3]
    assert sum(priors) == pytest.approx(1.0)


def test_play_arena_game_uses_candidate_when_candidate_has_current_turn() -> None:
    state = OneMoveState()
    config = ArenaConfig(games=1, max_turns=4, gumbel_simulations=1)

    result = play_arena_game(
        seed=7,
        candidate_model=FakeNetwork(2),
        best_model=FakeNetwork(3),
        candidate_player=1,
        config=config,
        state=state,
        search_factory=PriorSearch,
    )

    assert state.applied_actions == [2]
    assert result.seed == 7
    assert result.candidate_player == 1
    assert result.best_player == 2
    assert result.winner == 1
    assert result.moves == [MoveLog(turn=0, player=1, action=2)]


def test_play_arena_game_uses_gumbel_logits_backend() -> None:
    state = OneMoveState()
    search = LogitSearch()
    config = ArenaConfig(
        games=1,
        max_turns=4,
        gumbel_simulations=1,
    )

    result = play_arena_game(
        seed=7,
        candidate_model=FakeNetwork(2),
        best_model=FakeNetwork(3),
        candidate_player=1,
        config=config,
        state=state,
        search_factory=lambda: search,
    )

    assert search.root_logits is not None
    assert state.applied_actions == [2]
    assert result.moves == [MoveLog(turn=0, player=1, action=2)]


def test_play_arena_game_offsets_default_search_seeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_offsets: list[int] = []

    def fake_create_core_search_engine(
        config: ArenaConfig,
        *,
        seed_offset: int = 0,
    ) -> PriorSearch:
        del config
        seen_offsets.append(seed_offset)
        return PriorSearch()

    monkeypatch.setattr(
        evaluate_module,
        "create_core_search_engine",
        fake_create_core_search_engine,
    )

    play_arena_game(
        seed=7,
        candidate_model=FakeNetwork(2),
        best_model=FakeNetwork(3),
        candidate_player=1,
        config=ArenaConfig(games=1, max_turns=4, gumbel_simulations=1),
        state=OneMoveState(),
    )

    assert seen_offsets == [14, 15]


def test_run_arena_reports_progress_after_each_game() -> None:
    progress: list[tuple[int, int, int]] = []

    report = run_arena(
        candidate_model=FakeNetwork(2),
        best_model=FakeNetwork(3),
        config=ArenaConfig(games=2, max_turns=4, gumbel_simulations=1),
        state_factory=OneMoveState,
        search_factory=PriorSearch,
        progress_callback=lambda current, total, game: progress.append(
            (current, total, game.seed)
        ),
    )

    assert report.summary.games == 2
    assert progress == [(1, 2, 0), (2, 2, 1)]


def test_run_arena_dispatches_to_batched_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[FakeNetwork, FakeNetwork, ArenaConfig]] = []
    game = ArenaGameResult(
        seed=0,
        candidate_player=1,
        best_player=2,
        winner=1,
        end_reason=1,
        moves=[],
        territory_scores=(0, 0),
    )
    report = ArenaReport(
        config=ArenaConfig(games=1, batch_size=2),
        games=[game],
        summary=summarize_arena([game], promotion_threshold=0.55),
    )

    def fake_run_arena_batched(
        *,
        candidate_model: FakeNetwork,
        best_model: FakeNetwork,
        config: ArenaConfig | None = None,
        progress_callback: Any = None,
    ) -> ArenaReport:
        del progress_callback
        assert config is not None
        calls.append((candidate_model, best_model, config))
        return report

    monkeypatch.setattr(evaluate_module, "run_arena_batched", fake_run_arena_batched)
    config = ArenaConfig(games=1, batch_size=2)
    candidate = FakeNetwork(2)
    best = FakeNetwork(3)

    result = run_arena(
        candidate_model=candidate,
        best_model=best,
        config=config,
    )

    assert result is report
    assert calls == [(candidate, best, config)]


def test_run_arena_rejects_sequential_hooks_for_batched_dispatch() -> None:
    with pytest.raises(ValueError, match="only supported for batch_size=1"):
        run_arena(
            candidate_model=FakeNetwork(2),
            best_model=FakeNetwork(3),
            config=ArenaConfig(games=2, batch_size=2),
            state_factory=OneMoveState,
        )


def test_create_core_arena_batch_requires_rust_batch_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "great_kingdom_core", types.SimpleNamespace())

    with pytest.raises(RuntimeError, match="GumbelArenaBatch is not available"):
        create_core_arena_batch(
            ArenaConfig(games=1),
            game_count=1,
            seed_start=0,
        )


def test_core_search_constructors_receive_policy_target_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_search_kwargs: dict[str, Any] = {}
    seen_batch_kwargs: dict[str, Any] = {}

    class FakeCore:
        class GumbelSearch:
            def __init__(self, **kwargs: Any) -> None:
                seen_search_kwargs.update(kwargs)

        class GumbelArenaBatch:
            def __init__(self, **kwargs: Any) -> None:
                seen_batch_kwargs.update(kwargs)

    monkeypatch.setitem(sys.modules, "great_kingdom_core", FakeCore)
    config = ArenaConfig(
        games=1,
        gumbel_c_visit=31.0,
        gumbel_c_scale=0.75,
        policy_target_c_visit=31.0,
        policy_target_c_scale=0.75,
        policy_target_temperature=1.0,
    )

    evaluate_module.create_core_search_engine(config, seed_offset=3)
    create_core_arena_batch(config, game_count=1, seed_start=0)

    assert seen_search_kwargs["policy_target_c_visit"] == 31.0
    assert seen_search_kwargs["policy_target_c_scale"] == 0.75
    assert seen_search_kwargs["policy_target_temperature"] == 1.0
    assert seen_batch_kwargs["policy_target_c_visit"] == 31.0
    assert seen_batch_kwargs["policy_target_c_scale"] == 0.75
    assert seen_batch_kwargs["policy_target_temperature"] == 1.0


def test_run_arena_batched_splits_root_rows_by_candidate_player(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    progress: list[tuple[int, int, int]] = []

    def fake_create_core_arena_batch(
        config: ArenaConfig,
        *,
        game_count: int,
        seed_start: int,
        game_index_start: int = 0,
    ) -> FakeArenaBatch:
        del config
        return FakeArenaBatch(
            game_count=game_count,
            seed_start=seed_start,
            game_index_start=game_index_start,
        )

    monkeypatch.setattr(
        evaluate_module,
        "create_core_arena_batch",
        fake_create_core_arena_batch,
    )

    report = run_arena_batched(
        candidate_model=FakeNetwork(2),
        best_model=FakeNetwork(3),
        config=ArenaConfig(games=2, batch_size=2, max_turns=4, gumbel_simulations=1),
        progress_callback=lambda current, total, game: progress.append(
            (current, total, game.seed)
        ),
    )

    assert [game.seed for game in report.games] == [0, 1]
    assert [game.candidate_player for game in report.games] == [1, 2]
    assert [game.best_player for game in report.games] == [2, 1]
    assert [[move.action for move in game.moves] for game in report.games] == [[2], [3]]
    assert report.summary.candidate_wins == 2
    assert progress == [(1, 2, 0), (2, 2, 1)]


def test_run_arena_batched_reports_finished_games_in_seed_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    progress: list[tuple[int, int, int]] = []

    def fake_create_core_arena_batch(
        config: ArenaConfig,
        *,
        game_count: int,
        seed_start: int,
        game_index_start: int = 0,
    ) -> UnevenFakeArenaBatch:
        del config
        return UnevenFakeArenaBatch(
            game_count=game_count,
            seed_start=seed_start,
            game_index_start=game_index_start,
        )

    monkeypatch.setattr(
        evaluate_module,
        "create_core_arena_batch",
        fake_create_core_arena_batch,
    )

    report = run_arena_batched(
        candidate_model=FakeNetwork(2),
        best_model=FakeNetwork(3),
        config=ArenaConfig(games=2, batch_size=2, max_turns=4, gumbel_simulations=1),
        progress_callback=lambda current, total, game: progress.append(
            (current, total, game.seed)
        ),
    )

    assert [game.seed for game in report.games] == [0, 1]
    assert [len(game.moves) for game in report.games] == [1, 2]
    assert progress == [(1, 2, 0), (2, 2, 1)]


def test_run_arena_batched_requires_leaf_game_index_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class MissingMetadataArenaBatch(FakeArenaBatch):
        missing_leaf_metadata = True

    def fake_create_core_arena_batch(
        config: ArenaConfig,
        *,
        game_count: int,
        seed_start: int,
        game_index_start: int = 0,
    ) -> MissingMetadataArenaBatch:
        del config
        return MissingMetadataArenaBatch(
            game_count=game_count,
            seed_start=seed_start,
            game_index_start=game_index_start,
        )

    monkeypatch.setattr(
        evaluate_module,
        "create_core_arena_batch",
        fake_create_core_arena_batch,
    )

    with pytest.raises(RuntimeError, match="game index metadata"):
        run_arena_batched(
            candidate_model=FakeNetwork(2),
            best_model=FakeNetwork(3),
            config=ArenaConfig(games=1, batch_size=1, max_turns=4, gumbel_simulations=1),
        )


def test_summarize_arena_reports_side_split_and_promotion() -> None:
    games = [
        ArenaGameResult(
            seed=1,
            candidate_player=1,
            best_player=2,
            winner=1,
            end_reason=1,
            moves=[MoveLog(turn=0, player=1, action=2)],
            territory_scores=(0, 0),
        ),
        ArenaGameResult(
            seed=2,
            candidate_player=2,
            best_player=1,
            winner=1,
            end_reason=1,
            moves=[MoveLog(turn=0, player=1, action=3), MoveLog(turn=1, player=2, action=2)],
            territory_scores=(0, 0),
        ),
    ]

    summary = summarize_arena(games, promotion_threshold=0.5)

    assert summary.games == 2
    assert summary.candidate_wins == 1
    assert summary.best_wins == 1
    assert summary.candidate_win_rate == pytest.approx(0.5)
    assert summary.candidate_blue_games == 1
    assert summary.candidate_blue_wins == 1
    assert summary.candidate_orange_games == 1
    assert summary.candidate_orange_wins == 0
    assert summary.average_game_length == pytest.approx(1.5)
    assert summary.promoted is True


def test_save_report_and_promote_candidate_copy_checkpoint(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.pt"
    best = tmp_path / "best.pt"
    candidate.write_text("candidate", encoding="utf-8")
    best.write_text("best", encoding="utf-8")
    game = ArenaGameResult(
        seed=1,
        candidate_player=1,
        best_player=2,
        winner=1,
        end_reason=1,
        moves=[MoveLog(turn=0, player=1, action=2)],
        territory_scores=(0, 0),
    )
    summary = summarize_arena([game], promotion_threshold=1.0)
    report = ArenaReport(config=ArenaConfig(games=1), games=[game], summary=summary)

    promoted = promote_candidate_if_needed(
        candidate_checkpoint=candidate,
        best_checkpoint=best,
        report=report,
    )
    saved = save_arena_report(report, tmp_path / "reports" / "arena.json")

    assert promoted is True
    assert best.read_text(encoding="utf-8") == "candidate"
    assert saved.is_file()
    assert '"candidate_win_rate": 1.0' in saved.read_text(encoding="utf-8")
