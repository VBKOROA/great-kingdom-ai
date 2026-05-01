from __future__ import annotations

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
    evaluate_state_policy,
    play_arena_game,
    promote_candidate_if_needed,
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
    value: np.ndarray


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
        value=np.zeros((len(legal_masks),), dtype=np.float32),
    )


@pytest.fixture(autouse=True)
def patch_evaluator(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(evaluate_module, "evaluate_feature_batch", fake_evaluate_feature_batch)


def test_evaluate_state_policy_masks_and_normalizes_legal_actions() -> None:
    state = OneMoveState()

    priors = evaluate_state_policy(FakeNetwork(2), state)

    assert len(priors) == ACTION_SPACE
    assert priors[0] == 0.0
    assert priors[2] > priors[3]
    assert sum(priors) == pytest.approx(1.0)


def test_play_arena_game_uses_candidate_when_candidate_has_current_turn() -> None:
    state = OneMoveState()
    config = ArenaConfig(games=1, max_turns=4, simulations=1)

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
