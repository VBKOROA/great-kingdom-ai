from __future__ import annotations

from pathlib import Path

import pytest
from great_kingdom_ai.features import ACTION_SPACE, FEATURE_CHANNELS
from great_kingdom_ai.onnx_value_trace import (
    OnnxValueTraceConfig,
    action_label,
    parse_action_history,
    run_onnx_value_trace,
)


class FakeEvalRequest:
    def __init__(self, rows: list[list[float]]) -> None:
        self._rows = rows

    @staticmethod
    def from_feature_rows(rows: list[list[float]]) -> FakeEvalRequest:
        return FakeEvalRequest(rows)

    def feature_planes(self) -> list[list[float]]:
        return self._rows


class FakeState:
    def __init__(self) -> None:
        self.player = 1
        self.actions: list[int] = []

    def current_player(self) -> int:
        return self.player

    def feature_planes(self) -> list[float]:
        row = [0.0] * (FEATURE_CHANNELS * 9 * 9)
        row[0] = float(self.player)
        return row

    def legal_actions(self) -> list[int]:
        return [0, 1, 81]

    def apply_action(self, action: int) -> None:
        self.actions.append(action)
        self.player = 2 if self.player == 1 else 1

    def is_terminal(self) -> bool:
        return False

    def winner(self) -> None:
        return None

    def end_reason(self) -> None:
        return None

    def territory_scores(self) -> tuple[int, int]:
        return (0, 0)


class FakeEvaluator:
    def evaluate(self, request: FakeEvalRequest) -> tuple[list[list[float]], list[float]]:
        values = []
        policies = []
        for row in request.feature_planes():
            player = int(row[0])
            values.append(0.2 if player == 1 else -0.3)
            policies.append([0.0] * ACTION_SPACE)
        return policies, values


class FakeResult:
    def __init__(self, action: int, q: float, root_value: float, visit_count: int) -> None:
        self.action = action
        self.q = q
        self.value = root_value
        self.visit_count = visit_count

    def selected_action(self) -> int:
        return self.action

    def selected_action_q(self) -> float:
        return self.q

    def visit_counts(self) -> list[int]:
        counts = [0] * ACTION_SPACE
        counts[self.action] = self.visit_count
        return counts

    def selected_child_visit_counts(self) -> list[int]:
        counts = [0] * ACTION_SPACE
        counts[1] = 7
        return counts

    def selected_child_completed_q(self) -> list[float]:
        values = [0.0] * ACTION_SPACE
        values[0] = 0.5
        values[1] = 0.25
        return values

    def selected_child_log_priors(self) -> list[float]:
        values = [float("-inf")] * ACTION_SPACE
        values[0] = -0.35667494
        values[1] = -1.60943791
        values[81] = -2.30258509
        return values

    def root_value(self) -> float:
        return self.value


class FakeSearch:
    last_kwargs: dict[str, object] | None = None

    def __init__(self, **kwargs: object) -> None:
        type(self).last_kwargs = kwargs

    def search_with_logits_and_evaluator(
        self,
        state: FakeState,
        _root_logits: list[float],
        _leaf_evaluator: object,
        _root_value: float,
        _leaf_batch_size: int,
    ) -> FakeResult:
        if state.current_player() == 1:
            return FakeResult(action=0, q=0.5, root_value=0.6, visit_count=3)
        return FakeResult(action=1, q=-0.1, root_value=-0.4, visit_count=2)


class FakeCore:
    GameState = FakeState
    EvalRequest = FakeEvalRequest
    GumbelSearch = FakeSearch


def test_value_trace_converts_after_values_to_selected_player_perspective() -> None:
    report = run_onnx_value_trace(
        OnnxValueTraceConfig(onnx_model_path=Path("model.onnx"), max_turns=1),
        core=FakeCore(),
        evaluator=FakeEvaluator(),
        search=FakeSearch(),
    )

    assert len(report.rows) == 1
    row = report.rows[0]
    assert row.player == 1
    assert row.action == 0
    assert row.visit_count_a == 3
    assert row.v_net_my_s == pytest.approx(0.2)
    assert row.q_root_my_a == pytest.approx(0.5)
    assert row.v_net_my_s_after == pytest.approx(0.3)
    assert row.search_value_my_s_after == pytest.approx(0.4)
    assert row.s_after_response_action == 1
    assert row.s_after_response_label == "r0c1"
    assert row.s_after_response_q_my == pytest.approx(0.1)
    assert row.s_after_response_visit_count == 2
    assert row.parent_child_response_q_my == pytest.approx(0.25)
    assert row.parent_child_response_visit_count == 7
    assert row.parent_child_response_prior_rank == 2
    assert row.parent_child_response_prior_prob == pytest.approx(0.2)
    assert row.parent_child_response_q_rank == 2
    assert row.q_minus_v_net_s == pytest.approx(0.3)
    assert row.search_minus_v_net_s_after == pytest.approx(0.1)
    assert report.summary["q_minus_v_net_s"]["count"] == 1


def test_action_history_parser_accepts_pass_token() -> None:
    assert parse_action_history("0, pass, 81") == (0, 81, 81)
    assert action_label(81) == "pass"
    assert action_label(10) == "r1c1"


def test_value_trace_passes_gumbel_scale_to_search() -> None:
    FakeSearch.last_kwargs = None

    run_onnx_value_trace(
        OnnxValueTraceConfig(
            onnx_model_path=Path("model.onnx"),
            max_turns=1,
            gumbel_scale=0.75,
        ),
        core=FakeCore(),
        evaluator=FakeEvaluator(),
    )

    assert FakeSearch.last_kwargs is not None
    assert FakeSearch.last_kwargs["gumbel_scale"] == pytest.approx(0.75)


def test_value_trace_attaches_second_response_from_next_row() -> None:
    report = run_onnx_value_trace(
        OnnxValueTraceConfig(onnx_model_path=Path("model.onnx"), max_turns=2),
        core=FakeCore(),
        evaluator=FakeEvaluator(),
        search=FakeSearch(),
    )

    first = report.rows[0]
    assert first.s_after_response_action == report.rows[1].action
    assert first.s_after_response2_action == report.rows[1].s_after_response_action
    assert first.s_after_response2_q_my == pytest.approx(-report.rows[1].s_after_response_q_my)
    assert first.s_after_response2_visit_count == report.rows[1].s_after_response_visit_count
