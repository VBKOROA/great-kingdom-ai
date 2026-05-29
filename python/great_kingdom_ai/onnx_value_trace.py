"""Trace ONNX value/search consistency through one deterministic Gumbel game."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, NoReturn

import numpy as np

from great_kingdom_ai.features import ACTION_SPACE, BOARD_CELLS, BOARD_SIZE, FEATURE_CHANNELS
from great_kingdom_ai.self_play_data import value_target_for_player


@dataclass(frozen=True)
class OnnxValueTraceConfig:
    onnx_model_path: Path
    max_turns: int = 200
    gumbel_simulations: int = 128
    gumbel_max_considered_actions: int = 16
    gumbel_c_visit: float = 50.0
    gumbel_c_scale: float = 1.0
    gumbel_scale: float = 0.0
    gumbel_seed: int = 0
    policy_target_c_visit: float = 5.0
    policy_target_c_scale: float = 1.0
    policy_target_temperature: float = 1.0
    leaf_batch_size: int = 8
    onnx_max_batch_size: int = 128
    action_history: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.max_turns <= 0:
            raise ValueError("max_turns must be positive")
        if self.gumbel_simulations <= 0:
            raise ValueError("gumbel_simulations must be positive")
        if self.gumbel_max_considered_actions <= 0:
            raise ValueError("gumbel_max_considered_actions must be positive")
        if self.gumbel_c_visit <= 0.0:
            raise ValueError("gumbel_c_visit must be positive")
        if self.gumbel_c_scale <= 0.0:
            raise ValueError("gumbel_c_scale must be positive")
        if not math.isfinite(self.gumbel_scale) or self.gumbel_scale < 0.0:
            raise ValueError("gumbel_scale must be finite and non-negative")
        if self.policy_target_c_visit <= 0.0:
            raise ValueError("policy_target_c_visit must be positive")
        if self.policy_target_c_scale <= 0.0:
            raise ValueError("policy_target_c_scale must be positive")
        if self.policy_target_temperature <= 0.0:
            raise ValueError("policy_target_temperature must be positive")
        if self.leaf_batch_size <= 0:
            raise ValueError("leaf_batch_size must be positive")
        if self.onnx_max_batch_size <= 0:
            raise ValueError("onnx_max_batch_size must be positive")
        for action in self.action_history:
            if action < 0 or action >= ACTION_SPACE:
                raise ValueError(f"action_history contains invalid action: {action}")


@dataclass(frozen=True)
class PositionAnalysis:
    player: int
    net_value_to_play: float
    search_value_to_play: float
    result: Any


@dataclass(frozen=True)
class OnnxValueTraceRow:
    turn: int
    player: int
    action: int
    action_label: str
    visit_count_a: int
    v_net_my_s: float
    q_root_my_a: float
    v_net_my_s_after: float | None
    search_value_my_s_after: float | None
    s_after_response_action: int | None
    s_after_response_label: str | None
    s_after_response_q_my: float | None
    s_after_response_visit_count: int | None
    parent_child_response_q_my: float | None
    parent_child_response_visit_count: int | None
    parent_child_response_prior_rank: int | None
    parent_child_response_prior_prob: float | None
    parent_child_response_q_rank: int | None
    s_after_response2_action: int | None
    s_after_response2_label: str | None
    s_after_response2_q_my: float | None
    s_after_response2_visit_count: int | None
    terminal_value_my_s_after: float | None
    q_minus_v_net_s: float
    v_net_s_after_minus_q: float | None
    search_value_s_after_minus_q: float | None
    search_minus_v_net_s_after: float | None


@dataclass(frozen=True)
class OnnxValueTraceReport:
    rows: tuple[OnnxValueTraceRow, ...]
    summary: dict[str, dict[str, float | int]]
    winner: int | None
    end_reason: int | None
    territory_scores: tuple[int, int] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": [asdict(row) for row in self.rows],
            "summary": self.summary,
            "winner": self.winner,
            "end_reason": self.end_reason,
            "territory_scores": self.territory_scores,
        }


def run_onnx_value_trace(
    config: OnnxValueTraceConfig,
    *,
    core: Any | None = None,
    evaluator: Any | None = None,
    search: Any | None = None,
) -> OnnxValueTraceReport:
    core = _import_core() if core is None else core
    state = core.GameState()
    _apply_action_history(state, config.action_history)

    evaluator = evaluator or core.OnnxEvaluator(
        str(config.onnx_model_path),
        device="cpu",
        max_batch_size=config.onnx_max_batch_size,
    )
    search = search or core.GumbelSearch(
        simulations=config.gumbel_simulations,
        max_considered_actions=config.gumbel_max_considered_actions,
        c_visit=config.gumbel_c_visit,
        c_scale=config.gumbel_c_scale,
        seed=config.gumbel_seed,
        gumbel_scale=config.gumbel_scale,
        policy_target_temperature=config.policy_target_temperature,
        policy_target_c_visit=config.policy_target_c_visit,
        policy_target_c_scale=config.policy_target_c_scale,
    )

    rows: list[OnnxValueTraceRow] = []
    current = (
        None
        if state.is_terminal()
        else _analyze_position(core, evaluator, search, state, config)
    )
    start_turn = len(config.action_history)
    for turn in range(start_turn, start_turn + config.max_turns):
        if current is None or state.is_terminal():
            break
        player = current.player
        result = current.result
        action = _selected_action(result)
        q_root = _selected_action_q(result)
        visit_count = _selected_visit_count(result, action)
        state.apply_action(action)

        if state.is_terminal():
            winner = _optional_int(state.winner())
            terminal_value = None
            if winner is not None:
                terminal_value = value_target_for_player(player=player, winner=winner)
            after_net = None
            after_search = None
            response_action = None
            response_q_my = None
            response_visit_count = None
            parent_child_response_q = None
            parent_child_response_visit_count = None
            parent_child_response_prior_rank = None
            parent_child_response_prior_prob = None
            parent_child_response_q_rank = None
            next_analysis = None
        else:
            next_analysis = _analyze_position(core, evaluator, search, state, config)
            after_net = _value_from_player_perspective(
                next_analysis.net_value_to_play,
                value_player=next_analysis.player,
                target_player=player,
            )
            after_search = _value_from_player_perspective(
                next_analysis.search_value_to_play,
                value_player=next_analysis.player,
                target_player=player,
            )
            response_action = _selected_action(next_analysis.result)
            response_q_my = _value_from_player_perspective(
                _selected_action_q(next_analysis.result),
                value_player=next_analysis.player,
                target_player=player,
            )
            response_visit_count = _selected_visit_count(
                next_analysis.result,
                response_action,
            )
            parent_child_response_q = _selected_child_completed_q(result, response_action)
            parent_child_response_visit_count = _selected_child_visit_count(
                result,
                response_action,
            )
            parent_child_log_priors = _selected_child_log_priors(result)
            parent_child_q_values = _selected_child_completed_q_values(result)
            parent_child_response_prior_rank = _rank_descending(
                parent_child_log_priors,
                response_action,
                finite_only=True,
            )
            parent_child_response_prior_prob = _probability_from_log_prior(
                parent_child_log_priors[response_action],
            )
            parent_child_response_q_rank = _rank_descending(
                parent_child_q_values,
                response_action,
                finite_mask=parent_child_log_priors,
            )
            terminal_value = None

        rows.append(
            OnnxValueTraceRow(
                turn=turn,
                player=player,
                action=action,
                action_label=action_label(action),
                visit_count_a=visit_count,
                v_net_my_s=current.net_value_to_play,
                q_root_my_a=q_root,
                v_net_my_s_after=after_net,
                search_value_my_s_after=after_search,
                s_after_response_action=response_action,
                s_after_response_label=(
                    None if response_action is None else action_label(response_action)
                ),
                s_after_response_q_my=response_q_my,
                s_after_response_visit_count=response_visit_count,
                parent_child_response_q_my=parent_child_response_q,
                parent_child_response_visit_count=parent_child_response_visit_count,
                parent_child_response_prior_rank=parent_child_response_prior_rank,
                parent_child_response_prior_prob=parent_child_response_prior_prob,
                parent_child_response_q_rank=parent_child_response_q_rank,
                s_after_response2_action=None,
                s_after_response2_label=None,
                s_after_response2_q_my=None,
                s_after_response2_visit_count=None,
                terminal_value_my_s_after=terminal_value,
                q_minus_v_net_s=q_root - current.net_value_to_play,
                v_net_s_after_minus_q=None if after_net is None else after_net - q_root,
                search_value_s_after_minus_q=(
                    None if after_search is None else after_search - q_root
                ),
                search_minus_v_net_s_after=(
                    None if after_search is None or after_net is None else after_search - after_net
                ),
            )
        )
        current = next_analysis

    return OnnxValueTraceReport(
        rows=tuple(_attach_response2_columns(rows)),
        summary=summarize_trace_rows(rows),
        winner=_optional_int(state.winner()),
        end_reason=_optional_int(state.end_reason()),
        territory_scores=None if not state.is_terminal() else _int_pair(state.territory_scores()),
    )


def summarize_trace_rows(rows: list[OnnxValueTraceRow]) -> dict[str, dict[str, float | int]]:
    metrics = (
        "q_minus_v_net_s",
        "v_net_s_after_minus_q",
        "search_value_s_after_minus_q",
        "search_minus_v_net_s_after",
    )
    return {metric: _summary_stats(_metric_values(rows, metric)) for metric in metrics}


def _attach_response2_columns(rows: list[OnnxValueTraceRow]) -> list[OnnxValueTraceRow]:
    updated = list(rows)
    for index, row in enumerate(rows[:-1]):
        next_row = rows[index + 1]
        if row.s_after_response_action != next_row.action:
            continue
        response2_q = None
        if next_row.s_after_response_q_my is not None:
            response2_q = -next_row.s_after_response_q_my
        updated[index] = replace(
            row,
            s_after_response2_action=next_row.s_after_response_action,
            s_after_response2_label=next_row.s_after_response_label,
            s_after_response2_q_my=response2_q,
            s_after_response2_visit_count=next_row.s_after_response_visit_count,
        )
    return updated


def action_label(action: int) -> str:
    if action == BOARD_CELLS:
        return "pass"
    return f"r{action // BOARD_SIZE}c{action % BOARD_SIZE}"


def format_trace_report(report: OnnxValueTraceReport) -> str:
    lines = [
        "turn player action  N(a)  V_net_my(s)  Q_root_my(a)  V_net_my(s_after)  "
        "search_value_my(s_after)  resp  resp_Q_my  resp_N  parent_resp_Q  "
        "parent_resp_N  p_rank  p_prob  q_rank  resp2  resp2_Q_my  resp2_N  "
        "Q-V(s)  V_after-Q  search_after-Q"
    ]
    for row in report.rows:
        lines.append(
            f"{row.turn:>4} {row.player:>6} {row.action_label:>6} "
            f"{row.visit_count_a:>5} "
            f"{_fmt(row.v_net_my_s):>12} {_fmt(row.q_root_my_a):>13} "
            f"{_fmt(row.v_net_my_s_after):>18} "
            f"{_fmt(row.search_value_my_s_after):>24} "
            f"{_fmt_label(row.s_after_response_label):>6} "
            f"{_fmt(row.s_after_response_q_my):>10} "
            f"{_fmt_int(row.s_after_response_visit_count):>6} "
            f"{_fmt(row.parent_child_response_q_my):>13} "
            f"{_fmt_int(row.parent_child_response_visit_count):>13} "
            f"{_fmt_int(row.parent_child_response_prior_rank):>6} "
            f"{_fmt_prob(row.parent_child_response_prior_prob):>7} "
            f"{_fmt_int(row.parent_child_response_q_rank):>6} "
            f"{_fmt_label(row.s_after_response2_label):>6} "
            f"{_fmt(row.s_after_response2_q_my):>11} "
            f"{_fmt_int(row.s_after_response2_visit_count):>7} "
            f"{_fmt(row.q_minus_v_net_s):>7} "
            f"{_fmt(row.v_net_s_after_minus_q):>9} "
            f"{_fmt(row.search_value_s_after_minus_q):>14}"
        )
    lines.append("")
    lines.append("summary")
    for metric, stats in report.summary.items():
        lines.append(
            f"{metric}: n={stats['count']} mean={_fmt(stats['mean'])} "
            f"std={_fmt(stats['std'])} min={_fmt(stats['min'])} "
            f"max={_fmt(stats['max'])} max_abs={_fmt(stats['max_abs'])}"
        )
    if report.winner is not None:
        lines.append(
            f"terminal: winner={report.winner} end_reason={report.end_reason} "
            f"territory_scores={report.territory_scores}"
        )
    else:
        lines.append("terminal: not reached")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one CPU ONNX deterministic Gumbel play trace and print value sanity stats. "
            "All *_my columns are from the player who selected the root action."
        )
    )
    parser.add_argument("onnx_model_path", type=Path)
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--gumbel-simulations", type=int, default=128)
    parser.add_argument("--gumbel-max-considered-actions", type=int, default=16)
    parser.add_argument("--gumbel-c-visit", type=float, default=50.0)
    parser.add_argument("--gumbel-c-scale", type=float, default=1.0)
    parser.add_argument(
        "--gumbel-scale",
        type=float,
        default=0.0,
        help="root Gumbel noise scale; default 0.0 for deterministic play",
    )
    parser.add_argument("--gumbel-seed", type=int, default=0)
    parser.add_argument("--policy-target-c-visit", type=float, default=5.0)
    parser.add_argument("--policy-target-c-scale", type=float, default=1.0)
    parser.add_argument("--policy-target-temperature", type=float, default=1.0)
    parser.add_argument("--leaf-batch-size", type=int, default=8)
    parser.add_argument("--onnx-max-batch-size", type=int, default=128)
    parser.add_argument(
        "--action-history",
        default="",
        help="comma-separated actions to reach a starting position; use pass or 81 for pass",
    )
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    return parser


def main() -> NoReturn:
    args = build_parser().parse_args()
    config = OnnxValueTraceConfig(
        onnx_model_path=args.onnx_model_path,
        max_turns=args.max_turns,
        gumbel_simulations=args.gumbel_simulations,
        gumbel_max_considered_actions=args.gumbel_max_considered_actions,
        gumbel_c_visit=args.gumbel_c_visit,
        gumbel_c_scale=args.gumbel_c_scale,
        gumbel_scale=args.gumbel_scale,
        gumbel_seed=args.gumbel_seed,
        policy_target_c_visit=args.policy_target_c_visit,
        policy_target_c_scale=args.policy_target_c_scale,
        policy_target_temperature=args.policy_target_temperature,
        leaf_batch_size=args.leaf_batch_size,
        onnx_max_batch_size=args.onnx_max_batch_size,
        action_history=parse_action_history(args.action_history),
    )
    report = run_onnx_value_trace(config)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(format_trace_report(report))
    raise SystemExit(0)


def parse_action_history(raw: str) -> tuple[int, ...]:
    if not raw.strip():
        return ()
    actions = []
    for chunk in raw.split(","):
        token = chunk.strip().lower()
        if token == "pass":
            actions.append(BOARD_CELLS)
            continue
        actions.append(int(token))
    return tuple(actions)


def _analyze_position(
    core: Any,
    evaluator: Any,
    search: Any,
    state: Any,
    config: OnnxValueTraceConfig,
) -> PositionAnalysis:
    root_logits, root_values = _evaluate_state(core, evaluator, state)
    root_value = float(root_values[0])

    def leaf_evaluator(request: Any) -> tuple[list[list[float]], list[float]]:
        policies, values = evaluator.evaluate(request)
        return (
            [[float(value) for value in row] for row in policies],
            [float(value) for value in values],
        )

    result = search.search_with_logits_and_evaluator(
        state,
        [float(value) for value in root_logits[0]],
        leaf_evaluator,
        root_value,
        config.leaf_batch_size,
    )
    return PositionAnalysis(
        player=int(state.current_player()),
        net_value_to_play=root_value,
        search_value_to_play=_root_value(result),
        result=result,
    )


def _evaluate_state(core: Any, evaluator: Any, state: Any) -> tuple[list[list[float]], list[float]]:
    request = _eval_request_for_state(core, state)
    policies, values = evaluator.evaluate(request)
    if len(policies) != 1 or len(values) != 1:
        raise ValueError(
            f"expected one ONNX row, got {len(policies)} policies and {len(values)} values"
        )
    if len(policies[0]) != ACTION_SPACE:
        raise ValueError(f"expected {ACTION_SPACE} policy logits, got {len(policies[0])}")
    return (
        [[float(value) for value in policies[0]]],
        [float(values[0])],
    )


def _eval_request_for_state(core: Any, state: Any) -> Any:
    features = np.asarray(state.feature_planes(), dtype=np.float32)
    expected = FEATURE_CHANNELS * BOARD_CELLS
    if features.shape != (expected,):
        raise ValueError(f"expected flat feature shape {(expected,)}, got {features.shape}")
    if hasattr(core.EvalRequest, "from_feature_plane_bytes"):
        return core.EvalRequest.from_feature_plane_bytes(1, features.reshape(1, -1).tobytes())
    return core.EvalRequest.from_feature_rows([features.tolist()])


def _apply_action_history(state: Any, actions: tuple[int, ...]) -> None:
    for offset, action in enumerate(actions):
        if state.is_terminal():
            raise ValueError(f"action_history reaches terminal before offset {offset}")
        legal = set(int(value) for value in state.legal_actions())
        if action not in legal:
            raise ValueError(f"illegal action_history action at offset {offset}: {action}")
        state.apply_action(action)


def _selected_action(result: Any) -> int:
    selected = result.selected_action()
    if selected is None:
        raise RuntimeError("Gumbel result did not select an action")
    return int(selected)


def _selected_action_q(result: Any) -> float:
    selected_action_q = getattr(result, "selected_action_q", None)
    if selected_action_q is None:
        raise RuntimeError(
            "Gumbel result does not expose selected_action_q; rebuild great_kingdom_core."
        )
    q = selected_action_q()
    if q is None:
        raise RuntimeError("Gumbel result selected_action_q is missing")
    value = float(q)
    if not math.isfinite(value) or value < -1.0 or value > 1.0:
        raise ValueError("selected_action_q must be finite and in [-1, 1]")
    return value


def _selected_visit_count(result: Any, action: int) -> int:
    visit_counts = getattr(result, "visit_counts", None)
    if visit_counts is None:
        raise RuntimeError("Gumbel result does not expose visit_counts.")
    counts = visit_counts()
    if len(counts) != ACTION_SPACE:
        raise ValueError(f"expected {ACTION_SPACE} visit counts, got {len(counts)}")
    count = int(counts[action])
    if count < 0:
        raise ValueError("visit_count must be non-negative")
    return count


def _selected_child_visit_count(result: Any, action: int) -> int:
    visit_counts = getattr(result, "selected_child_visit_counts", None)
    if visit_counts is None:
        raise RuntimeError("Gumbel result does not expose selected_child_visit_counts.")
    counts = visit_counts()
    if len(counts) != ACTION_SPACE:
        raise ValueError(f"expected {ACTION_SPACE} selected child visit counts, got {len(counts)}")
    count = int(counts[action])
    if count < 0:
        raise ValueError("selected child visit_count must be non-negative")
    return count


def _selected_child_completed_q(result: Any, action: int) -> float:
    values = _selected_child_completed_q_values(result)
    value = float(values[action])
    if not math.isfinite(value) or value < -1.0 or value > 1.0:
        raise ValueError("selected child q must be finite and in [-1, 1]")
    return value


def _selected_child_completed_q_values(result: Any) -> list[float]:
    completed_q = getattr(result, "selected_child_completed_q", None)
    if completed_q is None:
        raise RuntimeError("Gumbel result does not expose selected_child_completed_q.")
    values = completed_q()
    if len(values) != ACTION_SPACE:
        raise ValueError(f"expected {ACTION_SPACE} selected child q values, got {len(values)}")
    return [float(value) for value in values]


def _selected_child_log_priors(result: Any) -> list[float]:
    log_priors = getattr(result, "selected_child_log_priors", None)
    if log_priors is None:
        raise RuntimeError("Gumbel result does not expose selected_child_log_priors.")
    values = log_priors()
    if len(values) != ACTION_SPACE:
        raise ValueError(f"expected {ACTION_SPACE} selected child log priors, got {len(values)}")
    return [float(value) for value in values]


def _rank_descending(
    values: list[float],
    action: int,
    *,
    finite_only: bool = False,
    finite_mask: list[float] | None = None,
) -> int | None:
    if action < 0 or action >= len(values):
        return None
    if finite_mask is not None and not math.isfinite(finite_mask[action]):
        return None
    action_value = values[action]
    if not math.isfinite(action_value):
        return None
    rank = 1
    for index, value in enumerate(values):
        if index == action:
            continue
        if finite_mask is not None and not math.isfinite(finite_mask[index]):
            continue
        if finite_only and not math.isfinite(value):
            continue
        if math.isfinite(value) and value > action_value:
            rank += 1
    return rank


def _probability_from_log_prior(log_prior: float) -> float | None:
    if not math.isfinite(log_prior):
        return None
    return math.exp(log_prior)


def _root_value(result: Any) -> float:
    root_value = getattr(result, "root_value", None)
    if root_value is None:
        raise RuntimeError("Gumbel result does not expose root_value; rebuild great_kingdom_core.")
    value = float(root_value())
    if not math.isfinite(value) or value < -1.0 or value > 1.0:
        raise ValueError("root_value must be finite and in [-1, 1]")
    return value


def _value_from_player_perspective(
    value: float,
    *,
    value_player: int,
    target_player: int,
) -> float:
    if value_player not in {1, 2} or target_player not in {1, 2}:
        raise ValueError("players must be 1 or 2")
    return value if value_player == target_player else -value


def _metric_values(rows: list[OnnxValueTraceRow], metric: str) -> list[float]:
    values = []
    for row in rows:
        value = getattr(row, metric)
        if value is not None and math.isfinite(float(value)):
            values.append(float(value))
    return values


def _summary_stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "max_abs": float("nan"),
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
        "max_abs": float(np.max(np.abs(array))),
    }


def _fmt(value: float | int | None) -> str:
    if value is None:
        return "NA"
    numeric = float(value)
    if not math.isfinite(numeric):
        return "nan"
    return f"{numeric:+.5f}"


def _fmt_int(value: int | None) -> str:
    return "NA" if value is None else str(value)


def _fmt_prob(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.4f}"


def _fmt_label(value: str | None) -> str:
    return "NA" if value is None else value


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _int_pair(values: tuple[Any, Any]) -> tuple[int, int]:
    return int(values[0]), int(values[1])


def _import_core() -> Any:
    try:
        import great_kingdom_core as core  # type: ignore[import-untyped]
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "great_kingdom_core is not installed. Build it with maturin before ONNX value trace."
        ) from exc
    return core


__all__ = [
    "OnnxValueTraceConfig",
    "OnnxValueTraceReport",
    "OnnxValueTraceRow",
    "action_label",
    "format_trace_report",
    "parse_action_history",
    "run_onnx_value_trace",
    "summarize_trace_rows",
]
