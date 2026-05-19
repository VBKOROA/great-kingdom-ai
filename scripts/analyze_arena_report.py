"""Analyze Great Kingdom arena report JSON files.

Usage:
    python scripts/analyze_arena_report.py data/pipeline/reports/arena-report.json
    python scripts/analyze_arena_report.py data/**/arena*.json --json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

END_REASON_NAMES = {
    1: "opponent castle destroyed",
    2: "own castle destroyed",
    3: "consecutive passes",
}

DEFAULT_PROMOTION_THRESHOLD = 0.55
WILSON_Z_95 = 1.959963984540054


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def analyze_report(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    games = _as_dict_list(report.get("games", []), field="games", path=path)
    raw_summary = report.get("summary", {})
    if raw_summary is not None and not isinstance(raw_summary, dict):
        raise ValueError(f"{path}: summary must be a JSON object when present")
    summary = raw_summary if isinstance(raw_summary, dict) else {}

    raw_config = report.get("config", {})
    if raw_config is not None and not isinstance(raw_config, dict):
        raise ValueError(f"{path}: config must be a JSON object when present")
    config = raw_config if isinstance(raw_config, dict) else {}

    threshold_value = _first_present(config, "promotion_threshold", fallback=summary)
    threshold = _float_value(threshold_value, default=DEFAULT_PROMOTION_THRESHOLD)
    require_side_win_rates = _bool_value(
        _first_present(config, "require_side_win_rates_for_promotion", fallback=summary),
        default=False,
    )
    stats = _stats_from_games(games)

    summary_games = _int_value(summary.get("games"), default=0)
    game_count = stats["games"] if games else summary_games
    candidate_wins = (
        stats["candidate_wins"]
        if games
        else _int_value(summary.get("candidate_wins"), default=0)
    )
    best_wins = stats["best_wins"] if games else _int_value(summary.get("best_wins"), default=0)
    candidate_win_rate = candidate_wins / game_count if game_count else 0.0
    best_win_rate = best_wins / game_count if game_count else 0.0

    candidate_blue_games = (
        stats["candidate_blue_games"]
        if games
        else _int_value(summary.get("candidate_blue_games"), default=0)
    )
    candidate_blue_wins = (
        stats["candidate_blue_wins"]
        if games
        else _int_value(summary.get("candidate_blue_wins"), default=0)
    )
    candidate_orange_games = (
        stats["candidate_orange_games"]
        if games
        else _int_value(summary.get("candidate_orange_games"), default=0)
    )
    candidate_orange_wins = (
        stats["candidate_orange_wins"]
        if games
        else _int_value(summary.get("candidate_orange_wins"), default=0)
    )
    candidate_blue_win_rate = _rate(candidate_blue_wins, candidate_blue_games)
    candidate_orange_win_rate = _rate(candidate_orange_wins, candidate_orange_games)

    ci_low, ci_high = wilson_interval(candidate_wins, game_count)
    return {
        "path": str(path),
        "games": game_count,
        "candidate_wins": candidate_wins,
        "best_wins": best_wins,
        "candidate_win_rate": candidate_win_rate,
        "best_win_rate": best_win_rate,
        "candidate_win_rate_ci95": [ci_low, ci_high],
        "promotion_threshold": threshold,
        "require_side_win_rates_for_promotion": require_side_win_rates,
        "promotion_margin": candidate_win_rate - threshold,
        "promoted": _is_promoted(
            game_count=game_count,
            candidate_win_rate=candidate_win_rate,
            candidate_blue_win_rate=candidate_blue_win_rate,
            candidate_orange_win_rate=candidate_orange_win_rate,
            threshold=threshold,
            require_side_win_rates=require_side_win_rates,
        ),
        "wins_needed_for_threshold": wins_needed_for_threshold(game_count, threshold),
        "candidate_blue_games": candidate_blue_games,
        "candidate_blue_wins": candidate_blue_wins,
        "candidate_blue_win_rate": candidate_blue_win_rate,
        "candidate_orange_games": candidate_orange_games,
        "candidate_orange_wins": candidate_orange_wins,
        "candidate_orange_win_rate": candidate_orange_win_rate,
        "overall_blue_wins": stats["overall_blue_wins"],
        "overall_orange_wins": stats["overall_orange_wins"],
        "overall_color_games": stats["overall_color_games"],
        "overall_blue_win_rate": _rate(stats["overall_blue_wins"], stats["overall_color_games"]),
        "overall_orange_win_rate": _rate(
            stats["overall_orange_wins"],
            stats["overall_color_games"],
        ),
        "lengths": stats["lengths"],
        "end_reasons": stats["end_reasons"],
        "territory": stats["territory"],
    }


def combine_analyses(path: str, analyses: list[dict[str, Any]]) -> dict[str, Any]:
    games = sum(int(analysis["games"]) for analysis in analyses)
    candidate_wins = sum(int(analysis["candidate_wins"]) for analysis in analyses)
    best_wins = sum(int(analysis["best_wins"]) for analysis in analyses)
    candidate_blue_games = sum(int(analysis["candidate_blue_games"]) for analysis in analyses)
    candidate_blue_wins = sum(int(analysis["candidate_blue_wins"]) for analysis in analyses)
    candidate_orange_games = sum(int(analysis["candidate_orange_games"]) for analysis in analyses)
    candidate_orange_wins = sum(int(analysis["candidate_orange_wins"]) for analysis in analyses)
    overall_blue_wins = sum(int(analysis["overall_blue_wins"]) for analysis in analyses)
    overall_orange_wins = sum(int(analysis["overall_orange_wins"]) for analysis in analyses)
    overall_color_games = sum(int(analysis["overall_color_games"]) for analysis in analyses)
    threshold = _common_threshold(analyses)
    require_side_win_rates = any(
        bool(analysis.get("require_side_win_rates_for_promotion", False))
        for analysis in analyses
    )
    ci_low, ci_high = wilson_interval(candidate_wins, games)

    lengths = _combine_length_stats([analysis["lengths"] for analysis in analyses])
    end_reasons = Counter[str]()
    for analysis in analyses:
        end_reasons.update({str(key): int(value) for key, value in analysis["end_reasons"].items()})

    return {
        "path": path,
        "games": games,
        "candidate_wins": candidate_wins,
        "best_wins": best_wins,
        "candidate_win_rate": _rate(candidate_wins, games),
        "best_win_rate": _rate(best_wins, games),
        "candidate_win_rate_ci95": [ci_low, ci_high],
        "promotion_threshold": threshold,
        "require_side_win_rates_for_promotion": require_side_win_rates,
        "promotion_margin": _rate(candidate_wins, games) - threshold,
        "promoted": _is_promoted(
            game_count=games,
            candidate_win_rate=_rate(candidate_wins, games),
            candidate_blue_win_rate=_rate(candidate_blue_wins, candidate_blue_games),
            candidate_orange_win_rate=_rate(candidate_orange_wins, candidate_orange_games),
            threshold=threshold,
            require_side_win_rates=require_side_win_rates,
        ),
        "wins_needed_for_threshold": wins_needed_for_threshold(games, threshold),
        "candidate_blue_games": candidate_blue_games,
        "candidate_blue_wins": candidate_blue_wins,
        "candidate_blue_win_rate": _rate(candidate_blue_wins, candidate_blue_games),
        "candidate_orange_games": candidate_orange_games,
        "candidate_orange_wins": candidate_orange_wins,
        "candidate_orange_win_rate": _rate(candidate_orange_wins, candidate_orange_games),
        "overall_blue_wins": overall_blue_wins,
        "overall_orange_wins": overall_orange_wins,
        "overall_color_games": overall_color_games,
        "overall_blue_win_rate": _rate(overall_blue_wins, overall_color_games),
        "overall_orange_win_rate": _rate(overall_orange_wins, overall_color_games),
        "lengths": lengths,
        "end_reasons": dict(sorted(end_reasons.items())),
        "territory": {},
    }


def wilson_interval(wins: int, total: int, z: float = WILSON_Z_95) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    phat = wins / total
    denominator = 1.0 + z * z / total
    center = phat + z * z / (2.0 * total)
    margin = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * total)) / total)
    return ((center - margin) / denominator, (center + margin) / denominator)


def wins_needed_for_threshold(games: int, threshold: float) -> int:
    if games <= 0:
        return 0
    return math.ceil(games * threshold)


def format_text(analyses: list[dict[str, Any]]) -> str:
    sections = [_format_one(analysis) for analysis in analyses]
    if len(analyses) > 1:
        sections.append(_format_one(combine_analyses("combined", analyses)))
    return "\n\n".join(sections)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze Great Kingdom arena report JSON files.")
    parser.add_argument("reports", type=Path, nargs="+", help="Arena report JSON path(s)")
    parser.add_argument("--json", action="store_true", help="Print machine-readable analysis JSON")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    analyses = [analyze_report(path, load_json(path)) for path in args.reports]
    if args.json:
        payload: dict[str, Any] = {"reports": analyses}
        if len(analyses) > 1:
            payload["combined"] = combine_analyses("combined", analyses)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    print(format_text(analyses))


def _stats_from_games(games: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_wins = 0
    best_wins = 0
    candidate_blue_games = 0
    candidate_blue_wins = 0
    candidate_orange_games = 0
    candidate_orange_wins = 0
    overall_blue_wins = 0
    overall_orange_wins = 0
    move_lengths: list[int] = []
    end_reasons: Counter[str] = Counter()
    territory_scores: list[tuple[int, int]] = []

    for game in games:
        candidate_player = _int_value(game.get("candidate_player"), default=0)
        best_player = _int_value(game.get("best_player"), default=_other_player(candidate_player))
        winner = _int_value(game.get("winner"), default=0)

        if winner == candidate_player:
            candidate_wins += 1
        if winner == best_player:
            best_wins += 1
        if candidate_player == 1:
            candidate_blue_games += 1
            if winner == 1:
                candidate_blue_wins += 1
        if candidate_player == 2:
            candidate_orange_games += 1
            if winner == 2:
                candidate_orange_wins += 1
        if winner == 1:
            overall_blue_wins += 1
        if winner == 2:
            overall_orange_wins += 1

        moves = game.get("moves", [])
        move_lengths.append(len(moves) if isinstance(moves, list) else 0)

        reason = _int_value(game.get("end_reason"), default=0)
        end_reasons[_end_reason_name(reason)] += 1

        territory = _score_pair(game.get("territory_scores"))
        if territory is not None:
            territory_scores.append(territory)

    return {
        "games": len(games),
        "candidate_wins": candidate_wins,
        "best_wins": best_wins,
        "candidate_blue_games": candidate_blue_games,
        "candidate_blue_wins": candidate_blue_wins,
        "candidate_orange_games": candidate_orange_games,
        "candidate_orange_wins": candidate_orange_wins,
        "overall_blue_wins": overall_blue_wins,
        "overall_orange_wins": overall_orange_wins,
        "overall_color_games": overall_blue_wins + overall_orange_wins,
        "lengths": _length_stats(move_lengths),
        "end_reasons": dict(sorted(end_reasons.items())),
        "territory": _territory_stats(territory_scores),
    }


def _format_one(analysis: dict[str, Any]) -> str:
    games = int(analysis["games"])
    candidate_wins = int(analysis["candidate_wins"])
    best_wins = int(analysis["best_wins"])
    ci_low, ci_high = analysis["candidate_win_rate_ci95"]
    threshold = float(analysis["promotion_threshold"])
    threshold_wins = int(analysis["wins_needed_for_threshold"])
    promoted = "yes" if analysis["promoted"] else "no"

    lines = [
        f"== {analysis['path']} ==",
        (
            f"games: {games} | candidate: {candidate_wins}-{best_wins} "
            f"({_pct(analysis['candidate_win_rate'])}, 95% CI {_pct(ci_low)}..{_pct(ci_high)})"
        ),
        (
            f"promotion: {promoted} | threshold {_pct(threshold)} "
            f"({threshold_wins}/{games}), margin {_pp(analysis['promotion_margin'])}"
        ),
        (
            "candidate side split: "
            f"Blue {analysis['candidate_blue_wins']}/{analysis['candidate_blue_games']} "
            f"= {_pct(analysis['candidate_blue_win_rate'])}, "
            f"Orange {analysis['candidate_orange_wins']}/{analysis['candidate_orange_games']} "
            f"= {_pct(analysis['candidate_orange_win_rate'])}"
        ),
        _format_overall_color_wins(analysis),
        f"lengths: {_format_lengths(analysis['lengths'])}",
        f"end reasons: {_format_counter(analysis['end_reasons'])}",
    ]

    territory = analysis["territory"]
    if territory:
        lines.append(
            "territory: "
            f"avg Blue {territory['average_blue']:.2f}, "
            f"avg Orange {territory['average_orange']:.2f}, "
            f"max Blue {territory['max_blue']:.0f}, "
            f"max Orange {territory['max_orange']:.0f}, "
            f"avg winner margin {territory['average_winner_margin']:.2f}, "
            f"max winner margin {territory['max_winner_margin']:.0f}"
        )
    return "\n".join(lines)


def _length_stats(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "min": 0, "median": 0.0, "average": 0.0, "p90": 0.0, "max": 0}
    ordered = sorted(values)
    p90_index = math.ceil(len(ordered) * 0.9) - 1
    return {
        "count": len(ordered),
        "min": ordered[0],
        "median": float(statistics.median(ordered)),
        "average": float(statistics.fmean(ordered)),
        "p90": float(ordered[max(0, min(p90_index, len(ordered) - 1))]),
        "max": ordered[-1],
    }


def _combine_length_stats(length_stats: list[dict[str, Any]]) -> dict[str, float | int]:
    count = sum(int(stats.get("count", 0)) for stats in length_stats)
    if count == 0:
        return _length_stats([])
    weighted_average = (
        sum(float(stats.get("average", 0.0)) * int(stats.get("count", 0)) for stats in length_stats)
        / count
    )
    non_empty_length_stats = [stats for stats in length_stats if int(stats.get("count", 0))]
    return {
        "count": count,
        "min": min(int(stats.get("min", 0)) for stats in non_empty_length_stats),
        "median": 0.0,
        "average": weighted_average,
        "p90": 0.0,
        "max": max(int(stats.get("max", 0)) for stats in non_empty_length_stats),
    }


def _territory_stats(scores: list[tuple[int, int]]) -> dict[str, float]:
    if not scores:
        return {}
    blue_scores = [score[0] for score in scores]
    orange_scores = [score[1] for score in scores]
    winner_margins = [abs(blue - orange) for blue, orange in scores]
    return {
        "average_blue": float(statistics.fmean(blue_scores)),
        "average_orange": float(statistics.fmean(orange_scores)),
        "max_blue": float(max(blue_scores)),
        "max_orange": float(max(orange_scores)),
        "average_winner_margin": float(statistics.fmean(winner_margins)),
        "max_winner_margin": float(max(winner_margins)),
    }


def _format_lengths(lengths: dict[str, Any]) -> str:
    if int(lengths.get("count", 0)) == 0:
        return "n/a"
    median = float(lengths["median"])
    p90 = float(lengths["p90"])
    median_text = "n/a" if median == 0.0 and p90 == 0.0 else f"{median:.1f}"
    p90_text = "n/a" if median == 0.0 and p90 == 0.0 else f"{p90:.1f}"
    return (
        f"avg {float(lengths['average']):.1f}, median {median_text}, "
        f"p90 {p90_text}, min {lengths['min']}, max {lengths['max']}"
    )


def _format_overall_color_wins(analysis: dict[str, Any]) -> str:
    color_games = int(analysis["overall_color_games"])
    if color_games == 0:
        return "overall color wins: n/a"
    return (
        f"overall color wins: Blue {analysis['overall_blue_wins']}/{color_games} "
        f"= {_pct(analysis['overall_blue_win_rate'])}, "
        f"Orange {analysis['overall_orange_wins']}/{color_games} "
        f"= {_pct(analysis['overall_orange_win_rate'])}"
    )


def _format_counter(counter: dict[str, Any]) -> str:
    if not counter:
        return "n/a"
    return ", ".join(f"{key} {value}" for key, value in counter.items())


def _as_dict_list(value: Any, *, field: str, path: Path) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{path}: {field} must be a list")
    result: list[dict[str, Any]] = []
    for index, row in enumerate(value):
        if not isinstance(row, dict):
            raise ValueError(f"{path}: {field}[{index}] must be a JSON object")
        result.append(row)
    return result


def _first_present(primary: dict[str, Any], key: str, *, fallback: dict[str, Any]) -> Any:
    if key in primary:
        return primary[key]
    return fallback.get(key)


def _common_threshold(analyses: list[dict[str, Any]]) -> float:
    thresholds = {float(analysis["promotion_threshold"]) for analysis in analyses}
    if len(thresholds) == 1:
        return thresholds.pop()
    return DEFAULT_PROMOTION_THRESHOLD


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _is_promoted(
    *,
    game_count: int,
    candidate_win_rate: float,
    candidate_blue_win_rate: float,
    candidate_orange_win_rate: float,
    threshold: float,
    require_side_win_rates: bool,
) -> bool:
    promoted = game_count > 0 and candidate_win_rate >= threshold
    if require_side_win_rates:
        promoted = (
            promoted
            and candidate_blue_win_rate >= threshold
            and candidate_orange_win_rate >= threshold
        )
    return promoted


def _int_value(value: Any, *, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float_value(value: Any, *, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bool_value(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _score_pair(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, list | tuple) or len(value) != 2:
        return None
    if not all(isinstance(item, int | float) for item in value):
        return None
    return (int(value[0]), int(value[1]))


def _other_player(player: int) -> int:
    return 2 if player == 1 else 1


def _end_reason_name(reason: int) -> str:
    return END_REASON_NAMES.get(reason, f"unknown({reason})")


def _pct(value: float) -> str:
    return f"{value * 100.0:.2f}%"


def _pp(value: float) -> str:
    return f"{value * 100.0:+.2f}pp"


if __name__ == "__main__":
    main()
