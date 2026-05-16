"""Diagnose Blue/Orange winner bias in self-play logs or trajectory replay."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NoReturn

import numpy as np

DEFAULT_WORK_DIR = Path("data/runpod/train-v2-gumbel-512k-adamw3e4-cscale01-r16")
DEFAULT_RECENT_WINDOWS = (100, 500, 1000, 5000)


@dataclass(frozen=True)
class GameRecord:
    index: int
    winner: int
    moves: int | None = None
    seed: int | None = None
    end_reason: int | None = None
    territory_scores: tuple[int, int] | None = None
    model_version: int | None = None
    created_iteration: int | None = None


@dataclass(frozen=True)
class ArraySummary:
    count: int
    mean: float
    std: float
    min: float
    p05: float
    p50: float
    p95: float
    max: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize Blue/Orange winner bias from self-play game_logs.jsonl, "
            "shard game_logs.json files, or trajectory-replay.npz."
        )
    )
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument(
        "--game-logs",
        type=Path,
        nargs="+",
        default=None,
        help="Explicit game log path(s). Supports JSON list files and JSONL files.",
    )
    parser.add_argument(
        "--replay",
        type=Path,
        default=None,
        help="Explicit trajectory-replay.npz path. Used when game logs are unavailable.",
    )
    parser.add_argument(
        "--include-shard-logs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When aggregate game_logs.jsonl is missing, scan shards/*/game_logs.json.",
    )
    parser.add_argument(
        "--recent-windows",
        type=int,
        nargs="+",
        default=list(DEFAULT_RECENT_WINDOWS),
        help="Recent game counts to summarize from the chronological tail.",
    )
    parser.add_argument(
        "--by-model-version",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For replay inputs, summarize episodes by first-row model_version.",
    )
    parser.add_argument("--pretty", action="store_true")
    return parser


def main() -> NoReturn:
    args = build_parser().parse_args()
    records, sources = load_records_from_args(args)
    payload = analyze_records(
        records,
        sources=sources,
        recent_windows=args.recent_windows,
        include_model_breakdown=args.by_model_version,
    )
    print(json.dumps(payload, indent=2 if args.pretty else None, sort_keys=True))
    raise SystemExit(0)


def load_records_from_args(args: argparse.Namespace) -> tuple[list[GameRecord], list[str]]:
    sources: list[Path] = []
    if args.game_logs is not None:
        sources.extend(args.game_logs)
        return load_game_log_records(sources), [str(path) for path in sources]

    aggregate_logs = args.work_dir / "replay" / "game_logs.jsonl"
    if aggregate_logs.is_file():
        sources.append(aggregate_logs)
        return load_game_log_records(sources), [str(path) for path in sources]

    if args.include_shard_logs:
        shard_logs = sorted((args.work_dir / "shards").glob("*/game_logs.json"))
        if shard_logs:
            sources.extend(shard_logs)
            return load_game_log_records(sources), [str(path) for path in sources]

    replay_path = args.replay or args.work_dir / "replay" / "trajectory-replay.npz"
    if replay_path.is_file():
        return load_replay_records(replay_path), [str(replay_path)]

    raise FileNotFoundError(
        "no game logs or trajectory replay found; pass --game-logs, --replay, or --work-dir"
    )


def load_game_log_records(paths: list[Path]) -> list[GameRecord]:
    records: list[GameRecord] = []
    for path in paths:
        records.extend(_load_one_game_log_path(path, start_index=len(records)))
    records.sort(key=lambda record: (math.inf if record.seed is None else record.seed, record.index))
    return [
        GameRecord(
            index=index,
            winner=record.winner,
            moves=record.moves,
            seed=record.seed,
            end_reason=record.end_reason,
            territory_scores=record.territory_scores,
            model_version=record.model_version,
            created_iteration=record.created_iteration,
        )
        for index, record in enumerate(records)
    ]


def _load_one_game_log_path(path: Path, *, start_index: int) -> list[GameRecord]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix == ".jsonl":
        items = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        loaded = json.loads(text)
        if not isinstance(loaded, list):
            raise ValueError(f"{path} must contain a JSON list or JSONL records")
        items = loaded
    records: list[GameRecord] = []
    for offset, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"{path}: game log item {offset} must be an object")
        records.append(_record_from_game_log(item, index=start_index + offset))
    return records


def _record_from_game_log(data: dict[str, Any], *, index: int) -> GameRecord:
    winner = int(data["winner"])
    if winner not in {1, 2}:
        raise ValueError(f"game log winner must be 1 or 2, got {winner}")
    moves = data.get("moves")
    territory = _territory_pair(data.get("territory_scores"))
    return GameRecord(
        index=index,
        seed=_optional_int(data.get("seed")),
        winner=winner,
        moves=len(moves) if isinstance(moves, list) else None,
        end_reason=_optional_int(data.get("end_reason")),
        territory_scores=territory,
    )


def load_replay_records(path: Path) -> list[GameRecord]:
    with np.load(path) as data:
        winners = np.asarray(data["episode_winners"], dtype=np.int64)
        offsets = np.asarray(data["episode_offsets"], dtype=np.int64)
        territory_scores = np.asarray(data["territory_scores"], dtype=np.int64)
        model_versions = (
            np.asarray(data["model_versions"], dtype=np.int64)
            if "model_versions" in data
            else None
        )
        created_iterations = (
            np.asarray(data["created_iterations"], dtype=np.int64)
            if "created_iterations" in data
            else None
        )

    records: list[GameRecord] = []
    for episode_index, winner in enumerate(winners.tolist()):
        if int(winner) not in {1, 2}:
            raise ValueError(f"replay episode winner must be 1 or 2, got {winner}")
        start = int(offsets[episode_index])
        end = int(offsets[episode_index + 1])
        model_version = None if model_versions is None else int(model_versions[start])
        created_iteration = (
            None if created_iterations is None else int(created_iterations[start])
        )
        records.append(
            GameRecord(
                index=episode_index,
                winner=int(winner),
                moves=max(0, end - start),
                territory_scores=(
                    int(territory_scores[episode_index, 0]),
                    int(territory_scores[episode_index, 1]),
                ),
                model_version=model_version,
                created_iteration=created_iteration,
            )
        )
    return records


def analyze_records(
    records: list[GameRecord],
    *,
    sources: list[str],
    recent_windows: list[int],
    include_model_breakdown: bool,
) -> dict[str, Any]:
    overall = summarize_records(records)
    payload: dict[str, Any] = {
        "sources": sources,
        "overall": overall,
        "recent_windows": [
            {
                "last_games": window,
                **summarize_records(records[-window:]),
            }
            for window in sorted({window for window in recent_windows if window > 0})
            if records
        ],
    }
    if include_model_breakdown:
        by_model = summarize_by(records, key=lambda record: record.model_version)
        if by_model:
            payload["by_model_version"] = by_model
        by_iteration = summarize_by(records, key=lambda record: record.created_iteration)
        if by_iteration:
            payload["by_created_iteration"] = by_iteration
    return payload


def summarize_records(records: list[GameRecord]) -> dict[str, Any]:
    games = len(records)
    blue_wins = sum(1 for record in records if record.winner == 1)
    orange_wins = sum(1 for record in records if record.winner == 2)
    moves = np.asarray(
        [record.moves for record in records if record.moves is not None],
        dtype=np.float32,
    )
    territory = [record.territory_scores for record in records if record.territory_scores is not None]
    end_reasons = Counter(
        str(record.end_reason)
        for record in records
        if record.end_reason is not None
    )
    return {
        "games": games,
        "blue_wins": blue_wins,
        "orange_wins": orange_wins,
        "blue_win_rate": _rate(blue_wins, games),
        "orange_win_rate": _rate(orange_wins, games),
        "blue_minus_orange_win_rate": _rate(blue_wins, games) - _rate(orange_wins, games),
        "winner_imbalance_abs": abs(_rate(blue_wins, games) - _rate(orange_wins, games)),
        "average_game_length": float(moves.mean()) if moves.size else None,
        "game_length": None if not moves.size else asdict(describe_array(moves)),
        "end_reasons": dict(sorted(end_reasons.items())),
        "territory": summarize_territory(territory),
    }


def summarize_by(
    records: list[GameRecord],
    *,
    key: Any,
) -> list[dict[str, Any]]:
    groups: dict[int, list[GameRecord]] = {}
    for record in records:
        value = key(record)
        if value is None:
            continue
        groups.setdefault(int(value), []).append(record)
    return [
        {"key": group_key, **summarize_records(group_records)}
        for group_key, group_records in sorted(groups.items())
    ]


def summarize_territory(scores: list[tuple[int, int]]) -> dict[str, float] | None:
    if not scores:
        return None
    array = np.asarray(scores, dtype=np.float32)
    margins = array[:, 0] - array[:, 1]
    winner_margins = np.abs(margins)
    return {
        "blue_score_mean": float(array[:, 0].mean()),
        "orange_score_mean": float(array[:, 1].mean()),
        "blue_minus_orange_mean": float(margins.mean()),
        "winner_margin_mean": float(winner_margins.mean()),
        "winner_margin_p95": float(np.percentile(winner_margins, 95)),
    }


def describe_array(values: np.ndarray) -> ArraySummary:
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        return ArraySummary(0, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan)
    return ArraySummary(
        count=int(array.size),
        mean=float(array.mean()),
        std=float(array.std()),
        min=float(array.min()),
        p05=float(np.percentile(array, 5)),
        p50=float(np.percentile(array, 50)),
        p95=float(np.percentile(array, 95)),
        max=float(array.max()),
    )


def _rate(count: int, total: int) -> float:
    return count / total if total > 0 else 0.0


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _territory_pair(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise ValueError("territory_scores must contain two values")
    return (int(value[0]), int(value[1]))


if __name__ == "__main__":
    main()
