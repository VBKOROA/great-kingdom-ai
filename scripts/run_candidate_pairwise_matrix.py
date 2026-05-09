"""Run a pairwise arena matrix over selected candidate checkpoints."""

from __future__ import annotations

import argparse
import gc
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NoReturn

from great_kingdom_ai.evaluate import (
    ArenaConfig,
    load_arena_config,
    load_model_from_checkpoint,
    run_arena,
    save_arena_report,
)

DEFAULT_WORK_DIR = Path("data/runpod/pure-gumbel-medium-plus")
DEFAULT_ARENA_CONFIG = Path("configs/runpod/arena.json")
_CANDIDATE_RE = re.compile(r"candidate-(\d+)\.pt$")


@dataclass(frozen=True)
class PairwiseMatchResult:
    match_index: int
    candidate: Path
    baseline: Path
    report_path: Path
    candidate_win_rate: float
    baseline_win_rate: float
    candidate_blue_games: int
    candidate_blue_wins: int
    candidate_orange_games: int
    candidate_orange_wins: int
    baseline_blue_games: int
    baseline_blue_wins: int
    baseline_orange_games: int
    baseline_orange_wins: int
    candidate_blue_win_rate: float
    candidate_orange_win_rate: float
    baseline_blue_win_rate: float
    baseline_orange_win_rate: float
    games: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "match_index": self.match_index,
            "candidate": str(self.candidate),
            "baseline": str(self.baseline),
            "report_path": str(self.report_path),
            "candidate_win_rate": self.candidate_win_rate,
            "baseline_win_rate": self.baseline_win_rate,
            "candidate_blue_games": self.candidate_blue_games,
            "candidate_blue_wins": self.candidate_blue_wins,
            "candidate_orange_games": self.candidate_orange_games,
            "candidate_orange_wins": self.candidate_orange_wins,
            "baseline_blue_games": self.baseline_blue_games,
            "baseline_blue_wins": self.baseline_blue_wins,
            "baseline_orange_games": self.baseline_orange_games,
            "baseline_orange_wins": self.baseline_orange_wins,
            "candidate_blue_win_rate": self.candidate_blue_win_rate,
            "candidate_orange_win_rate": self.candidate_orange_win_rate,
            "baseline_blue_win_rate": self.baseline_blue_win_rate,
            "baseline_orange_win_rate": self.baseline_orange_win_rate,
            "games": self.games,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run every selected candidate checkpoint against every earlier selected checkpoint, "
            "then write a pairwise win-rate matrix and ranking summary."
        )
    )
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--candidate-dir", type=Path, default=None)
    parser.add_argument("--arena-config", type=Path, default=DEFAULT_ARENA_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--candidates",
        type=Path,
        nargs="+",
        default=None,
        help="explicit checkpoint list; bypasses candidate-dir/glob/interval selection",
    )
    parser.add_argument("--glob", default="candidate-*.pt")
    parser.add_argument("--interval", type=int, default=1, help="take every Nth sorted candidate")
    parser.add_argument("--start-iteration", type=int, default=None)
    parser.add_argument("--end-iteration", type=int, default=None)
    parser.add_argument(
        "--include-latest",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="include the latest candidate even if interval skips it",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="keep only the last N selected candidates after interval filtering",
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--games", type=int, default=None)
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument("--gumbel-simulations", type=int, default=None)
    parser.add_argument("--gumbel-max-considered-actions", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="rerun existing match reports")
    return parser


def main() -> NoReturn:
    args = build_parser().parse_args()
    candidate_dir = args.candidate_dir or args.work_dir / "checkpoints" / "candidates"
    output_dir = args.output_dir or args.work_dir / "reports" / "candidate-pairwise"
    output_dir.mkdir(parents=True, exist_ok=True)

    arena_config = _load_effective_arena_config(
        args.arena_config,
        device=args.device,
        games=args.games,
        seed_start=args.seed_start,
        gumbel_simulations=args.gumbel_simulations,
        gumbel_max_considered_actions=args.gumbel_max_considered_actions,
    )
    candidates = (
        _validated_candidates(args.candidates)
        if args.candidates is not None
        else select_candidate_checkpoints(
            candidate_dir,
            glob=args.glob,
            interval=args.interval,
            start_iteration=args.start_iteration,
            end_iteration=args.end_iteration,
            include_latest=args.include_latest,
        )
    )
    if args.max_candidates is not None:
        if args.max_candidates <= 0:
            raise ValueError("max-candidates must be positive")
        candidates = candidates[-args.max_candidates :]
    if len(candidates) < 2:
        raise SystemExit("at least two candidate checkpoints are required")

    print(
        json.dumps(
            {
                "event": "candidate_pairwise_start",
                "candidate_count": len(candidates),
                "pair_count": len(candidates) * (len(candidates) - 1) // 2,
                "candidates": [str(path) for path in candidates],
                "arena_config": asdict(arena_config),
                "output_dir": str(output_dir),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    matches: list[PairwiseMatchResult] = []
    match_index = 0
    for baseline_index, baseline in enumerate(candidates):
        for candidate in candidates[baseline_index + 1 :]:
            match_index += 1
            report_path = output_dir / _match_report_name(match_index, candidate, baseline)
            result = _run_or_load_pairwise_match(
                match_index=match_index,
                candidate=candidate,
                baseline=baseline,
                report_path=report_path,
                arena_config=arena_config,
                force=args.force,
            )
            matches.append(result)
            print(
                json.dumps(
                    {"event": "candidate_pairwise_match", **result.to_dict()},
                    sort_keys=True,
                ),
                flush=True,
            )

    summary = summarize_pairwise(candidates, matches)
    payload = {
        "event": "candidate_pairwise_summary",
        "candidate_count": len(candidates),
        "pair_count": len(matches),
        "candidates": [str(path) for path in candidates],
        "matches": [match.to_dict() for match in matches],
        **summary,
    }
    _write_json(output_dir / "summary.json", payload)
    print(json.dumps(payload, sort_keys=True), flush=True)
    raise SystemExit(0)


def select_candidate_checkpoints(
    candidate_dir: Path,
    *,
    glob: str,
    interval: int,
    start_iteration: int | None,
    end_iteration: int | None,
    include_latest: bool,
) -> list[Path]:
    if interval <= 0:
        raise ValueError("interval must be positive")
    if not candidate_dir.is_dir():
        raise FileNotFoundError(f"candidate directory does not exist: {candidate_dir}")

    candidates = sorted(candidate_dir.glob(glob), key=_candidate_sort_key)
    filtered = [
        path
        for path in candidates
        if _iteration_in_range(
            _candidate_iteration(path),
            start=start_iteration,
            end=end_iteration,
        )
    ]
    selected = filtered[::interval]
    if include_latest and filtered and filtered[-1] not in selected:
        selected.append(filtered[-1])
    return selected


def summarize_pairwise(
    candidates: list[Path],
    matches: list[PairwiseMatchResult],
) -> dict[str, Any]:
    matrix: dict[str, dict[str, float | None]] = {
        _candidate_id(candidate): {_candidate_id(other): None for other in candidates}
        for candidate in candidates
    }
    side_totals: dict[str, dict[str, int]] = {
        _candidate_id(candidate): {
            "blue_wins": 0,
            "blue_games": 0,
            "orange_wins": 0,
            "orange_games": 0,
        }
        for candidate in candidates
    }

    for candidate in candidates:
        matrix[_candidate_id(candidate)][_candidate_id(candidate)] = 0.5

    for match in matches:
        candidate_id = _candidate_id(match.candidate)
        baseline_id = _candidate_id(match.baseline)
        matrix[candidate_id][baseline_id] = match.candidate_win_rate
        matrix[baseline_id][candidate_id] = match.baseline_win_rate

        _add_side_result(
            side_totals[candidate_id],
            blue_wins=match.candidate_blue_wins,
            blue_games=match.candidate_blue_games,
            orange_wins=match.candidate_orange_wins,
            orange_games=match.candidate_orange_games,
        )
        _add_side_result(
            side_totals[baseline_id],
            blue_wins=match.baseline_blue_wins,
            blue_games=match.baseline_blue_games,
            orange_wins=match.baseline_orange_wins,
            orange_games=match.baseline_orange_games,
        )

    ranking = [_candidate_summary(candidate, matrix, side_totals) for candidate in candidates]
    ranking.sort(
        key=lambda row: (
            float(row["average_win_rate"]),
            float(row["worst_win_rate"]),
            float(row["best_win_rate"]),
            str(row["candidate"]),
        ),
        reverse=True,
    )
    return {
        "matrix": matrix,
        "ranking": ranking,
        "winner": str(ranking[0]["candidate"]) if ranking else None,
    }


def _run_or_load_pairwise_match(
    *,
    match_index: int,
    candidate: Path,
    baseline: Path,
    report_path: Path,
    arena_config: ArenaConfig,
    force: bool,
) -> PairwiseMatchResult:
    if report_path.exists() and not force:
        summary = _load_arena_summary(report_path)
    else:
        candidate_model = load_model_from_checkpoint(candidate, device=arena_config.device)
        baseline_model = load_model_from_checkpoint(baseline, device=arena_config.device)
        report = run_arena(
            candidate_model=candidate_model,
            best_model=baseline_model,
            config=arena_config,
        )
        save_arena_report(report, report_path)
        summary = report.summary.to_dict()
        del candidate_model, baseline_model
        _release_cuda_cache()

    return _match_from_summary(
        match_index=match_index,
        candidate=candidate,
        baseline=baseline,
        report_path=report_path,
        summary=summary,
    )


def _match_from_summary(
    *,
    match_index: int,
    candidate: Path,
    baseline: Path,
    report_path: Path,
    summary: dict[str, Any],
) -> PairwiseMatchResult:
    candidate_blue_games = int(summary["candidate_blue_games"])
    candidate_orange_games = int(summary["candidate_orange_games"])
    candidate_blue_wins = int(summary["candidate_blue_wins"])
    candidate_orange_wins = int(summary["candidate_orange_wins"])
    baseline_blue_games = candidate_orange_games
    baseline_orange_games = candidate_blue_games
    baseline_blue_wins = candidate_orange_games - candidate_orange_wins
    baseline_orange_wins = candidate_blue_games - candidate_blue_wins
    return PairwiseMatchResult(
        match_index=match_index,
        candidate=candidate,
        baseline=baseline,
        report_path=report_path,
        candidate_win_rate=float(summary["candidate_win_rate"]),
        baseline_win_rate=float(summary["best_win_rate"]),
        candidate_blue_games=candidate_blue_games,
        candidate_blue_wins=candidate_blue_wins,
        candidate_orange_games=candidate_orange_games,
        candidate_orange_wins=candidate_orange_wins,
        baseline_blue_games=baseline_blue_games,
        baseline_blue_wins=baseline_blue_wins,
        baseline_orange_games=baseline_orange_games,
        baseline_orange_wins=baseline_orange_wins,
        candidate_blue_win_rate=_side_win_rate(
            wins=candidate_blue_wins,
            games=candidate_blue_games,
        ),
        candidate_orange_win_rate=_side_win_rate(
            wins=candidate_orange_wins,
            games=candidate_orange_games,
        ),
        baseline_blue_win_rate=_side_win_rate(
            wins=baseline_blue_wins,
            games=baseline_blue_games,
        ),
        baseline_orange_win_rate=_side_win_rate(
            wins=baseline_orange_wins,
            games=baseline_orange_games,
        ),
        games=int(summary["games"]),
    )


def _candidate_summary(
    candidate: Path,
    matrix: dict[str, dict[str, float | None]],
    side_totals: dict[str, dict[str, int]],
) -> dict[str, Any]:
    candidate_id = _candidate_id(candidate)
    win_rates = [
        float(value)
        for opponent_id, value in matrix[candidate_id].items()
        if opponent_id != candidate_id and value is not None
    ]
    totals = side_totals[candidate_id]
    return {
        "candidate": str(candidate),
        "candidate_id": candidate_id,
        "matches": len(win_rates),
        "average_win_rate": sum(win_rates) / len(win_rates) if win_rates else 0.0,
        "worst_win_rate": min(win_rates) if win_rates else 0.0,
        "best_win_rate": max(win_rates) if win_rates else 0.0,
        "winning_match_count": sum(1 for rate in win_rates if rate > 0.5),
        "non_losing_match_count": sum(1 for rate in win_rates if rate >= 0.5),
        "blue_win_rate": _side_win_rate(
            wins=totals["blue_wins"],
            games=totals["blue_games"],
        ),
        "orange_win_rate": _side_win_rate(
            wins=totals["orange_wins"],
            games=totals["orange_games"],
        ),
    }


def _add_side_result(
    totals: dict[str, int],
    *,
    blue_wins: int,
    blue_games: int,
    orange_wins: int,
    orange_games: int,
) -> None:
    totals["blue_wins"] += blue_wins
    totals["blue_games"] += blue_games
    totals["orange_wins"] += orange_wins
    totals["orange_games"] += orange_games


def _side_win_rate(*, wins: int, games: int) -> float:
    return wins / games if games > 0 else 0.0


def _load_effective_arena_config(
    path: Path,
    *,
    device: str | None,
    games: int | None,
    seed_start: int | None,
    gumbel_simulations: int | None,
    gumbel_max_considered_actions: int | None,
) -> ArenaConfig:
    config = load_arena_config(path)
    data = asdict(config)
    overrides = {
        "device": device,
        "games": games,
        "seed_start": seed_start,
        "gumbel_simulations": gumbel_simulations,
        "gumbel_max_considered_actions": gumbel_max_considered_actions,
    }
    data.update({key: value for key, value in overrides.items() if value is not None})
    return ArenaConfig(**data)


def _validated_candidates(paths: list[Path]) -> list[Path]:
    candidates = [path.expanduser() for path in paths]
    for path in candidates:
        if not path.is_file():
            raise FileNotFoundError(f"candidate checkpoint does not exist: {path}")
    return candidates


def _candidate_sort_key(path: Path) -> tuple[int, str]:
    iteration = _candidate_iteration(path)
    return (-1 if iteration is None else iteration, path.name)


def _candidate_iteration(path: Path) -> int | None:
    match = _CANDIDATE_RE.search(path.name)
    return int(match.group(1)) if match else None


def _iteration_in_range(
    iteration: int | None,
    *,
    start: int | None,
    end: int | None,
) -> bool:
    if iteration is None:
        return start is None and end is None
    if start is not None and iteration < start:
        return False
    if end is not None and iteration > end:
        return False
    return True


def _match_report_name(match_index: int, candidate: Path, baseline: Path) -> str:
    return (
        f"pair-{match_index:03d}-"
        f"{candidate.stem}-vs-{baseline.stem}-arena.json"
    )


def _candidate_id(path: Path) -> str:
    return path.stem


def _load_arena_summary(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = data.get("summary")
    if not isinstance(summary, dict):
        raise ValueError(f"{path} does not contain an arena summary")
    return summary


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _release_cuda_cache() -> None:
    gc.collect()
    try:
        import torch
    except ModuleNotFoundError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
