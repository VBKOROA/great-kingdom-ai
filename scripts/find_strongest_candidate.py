"""Find the strongest checkpoint among saved candidate checkpoints."""

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
class MatchResult:
    match_index: int
    champion_before: Path
    challenger: Path
    winner: Path
    report_path: Path
    challenger_win_rate: float
    champion_win_rate: float
    challenger_blue_win_rate: float
    challenger_orange_win_rate: float
    games: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "match_index": self.match_index,
            "champion_before": str(self.champion_before),
            "challenger": str(self.challenger),
            "winner": str(self.winner),
            "report_path": str(self.report_path),
            "challenger_win_rate": self.challenger_win_rate,
            "champion_win_rate": self.champion_win_rate,
            "challenger_blue_win_rate": self.challenger_blue_win_rate,
            "challenger_orange_win_rate": self.challenger_orange_win_rate,
            "games": self.games,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a king-of-the-hill arena over candidate checkpoints. Candidates are selected "
            "from candidate-*.pt files by iteration interval, then each selected checkpoint "
            "challenges the current champion."
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
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--games", type=int, default=None)
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument("--gumbel-simulations", type=int, default=None)
    parser.add_argument(
        "--win-threshold",
        type=float,
        default=0.5,
        help="challenger must reach this win rate as both Blue and Orange to become champion",
    )
    parser.add_argument("--force", action="store_true", help="rerun existing match reports")
    return parser


def main() -> NoReturn:
    args = build_parser().parse_args()
    candidate_dir = args.candidate_dir or args.work_dir / "checkpoints" / "candidates"
    output_dir = args.output_dir or args.work_dir / "reports" / "candidate-strength"
    output_dir.mkdir(parents=True, exist_ok=True)

    arena_config = _load_effective_arena_config(
        args.arena_config,
        device=args.device,
        games=args.games,
        seed_start=args.seed_start,
        gumbel_simulations=args.gumbel_simulations,
        promotion_threshold=args.win_threshold,
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
    if len(candidates) < 2:
        raise SystemExit("at least two candidate checkpoints are required")

    print(
        json.dumps(
            {
                "event": "candidate_strength_start",
                "candidate_count": len(candidates),
                "candidates": [str(path) for path in candidates],
                "arena_config": asdict(arena_config),
                "output_dir": str(output_dir),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    champion = candidates[0]
    matches: list[MatchResult] = []
    for match_index, challenger in enumerate(candidates[1:], start=1):
        report_path = output_dir / _match_report_name(match_index, champion, challenger)
        result = _run_or_load_match(
            match_index=match_index,
            champion=champion,
            challenger=challenger,
            report_path=report_path,
            arena_config=arena_config,
            force=args.force,
        )
        matches.append(result)
        champion = result.winner
        print(json.dumps({"event": "candidate_strength_match", **result.to_dict()}, sort_keys=True))

    summary = {
        "event": "candidate_strength_summary",
        "winner": str(champion),
        "candidate_count": len(candidates),
        "matches": [match.to_dict() for match in matches],
    }
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True))
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


def _run_or_load_match(
    *,
    match_index: int,
    champion: Path,
    challenger: Path,
    report_path: Path,
    arena_config: ArenaConfig,
    force: bool,
) -> MatchResult:
    if report_path.exists() and not force:
        summary = _load_arena_summary(report_path)
    else:
        challenger_model = load_model_from_checkpoint(challenger, device=arena_config.device)
        champion_model = load_model_from_checkpoint(champion, device=arena_config.device)
        report = run_arena(
            candidate_model=challenger_model,
            best_model=champion_model,
            config=arena_config,
        )
        save_arena_report(report, report_path)
        summary = report.summary.to_dict()
        del challenger_model, champion_model
        _release_cuda_cache()

    challenger_win_rate = float(summary["candidate_win_rate"])
    champion_win_rate = float(summary["best_win_rate"])
    challenger_blue_win_rate = _side_win_rate(
        wins=int(summary["candidate_blue_wins"]),
        games=int(summary["candidate_blue_games"]),
    )
    challenger_orange_win_rate = _side_win_rate(
        wins=int(summary["candidate_orange_wins"]),
        games=int(summary["candidate_orange_games"]),
    )
    challenger_promoted = (
        challenger_blue_win_rate >= arena_config.promotion_threshold
        and challenger_orange_win_rate >= arena_config.promotion_threshold
    )
    winner = challenger if challenger_promoted else champion
    return MatchResult(
        match_index=match_index,
        champion_before=champion,
        challenger=challenger,
        winner=winner,
        report_path=report_path,
        challenger_win_rate=challenger_win_rate,
        champion_win_rate=champion_win_rate,
        challenger_blue_win_rate=challenger_blue_win_rate,
        challenger_orange_win_rate=challenger_orange_win_rate,
        games=int(summary["games"]),
    )


def _side_win_rate(*, wins: int, games: int) -> float:
    return wins / games if games > 0 else 0.0


def _load_effective_arena_config(
    path: Path,
    *,
    device: str | None,
    games: int | None,
    seed_start: int | None,
    gumbel_simulations: int | None,
    promotion_threshold: float,
) -> ArenaConfig:
    config = load_arena_config(path)
    data = asdict(config)
    overrides = {
        "device": device,
        "games": games,
        "seed_start": seed_start,
        "gumbel_simulations": gumbel_simulations,
        "promotion_threshold": promotion_threshold,
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


def _match_report_name(match_index: int, champion: Path, challenger: Path) -> str:
    return (
        f"match-{match_index:03d}-"
        f"{challenger.stem}-vs-{champion.stem}-arena.json"
    )


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
