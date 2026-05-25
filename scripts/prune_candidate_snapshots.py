"""Prune candidate snapshots to keep only top-K survivors based on iterative evaluation."""

from __future__ import annotations

import argparse
import gc
import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any, NoReturn

from great_kingdom_ai.evaluate import (
    ArenaConfig,
    load_arena_config,
    load_model_from_checkpoint,
    run_arena,
    save_arena_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prune candidate snapshots to keep only the best ones."
    )
    parser.add_argument("--arena-config", type=Path, required=True, help="ArenaConfig json path")
    parser.add_argument("--top-k", type=int, required=True, help="Maximum number of survivors")
    parser.add_argument("--snapshot-dir", type=Path, required=True, help="Candidate snapshots dir")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--force", action="store_true", help="Rerun existing match reports")
    return parser


def main() -> NoReturn:
    args = build_parser().parse_args()

    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")

    snapshot_dir = args.snapshot_dir
    if not snapshot_dir.is_dir():
        raise FileNotFoundError(f"Snapshot directory not found: {snapshot_dir}")

    report_dir = default_report_dir(snapshot_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    arena_config = load_arena_config(args.arena_config)
    if args.device:
        arena_config = ArenaConfig(**{**asdict(arena_config), "device": args.device})

    all_snapshots = sorted(snapshot_dir.glob("*.pt"), key=snapshot_sort_key)
    if len(all_snapshots) < 2:
        raise SystemExit("At least two candidate checkpoints are required")

    def evaluate_fn(candidate: Path, pivot: Path, round_idx: int) -> dict[str, Any]:
        report_path = (
            report_dir / f"round-{round_idx:03d}-{candidate.stem}-vs-{pivot.stem}-arena.json"
        )

        if report_path.exists() and not args.force:
            summary = load_arena_summary(report_path)
        else:
            candidate_model = load_model_from_checkpoint(candidate, device=arena_config.device)
            pivot_model = load_model_from_checkpoint(pivot, device=arena_config.device)
            report = run_arena(
                candidate_model=candidate_model,
                best_model=pivot_model,
                config=arena_config,
            )
            save_arena_report(report, report_path)
            summary = report.summary.to_dict()
            del candidate_model, pivot_model
            release_cuda_cache()

        return {"win_rate": float(summary["candidate_win_rate"]), "report_path": str(report_path)}

    survivors_payload = prune_candidates(
        all_snapshots=all_snapshots,
        top_k=args.top_k,
        evaluate_fn=evaluate_fn,
        arena_config_path=args.arena_config,
        snapshot_dir=snapshot_dir,
    )

    for round_summary in survivors_payload["rounds"]:
        r_idx = round_summary["round"]
        write_json(report_dir / f"round-{r_idx:03d}-summary.json", round_summary)

    write_json(report_dir / "survivors.json", survivors_payload)
    print(json.dumps(survivors_payload, indent=2, sort_keys=True))
    raise SystemExit(0)


def prune_candidates(
    *,
    all_snapshots: list[Path],
    top_k: int,
    evaluate_fn: Callable[[Path, Path, int], dict[str, Any]],
    arena_config_path: Path,
    snapshot_dir: Path,
) -> dict[str, Any]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    active_candidates = list(all_snapshots)
    initial_count = len(active_candidates)
    round_summaries = []

    round_index = 1
    while len(active_candidates) > top_k:
        pivot_index = (0 + len(active_candidates) - 1) // 2
        pivot = active_candidates[pivot_index]

        matches = []
        survivors = []

        print(f"Round {round_index}: {len(active_candidates)} candidates, pivot: {pivot.name}")

        for candidate in active_candidates:
            if candidate == pivot:
                survivors.append(candidate)
                matches.append(
                    {
                        "candidate": candidate.name,
                        "pivot": pivot.name,
                        "candidate_win_rate": 0.5,
                        "survived": True,
                        "report_path": None,
                    }
                )
                continue

            eval_result = evaluate_fn(candidate, pivot, round_index)
            win_rate = eval_result["win_rate"]
            survived = win_rate >= 0.5
            if survived:
                survivors.append(candidate)

            matches.append(
                {
                    "candidate": candidate.name,
                    "pivot": pivot.name,
                    "candidate_win_rate": win_rate,
                    "survived": survived,
                    "report_path": eval_result["report_path"],
                }
            )

        survivors = sorted(survivors, key=snapshot_sort_key)

        round_summary = {
            "round": round_index,
            "pivot": pivot.name,
            "before_count": len(active_candidates),
            "after_count": len(survivors),
            "matches": matches,
        }
        round_summaries.append(round_summary)

        if len(survivors) == len(active_candidates):
            print("No shrinking occurred this round. Falling back to ranking.")
            candidate_to_win_rate = {m["candidate"]: m["candidate_win_rate"] for m in matches}

            ranked_candidates = sorted(
                active_candidates,
                key=lambda c: (
                    _fallback_win_rate(c, pivot, candidate_to_win_rate),
                    snapshot_mtime(c),
                    c.name,
                ),
                reverse=True,
            )
            active_candidates = sorted(ranked_candidates[:top_k], key=snapshot_sort_key)
            break

        active_candidates = survivors
        round_index += 1

    final_survivors = [c.name for c in active_candidates]
    return {
        "arena_config": str(arena_config_path),
        "top_k": top_k,
        "snapshot_dir": str(snapshot_dir),
        "initial_count": initial_count,
        "final_count": len(final_survivors),
        "survivors": final_survivors,
        "survivor_details": [
            {"path": str(candidate), "mtime": snapshot_mtime(candidate)}
            for candidate in active_candidates
        ],
        "rounds": round_summaries,
    }


def snapshot_mtime(path: Path) -> float:
    return path.stat().st_mtime


def snapshot_sort_key(path: Path) -> tuple[float, str]:
    return (snapshot_mtime(path), path.name)


def _fallback_win_rate(
    candidate: Path,
    pivot: Path,
    candidate_to_win_rate: dict[str, float],
) -> float:
    if candidate == pivot:
        return max(candidate_to_win_rate.values(), default=0.5)
    return candidate_to_win_rate[candidate.name]


def default_report_dir(snapshot_dir: Path) -> Path:
    if snapshot_dir.name == "candidates" and snapshot_dir.parent.name == "checkpoints":
        return snapshot_dir.parent.parent / "reports" / "pruning"
    return snapshot_dir.parent / "reports" / "pruning"


def load_arena_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    summary = data.get("summary")
    if not isinstance(summary, dict):
        raise ValueError(f"{path} does not contain an arena summary")
    return summary


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def release_cuda_cache() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
