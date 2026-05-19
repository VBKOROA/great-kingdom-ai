"""Run a pairwise arena matrix over every PyTorch checkpoint in a folder."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, NoReturn

from run_candidate_pairwise_matrix import (
    DEFAULT_ARENA_CONFIG,
    PairwiseMatchResult,
    _load_effective_arena_config,
    _run_or_load_pairwise_match,
    _write_json,
    summarize_pairwise,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run every .pt checkpoint in a folder against every other .pt checkpoint, "
            "then write a pairwise arena matrix and ranking summary."
        )
    )
    parser.add_argument("checkpoint_dir", type=Path, help="folder containing .pt checkpoints")
    parser.add_argument("--arena-config", type=Path, default=DEFAULT_ARENA_CONFIG)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="report output folder; defaults to CHECKPOINT_DIR/matrix-arena-reports",
    )
    parser.add_argument("--glob", default="*.pt", help="checkpoint filename pattern")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="search checkpoint_dir recursively using the glob pattern",
    )
    parser.add_argument(
        "--max-checkpoints",
        type=int,
        default=None,
        help="keep only the last N checkpoints after sorting",
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
    checkpoints = select_pt_checkpoints(
        args.checkpoint_dir,
        glob=args.glob,
        recursive=args.recursive,
    )
    if args.max_checkpoints is not None:
        if args.max_checkpoints <= 0:
            raise ValueError("max-checkpoints must be positive")
        checkpoints = checkpoints[-args.max_checkpoints :]
    if len(checkpoints) < 2:
        raise SystemExit("at least two .pt checkpoints are required")

    output_dir = args.output_dir or args.checkpoint_dir / "matrix-arena-reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    arena_config = _load_effective_arena_config(
        args.arena_config,
        device=args.device,
        games=args.games,
        seed_start=args.seed_start,
        gumbel_simulations=args.gumbel_simulations,
        gumbel_max_considered_actions=args.gumbel_max_considered_actions,
    )

    print(
        json.dumps(
            {
                "event": "pt_folder_matrix_start",
                "checkpoint_dir": str(args.checkpoint_dir),
                "checkpoint_count": len(checkpoints),
                "pair_count": len(checkpoints) * (len(checkpoints) - 1) // 2,
                "checkpoints": [str(path) for path in checkpoints],
                "arena_config": asdict(arena_config),
                "output_dir": str(output_dir),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    matches: list[PairwiseMatchResult] = []
    match_index = 0
    for baseline_index, baseline in enumerate(checkpoints):
        for candidate in checkpoints[baseline_index + 1 :]:
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
                    {"event": "pt_folder_matrix_match", **result.to_dict()},
                    sort_keys=True,
                ),
                flush=True,
            )

    summary = summarize_pairwise(checkpoints, matches)
    payload: dict[str, Any] = {
        "event": "pt_folder_matrix_summary",
        "checkpoint_dir": str(args.checkpoint_dir),
        "checkpoint_count": len(checkpoints),
        "pair_count": len(matches),
        "checkpoints": [str(path) for path in checkpoints],
        "matches": [match.to_dict() for match in matches],
        **summary,
    }
    _write_json(output_dir / "summary.json", payload)
    print(json.dumps(payload, sort_keys=True), flush=True)
    raise SystemExit(0)


def select_pt_checkpoints(
    checkpoint_dir: Path,
    *,
    glob: str,
    recursive: bool,
) -> list[Path]:
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {checkpoint_dir}")
    paths = checkpoint_dir.rglob(glob) if recursive else checkpoint_dir.glob(glob)
    return sorted((path for path in paths if path.is_file()), key=_checkpoint_sort_key)


def _checkpoint_sort_key(path: Path) -> tuple[str, str]:
    return (str(path.parent), path.name)


def _match_report_name(match_index: int, candidate: Path, baseline: Path) -> str:
    return f"pair-{match_index:03d}-{candidate.stem}-vs-{baseline.stem}-arena.json"


if __name__ == "__main__":
    main()
