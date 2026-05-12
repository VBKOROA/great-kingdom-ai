"""Prune regenerable Runpod training artifacts.

The script is dry-run by default. Pass ``--delete`` to actually remove files.
It deliberately keeps trajectory replay and active checkpoints because those are
the expensive state needed to resume v2 training.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import NoReturn

from great_kingdom_ai.runpod_pruning import (
    PruneItem,
    collect_prune_items,
    format_summary,
    prune_items,
)

DEFAULT_WORK_DIR = Path("data/runpod/train-v2-recommended-medium-plus")
__all__ = ["PruneItem", "collect_prune_items", "format_summary", "prune_items"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prune regenerable Runpod artifacts")
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--delete", action="store_true", help="actually delete selected paths")
    parser.add_argument("--keep-targets", type=int, default=2)
    parser.add_argument("--keep-candidates", type=int, default=3)
    parser.add_argument("--keep-onnx", type=int, default=1)
    parser.add_argument(
        "--include-build-cache",
        action="store_true",
        help="also prune local test/type/build caches such as .pytest_cache and rust target",
    )
    return parser


def main() -> NoReturn:
    args = build_parser().parse_args()
    items = collect_prune_items(
        args.work_dir,
        keep_targets=args.keep_targets,
        keep_candidates=args.keep_candidates,
        keep_onnx=args.keep_onnx,
        include_build_cache=args.include_build_cache,
    )
    print(format_summary(items, delete=args.delete))
    prune_items(items, delete=args.delete)
    raise SystemExit(0)


if __name__ == "__main__":
    main()
