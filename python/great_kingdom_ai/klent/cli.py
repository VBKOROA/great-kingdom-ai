"""Command line entry point for the synchronous KLENT trainer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import NoReturn

from great_kingdom_ai.klent.config import load_klent_train_config
from great_kingdom_ai.klent.trainer import run_klent_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="great-kingdom-klent-train",
        description="Run the synchronous KLENT collect/train iteration loop",
    )
    parser.add_argument("--config", type=Path, required=True, help="KLENT YAML config")
    parser.add_argument("--iterations", type=int, default=1, help="total iterations to reach")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="ignore the latest checkpoint and start a fresh run",
    )
    return parser


def main() -> NoReturn:
    args = build_parser().parse_args()
    config = load_klent_train_config(args.config)
    summaries = run_klent_training(
        config,
        iterations=args.iterations,
        resume=not args.no_resume,
    )
    print(
        json.dumps(
            {
                "event": "klent_train",
                "iterations": [summary.to_dict() for summary in summaries],
            },
            sort_keys=True,
        )
    )
    raise SystemExit(0)


if __name__ == "__main__":
    main()


__all__ = ["build_parser", "main"]