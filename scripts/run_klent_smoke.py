"""Run a tiny CPU KLENT zero-search collect/train smoke test.

This is the local smoke entry point from the KLENT implementation plan:
``.venv/bin/python -m scripts.run_klent_smoke``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import NoReturn

from great_kingdom_ai.klent.types import KlentConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a short KLENT collect/train smoke test")
    parser.add_argument("--work-dir", type=Path, default=Path("data/klent-smoke"))
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--model-preset", default="small_klent")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-transitions", type=int, default=32)
    parser.add_argument("--max-games", type=int, default=4)
    parser.add_argument("--fit-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--rust-actor",
        action="store_true",
        help="collect games with the Rust zero-search actor (requires actor ONNX export)",
    )
    return parser


def main() -> NoReturn:
    args = build_parser().parse_args()
    from great_kingdom_ai.klent.trainer import KlentTrainConfig, run_klent_training

    config = KlentTrainConfig(
        work_dir=args.work_dir,
        klent=KlentConfig(),
        model_preset=args.model_preset,
        device=args.device,
        seed=args.seed,
        min_transitions=args.min_transitions,
        max_games_per_iteration=args.max_games,
        fit_epochs=args.fit_epochs,
        batch_size=args.batch_size,
        max_turns=args.max_turns,
        use_rust_actor=args.rust_actor,
        export_onnx=False,
        rust_self_play_batch_size=args.max_games,
    )
    summaries = run_klent_training(
        config,
        iterations=args.iterations,
        resume=not args.no_resume,
    )
    print(
        json.dumps(
            {"event": "klent_smoke", "iterations": [summary.to_dict() for summary in summaries]},
            sort_keys=True,
        )
    )
    raise SystemExit(0)


if __name__ == "__main__":
    main()