"""CLI for reanalyze target snapshot generation."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import NoReturn

from great_kingdom_ai.reanalyze.builder import reanalyze_replay
from great_kingdom_ai.reanalyze.config import ReanalyzeConfig
from great_kingdom_ai.search_reanalyze import SearchReanalyzeConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Refresh trajectory replay value targets with a checkpoint"
    )
    parser.add_argument("--replay", type=Path, required=True, help="Input trajectory replay .npz")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Model checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="Output target snapshot .npz")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--onnx-model",
        type=Path,
        default=None,
        help="Optional ONNX model for policy/value refresh evaluation",
    )
    parser.add_argument("--onnx-device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--onnx-max-batch-size", type=int, default=1024)
    parser.add_argument("--bootstrap-td-steps", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument(
        "--dynamic-horizon-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--dynamic-horizon-tau", type=float, default=0.3)
    parser.add_argument("--dynamic-horizon-total-steps", type=int, default=None)
    parser.add_argument(
        "--value-bootstrap-source",
        choices=["value_head", "mcts_root"],
        default="value_head",
    )
    parser.add_argument(
        "--model-version",
        type=int,
        default=None,
        help="Override snapshot model version; defaults to checkpoint step",
    )
    parser.add_argument("--no-compress", action="store_true", help="Write an uncompressed npz")
    parser.add_argument(
        "--search-reanalyze-fraction",
        type=float,
        default=0.0,
        help="Fraction of high-priority rows whose policy targets are refreshed with Rust search",
    )
    parser.add_argument(
        "--search-reanalyze-budget",
        type=int,
        default=None,
        help="Maximum number of rows to refresh with Rust search",
    )
    parser.add_argument("--search-reanalyze-simulations", type=int, default=32)
    parser.add_argument("--search-reanalyze-max-considered-actions", type=int, default=16)
    parser.add_argument("--search-reanalyze-leaf-batch-size", type=int, default=8)
    parser.add_argument("--search-reanalyze-root-batch-size", type=int, default=128)
    parser.add_argument("--search-reanalyze-seed", type=int, default=0)
    return parser


def main() -> NoReturn:
    args = build_parser().parse_args()
    config = ReanalyzeConfig(
        batch_size=args.batch_size,
        device=args.device,
        onnx_model_path=None if args.onnx_model is None else str(args.onnx_model),
        onnx_device=args.onnx_device,
        onnx_max_batch_size=args.onnx_max_batch_size,
        bootstrap_td_steps=args.bootstrap_td_steps,
        gamma=args.gamma,
        dynamic_horizon_enabled=args.dynamic_horizon_enabled,
        dynamic_horizon_tau=args.dynamic_horizon_tau,
        dynamic_horizon_total_steps=args.dynamic_horizon_total_steps,
        value_bootstrap_source=args.value_bootstrap_source,
        model_version=args.model_version,
        compressed=not args.no_compress,
        search=SearchReanalyzeConfig(
            fraction=args.search_reanalyze_fraction,
            budget=args.search_reanalyze_budget,
            simulations=args.search_reanalyze_simulations,
            max_considered_actions=args.search_reanalyze_max_considered_actions,
            leaf_batch_size=args.search_reanalyze_leaf_batch_size,
            root_batch_size=args.search_reanalyze_root_batch_size,
            seed=args.search_reanalyze_seed,
        ),
    )
    print(
        json.dumps(
            {
                "event": "reanalyze_config",
                "config": asdict(config),
                "replay": str(args.replay),
                "checkpoint": str(args.checkpoint),
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    summary = reanalyze_replay(
        replay_path=args.replay,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        config=config,
    )
    print(json.dumps({"event": "reanalyze_summary", **summary.to_dict()}, sort_keys=True))
    raise SystemExit(0)


__all__ = ["build_parser", "main"]
