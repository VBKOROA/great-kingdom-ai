"""Checkpoint migration helpers for changing training-state policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import NoReturn

from great_kingdom_ai.train import (
    TrainingConfig,
    load_checkpoint_weights,
    load_training_config,
    save_checkpoint,
)


def migrate_checkpoint_training_state(
    source: str | Path,
    output: str | Path,
    config: TrainingConfig,
    *,
    force: bool = False,
) -> Path:
    """Copy model weights into a checkpoint with fresh optimizer/scheduler state."""
    source_path = Path(source)
    output_path = Path(output)
    if not source_path.is_file():
        raise FileNotFoundError(f"source checkpoint does not exist: {source_path}")
    if output_path.exists() and not force:
        raise FileExistsError(f"output checkpoint already exists: {output_path}")

    state = load_checkpoint_weights(source_path, config)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    if temporary_path.exists():
        temporary_path.unlink()
    try:
        save_checkpoint(state, temporary_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a checkpoint that preserves model weights but resets optimizer, "
            "scheduler, scaler, and step state from a TrainingConfig."
        )
    )
    parser.add_argument("--source", type=Path, required=True, help="Existing checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="Migrated checkpoint path")
    parser.add_argument("--config", type=Path, required=True, help="JSON TrainingConfig path")
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite output checkpoint if it already exists",
    )
    return parser


def main() -> NoReturn:
    args = build_parser().parse_args()
    config = load_training_config(args.config)
    output = migrate_checkpoint_training_state(
        args.source,
        args.output,
        config,
        force=args.force,
    )
    print(
        json.dumps(
            {
                "event": "checkpoint_migrated",
                "source": str(args.source),
                "output": str(output),
                "config": str(args.config),
                "step": 0,
                "lr_schedule": config.lr_schedule,
                "learning_rate": config.learning_rate,
            },
            sort_keys=True,
        )
    )
    raise SystemExit(0)


if __name__ == "__main__":
    main()


__all__ = [
    "build_parser",
    "main",
    "migrate_checkpoint_training_state",
]
