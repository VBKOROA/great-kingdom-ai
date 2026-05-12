"""Trajectory replay migration helpers for capacity-only changes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import NoReturn

import numpy as np

from great_kingdom_ai.trajectory_replay import TrajectoryReplayStore


def migrate_trajectory_replay_capacity(
    source: str | Path,
    output: str | Path,
    capacity: int,
    *,
    force: bool = False,
) -> Path:
    """Rewrite trajectory replay metadata with a larger preservation-safe capacity."""
    source_path = Path(source)
    output_path = Path(output)
    if not source_path.is_file():
        raise FileNotFoundError(f"source trajectory replay does not exist: {source_path}")
    if output_path.exists() and not force:
        raise FileExistsError(f"output trajectory replay already exists: {output_path}")
    if capacity <= 0:
        raise ValueError("capacity must be positive")

    replay = TrajectoryReplayStore.load(source_path)
    if capacity < len(replay):
        raise ValueError(
            "capacity must be at least the current transition count "
            f"to preserve replay data: capacity={capacity}, transitions={len(replay)}"
        )

    payload = replay.to_payload()
    payload["capacity"] = np.asarray(capacity, dtype=np.int64)
    migrated = TrajectoryReplayStore.from_payload(payload)

    temporary_path = output_path.with_name(f".{output_path.stem}.tmp{output_path.suffix}")
    if temporary_path.exists():
        temporary_path.unlink()
    try:
        migrated.save(temporary_path, compressed=False)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite a trajectory replay file with a new capacity while preserving all "
            "existing transitions."
        )
    )
    parser.add_argument("--source", type=Path, required=True, help="Existing replay .npz")
    parser.add_argument("--output", type=Path, required=True, help="Migrated replay .npz")
    parser.add_argument("--capacity", type=int, required=True, help="New replay capacity")
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite output replay if it already exists",
    )
    return parser


def main() -> NoReturn:
    args = build_parser().parse_args()
    source = TrajectoryReplayStore.load(args.source)
    output = migrate_trajectory_replay_capacity(
        args.source,
        args.output,
        args.capacity,
        force=args.force,
    )
    print(
        json.dumps(
            {
                "event": "trajectory_replay_capacity_migrated",
                "source": str(args.source),
                "output": str(output),
                "old_capacity": source.capacity,
                "new_capacity": args.capacity,
                "transitions": len(source),
                "episodes": source.episode_count,
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
    "migrate_trajectory_replay_capacity",
]
