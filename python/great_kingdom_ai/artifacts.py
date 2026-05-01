"""Artifact path and persistence helpers for self-play runs."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from great_kingdom_ai.replay_buffer import ReplayBuffer, ReplaySample
from great_kingdom_ai.self_play import GameLog


@dataclass(frozen=True)
class ArtifactPaths:
    replay_dir: Path
    checkpoint_dir: Path
    log_dir: Path

    @classmethod
    def from_mapping(cls, data: Mapping[str, str]) -> ArtifactPaths:
        return cls(
            replay_dir=Path(data["replay_dir"]),
            checkpoint_dir=Path(data["checkpoint_dir"]),
            log_dir=Path(data["log_dir"]),
        )

    def ensure_dirs(self) -> None:
        self.replay_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, str]:
        return {key: str(value) for key, value in asdict(self).items()}


def load_artifact_paths(path: str | Path) -> ArtifactPaths:
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)
    return ArtifactPaths.from_mapping(data)


def save_self_play_artifact(
    output_dir: str | Path,
    *,
    logs: Sequence[GameLog],
    samples: Sequence[ReplaySample],
    replay_capacity: int | None = None,
) -> dict[str, Path]:
    """Save replay samples and game logs so a later process can resume from disk."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    capacity = replay_capacity if replay_capacity is not None else max(1, len(samples))
    buffer = ReplayBuffer(capacity=capacity)
    buffer.extend(samples)

    replay_path = destination / "replay.npz"
    log_path = destination / "game_logs.json"
    buffer.save(replay_path)
    with log_path.open("w", encoding="utf-8") as file:
        json.dump([log.to_dict() for log in logs], file, indent=2, sort_keys=True)

    return {"replay": replay_path, "logs": log_path}


def load_self_play_artifact(output_dir: str | Path) -> tuple[ReplayBuffer, list[dict[str, Any]]]:
    source = Path(output_dir)
    buffer = ReplayBuffer.load(source / "replay.npz")
    with (source / "game_logs.json").open("r", encoding="utf-8") as file:
        logs = json.load(file)
    if not isinstance(logs, list):
        raise ValueError("game_logs.json must contain a list")
    return buffer, logs
