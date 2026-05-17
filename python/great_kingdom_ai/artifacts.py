"""Artifact path helpers for self-play runs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path


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


__all__ = ["ArtifactPaths", "load_artifact_paths"]
