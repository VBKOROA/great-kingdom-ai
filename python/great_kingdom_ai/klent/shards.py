"""KLENT shard metadata and atomic shard persistence."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from great_kingdom_ai.klent.types import KlentConfig
from great_kingdom_ai.replay.trajectory import TrajectoryReplayStore

KLENT_ALGORITHM = "klent"
KLENT_SCHEMA_VERSION = 1
KLENT_SHARD_METADATA_SUFFIX = ".metadata.json"


@dataclass(frozen=True)
class KlentShardMetadata:
    """Metadata required to accept a shard as KLENT self-play data."""

    iteration: int
    model_version: int
    alpha: float
    beta: float
    lambda_param: float
    gamma: float
    transitions: int
    games: int
    created_at: str
    algorithm: str = KLENT_ALGORITHM
    schema_version: int = KLENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.algorithm != KLENT_ALGORITHM:
            raise ValueError(f"algorithm must be {KLENT_ALGORITHM!r}, got {self.algorithm!r}")
        if self.schema_version != KLENT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported KLENT schema version: {self.schema_version}"
            )
        if self.iteration < 0:
            raise ValueError("iteration must be non-negative")
        if self.model_version < 0:
            raise ValueError("model_version must be non-negative")
        if self.transitions < 0:
            raise ValueError("transitions must be non-negative")
        if self.games < 0:
            raise ValueError("games must be non-negative")
        KlentConfig(
            alpha=self.alpha,
            beta=self.beta,
            lambda_param=self.lambda_param,
            gamma=self.gamma,
        )

    @classmethod
    def from_config(
        cls,
        config: KlentConfig,
        *,
        iteration: int,
        model_version: int,
        transitions: int,
        games: int,
        created_at: str | None = None,
    ) -> KlentShardMetadata:
        return cls(
            iteration=iteration,
            model_version=model_version,
            alpha=config.alpha,
            beta=config.beta,
            lambda_param=config.lambda_param,
            gamma=config.gamma,
            transitions=transitions,
            games=games,
            created_at=created_at or datetime.now(tz=UTC).isoformat(timespec="seconds"),
        )

    def matches_config(self, config: KlentConfig) -> bool:
        return (
            math.isclose(self.alpha, config.alpha, rel_tol=0.0, abs_tol=1e-9)
            and math.isclose(self.beta, config.beta, rel_tol=0.0, abs_tol=1e-9)
            and math.isclose(self.lambda_param, config.lambda_param, rel_tol=0.0, abs_tol=1e-9)
            and math.isclose(self.gamma, config.gamma, rel_tol=0.0, abs_tol=1e-9)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "schema_version": self.schema_version,
            "iteration": self.iteration,
            "model_version": self.model_version,
            "alpha": self.alpha,
            "beta": self.beta,
            "lambda_param": self.lambda_param,
            "gamma": self.gamma,
            "transitions": self.transitions,
            "games": self.games,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KlentShardMetadata:
        return cls(
            algorithm=str(data.get("algorithm", "")),
            schema_version=int(data["schema_version"]),
            iteration=int(data["iteration"]),
            model_version=int(data["model_version"]),
            alpha=float(data["alpha"]),
            beta=float(data["beta"]),
            lambda_param=float(data["lambda_param"]),
            gamma=float(data["gamma"]),
            transitions=int(data["transitions"]),
            games=int(data["games"]),
            created_at=str(data.get("created_at", "")),
        )


def shard_metadata_path(replay_path: str | Path) -> Path:
    """Return the per-shard metadata path (``<shard>.metadata.json``)."""
    path = Path(replay_path)
    return path.with_name(path.name + KLENT_SHARD_METADATA_SUFFIX)


def klent_config_hash(config: KlentConfig) -> str:
    """Stable tag stored on transitions to mark the KLENT target coefficients."""
    return (
        f"klent:alpha={config.alpha:g},beta={config.beta:g},"
        f"lambda={config.lambda_param:g},gamma={config.gamma:g}"
    )


def write_klent_shard_metadata(
    metadata: KlentShardMetadata,
    replay_path: str | Path,
) -> Path:
    """Atomically write only the shard metadata marker."""
    metadata_path = shard_metadata_path(replay_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = metadata_path.with_name(f"{metadata_path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(metadata.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(metadata_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return metadata_path


def save_klent_shard(
    replay: TrajectoryReplayStore,
    replay_path: str | Path,
    metadata: KlentShardMetadata,
    *,
    compressed: bool = True,
) -> Path:
    """Write the shard payload first and publish metadata last as the commit marker."""
    path = Path(replay_path)
    if metadata.transitions != len(replay):
        raise ValueError(
            f"metadata transitions {metadata.transitions} does not match "
            f"replay length {len(replay)}"
        )
    replay.save(path, compressed=compressed)
    return write_klent_shard_metadata(metadata, path)


def load_klent_shard(
    replay_path: str | Path,
) -> tuple[TrajectoryReplayStore, KlentShardMetadata]:
    path = Path(replay_path)
    metadata_path = shard_metadata_path(path)
    if not metadata_path.exists():
        raise ValueError(f"KLENT shard metadata is missing: {metadata_path}")
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{metadata_path} must contain a JSON object")
    metadata = KlentShardMetadata.from_dict(data)
    replay = TrajectoryReplayStore.load(path)
    if metadata.transitions != len(replay):
        raise ValueError(
            "KLENT shard metadata transitions does not match the replay payload"
        )
    return replay, metadata


__all__ = [
    "KLENT_ALGORITHM",
    "KLENT_SCHEMA_VERSION",
    "KLENT_SHARD_METADATA_SUFFIX",
    "KlentShardMetadata",
    "klent_config_hash",
    "load_klent_shard",
    "save_klent_shard",
    "shard_metadata_path",
    "write_klent_shard_metadata",
]