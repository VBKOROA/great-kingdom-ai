"""Replay import helpers for Rust ONNX self-play output."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from great_kingdom_ai.online_aggregate_replay import OnlineAggregateReplayBuffer
from great_kingdom_ai.replay_buffer import ReplayBuffer, ReplaySample
from great_kingdom_ai.self_play import GameLog


@dataclass(frozen=True)
class RustReplayImportSummary:
    artifact_dir: Path
    replay_path: Path
    imported_samples: int
    replay_samples: int
    imported_games: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_dir": str(self.artifact_dir),
            "replay_path": str(self.replay_path),
            "imported_samples": self.imported_samples,
            "replay_samples": self.replay_samples,
            "imported_games": self.imported_games,
        }


def import_rust_self_play_samples(
    *,
    artifact_dir: str | Path,
    samples: Sequence[ReplaySample],
    logs: Sequence[GameLog],
    replay_path: str | Path,
    replay_capacity: int,
    game_log_path: str | Path | None = None,
    aggregate_replay_path: str | Path | None = None,
    aggregate_replay_weight_mode: str = "sqrt_count",
    aggregate_replay_weight_cap: float | None = 16.0,
    materialize_raw_replay: bool = True,
    aggregate_replay: OnlineAggregateReplayBuffer | None = None,
    save_aggregate_replay: bool = True,
) -> RustReplayImportSummary:
    if not materialize_raw_replay and aggregate_replay_path is None:
        raise ValueError("materialize_raw_replay=False requires aggregate_replay_path")

    artifact = Path(artifact_dir)
    replay_file = Path(replay_path)
    replay_samples = 0
    if materialize_raw_replay:
        replay = (
            ReplayBuffer.load(replay_file)
            if replay_file.exists()
            else ReplayBuffer(replay_capacity)
        )
        replay.extend(samples)
        replay.save(replay_file, compressed=False)
        replay_samples = len(replay)
    if aggregate_replay_path is not None:
        aggregate_samples = _extend_online_aggregate_replay(
            aggregate_replay_path=Path(aggregate_replay_path),
            raw_replay_path=replay_file,
            replay_capacity=replay_capacity,
            samples=samples,
            sample_weight_mode=aggregate_replay_weight_mode,
            sample_weight_cap=aggregate_replay_weight_cap,
            raw_replay_includes_samples=materialize_raw_replay,
            aggregate_replay=aggregate_replay,
            save=save_aggregate_replay,
        )
        if not materialize_raw_replay:
            replay_samples = aggregate_samples

    if game_log_path is not None:
        log_path = Path(game_log_path)
        if log_path.suffix == ".jsonl":
            _append_jsonl(log_path, (log.to_dict() for log in logs))
        else:
            existing = _read_json_list(log_path) if log_path.exists() else []
            existing.extend(log.to_dict() for log in logs)
            _write_json(log_path, existing)

    return RustReplayImportSummary(
        artifact_dir=artifact,
        replay_path=replay_file,
        imported_samples=len(samples),
        replay_samples=replay_samples,
        imported_games=len(logs),
    )


def _extend_online_aggregate_replay(
    *,
    aggregate_replay_path: Path,
    raw_replay_path: Path,
    replay_capacity: int,
    samples: Sequence[ReplaySample],
    sample_weight_mode: str,
    sample_weight_cap: float | None,
    raw_replay_includes_samples: bool,
    aggregate_replay: OnlineAggregateReplayBuffer | None = None,
    save: bool = True,
) -> int:
    if aggregate_replay is not None:
        replay = aggregate_replay
        replay.extend(samples)
    elif aggregate_replay_path.exists():
        replay = OnlineAggregateReplayBuffer.load(
            aggregate_replay_path,
            capacity=replay_capacity,
            sample_weight_mode=sample_weight_mode,
            sample_weight_cap=sample_weight_cap,
        )
        replay.extend(samples)
    else:
        replay = OnlineAggregateReplayBuffer(
            replay_capacity,
            sample_weight_mode=sample_weight_mode,
            sample_weight_cap=sample_weight_cap,
        )
        if raw_replay_path.exists():
            _extend_online_aggregate_from_file(
                replay,
                raw_replay_path,
            )
        if not raw_replay_includes_samples:
            replay.extend(samples)
    if save:
        replay.save(aggregate_replay_path, compressed=False)
    return len(replay)


def _extend_online_aggregate_from_file(
    replay: OnlineAggregateReplayBuffer,
    raw_replay_path: Path,
) -> None:
    with np.load(raw_replay_path) as data:
        features = np.asarray(data["features"], dtype=np.float32)
        policies = np.asarray(data["policies"], dtype=np.float32)
        values = np.asarray(data["values"], dtype=np.float32)
        root_policy_logits = (
            np.asarray(data["root_policy_logits"], dtype=np.float32)
            if "root_policy_logits" in data
            else None
        )
        sample_weights = (
            np.asarray(data["sample_weights"], dtype=np.float32)
            if "sample_weights" in data
            else np.ones(values.shape, dtype=np.float32)
        )
    for index in range(features.shape[0]):
        replay.push(
            ReplaySample(
                features=features[index],
                policy=policies[index],
                value=float(values[index]),
                root_policy_logits=(
                    root_policy_logits[index]
                    if root_policy_logits is not None
                    and np.isfinite(root_policy_logits[index]).all()
                    else None
                ),
                sample_weight=float(sample_weights[index]),
            )
        )


def _read_json_list(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list")
    return [dict(item) for item in data if isinstance(item, dict)]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, sort_keys=True))
            file.write("\n")


__all__ = [
    "RustReplayImportSummary",
    "import_rust_self_play_samples",
]
