"""Replay import helpers for Rust ONNX self-play output."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from great_kingdom_ai.replay.sample import ReplaySample
from great_kingdom_ai.replay.schema import GameLogLike, TrajectoryEpisode
from great_kingdom_ai.replay.trajectory import (
    TrajectoryReplayStore,
    trajectory_episode_from_self_play_result,
)
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
) -> RustReplayImportSummary:
    artifact = Path(artifact_dir)
    replay_file = Path(replay_path)
    replay_samples = 0

    if samples:
        replay = (
            TrajectoryReplayStore.load(replay_file)
            if replay_file.exists()
            else TrajectoryReplayStore.empty(replay_capacity)
        )
        replay.extend_episodes(
            _episodes_from_logs_and_samples(
                logs=logs,
                samples=samples,
                first_episode_id=replay.episode_count,
            )
        )
        replay.save(replay_file, compressed=False)
        replay_samples = len(replay)

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


def _episodes_from_logs_and_samples(
    *,
    logs: Sequence[GameLog],
    samples: Sequence[ReplaySample],
    first_episode_id: int,
) -> list[TrajectoryEpisode]:
    episodes = []
    sample_offset = 0
    for index, log in enumerate(logs):
        sample_end = sample_offset + len(log.moves)
        if sample_end > len(samples):
            raise ValueError("not enough replay samples for game logs")
        episode_samples = samples[sample_offset:sample_end]
        sample_offset = sample_end
        if not episode_samples:
            continue
        episodes.append(
            trajectory_episode_from_self_play_result(
                cast(GameLogLike, log),
                episode_samples,
                episode_id=first_episode_id + index,
            )
        )
    if sample_offset != len(samples):
        raise ValueError("replay samples must be grouped by game log moves")
    return episodes


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
