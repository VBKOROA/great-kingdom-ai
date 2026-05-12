"""Shard metadata and persistence helpers for actor/learner processes."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from great_kingdom_ai.replay_buffer import ReplayBuffer, ReplaySample
from great_kingdom_ai.self_play import GameLog, MoveLog

ShardStatus = Literal["completed", "imported"]


@dataclass(frozen=True)
class ShardRecord:
    shard_id: str
    status: ShardStatus
    shard_dir: Path
    replay_path: Path
    log_path: Path
    model_version: str
    model_path: Path
    seed_start: int
    games: int
    samples: int
    created_at: str
    imported_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("shard_dir", "replay_path", "log_path", "model_path"):
            data[key] = str(data[key])
        return data


def load_shard_records(metadata_path: str | Path) -> list[ShardRecord]:
    path = Path(metadata_path)
    if not path.exists():
        return []
    records: dict[str, ShardRecord] = {}
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            event = json.loads(stripped)
            if not isinstance(event, dict):
                raise ValueError(f"{path} line {line_number} must contain a JSON object")
            event_name = event.get("event")
            shard_id = str(event.get("shard_id", ""))
            if not shard_id:
                raise ValueError(f"{path} line {line_number} missing shard_id")
            if event_name == "shard_completed":
                records[shard_id] = _record_from_completed_event(event)
            elif event_name == "shard_imported":
                records[shard_id] = _imported_record(
                    records,
                    shard_id=shard_id,
                    imported_at=str(event.get("imported_at", "")) or None,
                    path=path,
                    line_number=line_number,
                )
            else:
                raise ValueError(f"{path} line {line_number} has unknown event {event_name!r}")
    return list(records.values())


def pending_shards(metadata_path: str | Path) -> list[ShardRecord]:
    return [
        record
        for record in load_shard_records(metadata_path)
        if record.status == "completed"
    ]


def save_shard(
    shard_dir: Path,
    *,
    samples: Sequence[ReplaySample],
    logs: Sequence[GameLog],
) -> None:
    shard_dir.mkdir(parents=True, exist_ok=True)
    replay = ReplayBuffer(capacity=max(1, len(samples)))
    replay.extend(samples)
    replay.save(shard_dir / "replay.npz", compressed=False)
    with (shard_dir / "game_logs.json").open("w", encoding="utf-8") as file:
        json.dump([log.to_dict() for log in logs], file, indent=2, sort_keys=True)


def load_shard_samples(path: Path) -> list[ReplaySample]:
    return ReplayBuffer.load(path).to_samples()


def load_game_logs(path: Path) -> list[GameLog]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list")
    return [_game_log_from_dict(dict(item)) for item in data if isinstance(item, dict)]


def append_shard_event(metadata_path: Path, event: dict[str, Any]) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(event, sort_keys=True))
        file.write("\n")


def process_paths(work_dir: Path) -> dict[str, Path]:
    return {
        "shard_root": work_dir / "shards",
        "metadata_path": work_dir / "shards" / "metadata.jsonl",
        "replay_path": work_dir / "replay" / "replay.npz",
        "game_log_path": work_dir / "replay" / "game_logs.jsonl",
        "candidate_checkpoint": work_dir / "checkpoints" / "candidate.pt",
        "training_latest_checkpoint": work_dir / "checkpoints" / "training-latest.pt",
        "best_checkpoint": work_dir / "checkpoints" / "best.pt",
    }


def default_shard_id(*, model_version: str, seed_start: int, games: int) -> str:
    model_id = safe_id(model_version)
    return f"{model_id}-seed-{seed_start:08d}-games-{games:04d}"


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "shard"


def utc_now() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds")


def _imported_record(
    records: dict[str, ShardRecord],
    *,
    shard_id: str,
    imported_at: str | None,
    path: Path,
    line_number: int,
) -> ShardRecord:
    previous = records.get(shard_id)
    if previous is None:
        raise ValueError(f"{path} line {line_number} imports unknown shard {shard_id}")
    return ShardRecord(
        shard_id=previous.shard_id,
        status="imported",
        shard_dir=previous.shard_dir,
        replay_path=previous.replay_path,
        log_path=previous.log_path,
        model_version=previous.model_version,
        model_path=previous.model_path,
        seed_start=previous.seed_start,
        games=previous.games,
        samples=previous.samples,
        created_at=previous.created_at,
        imported_at=imported_at,
    )


def _record_from_completed_event(event: dict[str, Any]) -> ShardRecord:
    return ShardRecord(
        shard_id=str(event["shard_id"]),
        status="completed",
        shard_dir=Path(str(event["shard_dir"])),
        replay_path=Path(str(event["replay_path"])),
        log_path=Path(str(event["log_path"])),
        model_version=str(event["model_version"]),
        model_path=Path(str(event["model_path"])),
        seed_start=int(event["seed_start"]),
        games=int(event["games"]),
        samples=int(event["samples"]),
        created_at=str(event["created_at"]),
    )


def _game_log_from_dict(data: dict[str, Any]) -> GameLog:
    moves = [
        MoveLog(
            turn=int(move["turn"]),
            player=int(move["player"]),
            action=int(move["action"]),
        )
        for move in data["moves"]
    ]
    territory = data["territory_scores"]
    return GameLog(
        seed=int(data["seed"]),
        moves=moves,
        winner=int(data["winner"]),
        end_reason=int(data["end_reason"]),
        territory_scores=(int(territory[0]), int(territory[1])),
    )


__all__ = [
    "ShardRecord",
    "append_shard_event",
    "default_shard_id",
    "load_game_logs",
    "load_shard_records",
    "load_shard_samples",
    "pending_shards",
    "process_paths",
    "save_shard",
    "utc_now",
]
