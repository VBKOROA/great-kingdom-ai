"""Async v2 shard metadata and game-log persistence."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from great_kingdom_ai.self_play import GameLog, MoveLog

ShardStatus = Literal["completed", "imported"]

@dataclass(frozen=True)
class V2ShardRecord:
    shard_id: str
    status: ShardStatus
    shard_dir: Path
    replay_path: Path
    log_path: Path
    model_version: str
    model_path: Path
    seed_start: int
    games: int
    transitions: int
    created_at: str
    imported_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("shard_dir", "replay_path", "log_path", "model_path"):
            data[key] = str(data[key])
        return data

def load_v2_shard_records(metadata_path: str | Path) -> list[V2ShardRecord]:
    path = Path(metadata_path)
    if not path.exists():
        return []
    records: dict[str, V2ShardRecord] = {}
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            event = json.loads(stripped)
            if not isinstance(event, dict):
                raise ValueError(f"{path} line {line_number} must contain a JSON object")
            shard_id = str(event.get("shard_id", ""))
            if not shard_id:
                raise ValueError(f"{path} line {line_number} missing shard_id")
            if event.get("event") == "shard_completed":
                records[shard_id] = _record_from_completed_event(event)
            elif event.get("event") == "shard_imported":
                records[shard_id] = _imported_record(
                    records,
                    shard_id=shard_id,
                    imported_at=str(event.get("imported_at", "")) or None,
                    path=path,
                    line_number=line_number,
                )
            else:
                raise ValueError(
                    f"{path} line {line_number} has unknown event {event.get('event')!r}"
                )
    return list(records.values())

def pending_v2_shards(metadata_path: str | Path) -> list[V2ShardRecord]:
    return [
        record
        for record in load_v2_shard_records(metadata_path)
        if record.status == "completed"
    ]

def _append_game_logs(path: Path, shards: list[V2ShardRecord]) -> None:
    if not shards:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        for shard in shards:
            for log in _load_game_logs(shard.log_path):
                output.write(json.dumps(log.to_dict(), sort_keys=True))
                output.write("\n")

def _load_game_logs(path: Path) -> list[GameLog]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list")
    return [_game_log_from_dict(dict(item)) for item in data if isinstance(item, dict)]

def _append_event(metadata_path: Path, event: dict[str, Any]) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(event, sort_keys=True) + "\n").encode("utf-8")
    fd = os.open(metadata_path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(fd, line)
    finally:
        os.close(fd)

def _record_from_completed_event(event: dict[str, Any]) -> V2ShardRecord:
    return V2ShardRecord(
        shard_id=str(event["shard_id"]),
        status="completed",
        shard_dir=Path(str(event["shard_dir"])),
        replay_path=Path(str(event["replay_path"])),
        log_path=Path(str(event["log_path"])),
        model_version=str(event["model_version"]),
        model_path=Path(str(event["model_path"])),
        seed_start=int(event["seed_start"]),
        games=int(event["games"]),
        transitions=int(event["transitions"]),
        created_at=str(event["created_at"]),
    )

def _imported_record(
    records: dict[str, V2ShardRecord],
    *,
    shard_id: str,
    imported_at: str | None,
    path: Path,
    line_number: int,
) -> V2ShardRecord:
    previous = records.get(shard_id)
    if previous is None:
        raise ValueError(f"{path} line {line_number} imports unknown shard {shard_id}")
    return V2ShardRecord(
        shard_id=previous.shard_id,
        status="imported",
        shard_dir=previous.shard_dir,
        replay_path=previous.replay_path,
        log_path=previous.log_path,
        model_version=previous.model_version,
        model_path=previous.model_path,
        seed_start=previous.seed_start,
        games=previous.games,
        transitions=previous.transitions,
        created_at=previous.created_at,
        imported_at=imported_at,
    )

def _utc_now() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds")

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
