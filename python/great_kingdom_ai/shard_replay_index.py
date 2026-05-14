"""Event-log backed shard replay index for async learner replay storage."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from great_kingdom_ai.trajectory_replay import TrajectoryReplayStore


class ShardIndexSource(Protocol):
    shard_id: str
    replay_path: Path
    log_path: Path
    model_version: str
    model_path: Path
    seed_start: int
    games: int
    created_at: str


@dataclass(frozen=True)
class ShardIndexRecord:
    shard_id: str
    status: str
    replay_path: Path
    log_path: Path | None
    rows: int
    episodes: int
    seed_start: int | None
    games: int
    model_version: str
    model_path: Path | None
    created_at: str
    indexed_at: str
    evicted_at: str | None = None
    eviction_reason: str | None = None

    def to_event_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("replay_path", "log_path", "model_path"):
            value = data[key]
            data[key] = None if value is None else str(value)
        if self.status == "active":
            data["event"] = "shard_indexed"
        elif self.status == "evicted":
            data["event"] = "shard_evicted"
            data["reason"] = self.eviction_reason or "capacity"
        else:
            raise ValueError(f"unknown shard index status: {self.status}")
        return data


@dataclass(frozen=True)
class ActiveShardSpan:
    record: ShardIndexRecord
    start: int
    stop: int


class ShardReplayIndex:
    """Shard replay index reconstructed from append-only JSONL events."""

    def __init__(
        self,
        *,
        index_dir: Path,
        capacity: int,
        records: Iterable[ShardIndexRecord] = (),
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.index_dir = index_dir
        self.capacity = capacity
        self.index_path = index_dir / "index.jsonl"
        self.state_path = index_dir / "state.json"
        self._records = {record.shard_id: record for record in records}
        self._validate_active_files()

    @classmethod
    def load_or_create(cls, index_dir: str | Path, *, capacity: int) -> ShardReplayIndex:
        path = Path(index_dir)
        records = _load_records(path / "index.jsonl")
        index = cls(index_dir=path, capacity=capacity, records=records)
        index.write_state()
        return index

    @property
    def active_records(self) -> list[ShardIndexRecord]:
        return [record for record in self._records.values() if record.status == "active"]

    @property
    def evicted_records(self) -> list[ShardIndexRecord]:
        return [record for record in self._records.values() if record.status == "evicted"]

    @property
    def active_rows(self) -> int:
        return sum(record.rows for record in self.active_records)

    def __len__(self) -> int:
        return self.active_rows

    def active_spans(self) -> list[ActiveShardSpan]:
        spans: list[ActiveShardSpan] = []
        start = 0
        for record in self.active_records:
            stop = start + record.rows
            spans.append(ActiveShardSpan(record=record, start=start, stop=stop))
            start = stop
        return spans

    def add_shard(self, shard: ShardIndexSource) -> ShardIndexRecord:
        replay = TrajectoryReplayStore.load(shard.replay_path)
        replay.validate()
        existing = self._records.get(shard.shard_id)
        if existing is not None:
            if _same_indexed_shard(existing, shard, replay):
                return existing
            raise ValueError(f"shard index already contains different shard {shard.shard_id}")

        record = ShardIndexRecord(
            shard_id=shard.shard_id,
            status="active",
            replay_path=shard.replay_path,
            log_path=shard.log_path,
            rows=len(replay),
            episodes=replay.episode_count,
            seed_start=shard.seed_start,
            games=shard.games,
            model_version=shard.model_version,
            model_path=shard.model_path,
            created_at=shard.created_at,
            indexed_at=_utc_now(),
        )
        self._append_event(record.to_event_dict())
        self._records[record.shard_id] = record
        self.write_state()
        return record

    def evict_to_capacity(self) -> list[ShardIndexRecord]:
        evicted: list[ShardIndexRecord] = []
        while self.active_rows > self.capacity and self.active_records:
            oldest = self.active_records[0]
            replacement = ShardIndexRecord(
                shard_id=oldest.shard_id,
                status="evicted",
                replay_path=oldest.replay_path,
                log_path=oldest.log_path,
                rows=oldest.rows,
                episodes=oldest.episodes,
                seed_start=oldest.seed_start,
                games=oldest.games,
                model_version=oldest.model_version,
                model_path=oldest.model_path,
                created_at=oldest.created_at,
                indexed_at=oldest.indexed_at,
                evicted_at=_utc_now(),
                eviction_reason="capacity",
            )
            self._append_event(replacement.to_event_dict())
            self._records[replacement.shard_id] = replacement
            evicted.append(replacement)
        if evicted:
            self.write_state()
        return evicted

    def write_state(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "capacity": self.capacity,
            "active_rows": self.active_rows,
            "active_shards": len(self.active_records),
            "updated_at": _utc_now(),
        }
        temporary_path = self.state_path.with_name(f".{self.state_path.name}.tmp")
        temporary_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        temporary_path.replace(self.state_path)

    def _append_event(self, event: dict[str, Any]) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        line = (json.dumps(event, sort_keys=True) + "\n").encode("utf-8")
        fd = os.open(self.index_path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
        try:
            os.write(fd, line)
            os.fsync(fd)
        finally:
            os.close(fd)

    def _validate_active_files(self) -> None:
        for record in self.active_records:
            if not record.replay_path.is_file():
                raise FileNotFoundError(
                    f"active indexed shard replay is missing: {record.replay_path}"
                )


def _load_records(path: Path) -> list[ShardIndexRecord]:
    if not path.exists():
        return []
    records: dict[str, ShardIndexRecord] = {}
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
            if event_name == "shard_indexed":
                records[shard_id] = _record_from_indexed_event(event)
            elif event_name == "shard_evicted":
                previous = records.get(shard_id)
                if previous is None:
                    raise ValueError(f"{path} line {line_number} evicts unknown shard")
                records[shard_id] = ShardIndexRecord(
                    shard_id=previous.shard_id,
                    status="evicted",
                    replay_path=previous.replay_path,
                    log_path=previous.log_path,
                    rows=previous.rows,
                    episodes=previous.episodes,
                    seed_start=previous.seed_start,
                    games=previous.games,
                    model_version=previous.model_version,
                    model_path=previous.model_path,
                    created_at=previous.created_at,
                    indexed_at=previous.indexed_at,
                    evicted_at=str(event.get("evicted_at", "")) or _utc_now(),
                    eviction_reason=str(event.get("reason", "")) or "capacity",
                )
            else:
                raise ValueError(f"{path} line {line_number} has unknown event {event_name!r}")
    return list(records.values())


def _record_from_indexed_event(event: dict[str, Any]) -> ShardIndexRecord:
    return ShardIndexRecord(
        shard_id=str(event["shard_id"]),
        status="active",
        replay_path=Path(str(event["replay_path"])),
        log_path=None if event.get("log_path") is None else Path(str(event["log_path"])),
        rows=int(event["rows"]),
        episodes=int(event["episodes"]),
        seed_start=None if event.get("seed_start") is None else int(event["seed_start"]),
        games=int(event["games"]),
        model_version=str(event["model_version"]),
        model_path=None if event.get("model_path") is None else Path(str(event["model_path"])),
        created_at=str(event["created_at"]),
        indexed_at=str(event["indexed_at"]),
    )


def _same_indexed_shard(
    existing: ShardIndexRecord,
    shard: ShardIndexSource,
    replay: TrajectoryReplayStore,
) -> bool:
    return (
        existing.status == "active"
        and existing.replay_path == shard.replay_path
        and existing.rows == len(replay)
        and existing.episodes == replay.episode_count
    )


def _utc_now() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds")


__all__ = [
    "ActiveShardSpan",
    "ShardIndexRecord",
    "ShardReplayIndex",
]
