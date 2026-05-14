#!/usr/bin/env python3
"""Split a monolithic trajectory replay into shard-index replay storage."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import NoReturn

from great_kingdom_ai.shard_replay_index import ShardReplayIndex
from great_kingdom_ai.trajectory_replay import TrajectoryEpisode, TrajectoryReplayStore


@dataclass(frozen=True)
class MigrationShardSource:
    shard_id: str
    replay_path: Path
    log_path: Path | None
    model_version: str
    model_path: Path | None
    seed_start: int | None
    games: int
    created_at: str


@dataclass(frozen=True)
class MigrationShardSummary:
    shard_id: str
    replay_path: Path
    rows: int
    episodes: int

    def to_dict(self) -> dict[str, str | int]:
        return {
            "shard_id": self.shard_id,
            "replay_path": str(self.replay_path),
            "rows": self.rows,
            "episodes": self.episodes,
        }


@dataclass(frozen=True)
class MigrationSummary:
    source: Path
    output_shard_root: Path
    index_dir: Path
    dry_run: bool
    source_rows: int
    source_episodes: int
    shards: tuple[MigrationShardSummary, ...]

    @property
    def rows(self) -> int:
        return sum(shard.rows for shard in self.shards)

    @property
    def episodes(self) -> int:
        return sum(shard.episodes for shard in self.shards)

    def to_dict(self) -> dict[str, object]:
        return {
            "event": "monolithic_replay_migrated_to_shards",
            "source": str(self.source),
            "output_shard_root": str(self.output_shard_root),
            "index_dir": str(self.index_dir),
            "dry_run": self.dry_run,
            "source_rows": self.source_rows,
            "source_episodes": self.source_episodes,
            "rows": self.rows,
            "episodes": self.episodes,
            "shards": [shard.to_dict() for shard in self.shards],
        }


def migrate_monolithic_replay_to_shards(
    *,
    source: str | Path,
    output_shard_root: str | Path,
    index_dir: str | Path,
    target_rows_per_shard: int = 4096,
    capacity: int | None = None,
    drop_optional_arrays: bool = True,
    compressed: bool = False,
    dry_run: bool = False,
    shard_id_prefix: str = "migrated",
    model_version: str = "migrated",
) -> MigrationSummary:
    source_path = Path(source)
    shard_root = Path(output_shard_root)
    index_path = Path(index_dir)
    if not source_path.is_file():
        raise FileNotFoundError(f"source trajectory replay does not exist: {source_path}")
    if target_rows_per_shard <= 0:
        raise ValueError("target_rows_per_shard must be positive")
    if capacity is not None and capacity <= 0:
        raise ValueError("capacity must be positive")
    if not shard_id_prefix:
        raise ValueError("shard_id_prefix must be non-empty")

    source_replay = TrajectoryReplayStore.load(source_path)
    source_replay.validate()
    resolved_capacity = source_replay.capacity if capacity is None else capacity
    if resolved_capacity < len(source_replay):
        raise ValueError(
            "capacity must be at least the source transition count "
            "to preserve replay data: "
            f"capacity={resolved_capacity}, transitions={len(source_replay)}"
        )
    episode_chunks = _split_episodes(
        source_replay.episodes,
        target_rows_per_shard=target_rows_per_shard,
    )
    summaries: list[MigrationShardSummary] = []
    created_at = _utc_now()

    if dry_run:
        for shard_number, episodes in enumerate(episode_chunks, start=1):
            shard_id = _shard_id(shard_id_prefix, shard_number)
            summaries.append(
                MigrationShardSummary(
                    shard_id=shard_id,
                    replay_path=shard_root / shard_id / "trajectory-replay.npz",
                    rows=_episode_rows(episodes),
                    episodes=len(episodes),
                )
            )
        return _validated_summary(
            source=source_path,
            output_shard_root=shard_root,
            index_dir=index_path,
            dry_run=True,
            source_replay=source_replay,
            shards=summaries,
        )

    index = ShardReplayIndex.load_or_create(index_path, capacity=resolved_capacity)
    for shard_number, episodes in enumerate(episode_chunks, start=1):
        shard_id = _shard_id(shard_id_prefix, shard_number)
        shard_dir = shard_root / shard_id
        replay_path = shard_dir / "trajectory-replay.npz"
        if shard_dir.exists() and not replay_path.is_file():
            raise FileExistsError(f"migration shard directory already exists: {shard_dir}")
        if replay_path.exists():
            raise FileExistsError(f"migration shard replay already exists: {replay_path}")

        rows = _episode_rows(episodes)
        shard_replay = TrajectoryReplayStore.from_episodes(max(1, rows), episodes)
        if drop_optional_arrays:
            _drop_optional_replay_arrays(shard_replay)
        shard_replay.validate()
        shard_dir.mkdir(parents=True, exist_ok=False)
        shard_replay.save(replay_path, compressed=compressed)

        source_record = MigrationShardSource(
            shard_id=shard_id,
            replay_path=replay_path,
            log_path=None,
            model_version=model_version,
            model_path=None,
            seed_start=None,
            games=len(episodes),
            created_at=created_at,
        )
        indexed = index.add_shard(source_record)
        summaries.append(
            MigrationShardSummary(
                shard_id=shard_id,
                replay_path=replay_path,
                rows=indexed.rows,
                episodes=indexed.episodes,
            )
        )

    return _validated_summary(
        source=source_path,
        output_shard_root=shard_root,
        index_dir=index_path,
        dry_run=False,
        source_replay=source_replay,
        shards=summaries,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Split an existing trajectory-replay.npz into episode-boundary shards and "
            "register them in a shard replay index."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Existing trajectory replay .npz",
    )
    parser.add_argument(
        "--output-shard-root",
        type=Path,
        required=True,
        help="Directory that will contain migrated shard directories",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        required=True,
        help="Replay index directory that contains index.jsonl/state.json",
    )
    parser.add_argument("--target-rows-per-shard", type=int, default=4096)
    parser.add_argument(
        "--capacity",
        type=int,
        default=None,
        help="Replay index row capacity. Defaults to the source replay capacity.",
    )
    parser.add_argument(
        "--drop-optional-arrays",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop root_policy_logits and next_features from migrated shards.",
    )
    parser.add_argument("--compressed", action="store_true", help="Write compressed NPZ shards")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan migration without writing files",
    )
    parser.add_argument("--shard-id-prefix", default="migrated")
    parser.add_argument("--model-version", default="migrated")
    return parser


def main() -> NoReturn:
    args = build_parser().parse_args()
    summary = migrate_monolithic_replay_to_shards(
        source=args.source,
        output_shard_root=args.output_shard_root,
        index_dir=args.index_dir,
        target_rows_per_shard=args.target_rows_per_shard,
        capacity=args.capacity,
        drop_optional_arrays=args.drop_optional_arrays,
        compressed=args.compressed,
        dry_run=args.dry_run,
        shard_id_prefix=args.shard_id_prefix,
        model_version=args.model_version,
    )
    print(json.dumps(summary.to_dict(), sort_keys=True))
    raise SystemExit(0)


def _split_episodes(
    episodes: tuple[TrajectoryEpisode, ...],
    *,
    target_rows_per_shard: int,
) -> tuple[tuple[TrajectoryEpisode, ...], ...]:
    chunks: list[tuple[TrajectoryEpisode, ...]] = []
    current: list[TrajectoryEpisode] = []
    current_rows = 0
    for episode in episodes:
        episode_rows = len(episode.transitions)
        if current and current_rows + episode_rows > target_rows_per_shard:
            chunks.append(tuple(current))
            current = []
            current_rows = 0
        current.append(episode)
        current_rows += episode_rows
    if current:
        chunks.append(tuple(current))
    return tuple(chunks)


def _validated_summary(
    *,
    source: Path,
    output_shard_root: Path,
    index_dir: Path,
    dry_run: bool,
    source_replay: TrajectoryReplayStore,
    shards: list[MigrationShardSummary],
) -> MigrationSummary:
    summary = MigrationSummary(
        source=source,
        output_shard_root=output_shard_root,
        index_dir=index_dir,
        dry_run=dry_run,
        source_rows=len(source_replay),
        source_episodes=source_replay.episode_count,
        shards=tuple(shards),
    )
    if summary.rows != len(source_replay):
        raise RuntimeError("migrated shard row count does not match source replay")
    if summary.episodes != source_replay.episode_count:
        raise RuntimeError("migrated shard episode count does not match source replay")
    return summary


def _drop_optional_replay_arrays(replay: TrajectoryReplayStore) -> None:
    replay.root_policy_logits = None
    replay.root_policy_logits_present = None
    replay.next_features = None
    replay.next_features_present = None


def _episode_rows(episodes: tuple[TrajectoryEpisode, ...]) -> int:
    return sum(len(episode.transitions) for episode in episodes)


def _shard_id(prefix: str, number: int) -> str:
    return f"{prefix}-{number:06d}"


def _utc_now() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds")


if __name__ == "__main__":
    main()
