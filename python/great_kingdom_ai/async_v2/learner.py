"""Async v2 learner cycle runner."""

from __future__ import annotations

import json
import math
import shutil
import time
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from great_kingdom_ai.async_v2.config import LearnerV2Config, LearnerV2Summary
from great_kingdom_ai.async_v2.metadata import (
    V2ShardRecord,
    _append_event,
    _append_game_logs,
    _utc_now,
    load_v2_shard_records,
    pending_v2_shards,
)
from great_kingdom_ai.async_v2.paths import (
    _candidate_checkpoint,
    _ensure_learner_dirs,
    _onnx_output_path,
    _paths,
    _source_checkpoint,
    _training_latest_checkpoint,
)
from great_kingdom_ai.onnx_export import export_checkpoint_to_onnx
from great_kingdom_ai.pipeline_printer import PipelinePrinter
from great_kingdom_ai.replay import TrajectoryReplayDataset, TrajectoryReplayStore
from great_kingdom_ai.runpod_pruning import PruneItem, prune_items
from great_kingdom_ai.training import (
    TrainingConfig,
    summarize_checkpoint_optimizer_state,
    train_from_replay,
)


def run_learner_v2_once(
    config: LearnerV2Config,
    train_config: TrainingConfig,
    *,
    trainer: Callable[..., Any] | None = None,
    onnx_exporter: Callable[..., Any] | None = None,
    resume_optimizer_lr_override: float | None = None,
    printer: PipelinePrinter | None = None,
) -> LearnerV2Summary:
    cycle_started_at = time.monotonic()
    _validate_learner_config(config)
    printer = printer if printer is not None else PipelinePrinter()
    train = trainer if trainer is not None else train_from_replay
    export_onnx = onnx_exporter if onnx_exporter is not None else export_checkpoint_to_onnx
    paths = _paths(config.work_dir)
    _ensure_learner_dirs(paths)
    pending = pending_v2_shards(paths["metadata_path"])

    printer.title("Learner V2")
    printer.metric("work dir", config.work_dir)
    printer.metric("pending shards", len(pending))
    printer.metric("replay capacity", config.replay_capacity)
    printer.metric("min replay rows", config.min_replay_transitions)
    printer.metric("train batch", train_config.batch_size)
    printer.metric("train steps", train_config.steps)
    printer.metric("recent window", train_config.recent_sample_window)
    printer.metric("recent fraction", train_config.recent_sample_fraction)
    printer.metric("ema decay", train_config.ema_decay)
    printer.metric("onnx weights", "ema" if config.onnx_prefer_ema else "raw")
    if not pending:
        cycle_seconds = time.monotonic() - cycle_started_at
        printer.done(f"waiting for shards: pending=0, cycle={cycle_seconds:.1f}s")
        return LearnerV2Summary(
            imported_shards=[],
            imported_transitions=0,
            imported_games=0,
            replay_transitions=None,
            trained=False,
            train_start_step=None,
            train_end_step=None,
            candidate_checkpoint=None,
            training_latest_checkpoint=None,
            onnx_output_path=None,
            pruned_artifacts=0,
            pruned_bytes=0,
            cycle_seconds=cycle_seconds,
        )
    replay = _load_or_create_replay(paths["replay_path"], capacity=config.replay_capacity)
    imported_events = _import_shards_into_replay(
        replay,
        pending,
        defer_capacity_eviction=True,
        printer=printer,
    )
    imported_transitions = sum(stats.transitions for _, stats, _ in imported_events)
    imported_games = sum(stats.games for _, stats, _ in imported_events)
    printer.metric("imported games", imported_games)
    printer.metric("imported rows", imported_transitions)
    printer.metric("train replay transitions", len(replay))

    if len(replay) < config.min_replay_transitions:
        _persist_imported_replay(
            config=config,
            paths=paths,
            replay=replay,
            pending=pending,
            imported_events=imported_events,
            printer=printer,
        )
        cycle_seconds = time.monotonic() - cycle_started_at
        printer.done(
            f"waiting for replay: {len(replay)}/{config.min_replay_transitions} "
            f"transitions, cycle={cycle_seconds:.1f}s"
        )
        return LearnerV2Summary(
            imported_shards=[shard.shard_id for shard in pending],
            imported_transitions=imported_transitions,
            imported_games=imported_games,
            replay_transitions=len(replay),
            trained=False,
            train_start_step=None,
            train_end_step=None,
            candidate_checkpoint=None,
            training_latest_checkpoint=None,
            onnx_output_path=None,
            pruned_artifacts=0,
            pruned_bytes=0,
            cycle_seconds=cycle_seconds,
        )

    dataset = TrajectoryReplayDataset(replay)
    candidate_checkpoint = _candidate_checkpoint(config)
    training_latest = _training_latest_checkpoint(config)
    kwargs = _train_checkpoint_kwargs(
        train_checkpoint_mode=config.train_checkpoint_mode,
        source_checkpoint=_source_checkpoint(config),
    )
    printer.step(f"training candidate -> {candidate_checkpoint}")
    train_kwargs: dict[str, Any] = {**kwargs}
    _print_learner_optimizer_state(
        printer,
        train_checkpoint_mode=config.train_checkpoint_mode,
        resume_path=train_kwargs.get("resume_path"),
        bootstrap_weights_path=train_kwargs.get("bootstrap_weights_path"),
    )
    if resume_optimizer_lr_override is not None:
        train_kwargs["resume_optimizer_lr_override"] = resume_optimizer_lr_override
        printer.metric("optimizer lr override", resume_optimizer_lr_override)
    train_summary = train(
        dataset,
        train_config,
        checkpoint_path=candidate_checkpoint,
        **train_kwargs,
        log_every=max(1, train_config.steps // 10),
        progress_callback=lambda current, target, loss: printer.progress(
            "train",
            current,
            target,
            detail=_format_train_loss_detail(loss),
        ),
    )
    _persist_imported_replay(
        config=config,
        paths=paths,
        replay=replay,
        pending=pending,
        imported_events=imported_events,
        printer=printer,
    )
    training_latest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidate_checkpoint, training_latest)
    printer.done(f"training complete: {train_summary.start_step}->{train_summary.end_step}")
    onnx_path: Path | None = _onnx_output_path(config)
    if config.export_onnx:
        assert onnx_path is not None
        printer.step(f"exporting learner checkpoint -> {onnx_path}")
        temporary_onnx_path = onnx_path.with_suffix(f"{onnx_path.suffix}.tmp")
        export_onnx(
            training_latest,
            temporary_onnx_path,
            device=config.onnx_device,
            precision=config.onnx_precision,
            dummy_batch_size=config.onnx_dummy_batch_size,
            prefer_ema=config.onnx_prefer_ema,
        )
        temporary_onnx_path.replace(onnx_path)
        printer.done(f"onnx ready: {onnx_path}")
    else:
        onnx_path = None

    pruned_artifacts = 0
    pruned_bytes = 0
    if config.prune_artifacts:
        prune_summary = _prune_learner_artifacts(config=config, printer=printer)
        pruned_artifacts = prune_summary["items"]
        pruned_bytes = prune_summary["bytes"]

    cycle_seconds = time.monotonic() - cycle_started_at
    printer.done(f"learner cycle complete in {cycle_seconds:.1f}s")
    return LearnerV2Summary(
        imported_shards=[shard.shard_id for shard in pending],
        imported_transitions=imported_transitions,
        imported_games=imported_games,
        replay_transitions=len(replay),
        trained=True,
        train_start_step=int(train_summary.start_step),
        train_end_step=int(train_summary.end_step),
        candidate_checkpoint=candidate_checkpoint,
        training_latest_checkpoint=training_latest,
        onnx_output_path=onnx_path,
        pruned_artifacts=pruned_artifacts,
        pruned_bytes=pruned_bytes,
        cycle_seconds=cycle_seconds,
    )

def _load_or_create_replay(path: Path, *, capacity: int) -> TrajectoryReplayStore:
    if path.exists():
        replay = TrajectoryReplayStore.load(path)
        if replay.capacity == capacity:
            _drop_async_unused_replay_arrays(replay)
            return replay
        resized = TrajectoryReplayStore.from_episodes(capacity, replay.episodes)
        _drop_async_unused_replay_arrays(resized)
        return resized
    return TrajectoryReplayStore.empty(capacity)

def _drop_async_unused_replay_arrays(replay: TrajectoryReplayStore) -> None:
    replay.next_features = None
    replay.next_features_present = None


@dataclass(frozen=True)
class ShardImportStats:
    transitions: int
    games: int
    load_seconds: float
    extend_seconds: float
    total_seconds: float


@dataclass(frozen=True)
class _NpzReplayFileStats:
    size_bytes: int
    uncompressed_bytes: int
    compressed_members: int
    total_members: int
    optional_keys: tuple[str, ...]


@dataclass(frozen=True)
class _LoadedShardReplay:
    shard_id: str
    replay: TrajectoryReplayStore
    file_stats: _NpzReplayFileStats
    load_seconds: float
    total_seconds_before_extend: float


def _import_shard_into_replay(
    replay: TrajectoryReplayStore,
    *,
    shard_id: str,
    replay_path: Path,
    printer: PipelinePrinter,
) -> ShardImportStats:
    return _import_shards_into_replay(
        replay,
        [
            V2ShardRecord(
                shard_id=shard_id,
                status="completed",
                shard_dir=replay_path.parent,
                replay_path=replay_path,
                log_path=replay_path.parent / "game_logs.json",
                model_version="",
                model_path=Path(),
                seed_start=0,
                games=0,
                transitions=0,
                created_at="",
            )
        ],
        printer=printer,
    )[0][1]


def _import_shards_into_replay(
    replay: TrajectoryReplayStore,
    shards: list[V2ShardRecord],
    *,
    defer_capacity_eviction: bool = False,
    printer: PipelinePrinter,
) -> list[tuple[str, ShardImportStats, int]]:
    loaded = [
        _load_shard_replay(
            shard_id=shard.shard_id,
            replay_path=shard.replay_path,
            printer=printer,
        )
        for shard in shards
    ]
    if not loaded:
        return []
    extend_started_at = time.monotonic()
    replay.extend_stores(
        [item.replay for item in loaded],
        defer_capacity_eviction=defer_capacity_eviction,
    )
    extend_seconds = time.monotonic() - extend_started_at
    total_transitions = sum(len(item.replay) for item in loaded)
    imported_events: list[tuple[str, ShardImportStats, int]] = []
    for item in loaded:
        extend_share = _proportional_seconds(
            extend_seconds,
            part=len(item.replay),
            total=total_transitions,
        )
        total_seconds = item.total_seconds_before_extend + extend_share
        stats = ShardImportStats(
            transitions=len(item.replay),
            games=item.replay.episode_count,
            load_seconds=item.load_seconds,
            extend_seconds=extend_share,
            total_seconds=total_seconds,
        )
        _print_shard_import_stats(
            printer,
            shard_id=item.shard_id,
            file_stats=item.file_stats,
            transitions=stats.transitions,
            games=stats.games,
            load_seconds=stats.load_seconds,
            extend_seconds=stats.extend_seconds,
            total_seconds=stats.total_seconds,
        )
        imported_events.append((item.shard_id, stats, len(replay)))
    return imported_events


def _persist_imported_replay(
    *,
    config: LearnerV2Config,
    paths: dict[str, Path],
    replay: TrajectoryReplayStore,
    pending: list[V2ShardRecord],
    imported_events: list[tuple[str, ShardImportStats, int]],
    printer: PipelinePrinter,
) -> None:
    if len(replay) > config.replay_capacity or replay.capacity != config.replay_capacity:
        before = len(replay)
        replay.compact_to_capacity(config.replay_capacity)
        printer.metric("compacted replay rows", f"{before}->{len(replay)}")
    _save_replay_with_timing(replay, paths["replay_path"], printer=printer)
    for shard_id, stats, _replay_transitions in imported_events:
        _append_shard_import_event(
            paths["metadata_path"],
            shard_id=shard_id,
            stats=stats,
            replay_transitions=len(replay),
        )
    _append_game_logs(paths["game_log_path"], pending)


def _load_shard_replay(
    *,
    shard_id: str,
    replay_path: Path,
    printer: PipelinePrinter,
) -> _LoadedShardReplay:
    started_at = time.monotonic()
    printer.step(f"importing shard {shard_id}")
    file_stats = _inspect_npz_replay_file(replay_path)
    load_started_at = time.monotonic()
    shard_replay = TrajectoryReplayStore.load(replay_path)
    load_seconds = time.monotonic() - load_started_at
    _drop_async_unused_replay_arrays(shard_replay)
    return _LoadedShardReplay(
        shard_id=shard_id,
        replay=shard_replay,
        file_stats=file_stats,
        load_seconds=load_seconds,
        total_seconds_before_extend=time.monotonic() - started_at,
    )


def _proportional_seconds(seconds: float, *, part: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return seconds * (part / total)


def _inspect_npz_replay_file(path: Path) -> _NpzReplayFileStats:
    size_bytes = path.stat().st_size
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
    keys = tuple(member.filename.removesuffix(".npy") for member in members)
    optional_keys = tuple(
        key
        for key in ("root_policy_logits", "next_features")
        if key in keys
    )
    return _NpzReplayFileStats(
        size_bytes=size_bytes,
        uncompressed_bytes=sum(member.file_size for member in members),
        compressed_members=sum(
            1 for member in members if member.compress_type != zipfile.ZIP_STORED
        ),
        total_members=len(members),
        optional_keys=optional_keys,
    )


def _print_shard_import_stats(
    printer: PipelinePrinter,
    *,
    shard_id: str,
    file_stats: _NpzReplayFileStats,
    transitions: int,
    games: int,
    load_seconds: float,
    extend_seconds: float,
    total_seconds: float,
) -> None:
    optional = ",".join(file_stats.optional_keys) if file_stats.optional_keys else "none"
    printer.done(
        f"imported shard {shard_id}: rows={transitions}, games={games}, "
        f"file={_format_bytes(file_stats.size_bytes)}, "
        f"npz_raw={_format_bytes(file_stats.uncompressed_bytes)}, "
        f"compressed_members={file_stats.compressed_members}/{file_stats.total_members}, "
        f"optional={optional}, load={load_seconds:.2f}s, "
        f"extend={extend_seconds:.2f}s, total={total_seconds:.2f}s"
    )


def _append_shard_import_event(
    metadata_path: Path,
    *,
    shard_id: str,
    stats: ShardImportStats,
    replay_transitions: int,
) -> None:
    _append_event(
        metadata_path,
        {
            "event": "shard_imported",
            "shard_id": shard_id,
            "imported_at": _utc_now(),
            "imported_transitions": stats.transitions,
            "replay_transitions": replay_transitions,
            "import_load_seconds": stats.load_seconds,
            "import_extend_seconds": stats.extend_seconds,
            "import_total_seconds": stats.total_seconds,
        },
    )


def _save_replay_with_timing(
    replay: TrajectoryReplayStore,
    path: Path,
    *,
    printer: PipelinePrinter,
) -> float:
    printer.step(f"saving replay -> {path}")
    started_at = time.monotonic()
    save_stats = replay.save(path, compressed=False)
    seconds = time.monotonic() - started_at
    printer.done(
        f"saved replay: file={_format_bytes(path.stat().st_size)}, "
        f"write={save_stats.write_seconds:.2f}s, "
        f"close={save_stats.close_seconds:.2f}s, "
        f"replace={save_stats.replace_seconds:.2f}s, "
        f"total={seconds:.2f}s, slowest={_format_slowest_npz_writes(save_stats)}"
    )
    return seconds


def _format_slowest_npz_writes(save_stats: Any) -> str:
    slowest = sorted(save_stats.array_stats, key=lambda stat: stat.seconds, reverse=True)[:3]
    if not slowest:
        return "none"
    return ",".join(
        f"{stat.key}:{stat.seconds:.2f}s/{_format_bytes(stat.bytes)}"
        for stat in slowest
    )


def _format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024.0 or unit == "GiB":
            return f"{value:.1f}{unit}"
        value /= 1024.0
    raise AssertionError("unreachable")

def _continuous_train_steps(
    *,
    train_budget_samples: float,
    replay_transitions: int,
    train_config: TrainingConfig,
    min_replay_transitions: int,
) -> int:
    if replay_transitions < min_replay_transitions:
        return 0
    if train_budget_samples < train_config.batch_size:
        return 0
    budget_steps = math.floor(train_budget_samples / train_config.batch_size)
    return max(0, min(train_config.steps, budget_steps))

def _format_train_loss_detail(loss: dict[str, float]) -> str:
    detail = f"loss={loss['total']:.4f}"
    if {"policy", "value", "policy_kl"}.issubset(loss):
        detail += (
            f" policy={loss['policy']:.4f}"
            f" value={loss['value']:.4f}"
            f" kl={loss['policy_kl']:.4f}"
        )
    return detail

def _prune_learner_artifacts(
    *,
    config: LearnerV2Config,
    printer: PipelinePrinter,
) -> dict[str, int]:
    items = _collect_imported_shard_prune_items(
        config.work_dir,
        keep_imported_shards=config.prune_keep_imported_shards,
    )
    total_bytes = sum(item.size_bytes for item in items)
    printer.step(
        "pruning learner artifacts "
        f"(items={len(items)}, bytes={total_bytes}, elapsed={printer.elapsed()})"
    )
    prune_items(items, delete=True)
    return {"items": len(items), "bytes": total_bytes}

def _collect_imported_shard_prune_items(
    work_dir: Path,
    *,
    keep_imported_shards: int,
) -> list[PruneItem]:
    if keep_imported_shards < 0:
        raise ValueError("prune_keep_imported_shards must be non-negative")
    paths = _paths(work_dir)
    imported = [
        record
        for record in load_v2_shard_records(paths["metadata_path"])
        if record.status == "imported" and record.shard_dir.exists()
    ]
    if keep_imported_shards > 0:
        imported = sorted(
            imported,
            key=lambda record: (record.imported_at or record.created_at, record.shard_id),
        )
        imported = imported[: max(0, len(imported) - keep_imported_shards)]
    return [
        PruneItem(
            path=record.shard_dir,
            reason="imported learner shard directory",
            size_bytes=_path_size(record.shard_dir),
        )
        for record in imported
    ]

def _path_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file() or path.is_symlink():
        return path.lstat().st_size
    return sum(child.lstat().st_size for child in path.rglob("*") if child.exists())

def _train_checkpoint_kwargs(
    *,
    train_checkpoint_mode: str,
    source_checkpoint: Path,
) -> dict[str, Path | None]:
    if not source_checkpoint.exists():
        return {"resume_path": None, "bootstrap_weights_path": None}
    if train_checkpoint_mode == "resume":
        return {"resume_path": source_checkpoint, "bootstrap_weights_path": None}
    if train_checkpoint_mode == "bootstrap":
        return {"resume_path": None, "bootstrap_weights_path": source_checkpoint}
    raise ValueError("train_checkpoint_mode must be one of: resume, bootstrap")

def _print_learner_optimizer_state(
    printer: PipelinePrinter,
    *,
    train_checkpoint_mode: str,
    resume_path: str | Path | None,
    bootstrap_weights_path: str | Path | None,
) -> None:
    printer.metric("train checkpoint mode", train_checkpoint_mode)
    if resume_path is not None:
        checkpoint_path = Path(resume_path)
        printer.metric("optimizer checkpoint", checkpoint_path)
        printer.metric(
            "optimizer state",
            json.dumps(
                summarize_checkpoint_optimizer_state(checkpoint_path),
                sort_keys=True,
            ),
        )
        return
    if bootstrap_weights_path is not None:
        printer.metric("optimizer checkpoint", "fresh (bootstrap weights)")
        return
    printer.metric("optimizer checkpoint", "fresh (no source checkpoint)")

def _validate_learner_config(config: LearnerV2Config) -> None:
    if config.replay_capacity <= 0:
        raise ValueError("replay_capacity must be positive")
    if config.min_replay_transitions < 0:
        raise ValueError("min_replay_transitions must be non-negative")
    if config.prune_keep_imported_shards < 0:
        raise ValueError("prune_keep_imported_shards must be non-negative")
    if not math.isfinite(config.train_reuse_factor) or config.train_reuse_factor < 0.0:
        raise ValueError("train_reuse_factor must be non-negative")
    if config.train_checkpoint_mode not in {"resume", "bootstrap"}:
        raise ValueError("train_checkpoint_mode must be one of: resume, bootstrap")
    if config.onnx_precision not in {"fp32", "fp16"}:
        raise ValueError("onnx_precision must be one of: fp32, fp16")
    if config.onnx_dummy_batch_size <= 0:
        raise ValueError("onnx_dummy_batch_size must be positive")
