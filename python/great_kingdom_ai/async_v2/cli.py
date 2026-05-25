"""Async v2 command-line entrypoints."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, replace
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

from great_kingdom_ai.async_v2.actor import _reserve_actor_seed_start, run_actor_v2_once
from great_kingdom_ai.async_v2.config import (
    ActorV2Config,
    FactoryInitV2Config,
    LearnerV2Config,
    LearnerV2Summary,
    load_actor_v2_config,
    load_learner_v2_config,
)
from great_kingdom_ai.async_v2.factory import run_factory_init_v2_once
from great_kingdom_ai.async_v2.learner import (
    _continuous_train_steps,
    _drop_async_unused_replay_arrays,
    _format_train_loss_detail,
    _import_shards_into_replay,
    _load_or_create_replay,
    _append_shard_import_event,
    _persist_imported_replay,
    _print_learner_optimizer_state,
    _prune_learner_artifacts,
    _save_replay_with_timing,
    _train_checkpoint_kwargs,
    _validate_learner_config,
    run_learner_v2_once,
)
from great_kingdom_ai.async_v2.metadata import _append_game_logs, pending_v2_shards
from great_kingdom_ai.async_v2.paths import (
    _candidate_checkpoint,
    _ema_onnx_output_path,
    _ensure_learner_dirs,
    _onnx_output_path,
    _paths,
    _source_checkpoint,
    _training_latest_checkpoint,
)
from great_kingdom_ai.onnx_export import export_checkpoint_to_onnx
from great_kingdom_ai.pipeline_printer import PipelinePrinter
from great_kingdom_ai.replay import TrajectoryReplayDataset
from great_kingdom_ai.replay.persistence import copy_file_atomic
from great_kingdom_ai.self_play import SelfPlayConfig
from great_kingdom_ai.training import TrainingConfig, load_training_config, train_from_replay


def build_actor_v2_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="great-kingdom-actor-v2",
        description="Generate v2 trajectory self-play shards.",
    )
    parser.add_argument("--actor-config", type=Path, default=None)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--onnx-model", type=Path, default=None)
    parser.add_argument("--ema-onnx-model", type=Path, default=None)
    parser.add_argument("--ema-opponent-fraction", type=float, default=None)
    parser.add_argument("--model-version", default=None)
    parser.add_argument("--model-iteration", type=int, default=None)
    parser.add_argument("--games", type=int, default=None)
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument("--onnx-device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--onnx-max-batch-size", type=int, default=None)
    parser.add_argument("--rust-self-play-batch-size", type=int, default=None)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--max-cycles", type=int, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument("--json", action="store_true")
    return parser

def build_learner_v2_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="great-kingdom-learner-v2",
        description="Import v2 trajectory shards and train continuously.",
    )
    parser.add_argument("--learner-config", type=Path, default=None)
    parser.add_argument("--train-config", type=Path, default=Path("configs/runpod/train.json"))
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--replay-capacity", type=int, default=None)
    parser.add_argument("--min-replay-transitions", type=int, default=None)
    parser.add_argument("--source-checkpoint", type=Path, default=None)
    parser.add_argument("--train-checkpoint-mode", choices=["resume", "bootstrap"], default=None)
    parser.add_argument(
        "--bootstrap-once",
        action="store_true",
        help=(
            "Use bootstrap checkpoint loading for the first training call only, then resume "
            "from the newly written training-latest checkpoint."
        ),
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--train-steps", type=int, default=None)
    parser.add_argument("--no-export-onnx", action="store_true")
    parser.add_argument("--onnx-device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--onnx-precision", choices=["fp32", "fp16"], default=None)
    parser.add_argument("--onnx-prefer-ema", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--prune-artifacts", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--prune-keep-imported-shards", type=int, default=None)
    parser.add_argument("--train-reuse-factor", type=float, default=None)
    parser.add_argument("--replay-save-temp-dir", type=Path, default=None)
    parser.add_argument("--replay-local-dir", type=Path, default=None)
    parser.add_argument(
        "--defer-replay-save-until-train",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "In continuous mode, keep imported shards staged in memory and write the full "
            "replay only after a training chunk runs."
        ),
    )
    parser.add_argument(
        "--lr-override",
        "--override-optimizer-lr",
        dest="override_optimizer_lr",
        type=float,
        default=None,
        help=(
            "Override the resumed optimizer learning rate for the first training call only. "
            "This is a CLI-only one-shot override and is not saved to learner config."
        ),
    )
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--max-cycles", type=int, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument("--json", action="store_true")
    return parser

def build_factory_init_v2_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="great-kingdom-init-async-v2",
        description="Create factory checkpoint and ONNX artifacts for async v2.",
    )
    parser.add_argument("--train-config", type=Path, default=Path("configs/runpod/train.json"))
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--onnx-output", type=Path, default=None)
    parser.add_argument("--ema-onnx-output", type=Path, default=None)
    parser.add_argument("--no-export-ema-onnx", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument(
        "--model-preset",
        choices=[
            "small",
            "medium",
            "medium_plus",
            "strong",
            "large",
            "large_policy",
            "large_plus",
        ],
        default=None,
    )
    parser.add_argument("--ema-decay", type=float, default=None)
    parser.add_argument("--onnx-device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--onnx-precision", choices=["fp32", "fp16"], default=None)
    parser.add_argument("--onnx-dummy-batch-size", type=int, default=None)
    parser.add_argument("--onnx-prefer-ema", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser

def factory_init_v2_main() -> NoReturn:
    args = build_factory_init_v2_parser().parse_args()
    train = load_training_config(args.train_config)
    train_data = asdict(train)
    if args.device is not None:
        train_data["device"] = args.device
    if args.model_preset is not None:
        train_data["model_preset"] = args.model_preset
    if args.ema_decay is not None:
        train_data["ema_decay"] = args.ema_decay
    config = FactoryInitV2Config(
        work_dir=args.work_dir or FactoryInitV2Config.work_dir,
        checkpoint_path=args.checkpoint,
        onnx_output_path=args.onnx_output,
        ema_onnx_output_path=args.ema_onnx_output,
        export_ema_onnx=not args.no_export_ema_onnx,
        overwrite=args.overwrite,
        onnx_device=args.onnx_device or FactoryInitV2Config.onnx_device,
        onnx_precision=args.onnx_precision or FactoryInitV2Config.onnx_precision,
        onnx_dummy_batch_size=(
            args.onnx_dummy_batch_size
            if args.onnx_dummy_batch_size is not None
            else FactoryInitV2Config.onnx_dummy_batch_size
        ),
        onnx_prefer_ema=(
            args.onnx_prefer_ema
            if args.onnx_prefer_ema is not None
            else FactoryInitV2Config.onnx_prefer_ema
        ),
    )
    summary = run_factory_init_v2_once(
        config,
        TrainingConfig(**train_data),
        printer=PipelinePrinter(enabled=not args.json),
    )
    print(_json_payload([summary.to_dict()], args.json))
    raise SystemExit(0)

def actor_v2_main() -> NoReturn:
    args = build_actor_v2_parser().parse_args()
    base = load_actor_v2_config(args.actor_config) if args.actor_config else ActorV2Config()
    data = asdict(base)
    for key, value in {
        "work_dir": args.work_dir,
        "onnx_model_path": args.onnx_model,
        "ema_onnx_model_path": args.ema_onnx_model,
        "ema_opponent_fraction": args.ema_opponent_fraction,
        "model_version": args.model_version,
        "model_iteration": args.model_iteration,
        "games": args.games,
        "seed_start": args.seed_start,
        "onnx_device": args.onnx_device,
        "onnx_max_batch_size": args.onnx_max_batch_size,
        "rust_self_play_batch_size": args.rust_self_play_batch_size,
    }.items():
        if value is not None:
            data[key] = value
    if isinstance(data.get("self_play"), dict):
        data["self_play"] = SelfPlayConfig(**data["self_play"])
    config = ActorV2Config(**data)
    summaries = _run_actor_cli(config, args)
    print(_json_payload(summaries, args.json))
    raise SystemExit(0)

def learner_v2_main() -> NoReturn:
    args = build_learner_v2_parser().parse_args()
    base = (
        load_learner_v2_config(args.learner_config)
        if args.learner_config
        else LearnerV2Config()
    )
    data = asdict(base)
    for key, value in {
        "work_dir": args.work_dir,
        "replay_capacity": args.replay_capacity,
        "min_replay_transitions": args.min_replay_transitions,
        "source_checkpoint": args.source_checkpoint,
        "train_checkpoint_mode": args.train_checkpoint_mode,
        "onnx_device": args.onnx_device,
        "onnx_precision": args.onnx_precision,
        "onnx_prefer_ema": args.onnx_prefer_ema,
        "prune_artifacts": args.prune_artifacts,
        "prune_keep_imported_shards": args.prune_keep_imported_shards,
        "train_reuse_factor": args.train_reuse_factor,
        "defer_replay_save_until_train": args.defer_replay_save_until_train,
        "replay_save_temp_dir": args.replay_save_temp_dir,
        "replay_local_dir": args.replay_local_dir,
    }.items():
        if value is not None:
            data[key] = value
    if args.no_export_onnx:
        data["export_onnx"] = False
    train = load_training_config(args.train_config)
    train_data = asdict(train)
    if args.device is not None:
        train_data["device"] = args.device
    if args.train_steps is not None:
        train_data["steps"] = args.train_steps
    summaries = _run_learner_cli(LearnerV2Config(**data), TrainingConfig(**train_data), args)
    print(_json_payload(summaries, args.json))
    raise SystemExit(0)

def _run_actor_cli(config: ActorV2Config, args: argparse.Namespace) -> list[dict[str, Any]]:
    summaries = []
    cycles = args.max_cycles if args.loop else 1
    cycle = 0
    while cycles is None or cycle < cycles:
        seed_start = (
            config.seed_start
            if config.shard_id is not None
            else _reserve_actor_seed_start(config)
        )
        shard_config = ActorV2Config(
            work_dir=config.work_dir,
            onnx_model_path=config.onnx_model_path,
            ema_onnx_model_path=config.ema_onnx_model_path,
            ema_opponent_fraction=config.ema_opponent_fraction,
            model_version=config.model_version,
            model_iteration=config.model_iteration,
            shard_id=config.shard_id if not args.loop else None,
            games=config.games,
            seed_start=seed_start,
            onnx_device=config.onnx_device,
            onnx_max_batch_size=config.onnx_max_batch_size,
            rust_self_play_batch_size=config.rust_self_play_batch_size,
            self_play=config.self_play,
        )
        summary = run_actor_v2_once(
            shard_config,
            printer=PipelinePrinter(enabled=not args.json),
        )
        summaries.append(summary.to_dict())
        cycle += 1
        if not args.loop or (cycles is not None and cycle >= cycles):
            break
        time.sleep(args.sleep_seconds)
    return summaries

def _run_learner_cli(
    config: LearnerV2Config,
    train_config: TrainingConfig,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    bootstrap_once = getattr(args, "bootstrap_once", False)
    override_optimizer_lr = getattr(args, "override_optimizer_lr", None)
    if bootstrap_once:
        if override_optimizer_lr is not None:
            raise ValueError("--bootstrap-once cannot be combined with --lr-override")
        if config.train_checkpoint_mode != "resume":
            raise ValueError("--bootstrap-once requires train_checkpoint_mode=resume")
    if args.loop:
        return _run_learner_continuous_cli(config, train_config, args)
    summaries = []
    if bootstrap_once:
        config = replace(config, train_checkpoint_mode="bootstrap")
    summary = run_learner_v2_once(
        config,
        train_config,
        resume_optimizer_lr_override=override_optimizer_lr,
        printer=PipelinePrinter(enabled=not args.json),
    )
    summaries.append(summary.to_dict())
    return summaries


@dataclass(frozen=True)
class _ReplayBackupRequest:
    source_snapshot: Path
    replay_backup_path: Path
    metadata_path: Path
    game_log_path: Path
    pending: tuple[Any, ...]
    imported_events: tuple[tuple[str, Any, int], ...]

    @property
    def shard_ids(self) -> set[str]:
        return {shard.shard_id for shard in self.pending}


class _ReplayBackupManager:
    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="replay-backup")
        self._future: Future[None] | None = None
        self._request: _ReplayBackupRequest | None = None

    def close(self) -> None:
        self._executor.shutdown(wait=True)
        if self._future is not None:
            self._future.result()

    def poll(self, printer: PipelinePrinter) -> _ReplayBackupRequest | None:
        if self._future is None or self._request is None or not self._future.done():
            return None
        future = self._future
        request = self._request
        self._future = None
        self._request = None
        future.result()
        printer.done(
            "replay backup complete: "
            f"shards={len(request.pending)}, path={request.replay_backup_path}"
        )
        return request

    def start_if_idle(
        self,
        *,
        replay_path: Path,
        backup_path: Path,
        metadata_path: Path,
        game_log_path: Path,
        pending: list[Any],
        imported_events: list[tuple[str, Any, int]],
        printer: PipelinePrinter,
    ) -> bool:
        if not pending:
            return False
        if self._future is not None:
            printer.done("replay backup already running")
            return False
        snapshot_path = _local_replay_backup_snapshot_path(replay_path)
        _link_or_copy_local_snapshot(replay_path, snapshot_path)
        request = _ReplayBackupRequest(
            source_snapshot=snapshot_path,
            replay_backup_path=backup_path,
            metadata_path=metadata_path,
            game_log_path=game_log_path,
            pending=tuple(pending),
            imported_events=tuple(imported_events),
        )
        self._request = request
        self._future = self._executor.submit(_run_replay_backup, request)
        printer.done(
            "started replay backup: "
            f"shards={len(request.pending)}, source={snapshot_path}, path={backup_path}"
        )
        return True


def _run_replay_backup(request: _ReplayBackupRequest) -> None:
    try:
        copy_file_atomic(request.source_snapshot, request.replay_backup_path)
        for shard_id, stats, replay_transitions in request.imported_events:
            _append_shard_import_event(
                request.metadata_path,
                shard_id=shard_id,
                stats=stats,
                replay_transitions=replay_transitions,
            )
        _append_game_logs(request.game_log_path, list(request.pending))
    finally:
        if request.source_snapshot.exists():
            request.source_snapshot.unlink()


def _local_replay_backup_snapshot_path(replay_path: Path) -> Path:
    return replay_path.with_name(f"{replay_path.name}.backup-{time.monotonic_ns()}.tmp")


def _link_or_copy_local_snapshot(source: Path, snapshot: Path) -> None:
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, snapshot)
    except OSError:
        copy_file_atomic(source, snapshot)


def _continuous_replay_path(config: LearnerV2Config, backup_replay_path: Path) -> Path:
    if config.replay_local_dir is None:
        return backup_replay_path
    return config.replay_local_dir / "trajectory-replay.npz"


def _prepare_continuous_replay_path(
    *,
    active_replay_path: Path,
    backup_replay_path: Path,
    printer: PipelinePrinter,
) -> None:
    active_replay_path.parent.mkdir(parents=True, exist_ok=True)
    if active_replay_path.exists() or not backup_replay_path.exists():
        return
    printer.step(f"restoring replay backup -> {active_replay_path}")
    started_at = time.monotonic()
    copy_file_atomic(backup_replay_path, active_replay_path)
    printer.done(f"restored replay backup in {time.monotonic() - started_at:.2f}s")


def _remove_backed_up_staged_shards(
    *,
    staged_pending: list[Any],
    staged_imported_events: list[tuple[str, Any, int]],
    backed_up_ids: set[str],
) -> tuple[list[Any], list[tuple[str, Any, int]], set[str]]:
    remaining_pending = [
        shard for shard in staged_pending if shard.shard_id not in backed_up_ids
    ]
    remaining_events = [
        event for event in staged_imported_events if event[0] not in backed_up_ids
    ]
    remaining_ids = {shard.shard_id for shard in remaining_pending}
    return remaining_pending, remaining_events, remaining_ids


def _save_continuous_local_replay(
    *,
    config: LearnerV2Config,
    replay: Any,
    path: Path,
    printer: PipelinePrinter,
) -> None:
    if len(replay) > config.replay_capacity or replay.capacity != config.replay_capacity:
        before = len(replay)
        replay.compact_to_capacity(config.replay_capacity)
        printer.metric("compacted replay rows", f"{before}->{len(replay)}")
    _save_replay_with_timing(replay, path, config=config, printer=printer)


def _run_learner_continuous_cli(
    config: LearnerV2Config,
    train_config: TrainingConfig,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    _validate_learner_config(config)
    paths = _paths(config.work_dir)
    _ensure_learner_dirs(paths)
    setup_printer = PipelinePrinter(enabled=not args.json)
    active_replay_path = _continuous_replay_path(config, paths["replay_path"])
    _prepare_continuous_replay_path(
        active_replay_path=active_replay_path,
        backup_replay_path=paths["replay_path"],
        printer=setup_printer,
    )
    replay = _load_or_create_replay(active_replay_path, capacity=config.replay_capacity)
    _drop_async_unused_replay_arrays(replay)
    train = train_from_replay
    export_onnx = export_checkpoint_to_onnx
    summaries: list[dict[str, Any]] = []
    train_budget_samples = 0.0
    resume_optimizer_lr_override = getattr(args, "override_optimizer_lr", None)
    bootstrap_once_pending = getattr(args, "bootstrap_once", False)
    cycles = args.max_cycles
    cycle = 0
    train_chunks = 0
    staged_pending: list[Any] = []
    staged_imported_events: list[tuple[str, Any, int]] = []
    staged_shard_ids: set[str] = set()
    backup_manager = _ReplayBackupManager() if config.replay_local_dir is not None else None

    try:
        while cycles is None or cycle < cycles:
            cycle_started_at = time.monotonic()
            printer = PipelinePrinter(enabled=not args.json)
            pruned_artifacts = 0
            pruned_bytes = 0
            if backup_manager is not None:
                completed_backup = backup_manager.poll(printer)
                if completed_backup is not None:
                    (
                        staged_pending,
                        staged_imported_events,
                        staged_shard_ids,
                    ) = _remove_backed_up_staged_shards(
                        staged_pending=staged_pending,
                        staged_imported_events=staged_imported_events,
                        backed_up_ids=completed_backup.shard_ids,
                    )
                    if config.prune_artifacts:
                        prune_summary = _prune_learner_artifacts(
                            config=config,
                            printer=printer,
                        )
                        pruned_artifacts = prune_summary["items"]
                        pruned_bytes = prune_summary["bytes"]
                    if staged_pending:
                        backup_manager.start_if_idle(
                            replay_path=active_replay_path,
                            backup_path=paths["replay_path"],
                            metadata_path=paths["metadata_path"],
                            game_log_path=paths["game_log_path"],
                            pending=staged_pending,
                            imported_events=staged_imported_events,
                            printer=printer,
                        )

            pending = [
                shard
                for shard in pending_v2_shards(paths["metadata_path"])
                if shard.shard_id not in staged_shard_ids
            ]
            imported_transitions = 0
            imported_games = 0
            imported_events: list[tuple[str, Any, int]] = []

            printer.title("Learner V2 Continuous")
            printer.metric("work dir", config.work_dir)
            if config.replay_local_dir is not None:
                printer.metric("active replay", active_replay_path)
                printer.metric("backup replay", paths["replay_path"])
            printer.metric("pending shards", len(pending))
            if staged_pending:
                printer.metric("staged shards", len(staged_pending))
            printer.metric("replay transitions", len(replay))
            printer.metric("train batch", train_config.batch_size)
            printer.metric("max train steps", train_config.steps)
            printer.metric("reuse factor", config.train_reuse_factor)
            printer.metric("budget samples", int(train_budget_samples))
            printer.metric("onnx weights", "raw + ema" if config.export_ema_onnx else "raw")

            imported_events = _import_shards_into_replay(
                replay,
                pending,
                defer_capacity_eviction=True,
                printer=printer,
            )
            imported_transitions = sum(stats.transitions for _, stats, _ in imported_events)
            imported_games = sum(stats.games for _, stats, _ in imported_events)

            if pending:
                staged_pending.extend(pending)
                staged_imported_events.extend(imported_events)
                staged_shard_ids.update(shard.shard_id for shard in pending)
                train_budget_samples += imported_transitions * config.train_reuse_factor
                printer.metric("imported games", imported_games)
                printer.metric("imported rows", imported_transitions)
                printer.metric("budget samples", int(train_budget_samples))

            train_steps = _continuous_train_steps(
                train_budget_samples=train_budget_samples,
                replay_transitions=len(replay),
                train_config=train_config,
                min_replay_transitions=config.min_replay_transitions,
            )
            trained = False
            train_start_step: int | None = None
            train_end_step: int | None = None
            candidate_checkpoint: Path | None = None
            training_latest: Path | None = None
            onnx_path: Path | None = None

            if train_steps > 0:
                effective_train_config = TrainingConfig(
                    **{
                        **asdict(train_config),
                        "steps": train_steps,
                        "seed": train_config.seed + train_chunks,
                    }
                )
                dataset = TrajectoryReplayDataset(replay)
                candidate_checkpoint = _candidate_checkpoint(config)
                training_latest = _training_latest_checkpoint(config)
                if bootstrap_once_pending:
                    train_checkpoint_mode = "bootstrap"
                elif getattr(args, "bootstrap_once", False):
                    train_checkpoint_mode = "resume"
                else:
                    train_checkpoint_mode = config.train_checkpoint_mode
                kwargs = _train_checkpoint_kwargs(
                    train_checkpoint_mode=train_checkpoint_mode,
                    source_checkpoint=_source_checkpoint(config),
                )
                printer.step(
                    f"training candidate -> {candidate_checkpoint} "
                    f"(steps={train_steps}, budget={int(train_budget_samples)})"
                )
                train_kwargs: dict[str, Any] = {**kwargs}
                _print_learner_optimizer_state(
                    printer,
                    train_checkpoint_mode=train_checkpoint_mode,
                    resume_path=train_kwargs.get("resume_path"),
                    bootstrap_weights_path=train_kwargs.get("bootstrap_weights_path"),
                )
                if resume_optimizer_lr_override is not None:
                    train_kwargs["resume_optimizer_lr_override"] = resume_optimizer_lr_override
                    printer.metric("optimizer lr override", resume_optimizer_lr_override)

                def progress_callback(
                    current: int,
                    target: int,
                    loss: dict[str, float],
                    p: PipelinePrinter = printer,
                ) -> None:
                    p.progress(
                        "train",
                        current,
                        target,
                        detail=_format_train_loss_detail(loss),
                    )

                train_summary = train(
                    dataset,
                    effective_train_config,
                    checkpoint_path=candidate_checkpoint,
                    **train_kwargs,
                    log_every=max(1, effective_train_config.steps // 10),
                    progress_callback=progress_callback,
                )
                bootstrap_once_pending = False
                if staged_pending:
                    if backup_manager is None:
                        _persist_imported_replay(
                            config=config,
                            paths=paths,
                            replay=replay,
                            pending=staged_pending,
                            imported_events=staged_imported_events,
                            printer=printer,
                        )
                        staged_pending = []
                        staged_imported_events = []
                        staged_shard_ids.clear()
                    else:
                        _save_continuous_local_replay(
                            config=config,
                            replay=replay,
                            path=active_replay_path,
                            printer=printer,
                        )
                        backup_manager.start_if_idle(
                            replay_path=active_replay_path,
                            backup_path=paths["replay_path"],
                            metadata_path=paths["metadata_path"],
                            game_log_path=paths["game_log_path"],
                            pending=staged_pending,
                            imported_events=staged_imported_events,
                            printer=printer,
                        )
                resume_optimizer_lr_override = None
                train_budget_samples = max(
                    0.0,
                    train_budget_samples - train_steps * train_config.batch_size,
                )
                training_latest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(candidate_checkpoint, training_latest)
                printer.done(
                    f"training complete: {train_summary.start_step}->{train_summary.end_step}"
                )
                onnx_path = _onnx_output_path(config)
                if config.export_onnx:
                    temporary_onnx_path = onnx_path.with_suffix(f"{onnx_path.suffix}.tmp")
                    printer.step(f"exporting raw learner checkpoint -> {onnx_path}")
                    export_onnx(
                        training_latest,
                        temporary_onnx_path,
                        device=config.onnx_device,
                        precision=config.onnx_precision,
                        dummy_batch_size=config.onnx_dummy_batch_size,
                        prefer_ema=False,
                    )
                    temporary_onnx_path.replace(onnx_path)
                    printer.done(f"onnx ready: {onnx_path}")
                    if config.export_ema_onnx:
                        ema_onnx_path = _ema_onnx_output_path(config)
                        temporary_ema_onnx_path = ema_onnx_path.with_suffix(
                            f"{ema_onnx_path.suffix}.tmp"
                        )
                        printer.step(f"exporting ema learner checkpoint -> {ema_onnx_path}")
                        export_onnx(
                            training_latest,
                            temporary_ema_onnx_path,
                            device=config.onnx_device,
                            precision=config.onnx_precision,
                            dummy_batch_size=config.onnx_dummy_batch_size,
                            prefer_ema=True,
                        )
                        temporary_ema_onnx_path.replace(ema_onnx_path)
                        printer.done(f"ema onnx ready: {ema_onnx_path}")
                else:
                    onnx_path = None
                trained = True
                train_start_step = int(train_summary.start_step)
                train_end_step = int(train_summary.end_step)
                train_chunks += 1
            elif len(replay) < config.min_replay_transitions:
                printer.done(
                    f"waiting for replay: {len(replay)}/{config.min_replay_transitions} "
                    "transitions"
                )
            elif train_budget_samples < train_config.batch_size:
                printer.done(
                    f"waiting for train budget: {int(train_budget_samples)}/"
                    f"{train_config.batch_size} samples"
                )
            else:
                printer.done("waiting for learner work")

            if pending and not trained and not config.defer_replay_save_until_train:
                if backup_manager is None:
                    _persist_imported_replay(
                        config=config,
                        paths=paths,
                        replay=replay,
                        pending=staged_pending,
                        imported_events=staged_imported_events,
                        printer=printer,
                    )
                    staged_pending = []
                    staged_imported_events = []
                    staged_shard_ids.clear()
                else:
                    _save_continuous_local_replay(
                        config=config,
                        replay=replay,
                        path=active_replay_path,
                        printer=printer,
                    )
                    backup_manager.start_if_idle(
                        replay_path=active_replay_path,
                        backup_path=paths["replay_path"],
                        metadata_path=paths["metadata_path"],
                        game_log_path=paths["game_log_path"],
                        pending=staged_pending,
                        imported_events=staged_imported_events,
                        printer=printer,
                    )
            elif pending and not trained:
                printer.done(
                    "deferred replay save until train: "
                    f"staged_shards={len(staged_pending)}"
                )
            if pending and config.prune_artifacts and backup_manager is None:
                prune_summary = _prune_learner_artifacts(config=config, printer=printer)
                pruned_artifacts = prune_summary["items"]
                pruned_bytes = prune_summary["bytes"]

            cycle_seconds = time.monotonic() - cycle_started_at
            printer.done(f"learner cycle complete in {cycle_seconds:.1f}s")
            summary = LearnerV2Summary(
                imported_shards=[shard.shard_id for shard in pending],
                imported_transitions=imported_transitions,
                imported_games=imported_games,
                replay_transitions=len(replay),
                trained=trained,
                train_start_step=train_start_step,
                train_end_step=train_end_step,
                candidate_checkpoint=candidate_checkpoint,
                training_latest_checkpoint=training_latest,
                onnx_output_path=onnx_path,
                pruned_artifacts=pruned_artifacts,
                pruned_bytes=pruned_bytes,
                cycle_seconds=cycle_seconds,
            )
            if cycles is not None:
                summaries.append(summary.to_dict())
            cycle += 1
            if cycles is not None and cycle >= cycles:
                break
            has_train_work = (
                _continuous_train_steps(
                    train_budget_samples=train_budget_samples,
                    replay_transitions=len(replay),
                    train_config=train_config,
                    min_replay_transitions=config.min_replay_transitions,
                )
                > 0
            )
            if not has_train_work:
                time.sleep(args.sleep_seconds)
    finally:
        if backup_manager is not None:
            backup_manager.close()

    return summaries

def _json_payload(summaries: list[dict[str, Any]], compact: bool) -> str:
    payload: dict[str, Any] = {"summaries": summaries}
    if len(summaries) == 1:
        payload = summaries[0]
    return json.dumps(payload, indent=None if compact else 2, sort_keys=True)
