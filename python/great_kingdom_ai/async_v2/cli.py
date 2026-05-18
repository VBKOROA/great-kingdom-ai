"""Async v2 command-line entrypoints."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import asdict
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
    _append_shard_import_event,
    _continuous_train_steps,
    _drop_async_unused_replay_arrays,
    _format_train_loss_detail,
    _import_shards_into_replay,
    _load_or_create_replay,
    _print_learner_optimizer_state,
    _prune_learner_artifacts,
    _save_replay_with_timing,
    _train_checkpoint_kwargs,
    _validate_learner_config,
    run_learner_v2_once,
)
from great_kingdom_ai.async_v2.metadata import (
    _append_game_logs,
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
from great_kingdom_ai.replay import TrajectoryReplayDataset
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
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--train-steps", type=int, default=None)
    parser.add_argument("--no-export-onnx", action="store_true")
    parser.add_argument("--onnx-device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--onnx-precision", choices=["fp32", "fp16"], default=None)
    parser.add_argument("--prune-artifacts", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--prune-keep-imported-shards", type=int, default=None)
    parser.add_argument("--train-reuse-factor", type=float, default=None)
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
        overwrite=args.overwrite,
        onnx_device=args.onnx_device or FactoryInitV2Config.onnx_device,
        onnx_precision=args.onnx_precision or FactoryInitV2Config.onnx_precision,
        onnx_dummy_batch_size=(
            args.onnx_dummy_batch_size
            if args.onnx_dummy_batch_size is not None
            else FactoryInitV2Config.onnx_dummy_batch_size
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
        "prune_artifacts": args.prune_artifacts,
        "prune_keep_imported_shards": args.prune_keep_imported_shards,
        "train_reuse_factor": args.train_reuse_factor,
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
    if args.loop:
        return _run_learner_continuous_cli(config, train_config, args)
    summaries = []
    summary = run_learner_v2_once(
        config,
        train_config,
        resume_optimizer_lr_override=args.override_optimizer_lr,
        printer=PipelinePrinter(enabled=not args.json),
    )
    summaries.append(summary.to_dict())
    return summaries

def _run_learner_continuous_cli(
    config: LearnerV2Config,
    train_config: TrainingConfig,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    _validate_learner_config(config)
    paths = _paths(config.work_dir)
    _ensure_learner_dirs(paths)
    replay = _load_or_create_replay(paths["replay_path"], capacity=config.replay_capacity)
    _drop_async_unused_replay_arrays(replay)
    train = train_from_replay
    export_onnx = export_checkpoint_to_onnx
    summaries: list[dict[str, Any]] = []
    train_budget_samples = 0.0
    resume_optimizer_lr_override = args.override_optimizer_lr
    cycles = args.max_cycles
    cycle = 0
    train_chunks = 0

    while cycles is None or cycle < cycles:
        cycle_started_at = time.monotonic()
        printer = PipelinePrinter(enabled=not args.json)
        pending = pending_v2_shards(paths["metadata_path"])
        imported_transitions = 0
        imported_games = 0
        imported_events: list[tuple[str, Any, int]] = []

        printer.title("Learner V2 Continuous")
        printer.metric("work dir", config.work_dir)
        printer.metric("pending shards", len(pending))
        printer.metric("replay transitions", len(replay))
        printer.metric("train batch", train_config.batch_size)
        printer.metric("max train steps", train_config.steps)
        printer.metric("reuse factor", config.train_reuse_factor)
        printer.metric("budget samples", int(train_budget_samples))

        imported_events = _import_shards_into_replay(replay, pending, printer=printer)
        imported_transitions = sum(stats.transitions for _, stats, _ in imported_events)
        imported_games = sum(stats.games for _, stats, _ in imported_events)

        if pending:
            _save_replay_with_timing(replay, paths["replay_path"], printer=printer)
            for shard_id, stats, replay_transitions in imported_events:
                _append_shard_import_event(
                    paths["metadata_path"],
                    shard_id=shard_id,
                    stats=stats,
                    replay_transitions=replay_transitions,
                )
            _append_game_logs(paths["game_log_path"], pending)
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
        pruned_artifacts = 0
        pruned_bytes = 0

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
            kwargs = _train_checkpoint_kwargs(
                train_checkpoint_mode=config.train_checkpoint_mode,
                source_checkpoint=_source_checkpoint(config),
            )
            printer.step(
                f"training candidate -> {candidate_checkpoint} "
                f"(steps={train_steps}, budget={int(train_budget_samples)})"
            )
            train_kwargs: dict[str, Any] = {
                **kwargs,
            }
            _print_learner_optimizer_state(
                printer,
                train_checkpoint_mode=config.train_checkpoint_mode,
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
                printer.step(f"exporting learner checkpoint -> {onnx_path}")
                export_onnx(
                    training_latest,
                    temporary_onnx_path,
                    device=config.onnx_device,
                    precision=config.onnx_precision,
                    dummy_batch_size=config.onnx_dummy_batch_size,
                    prefer_ema=True,
                )
                temporary_onnx_path.replace(onnx_path)
                printer.done(f"onnx ready: {onnx_path}")
            else:
                onnx_path = None
            if config.prune_artifacts:
                prune_summary = _prune_learner_artifacts(config=config, printer=printer)
                pruned_artifacts = prune_summary["items"]
                pruned_bytes = prune_summary["bytes"]
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

    return summaries

def _json_payload(summaries: list[dict[str, Any]], compact: bool) -> str:
    payload: dict[str, Any] = {"summaries": summaries}
    if len(summaries) == 1:
        payload = summaries[0]
    return json.dumps(payload, indent=None if compact else 2, sort_keys=True)
