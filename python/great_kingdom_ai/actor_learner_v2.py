"""V2 trajectory-replay actor and learner process entrypoints."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, NoReturn

from great_kingdom_ai.onnx_export import export_checkpoint_to_onnx
from great_kingdom_ai.pipeline_printer import PipelinePrinter
from great_kingdom_ai.runpod_pruning import PruneItem, prune_items
from great_kingdom_ai.rust_onnx_self_play import (
    RustOnnxSelfPlayConfig,
    RustSelfPlayRunSummary,
    run_rust_onnx_self_play,
)
from great_kingdom_ai.self_play import GameLog, MoveLog, SelfPlayConfig
from great_kingdom_ai.train import TrainingConfig, load_training_config, train_from_replay
from great_kingdom_ai.trajectory_dataset import TrajectoryReplayDataset
from great_kingdom_ai.trajectory_replay import TrajectoryEpisode, TrajectoryReplayStore

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


@dataclass(frozen=True)
class ActorV2Config:
    work_dir: Path = Path("data/runpod/async-v2")
    onnx_model_path: Path = Path("data/runpod/async-v2/checkpoints/onnx/training-latest.onnx")
    model_version: str = "latest"
    shard_id: str | None = None
    games: int = 64
    seed_start: int = 0
    onnx_device: str = "cuda"
    onnx_max_batch_size: int = 4096
    rust_self_play_batch_size: int = 512
    self_play: SelfPlayConfig = SelfPlayConfig()


@dataclass(frozen=True)
class LearnerV2Config:
    work_dir: Path = Path("data/runpod/async-v2")
    replay_capacity: int = 512000
    min_replay_transitions: int = 8192
    source_checkpoint: Path | None = None
    candidate_checkpoint: Path | None = None
    training_latest_checkpoint: Path | None = None
    train_checkpoint_mode: str = "resume"
    export_onnx: bool = True
    onnx_output_path: Path | None = None
    onnx_device: str = "cuda"
    onnx_precision: str = "fp16"
    onnx_dummy_batch_size: int = 2
    prune_artifacts: bool = False
    prune_keep_imported_shards: int = 0
    train_reuse_factor: float = 16.0


@dataclass(frozen=True)
class ActorV2Summary:
    shard: V2ShardRecord

    def to_dict(self) -> dict[str, Any]:
        return {"shard": self.shard.to_dict()}


@dataclass(frozen=True)
class LearnerV2Summary:
    imported_shards: list[str]
    imported_transitions: int
    imported_games: int
    replay_transitions: int | None
    trained: bool
    train_start_step: int | None
    train_end_step: int | None
    candidate_checkpoint: Path | None
    training_latest_checkpoint: Path | None
    onnx_output_path: Path | None
    pruned_artifacts: int = 0
    pruned_bytes: int = 0
    cycle_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "imported_shards": self.imported_shards,
            "imported_transitions": self.imported_transitions,
            "imported_games": self.imported_games,
            "replay_transitions": self.replay_transitions,
            "trained": self.trained,
            "train_start_step": self.train_start_step,
            "train_end_step": self.train_end_step,
            "candidate_checkpoint": (
                None if self.candidate_checkpoint is None else str(self.candidate_checkpoint)
            ),
            "training_latest_checkpoint": (
                None
                if self.training_latest_checkpoint is None
                else str(self.training_latest_checkpoint)
            ),
            "onnx_output_path": (
                None if self.onnx_output_path is None else str(self.onnx_output_path)
            ),
            "pruned_artifacts": self.pruned_artifacts,
            "pruned_bytes": self.pruned_bytes,
            "cycle_seconds": self.cycle_seconds,
        }


def run_actor_v2_once(
    config: ActorV2Config,
    *,
    runner: Callable[[RustOnnxSelfPlayConfig], RustSelfPlayRunSummary] | None = None,
    printer: PipelinePrinter | None = None,
) -> ActorV2Summary:
    _validate_actor_config(config)
    printer = printer if printer is not None else PipelinePrinter()
    run_self_play = runner if runner is not None else run_rust_onnx_self_play
    shard_id = config.shard_id or _default_shard_id(
        model_version=config.model_version,
        seed_start=config.seed_start,
        games=config.games,
    )
    paths = _paths(config.work_dir)
    shard_dir = paths["shard_root"] / shard_id
    replay_path = shard_dir / "trajectory-replay.npz"
    log_path = shard_dir / "game_logs.json"
    if replay_path.exists() or log_path.exists():
        raise FileExistsError(f"shard already exists: {shard_dir}")

    printer.title("Actor V2")
    printer.metric("work dir", config.work_dir)
    printer.metric("onnx model", config.onnx_model_path)
    printer.metric("model version", config.model_version)
    printer.metric("games", config.games)
    printer.metric("seed start", config.seed_start)
    printer.metric("onnx device", config.onnx_device)
    printer.metric("onnx max batch", config.onnx_max_batch_size)
    printer.metric("self-play batch", config.rust_self_play_batch_size)
    printer.metric("gumbel sims", config.self_play.gumbel_simulations)
    printer.metric("leaf batch", config.self_play.leaf_batch_size)
    printer.step(f"generating trajectory shard {shard_id}")
    summary = run_self_play(
        RustOnnxSelfPlayConfig(
            onnx_model_path=config.onnx_model_path,
            output_dir=shard_dir,
            games=config.games,
            seed_start=config.seed_start,
            onnx_device=config.onnx_device,
            onnx_max_batch_size=config.onnx_max_batch_size,
            rust_self_play_batch_size=config.rust_self_play_batch_size,
            self_play=config.self_play,
        )
    )
    if not summary.trajectory_episodes:
        raise RuntimeError("actor runner did not return trajectory episodes")
    transitions = sum(len(episode.transitions) for episode in summary.trajectory_episodes)
    if len(summary.game_logs) != summary.games:
        raise RuntimeError("actor runner returned inconsistent game log count")

    _save_trajectory_shard(
        shard_dir,
        episodes=summary.trajectory_episodes,
        logs=summary.game_logs,
    )
    record = V2ShardRecord(
        shard_id=shard_id,
        status="completed",
        shard_dir=shard_dir,
        replay_path=replay_path,
        log_path=log_path,
        model_version=config.model_version,
        model_path=config.onnx_model_path,
        seed_start=config.seed_start,
        games=summary.games,
        transitions=transitions,
        created_at=_utc_now(),
    )
    _append_event(paths["metadata_path"], {"event": "shard_completed", **record.to_dict()})
    average_length = transitions / max(1, record.games)
    if summary.samples != transitions:
        printer.metric("full-search samples", summary.samples)
    printer.metric("new games", record.games)
    printer.metric("new transitions", transitions)
    printer.metric("avg game length", f"{average_length:.1f}")
    printer.done(f"wrote shard {shard_id} in {printer.elapsed()}")
    return ActorV2Summary(shard=record)


def run_learner_v2_once(
    config: LearnerV2Config,
    train_config: TrainingConfig,
    *,
    trainer: Callable[..., Any] | None = None,
    onnx_exporter: Callable[..., Any] | None = None,
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
    imported_transitions = 0
    imported_games = 0
    for shard in pending:
        printer.step(f"importing shard {shard.shard_id}")
        shard_replay = TrajectoryReplayStore.load(shard.replay_path)
        replay.extend_episodes(shard_replay.episodes)
        imported_transitions += len(shard_replay)
        imported_games += shard_replay.episode_count
        _append_event(
            paths["metadata_path"],
            {
                "event": "shard_imported",
                "shard_id": shard.shard_id,
                "imported_at": _utc_now(),
                "imported_transitions": len(shard_replay),
                "replay_transitions": len(replay),
            },
        )
    replay.save(paths["replay_path"], compressed=False)
    _append_game_logs(paths["game_log_path"], pending)
    printer.metric("imported games", imported_games)
    printer.metric("imported rows", imported_transitions)
    printer.metric("replay transitions", len(replay))

    if len(replay) < config.min_replay_transitions:
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
    train_summary = train(
        dataset,
        train_config,
        checkpoint_path=candidate_checkpoint,
        **kwargs,
        log_every=max(1, train_config.steps // 10),
        progress_callback=lambda current, target, loss: printer.progress(
            "train",
            current,
            target,
            detail=_format_train_loss_detail(loss),
        ),
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
            prefer_ema=True,
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


def load_actor_v2_config(path: str | Path) -> ActorV2Config:
    data = _load_json_object(path, "actor v2 config")
    for key in ("work_dir", "onnx_model_path"):
        if key in data:
            data[key] = Path(data[key])
    self_play_data = data.pop("self_play", None)
    if isinstance(self_play_data, dict):
        data["self_play"] = SelfPlayConfig(**self_play_data)
    return ActorV2Config(**data)


def load_learner_v2_config(path: str | Path) -> LearnerV2Config:
    data = _load_json_object(path, "learner v2 config")
    for key in (
        "work_dir",
        "source_checkpoint",
        "candidate_checkpoint",
        "training_latest_checkpoint",
        "onnx_output_path",
    ):
        if data.get(key) is not None:
            data[key] = Path(data[key])
    return LearnerV2Config(**data)


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


def build_actor_v2_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="great-kingdom-actor-v2",
        description="Generate v2 trajectory self-play shards.",
    )
    parser.add_argument("--actor-config", type=Path, default=None)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--onnx-model", type=Path, default=None)
    parser.add_argument("--model-version", default=None)
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
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--max-cycles", type=int, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument("--json", action="store_true")
    return parser


def actor_v2_main() -> NoReturn:
    args = build_actor_v2_parser().parse_args()
    base = load_actor_v2_config(args.actor_config) if args.actor_config else ActorV2Config()
    data = asdict(base)
    for key, value in {
        "work_dir": args.work_dir,
        "onnx_model_path": args.onnx_model,
        "model_version": args.model_version,
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
    seed_cursor = _next_actor_seed_start(config)
    while cycles is None or cycle < cycles:
        shard_config = ActorV2Config(
            work_dir=config.work_dir,
            onnx_model_path=config.onnx_model_path,
            model_version=config.model_version,
            shard_id=config.shard_id if not args.loop else None,
            games=config.games,
            seed_start=seed_cursor,
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
        seed_cursor = summary.shard.seed_start + summary.shard.games
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
    train = train_from_replay
    export_onnx = export_checkpoint_to_onnx
    summaries: list[dict[str, Any]] = []
    train_budget_samples = 0.0
    cycles = args.max_cycles
    cycle = 0
    train_chunks = 0

    while cycles is None or cycle < cycles:
        cycle_started_at = time.monotonic()
        printer = PipelinePrinter(enabled=not args.json)
        pending = pending_v2_shards(paths["metadata_path"])
        imported_transitions = 0
        imported_games = 0

        printer.title("Learner V2 Continuous")
        printer.metric("work dir", config.work_dir)
        printer.metric("pending shards", len(pending))
        printer.metric("replay transitions", len(replay))
        printer.metric("train batch", train_config.batch_size)
        printer.metric("max train steps", train_config.steps)
        printer.metric("reuse factor", config.train_reuse_factor)
        printer.metric("budget samples", int(train_budget_samples))

        for shard in pending:
            printer.step(f"importing shard {shard.shard_id}")
            shard_replay = TrajectoryReplayStore.load(shard.replay_path)
            replay.extend_episodes(shard_replay.episodes)
            imported_transitions += len(shard_replay)
            imported_games += shard_replay.episode_count
            _append_event(
                paths["metadata_path"],
                {
                    "event": "shard_imported",
                    "shard_id": shard.shard_id,
                    "imported_at": _utc_now(),
                    "imported_transitions": len(shard_replay),
                    "replay_transitions": len(replay),
                },
            )

        if pending:
            replay.save(paths["replay_path"], compressed=False)
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
            train_summary = train(
                dataset,
                effective_train_config,
                checkpoint_path=candidate_checkpoint,
                **kwargs,
                log_every=max(1, effective_train_config.steps // 10),
                progress_callback=lambda current, target, loss, p=printer: p.progress(
                    "train",
                    current,
                    target,
                    detail=_format_train_loss_detail(loss),
                ),
            )
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


def _save_trajectory_shard(
    shard_dir: Path,
    *,
    episodes: tuple[TrajectoryEpisode, ...],
    logs: tuple[GameLog, ...],
) -> None:
    shard_dir.mkdir(parents=True, exist_ok=True)
    transitions = sum(len(episode.transitions) for episode in episodes)
    TrajectoryReplayStore.from_episodes(max(1, transitions), episodes).save(
        shard_dir / "trajectory-replay.npz",
        compressed=False,
    )
    with (shard_dir / "game_logs.json").open("w", encoding="utf-8") as file:
        json.dump([log.to_dict() for log in logs], file, indent=2, sort_keys=True)


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


def _load_or_create_replay(path: Path, *, capacity: int) -> TrajectoryReplayStore:
    if path.exists():
        replay = TrajectoryReplayStore.load(path)
        if replay.capacity == capacity:
            return replay
        return TrajectoryReplayStore.from_episodes(capacity, replay.episodes)
    return TrajectoryReplayStore.empty(capacity)


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


def _next_actor_seed_start(config: ActorV2Config) -> int:
    records = load_v2_shard_records(_paths(config.work_dir)["metadata_path"])
    next_seed = config.seed_start
    for record in records:
        if record.model_version != config.model_version:
            continue
        next_seed = max(next_seed, record.seed_start + record.games)
    return next_seed


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


def _paths(work_dir: Path) -> dict[str, Path]:
    return {
        "shard_root": work_dir / "shards",
        "metadata_path": work_dir / "shards" / "metadata.jsonl",
        "replay_path": work_dir / "replay" / "trajectory-replay.npz",
        "game_log_path": work_dir / "replay" / "game_logs.jsonl",
        "candidate_checkpoint": work_dir / "checkpoints" / "candidate.pt",
        "training_latest_checkpoint": work_dir / "checkpoints" / "training-latest.pt",
        "best_checkpoint": work_dir / "checkpoints" / "best.pt",
        "onnx_output_path": work_dir / "checkpoints" / "onnx" / "training-latest.onnx",
    }


def _ensure_learner_dirs(paths: dict[str, Path]) -> None:
    paths["replay_path"].parent.mkdir(parents=True, exist_ok=True)
    paths["candidate_checkpoint"].parent.mkdir(parents=True, exist_ok=True)
    paths["onnx_output_path"].parent.mkdir(parents=True, exist_ok=True)


def _candidate_checkpoint(config: LearnerV2Config) -> Path:
    return config.candidate_checkpoint or _paths(config.work_dir)["candidate_checkpoint"]


def _training_latest_checkpoint(config: LearnerV2Config) -> Path:
    return (
        config.training_latest_checkpoint
        or _paths(config.work_dir)["training_latest_checkpoint"]
    )


def _source_checkpoint(config: LearnerV2Config) -> Path:
    if config.source_checkpoint is not None:
        return config.source_checkpoint
    training_latest = _training_latest_checkpoint(config)
    if training_latest.exists():
        return training_latest
    return _paths(config.work_dir)["best_checkpoint"]


def _onnx_output_path(config: LearnerV2Config) -> Path:
    return config.onnx_output_path or _paths(config.work_dir)["onnx_output_path"]


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


def _default_shard_id(*, model_version: str, seed_start: int, games: int) -> str:
    return f"{_safe_id(model_version)}-seed-{seed_start:08d}-games-{games:04d}"


def _safe_id(value: str) -> str:
    return "".join(char if char.isalnum() or char in "_.-" else "-" for char in value).strip("-")


def _utc_now() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds")


def _load_json_object(path: str | Path, label: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"{label} must be a JSON object")
    return data


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


def _json_payload(summaries: list[dict[str, Any]], compact: bool) -> str:
    payload: dict[str, Any] = {"summaries": summaries}
    if len(summaries) == 1:
        payload = summaries[0]
    return json.dumps(payload, indent=None if compact else 2, sort_keys=True)


def _validate_actor_config(config: ActorV2Config) -> None:
    if config.games <= 0:
        raise ValueError("games must be positive")
    if config.seed_start < 0:
        raise ValueError("seed_start must be non-negative")
    if config.onnx_max_batch_size <= 0:
        raise ValueError("onnx_max_batch_size must be positive")
    if config.rust_self_play_batch_size <= 0:
        raise ValueError("rust_self_play_batch_size must be positive")


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


__all__ = [
    "ActorV2Config",
    "ActorV2Summary",
    "LearnerV2Config",
    "LearnerV2Summary",
    "V2ShardRecord",
    "actor_v2_main",
    "learner_v2_main",
    "load_actor_v2_config",
    "load_learner_v2_config",
    "load_v2_shard_records",
    "pending_v2_shards",
    "run_actor_v2_once",
    "run_learner_v2_once",
]
