"""Separate actor and learner process entrypoints for shard-based training."""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NoReturn

from great_kingdom_ai.actor_learner_shards import (
    ShardRecord,
    append_shard_event,
    default_shard_id,
    load_game_logs,
    load_shard_records,
    load_shard_samples,
    pending_shards,
    process_paths,
    save_shard,
    utc_now,
)
from great_kingdom_ai.pipeline import PipelinePrinter
from great_kingdom_ai.replay_buffer import ReplayBuffer
from great_kingdom_ai.rust_onnx_replay import RustReplayImportSummary, import_rust_self_play_samples
from great_kingdom_ai.rust_onnx_self_play import (
    RustOnnxSelfPlayConfig,
    RustSelfPlayRunSummary,
    run_rust_onnx_self_play,
)
from great_kingdom_ai.self_play import SelfPlayConfig
from great_kingdom_ai.train import TrainingConfig, load_training_config, train_from_replay


@dataclass(frozen=True)
class ActorProcessConfig:
    work_dir: Path = Path("data/actor-learner")
    onnx_model_path: Path = Path("data/actor-learner/checkpoints/onnx/best.onnx")
    model_version: str = "best"
    shard_id: str | None = None
    games: int = 2
    seed_start: int = 0
    onnx_device: str = "cpu"
    onnx_max_batch_size: int = 128
    rust_self_play_batch_size: int = 2
    self_play: SelfPlayConfig = SelfPlayConfig()


@dataclass(frozen=True)
class LearnerProcessConfig:
    work_dir: Path = Path("data/actor-learner")
    replay_capacity: int = 10000
    source_checkpoint: Path | None = None
    candidate_checkpoint: Path | None = None
    training_latest_checkpoint: Path | None = None
    train_checkpoint_mode: str = "resume"


@dataclass(frozen=True)
class ActorProcessSummary:
    shard: ShardRecord

    def to_dict(self) -> dict[str, Any]:
        return {"shard": self.shard.to_dict()}


@dataclass(frozen=True)
class LearnerProcessSummary:
    imported_shards: list[str]
    imported_samples: int
    imported_games: int
    replay_samples: int
    train_start_step: int
    train_end_step: int
    candidate_checkpoint: Path
    training_latest_checkpoint: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "imported_shards": self.imported_shards,
            "imported_samples": self.imported_samples,
            "imported_games": self.imported_games,
            "replay_samples": self.replay_samples,
            "train_start_step": self.train_start_step,
            "train_end_step": self.train_end_step,
            "candidate_checkpoint": str(self.candidate_checkpoint),
            "training_latest_checkpoint": str(self.training_latest_checkpoint),
        }


def run_actor_process_once(
    config: ActorProcessConfig,
    *,
    runner: Callable[[RustOnnxSelfPlayConfig], RustSelfPlayRunSummary] | None = None,
    printer: PipelinePrinter | None = None,
) -> ActorProcessSummary:
    _validate_actor_config(config)
    printer = printer if printer is not None else PipelinePrinter()
    run_self_play = runner if runner is not None else run_rust_onnx_self_play
    shard_id = config.shard_id or default_shard_id(
        model_version=config.model_version,
        seed_start=config.seed_start,
        games=config.games,
    )
    paths = process_paths(config.work_dir)
    shard_dir = paths["shard_root"] / shard_id
    replay_path = shard_dir / "replay.npz"
    log_path = shard_dir / "game_logs.json"
    if replay_path.exists() or log_path.exists():
        raise FileExistsError(f"shard already exists: {shard_dir}")

    printer.title("Actor")
    printer.metric("work dir", config.work_dir)
    printer.metric("model version", config.model_version)
    printer.metric("onnx model", config.onnx_model_path)
    printer.step(f"generating shard {shard_id}")
    self_play_summary = run_self_play(
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
    if len(self_play_summary.replay_samples) != self_play_summary.samples:
        raise RuntimeError("actor runner returned inconsistent sample count")
    if len(self_play_summary.game_logs) != self_play_summary.games:
        raise RuntimeError("actor runner returned inconsistent game count")

    save_shard(
        shard_dir,
        samples=self_play_summary.replay_samples,
        logs=self_play_summary.game_logs,
    )
    record = ShardRecord(
        shard_id=shard_id,
        status="completed",
        shard_dir=shard_dir,
        replay_path=replay_path,
        log_path=log_path,
        model_version=config.model_version,
        model_path=config.onnx_model_path,
        seed_start=config.seed_start,
        games=self_play_summary.games,
        samples=self_play_summary.samples,
        created_at=utc_now(),
    )
    append_shard_event(paths["metadata_path"], {"event": "shard_completed", **record.to_dict()})
    printer.done(f"wrote shard {shard_id}: games={record.games}, samples={record.samples}")
    return ActorProcessSummary(shard=record)


def run_learner_process_once(
    config: LearnerProcessConfig,
    train_config: TrainingConfig,
    *,
    trainer: Callable[..., Any] | None = None,
    printer: PipelinePrinter | None = None,
) -> LearnerProcessSummary:
    _validate_learner_config(config)
    printer = printer if printer is not None else PipelinePrinter()
    train = trainer if trainer is not None else train_from_replay
    paths = process_paths(config.work_dir)
    _ensure_learner_dirs(paths)
    pending = pending_shards(paths["metadata_path"])

    printer.title("Learner")
    printer.metric("work dir", config.work_dir)
    printer.metric("pending shards", len(pending))
    imported: list[RustReplayImportSummary] = []
    for shard in pending:
        printer.step(f"importing shard {shard.shard_id}")
        samples = load_shard_samples(shard.replay_path)
        logs = load_game_logs(shard.log_path)
        summary = import_rust_self_play_samples(
            artifact_dir=shard.shard_dir,
            samples=samples,
            logs=logs,
            replay_path=paths["replay_path"],
            replay_capacity=config.replay_capacity,
            game_log_path=paths["game_log_path"],
            materialize_raw_replay=True,
        )
        imported.append(summary)
        append_shard_event(
            paths["metadata_path"],
            {
                "event": "shard_imported",
                "shard_id": shard.shard_id,
                "imported_at": utc_now(),
                "imported_samples": summary.imported_samples,
                "imported_games": summary.imported_games,
                "replay_samples": summary.replay_samples,
            },
        )

    if not paths["replay_path"].exists():
        raise FileNotFoundError(f"no replay is available for learner: {paths['replay_path']}")
    replay = ReplayBuffer.load(paths["replay_path"])
    printer.metric("replay samples", len(replay))

    candidate_checkpoint = _candidate_checkpoint(config)
    training_latest = _training_latest_checkpoint(config)
    source_checkpoint = _source_checkpoint(config)
    kwargs = _train_checkpoint_kwargs(
        train_checkpoint_mode=config.train_checkpoint_mode,
        source_checkpoint=source_checkpoint,
    )
    printer.step(f"training candidate -> {candidate_checkpoint}")
    train_summary = train(
        replay,
        train_config,
        checkpoint_path=candidate_checkpoint,
        **kwargs,
        log_every=max(1, train_config.steps // 10),
        progress_callback=lambda current, target, loss: printer.progress(
            "train",
            current,
            target,
            detail=f"loss={loss['total']:.4f}",
        ),
    )
    training_latest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidate_checkpoint, training_latest)
    printer.done(f"training complete: {train_summary.start_step}->{train_summary.end_step}")

    return LearnerProcessSummary(
        imported_shards=[shard.shard_id for shard in pending],
        imported_samples=sum(summary.imported_samples for summary in imported),
        imported_games=sum(summary.imported_games for summary in imported),
        replay_samples=len(replay),
        train_start_step=int(train_summary.start_step),
        train_end_step=int(train_summary.end_step),
        candidate_checkpoint=candidate_checkpoint,
        training_latest_checkpoint=training_latest,
    )


def load_actor_config(path: str | Path) -> ActorProcessConfig:
    data = _load_json_object(path, "actor config")
    if "work_dir" in data:
        data["work_dir"] = Path(data["work_dir"])
    if "onnx_model_path" in data:
        data["onnx_model_path"] = Path(data["onnx_model_path"])
    self_play_data = data.pop("self_play", None)
    if isinstance(self_play_data, dict):
        data["self_play"] = SelfPlayConfig(**self_play_data)
    return ActorProcessConfig(**data)


def load_learner_config(path: str | Path) -> LearnerProcessConfig:
    data = _load_json_object(path, "learner config")
    path_keys = (
        "work_dir",
        "source_checkpoint",
        "candidate_checkpoint",
        "training_latest_checkpoint",
    )
    for key in path_keys:
        if data.get(key) is not None:
            data[key] = Path(data[key])
    return LearnerProcessConfig(**data)


def build_actor_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="great-kingdom-actor",
        description="Generate one append-only self-play shard for a learner process.",
    )
    parser.add_argument("--actor-config", type=Path, default=None)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--onnx-model", type=Path, default=None)
    parser.add_argument("--model-version", default=None)
    parser.add_argument("--shard-id", default=None)
    parser.add_argument("--games", type=int, default=None)
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument("--onnx-device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--onnx-max-batch-size", type=int, default=None)
    parser.add_argument("--rust-self-play-batch-size", type=int, default=None)
    parser.add_argument("--json", action="store_true")
    return parser


def build_learner_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="great-kingdom-learner",
        description="Import completed actor shards and train one candidate checkpoint.",
    )
    parser.add_argument("--learner-config", type=Path, default=None)
    parser.add_argument(
        "--train-config",
        type=Path,
        default=Path("configs/test/m8-train-smoke.json"),
    )
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--replay-capacity", type=int, default=None)
    parser.add_argument("--source-checkpoint", type=Path, default=None)
    parser.add_argument("--candidate-checkpoint", type=Path, default=None)
    parser.add_argument("--training-latest-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--train-checkpoint-mode",
        choices=["resume", "bootstrap"],
        default=None,
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--train-steps", type=int, default=None)
    parser.add_argument("--json", action="store_true")
    return parser


def actor_main() -> NoReturn:
    args = build_actor_parser().parse_args()
    config = load_actor_config(args.actor_config) if args.actor_config else ActorProcessConfig()
    data = asdict(config)
    for key, value in {
        "work_dir": args.work_dir,
        "onnx_model_path": args.onnx_model,
        "model_version": args.model_version,
        "shard_id": args.shard_id,
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
    summary = run_actor_process_once(
        ActorProcessConfig(**data),
        printer=PipelinePrinter(enabled=not args.json),
    )
    print(json.dumps(summary.to_dict(), indent=None if args.json else 2, sort_keys=True))
    raise SystemExit(0)


def learner_main() -> NoReturn:
    args = build_learner_parser().parse_args()
    config = (
        load_learner_config(args.learner_config)
        if args.learner_config
        else LearnerProcessConfig()
    )
    data = asdict(config)
    for key, value in {
        "work_dir": args.work_dir,
        "replay_capacity": args.replay_capacity,
        "source_checkpoint": args.source_checkpoint,
        "candidate_checkpoint": args.candidate_checkpoint,
        "training_latest_checkpoint": args.training_latest_checkpoint,
        "train_checkpoint_mode": args.train_checkpoint_mode,
    }.items():
        if value is not None:
            data[key] = value
    train = load_training_config(args.train_config)
    train_data = asdict(train)
    if args.device is not None:
        train_data["device"] = args.device
    if args.train_steps is not None:
        train_data["steps"] = args.train_steps
    summary = run_learner_process_once(
        LearnerProcessConfig(**data),
        TrainingConfig(**train_data),
        printer=PipelinePrinter(enabled=not args.json),
    )
    print(json.dumps(summary.to_dict(), indent=None if args.json else 2, sort_keys=True))
    raise SystemExit(0)


def _validate_actor_config(config: ActorProcessConfig) -> None:
    if config.games <= 0:
        raise ValueError("games must be positive")
    if config.seed_start < 0:
        raise ValueError("seed_start must be non-negative")
    if config.onnx_max_batch_size <= 0:
        raise ValueError("onnx_max_batch_size must be positive")
    if config.rust_self_play_batch_size <= 0:
        raise ValueError("rust_self_play_batch_size must be positive")


def _validate_learner_config(config: LearnerProcessConfig) -> None:
    if config.replay_capacity <= 0:
        raise ValueError("replay_capacity must be positive")
    if config.train_checkpoint_mode not in {"resume", "bootstrap"}:
        raise ValueError("train_checkpoint_mode must be one of: resume, bootstrap")


def _ensure_learner_dirs(paths: dict[str, Path]) -> None:
    paths["replay_path"].parent.mkdir(parents=True, exist_ok=True)
    paths["candidate_checkpoint"].parent.mkdir(parents=True, exist_ok=True)


def _candidate_checkpoint(config: LearnerProcessConfig) -> Path:
    return config.candidate_checkpoint or process_paths(config.work_dir)["candidate_checkpoint"]


def _training_latest_checkpoint(config: LearnerProcessConfig) -> Path:
    return (
        config.training_latest_checkpoint
        or process_paths(config.work_dir)["training_latest_checkpoint"]
    )


def _source_checkpoint(config: LearnerProcessConfig) -> Path:
    return config.source_checkpoint or process_paths(config.work_dir)["training_latest_checkpoint"]


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


def _load_json_object(path: str | Path, label: str) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{label} must be a JSON object")
    return data


if __name__ == "__main__":
    actor_main()


__all__ = [
    "ActorProcessConfig",
    "ActorProcessSummary",
    "LearnerProcessConfig",
    "LearnerProcessSummary",
    "ShardRecord",
    "actor_main",
    "learner_main",
    "load_actor_config",
    "load_learner_config",
    "load_shard_records",
    "pending_shards",
    "run_actor_process_once",
    "run_learner_process_once",
]
