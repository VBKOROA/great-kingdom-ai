"""Async v2 path calculation helpers."""

from __future__ import annotations

from pathlib import Path

from great_kingdom_ai.async_v2.config import FactoryInitV2Config, LearnerV2Config


def _paths(work_dir: Path) -> dict[str, Path]:
    return {
        "shard_root": work_dir / "shards",
        "metadata_path": work_dir / "shards" / "metadata.jsonl",
        "actor_seed_lock_path": work_dir / "shards" / "actor-seed.lock",
        "actor_seed_state_path": work_dir / "shards" / "actor-seed-state.json",
        "replay_path": work_dir / "replay" / "trajectory-replay.npz",
        "game_log_path": work_dir / "replay" / "game_logs.jsonl",
        "candidate_checkpoint": work_dir / "checkpoints" / "candidate.pt",
        "training_latest_checkpoint": work_dir / "checkpoints" / "training-latest.pt",
        "best_checkpoint": work_dir / "checkpoints" / "best.pt",
        "onnx_output_path": work_dir / "checkpoints" / "onnx" / "training-latest.onnx",
        "ema_onnx_output_path": work_dir / "checkpoints" / "onnx" / "training-latest-ema.onnx",
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

def _ema_onnx_output_path(config: LearnerV2Config) -> Path:
    return config.ema_onnx_output_path or _paths(config.work_dir)["ema_onnx_output_path"]

def _factory_checkpoint_path(config: FactoryInitV2Config) -> Path:
    return config.checkpoint_path or _paths(config.work_dir)["training_latest_checkpoint"]

def _factory_onnx_output_path(config: FactoryInitV2Config) -> Path:
    return config.onnx_output_path or _paths(config.work_dir)["onnx_output_path"]

def _factory_ema_onnx_output_path(config: FactoryInitV2Config) -> Path:
    return config.ema_onnx_output_path or _paths(config.work_dir)["ema_onnx_output_path"]
