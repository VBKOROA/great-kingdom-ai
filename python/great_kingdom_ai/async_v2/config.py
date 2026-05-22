"""Async v2 configuration and summaries."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from great_kingdom_ai.self_play import SelfPlayConfig

if TYPE_CHECKING:
    from great_kingdom_ai.async_v2.metadata import V2ShardRecord

@dataclass(frozen=True)
class ActorV2Config:
    work_dir: Path = Path("data/runpod/train-v3")
    onnx_model_path: Path = Path("data/runpod/train-v3/checkpoints/onnx/training-latest.onnx")
    ema_onnx_model_path: Path | None = None
    ema_opponent_fraction: float = 0.0
    model_version: str = "latest"
    model_iteration: int | None = None
    shard_id: str | None = None
    games: int = 64
    seed_start: int = 0
    onnx_device: str = "cuda"
    onnx_max_batch_size: int = 4096
    rust_self_play_batch_size: int = 512
    self_play: SelfPlayConfig = SelfPlayConfig()

@dataclass(frozen=True)
class LearnerV2Config:
    work_dir: Path = Path("data/runpod/train-v3")
    replay_capacity: int = 512000
    min_replay_transitions: int = 8192
    source_checkpoint: Path | None = None
    candidate_checkpoint: Path | None = None
    training_latest_checkpoint: Path | None = None
    train_checkpoint_mode: str = "resume"
    export_onnx: bool = True
    onnx_output_path: Path | None = None
    ema_onnx_output_path: Path | None = None
    export_ema_onnx: bool = True
    onnx_device: str = "cuda"
    onnx_precision: str = "fp16"
    onnx_dummy_batch_size: int = 2
    onnx_prefer_ema: bool = False
    prune_artifacts: bool = False
    prune_keep_imported_shards: int = 0
    train_reuse_factor: float = 16.0

@dataclass(frozen=True)
class FactoryInitV2Config:
    work_dir: Path = Path("data/runpod/train-v3")
    checkpoint_path: Path | None = None
    onnx_output_path: Path | None = None
    ema_onnx_output_path: Path | None = None
    export_ema_onnx: bool = True
    overwrite: bool = False
    onnx_device: str = "cpu"
    onnx_precision: str = "fp32"
    onnx_dummy_batch_size: int = 2
    onnx_prefer_ema: bool = False

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

@dataclass(frozen=True)
class FactoryInitV2Summary:
    checkpoint_path: Path
    onnx_output_path: Path
    model_preset: str
    step: int
    overwritten: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_path": str(self.checkpoint_path),
            "onnx_output_path": str(self.onnx_output_path),
            "model_preset": self.model_preset,
            "step": self.step,
            "overwritten": self.overwritten,
        }

def load_actor_v2_config(path: str | Path) -> ActorV2Config:
    data = _load_json_object(path, "actor v2 config")
    for key in ("work_dir", "onnx_model_path", "ema_onnx_model_path"):
        if data.get(key) is not None:
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
        "ema_onnx_output_path",
    ):
        if data.get(key) is not None:
            data[key] = Path(data[key])
    return LearnerV2Config(**data)

def _load_json_object(path: str | Path, label: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"{label} must be a JSON object")
    return data
