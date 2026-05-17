"""Training configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 32
    steps: int = 1000
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    value_loss_weight: float = 1.0
    policy_loss_weight: float = 1.0
    l2_loss_weight: float = 0.0
    lr_schedule: str = "step"
    lr_decay_gamma: float = 0.99
    lr_decay_steps: int = 100
    lr_warmup_steps: int = 0
    lr_min_factor: float = 0.1
    lr_cosine_steps: int = 0
    seed: int = 0
    device: str = "cpu"
    model_preset: str = "small"
    symmetry_augmentation: bool = True
    mask_policy_loss: bool = True
    amp: bool = False
    recent_sample_fraction: float = 0.0
    recent_sample_window: int = 0
    priority_enabled: bool = False
    priority_alpha: float = 0.6
    priority_beta: float = 0.4
    priority_value_error_weight: float = 1.0
    priority_policy_kl_weight: float = 1.0
    priority_target_age_weight: float = 0.25
    priority_search_reanalyzed_boost: float = 1.0
    priority_max_priority: float | None = 64.0
    prefetch_batches: int = 1
    ema_decay: float | None = None



def load_training_config(path: str | Path) -> TrainingConfig:
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError("training config must be a JSON object")
    return TrainingConfig(**data)


__all__ = ["TrainingConfig", "load_training_config"]
