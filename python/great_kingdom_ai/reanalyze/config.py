"""Reanalyze configuration types."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

from great_kingdom_ai.search_reanalyze import SearchReanalyzeConfig

ReanalyzeProgressCallback = Callable[[str, int, int, str], None]


@dataclass(frozen=True)
class ReanalyzeConfig:
    batch_size: int = 1024
    device: str = "cpu"
    onnx_model_path: str | None = None
    onnx_device: str | None = None
    onnx_max_batch_size: int = 1024
    bootstrap_td_steps: int = 0
    gamma: float = 1.0
    dynamic_horizon_enabled: bool = False
    dynamic_horizon_tau: float = 0.3
    dynamic_horizon_total_steps: int | None = None
    value_bootstrap_source: str = "value_head"
    policy_reanalyze_ratio: float = 0.0
    model_version: int | None = None
    compressed: bool = True
    search: SearchReanalyzeConfig = field(default_factory=SearchReanalyzeConfig)

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.onnx_max_batch_size <= 0:
            raise ValueError("onnx_max_batch_size must be positive")
        if self.bootstrap_td_steps < 0:
            raise ValueError("bootstrap_td_steps must be non-negative")
        if not math.isfinite(self.gamma) or not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be finite and in [0, 1]")
        if not math.isfinite(self.dynamic_horizon_tau) or self.dynamic_horizon_tau <= 0.0:
            raise ValueError("dynamic_horizon_tau must be finite and positive")
        if self.dynamic_horizon_total_steps is not None and self.dynamic_horizon_total_steps <= 0:
            raise ValueError("dynamic_horizon_total_steps must be positive")
        if self.dynamic_horizon_enabled and self.dynamic_horizon_total_steps is None:
            raise ValueError(
                "dynamic_horizon_total_steps is required when dynamic horizon is enabled"
            )
        if self.value_bootstrap_source not in {"value_head", "mcts_root"}:
            raise ValueError("value_bootstrap_source must be one of: value_head, mcts_root")
        if not math.isfinite(self.policy_reanalyze_ratio) or not 0.0 <= (
            self.policy_reanalyze_ratio
        ) <= 1.0:
            raise ValueError("policy_reanalyze_ratio must be finite and in [0, 1]")
        if self.model_version is not None and self.model_version < 0:
            raise ValueError("model_version must be non-negative")


__all__ = ["ReanalyzeConfig", "ReanalyzeProgressCallback"]
