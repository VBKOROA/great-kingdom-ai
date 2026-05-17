"""Reanalyze run summary types."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ReanalyzeSummary:
    replay_path: Path
    checkpoint_path: Path
    output_path: Path
    transitions: int
    model_version: int
    bootstrap_td_steps: int
    gamma: float
    search_reanalyzed: int = 0
    reanalyze_mode: str = "snapshot"
    value_bootstrap_source: str = "value_head"
    dynamic_horizon_enabled: bool = False
    sampled_batches: int = 0
    sampled_rows: int = 0
    policy_reanalyzed: int = 0
    sampled_rows_per_batch: float = 0.0
    value_eval_seconds: float = 0.0
    search_seconds: float = 0.0
    policy_reanalyze_ratio_applied: float = 0.0
    stale_policy_fallbacks: int = 0
    policy_cache_hits: int = 0
    policy_cache_misses: int = 0
    mcts_root_cache_hits: int = 0
    mcts_root_cache_misses: int = 0
    bootstrap_horizon_counts: dict[int, int] = field(default_factory=dict)
    bootstrap_source_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "replay": str(self.replay_path),
            "checkpoint": str(self.checkpoint_path),
            "output": str(self.output_path),
            "transitions": self.transitions,
            "model_version": self.model_version,
            "bootstrap_td_steps": self.bootstrap_td_steps,
            "gamma": self.gamma,
            "search_reanalyzed": self.search_reanalyzed,
            "reanalyze_mode": self.reanalyze_mode,
            "value_bootstrap_source": self.value_bootstrap_source,
            "dynamic_horizon_enabled": self.dynamic_horizon_enabled,
            "sampled_batches": self.sampled_batches,
            "sampled_rows": self.sampled_rows,
            "policy_reanalyzed": self.policy_reanalyzed,
            "sampled_rows_per_batch": self.sampled_rows_per_batch,
            "value_eval_seconds": self.value_eval_seconds,
            "search_seconds": self.search_seconds,
            "policy_reanalyze_ratio_applied": self.policy_reanalyze_ratio_applied,
            "stale_policy_fallbacks": self.stale_policy_fallbacks,
            "policy_cache_hits": self.policy_cache_hits,
            "policy_cache_misses": self.policy_cache_misses,
            "mcts_root_cache_hits": self.mcts_root_cache_hits,
            "mcts_root_cache_misses": self.mcts_root_cache_misses,
            "bootstrap_horizon_counts": {
                str(horizon): count
                for horizon, count in sorted(self.bootstrap_horizon_counts.items())
            },
            "bootstrap_source_counts": dict(sorted(self.bootstrap_source_counts.items())),
        }


__all__ = ["ReanalyzeSummary"]
