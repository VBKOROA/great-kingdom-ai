"""Network-only reanalyze target snapshots for trajectory replay."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import random
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, NoReturn, cast

import numpy as np

from great_kingdom_ai.evaluator import evaluate_feature_arrays_logits_values
from great_kingdom_ai.features import ACTION_SPACE
from great_kingdom_ai.priority_sampling import (
    PrioritySamplingConfig,
    legal_masks_from_features,
    priority_scores,
    sample_priority_indexes,
)
from great_kingdom_ai.replay import FEATURE_SHAPE, TrajectoryEpisode, TrajectoryReplayStore
from great_kingdom_ai.replay_buffer import ReplaySample
from great_kingdom_ai.search_reanalyze import (
    SearchReanalyzeConfig,
    refresh_policies_with_search,
    refresh_sampled_policies_with_search,
)
from great_kingdom_ai.self_play_data import value_target_for_player

SNAPSHOT_FORMAT = "reanalyze-target-v1"
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


@dataclass(frozen=True)
class ReanalyzeTargetBatch:
    features: np.ndarray
    policies: np.ndarray
    values: np.ndarray
    sample_weights: np.ndarray
    legal_masks: np.ndarray


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


@dataclass(frozen=True)
class ReanalyzeTargetSnapshot:
    """Training-ready target snapshot generated from trajectory replay."""

    features: np.ndarray
    policies: np.ndarray
    values: np.ndarray
    refreshed_values: np.ndarray
    sample_weights: np.ndarray
    episode_ids: np.ndarray
    timesteps: np.ndarray
    players: np.ndarray
    source_model_versions: np.ndarray
    created_iterations: np.ndarray
    target_ages: np.ndarray
    model_version: int
    bootstrap_td_steps: int
    gamma: float
    checkpoint_path: str = ""
    policy_logits: np.ndarray | None = None
    search_reanalyzed: np.ndarray | None = None
    _priority_score_cache: dict[PrioritySamplingConfig, np.ndarray] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    @property
    def capacity(self) -> int:
        return len(self)

    def __len__(self) -> int:
        return int(self.values.shape[0])

    def sample(self, batch_size: int, rng: random.Random) -> list[ReplaySample]:
        indexes = _sample_indexes(len(self), batch_size, rng)
        return [
            ReplaySample(
                features=self.features[index],
                policy=self.policies[index],
                value=float(self.values[index]),
                sample_weight=float(self.sample_weights[index]),
            )
            for index in indexes
        ]

    def sample_arrays(
        self,
        batch_size: int,
        rng: random.Random,
        *,
        recent_fraction: float = 0.0,
        recent_window: int = 0,
        priority_config: PrioritySamplingConfig | None = None,
    ) -> ReanalyzeTargetBatch:
        if priority_config is not None and priority_config.enabled:
            priorities = self._cached_priority_scores(priority_config)
            sampled = sample_priority_indexes(
                priorities=priorities ** np.float32(priority_config.alpha),
                batch_size=batch_size,
                rng=rng,
                beta=priority_config.beta,
                recent_fraction=recent_fraction,
                recent_window=recent_window,
            )
            indexes = sampled.indexes
            sample_weights = (
                self.sample_weights[indexes].astype(np.float32, copy=True)
                * sampled.importance_weights
            )
        else:
            indexes = _sample_indexes(
                len(self),
                batch_size,
                rng,
                recent_fraction=recent_fraction,
                recent_window=recent_window,
            )
            sample_weights = self.sample_weights[indexes].astype(np.float32, copy=True)
        features = self.features[indexes].astype(np.float32, copy=True)
        return ReanalyzeTargetBatch(
            features=np.ascontiguousarray(features, dtype=np.float32),
            policies=np.ascontiguousarray(
                self.policies[indexes].astype(np.float32, copy=True),
                dtype=np.float32,
            ),
            values=np.ascontiguousarray(
                self.values[indexes].astype(np.float32, copy=True),
                dtype=np.float32,
            ),
            sample_weights=np.ascontiguousarray(sample_weights, dtype=np.float32),
            legal_masks=legal_masks_from_features(features),
        )

    def priority_scores(self, config: PrioritySamplingConfig) -> np.ndarray:
        return self._cached_priority_scores(config).copy()

    def _cached_priority_scores(self, config: PrioritySamplingConfig) -> np.ndarray:
        cached = self._priority_score_cache.get(config)
        if cached is not None:
            return cached
        scores = priority_scores(
            values=self.values,
            value_predictions=self.refreshed_values,
            policies=self.policies,
            policy_logits=self.policy_logits,
            legal_masks=legal_masks_from_features(self.features),
            target_ages=self.target_ages,
            search_reanalyzed=self.search_reanalyzed,
            config=config,
        )
        self._priority_score_cache[config] = scores
        return scores

    def save(self, path: str | Path, *, compressed: bool = True) -> None:
        snapshot = _validated_snapshot(self)
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "snapshot_format": np.asarray(SNAPSHOT_FORMAT, dtype=np.str_),
            "features": snapshot.features,
            "policies": snapshot.policies,
            "values": snapshot.values,
            "refreshed_values": snapshot.refreshed_values,
            "sample_weights": snapshot.sample_weights,
            "episode_ids": snapshot.episode_ids,
            "timesteps": snapshot.timesteps,
            "players": snapshot.players,
            "source_model_versions": snapshot.source_model_versions,
            "created_iterations": snapshot.created_iterations,
            "target_ages": snapshot.target_ages,
            "model_version": np.asarray(snapshot.model_version, dtype=np.int64),
            "bootstrap_td_steps": np.asarray(snapshot.bootstrap_td_steps, dtype=np.int64),
            "gamma": np.asarray(snapshot.gamma, dtype=np.float32),
            "checkpoint_path": np.asarray(snapshot.checkpoint_path, dtype=np.str_),
        }
        if snapshot.policy_logits is not None:
            payload["policy_logits"] = snapshot.policy_logits
        if snapshot.search_reanalyzed is not None:
            payload["search_reanalyzed"] = snapshot.search_reanalyzed
        save = np.savez_compressed if compressed else np.savez
        save(destination, **cast(dict[str, Any], payload))

    @classmethod
    def load(cls, path: str | Path) -> ReanalyzeTargetSnapshot:
        with np.load(Path(path)) as data:
            if str(data["snapshot_format"]) != SNAPSHOT_FORMAT:
                raise ValueError("unsupported reanalyze target snapshot format")
            snapshot = cls(
                features=np.asarray(data["features"], dtype=np.float32),
                policies=np.asarray(data["policies"], dtype=np.float32),
                values=np.asarray(data["values"], dtype=np.float32),
                refreshed_values=np.asarray(data["refreshed_values"], dtype=np.float32),
                sample_weights=np.asarray(data["sample_weights"], dtype=np.float32),
                episode_ids=np.asarray(data["episode_ids"], dtype=np.int64),
                timesteps=np.asarray(data["timesteps"], dtype=np.int64),
                players=np.asarray(data["players"], dtype=np.int64),
                source_model_versions=np.asarray(data["source_model_versions"], dtype=np.int64),
                created_iterations=np.asarray(data["created_iterations"], dtype=np.int64),
                target_ages=np.asarray(data["target_ages"], dtype=np.int64),
                model_version=int(data["model_version"]),
                bootstrap_td_steps=int(data["bootstrap_td_steps"]),
                gamma=float(data["gamma"]),
                checkpoint_path=str(data["checkpoint_path"]),
                policy_logits=(
                    np.asarray(data["policy_logits"], dtype=np.float32)
                    if "policy_logits" in data
                    else None
                ),
                search_reanalyzed=(
                    np.asarray(data["search_reanalyzed"], dtype=np.bool_)
                    if "search_reanalyzed" in data
                    else None
                ),
            )
        return _validated_snapshot(snapshot)


def reanalyze_replay(
    *,
    replay_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    config: ReanalyzeConfig | None = None,
    progress_callback: ReanalyzeProgressCallback | None = None,
) -> ReanalyzeSummary:
    resolved_config = config if config is not None else ReanalyzeConfig()
    _report_progress(progress_callback, "load", 0, 1, f"replay={replay_path}")
    replay = TrajectoryReplayStore.load(replay_path)
    _report_progress(progress_callback, "load", 1, 1, f"transitions={len(replay)}")
    snapshot = build_reanalyze_snapshot_from_store(
        replay,
        checkpoint_path=checkpoint_path,
        config=resolved_config,
        progress_callback=progress_callback,
    )
    _report_progress(progress_callback, "save", 0, 1, f"output={output_path}")
    snapshot.save(output_path, compressed=resolved_config.compressed)
    _report_progress(progress_callback, "save", 1, 1, f"rows={len(snapshot)}")
    return ReanalyzeSummary(
        replay_path=Path(replay_path),
        checkpoint_path=Path(checkpoint_path),
        output_path=Path(output_path),
        transitions=len(snapshot),
        model_version=snapshot.model_version,
        bootstrap_td_steps=snapshot.bootstrap_td_steps,
        gamma=snapshot.gamma,
        search_reanalyzed=(
            0
            if snapshot.search_reanalyzed is None
            else int(np.count_nonzero(snapshot.search_reanalyzed))
        ),
    )


def reanalyze_replay_store(
    *,
    replay: TrajectoryReplayStore,
    checkpoint_path: str | Path,
    output_path: str | Path,
    replay_path: str | Path,
    config: ReanalyzeConfig | None = None,
    progress_callback: ReanalyzeProgressCallback | None = None,
) -> ReanalyzeSummary:
    resolved_config = config if config is not None else ReanalyzeConfig()
    snapshot = build_reanalyze_snapshot_from_store(
        replay,
        checkpoint_path=checkpoint_path,
        config=resolved_config,
        progress_callback=progress_callback,
    )
    _report_progress(progress_callback, "save", 0, 1, f"output={output_path}")
    snapshot.save(output_path, compressed=resolved_config.compressed)
    _report_progress(progress_callback, "save", 1, 1, f"rows={len(snapshot)}")
    return ReanalyzeSummary(
        replay_path=Path(replay_path),
        checkpoint_path=Path(checkpoint_path),
        output_path=Path(output_path),
        transitions=len(snapshot),
        model_version=snapshot.model_version,
        bootstrap_td_steps=snapshot.bootstrap_td_steps,
        gamma=snapshot.gamma,
        search_reanalyzed=(
            0
            if snapshot.search_reanalyzed is None
            else int(np.count_nonzero(snapshot.search_reanalyzed))
        ),
    )


def build_reanalyze_snapshot_from_store(
    replay: TrajectoryReplayStore,
    *,
    checkpoint_path: str | Path,
    config: ReanalyzeConfig,
    progress_callback: ReanalyzeProgressCallback | None = None,
) -> ReanalyzeTargetSnapshot:
    from great_kingdom_ai.train import load_checkpoint

    if len(replay) == 0:
        raise ValueError("trajectory replay must contain at least one transition")
    _report_progress(progress_callback, "checkpoint", 0, 1, f"loading {checkpoint_path}")
    checkpoint_device = "cpu" if config.onnx_model_path is not None else config.device
    state = load_checkpoint(checkpoint_path, device=checkpoint_device, prefer_ema=True)
    state.model.eval()
    eval_backend = "onnx" if config.onnx_model_path is not None else "pytorch"
    _report_progress(
        progress_callback,
        "checkpoint",
        1,
        1,
        f"device={config.device}, eval_backend={eval_backend}",
    )
    model_version = state.step if config.model_version is None else config.model_version

    features = np.ascontiguousarray(replay.features, dtype=np.float32)
    legal_masks = np.ascontiguousarray(replay.legal_masks, dtype=np.bool_)
    policies = np.ascontiguousarray(replay.policy_targets, dtype=np.float32)
    _report_progress(progress_callback, "arrays", 1, 1, f"rows={features.shape[0]}")
    onnx_evaluator = (
        None
        if config.onnx_model_path is None
        else _create_onnx_evaluator(
            config.onnx_model_path,
            device=config.onnx_device or config.device,
            max_batch_size=config.onnx_max_batch_size,
        )
    )
    if onnx_evaluator is None:
        policy_logits, refreshed_values = _evaluate_policy_logits_values(
            state.model,
            features,
            legal_masks,
            batch_size=config.batch_size,
            device=config.device,
            progress_callback=progress_callback,
        )
    else:
        policy_logits, refreshed_values = _evaluate_policy_logits_values_with_onnx(
            onnx_evaluator,
            features,
            batch_size=config.batch_size,
            device=config.onnx_device or config.device,
            progress_callback=progress_callback,
        )
    _report_progress(
        progress_callback,
        "bootstrap",
        0,
        1,
        f"td_steps={config.bootstrap_td_steps}, gamma={config.gamma:g}",
    )
    bootstrap_values = (
        refreshed_values
        if config.value_bootstrap_source == "value_head"
        else _mcts_root_bootstrap_values_from_store(
            replay,
            policy_logits=policy_logits,
            refreshed_values=refreshed_values,
            model=state.model,
            device=config.device,
            onnx_evaluator=onnx_evaluator,
            config=config.search,
        )
    )
    values = _bootstrap_targets_from_store(
        replay,
        bootstrap_values,
        td_steps=config.bootstrap_td_steps,
        gamma=config.gamma,
        model_version=model_version,
        dynamic_horizon_enabled=config.dynamic_horizon_enabled,
        dynamic_horizon_tau=config.dynamic_horizon_tau,
        dynamic_horizon_total_steps=config.dynamic_horizon_total_steps,
    )
    _report_progress(progress_callback, "bootstrap", 1, 1, f"rows={values.shape[0]}")
    source_model_versions = np.asarray(replay.model_versions, dtype=np.int64)
    target_ages = np.maximum(model_version - source_model_versions, 0).astype(np.int64)
    search_reanalyzed: np.ndarray | None = None
    if config.search.enabled:
        episodes = replay.episodes
        search_result = refresh_policies_with_search(
            episodes=episodes,
            policies=policies,
            values=values,
            policy_logits=policy_logits,
            refreshed_values=refreshed_values,
            target_ages=target_ages,
            model=state.model,
            device=config.device,
            onnx_evaluator=onnx_evaluator,
            config=config.search,
            progress_callback=progress_callback,
        )
        policies = search_result.policies
        search_reanalyzed = search_result.search_reanalyzed
    else:
        _report_progress(progress_callback, "search", 0, 0, "disabled")

    return _validated_snapshot(
        ReanalyzeTargetSnapshot(
            features=features,
            policies=policies,
            values=values,
            refreshed_values=refreshed_values,
            sample_weights=np.asarray(replay.sample_weights, dtype=np.float32),
            episode_ids=_transition_episode_values(replay, replay.episode_ids),
            timesteps=np.asarray(replay.timesteps, dtype=np.int64),
            players=np.asarray(replay.players, dtype=np.int64),
            source_model_versions=source_model_versions,
            created_iterations=np.asarray(replay.created_iterations, dtype=np.int64),
            target_ages=target_ages,
            model_version=model_version,
            bootstrap_td_steps=config.bootstrap_td_steps,
            gamma=config.gamma,
            checkpoint_path=str(checkpoint_path),
            policy_logits=policy_logits,
            search_reanalyzed=search_reanalyzed,
        )
    )


def build_reanalyze_snapshot(
    replay: TrajectoryReplayStore,
    *,
    checkpoint_path: str | Path,
    config: ReanalyzeConfig,
    progress_callback: ReanalyzeProgressCallback | None = None,
) -> ReanalyzeTargetSnapshot:
    return build_reanalyze_snapshot_from_store(
        replay,
        checkpoint_path=checkpoint_path,
        config=config,
        progress_callback=progress_callback,
    )


def is_reanalyze_target_snapshot(path: str | Path) -> bool:
    try:
        with np.load(Path(path)) as data:
            return "snapshot_format" in data and str(data["snapshot_format"]) == SNAPSHOT_FORMAT
    except (OSError, ValueError, KeyError):
        return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Refresh trajectory replay value targets with a checkpoint"
    )
    parser.add_argument("--replay", type=Path, required=True, help="Input trajectory replay .npz")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Model checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="Output target snapshot .npz")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--onnx-model",
        type=Path,
        default=None,
        help="Optional ONNX model for policy/value refresh evaluation",
    )
    parser.add_argument("--onnx-device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--onnx-max-batch-size", type=int, default=1024)
    parser.add_argument("--bootstrap-td-steps", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument(
        "--dynamic-horizon-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--dynamic-horizon-tau", type=float, default=0.3)
    parser.add_argument("--dynamic-horizon-total-steps", type=int, default=None)
    parser.add_argument(
        "--value-bootstrap-source",
        choices=["value_head", "mcts_root"],
        default="value_head",
    )
    parser.add_argument(
        "--model-version",
        type=int,
        default=None,
        help="Override snapshot model version; defaults to checkpoint step",
    )
    parser.add_argument("--no-compress", action="store_true", help="Write an uncompressed npz")
    parser.add_argument(
        "--search-reanalyze-fraction",
        type=float,
        default=0.0,
        help="Fraction of high-priority rows whose policy targets are refreshed with Rust search",
    )
    parser.add_argument(
        "--search-reanalyze-budget",
        type=int,
        default=None,
        help="Maximum number of rows to refresh with Rust search",
    )
    parser.add_argument("--search-reanalyze-simulations", type=int, default=32)
    parser.add_argument("--search-reanalyze-max-considered-actions", type=int, default=16)
    parser.add_argument("--search-reanalyze-leaf-batch-size", type=int, default=8)
    parser.add_argument("--search-reanalyze-root-batch-size", type=int, default=128)
    parser.add_argument("--search-reanalyze-seed", type=int, default=0)
    return parser


def main() -> NoReturn:
    args = build_parser().parse_args()
    config = ReanalyzeConfig(
        batch_size=args.batch_size,
        device=args.device,
        onnx_model_path=None if args.onnx_model is None else str(args.onnx_model),
        onnx_device=args.onnx_device,
        onnx_max_batch_size=args.onnx_max_batch_size,
        bootstrap_td_steps=args.bootstrap_td_steps,
        gamma=args.gamma,
        dynamic_horizon_enabled=args.dynamic_horizon_enabled,
        dynamic_horizon_tau=args.dynamic_horizon_tau,
        dynamic_horizon_total_steps=args.dynamic_horizon_total_steps,
        value_bootstrap_source=args.value_bootstrap_source,
        model_version=args.model_version,
        compressed=not args.no_compress,
        search=SearchReanalyzeConfig(
            fraction=args.search_reanalyze_fraction,
            budget=args.search_reanalyze_budget,
            simulations=args.search_reanalyze_simulations,
            max_considered_actions=args.search_reanalyze_max_considered_actions,
            leaf_batch_size=args.search_reanalyze_leaf_batch_size,
            root_batch_size=args.search_reanalyze_root_batch_size,
            seed=args.search_reanalyze_seed,
        ),
    )
    print(
        json.dumps(
            {
                "event": "reanalyze_config",
                "config": asdict(config),
                "replay": str(args.replay),
                "checkpoint": str(args.checkpoint),
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    summary = reanalyze_replay(
        replay_path=args.replay,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        config=config,
    )
    print(json.dumps({"event": "reanalyze_summary", **summary.to_dict()}, sort_keys=True))
    raise SystemExit(0)


def _evaluate_policy_logits_values(
    model: Any,
    features: np.ndarray,
    legal_masks: np.ndarray,
    *,
    batch_size: int,
    device: str,
    progress_callback: ReanalyzeProgressCallback | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    policy_logits: list[np.ndarray] = []
    values: list[np.ndarray] = []
    total_batches = math.ceil(features.shape[0] / batch_size)
    _report_progress(
        progress_callback,
        "eval",
        0,
        total_batches,
        f"rows={features.shape[0]}, batch_size={batch_size}, device={device}",
    )
    for start in range(0, features.shape[0], batch_size):
        end = min(start + batch_size, features.shape[0])
        batch_number = start // batch_size + 1
        evaluation = evaluate_feature_arrays_logits_values(
            model,
            features[start:end],
            legal_masks[start:end],
            device=device,
        )
        policy_logits.append(evaluation.policy_logits)
        values.append(evaluation.value)
        _report_progress(
            progress_callback,
            "eval",
            batch_number,
            total_batches,
            f"rows={start}:{end}",
        )
    return (
        np.concatenate(policy_logits, axis=0).astype(np.float32),
        np.concatenate(values, axis=0).astype(np.float32),
    )


def _create_onnx_evaluator(
    onnx_model_path: str,
    *,
    device: str,
    max_batch_size: int,
) -> Any:
    core = _import_core()
    return core.OnnxEvaluator(
        str(onnx_model_path),
        device=device,
        max_batch_size=max_batch_size,
    )


def _evaluate_policy_logits_values_with_onnx(
    evaluator: Any,
    features: np.ndarray,
    *,
    batch_size: int,
    device: str,
    progress_callback: ReanalyzeProgressCallback | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    core = _import_core()
    policy_logits: list[np.ndarray] = []
    values: list[np.ndarray] = []
    total_batches = math.ceil(features.shape[0] / batch_size)
    _report_progress(
        progress_callback,
        "eval",
        0,
        total_batches,
        (
            f"rows={features.shape[0]}, batch_size={batch_size}, "
            f"device={device}, backend=onnx"
        ),
    )
    for start in range(0, features.shape[0], batch_size):
        end = min(start + batch_size, features.shape[0])
        batch_number = start // batch_size + 1
        request = _eval_request_from_feature_array(core, features[start:end])
        batch_logits, batch_values = evaluator.evaluate(request)
        policy_logits.append(np.asarray(batch_logits, dtype=np.float32))
        values.append(np.asarray(batch_values, dtype=np.float32))
        _report_progress(
            progress_callback,
            "eval",
            batch_number,
            total_batches,
            f"rows={start}:{end}",
        )
    return (
        np.concatenate(policy_logits, axis=0).astype(np.float32),
        np.concatenate(values, axis=0).astype(np.float32),
    )


def _eval_request_from_feature_array(core: Any, features: np.ndarray) -> Any:
    rows = np.ascontiguousarray(features.reshape(features.shape[0], -1), dtype=np.float32)
    if hasattr(core.EvalRequest, "from_feature_plane_bytes"):
        return core.EvalRequest.from_feature_plane_bytes(rows.shape[0], rows.tobytes())
    return core.EvalRequest.from_feature_rows(rows.tolist())


def _import_core() -> Any:
    try:
        return importlib.import_module("great_kingdom_core")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "great_kingdom_core is not installed. Build it with maturin before ONNX reanalyze."
        ) from exc


def _bootstrap_targets_from_refreshed_values(
    episodes: Sequence[TrajectoryEpisode],
    refreshed_values: np.ndarray,
    *,
    td_steps: int,
    gamma: float,
    model_version: int = 0,
    dynamic_horizon_enabled: bool = False,
    dynamic_horizon_tau: float = 0.3,
    dynamic_horizon_total_steps: int | None = None,
) -> np.ndarray:
    targets = np.empty((refreshed_values.shape[0],), dtype=np.float32)
    row_offset = 0
    for episode in episodes:
        terminal_index = len(episode.transitions) - 1
        for index, transition in enumerate(episode.transitions):
            row = row_offset + index
            effective_td_steps = _effective_bootstrap_td_steps(
                td_steps=td_steps,
                model_version=model_version,
                created_iteration=transition.created_iteration,
                dynamic_horizon_enabled=dynamic_horizon_enabled,
                dynamic_horizon_tau=dynamic_horizon_tau,
                dynamic_horizon_total_steps=dynamic_horizon_total_steps,
            )
            target_index = index + effective_td_steps
            if (
                effective_td_steps == 0
                or transition.terminal
                or target_index >= terminal_index
            ):
                targets[row] = value_target_for_player(
                    player=transition.player,
                    winner=episode.winner,
                )
                continue

            bootstrap = float(refreshed_values[row_offset + target_index])
            bootstrap_transition = episode.transitions[target_index]
            if bootstrap_transition.player != transition.player:
                bootstrap = -bootstrap
            targets[row] = np.float32((gamma**effective_td_steps) * bootstrap)
        row_offset += len(episode.transitions)
    return targets


def _mcts_root_bootstrap_values_from_store(
    replay: TrajectoryReplayStore,
    *,
    policy_logits: np.ndarray,
    refreshed_values: np.ndarray,
    model: Any,
    device: str,
    onnx_evaluator: Any | None,
    config: SearchReanalyzeConfig,
) -> np.ndarray:
    search_result = refresh_sampled_policies_with_search(
        transitions=replay.transition_refs(list(range(len(replay)))),
        policies=np.ascontiguousarray(replay.policy_targets, dtype=np.float32),
        policy_logits=policy_logits,
        refreshed_values=refreshed_values,
        model=model,
        device=device,
        onnx_evaluator=onnx_evaluator,
        config=config,
    )
    if search_result.root_values is None or not np.isfinite(search_result.root_values).all():
        raise RuntimeError("MCTS root bootstrap requires root values from search results")
    return np.ascontiguousarray(search_result.root_values, dtype=np.float32)


def _bootstrap_targets_from_store(
    replay: TrajectoryReplayStore,
    refreshed_values: np.ndarray,
    *,
    td_steps: int,
    gamma: float,
    model_version: int = 0,
    dynamic_horizon_enabled: bool = False,
    dynamic_horizon_tau: float = 0.3,
    dynamic_horizon_total_steps: int | None = None,
) -> np.ndarray:
    targets = np.empty((refreshed_values.shape[0],), dtype=np.float32)
    for episode_index in range(replay.episode_count):
        start = int(replay.episode_offsets[episode_index])
        end = int(replay.episode_offsets[episode_index + 1])
        terminal_index = end - 1
        winner = int(replay.episode_winners[episode_index])
        for row in range(start, end):
            effective_td_steps = _effective_bootstrap_td_steps(
                td_steps=td_steps,
                model_version=model_version,
                created_iteration=int(replay.created_iterations[row]),
                dynamic_horizon_enabled=dynamic_horizon_enabled,
                dynamic_horizon_tau=dynamic_horizon_tau,
                dynamic_horizon_total_steps=dynamic_horizon_total_steps,
            )
            target_index = row + effective_td_steps
            if (
                effective_td_steps == 0
                or bool(replay.terminals[row])
                or target_index >= terminal_index
            ):
                targets[row] = value_target_for_player(
                    player=int(replay.players[row]),
                    winner=winner,
                )
                continue

            bootstrap = float(refreshed_values[target_index])
            if int(replay.players[target_index]) != int(replay.players[row]):
                bootstrap = -bootstrap
            targets[row] = np.float32((gamma**effective_td_steps) * bootstrap)
    return targets


def _effective_bootstrap_td_steps(
    *,
    td_steps: int,
    model_version: int,
    created_iteration: int,
    dynamic_horizon_enabled: bool,
    dynamic_horizon_tau: float,
    dynamic_horizon_total_steps: int | None,
) -> int:
    if td_steps <= 0 or not dynamic_horizon_enabled:
        return td_steps
    if dynamic_horizon_total_steps is None:
        raise ValueError("dynamic_horizon_total_steps is required")
    age = max(model_version - created_iteration, 0)
    shrink = math.floor(age / (dynamic_horizon_tau * dynamic_horizon_total_steps))
    return min(td_steps, max(1, td_steps - shrink))


def _transition_episode_values(
    replay: TrajectoryReplayStore,
    episode_values: np.ndarray,
) -> np.ndarray:
    counts = np.diff(replay.episode_offsets)
    return np.repeat(np.asarray(episode_values, dtype=np.int64), counts).astype(np.int64)


def _flatten_episodes(episodes: Sequence[TrajectoryEpisode]) -> list[Any]:
    return [transition for episode in episodes for transition in episode.transitions]


def _sample_indexes(
    size: int,
    batch_size: int,
    rng: random.Random,
    *,
    recent_fraction: float = 0.0,
    recent_window: int = 0,
) -> list[int]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if batch_size > size:
        raise ValueError("batch_size exceeds reanalyze target snapshot size")
    if recent_fraction <= 0.0:
        return rng.sample(range(size), batch_size)
    if recent_fraction > 1.0:
        raise ValueError("recent_fraction must be in [0, 1]")
    if recent_window <= 0:
        raise ValueError("recent_window must be positive when recency sampling is enabled")

    recent_count = min(recent_window, size)
    old_count = size - recent_count
    recent_take = min(round(batch_size * recent_fraction), recent_count, batch_size)
    old_take = min(batch_size - recent_take, old_count)
    recent_take = batch_size - old_take
    if recent_take > recent_count:
        raise ValueError("not enough rows to satisfy recency-biased sample")
    recent_start = size - recent_count
    indexes = [recent_start + index for index in rng.sample(range(recent_count), recent_take)]
    indexes.extend(rng.sample(range(old_count), old_take))
    rng.shuffle(indexes)
    return indexes


def _validated_snapshot(snapshot: ReanalyzeTargetSnapshot) -> ReanalyzeTargetSnapshot:
    features = np.asarray(snapshot.features, dtype=np.float32)
    policies = np.asarray(snapshot.policies, dtype=np.float32)
    values = np.asarray(snapshot.values, dtype=np.float32)
    refreshed_values = np.asarray(snapshot.refreshed_values, dtype=np.float32)
    sample_weights = np.asarray(snapshot.sample_weights, dtype=np.float32)
    episode_ids = np.asarray(snapshot.episode_ids, dtype=np.int64)
    timesteps = np.asarray(snapshot.timesteps, dtype=np.int64)
    players = np.asarray(snapshot.players, dtype=np.int64)
    source_model_versions = np.asarray(snapshot.source_model_versions, dtype=np.int64)
    created_iterations = np.asarray(snapshot.created_iterations, dtype=np.int64)
    target_ages = np.asarray(snapshot.target_ages, dtype=np.int64)
    policy_logits = (
        None
        if snapshot.policy_logits is None
        else np.asarray(snapshot.policy_logits, dtype=np.float32)
    )
    search_reanalyzed = (
        None
        if snapshot.search_reanalyzed is None
        else np.asarray(snapshot.search_reanalyzed, dtype=np.bool_)
    )
    row_count = values.shape[0]

    if features.shape != (row_count, *FEATURE_SHAPE):
        raise ValueError(f"expected features shape {(row_count, *FEATURE_SHAPE)}")
    if policies.shape != (row_count, ACTION_SPACE):
        raise ValueError(f"expected policies shape {(row_count, ACTION_SPACE)}")
    if policy_logits is not None and policy_logits.shape != (row_count, ACTION_SPACE):
        raise ValueError(f"expected policy_logits shape {(row_count, ACTION_SPACE)}")
    if search_reanalyzed is not None and search_reanalyzed.shape != (row_count,):
        raise ValueError("search_reanalyzed shape must match values")
    row_arrays: tuple[tuple[str, np.ndarray], ...] = (
        ("refreshed_values", refreshed_values),
        ("sample_weights", sample_weights),
        ("episode_ids", episode_ids),
        ("timesteps", timesteps),
        ("players", players),
        ("source_model_versions", source_model_versions),
        ("created_iterations", created_iterations),
        ("target_ages", target_ages),
    )
    for label, array in row_arrays:
        if array.shape != (row_count,):
            raise ValueError(f"{label} shape must match values")
    if not np.isfinite(features).all() or not np.isfinite(policies).all():
        raise ValueError("snapshot feature and policy arrays must be finite")
    if policy_logits is not None and not np.isfinite(policy_logits).all():
        raise ValueError("snapshot policy_logits must be finite")
    if not np.isfinite(values).all() or not np.isfinite(refreshed_values).all():
        raise ValueError("snapshot value arrays must be finite")
    if np.any(values < -1.0) or np.any(values > 1.0):
        raise ValueError("snapshot value targets must be in [-1, 1]")
    if np.any(refreshed_values < -1.0) or np.any(refreshed_values > 1.0):
        raise ValueError("refreshed values must be in [-1, 1]")
    if np.any(sample_weights <= 0.0) or not np.isfinite(sample_weights).all():
        raise ValueError("sample weights must be finite and positive")
    if np.any(players < 1) or np.any(players > 2):
        raise ValueError("players must be 1 or 2")
    if snapshot.model_version < 0:
        raise ValueError("model_version must be non-negative")
    if snapshot.bootstrap_td_steps < 0:
        raise ValueError("bootstrap_td_steps must be non-negative")
    if not math.isfinite(snapshot.gamma) or not 0.0 <= snapshot.gamma <= 1.0:
        raise ValueError("gamma must be finite and in [0, 1]")

    return ReanalyzeTargetSnapshot(
        features=features.copy(),
        policies=policies.copy(),
        values=values.copy(),
        refreshed_values=refreshed_values.copy(),
        sample_weights=sample_weights.copy(),
        episode_ids=episode_ids.copy(),
        timesteps=timesteps.copy(),
        players=players.copy(),
        source_model_versions=source_model_versions.copy(),
        created_iterations=created_iterations.copy(),
        target_ages=target_ages.copy(),
        model_version=int(snapshot.model_version),
        bootstrap_td_steps=int(snapshot.bootstrap_td_steps),
        gamma=float(snapshot.gamma),
        checkpoint_path=str(snapshot.checkpoint_path),
        policy_logits=None if policy_logits is None else policy_logits.copy(),
        search_reanalyzed=None if search_reanalyzed is None else search_reanalyzed.copy(),
    )


def _report_progress(
    progress_callback: ReanalyzeProgressCallback | None,
    stage: str,
    current: int,
    total: int,
    detail: str,
) -> None:
    if progress_callback is not None:
        progress_callback(stage, current, total, detail)


if __name__ == "__main__":
    main()


__all__ = [
    "ReanalyzeConfig",
    "ReanalyzeProgressCallback",
    "ReanalyzeSummary",
    "ReanalyzeTargetBatch",
    "ReanalyzeTargetSnapshot",
    "SearchReanalyzeConfig",
    "build_parser",
    "build_reanalyze_snapshot",
    "build_reanalyze_snapshot_from_store",
    "is_reanalyze_target_snapshot",
    "reanalyze_replay",
    "reanalyze_replay_store",
]
