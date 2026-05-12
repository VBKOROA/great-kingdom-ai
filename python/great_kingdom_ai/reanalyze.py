"""Network-only reanalyze target snapshots for trajectory replay."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections.abc import Sequence
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
from great_kingdom_ai.replay_buffer import FEATURE_SHAPE, ReplaySample
from great_kingdom_ai.search_reanalyze import (
    SearchReanalyzeConfig,
    refresh_policies_with_search,
)
from great_kingdom_ai.self_play_data import value_target_for_player
from great_kingdom_ai.trajectory_replay import TrajectoryEpisode, TrajectoryReplayBuffer

SNAPSHOT_FORMAT = "reanalyze-target-v1"


@dataclass(frozen=True)
class ReanalyzeConfig:
    batch_size: int = 1024
    device: str = "cpu"
    bootstrap_td_steps: int = 0
    gamma: float = 1.0
    model_version: int | None = None
    compressed: bool = True
    search: SearchReanalyzeConfig = field(default_factory=SearchReanalyzeConfig)

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.bootstrap_td_steps < 0:
            raise ValueError("bootstrap_td_steps must be non-negative")
        if not math.isfinite(self.gamma) or not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be finite and in [0, 1]")
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
            priorities = self.priority_scores(priority_config)
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
        return priority_scores(
            values=self.values,
            value_predictions=self.refreshed_values,
            policies=self.policies,
            policy_logits=self.policy_logits,
            legal_masks=legal_masks_from_features(self.features),
            target_ages=self.target_ages,
            config=config,
        )

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
) -> ReanalyzeSummary:
    resolved_config = config if config is not None else ReanalyzeConfig()
    replay = TrajectoryReplayBuffer.load(replay_path)
    snapshot = build_reanalyze_snapshot(
        replay,
        checkpoint_path=checkpoint_path,
        config=resolved_config,
    )
    snapshot.save(output_path, compressed=resolved_config.compressed)
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


def build_reanalyze_snapshot(
    replay: TrajectoryReplayBuffer,
    *,
    checkpoint_path: str | Path,
    config: ReanalyzeConfig,
) -> ReanalyzeTargetSnapshot:
    from great_kingdom_ai.train import load_checkpoint

    state = load_checkpoint(checkpoint_path, device=config.device)
    state.model.eval()
    model_version = state.step if config.model_version is None else config.model_version
    episodes = replay.episodes
    rows = _flatten_episodes(episodes)
    if not rows:
        raise ValueError("trajectory replay must contain at least one transition")

    features = np.stack([row.features for row in rows], axis=0).astype(np.float32)
    legal_masks = np.stack([row.legal_mask for row in rows], axis=0).astype(np.bool_)
    policies = np.stack([row.policy_target for row in rows], axis=0).astype(np.float32)
    policy_logits, refreshed_values = _evaluate_policy_logits_values(
        state.model,
        features,
        legal_masks,
        batch_size=config.batch_size,
        device=config.device,
    )
    values = _bootstrap_targets_from_refreshed_values(
        episodes,
        refreshed_values,
        td_steps=config.bootstrap_td_steps,
        gamma=config.gamma,
    )
    source_model_versions = np.asarray([row.model_version for row in rows], dtype=np.int64)
    target_ages = np.maximum(model_version - source_model_versions, 0).astype(np.int64)
    search_reanalyzed: np.ndarray | None = None
    if config.search.enabled:
        search_result = refresh_policies_with_search(
            episodes=episodes,
            policies=policies,
            values=values,
            policy_logits=policy_logits,
            refreshed_values=refreshed_values,
            target_ages=target_ages,
            model=state.model,
            device=config.device,
            config=config.search,
        )
        policies = search_result.policies
        search_reanalyzed = search_result.search_reanalyzed
    return _validated_snapshot(
        ReanalyzeTargetSnapshot(
            features=features,
            policies=policies,
            values=values,
            refreshed_values=refreshed_values,
            sample_weights=np.asarray([row.sample_weight for row in rows], dtype=np.float32),
            episode_ids=np.asarray([row.episode_id for row in rows], dtype=np.int64),
            timesteps=np.asarray([row.timestep for row in rows], dtype=np.int64),
            players=np.asarray([row.player for row in rows], dtype=np.int64),
            source_model_versions=source_model_versions,
            created_iterations=np.asarray([row.created_iteration for row in rows], dtype=np.int64),
            target_ages=target_ages,
            model_version=model_version,
            bootstrap_td_steps=config.bootstrap_td_steps,
            gamma=config.gamma,
            checkpoint_path=str(checkpoint_path),
            policy_logits=policy_logits,
            search_reanalyzed=search_reanalyzed,
        )
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
    parser.add_argument("--bootstrap-td-steps", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=1.0)
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
    parser.add_argument("--search-reanalyze-seed", type=int, default=0)
    return parser


def main() -> NoReturn:
    args = build_parser().parse_args()
    config = ReanalyzeConfig(
        batch_size=args.batch_size,
        device=args.device,
        bootstrap_td_steps=args.bootstrap_td_steps,
        gamma=args.gamma,
        model_version=args.model_version,
        compressed=not args.no_compress,
        search=SearchReanalyzeConfig(
            fraction=args.search_reanalyze_fraction,
            budget=args.search_reanalyze_budget,
            simulations=args.search_reanalyze_simulations,
            max_considered_actions=args.search_reanalyze_max_considered_actions,
            leaf_batch_size=args.search_reanalyze_leaf_batch_size,
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
) -> tuple[np.ndarray, np.ndarray]:
    policy_logits: list[np.ndarray] = []
    values: list[np.ndarray] = []
    for start in range(0, features.shape[0], batch_size):
        end = min(start + batch_size, features.shape[0])
        evaluation = evaluate_feature_arrays_logits_values(
            model,
            features[start:end],
            legal_masks[start:end],
            device=device,
        )
        policy_logits.append(evaluation.policy_logits)
        values.append(evaluation.value)
    return (
        np.concatenate(policy_logits, axis=0).astype(np.float32),
        np.concatenate(values, axis=0).astype(np.float32),
    )


def _bootstrap_targets_from_refreshed_values(
    episodes: Sequence[TrajectoryEpisode],
    refreshed_values: np.ndarray,
    *,
    td_steps: int,
    gamma: float,
) -> np.ndarray:
    targets = np.empty((refreshed_values.shape[0],), dtype=np.float32)
    row_offset = 0
    for episode in episodes:
        terminal_index = len(episode.transitions) - 1
        for index, transition in enumerate(episode.transitions):
            row = row_offset + index
            target_index = index + td_steps
            if td_steps == 0 or transition.terminal or target_index >= terminal_index:
                targets[row] = value_target_for_player(
                    player=transition.player,
                    winner=episode.winner,
                )
                continue

            bootstrap = float(refreshed_values[row_offset + target_index])
            bootstrap_transition = episode.transitions[target_index]
            if bootstrap_transition.player != transition.player:
                bootstrap = -bootstrap
            targets[row] = np.float32((gamma**td_steps) * bootstrap)
        row_offset += len(episode.transitions)
    return targets


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


if __name__ == "__main__":
    main()


__all__ = [
    "ReanalyzeConfig",
    "ReanalyzeSummary",
    "ReanalyzeTargetBatch",
    "ReanalyzeTargetSnapshot",
    "SearchReanalyzeConfig",
    "build_parser",
    "build_reanalyze_snapshot",
    "is_reanalyze_target_snapshot",
    "reanalyze_replay",
]
