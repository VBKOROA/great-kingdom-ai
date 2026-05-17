"""Training target snapshot storage and sampling."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np

from great_kingdom_ai.features import ACTION_SPACE
from great_kingdom_ai.priority_sampling import (
    PrioritySamplingConfig,
    legal_masks_from_features,
    sample_priority_indexes,
)
from great_kingdom_ai.reanalyze_sampling import sample_indexes
from great_kingdom_ai.replay import FEATURE_SHAPE
from great_kingdom_ai.replay.sample import ReplaySample

SNAPSHOT_FORMAT = "reanalyze-target-v1"


@dataclass(frozen=True)
class ReanalyzeTargetBatch:
    features: np.ndarray
    policies: np.ndarray
    values: np.ndarray
    sample_weights: np.ndarray
    legal_masks: np.ndarray


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
        indexes = sample_indexes(len(self), batch_size, rng)
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
            indexes = sample_indexes(
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
        from great_kingdom_ai import reanalyze as reanalyze_api

        scores = reanalyze_api.priority_scores(
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


def is_reanalyze_target_snapshot(path: str | Path) -> bool:
    try:
        with np.load(Path(path)) as data:
            return "snapshot_format" in data and str(data["snapshot_format"]) == SNAPSHOT_FORMAT
    except (OSError, ValueError, KeyError):
        return False


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


__all__ = [
    "SNAPSHOT_FORMAT",
    "ReanalyzeTargetBatch",
    "ReanalyzeTargetSnapshot",
    "is_reanalyze_target_snapshot",
]
