"""Replay sample objects used at training and self-play boundaries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from great_kingdom_ai.features import ACTION_SPACE
from great_kingdom_ai.replay.schema import FEATURE_SHAPE


@dataclass(frozen=True)
class ReplaySample:
    features: np.ndarray
    policy: np.ndarray
    value: float
    root_policy_logits: np.ndarray | None = None
    sample_weight: float = 1.0


@dataclass(frozen=True)
class ReplayArrayBatch:
    features: np.ndarray
    policies: np.ndarray
    values: np.ndarray
    sample_weights: np.ndarray
    legal_masks: np.ndarray


def validate_replay_sample(sample: ReplaySample) -> ReplaySample:
    features = np.asarray(sample.features, dtype=np.float32)
    policy = np.asarray(sample.policy, dtype=np.float32)
    value = float(sample.value)
    sample_weight = float(sample.sample_weight)

    if features.shape != FEATURE_SHAPE:
        raise ValueError(f"expected feature shape {FEATURE_SHAPE}, got {features.shape}")
    if policy.shape != (ACTION_SPACE,):
        raise ValueError(f"expected policy shape {(ACTION_SPACE,)}, got {policy.shape}")
    root_policy_logits = (
        None
        if sample.root_policy_logits is None
        else np.asarray(sample.root_policy_logits, dtype=np.float32)
    )
    if root_policy_logits is not None:
        if root_policy_logits.shape != (ACTION_SPACE,):
            raise ValueError(
                f"expected root_policy_logits shape {(ACTION_SPACE,)}, "
                f"got {root_policy_logits.shape}"
            )
        if not np.isfinite(root_policy_logits).all():
            raise ValueError("root_policy_logits must be finite")
    if not np.isclose(policy.sum(), 1.0):
        raise ValueError("policy target must sum to 1")
    if np.any(policy < 0.0):
        raise ValueError("policy target must be non-negative")
    if value < -1.0 or value > 1.0:
        raise ValueError("value target must be in [-1, 1]")
    if not np.isfinite(sample_weight) or sample_weight <= 0.0:
        raise ValueError("sample_weight must be finite and positive")

    return ReplaySample(
        features=features.copy(),
        policy=policy.copy(),
        value=value,
        root_policy_logits=(
            None if root_policy_logits is None else root_policy_logits.copy()
        ),
        sample_weight=sample_weight,
    )


__all__ = ["ReplayArrayBatch", "ReplaySample", "validate_replay_sample"]
