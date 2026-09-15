"""Replay sample objects used at training and self-play boundaries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from great_kingdom_ai.features import ACTION_SPACE
from great_kingdom_ai.replay.schema import FEATURE_SHAPE
from great_kingdom_ai.replay.terminal_board import (
    TERMINAL_BOARD_CLASSES,
    TERMINAL_BOARD_SHAPE,
)


@dataclass(frozen=True)
class ReplaySample:
    features: np.ndarray
    policy: np.ndarray
    value: float
    root_policy_logits: np.ndarray | None = None
    sample_weight: float = 1.0
    terminal_board_target: np.ndarray | None = None
    action: int | None = None


@dataclass(frozen=True)
class ReplayArrayBatch:
    features: np.ndarray
    policies: np.ndarray
    values: np.ndarray
    sample_weights: np.ndarray
    legal_masks: np.ndarray
    actions: np.ndarray | None = None


def validate_replay_sample(sample: ReplaySample) -> ReplaySample:
    features = np.asarray(sample.features, dtype=np.float32)
    policy = np.asarray(sample.policy, dtype=np.float32)
    value = float(sample.value)
    sample_weight = float(sample.sample_weight)
    action = None if sample.action is None else int(sample.action)

    if features.shape != FEATURE_SHAPE:
        raise ValueError(f"expected feature shape {FEATURE_SHAPE}, got {features.shape}")
    if policy.shape != (ACTION_SPACE,):
        raise ValueError(f"expected policy shape {(ACTION_SPACE,)}, got {policy.shape}")
    root_policy_logits = (
        None
        if sample.root_policy_logits is None
        else np.asarray(sample.root_policy_logits, dtype=np.float32)
    )
    terminal_board_target = (
        None
        if sample.terminal_board_target is None
        else np.asarray(sample.terminal_board_target)
    )
    if terminal_board_target is not None:
        if terminal_board_target.shape != TERMINAL_BOARD_SHAPE:
            raise ValueError(
                "expected terminal_board_target shape "
                f"{TERMINAL_BOARD_SHAPE}, got {terminal_board_target.shape}"
            )
        if not np.issubdtype(terminal_board_target.dtype, np.integer):
            raise ValueError("terminal_board_target must be an integer array")
        if np.any(terminal_board_target < 0) or np.any(
            terminal_board_target >= TERMINAL_BOARD_CLASSES
        ):
            raise ValueError(
                f"terminal_board_target classes must be in [0, {TERMINAL_BOARD_CLASSES})"
            )
        terminal_board_target = terminal_board_target.astype(np.int64, copy=True)
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
    if action is not None and (action < 0 or action >= ACTION_SPACE):
        raise ValueError(f"action must be in [0, {ACTION_SPACE})")

    return ReplaySample(
        features=features.copy(),
        policy=policy.copy(),
        value=value,
        root_policy_logits=(
            None if root_policy_logits is None else root_policy_logits.copy()
        ),
        sample_weight=sample_weight,
        terminal_board_target=terminal_board_target,
        action=action,
    )


__all__ = ["ReplayArrayBatch", "ReplaySample", "validate_replay_sample"]
