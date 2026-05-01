"""Self-play target helpers for AlphaZero-lite training data."""

from __future__ import annotations

import random
from collections.abc import Sequence

import numpy as np

from great_kingdom_ai.features import ACTION_SPACE


def policy_target_from_visit_counts(visit_counts: Sequence[int]) -> np.ndarray:
    """Normalize MCTS root visit counts into an 82-action policy target."""
    counts = np.asarray(visit_counts, dtype=np.float32)
    if counts.shape != (ACTION_SPACE,):
        raise ValueError(f"expected {ACTION_SPACE} visit counts, got shape {counts.shape}")
    if np.any(counts < 0):
        raise ValueError("visit counts must be non-negative")

    total = float(counts.sum())
    if total <= 0.0:
        raise ValueError("visit counts must contain at least one visit")
    return counts / total


def value_target_for_player(*, player: int, winner: int) -> float:
    """Return the terminal value from the sample player's perspective."""
    if player not in {1, 2}:
        raise ValueError(f"unknown player: {player}")
    if winner not in {1, 2}:
        raise ValueError(f"unknown winner: {winner}")
    return 1.0 if player == winner else -1.0


def select_action_from_visit_counts(
    visit_counts: Sequence[int],
    rng: random.Random,
    *,
    temperature: float,
) -> int:
    """Select a move from MCTS visits using sampling for hot turns and argmax when cold."""
    target = policy_target_from_visit_counts(visit_counts)
    if temperature < 0.0:
        raise ValueError("temperature must be non-negative")
    if temperature == 0.0:
        return int(np.argmax(target))

    adjusted = np.power(target, 1.0 / temperature, dtype=np.float32)
    adjusted_total = float(adjusted.sum())
    if adjusted_total <= 0.0:
        raise ValueError("temperature adjustment removed all probability mass")
    probabilities = adjusted / adjusted_total

    threshold = rng.random()
    cumulative = 0.0
    for action, probability in enumerate(probabilities):
        cumulative += float(probability)
        if threshold <= cumulative:
            return action
    return int(np.flatnonzero(probabilities)[-1])
