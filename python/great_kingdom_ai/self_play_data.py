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


def apply_root_dirichlet_noise(
    priors: Sequence[float],
    legal_mask: Sequence[bool],
    rng: random.Random,
    *,
    alpha: float = 0.3,
    epsilon: float = 0.25,
) -> np.ndarray:
    """Mix Dirichlet noise into root priors for self-play exploration only."""
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be between 0 and 1")

    prior_array = np.asarray(priors, dtype=np.float32)
    mask_array = np.asarray(legal_mask, dtype=np.bool_)
    if prior_array.shape != (ACTION_SPACE,):
        raise ValueError(f"expected {ACTION_SPACE} priors, got shape {prior_array.shape}")
    if mask_array.shape != (ACTION_SPACE,):
        raise ValueError(f"expected {ACTION_SPACE} legal mask, got shape {mask_array.shape}")
    if np.any(prior_array < 0.0) or not np.all(np.isfinite(prior_array)):
        raise ValueError("priors must be finite non-negative values")

    legal_indexes = np.flatnonzero(mask_array)
    if legal_indexes.size == 0:
        raise ValueError("legal mask must contain at least one legal action")

    legal_priors = prior_array[legal_indexes]
    legal_total = float(legal_priors.sum())
    if legal_total > 0.0:
        legal_priors = legal_priors / legal_total
    else:
        legal_priors = np.full(legal_indexes.size, 1.0 / legal_indexes.size, dtype=np.float32)

    noise_values = np.asarray(
        [rng.gammavariate(alpha, 1.0) for _ in range(legal_indexes.size)],
        dtype=np.float32,
    )
    noise_values /= float(noise_values.sum())

    mixed = (1.0 - epsilon) * legal_priors + epsilon * noise_values
    result = np.zeros(ACTION_SPACE, dtype=np.float32)
    result[legal_indexes] = mixed
    result /= float(result.sum())
    return result
