"""Priority and target-age aware replay sampling helpers."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np

from great_kingdom_ai.features import (
    BOARD_CELLS,
    LEGAL_PLACE_FEATURE_CHANNEL,
    PASS_ACTION,
)


@dataclass(frozen=True)
class PrioritySamplingConfig:
    enabled: bool = False
    alpha: float = 0.6
    beta: float = 0.4
    value_error_weight: float = 1.0
    policy_kl_weight: float = 1.0
    target_age_weight: float = 0.25
    search_reanalyzed_boost: float = 1.0
    max_priority: float | None = 64.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.alpha) or self.alpha < 0.0:
            raise ValueError("priority alpha must be finite and non-negative")
        if not math.isfinite(self.beta) or not 0.0 <= self.beta <= 1.0:
            raise ValueError("priority beta must be finite and in [0, 1]")
        weights = (
            ("value_error_weight", self.value_error_weight),
            ("policy_kl_weight", self.policy_kl_weight),
            ("target_age_weight", self.target_age_weight),
        )
        for label, value in weights:
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{label} must be finite and non-negative")
        if not math.isfinite(self.search_reanalyzed_boost) or self.search_reanalyzed_boost < 1.0:
            raise ValueError("search_reanalyzed_boost must be finite and at least 1")
        if self.max_priority is not None:
            if not math.isfinite(self.max_priority) or self.max_priority <= 1.0:
                raise ValueError("max_priority must be greater than 1")


@dataclass(frozen=True)
class PrioritySampleResult:
    indexes: list[int]
    importance_weights: np.ndarray


def priority_scores(
    *,
    values: np.ndarray,
    value_predictions: np.ndarray | None,
    policies: np.ndarray | None,
    policy_logits: np.ndarray | None,
    legal_masks: np.ndarray | None,
    target_ages: np.ndarray | None,
    config: PrioritySamplingConfig,
    search_reanalyzed: np.ndarray | None = None,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    scores = np.ones(values.shape, dtype=np.float32)

    if config.value_error_weight > 0.0 and value_predictions is not None:
        predictions = np.asarray(value_predictions, dtype=np.float32)
        _validate_row_vector("value_predictions", predictions, values.shape)
        scores += np.float32(config.value_error_weight) * np.abs(values - predictions)

    if config.policy_kl_weight > 0.0 and policies is not None and policy_logits is not None:
        policy_kl = policy_kl_scores(
            np.asarray(policies, dtype=np.float32),
            np.asarray(policy_logits, dtype=np.float32),
            None if legal_masks is None else np.asarray(legal_masks, dtype=np.bool_),
        )
        _validate_row_vector("policy_kl", policy_kl, values.shape)
        scores += np.float32(config.policy_kl_weight) * policy_kl

    if config.target_age_weight > 0.0 and target_ages is not None:
        ages = np.asarray(target_ages, dtype=np.float32)
        _validate_row_vector("target_ages", ages, values.shape)
        max_age = float(ages.max(initial=0.0))
        if max_age > 0.0:
            scores += np.float32(config.target_age_weight) * (ages / np.float32(max_age))

    if config.search_reanalyzed_boost > 1.0 and search_reanalyzed is not None:
        refreshed_by_search = np.asarray(search_reanalyzed, dtype=np.bool_)
        _validate_row_vector("search_reanalyzed", refreshed_by_search, values.shape)
        scores = np.where(
            refreshed_by_search,
            scores * np.float32(config.search_reanalyzed_boost),
            scores,
        )

    if config.max_priority is not None:
        scores = np.minimum(scores, np.float32(config.max_priority))
    if not np.isfinite(scores).all() or np.any(scores <= 0.0):
        raise ValueError("priority scores must be finite and positive")
    return scores.astype(np.float32, copy=False)


def policy_kl_scores(
    policies: np.ndarray,
    policy_logits: np.ndarray,
    legal_masks: np.ndarray | None = None,
) -> np.ndarray:
    if policies.shape != policy_logits.shape:
        raise ValueError("policy logits shape must match policies shape")
    if policies.ndim != 2:
        raise ValueError("policies must be a 2D array")
    if legal_masks is not None and legal_masks.shape != policies.shape:
        raise ValueError("legal_masks shape must match policies shape")
    if np.any(policies < 0.0) or not np.allclose(policies.sum(axis=1), 1.0):
        raise ValueError("policy targets must be normalized and non-negative")

    logits = policy_logits.astype(np.float32, copy=True)
    if legal_masks is not None:
        logits = np.where(legal_masks, logits, -np.inf).astype(np.float32, copy=False)
    row_max = np.max(logits, axis=1, keepdims=True)
    if not np.isfinite(row_max).all():
        raise ValueError("each policy row must have at least one finite legal logit")
    logsumexp = row_max + np.log(np.exp(logits - row_max).sum(axis=1, keepdims=True))
    log_probs = logits - logsumexp
    positive = policies > 0.0
    target_log = np.log(np.clip(policies, 1e-45, 1.0))
    terms = np.zeros_like(policies, dtype=np.float32)
    terms[positive] = policies[positive] * (target_log[positive] - log_probs[positive])
    kl = terms.sum(axis=1)
    return np.asarray(np.maximum(kl, 0.0), dtype=np.float32)


def sample_priority_indexes(
    *,
    priorities: np.ndarray,
    batch_size: int,
    rng: random.Random,
    beta: float = 0.0,
    recent_fraction: float = 0.0,
    recent_window: int = 0,
) -> PrioritySampleResult:
    priority_array = np.asarray(priorities, dtype=np.float64)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if batch_size > priority_array.shape[0]:
        raise ValueError("batch_size exceeds priority array size")
    if not np.isfinite(priority_array).all() or np.any(priority_array <= 0.0):
        raise ValueError("priorities must be finite and positive")
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be in [0, 1]")

    probabilities = priority_array / priority_array.sum()
    if recent_fraction <= 0.0:
        indexes = _choice_without_replacement(
            np.arange(priority_array.shape[0], dtype=np.int64),
            probabilities,
            batch_size,
            rng,
        )
    else:
        indexes = _sample_recent_priority_indexes(
            probabilities,
            batch_size,
            rng,
            recent_fraction=recent_fraction,
            recent_window=recent_window,
        )

    selected_probabilities = probabilities[indexes]
    importance_weights = np.ones((len(indexes),), dtype=np.float32)
    if beta > 0.0:
        importance_weights = np.power(
            priority_array.shape[0] * selected_probabilities,
            -beta,
        ).astype(np.float32)
        importance_weights /= importance_weights.max(initial=1.0)
    return PrioritySampleResult(
        indexes=[int(index) for index in indexes.tolist()],
        importance_weights=importance_weights,
    )


def legal_masks_from_features(features: np.ndarray) -> np.ndarray:
    feature_array = np.asarray(features, dtype=np.float32)
    legal_place = feature_array[:, LEGAL_PLACE_FEATURE_CHANNEL].reshape(-1, BOARD_CELLS) > 0.5
    legal_mask = np.zeros((feature_array.shape[0], BOARD_CELLS + 1), dtype=np.bool_)
    legal_mask[:, :BOARD_CELLS] = legal_place
    legal_mask[:, PASS_ACTION] = True
    return legal_mask


def _sample_recent_priority_indexes(
    probabilities: np.ndarray,
    batch_size: int,
    rng: random.Random,
    *,
    recent_fraction: float,
    recent_window: int,
) -> np.ndarray:
    if not 0.0 <= recent_fraction <= 1.0:
        raise ValueError("recent_fraction must be in [0, 1]")
    if recent_window <= 0:
        raise ValueError("recent_window must be positive when recency sampling is enabled")

    size = probabilities.shape[0]
    recent_count = min(recent_window, size)
    old_count = size - recent_count
    target_recent = round(batch_size * recent_fraction)
    recent_take = min(target_recent, recent_count, batch_size)
    old_take = min(batch_size - recent_take, old_count)
    recent_take = min(batch_size - old_take, recent_count)
    old_take = batch_size - recent_take
    if old_take > old_count:
        old_take = old_count
        recent_take = batch_size - old_take
    if recent_take > recent_count:
        raise ValueError("not enough rows to satisfy recency-biased sample")

    chunks: list[np.ndarray] = []
    if recent_take > 0:
        recent_start = size - recent_count
        recent_candidates = np.arange(recent_start, size, dtype=np.int64)
        chunks.append(
            _choice_without_replacement(
                recent_candidates,
                probabilities[recent_candidates],
                recent_take,
                rng,
            )
        )
    if old_take > 0:
        old_candidates = np.arange(old_count, dtype=np.int64)
        chunks.append(
            _choice_without_replacement(
                old_candidates,
                probabilities[old_candidates],
                old_take,
                rng,
            )
        )
    indexes = np.concatenate(chunks) if chunks else np.empty((0,), dtype=np.int64)
    index_list = [int(index) for index in indexes.tolist()]
    rng.shuffle(index_list)
    return np.asarray(index_list, dtype=np.int64)


def _choice_without_replacement(
    candidates: np.ndarray,
    weights: np.ndarray,
    count: int,
    rng: random.Random,
) -> np.ndarray:
    if count > candidates.shape[0]:
        raise ValueError("cannot sample more candidates than available")
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()
    np_rng = np.random.default_rng(rng.randrange(0, 2**63))
    return np_rng.choice(candidates, size=count, replace=False, p=weights)


def _validate_row_vector(label: str, array: np.ndarray, shape: tuple[int, ...]) -> None:
    if array.shape != shape:
        raise ValueError(f"{label} shape must match values")


__all__ = [
    "PrioritySampleResult",
    "PrioritySamplingConfig",
    "legal_masks_from_features",
    "policy_kl_scores",
    "priority_scores",
    "sample_priority_indexes",
]
