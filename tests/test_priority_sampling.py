import random

import numpy as np
import pytest
from great_kingdom_ai.priority_sampling import (
    PrioritySamplingConfig,
    policy_kl_scores,
    priority_scores,
    sample_priority_indexes,
)


def test_priority_scores_combine_value_error_policy_kl_and_age() -> None:
    policies = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    policy_logits = np.asarray(
        [
            [4.0, -4.0],
            [4.0, -4.0],
        ],
        dtype=np.float32,
    )
    config = PrioritySamplingConfig(
        enabled=True,
        alpha=1.0,
        value_error_weight=2.0,
        policy_kl_weight=1.0,
        target_age_weight=0.5,
    )

    scores = priority_scores(
        values=np.asarray([1.0, 0.0], dtype=np.float32),
        value_predictions=np.asarray([0.0, 0.0], dtype=np.float32),
        policies=policies,
        policy_logits=policy_logits,
        legal_masks=np.ones_like(policies, dtype=np.bool_),
        target_ages=np.asarray([0, 4], dtype=np.int64),
        config=config,
    )

    assert scores[0] > 1.0
    assert scores[1] > scores[0]


def test_policy_kl_scores_use_masked_log_probs() -> None:
    policies = np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32)
    logits = np.asarray([[100.0, 0.0, 100.0]], dtype=np.float32)
    legal_masks = np.asarray([[False, True, False]], dtype=np.bool_)

    scores = policy_kl_scores(policies, logits, legal_masks)

    assert scores.tolist() == pytest.approx([0.0])


def test_priority_sampling_respects_recent_window() -> None:
    result = sample_priority_indexes(
        priorities=np.asarray([100.0, 100.0, 100.0, 1.0, 1.0], dtype=np.float32),
        batch_size=2,
        rng=random.Random(3),
        beta=0.0,
        recent_fraction=1.0,
        recent_window=2,
    )

    assert set(result.indexes) == {3, 4}
    assert result.importance_weights.tolist() == pytest.approx([1.0, 1.0])


def test_priority_sampling_returns_importance_weights() -> None:
    result = sample_priority_indexes(
        priorities=np.asarray([1.0, 1000.0], dtype=np.float32),
        batch_size=1,
        rng=random.Random(0),
        beta=0.4,
    )

    assert result.indexes == [1]
    assert result.importance_weights.shape == (1,)
    assert 0.0 < result.importance_weights[0] <= 1.0
