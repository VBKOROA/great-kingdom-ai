import random

import numpy as np
import pytest
from great_kingdom_ai.augmentation import (
    ALL_SYMMETRIES,
    augment_all_symmetries,
    augment_sample,
    augment_samples_randomly,
)
from great_kingdom_ai.features import ACTION_SPACE, BOARD_SIZE, FEATURE_CHANNELS
from great_kingdom_ai.replay_buffer import ReplaySample


def make_sample(action: int) -> ReplaySample:
    features = np.zeros((FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    features[0, 1, 2] = 1.0
    policy = np.zeros(ACTION_SPACE, dtype=np.float32)
    policy[action] = 0.75
    policy[81] = 0.25
    return ReplaySample(features=features, policy=policy, value=-1.0)


def test_augment_sample_transforms_features_and_policy_together() -> None:
    sample = make_sample(action=1 * BOARD_SIZE + 2)
    root_policy_logits = np.zeros(ACTION_SPACE, dtype=np.float32)
    root_policy_logits[1 * BOARD_SIZE + 2] = 7.0
    sample = ReplaySample(
        features=sample.features,
        policy=sample.policy,
        value=sample.value,
        root_policy_logits=root_policy_logits,
    )

    augmented = augment_sample(sample, "rot90")

    assert augmented.features[0, 6, 1] == 1.0
    assert augmented.policy[6 * BOARD_SIZE + 1] == 0.75
    assert augmented.root_policy_logits is not None
    assert augmented.root_policy_logits[6 * BOARD_SIZE + 1] == 7.0
    assert augmented.policy[81] == 0.25
    assert augmented.value == -1.0
    assert np.isclose(augmented.policy.sum(), 1.0)


def test_augment_sample_keeps_pass_policy_on_pass_index() -> None:
    features = np.zeros((FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    policy = np.zeros(ACTION_SPACE, dtype=np.float32)
    policy[81] = 1.0
    sample = ReplaySample(features=features, policy=policy, value=1.0)

    augmented = augment_sample(sample, "flip_horizontal")

    assert augmented.policy[81] == 1.0
    assert np.count_nonzero(augmented.policy[:81]) == 0


def test_augment_all_symmetries_returns_configured_transforms() -> None:
    sample = make_sample(action=10)

    augmented = augment_all_symmetries(sample)

    assert len(augmented) == len(ALL_SYMMETRIES)
    assert all(item.features.shape == sample.features.shape for item in augmented)


def test_augment_samples_randomly_applies_configured_symmetries() -> None:
    sample = make_sample(action=1 * BOARD_SIZE + 2)

    augmented = augment_samples_randomly(
        [sample],
        random.Random(1),
        symmetries=("rot90",),
    )

    assert len(augmented) == 1
    assert augmented[0].features[0, 6, 1] == 1.0
    assert augmented[0].policy[6 * BOARD_SIZE + 1] == 0.75


def test_augment_samples_randomly_rejects_empty_symmetry_set() -> None:
    with pytest.raises(ValueError, match="at least one"):
        augment_samples_randomly([make_sample(action=10)], random.Random(1), symmetries=())
