import random

import numpy as np
import pytest
from great_kingdom_ai.augmentation import (
    ALL_SYMMETRIES,
    augment_all_symmetries,
    augment_sample,
    augment_samples_randomly,
    augment_training_arrays_randomly,
)
from great_kingdom_ai.features import ACTION_SPACE, BOARD_SIZE, FEATURE_CHANNELS
from great_kingdom_ai.replay.sample import ReplaySample


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


def test_augment_training_arrays_transforms_legal_masks_with_policies() -> None:
    sample = make_sample(action=1 * BOARD_SIZE + 2)
    legal_masks = np.zeros((1, ACTION_SPACE), dtype=np.bool_)
    legal_masks[0, 1 * BOARD_SIZE + 2] = True
    legal_masks[0, 81] = True

    features, policies, transformed_legal_masks, _terminal, _actions = (
        augment_training_arrays_randomly(
            sample.features[np.newaxis],
            sample.policy[np.newaxis],
            legal_masks,
            random.Random(1),
            symmetries=("rot90",),
        )
    )

    assert features[0, 0, 6, 1] == 1.0
    assert policies[0, 6 * BOARD_SIZE + 1] == 0.75
    assert transformed_legal_masks is not None
    assert transformed_legal_masks[0, 6 * BOARD_SIZE + 1]
    assert transformed_legal_masks[0, 81]


def test_augment_sample_permutes_action_like_policy() -> None:
    for symmetry in ALL_SYMMETRIES:
        action = 1 * BOARD_SIZE + 2
        sample = make_sample(action=action)
        sample = ReplaySample(
            features=sample.features,
            policy=sample.policy,
            value=sample.value,
            action=action,
        )

        augmented = augment_sample(sample, symmetry)

        assert augmented.action is not None
        expected_action = int(np.argmax(augmented.policy[:81]))
        assert augmented.action == expected_action
        assert augmented.action != action or symmetry == "identity"


def test_augment_sample_keeps_pass_action_on_pass_index() -> None:
    features = np.zeros((FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    policy = np.zeros(ACTION_SPACE, dtype=np.float32)
    policy[81] = 1.0
    sample = ReplaySample(features=features, policy=policy, value=1.0, action=81)

    augmented = augment_sample(sample, "rot270")

    assert augmented.action == 81


def test_augment_training_arrays_permutes_actions_like_policies() -> None:
    actions = np.asarray([1 * BOARD_SIZE + 2, 81], dtype=np.int64)
    policies = np.zeros((2, ACTION_SPACE), dtype=np.float32)
    policies[0, actions[0]] = 1.0
    policies[1, actions[1]] = 1.0
    features = np.zeros((2, FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

    _features, transformed_policies, _masks, _boards, transformed_actions = (
        augment_training_arrays_randomly(
            features,
            policies,
            None,
            random.Random(1),
            symmetries=("rot90",),
            actions=actions,
        )
    )

    assert transformed_actions is not None
    assert transformed_actions[0] == int(np.argmax(transformed_policies[0, :81]))
    assert transformed_actions[0] == 6 * BOARD_SIZE + 1
    assert transformed_actions[1] == 81
