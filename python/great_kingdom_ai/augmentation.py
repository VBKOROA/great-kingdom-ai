"""Board symmetry augmentation for self-play samples."""

from __future__ import annotations

import random
from collections.abc import Iterable
from typing import Literal

import numpy as np

from great_kingdom_ai.features import ACTION_SPACE, BOARD_SIZE
from great_kingdom_ai.replay_buffer import FEATURE_SHAPE, ReplaySample

Symmetry = Literal[
    "identity",
    "rot90",
    "rot180",
    "rot270",
    "flip_horizontal",
    "flip_vertical",
    "transpose",
    "anti_transpose",
]

ALL_SYMMETRIES: tuple[Symmetry, ...] = (
    "identity",
    "rot90",
    "rot180",
    "rot270",
    "flip_horizontal",
    "flip_vertical",
    "transpose",
    "anti_transpose",
)


def augment_sample(sample: ReplaySample, symmetry: Symmetry) -> ReplaySample:
    """Apply one board symmetry to feature planes and policy indexes together."""
    features = np.asarray(sample.features, dtype=np.float32)
    policy = np.asarray(sample.policy, dtype=np.float32)
    if features.shape != FEATURE_SHAPE:
        raise ValueError(f"expected feature shape {FEATURE_SHAPE}, got {features.shape}")
    if policy.shape != (ACTION_SPACE,):
        raise ValueError(f"expected policy shape {(ACTION_SPACE,)}, got {policy.shape}")

    board_policy = policy[: BOARD_SIZE * BOARD_SIZE].reshape(BOARD_SIZE, BOARD_SIZE)
    transformed_policy = np.empty(ACTION_SPACE, dtype=np.float32)
    transformed_policy[: BOARD_SIZE * BOARD_SIZE] = _transform_spatial(
        board_policy,
        symmetry,
    ).reshape(BOARD_SIZE * BOARD_SIZE)
    transformed_policy[-1] = policy[-1]
    transformed_root_logits = (
        None
        if sample.root_policy_logits is None
        else _transform_policy_like_array(
            np.asarray(sample.root_policy_logits, dtype=np.float32),
            symmetry,
            "root_policy_logits",
        )
    )

    return ReplaySample(
        features=_transform_spatial(features, symmetry).copy(),
        policy=transformed_policy,
        value=sample.value,
        root_policy_logits=transformed_root_logits,
        sample_weight=sample.sample_weight,
    )


def augment_all_symmetries(
    sample: ReplaySample,
    *,
    symmetries: Iterable[Symmetry] = ALL_SYMMETRIES,
) -> list[ReplaySample]:
    return [augment_sample(sample, symmetry) for symmetry in symmetries]


def augment_samples_randomly(
    samples: Iterable[ReplaySample],
    rng: random.Random,
    *,
    symmetries: Iterable[Symmetry] = ALL_SYMMETRIES,
) -> list[ReplaySample]:
    choices = tuple(symmetries)
    if not choices:
        raise ValueError("symmetries must contain at least one transform")
    return [augment_sample(sample, rng.choice(choices)) for sample in samples]


def augment_policy_training_arrays_randomly(
    features: np.ndarray,
    policies: np.ndarray,
    rng: random.Random,
    *,
    symmetries: Iterable[Symmetry] = ALL_SYMMETRIES,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply random board symmetries to feature and policy training arrays."""
    choices = tuple(symmetries)
    if not choices:
        raise ValueError("symmetries must contain at least one transform")
    if features.ndim != 4 or features.shape[1:] != FEATURE_SHAPE:
        raise ValueError(f"expected features shape [N, {FEATURE_SHAPE}], got {features.shape}")
    if policies.shape != (features.shape[0], ACTION_SPACE):
        raise ValueError(
            f"expected policies shape {(features.shape[0], ACTION_SPACE)}, got {policies.shape}"
        )

    transformed_features = np.empty_like(features, dtype=np.float32)
    transformed_policies = np.empty_like(policies, dtype=np.float32)
    selected = [rng.choice(choices) for _ in range(features.shape[0])]
    for symmetry in choices:
        indexes = [index for index, candidate in enumerate(selected) if candidate == symmetry]
        if not indexes:
            continue
        index_array = np.asarray(indexes, dtype=np.int64)
        transformed_features[index_array] = _transform_spatial(
            features[index_array],
            symmetry,
        )
        board_policy = policies[index_array, : BOARD_SIZE * BOARD_SIZE].reshape(
            len(indexes),
            BOARD_SIZE,
            BOARD_SIZE,
        )
        transformed_policies[index_array, : BOARD_SIZE * BOARD_SIZE] = _transform_spatial(
            board_policy,
            symmetry,
        ).reshape(len(indexes), BOARD_SIZE * BOARD_SIZE)
        transformed_policies[index_array, -1] = policies[index_array, -1]
    return transformed_features, transformed_policies


def _transform_spatial(array: np.ndarray, symmetry: Symmetry) -> np.ndarray:
    if symmetry == "identity":
        return array
    if symmetry == "rot90":
        return np.rot90(array, k=1, axes=(-2, -1))
    if symmetry == "rot180":
        return np.rot90(array, k=2, axes=(-2, -1))
    if symmetry == "rot270":
        return np.rot90(array, k=3, axes=(-2, -1))
    if symmetry == "flip_horizontal":
        return np.flip(array, axis=-1)
    if symmetry == "flip_vertical":
        return np.flip(array, axis=-2)
    if symmetry == "transpose":
        return np.swapaxes(array, -2, -1)
    if symmetry == "anti_transpose":
        return np.flip(np.swapaxes(array, -2, -1), axis=(-2, -1))
    raise ValueError(f"unknown symmetry: {symmetry}")


def _transform_policy_like_array(
    values: np.ndarray,
    symmetry: Symmetry,
    label: str,
) -> np.ndarray:
    if values.shape != (ACTION_SPACE,):
        raise ValueError(f"expected {label} shape {(ACTION_SPACE,)}, got {values.shape}")
    board_values = values[: BOARD_SIZE * BOARD_SIZE].reshape(BOARD_SIZE, BOARD_SIZE)
    transformed = np.empty(ACTION_SPACE, dtype=np.float32)
    transformed[: BOARD_SIZE * BOARD_SIZE] = _transform_spatial(
        board_values,
        symmetry,
    ).reshape(BOARD_SIZE * BOARD_SIZE)
    transformed[-1] = values[-1]
    return transformed
