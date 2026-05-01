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

    return ReplaySample(
        features=_transform_spatial(features, symmetry).copy(),
        policy=transformed_policy,
        value=sample.value,
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
