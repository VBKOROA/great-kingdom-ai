"""Tensor conversion helpers for Rust-provided Great Kingdom features."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

if TYPE_CHECKING:
    import torch

BOARD_SIZE = 9
BOARD_CELLS = BOARD_SIZE * BOARD_SIZE
ACTION_SPACE = BOARD_CELLS + 1
PASS_ACTION = BOARD_CELLS
FEATURE_CHANNELS = 11
LEGAL_PLACE_FEATURE_CHANNEL = 4


class FeatureState(Protocol):
    def feature_planes(self) -> list[float]: ...

    def legal_mask(self) -> list[bool]: ...


def states_to_feature_tensor(
    states: Sequence[FeatureState],
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Convert Rust feature planes to a float tensor shaped [batch, channels, 9, 9]."""
    torch = _import_torch()
    if not states:
        return torch.empty((0, FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=torch.float32)

    features = np.asarray([state.feature_planes() for state in states], dtype=np.float32)
    expected = FEATURE_CHANNELS * BOARD_CELLS
    if features.shape != (len(states), expected):
        raise ValueError(f"expected feature shape {(len(states), expected)}, got {features.shape}")

    tensor = torch.from_numpy(
        features.reshape(len(states), FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    )
    return tensor.to(device=device) if device is not None else tensor


def states_to_legal_mask_tensor(
    states: Sequence[FeatureState],
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Convert Rust legal masks to a bool tensor shaped [batch, 82]."""
    torch = _import_torch()
    if not states:
        return torch.empty((0, ACTION_SPACE), dtype=torch.bool)

    masks = np.asarray([state.legal_mask() for state in states], dtype=np.bool_)
    if masks.shape != (len(states), ACTION_SPACE):
        raise ValueError(
            f"expected legal mask shape {(len(states), ACTION_SPACE)}, got {masks.shape}"
        )

    tensor = torch.from_numpy(masks)
    return tensor.to(device=device) if device is not None else tensor


def _import_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required for feature tensor conversion") from exc
    return torch
