from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass

import pytest

_torch_spec = importlib.util.find_spec("torch")
pytestmark = pytest.mark.skipif(
    _torch_spec is None,
    reason="torch is not installed",
)
torch = importlib.import_module("torch") if _torch_spec is not None else None

from great_kingdom_ai.features import (  # noqa: E402
    ACTION_SPACE,
    BOARD_SIZE,
    FEATURE_CHANNELS,
    states_to_feature_tensor,
    states_to_legal_mask_tensor,
)


@dataclass(frozen=True)
class FakeFeatureState:
    features: list[float]
    mask: list[bool]

    def feature_planes(self) -> list[float]:
        return self.features

    def legal_mask(self) -> list[bool]:
        return self.mask


def test_states_to_feature_tensor_uses_batch_channel_board_shape() -> None:
    state = FakeFeatureState(
        features=[1.0] + [0.0] * (FEATURE_CHANNELS * BOARD_SIZE * BOARD_SIZE - 1),
        mask=[True] * ACTION_SPACE,
    )

    tensor = states_to_feature_tensor([state])

    assert tensor.shape == (1, FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    assert tensor.dtype == torch.float32
    assert tensor[0, 0, 0, 0].item() == 1.0


def test_states_to_legal_mask_tensor_uses_action_space_shape() -> None:
    mask = [False] * ACTION_SPACE
    mask[81] = True
    state = FakeFeatureState(
        features=[0.0] * (FEATURE_CHANNELS * BOARD_SIZE * BOARD_SIZE),
        mask=mask,
    )

    tensor = states_to_legal_mask_tensor([state])

    assert tensor.shape == (1, ACTION_SPACE)
    assert tensor.dtype == torch.bool
    assert tensor[0, 81].item() is True


def test_feature_tensor_rejects_wrong_flat_feature_length() -> None:
    state = FakeFeatureState(features=[0.0], mask=[True] * ACTION_SPACE)

    with pytest.raises(ValueError, match="feature shape"):
        states_to_feature_tensor([state])
