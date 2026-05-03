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

from great_kingdom_ai.evaluator import (  # noqa: E402
    evaluate_feature_batch_logits_values,
    evaluate_request,
    evaluate_request_bytes,
    evaluate_request_bytes_logits_values,
)
from great_kingdom_ai.features import ACTION_SPACE, BOARD_SIZE, FEATURE_CHANNELS  # noqa: E402


@dataclass(frozen=True)
class FakeEvalRequest:
    features: list[list[float]]
    masks: list[list[bool]]

    def feature_planes(self) -> list[list[float]]:
        return self.features

    def legal_masks(self) -> list[list[bool]]:
        return self.masks


@dataclass(frozen=True)
class FakeByteEvalRequest:
    features: bytes
    masks: bytes
    batch_size: int

    def len(self) -> int:
        return self.batch_size

    def feature_plane_bytes(self) -> bytes:
        return self.features

    def legal_mask_bytes(self) -> bytes:
        return self.masks


if torch is not None:

    class FixedNetwork(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            batch = x.shape[0]
            policy = torch.arange(ACTION_SPACE, dtype=torch.float32).repeat(batch, 1)
            value = torch.linspace(-0.5, 0.5, steps=batch)
            return policy, value


def test_evaluate_request_returns_masked_policy_and_value_batch() -> None:
    mask = [False] * ACTION_SPACE
    mask[0] = True
    mask[81] = True
    request = FakeEvalRequest(
        features=[[0.0] * (FEATURE_CHANNELS * BOARD_SIZE * BOARD_SIZE)],
        masks=[mask],
    )

    result = evaluate_request(FixedNetwork(), request)

    assert result.policy.shape == (1, ACTION_SPACE)
    assert result.value.shape == (1,)
    assert result.policy[0, 1] == 0.0
    assert result.policy[0, 81] > result.policy[0, 0]
    assert result.policy[0].sum() == pytest.approx(1.0)
    assert result.value[0] == pytest.approx(-0.5)


def test_evaluate_request_bytes_returns_masked_policy_and_value_batch() -> None:
    mask = [False] * ACTION_SPACE
    mask[0] = True
    mask[81] = True
    features = torch.zeros(
        (1, FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE),
        dtype=torch.float32,
    ).numpy()
    masks = torch.tensor([mask], dtype=torch.bool).numpy()
    request = FakeByteEvalRequest(
        features=features.tobytes(),
        masks=masks.tobytes(),
        batch_size=1,
    )

    result = evaluate_request_bytes(FixedNetwork(), request)

    assert result.policy.shape == (1, ACTION_SPACE)
    assert result.value.shape == (1,)
    assert result.policy[0, 1] == 0.0
    assert result.policy[0, 81] > result.policy[0, 0]
    assert result.policy[0].sum() == pytest.approx(1.0)
    assert result.value[0] == pytest.approx(-0.5)


def test_evaluate_feature_batch_logits_values_skips_policy_output() -> None:
    mask = [False] * ACTION_SPACE
    mask[0] = True
    mask[81] = True

    result = evaluate_feature_batch_logits_values(
        FixedNetwork(),
        [[0.0] * (FEATURE_CHANNELS * BOARD_SIZE * BOARD_SIZE)],
        [mask],
    )

    assert result.policy_logits.shape == (1, ACTION_SPACE)
    assert result.value.shape == (1,)
    assert not hasattr(result, "policy")


def test_evaluate_request_bytes_logits_values_returns_logits_and_value() -> None:
    mask = [False] * ACTION_SPACE
    mask[0] = True
    mask[81] = True
    features = torch.zeros(
        (1, FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE),
        dtype=torch.float32,
    ).numpy()
    masks = torch.tensor([mask], dtype=torch.bool).numpy()
    request = FakeByteEvalRequest(
        features=features.tobytes(),
        masks=masks.tobytes(),
        batch_size=1,
    )

    result = evaluate_request_bytes_logits_values(FixedNetwork(), request)

    assert result.policy_logits.shape == (1, ACTION_SPACE)
    assert result.value.shape == (1,)


def test_evaluate_request_rejects_mismatched_batch_sizes() -> None:
    request = FakeEvalRequest(
        features=[[0.0] * (FEATURE_CHANNELS * BOARD_SIZE * BOARD_SIZE)],
        masks=[],
    )

    with pytest.raises(ValueError, match="same length"):
        evaluate_request(FixedNetwork(), request)
