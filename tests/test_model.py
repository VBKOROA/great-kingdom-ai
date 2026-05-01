from __future__ import annotations

import importlib
import importlib.util

import pytest

_torch_spec = importlib.util.find_spec("torch")
pytestmark = pytest.mark.skipif(
    _torch_spec is None,
    reason="torch is not installed",
)
torch = importlib.import_module("torch") if _torch_spec is not None else None

from great_kingdom_ai.features import ACTION_SPACE, BOARD_SIZE, FEATURE_CHANNELS  # noqa: E402


@pytest.mark.parametrize("preset", ["small", "medium"])
def test_model_presets_return_policy_logits_and_value_scalar(preset: str) -> None:
    from great_kingdom_ai.model import create_model

    model = create_model(preset)
    model.eval()
    inputs = torch.zeros((2, FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=torch.float32)

    with torch.no_grad():
        policy_logits, value = model(inputs)

    assert policy_logits.shape == (2, ACTION_SPACE)
    assert value.shape == (2,)
    assert torch.all(value <= 1.0)
    assert torch.all(value >= -1.0)


def test_policy_head_splits_board_locations_and_pass_logit() -> None:
    from great_kingdom_ai.model import create_model

    model = create_model("small")
    model.eval()
    inputs = torch.zeros((3, FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=torch.float32)

    with torch.no_grad():
        features = model.backbone(model.stem(inputs))
        board_logits = model.policy_spatial(features).flatten(start_dim=1)
        pass_logits = model.policy_pass(features)
        policy_logits, _ = model(inputs)

    assert board_logits.shape == (3, BOARD_SIZE * BOARD_SIZE)
    assert pass_logits.shape == (3, 1)
    assert torch.equal(policy_logits[:, :81], board_logits)
    assert torch.equal(policy_logits[:, 81:], pass_logits)


def test_value_head_uses_global_pooling_without_flattening_board_cells() -> None:
    from great_kingdom_ai.model import ModelConfig, PolicyValueNetwork

    model = PolicyValueNetwork(ModelConfig(channels=16, residual_blocks=1, value_hidden=8))

    assert any(isinstance(module, torch.nn.AdaptiveAvgPool2d) for module in model.value_head)
    assert not any(
        isinstance(module, torch.nn.Linear) and module.in_features == 16 * BOARD_SIZE * BOARD_SIZE
        for module in model.value_head
    )


def test_model_rejects_wrong_input_shape() -> None:
    from great_kingdom_ai.model import create_model

    model = create_model("small")

    with pytest.raises(ValueError, match="expected input shape"):
        model(torch.zeros((1, FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE - 1)))
