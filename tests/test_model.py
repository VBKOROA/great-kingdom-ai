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


@pytest.mark.parametrize(
    "preset",
    ["small", "medium", "medium_plus", "large", "large_policy", "large_plus", "strong"],
)
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


def test_medium_plus_increases_capacity_and_policy_head_width() -> None:
    from great_kingdom_ai.model import create_model

    medium = create_model("medium")
    medium_plus = create_model("medium_plus")

    assert medium_plus.config.channels == 96
    assert medium_plus.config.residual_blocks == 6
    assert medium_plus.config.policy_channels == 16
    assert sum(p.numel() for p in medium_plus.parameters()) > sum(
        p.numel() for p in medium.parameters()
    )


def test_strong_preset_uses_spatial_value_head_and_policy_context() -> None:
    from great_kingdom_ai.model import create_model

    strong = create_model("strong")

    assert strong.config.channels == 128
    assert strong.config.residual_blocks == 10
    assert strong.config.policy_channels == 32
    assert strong.config.policy_kernel_size == 3
    assert strong.config.spatial_value_head is True
    assert any(
        isinstance(module, torch.nn.Conv2d)
        and module.out_channels == strong.config.value_spatial_channels
        for module in strong.value_head
    )
    assert any(
        isinstance(module, torch.nn.Linear)
        and module.in_features == strong.config.value_spatial_channels * BOARD_SIZE * BOARD_SIZE
        for module in strong.value_head
    )
    assert not any(isinstance(module, torch.nn.AdaptiveAvgPool2d) for module in strong.value_head)


def test_large_plus_preset_adds_context_heads_without_extra_backbone_depth() -> None:
    from great_kingdom_ai.model import create_model

    large = create_model("large")
    large_plus = create_model("large_plus")
    strong = create_model("strong")

    assert large_plus.config.channels == 128
    assert large_plus.config.residual_blocks == 8
    assert large_plus.config.policy_channels == 16
    assert large_plus.config.policy_kernel_size == 3
    assert large_plus.config.spatial_value_head is True
    assert sum(p.numel() for p in large.parameters()) < sum(
        p.numel() for p in large_plus.parameters()
    ) < sum(p.numel() for p in strong.parameters())
    assert any(
        isinstance(module, torch.nn.Conv2d)
        and module.out_channels == large_plus.config.value_spatial_channels
        for module in large_plus.value_head
    )
    assert any(
        isinstance(module, torch.nn.Linear)
        and module.in_features == large_plus.config.value_spatial_channels * BOARD_SIZE * BOARD_SIZE
        for module in large_plus.value_head
    )
    assert not any(
        isinstance(module, torch.nn.AdaptiveAvgPool2d) for module in large_plus.value_head
    )



def test_large_policy_preset_adds_policy_context_without_spatial_value_head() -> None:
    from great_kingdom_ai.model import create_model

    large = create_model("large")
    large_policy = create_model("large_policy")
    large_plus = create_model("large_plus")

    assert large_policy.config.channels == 128
    assert large_policy.config.residual_blocks == 8
    assert large_policy.config.policy_channels == 16
    assert large_policy.config.policy_kernel_size == 3
    assert large_policy.config.spatial_value_head is False
    assert sum(p.numel() for p in large.parameters()) < sum(
        p.numel() for p in large_policy.parameters()
    ) < sum(p.numel() for p in large_plus.parameters())
    assert any(isinstance(module, torch.nn.AdaptiveAvgPool2d) for module in large_policy.value_head)
    assert not any(
        isinstance(module, torch.nn.Linear)
        and module.in_features == large_policy.config.channels * BOARD_SIZE * BOARD_SIZE
        for module in large_policy.value_head
    )


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
