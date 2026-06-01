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
    [
        "small",
        "medium",
        "medium_plus",
        "large",
        "large_policy",
        "large_plus",
        "strong",
        "strong_clean",
        "strong_attn",
    ],
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


def test_attention_block_keeps_input_shape() -> None:
    from great_kingdom_ai.model import BoardSelfAttentionBlock

    block = BoardSelfAttentionBlock(channels=64, num_heads=4)
    block.eval()
    inputs = torch.randn(2, 64, BOARD_SIZE, BOARD_SIZE)
    with torch.no_grad():
        outputs = block(inputs)
    assert outputs.shape == inputs.shape


def test_full_2d_relative_position_bias_shapes_and_values() -> None:
    from great_kingdom_ai.model import Full2DRelativePositionBias

    num_heads = 4
    bias_module = Full2DRelativePositionBias(num_heads=num_heads, board_size=BOARD_SIZE)
    assert bias_module.relative_bias_table.shape == (289, num_heads)
    assert bias_module.relative_index.shape == (81, 81)

    # Check that indices are within 0..288
    assert torch.all(bias_module.relative_index >= 0)
    assert torch.all(bias_module.relative_index < 289)

    # Output shape should be [1, heads, 81, 81]
    with torch.no_grad():
        bias = bias_module()
    assert bias.shape == (1, num_heads, 81, 81)


def test_full_2d_relative_position_bias_known_coordinate_mapping() -> None:
    from great_kingdom_ai.model import Full2DRelativePositionBias

    board_size = 9
    bias_module = Full2DRelativePositionBias(num_heads=2, board_size=board_size)
    rel_idx = bias_module.relative_index

    # helper to convert row, col to flat index
    def to_flat(row: int, col: int) -> int:
        return row * board_size + col

    # Same cell: (0, 0) -> (0, 0) maps to center offset (0, 0)
    # dr = 0, dc = 0
    # expected index = (0 + 8) * 17 + (0 + 8) = 144
    q1 = to_flat(0, 0)
    k1 = to_flat(0, 0)
    assert rel_idx[q1, k1].item() == 144

    # Horizontal neighbor: (0, 0) -> (0, 1) maps to offset (0, 1)
    # dr = 0, dc = 1
    # expected index = (0 + 8) * 17 + (1 + 8) = 145
    q2 = to_flat(0, 0)
    k2 = to_flat(0, 1)
    assert rel_idx[q2, k2].item() == 145

    # Vertical neighbor: (1, 0) -> (0, 0) maps to offset (-1, 0)
    # dr = -1, dc = 0
    # expected index = (-1 + 8) * 17 + (0 + 8) = 127
    q3 = to_flat(1, 0)
    k3 = to_flat(0, 0)
    assert rel_idx[q3, k3].item() == 127

    # Corner to opposite corner: (0, 0) -> (8, 8) maps to offset (8, 8)
    # dr = 8, dc = 8
    # expected index = (8 + 8) * 17 + (8 + 8) = 288
    q4 = to_flat(0, 0)
    k4 = to_flat(8, 8)
    assert rel_idx[q4, k4].item() == 288

    # Opposite corner back: (8, 8) -> (0, 0) maps to offset (-8, -8)
    # dr = -8, dc = -8
    # expected index = (-8 + 8) * 17 + (-8 + 8) = 0
    q5 = to_flat(8, 8)
    k5 = to_flat(0, 0)
    assert rel_idx[q5, k5].item() == 0

    # Mixed interior pair: (4, 4) -> (5, 3) maps to offset (1, -1)
    # dr = 1, dc = -1
    # expected index = (1 + 8) * 17 + (-1 + 8) = 160
    q6 = to_flat(4, 4)
    k6 = to_flat(5, 3)
    assert rel_idx[q6, k6].item() == 160


def test_full_2d_relative_position_bias_same_offset_reuses_index() -> None:
    from great_kingdom_ai.model import Full2DRelativePositionBias

    board_size = 9
    bias_module = Full2DRelativePositionBias(num_heads=2, board_size=board_size)
    rel_idx = bias_module.relative_index

    def to_flat(row: int, col: int) -> int:
        return row * board_size + col

    # Example pairs for (dr=1, dc=-2) (i.e. key_row - query_row = 1, key_col - query_col = -2)
    # Pair A: Query (2, 3), key (3, 1) -> dr = 1, dc = -2
    # Pair B: Query (5, 4), key (6, 2) -> dr = 1, dc = -2
    # Pair C: Query (7, 2), key (8, 0) -> dr = 1, dc = -2

    q_a, k_a = to_flat(2, 3), to_flat(3, 1)
    q_b, k_b = to_flat(5, 4), to_flat(6, 2)
    q_c, k_c = to_flat(7, 2), to_flat(8, 0)

    val_a = rel_idx[q_a, k_a].item()
    val_b = rel_idx[q_b, k_b].item()
    val_c = rel_idx[q_c, k_c].item()

    assert val_a == val_b
    assert val_b == val_c

    # Also verify it matches the computed formula index for dr=1, dc=-2:
    # expected index = (1 + 8) * 17 + (-2 + 8) = 9 * 17 + 6 = 159
    assert val_a == 159


def test_full_2d_relative_position_bias_offsets_do_not_collide() -> None:
    from great_kingdom_ai.model import Full2DRelativePositionBias

    board_size = 9
    bias_module = Full2DRelativePositionBias(num_heads=2, board_size=board_size)
    rel_idx = bias_module.relative_index

    offset_to_idx = {}
    idx_to_offset = {}

    for q_r in range(board_size):
        for q_c in range(board_size):
            for k_r in range(board_size):
                for k_c in range(board_size):
                    dr = k_r - q_r
                    dc = k_c - q_c
                    offset = (dr, dc)

                    q_flat = q_r * board_size + q_c
                    k_flat = k_r * board_size + k_c

                    idx = rel_idx[q_flat, k_flat].item()

                    # Assert every (dr, dc) maps to exactly one index
                    if offset in offset_to_idx:
                        assert offset_to_idx[offset] == idx
                    else:
                        offset_to_idx[offset] = idx

                    # Assert no index maps to more than one (dr, dc)
                    if idx in idx_to_offset:
                        assert idx_to_offset[idx] == offset
                    else:
                        idx_to_offset[idx] = offset

    # The number of unique offsets is 289 (17 * 17)
    expected_unique = (2 * board_size - 1) ** 2
    assert len(offset_to_idx) == expected_unique
    assert len(idx_to_offset) == expected_unique


def test_full_2d_relative_position_bias_forward_gathers_table_values() -> None:
    from great_kingdom_ai.model import Full2DRelativePositionBias

    num_heads = 4
    board_size = 9
    bias_module = Full2DRelativePositionBias(num_heads=num_heads, board_size=board_size)

    # Fill relative_bias_table with deterministic values
    # relative_bias_table.shape == (289, num_heads)
    with torch.no_grad():
        for index in range(bias_module.num_relative_positions):
            for head in range(num_heads):
                bias_module.relative_bias_table[index, head] = index * 10.0 + head

    # Forward pass
    with torch.no_grad():
        bias = bias_module()  # shape [1, num_heads, 81, 81]

    assert bias.shape == (1, num_heads, 81, 81)

    # Verify selected values
    def to_flat(row: int, col: int) -> int:
        return row * board_size + col
    rel_idx = bias_module.relative_index

    test_pairs = [
        ((0, 0), (0, 0)),
        ((0, 0), (0, 1)),
        ((4, 4), (5, 3)),
        ((8, 8), (0, 0)),
    ]

    for (q_r, q_c), (k_r, k_c) in test_pairs:
        q_flat = to_flat(q_r, q_c)
        k_flat = to_flat(k_r, k_c)

        idx = rel_idx[q_flat, k_flat].item()

        for head in range(num_heads):
            expected_val = idx * 10.0 + head
            actual_val = bias[0, head, q_flat, k_flat].item()
            assert actual_val == expected_val


def test_full_2d_relative_position_bias_table_receives_gradient() -> None:
    from great_kingdom_ai.model import Full2DRelativePositionBias

    num_heads = 2
    board_size = 9
    bias_module = Full2DRelativePositionBias(num_heads=num_heads, board_size=board_size)

    # Check that parameters are initialized with requires_grad=True
    assert bias_module.relative_bias_table.requires_grad is True

    loss = bias_module().sum()
    loss.backward()

    assert bias_module.relative_bias_table.grad is not None
    assert bias_module.relative_bias_table.grad.shape == bias_module.relative_bias_table.shape
    assert torch.all(torch.isfinite(bias_module.relative_bias_table.grad))

    # Since every relative offset (289 total) appears at least once on a 9x9 board,
    # all indices in the table must receive positive gradients.
    assert torch.all(bias_module.relative_bias_table.grad > 0)


@pytest.mark.parametrize("board_size", [1, 2, 3, 9])
def test_full_2d_relative_position_bias_board_size_invariants(board_size: int) -> None:
    from great_kingdom_ai.model import Full2DRelativePositionBias

    num_heads = 3
    bias_module = Full2DRelativePositionBias(num_heads=num_heads, board_size=board_size)

    expected_rows = (2 * board_size - 1) ** 2
    expected_positions = board_size * board_size

    # Table rows check
    assert bias_module.relative_bias_table.shape == (expected_rows, num_heads)

    # Index shape check
    assert bias_module.relative_index.shape == (expected_positions, expected_positions)

    # Unique used indexes check
    unique_indices = torch.unique(bias_module.relative_index)
    assert len(unique_indices) == expected_rows
    assert torch.all(unique_indices >= 0)
    assert torch.all(unique_indices < expected_rows)

    # Forward shape check
    with torch.no_grad():
        bias = bias_module()
    assert bias.shape == (1, num_heads, expected_positions, expected_positions)


def test_attention_block_scale_initialization() -> None:
    from great_kingdom_ai.model import BoardSelfAttentionBlock

    residual_scale_init = 1e-3
    block = BoardSelfAttentionBlock(
        channels=64, num_heads=4, residual_scale_init=residual_scale_init
    )

    assert torch.allclose(block.attn_scale, torch.full((64,), residual_scale_init))
    assert torch.allclose(block.ffn_scale, torch.full((64,), residual_scale_init))


def test_strong_attn_preset_contains_two_attention_blocks() -> None:
    from great_kingdom_ai.model import BoardSelfAttentionBlock, create_model

    model = create_model("strong_attn")

    # Check ModelConfig fields
    assert model.config.attention_blocks == 2
    assert model.config.attention_heads == 4
    assert model.config.attention_ffn_multiplier == 4
    assert model.config.attention_residual_scale_init == 1e-3

    # Check model architecture
    assert len(model.attention) == 2
    for block in model.attention:
        assert isinstance(block, BoardSelfAttentionBlock)


def test_attention_block_rejects_wrong_spatial_dimensions() -> None:
    from great_kingdom_ai.model import BoardSelfAttentionBlock

    block = BoardSelfAttentionBlock(channels=64, num_heads=4, board_size=9)
    block.eval()

    # 3x27 has 81 cells, but wrong spatial dims
    inputs_wrong_dims = torch.randn(2, 64, 3, 27)
    with pytest.raises(ValueError, match="expected spatial shape"):
        block(inputs_wrong_dims)
