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
        features = model.forward_features(inputs)
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


def test_attention_block_is_d4_equivariant_for_non_contiguous_inputs() -> None:
    from great_kingdom_ai.model import BoardSelfAttentionBlock

    torch.manual_seed(0)
    block = BoardSelfAttentionBlock(
        channels=16, num_heads=4, residual_scale_init=1.0
    )
    block.eval()
    # Randomize the relative bias so the test exercises the D4 weight sharing
    # (a zero table would be trivially symmetric).
    with torch.no_grad():
        block.relative_bias.relative_bias_table.normal_()

    # A transposed tensor is non-contiguous, which used to crash the internal view().
    base = torch.randn(2, 16, BOARD_SIZE, BOARD_SIZE)
    inputs = base.transpose(2, 3)
    assert not inputs.is_contiguous()

    with torch.no_grad():
        reference = block(inputs)

    for k in (1, 2, 3):
        with torch.no_grad():
            rotated = block(torch.rot90(inputs, k, dims=(2, 3)))
        assert torch.allclose(
            rotated, torch.rot90(reference, k, dims=(2, 3)), atol=1e-5, rtol=1e-4
        )

    for dim in (2, 3):
        with torch.no_grad():
            flipped = block(torch.flip(inputs, dims=(dim,)))
        assert torch.allclose(
            flipped, torch.flip(reference, dims=(dim,)), atol=1e-5, rtol=1e-4
        )


def test_d4_relative_position_bias_shapes_and_values() -> None:
    from great_kingdom_ai.model import D4RelativePositionBias

    num_heads = 4
    expected_positions = BOARD_SIZE * (BOARD_SIZE + 1) // 2
    bias_module = D4RelativePositionBias(num_heads=num_heads, board_size=BOARD_SIZE)
    assert bias_module.relative_bias_table.shape == (expected_positions, num_heads)
    assert bias_module.relative_index.shape == (81, 81)

    # Check that indices are within 0..num_relative_positions-1
    assert torch.all(bias_module.relative_index >= 0)
    assert torch.all(bias_module.relative_index < expected_positions)

    # Output shape should be [1, heads, 81, 81]
    with torch.no_grad():
        bias = bias_module()
    assert bias.shape == (1, num_heads, 81, 81)


def test_d4_relative_position_bias_known_coordinate_mapping() -> None:
    from great_kingdom_ai.model import D4RelativePositionBias

    board_size = 9
    bias_module = D4RelativePositionBias(num_heads=2, board_size=board_size)
    rel_idx = bias_module.relative_index

    # helper to convert row, col to flat index
    def to_flat(row: int, col: int) -> int:
        return row * board_size + col

    def orbit_index(dr: int, dc: int) -> int:
        major = max(abs(dr), abs(dc))
        minor = min(abs(dr), abs(dc))
        return major * (major + 1) // 2 + minor

    # Same cell: displacement (0, 0) -> orbit (0, 0) -> index 0
    assert rel_idx[to_flat(0, 0), to_flat(0, 0)].item() == orbit_index(0, 0) == 0

    # Horizontal and vertical neighbors share the same D4 orbit (1, 0)
    horizontal = rel_idx[to_flat(0, 0), to_flat(0, 1)].item()
    vertical = rel_idx[to_flat(1, 0), to_flat(0, 0)].item()
    assert horizontal == orbit_index(0, 1) == 1
    assert horizontal == vertical

    # Corner to opposite corner: (0, 0) -> (8, 8) is the furthest orbit (8, 8)
    expected_far = board_size * (board_size - 1) // 2 + (board_size - 1)
    assert rel_idx[to_flat(0, 0), to_flat(8, 8)].item() == expected_far
    assert rel_idx[to_flat(8, 8), to_flat(0, 0)].item() == expected_far


def test_d4_relative_position_bias_same_offset_reuses_index() -> None:
    from great_kingdom_ai.model import D4RelativePositionBias

    board_size = 9
    bias_module = D4RelativePositionBias(num_heads=2, board_size=board_size)
    rel_idx = bias_module.relative_index

    def to_flat(row: int, col: int) -> int:
        return row * board_size + col

    # Example pairs for (dr=1, dc=-2) (i.e. key_row - query_row = 1, key_col - query_col = -2)
    # Pair A: Query (2, 3), key (3, 1) -> dr = 1, dc = -2
    # Pair B: Query (5, 4), key (6, 2) -> dr = 1, dc = -2

    q_a, k_a = to_flat(2, 3), to_flat(3, 1)
    q_b, k_b = to_flat(5, 4), to_flat(6, 2)

    val_a = rel_idx[q_a, k_a].item()
    val_b = rel_idx[q_b, k_b].item()

    assert val_a == val_b

    # major = max(1, 2) = 2, minor = min(1, 2) = 1 -> 2 * 3 // 2 + 1 = 4
    assert val_a == 4


def test_d4_relative_position_bias_shares_d4_orbits() -> None:
    from great_kingdom_ai.model import D4RelativePositionBias

    board_size = 9
    bias_module = D4RelativePositionBias(num_heads=2, board_size=board_size)
    rel_idx = bias_module.relative_index

    def to_flat(row: int, col: int) -> int:
        return row * board_size + col

    def orbit(dr: int, dc: int) -> tuple[int, int]:
        return (max(abs(dr), abs(dc)), min(abs(dr), abs(dc)))

    orbit_to_index: dict[tuple[int, int], int] = {}
    for q_r in range(board_size):
        for q_c in range(board_size):
            for k_r in range(board_size):
                for k_c in range(board_size):
                    dr = k_r - q_r
                    dc = k_c - q_c
                    key = orbit(dr, dc)
                    idx = rel_idx[to_flat(q_r, q_c), to_flat(k_r, k_c)].item()

                    # Assert every displacement in the same D4 orbit maps to one index
                    if key in orbit_to_index:
                        assert orbit_to_index[key] == idx
                    else:
                        orbit_to_index[key] = idx

    # D4 orbits of a 9x9 board: 45 distinct (major, minor) pairs.
    expected_unique = board_size * (board_size + 1) // 2
    assert len(orbit_to_index) == expected_unique
    assert set(orbit_to_index.values()) == set(range(expected_unique))


def test_d4_relative_position_bias_forward_gathers_table_values() -> None:
    from great_kingdom_ai.model import D4RelativePositionBias

    num_heads = 4
    board_size = 9
    bias_module = D4RelativePositionBias(num_heads=num_heads, board_size=board_size)

    # Fill relative_bias_table with deterministic values
    # relative_bias_table.shape == (num_relative_positions, num_heads)
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


def test_d4_relative_position_bias_table_receives_gradient() -> None:
    from great_kingdom_ai.model import D4RelativePositionBias

    num_heads = 2
    board_size = 9
    bias_module = D4RelativePositionBias(num_heads=num_heads, board_size=board_size)

    # Check that parameters are initialized with requires_grad=True
    assert bias_module.relative_bias_table.requires_grad is True

    loss = bias_module().sum()
    loss.backward()

    assert bias_module.relative_bias_table.grad is not None
    assert bias_module.relative_bias_table.grad.shape == bias_module.relative_bias_table.shape
    assert torch.all(torch.isfinite(bias_module.relative_bias_table.grad))

    # Every D4 orbit appears at least once on a 9x9 board, so all indices
    # in the table must receive positive gradients.
    assert torch.all(bias_module.relative_bias_table.grad > 0)


@pytest.mark.parametrize("board_size", [1, 2, 3, 9])
def test_d4_relative_position_bias_board_size_invariants(board_size: int) -> None:
    from great_kingdom_ai.model import D4RelativePositionBias

    num_heads = 3
    bias_module = D4RelativePositionBias(num_heads=num_heads, board_size=board_size)

    expected_rows = board_size * (board_size + 1) // 2
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


def test_strong_attn_preset_interleaves_two_attention_blocks() -> None:
    from great_kingdom_ai.model import BoardSelfAttentionBlock, create_model

    model = create_model("strong_attn")

    # Check ModelConfig fields
    assert model.config.attention_blocks == 2
    assert model.config.attention_heads == 4
    assert model.config.attention_ffn_multiplier == 4
    assert model.config.attention_residual_scale_init == 1e-2
    assert model.config.attention_insert_every == 5

    # Check model architecture
    assert len(model.backbone) == 10
    assert len(model.attention) == 2
    for block in model.attention:
        assert isinstance(block, BoardSelfAttentionBlock)

    # Attention is interleaved after every 5 residual blocks rather than stacked at the end
    expected_plan = (
        *(("residual", index) for index in range(5)),
        ("attention", 0),
        *(("residual", index) for index in range(5, 10)),
        ("attention", 1),
    )
    assert model.stage_plan == expected_plan


def test_build_stage_plan_interleaves_and_supports_legacy_layout() -> None:
    from great_kingdom_ai.model import build_stage_plan

    assert build_stage_plan(10, 2, 5) == (
        *(("residual", index) for index in range(5)),
        ("attention", 0),
        *(("residual", index) for index in range(5, 10)),
        ("attention", 1),
    )
    assert build_stage_plan(4, 0, 0) == (
        ("residual", 0),
        ("residual", 1),
        ("residual", 2),
        ("residual", 3),
    )
    assert build_stage_plan(2, 2, 0) == (
        ("residual", 0),
        ("residual", 1),
        ("attention", 0),
        ("attention", 1),
    )
    # Uneven grouping still drains every block exactly once
    assert build_stage_plan(3, 2, 5) == (
        ("residual", 0),
        ("residual", 1),
        ("residual", 2),
        ("attention", 0),
        ("attention", 1),
    )


def test_attention_block_rejects_wrong_spatial_dimensions() -> None:
    from great_kingdom_ai.model import BoardSelfAttentionBlock

    block = BoardSelfAttentionBlock(channels=64, num_heads=4, board_size=9)
    block.eval()

    # 3x27 has 81 cells, but wrong spatial dims
    inputs_wrong_dims = torch.randn(2, 64, 3, 27)
    with pytest.raises(ValueError, match="expected spatial shape"):
        block(inputs_wrong_dims)


def test_action_value_head_is_off_by_default() -> None:
    from great_kingdom_ai.model import ModelConfig, create_model

    assert ModelConfig().action_value_head is False
    model = create_model("small")
    assert model.has_action_value_head is False
    assert not hasattr(model, "q_spatial") or model.q_spatial is None
    assert model.value_head is not None

    with pytest.raises(ValueError, match="action value head"):
        model.forward_q(torch.zeros((1, FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE)))


def test_strong_attn_klent_preset_replaces_state_value_head_with_q_head() -> None:
    from great_kingdom_ai.model import create_model

    model = create_model("strong_attn_klent")

    assert model.config.action_value_head is True
    assert model.config.spatial_value_head is True
    assert model.has_action_value_head is True
    assert model.value_head is None
    assert model.q_spatial is not None
    assert model.q_pass is not None
    assert any(isinstance(module, torch.nn.Tanh) for module in model.q_spatial)
    assert any(isinstance(module, torch.nn.Tanh) for module in model.q_pass)


def test_action_value_head_returns_policy_logits_and_masked_value() -> None:
    from great_kingdom_ai.features import LEGAL_PLACE_FEATURE_CHANNEL
    from great_kingdom_ai.klent.targets import legal_mask_from_features, masked_state_value
    from great_kingdom_ai.model import ModelConfig, PolicyValueNetwork

    model = PolicyValueNetwork(ModelConfig(channels=16, residual_blocks=1, action_value_head=True))
    model.eval()
    features = torch.zeros((2, FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=torch.float32)
    features[:, LEGAL_PLACE_FEATURE_CHANNEL] = 0.0
    features[0, LEGAL_PLACE_FEATURE_CHANNEL, 0, 0] = 1.0
    features[1, LEGAL_PLACE_FEATURE_CHANNEL, :, :] = 1.0

    with torch.no_grad():
        policy_logits, value = model(features)
        q_policy_logits, q_values = model.forward_q(features)

    assert value.shape == (2,)
    assert q_values.shape == (2, ACTION_SPACE)
    assert torch.equal(policy_logits, q_policy_logits)
    assert torch.all(q_values <= 1.0)
    assert torch.all(q_values >= -1.0)
    legal_mask = legal_mask_from_features(features)
    expected = masked_state_value(q_policy_logits, q_values, legal_mask)
    assert torch.allclose(value, expected, atol=1e-6)

    unmasked = (torch.softmax(q_policy_logits, dim=1) * q_values).sum(dim=1)
    assert not torch.allclose(value, unmasked, atol=1e-6)


def test_action_value_head_forward_with_aux_uses_masked_value() -> None:
    from great_kingdom_ai.model import ModelConfig, PolicyValueNetwork

    model = PolicyValueNetwork(
        ModelConfig(
            channels=16,
            residual_blocks=1,
            action_value_head=True,
            terminal_board_head=True,
        )
    )
    model.eval()
    features = torch.zeros((1, FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=torch.float32)

    with torch.no_grad():
        policy_logits, value, aux_logits = model.forward_with_aux(features)

    assert policy_logits.shape == (1, ACTION_SPACE)
    assert value.shape == (1,)
    assert aux_logits.shape[0] == 1
