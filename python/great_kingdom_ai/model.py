"""PyTorch policy-value network for Great Kingdom AlphaZero-lite."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import cast

import torch
from torch import nn

from great_kingdom_ai.features import ACTION_SPACE, BOARD_SIZE, FEATURE_CHANNELS
from great_kingdom_ai.klent.targets import legal_mask_from_features, masked_state_value
from great_kingdom_ai.replay.terminal_board import TERMINAL_BOARD_CLASSES


@dataclass(frozen=True)
class ModelConfig:
    input_channels: int = FEATURE_CHANNELS
    channels: int = 64
    residual_blocks: int = 4
    value_hidden: int = 64
    policy_channels: int = 2
    policy_kernel_size: int = 1
    spatial_value_head: bool = False
    value_spatial_channels: int = 2
    attention_blocks: int = 0
    attention_heads: int = 4
    attention_ffn_multiplier: int = 4
    attention_residual_scale_init: float = 1e-2
    attention_insert_every: int = 0
    terminal_board_head: bool = False
    terminal_board_hidden_channels: int = 32
    terminal_board_classes: int = TERMINAL_BOARD_CLASSES
    action_value_head: bool = False

    def __post_init__(self) -> None:
        if self.residual_blocks < 0:
            raise ValueError(f"residual_blocks must be non-negative, got {self.residual_blocks}")
        if self.attention_blocks < 0:
            raise ValueError(f"attention_blocks must be non-negative, got {self.attention_blocks}")
        if self.attention_insert_every < 0:
            raise ValueError(
                "attention_insert_every must be non-negative, "
                f"got {self.attention_insert_every}"
            )
        if self.terminal_board_hidden_channels <= 0:
            raise ValueError("terminal_board_hidden_channels must be positive")
        if self.terminal_board_classes <= 0:
            raise ValueError("terminal_board_classes must be positive")


MODEL_PRESETS: dict[str, ModelConfig] = {
    "small": ModelConfig(channels=32, residual_blocks=2, value_hidden=64),
    "medium": ModelConfig(channels=64, residual_blocks=4, value_hidden=128),
    "medium_plus": ModelConfig(
        channels=96,
        residual_blocks=6,
        value_hidden=192,
        policy_channels=16,
    ),
    "strong": ModelConfig(
        channels=128,
        residual_blocks=10,
        value_hidden=256,
        policy_channels=32,
        policy_kernel_size=3,
        spatial_value_head=True,
    ),
    "strong_clean": ModelConfig(
        channels=128,
        residual_blocks=10,
        value_hidden=256,
        policy_channels=16,
        attention_blocks=0,
        spatial_value_head=True,
    ),
    "strong_attn": ModelConfig(
        channels=128,
        residual_blocks=10,
        value_hidden=256,
        policy_channels=16,
        attention_blocks=2,
        attention_heads=4,
        attention_insert_every=5,
        attention_residual_scale_init=1e-2,
        spatial_value_head=True,
    ),
    "large": ModelConfig(channels=128, residual_blocks=8, value_hidden=256),
    "large_policy": ModelConfig(
        channels=128,
        residual_blocks=8,
        value_hidden=256,
        policy_channels=16,
        policy_kernel_size=3,
    ),
    "large_plus": ModelConfig(
        channels=128,
        residual_blocks=8,
        value_hidden=256,
        policy_channels=16,
        policy_kernel_size=3,
        spatial_value_head=True,
    ),
}

MODEL_PRESETS["strong_attn_terminal_board"] = replace(
    MODEL_PRESETS["strong_attn"],
    terminal_board_head=True,
)

MODEL_PRESETS["strong_attn_klent"] = replace(
    MODEL_PRESETS["strong_attn"],
    action_value_head=True,
)

MODEL_PRESETS["small_klent"] = replace(
    MODEL_PRESETS["small"],
    action_value_head=True,
)

RESIDUAL_STAGE = "residual"
ATTENTION_STAGE = "attention"


def build_stage_plan(
    residual_blocks: int,
    attention_blocks: int,
    attention_insert_every: int,
) -> tuple[tuple[str, int], ...]:
    """Order residual and attention stages for execution.

    With ``attention_insert_every > 0`` an attention block is interleaved after
    every N residual blocks, so the CNN can spatially refine the global rela-
    tions created by attention. With ``0`` all attention blocks run after the
    full residual backbone (legacy layout).
    """
    if residual_blocks < 0 or attention_blocks < 0 or attention_insert_every < 0:
        raise ValueError("block counts must be non-negative")

    if attention_blocks == 0 or attention_insert_every == 0:
        ordered = [(RESIDUAL_STAGE, i) for i in range(residual_blocks)]
        ordered += [(ATTENTION_STAGE, i) for i in range(attention_blocks)]
        return tuple(ordered)

    plan: list[tuple[str, int]] = []
    residual_index = 0
    attention_index = 0
    while residual_index < residual_blocks or attention_index < attention_blocks:
        for _ in range(attention_insert_every):
            if residual_index >= residual_blocks:
                break
            plan.append((RESIDUAL_STAGE, residual_index))
            residual_index += 1
        if attention_index < attention_blocks:
            plan.append((ATTENTION_STAGE, attention_index))
            attention_index += 1
    return tuple(plan)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.activation(x + self.block(x)))


class D4RelativePositionBias(nn.Module):
    """Relative position bias shared across the board's D4 symmetries.

    A displacement ``(dr, dc)`` is keyed by its D4 orbit
    ``(max(|dr|, |dc|), min(|dr|, |dc|))`` so rotations and reflections of the
    board use the same learned bias. The game rules have no absolute
    up/down/left/right, so independent per-direction bias is unnecessary.
    """

    def __init__(self, num_heads: int, board_size: int = BOARD_SIZE) -> None:
        super().__init__()
        if board_size < 1:
            raise ValueError(f"board_size must be at least 1, got {board_size}")

        self.num_heads = num_heads
        self.board_size = board_size
        self.num_positions = board_size * board_size
        self.max_relative_distance = board_size - 1
        self.num_relative_positions = board_size * (board_size + 1) // 2

        self.relative_bias_table = nn.Parameter(
            torch.zeros(self.num_relative_positions, num_heads)
        )

        self.relative_index: torch.Tensor
        self.register_buffer("relative_index", self._compute_relative_index())

    def _compute_relative_index(self) -> torch.Tensor:
        coords = torch.arange(self.num_positions)
        rows = coords // self.board_size
        cols = coords % self.board_size

        dr = rows.unsqueeze(0) - rows.unsqueeze(1)
        dc = cols.unsqueeze(0) - cols.unsqueeze(1)

        major = torch.maximum(dr.abs(), dc.abs())
        minor = torch.minimum(dr.abs(), dc.abs())
        return major * (major + 1) // 2 + minor

    def forward(self) -> torch.Tensor:
        # [num_positions, num_positions, num_heads] -> [1, num_heads, N, N]
        bias = self.relative_bias_table[self.relative_index]
        return bias.permute(2, 0, 1).unsqueeze(0)


class BoardSelfAttentionBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        ffn_multiplier: int = 4,
        residual_scale_init: float = 1e-2,
        board_size: int = BOARD_SIZE,
    ) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels ({channels}) must be divisible by num_heads ({num_heads})")

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.board_size = board_size

        self.norm1 = nn.LayerNorm(channels)
        self.qkv_proj = nn.Linear(channels, channels * 3, bias=True)
        self.out_proj = nn.Linear(channels, channels, bias=True)

        self.relative_bias = D4RelativePositionBias(num_heads, board_size=board_size)

        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * ffn_multiplier),
            nn.GELU(),
            nn.Linear(channels * ffn_multiplier, channels),
        )

        self.attn_scale = nn.Parameter(torch.full((channels,), residual_scale_init))
        self.ffn_scale = nn.Parameter(torch.full((channels,), residual_scale_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        is_tracing = cast("Callable[[], bool]", torch.jit.is_tracing)  # type: ignore[attr-defined]
        if not is_tracing():
            if H != self.board_size or W != self.board_size:
                raise ValueError(
                    f"expected spatial shape ({self.board_size}, {self.board_size}), got ({H}, {W})"
                )
        N = H * W

        # Flatten spatial dimensions: [B, C, H, W] -> [B, N, C]
        # reshape (not view) so non-contiguous inputs are accepted
        x_flat = x.permute(0, 2, 3, 1).reshape(B, N, C)

        # Self-Attention Branch
        norm_x = self.norm1(x_flat)
        qkv = self.qkv_proj(norm_x)  # [B, N, 3 * C]
        q, k, v = qkv.chunk(3, dim=-1)  # Each [B, N, C]

        # Reshape to [B, num_heads, N, head_dim]
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute scaled dot-product attention scores
        # q: [B, num_heads, N, head_dim], k^T: [B, num_heads, head_dim, N]
        # scores: [B, num_heads, N, N]
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Add full 2D relative position bias
        # bias: [1, num_heads, N, N]
        scores = scores + self.relative_bias()

        attn_weights = torch.softmax(scores, dim=-1)

        # Compute context vector
        # context: [B, num_heads, N, head_dim]
        context = torch.matmul(attn_weights, v)
        # Reshape back to [B, N, C]
        context = context.transpose(1, 2).contiguous().view(B, N, C)

        attn_out = cast(torch.Tensor, self.out_proj(context))

        # Apply LayerScale and residual connection
        x_flat = x_flat + self.attn_scale * attn_out

        # FFN Branch
        ffn_out = cast(torch.Tensor, self.ffn(self.norm2(x_flat)))
        x_flat = x_flat + self.ffn_scale * ffn_out

        # Reshape back to [B, C, H, W]
        x_out = x_flat.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return x_out


class PolicyValueNetwork(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.stem = nn.Sequential(
            nn.Conv2d(config.input_channels, config.channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(config.channels),
            nn.ReLU(inplace=True),
        )
        self.backbone = nn.ModuleList(
            [ResidualBlock(config.channels) for _ in range(config.residual_blocks)]
        )
        self.attention = nn.ModuleList(
            [
                BoardSelfAttentionBlock(
                    channels=config.channels,
                    num_heads=config.attention_heads,
                    ffn_multiplier=config.attention_ffn_multiplier,
                    residual_scale_init=config.attention_residual_scale_init,
                )
                for _ in range(config.attention_blocks)
            ]
        )
        self.stage_plan = build_stage_plan(
            config.residual_blocks,
            config.attention_blocks,
            config.attention_insert_every,
        )
        self.policy_spatial = nn.Sequential(
            nn.Conv2d(
                config.channels,
                config.policy_channels,
                kernel_size=config.policy_kernel_size,
                padding=config.policy_kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm2d(config.policy_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(config.policy_channels, 1, kernel_size=1),
        )
        self.policy_pass = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(config.channels, 1),
        )
        self.q_spatial: nn.Sequential | None = None
        self.q_pass: nn.Sequential | None = None
        if config.action_value_head:
            self.q_spatial = nn.Sequential(
                nn.Conv2d(
                    config.channels,
                    config.policy_channels,
                    kernel_size=config.policy_kernel_size,
                    padding=config.policy_kernel_size // 2,
                    bias=False,
                ),
                nn.BatchNorm2d(config.policy_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(config.policy_channels, 1, kernel_size=1),
                nn.Tanh(),
            )
            self.q_pass = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(config.channels, 1),
                nn.Tanh(),
            )

        self.value_head: nn.Module | None = None
        if not config.action_value_head:
            if config.spatial_value_head:
                self.value_head = nn.Sequential(
                    nn.Conv2d(
                        config.channels, config.value_spatial_channels, kernel_size=1, bias=False
                    ),
                    nn.BatchNorm2d(config.value_spatial_channels),
                    nn.ReLU(inplace=True),
                    nn.Flatten(),
                    nn.Linear(
                        config.value_spatial_channels * BOARD_SIZE * BOARD_SIZE, config.value_hidden
                    ),
                    nn.ReLU(inplace=True),
                    nn.Linear(config.value_hidden, 1),
                    nn.Tanh(),
                )

            else:
                self.value_head = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(config.channels, config.value_hidden),
                    nn.ReLU(inplace=True),
                    nn.Linear(config.value_hidden, 1),
                    nn.Tanh(),
                )
        self.terminal_board_head: nn.Module | None = None
        if config.terminal_board_head:
            self.terminal_board_head = nn.Sequential(
                nn.Conv2d(config.channels, config.terminal_board_hidden_channels, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    config.terminal_board_hidden_channels,
                    config.terminal_board_classes,
                    kernel_size=1,
                ),
            )

    @property
    def has_terminal_board_head(self) -> bool:
        return self.terminal_board_head is not None

    @property
    def has_action_value_head(self) -> bool:
        return self.config.action_value_head

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        features = cast(torch.Tensor, self.stem(x))
        for kind, index in self.stage_plan:
            if kind == ATTENTION_STAGE:
                features = cast(torch.Tensor, self.attention[index](features))
            else:
                features = cast(torch.Tensor, self.backbone[index](features))
        return features

    def _policy_logits(self, features: torch.Tensor) -> torch.Tensor:
        board_logits = self.policy_spatial(features).flatten(start_dim=1)
        pass_logits = self.policy_pass(features)
        return torch.cat([board_logits, pass_logits], dim=1)

    def _action_values(self, features: torch.Tensor) -> torch.Tensor:
        if self.q_spatial is None or self.q_pass is None:
            raise ValueError("model does not have an action value head")
        board_values = self.q_spatial(features).flatten(start_dim=1)
        pass_values = self.q_pass(features)
        return torch.cat([board_values, pass_values], dim=1)

    def _policy_and_value(
        self,
        features: torch.Tensor,
        inputs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        policy_logits = self._policy_logits(features)
        if self.config.action_value_head:
            if inputs is None:
                raise ValueError("action value head requires the input features for legal masking")
            q_values = self._action_values(features)
            legal_mask = legal_mask_from_features(inputs)
            value = masked_state_value(policy_logits, q_values, legal_mask)
            return policy_logits, value
        if self.value_head is None:
            raise ValueError("model does not have a value head")
        value = self.value_head(features).squeeze(-1)
        return policy_logits, value

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        is_tracing = cast("Callable[[], bool]", torch.jit.is_tracing)  # type: ignore[attr-defined]
        if not is_tracing():
            self._validate_input_shape(x)

        features = self.forward_features(x)
        return self._policy_and_value(features, x)

    def forward_q(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return policy logits and per-action Q values with one backbone pass."""
        if not self.config.action_value_head:
            raise ValueError("model does not have an action value head")
        is_tracing = cast("Callable[[], bool]", torch.jit.is_tracing)  # type: ignore[attr-defined]
        if not is_tracing():
            self._validate_input_shape(x)

        features = self.forward_features(x)
        return self._policy_logits(features), self._action_values(features)

    def forward_with_aux(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return policy/value plus terminal board logits with one backbone pass."""
        if self.terminal_board_head is None:
            raise ValueError("model does not have a terminal board head")
        is_tracing = cast("Callable[[], bool]", torch.jit.is_tracing)  # type: ignore[attr-defined]
        if not is_tracing():
            self._validate_input_shape(x)

        features = self.forward_features(x)
        policy_logits, value = self._policy_and_value(features, x)
        aux_logits = cast(torch.Tensor, self.terminal_board_head(features))
        return policy_logits, value, aux_logits

    def _validate_input_shape(self, x: torch.Tensor) -> None:
        if x.ndim != 4:
            raise ValueError(f"expected input rank 4 [batch, channels, 9, 9], got {x.ndim}")
        if x.shape[1:] != (self.config.input_channels, BOARD_SIZE, BOARD_SIZE):
            raise ValueError(
                "expected input shape "
                f"[batch, {self.config.input_channels}, {BOARD_SIZE}, {BOARD_SIZE}], "
                f"got {tuple(x.shape)}"
            )


def create_model(preset: str = "small", **overrides: int) -> PolicyValueNetwork:
    try:
        config = MODEL_PRESETS[preset]
    except KeyError as exc:
        choices = ", ".join(sorted(MODEL_PRESETS))
        raise ValueError(f"unknown model preset {preset!r}; choose one of: {choices}") from exc

    if overrides:
        config = ModelConfig(**{**config.__dict__, **overrides})
    return PolicyValueNetwork(config)


__all__ = [
    "ACTION_SPACE",
    "MODEL_PRESETS",
    "ModelConfig",
    "PolicyValueNetwork",
    "BoardSelfAttentionBlock",
    "D4RelativePositionBias",
    "build_stage_plan",
    "create_model",
]
