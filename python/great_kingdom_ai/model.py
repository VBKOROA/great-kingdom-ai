"""PyTorch policy-value network for Great Kingdom AlphaZero-lite."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import torch
from torch import nn

from great_kingdom_ai.features import ACTION_SPACE, BOARD_SIZE, FEATURE_CHANNELS


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
    attention_residual_scale_init: float = 1e-3



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


class Full2DRelativePositionBias(nn.Module):
    def __init__(self, num_heads: int, board_size: int = 9) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.board_size = board_size
        self.num_positions = board_size * board_size

        self.max_relative_position = 2 * board_size - 1
        self.num_relative_positions = self.max_relative_position * self.max_relative_position

        self.relative_bias_table = nn.Parameter(
            torch.zeros(self.num_relative_positions, num_heads)
        )

        relative_index = self._compute_relative_index()
        self.register_buffer("relative_index", relative_index)

    def _compute_relative_index(self) -> torch.Tensor:
        coords = torch.arange(self.num_positions)
        r = coords // self.board_size
        c = coords % self.board_size

        dr = r.unsqueeze(0) - r.unsqueeze(1)
        dc = c.unsqueeze(0) - c.unsqueeze(1)

        dr_idx = dr + (self.board_size - 1)
        dc_idx = dc + (self.board_size - 1)

        relative_index = dr_idx * self.max_relative_position + dc_idx
        return relative_index

    def forward(self) -> torch.Tensor:
        bias = self.relative_bias_table[self.relative_index]  # [81, 81, num_heads]
        bias = bias.permute(2, 0, 1).unsqueeze(0)  # [1, num_heads, 81, 81]
        return bias


class BoardSelfAttentionBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        ffn_multiplier: int = 4,
        residual_scale_init: float = 1e-3,
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

        self.relative_bias = Full2DRelativePositionBias(num_heads, board_size=board_size)

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
        x_flat = x.permute(0, 2, 3, 1).view(B, N, C)

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

        attn_out = self.out_proj(context)

        # Apply LayerScale and residual connection
        x_flat = x_flat + self.attn_scale * attn_out

        # FFN Branch
        ffn_out = self.ffn(self.norm2(x_flat))
        x_flat = x_flat + self.ffn_scale * ffn_out

        # Reshape back to [B, C, H, W]
        x_out = x_flat.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
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
        self.backbone = nn.Sequential(
            *[ResidualBlock(config.channels) for _ in range(config.residual_blocks)]
        )
        self.attention = nn.Sequential(
            *[
                BoardSelfAttentionBlock(
                    channels=config.channels,
                    num_heads=config.attention_heads,
                    ffn_multiplier=config.attention_ffn_multiplier,
                    residual_scale_init=config.attention_residual_scale_init,
                )
                for _ in range(config.attention_blocks)
            ]
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
        if config.spatial_value_head:
            self.value_head = nn.Sequential(
                nn.Conv2d(config.channels, config.value_spatial_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(config.value_spatial_channels),
                nn.ReLU(inplace=True),
                nn.Flatten(),
                nn.Linear(config.value_spatial_channels * BOARD_SIZE * BOARD_SIZE, config.value_hidden),
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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        is_tracing = cast("Callable[[], bool]", torch.jit.is_tracing)  # type: ignore[attr-defined]
        if not is_tracing():
            self._validate_input_shape(x)

        features = self.backbone(self.stem(x))
        features = self.attention(features)
        board_logits = self.policy_spatial(features).flatten(start_dim=1)
        pass_logits = self.policy_pass(features)
        policy_logits = torch.cat([board_logits, pass_logits], dim=1)
        value = self.value_head(features).squeeze(-1)
        return policy_logits, value

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
    "Full2DRelativePositionBias",
    "create_model",
]
