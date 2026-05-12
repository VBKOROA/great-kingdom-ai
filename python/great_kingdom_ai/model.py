"""PyTorch policy-value network for Great Kingdom AlphaZero-lite."""

from __future__ import annotations

from dataclasses import dataclass

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
        return self.activation(x + self.block(x))


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
                nn.Flatten(),
                nn.Linear(config.channels * BOARD_SIZE * BOARD_SIZE, config.value_hidden),
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
        if not torch.jit.is_tracing():
            self._validate_input_shape(x)

        features = self.backbone(self.stem(x))
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
    "create_model",
]
