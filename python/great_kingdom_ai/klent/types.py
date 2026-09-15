"""KLENT configuration types and coefficient validation."""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from torch import Tensor


class KlentPolicyValueModel(Protocol):
    """Structural boundary for models consumed by KLENT self-play and fitting."""

    training: bool

    def forward_q(self, features: Tensor) -> tuple[Tensor, Tensor]: ...

    def parameters(self) -> Iterator[Tensor]: ...

    def eval(self) -> Any: ...

    def train(self, mode: bool = True) -> Any: ...


def validate_alpha_beta(alpha: float, beta: float) -> None:
    """Validate KLENT regularization coefficients."""
    if not math.isfinite(alpha) or alpha < 0.0:
        raise ValueError(f"alpha must be finite and non-negative, got {alpha}")
    if not math.isfinite(beta) or beta < 0.0:
        raise ValueError(f"beta must be finite and non-negative, got {beta}")
    if alpha + beta <= 0.0:
        raise ValueError("alpha + beta must be positive")


def validate_lambda_gamma(lambda_param: float, gamma: float) -> None:
    """Validate the lambda-return coefficients."""
    if not math.isfinite(lambda_param) or not 0.0 <= lambda_param <= 1.0:
        raise ValueError(f"lambda_param must be finite and in [0, 1], got {lambda_param}")
    if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must be finite and in [0, 1], got {gamma}")


@dataclass(frozen=True)
class KlentConfig:
    """KLENT target and loss coefficients."""

    alpha: float = 0.03
    beta: float = 0.1
    lambda_param: float = 0.8825
    gamma: float = 1.0

    def __post_init__(self) -> None:
        validate_alpha_beta(self.alpha, self.beta)
        validate_lambda_gamma(self.lambda_param, self.gamma)


__all__ = [
    "KlentConfig",
    "KlentPolicyValueModel",
    "validate_alpha_beta",
    "validate_lambda_gamma",
]