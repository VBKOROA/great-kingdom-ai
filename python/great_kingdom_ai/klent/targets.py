"""KLENT closed-form targets and shared tensor helpers.

The helpers in this module are consumed by the model, ONNX export, and
self-play so that masking rules stay identical everywhere.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

from great_kingdom_ai.features import (
    BOARD_CELLS,
    BOARD_SIZE,
    LEGAL_PLACE_FEATURE_CHANNEL,
)
from great_kingdom_ai.klent._torch import _import_torch
from great_kingdom_ai.klent.types import validate_alpha_beta, validate_lambda_gamma

if TYPE_CHECKING:
    from torch import Tensor


def compute_analytical_policy(
    policy_logits: Tensor,
    q_values: Tensor,
    legal_mask: Tensor,
    *,
    alpha: float,
    beta: float,
) -> Tensor:
    """Compute ``pi'(a|s) = softmax((Q + beta * log pi) / (alpha + beta))``.

    Only legal actions are normalized. The result is a detached target tensor:
    callers must not backpropagate through it, and illegal actions are exactly
    zero.
    """
    torch = _import_torch()
    validate_alpha_beta(alpha, beta)
    _validate_action_tensors(torch, policy_logits, q_values, legal_mask)

    with torch.no_grad():
        logits = policy_logits.detach().to(dtype=torch.float32)
        q = q_values.detach().to(dtype=torch.float32)
        mask = legal_mask.detach().to(dtype=torch.bool)
        if not _has_legal_action(torch, mask):
            raise ValueError("legal_mask must contain at least one legal action per row")

        masked_logits = logits.masked_fill(~mask, torch.finfo(torch.float32).min)
        log_policy = torch.log_softmax(masked_logits, dim=1)
        safe_log_policy = torch.where(mask, log_policy, torch.zeros_like(log_policy))
        scores = (q + float(beta) * safe_log_policy) / (float(alpha) + float(beta))
        scores = scores.masked_fill(~mask, float("-inf"))
        probabilities = torch.softmax(scores, dim=1)
        probabilities = probabilities.masked_fill(~mask, 0.0)
    return cast("Tensor", probabilities)


def compute_lambda_returns(
    rewards: Tensor | Sequence[float],
    state_values: Tensor | Sequence[float],
    players: Tensor | Sequence[int],
    *,
    lambda_param: float,
    gamma: float = 1.0,
) -> Tensor:
    """Compute backward TD(lambda) returns for one finished two-player game.

    ``T`` is the index of the last move and every input has length ``T + 1``.
    ``rewards[t]`` is the reward for taking ``A_t`` in ``S_t`` from the
    perspective of ``players[t]``: zero for non-terminal moves and +/-1 on the
    final move. ``state_values[t]`` estimates ``E[Q(S_t, A)]`` under the
    analytical policy and is only needed up to ``t = T``; the value of the
    terminal successor ``S_{T+1}`` is never used.
    """
    torch = _import_torch()
    validate_lambda_gamma(lambda_param, gamma)
    reward_list = _float_list(torch, rewards)
    value_list = _float_list(torch, state_values)
    player_list = _int_list(torch, players)

    length = len(reward_list)
    if length < 1:
        raise ValueError("lambda returns require at least one transition")
    if len(value_list) != length:
        raise ValueError("state_values length must match rewards length")
    if len(player_list) != length:
        raise ValueError("players length must match rewards length")

    for index, player in enumerate(player_list):
        if player not in {1, 2}:
            raise ValueError(f"players[{index}] must be 1 or 2, got {player}")
    for index, reward in enumerate(reward_list):
        if not math.isfinite(reward) or reward < -1.0 or reward > 1.0:
            raise ValueError(f"rewards[{index}] must be finite and in [-1, 1], got {reward}")
    for index, value in enumerate(value_list):
        if not math.isfinite(value) or value < -1.0 or value > 1.0:
            raise ValueError(f"state_values[{index}] must be finite and in [-1, 1], got {value}")

    if any(reward != 0.0 for reward in reward_list[:-1]):
        raise ValueError("non-terminal rewards must be zero")
    if reward_list[-1] not in {-1.0, 1.0}:
        raise ValueError("the final reward must be +1 or -1 for a finished game")

    returns = [0.0] * length
    returns[-1] = reward_list[-1]
    for index in range(length - 2, -1, -1):
        sign = 1.0 if player_list[index] == player_list[index + 1] else -1.0
        continuation = (
            (1.0 - float(lambda_param)) * value_list[index + 1]
            + float(lambda_param) * returns[index + 1]
        )
        returns[index] = reward_list[index] + float(gamma) * sign * continuation

    for index, value in enumerate(returns):
        if not math.isfinite(value) or value < -1.0 - 1e-6 or value > 1.0 + 1e-6:
            raise ValueError(f"lambda return at index {index} is out of range: {value}")

    device = getattr(rewards, "device", None)
    return cast("Tensor", torch.tensor(returns, dtype=torch.float32, device=device))


def legal_mask_from_features(features: Tensor) -> Tensor:
    """Reconstruct the legal action mask from feature planes (channel 4 + pass)."""
    torch = _import_torch()
    if features.ndim != 4:
        raise ValueError(f"expected features rank 4 [batch, channels, 9, 9], got {features.ndim}")
    if features.shape[2:] != (BOARD_SIZE, BOARD_SIZE):
        raise ValueError(
            f"expected features spatial shape {(BOARD_SIZE, BOARD_SIZE)}, "
            f"got {tuple(features.shape[2:])}"
        )
    legal_place = features[:, LEGAL_PLACE_FEATURE_CHANNEL].flatten(start_dim=1) > 0.5
    if legal_place.shape[1] != BOARD_CELLS:
        raise ValueError(
            f"expected {BOARD_CELLS} board cells in feature channel "
            f"{LEGAL_PLACE_FEATURE_CHANNEL}, got {legal_place.shape[1]}"
        )
    pass_column = torch.ones_like(legal_place[:, :1])
    return cast("Tensor", torch.cat([legal_place, pass_column], dim=1))


def masked_state_value(
    policy_logits: Tensor,
    q_values: Tensor,
    legal_mask: Tensor,
) -> Tensor:
    """Return ``V(s) = sum_a softmax(pi_masked)_a * Q(s, a)`` as a rank-1 tensor."""
    torch = _import_torch()
    _validate_action_tensors(torch, policy_logits, q_values, legal_mask)
    if not _has_legal_action(torch, legal_mask):
        raise ValueError("legal_mask must contain at least one legal action per row")

    logits = policy_logits.to(dtype=torch.float32)
    q = q_values.to(dtype=torch.float32)
    mask = legal_mask.to(dtype=torch.bool)
    masked_logits = logits.masked_fill(~mask, torch.finfo(torch.float32).min)
    probabilities = torch.softmax(masked_logits, dim=1)
    probabilities = probabilities * mask.to(dtype=probabilities.dtype)
    return cast("Tensor", (probabilities * q).sum(dim=1))


def _validate_action_tensors(
    torch: Any,
    policy_logits: Tensor,
    q_values: Tensor,
    legal_mask: Tensor,
) -> None:
    if policy_logits.ndim != 2 or q_values.ndim != 2 or legal_mask.ndim != 2:
        raise ValueError("policy_logits, q_values, and legal_mask must be rank 2 [batch, actions]")
    if policy_logits.shape != q_values.shape or policy_logits.shape != legal_mask.shape:
        raise ValueError(
            "policy_logits, q_values, and legal_mask must share the same shape, got "
            f"{tuple(policy_logits.shape)}, {tuple(q_values.shape)}, "
            f"{tuple(legal_mask.shape)}"
        )
    if policy_logits.shape[1] == 0:
        raise ValueError("action dimension must be non-empty")
    del torch


def _has_legal_action(torch: Any, legal_mask: Tensor) -> bool:
    if _is_tracing(torch):
        return True
    return bool(legal_mask.to(dtype=torch.bool).any(dim=1).all())


def _is_tracing(torch: Any) -> bool:
    is_tracing = getattr(torch.jit, "is_tracing", None)
    return bool(is_tracing()) if is_tracing is not None else False


def _float_list(torch: Any, values: Tensor | Sequence[float]) -> list[float]:
    tensor = torch.as_tensor(values, dtype=torch.float32).detach().reshape(-1)
    return [float(value) for value in tensor.cpu()]


def _int_list(torch: Any, values: Tensor | Sequence[int]) -> list[int]:
    tensor = torch.as_tensor(values).detach().reshape(-1)
    return [int(value) for value in tensor.cpu()]


__all__ = [
    "compute_analytical_policy",
    "compute_lambda_returns",
    "legal_mask_from_features",
    "masked_state_value",
]