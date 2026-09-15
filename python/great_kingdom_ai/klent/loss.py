"""KLENT policy cross-entropy and selected-action Q-value loss."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from great_kingdom_ai.klent._torch import _import_torch
from great_kingdom_ai.klent.types import KlentConfig

if TYPE_CHECKING:
    from torch import Tensor, nn

    from great_kingdom_ai.training.batch import TrainingBatch


@dataclass(frozen=True)
class KlentLossBreakdown:
    """Scalar losses plus per-sample diagnostics for priority bookkeeping."""

    policy: Tensor
    q_value: Tensor
    total: Tensor
    per_sample_policy_loss: Tensor
    per_sample_q_error: Tensor

    def to_float_dict(self) -> dict[str, float]:
        return {
            "policy": float(self.policy.detach().cpu()),
            "q_value": float(self.q_value.detach().cpu()),
            "total": float(self.total.detach().cpu()),
        }


def compute_klent_losses(
    model: nn.Module,
    batch: TrainingBatch,
    config: KlentConfig,
    *,
    policy_loss_weight: float = 1.0,
    value_loss_weight: float = 1.0,
) -> KlentLossBreakdown:
    """Compute soft-target cross-entropy and selected-action Q MSE.

    The policy target is the stored analytical policy ``pi'`` and the value
    target is the stored lambda-return. Both targets are detached and all
    aggregations run in FP32.
    """
    torch = _import_torch()
    _validate_loss_weight(policy_loss_weight, "policy_loss_weight")
    _validate_loss_weight(value_loss_weight, "value_loss_weight")
    forward_q = getattr(model, "forward_q", None)
    if forward_q is None:
        raise ValueError("KLENT loss requires a model with forward_q")
    if batch.action is None:
        raise ValueError("KLENT loss requires batch.action (the taken action index)")

    policy_logits, q_values = cast(
        "tuple[Tensor, Tensor]",
        forward_q(batch.features),
    )
    logits = policy_logits.float()
    q = q_values.float()
    targets = batch.policy.detach().float()
    values = batch.value.detach().float()
    weights = batch.sample_weight.detach().float()
    legal_mask = batch.legal_mask.detach().to(device=logits.device, dtype=torch.bool)
    actions = batch.action.detach().to(device=logits.device, dtype=torch.long)

    _validate_shapes(torch, logits, q, targets, values, weights, legal_mask, actions)
    _validate_action_targets(torch, targets, legal_mask)
    _validate_actions(torch, actions, q.shape[1], legal_mask)

    masked_logits = logits.masked_fill(~legal_mask, torch.finfo(logits.dtype).min)
    log_policy = torch.log_softmax(masked_logits, dim=1)
    safe_log_policy = torch.where(legal_mask, log_policy, torch.zeros_like(log_policy))
    per_sample_policy_loss = -(targets * safe_log_policy).sum(dim=1)
    policy_loss = _weighted_mean(per_sample_policy_loss, weights)

    taken_q = q.gather(1, actions.unsqueeze(1)).squeeze(1)
    per_sample_q_error = (taken_q - values).pow(2)
    q_loss = _weighted_mean(per_sample_q_error, weights)

    return KlentLossBreakdown(
        policy=policy_loss,
        q_value=q_loss,
        total=policy_loss_weight * policy_loss + value_loss_weight * q_loss,
        per_sample_policy_loss=per_sample_policy_loss,
        per_sample_q_error=per_sample_q_error,
    )


def _validate_loss_weight(value: float, label: str) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{label} must be finite and non-negative, got {value}")


def _validate_shapes(
    torch: Any,
    logits: Tensor,
    q_values: Tensor,
    targets: Tensor,
    values: Tensor,
    weights: Tensor,
    legal_mask: Tensor,
    actions: Tensor,
) -> None:
    if logits.shape != q_values.shape:
        raise ValueError("policy logits and q values must share the same shape")
    batch_size, action_space = logits.shape
    if targets.shape != logits.shape:
        raise ValueError("policy target shape must match policy logits")
    if legal_mask.shape != logits.shape:
        raise ValueError("legal_mask shape must match policy logits")
    if values.shape != (batch_size,):
        raise ValueError("lambda return target must be rank 1 [batch]")
    if weights.shape != (batch_size,):
        raise ValueError("sample_weight must be rank 1 [batch]")
    if actions.shape != (batch_size,):
        raise ValueError("action must be rank 1 [batch]")
    if action_space == 0:
        raise ValueError("action space must be non-empty")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("lambda return target must be finite")
    if bool((values < -1.0).any()) or bool((values > 1.0).any()):
        raise ValueError("lambda return target must be in [-1, 1]")
    if not bool(torch.isfinite(weights).all()) or float(weights.min()) <= 0.0:
        raise ValueError("sample_weight must be finite and positive")


def _validate_action_targets(torch: Any, targets: Tensor, legal_mask: Tensor) -> None:
    illegal_mass = targets.masked_select(~legal_mask).sum()
    if float(illegal_mass.detach()) > 1e-5:
        raise ValueError("policy target assigns probability to illegal actions")
    target_sums = targets.sum(dim=1)
    if bool((target_sums <= 0.0).any()):
        raise ValueError("policy target must contain at least one positive probability")


def _validate_actions(
    torch: Any,
    actions: Tensor,
    action_space: int,
    legal_mask: Tensor,
) -> None:
    if bool((actions < 0).any()) or bool((actions >= action_space).any()):
        raise ValueError(f"action must be in [0, {action_space})")
    taken_legal = legal_mask.gather(1, actions.unsqueeze(1)).squeeze(1)
    if not bool(taken_legal.all()):
        raise ValueError("the taken action must be legal")


def _weighted_mean(values: Tensor, sample_weight: Tensor) -> Tensor:
    weights = sample_weight.to(dtype=values.dtype)
    return (values * weights).sum() / weights.sum()


__all__ = ["KlentLossBreakdown", "compute_klent_losses"]