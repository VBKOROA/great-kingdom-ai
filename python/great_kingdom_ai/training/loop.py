"""Core training loop and loss computation."""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from great_kingdom_ai.training.batch import (
    ReplayDataset,
    TrainingBatch,
    _iter_training_batches,
)
from great_kingdom_ai.training.checkpoint import (
    TrainState,
    _update_ema_model,
    create_train_state,
    load_checkpoint,
    load_checkpoint_weights,
    save_checkpoint,
)
from great_kingdom_ai.training.config import TrainingConfig
from great_kingdom_ai.training.torch_utils import (
    _autocast_context,
    _cuda_amp_enabled,
    _import_torch,
)

if TYPE_CHECKING:
    import torch
    from torch import Tensor, nn

@dataclass(frozen=True)
class LossBreakdown:
    policy: torch.Tensor
    value: torch.Tensor
    regularization: torch.Tensor
    policy_entropy: torch.Tensor
    policy_kl: torch.Tensor
    total: torch.Tensor
    per_sample_policy_kl: torch.Tensor
    per_sample_value_abs_error: torch.Tensor

    def to_float_dict(self) -> dict[str, float]:
        return {
            "policy": float(self.policy.detach().cpu()),
            "value": float(self.value.detach().cpu()),
            "regularization": float(self.regularization.detach().cpu()),
            "policy_entropy": float(self.policy_entropy.detach().cpu()),
            "policy_kl": float(self.policy_kl.detach().cpu()),
            "total": float(self.total.detach().cpu()),
        }


@dataclass(frozen=True)
class TrainSummary:
    start_step: int
    end_step: int
    checkpoint_path: Path | None
    losses: list[dict[str, float]]


def compute_losses(
    model: nn.Module,
    batch: TrainingBatch,
    *,
    policy_loss_weight: float = 1.0,
    value_loss_weight: float = 1.0,
    l2_loss_weight: float = 0.0,
    mask_policy_loss: bool = True,
) -> LossBreakdown:
    """Compute AlphaZero policy cross-entropy, value MSE, and optional L2 loss."""
    torch = _import_torch()
    if policy_loss_weight < 0.0 or value_loss_weight < 0.0 or l2_loss_weight < 0.0:
        raise ValueError("loss weights must be non-negative")

    policy_logits, value = model(batch.features)
    if mask_policy_loss:
        policy_logits = policy_logits.masked_fill(
            ~batch.legal_mask,
            torch.finfo(policy_logits.dtype).min,
        )
        _validate_policy_targets_match_legal_mask(batch)
    _validate_sample_weight(batch)
    log_policy = torch.log_softmax(policy_logits, dim=1)
    per_sample_policy_loss = -(batch.policy * log_policy).sum(dim=1)
    per_sample_policy_entropy = _policy_target_entropy_rows(batch.policy)
    per_sample_policy_kl = torch.clamp(
        per_sample_policy_loss - per_sample_policy_entropy,
        min=0.0,
    )
    policy_loss = _weighted_mean(per_sample_policy_loss, batch.sample_weight)
    policy_entropy = _weighted_mean(per_sample_policy_entropy, batch.sample_weight)
    policy_kl = policy_loss - policy_entropy
    per_sample_value_error = value - batch.value
    value_loss = _weighted_mean(per_sample_value_error.pow(2), batch.sample_weight)
    regularization = _l2_regularization(model) * l2_loss_weight
    total = policy_loss_weight * policy_loss + value_loss_weight * value_loss + regularization
    return LossBreakdown(
        policy=policy_loss,
        value=value_loss,
        regularization=regularization,
        policy_entropy=policy_entropy,
        policy_kl=policy_kl,
        total=total,
        per_sample_policy_kl=per_sample_policy_kl,
        per_sample_value_abs_error=per_sample_value_error.abs(),
    )

def train_step(state: TrainState, batch: TrainingBatch, config: TrainingConfig) -> LossBreakdown:
    torch = _import_torch()
    state.model.train()
    state.optimizer.zero_grad(set_to_none=True)
    amp_enabled = _amp_enabled(config)
    with _autocast_context(torch, enabled=amp_enabled):
        losses = compute_losses(
            state.model,
            batch,
            policy_loss_weight=config.policy_loss_weight,
            value_loss_weight=config.value_loss_weight,
            l2_loss_weight=config.l2_loss_weight,
            mask_policy_loss=config.mask_policy_loss,
        )
    optimizer_stepped = True
    if amp_enabled and state.scaler is not None:
        scale_before = float(state.scaler.get_scale())
        state.scaler.scale(losses.total).backward()
        if config.gradient_clip_norm is not None:
            state.scaler.unscale_(state.optimizer)
            torch.nn.utils.clip_grad_norm_(state.model.parameters(), config.gradient_clip_norm)
        state.scaler.step(state.optimizer)
        state.scaler.update()
        optimizer_stepped = float(state.scaler.get_scale()) >= scale_before
    else:
        losses.total.backward()  # type: ignore[no-untyped-call]
        if config.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(state.model.parameters(), config.gradient_clip_norm)
        state.optimizer.step()
    if optimizer_stepped:
        state.scheduler.step()
        _update_ema_model(state)
    return losses

def train_from_replay(
    replay: ReplayDataset,
    config: TrainingConfig,
    *,
    checkpoint_path: str | Path | None = None,
    resume_path: str | Path | None = None,
    bootstrap_weights_path: str | Path | None = None,
    resume_optimizer_lr_override: float | None = None,
    log_every: int = 0,
    progress_callback: Callable[[int, int, dict[str, float]], None] | None = None,
) -> TrainSummary:
    if len(replay) < config.batch_size:
        raise ValueError("replay buffer must contain at least batch_size samples")
    if config.prefetch_batches < 0:
        raise ValueError("prefetch_batches must be non-negative")
    if resume_path is not None and bootstrap_weights_path is not None:
        raise ValueError("resume_path and bootstrap_weights_path are mutually exclusive")
    if resume_optimizer_lr_override is not None and resume_path is None:
        raise ValueError("resume_optimizer_lr_override requires resume_path")

    torch = _import_torch()
    torch.manual_seed(config.seed)
    rng = random.Random(config.seed)

    if bootstrap_weights_path is not None:
        state = load_checkpoint_weights(bootstrap_weights_path, config)
    elif resume_path is None:
        state = create_train_state(config)
    else:
        state = load_checkpoint(
            resume_path,
            device=config.device,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            lr_schedule=config.lr_schedule,
            lr_decay_steps=config.lr_decay_steps,
            lr_decay_gamma=config.lr_decay_gamma,
            lr_warmup_steps=config.lr_warmup_steps,
            lr_min_factor=config.lr_min_factor,
            lr_cosine_steps=config.lr_cosine_steps,
            steps=config.steps,
            optimizer=config.optimizer,
            momentum=config.momentum,
            nesterov=config.nesterov,
            amp=config.amp,
            ema_decay=config.ema_decay,
            optimizer_lr_override=resume_optimizer_lr_override,
        )

    start_step = state.step
    losses: list[dict[str, float]] = []
    for step, batch in zip(
        range(start_step, start_step + config.steps),
        _iter_training_batches(replay, config, rng, torch=torch),
        strict=True,
    ):
        train_step_start = time.perf_counter()
        loss = train_step(state, batch, config)
        train_step_seconds = time.perf_counter() - train_step_start
        _update_batch_local_priorities(replay, batch, loss, config)
        state = TrainState(
            model=state.model,
            optimizer=state.optimizer,
            scheduler=state.scheduler,
            scaler=state.scaler,
            ema_model=state.ema_model,
            ema_decay=state.ema_decay,
            step=step + 1,
            model_preset=state.model_preset,
        )
        if log_every > 0:
            should_log = state.step == start_step + config.steps or state.step % log_every == 0
        else:
            should_log = False
        if should_log:
            loss_values = loss.to_float_dict()
            loss_values["train_step_seconds"] = train_step_seconds
            losses.append({"step": float(state.step), **loss_values})
            if progress_callback is not None:
                progress_callback(state.step - start_step, config.steps, loss_values)

    saved_path = save_checkpoint(state, checkpoint_path) if checkpoint_path is not None else None
    return TrainSummary(
        start_step=start_step,
        end_step=state.step,
        checkpoint_path=saved_path,
        losses=losses,
    )

def _l2_regularization(model: nn.Module) -> torch.Tensor:
    torch = _import_torch()
    parameters = [
        parameter.pow(2).sum()
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    if not parameters:
        return cast("Tensor", torch.tensor(0.0))
    return cast("Tensor", torch.stack(parameters).sum())


def _validate_policy_targets_match_legal_mask(batch: TrainingBatch) -> None:
    illegal_target_mass = batch.policy.masked_select(~batch.legal_mask).sum()
    if float(illegal_target_mass.detach().cpu()) > 1e-5:
        raise ValueError("policy target assigns probability to illegal actions")


def _validate_sample_weight(batch: TrainingBatch) -> None:
    torch = _import_torch()
    if batch.sample_weight.shape != batch.value.shape:
        raise ValueError("sample_weight shape must match value target shape")
    if not bool(torch.isfinite(batch.sample_weight).all()):
        raise ValueError("sample_weight must be finite")
    if float(batch.sample_weight.min().detach().cpu()) <= 0.0:
        raise ValueError("sample_weight must be positive")


def _policy_target_entropy(policy: torch.Tensor, sample_weight: torch.Tensor) -> torch.Tensor:
    return _weighted_mean(_policy_target_entropy_rows(policy), sample_weight)


def _policy_target_entropy_rows(policy: torch.Tensor) -> torch.Tensor:
    torch = _import_torch()
    positive = policy > 0.0
    return -(torch.where(positive, policy * torch.log(policy.clamp_min(1e-45)), 0.0)).sum(dim=1)


def _weighted_mean(values: torch.Tensor, sample_weight: torch.Tensor) -> torch.Tensor:
    weights = sample_weight.to(dtype=values.dtype)
    return (values * weights).sum() / weights.sum()


def _amp_enabled(config: TrainingConfig) -> bool:
    torch = _import_torch()
    return _cuda_amp_enabled(torch, config.device, enabled=config.amp)


def _update_batch_local_priorities(
    replay: ReplayDataset,
    batch: TrainingBatch,
    losses: LossBreakdown,
    config: TrainingConfig,
) -> None:
    if not config.priority_enabled or batch.replay_indexes is None:
        return
    updater = getattr(replay, "update_sampling_priorities", None)
    if updater is None:
        return
    priorities = (
        config.priority_epsilon
        + config.priority_policy_kl_weight
        * losses.per_sample_policy_kl.detach().float().cpu().numpy()
        + config.priority_value_error_weight
        * losses.per_sample_value_abs_error.detach().float().cpu().numpy()
    )
    updater(
        batch.replay_indexes,
        priorities.astype("float32", copy=False),
        ema=config.priority_ema,
        epsilon=config.priority_epsilon,
        max_priority=config.priority_max_priority,
    )


__all__ = [
    "LossBreakdown",
    "TrainSummary",
    "compute_losses",
    "train_from_replay",
    "train_step",
]
