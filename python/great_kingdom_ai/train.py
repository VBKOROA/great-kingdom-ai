"""Training loop and checkpoint helpers for AlphaZero-lite models."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn, Protocol, cast

import numpy as np

from great_kingdom_ai.augmentation import (
    augment_samples_randomly,
    augment_training_arrays_randomly,
)
from great_kingdom_ai.features import BOARD_CELLS, LEGAL_PLACE_FEATURE_CHANNEL, PASS_ACTION
from great_kingdom_ai.learner_prefetch import PrefetchIterator
from great_kingdom_ai.priority_sampling import PrioritySamplingConfig
from great_kingdom_ai.replay_buffer import ReplaySample

if TYPE_CHECKING:
    import torch
    from torch import Tensor, nn
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler

    from great_kingdom_ai.model import PolicyValueNetwork


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 32
    steps: int = 1000
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    value_loss_weight: float = 1.0
    policy_loss_weight: float = 1.0
    l2_loss_weight: float = 0.0
    lr_schedule: str = "step"
    lr_decay_gamma: float = 0.99
    lr_decay_steps: int = 100
    lr_warmup_steps: int = 0
    lr_min_factor: float = 0.1
    lr_cosine_steps: int = 0
    seed: int = 0
    device: str = "cpu"
    model_preset: str = "small"
    symmetry_augmentation: bool = True
    mask_policy_loss: bool = True
    amp: bool = False
    recent_sample_fraction: float = 0.0
    recent_sample_window: int = 0
    priority_enabled: bool = False
    priority_alpha: float = 0.6
    priority_beta: float = 0.4
    priority_value_error_weight: float = 1.0
    priority_policy_kl_weight: float = 1.0
    priority_target_age_weight: float = 0.25
    priority_search_reanalyzed_boost: float = 1.0
    priority_max_priority: float | None = 64.0
    prefetch_batches: int = 1
    ema_decay: float | None = None


@dataclass(frozen=True)
class TrainingBatch:
    features: torch.Tensor
    policy: torch.Tensor
    value: torch.Tensor
    legal_mask: torch.Tensor
    sample_weight: torch.Tensor


@dataclass(frozen=True)
class TrainingArrays:
    features: np.ndarray
    policies: np.ndarray
    values: np.ndarray
    sample_weights: np.ndarray
    legal_masks: np.ndarray | None = None


@dataclass(frozen=True)
class LossBreakdown:
    policy: torch.Tensor
    value: torch.Tensor
    regularization: torch.Tensor
    policy_entropy: torch.Tensor
    policy_kl: torch.Tensor
    total: torch.Tensor

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
class TrainState:
    model: PolicyValueNetwork
    optimizer: Optimizer
    scheduler: LRScheduler
    scaler: Any | None = None
    ema_model: PolicyValueNetwork | None = None
    ema_decay: float | None = None
    step: int = 0
    model_preset: str = "small"


@dataclass(frozen=True)
class TrainSummary:
    start_step: int
    end_step: int
    checkpoint_path: Path | None
    losses: list[dict[str, float]]


class ReplayDataset(Protocol):
    def __len__(self) -> int: ...

    def sample(self, batch_size: int, rng: random.Random) -> list[ReplaySample]: ...


def samples_to_batch(
    samples: Sequence[ReplaySample],
    *,
    device: torch.device | str | None = None,
    pin_memory: bool = False,
) -> TrainingBatch:
    """Convert replay samples into tensors shaped for the policy-value network."""
    torch = _import_torch()
    if not samples:
        raise ValueError("training batch must contain at least one sample")

    features = np.stack([sample.features for sample in samples], axis=0).astype(np.float32)
    policies = np.stack([sample.policy for sample in samples], axis=0).astype(np.float32)
    values = np.asarray([sample.value for sample in samples], dtype=np.float32)
    sample_weights = np.asarray([sample.sample_weight for sample in samples], dtype=np.float32)
    legal_masks = _legal_masks_from_features(features)

    return TrainingBatch(
        features=_tensor_from_numpy(torch, features, pin_memory=pin_memory).to(device=device),
        policy=_tensor_from_numpy(torch, policies, pin_memory=pin_memory).to(device=device),
        value=_tensor_from_numpy(torch, values, pin_memory=pin_memory).to(device=device),
        legal_mask=_tensor_from_numpy(torch, legal_masks, pin_memory=pin_memory).to(
            device=device
        ),
        sample_weight=_tensor_from_numpy(torch, sample_weights, pin_memory=pin_memory).to(
            device=device
        ),
    )


def arrays_to_batch(
    arrays: TrainingArrays,
    *,
    device: torch.device | str | None = None,
    pin_memory: bool = False,
) -> TrainingBatch:
    torch = _import_torch()
    features = np.ascontiguousarray(arrays.features, dtype=np.float32)
    policies = np.ascontiguousarray(arrays.policies, dtype=np.float32)
    values = np.ascontiguousarray(arrays.values, dtype=np.float32)
    sample_weights = np.ascontiguousarray(arrays.sample_weights, dtype=np.float32)
    if features.shape[0] == 0:
        raise ValueError("training batch must contain at least one sample")
    legal_masks = (
        _legal_masks_from_features(features)
        if arrays.legal_masks is None
        else np.ascontiguousarray(arrays.legal_masks, dtype=np.bool_)
    )
    if legal_masks.shape != policies.shape:
        raise ValueError("legal_masks shape must match policies shape")
    return TrainingBatch(
        features=_tensor_from_numpy(torch, features, pin_memory=pin_memory).to(device=device),
        policy=_tensor_from_numpy(torch, policies, pin_memory=pin_memory).to(device=device),
        value=_tensor_from_numpy(torch, values, pin_memory=pin_memory).to(device=device),
        legal_mask=_tensor_from_numpy(torch, legal_masks, pin_memory=pin_memory).to(
            device=device
        ),
        sample_weight=_tensor_from_numpy(torch, sample_weights, pin_memory=pin_memory).to(
            device=device
        ),
    )


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
    policy_loss = _weighted_mean(-(batch.policy * log_policy).sum(dim=1), batch.sample_weight)
    policy_entropy = _policy_target_entropy(batch.policy, batch.sample_weight)
    policy_kl = policy_loss - policy_entropy
    value_loss = _weighted_mean(
        torch.nn.functional.mse_loss(value, batch.value, reduction="none"),
        batch.sample_weight,
    )
    regularization = _l2_regularization(model) * l2_loss_weight
    total = policy_loss_weight * policy_loss + value_loss_weight * value_loss + regularization
    return LossBreakdown(
        policy=policy_loss,
        value=value_loss,
        regularization=regularization,
        policy_entropy=policy_entropy,
        policy_kl=policy_kl,
        total=total,
    )


def create_train_state(config: TrainingConfig) -> TrainState:
    torch = _import_torch()
    from great_kingdom_ai.model import create_model

    _validate_ema_decay(config.ema_decay)
    model = create_model(config.model_preset).to(config.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = create_lr_scheduler(torch, optimizer, config)
    ema_model = _create_ema_model(model) if config.ema_decay is not None else None
    return TrainState(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=_create_grad_scaler(config),
        ema_model=ema_model,
        ema_decay=config.ema_decay,
        step=0,
        model_preset=config.model_preset,
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
        state.scaler.step(state.optimizer)
        state.scaler.update()
        optimizer_stepped = float(state.scaler.get_scale()) >= scale_before
    else:
        losses.total.backward()  # type: ignore[no-untyped-call]
        state.optimizer.step()
    if optimizer_stepped:
        state.scheduler.step()
        _update_ema_model(state)
    return losses


def save_checkpoint(state: TrainState, path: str | Path) -> Path:
    torch = _import_torch()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": state.step,
            "model_preset": state.model_preset,
            "model_config": asdict(state.model.config),
            "model_state": state.model.state_dict(),
            "optimizer_state": state.optimizer.state_dict(),
            "scheduler_state": state.scheduler.state_dict(),
            "scaler_state": None if state.scaler is None else state.scaler.state_dict(),
            "ema_model_state": (
                None if state.ema_model is None else state.ema_model.state_dict()
            ),
            "ema_decay": state.ema_decay,
        },
        destination,
    )
    return destination


def summarize_checkpoint_optimizer_state(path: str | Path) -> dict[str, Any]:
    """Return a compact, JSON-friendly optimizer-state summary for diagnostics."""
    torch = _import_torch()
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    optimizer_state = checkpoint.get("optimizer_state")
    if not isinstance(optimizer_state, Mapping):
        raise ValueError("checkpoint does not contain optimizer_state")
    summary = summarize_optimizer_state_dict(optimizer_state)
    summary["checkpoint_step"] = int(checkpoint.get("step", 0))
    return summary


def summarize_optimizer_state_dict(optimizer_state: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize optimizer state without printing large tensors."""
    state = optimizer_state.get("state", {})
    param_groups = optimizer_state.get("param_groups", [])
    if not isinstance(state, Mapping):
        raise ValueError("optimizer state must contain a mapping 'state'")
    if not isinstance(param_groups, Sequence):
        raise ValueError("optimizer state must contain a sequence 'param_groups'")

    step_values: list[float] = []
    tensor_buffers: dict[str, dict[str, int]] = {}
    for entry in state.values():
        if not isinstance(entry, Mapping):
            continue
        step = _optimizer_scalar(entry.get("step"))
        if step is not None:
            step_values.append(step)
        for key, value in entry.items():
            if key == "step" or not _is_torch_tensor(value):
                continue
            buffer_summary = tensor_buffers.setdefault(
                str(key),
                {"tensors": 0, "elements": 0},
            )
            buffer_summary["tensors"] += 1
            buffer_summary["elements"] += int(value.numel())

    summary: dict[str, Any] = {
        "param_groups": [
            _optimizer_param_group_summary(group)
            for group in param_groups
            if isinstance(group, Mapping)
        ],
        "state_entries": len(state),
        "tensor_buffers": tensor_buffers,
    }
    if step_values:
        summary["step"] = {
            "min": min(step_values),
            "max": max(step_values),
            "mean": sum(step_values) / len(step_values),
        }
    else:
        summary["step"] = None
    return summary


def load_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str | None = None,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    lr_schedule: str = "step",
    lr_decay_steps: int = 100,
    lr_decay_gamma: float = 0.99,
    lr_warmup_steps: int = 0,
    lr_min_factor: float = 0.1,
    lr_cosine_steps: int = 0,
    steps: int = 1000,
    amp: bool = False,
    ema_decay: float | None = None,
    prefer_ema: bool = False,
    optimizer_lr_override: float | None = None,
) -> TrainState:
    torch = _import_torch()
    from great_kingdom_ai.model import ModelConfig, PolicyValueNetwork

    _validate_ema_decay(ema_decay)
    checkpoint = torch.load(Path(path), map_location=device or "cpu", weights_only=False)
    config = ModelConfig(**checkpoint["model_config"])
    model = PolicyValueNetwork(config).to(device=device)
    model_state = checkpoint["model_state"]
    ema_model_state = checkpoint.get("ema_model_state")
    if prefer_ema and ema_model_state is not None:
        model_state = ema_model_state
    model.load_state_dict(model_state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler = create_lr_scheduler(
        torch,
        optimizer,
        TrainingConfig(
            steps=steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            lr_schedule=lr_schedule,
            lr_decay_steps=lr_decay_steps,
            lr_decay_gamma=lr_decay_gamma,
            lr_warmup_steps=lr_warmup_steps,
            lr_min_factor=lr_min_factor,
            lr_cosine_steps=lr_cosine_steps,
            device=str(device or "cpu"),
        ),
    )
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    _restore_optimizer_lrs_from_scheduler(optimizer, scheduler)
    if optimizer_lr_override is not None:
        _override_optimizer_learning_rate(optimizer, scheduler, optimizer_lr_override)
    scaler = _create_grad_scaler_for_device(torch, device, enabled=amp)
    scaler_state = checkpoint.get("scaler_state")
    if scaler is not None and scaler_state is not None:
        scaler.load_state_dict(scaler_state)
    resolved_ema_decay = ema_decay if ema_decay is not None else checkpoint.get("ema_decay")
    _validate_ema_decay(resolved_ema_decay)
    ema_model = None
    if resolved_ema_decay is not None:
        ema_model = PolicyValueNetwork(config).to(device=device)
        ema_model.load_state_dict(
            ema_model_state if ema_model_state is not None else checkpoint["model_state"]
        )
        ema_model.eval()
        for parameter in ema_model.parameters():
            parameter.requires_grad_(False)
    return TrainState(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        ema_model=ema_model,
        ema_decay=resolved_ema_decay,
        step=int(checkpoint["step"]),
        model_preset=str(checkpoint.get("model_preset", "custom")),
    )


def _optimizer_param_group_summary(group: Mapping[str, Any]) -> dict[str, Any]:
    keys = ("lr", "weight_decay", "betas", "eps", "amsgrad")
    return {key: _optimizer_json_value(group[key]) for key in keys if key in group}


def _optimizer_json_value(value: Any) -> Any:
    scalar = _optimizer_scalar(value)
    if scalar is not None:
        return scalar
    if isinstance(value, tuple):
        return [_optimizer_json_value(item) for item in value]
    if isinstance(value, list):
        return [_optimizer_json_value(item) for item in value]
    return value


def _optimizer_scalar(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if _is_torch_tensor(value) and value.numel() == 1:
        return float(value.detach().cpu().item())
    return None


def _is_torch_tensor(value: Any) -> bool:
    return hasattr(value, "detach") and hasattr(value, "numel")


def load_checkpoint_weights(
    path: str | Path,
    config: TrainingConfig,
) -> TrainState:
    """Load only model weights from a checkpoint and create fresh training state."""
    torch = _import_torch()
    from great_kingdom_ai.model import ModelConfig, PolicyValueNetwork

    _validate_ema_decay(config.ema_decay)
    checkpoint = torch.load(Path(path), map_location=config.device, weights_only=False)
    model_config = ModelConfig(**checkpoint["model_config"])
    model = PolicyValueNetwork(model_config).to(device=config.device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = create_lr_scheduler(torch, optimizer, config)
    ema_model = None
    if config.ema_decay is not None:
        ema_model = PolicyValueNetwork(model_config).to(device=config.device)
        ema_model.load_state_dict(checkpoint.get("ema_model_state") or checkpoint["model_state"])
        ema_model.eval()
        for parameter in ema_model.parameters():
            parameter.requires_grad_(False)
    return TrainState(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=_create_grad_scaler(config),
        ema_model=ema_model,
        ema_decay=config.ema_decay,
        step=0,
        model_preset=str(checkpoint.get("model_preset", "custom")),
    )


def create_lr_scheduler(
    torch: Any,
    optimizer: Optimizer,
    config: TrainingConfig,
) -> LRScheduler:
    """Create the configured learning-rate scheduler.

    ``step`` preserves the original training behavior. ``constant_with_warmup``
    ramps the learning rate up for ``lr_warmup_steps`` then keeps it constant.
    ``warmup_cosine`` ramps up then decays it to ``lr_min_factor``.
    """
    if config.lr_schedule == "step":
        if config.lr_decay_steps <= 0:
            raise ValueError("lr_decay_steps must be positive")
        if not math.isfinite(config.lr_decay_gamma) or config.lr_decay_gamma <= 0.0:
            raise ValueError("lr_decay_gamma must be finite and positive")
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.lr_decay_steps,
            gamma=config.lr_decay_gamma,
        )
    if config.lr_schedule not in {"constant_with_warmup", "warmup_cosine"}:
        raise ValueError(
            "lr_schedule must be one of: step, constant_with_warmup, warmup_cosine"
        )
    if config.lr_warmup_steps < 0:
        raise ValueError("lr_warmup_steps must be non-negative")
    if config.lr_schedule == "constant_with_warmup":
        def lr_factor(step_index: int) -> float:
            if config.lr_warmup_steps > 0 and step_index < config.lr_warmup_steps:
                return (step_index + 1) / config.lr_warmup_steps
            return 1.0

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)

    if not math.isfinite(config.lr_min_factor) or not 0.0 <= config.lr_min_factor <= 1.0:
        raise ValueError("lr_min_factor must be in [0, 1]")

    total_steps = config.lr_cosine_steps if config.lr_cosine_steps > 0 else config.steps
    if total_steps <= 0:
        raise ValueError("lr_cosine_steps or steps must be positive")
    warmup_steps = min(config.lr_warmup_steps, total_steps)

    def lr_factor(step_index: int) -> float:
        if warmup_steps > 0 and step_index < warmup_steps:
            return (step_index + 1) / warmup_steps
        decay_steps = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, (step_index - warmup_steps) / decay_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return config.lr_min_factor + (1.0 - config.lr_min_factor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)


def _restore_optimizer_lrs_from_scheduler(
    optimizer: Optimizer,
    scheduler: LRScheduler,
) -> None:
    for group, learning_rate in zip(optimizer.param_groups, scheduler.get_last_lr(), strict=True):
        group["lr"] = learning_rate


def _override_optimizer_learning_rate(
    optimizer: Optimizer,
    scheduler: LRScheduler,
    learning_rate: float,
) -> None:
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("optimizer_lr_override must be finite and positive")
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
        if "initial_lr" in group:
            group["initial_lr"] = learning_rate
    if hasattr(scheduler, "base_lrs"):
        scheduler.base_lrs = [learning_rate for _ in scheduler.base_lrs]
    if hasattr(scheduler, "_last_lr"):
        scheduler._last_lr = [learning_rate for _ in optimizer.param_groups]


def _create_ema_model(model: PolicyValueNetwork) -> PolicyValueNetwork:
    import copy

    ema_model = copy.deepcopy(model)
    ema_model.eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)
    return ema_model


def _update_ema_model(state: TrainState) -> None:
    if state.ema_model is None or state.ema_decay is None:
        return
    torch = _import_torch()
    model_state = state.model.state_dict()
    ema_state = state.ema_model.state_dict()
    with torch.no_grad():
        for key, ema_tensor in ema_state.items():
            model_tensor = model_state[key].detach().to(
                device=ema_tensor.device,
                dtype=ema_tensor.dtype,
            )
            if ema_tensor.is_floating_point() and model_tensor.is_floating_point():
                ema_tensor.mul_(state.ema_decay).add_(model_tensor, alpha=1.0 - state.ema_decay)
            else:
                ema_tensor.copy_(model_tensor)


def _validate_ema_decay(decay: float | None) -> None:
    if decay is None:
        return
    if not math.isfinite(decay) or not 0.0 <= decay < 1.0:
        raise ValueError("ema_decay must be in [0, 1)")


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


def _iter_training_batches(
    replay: ReplayDataset,
    config: TrainingConfig,
    rng: random.Random,
    *,
    torch: Any,
) -> Iterator[TrainingBatch]:
    if not _use_cuda_prefetch(torch, config):
        for _ in range(config.steps):
            yield _sample_training_batch(
                replay,
                config,
                rng,
                device=config.device,
                pin_memory=False,
            )
        return

    cpu_batches = PrefetchIterator(
        producer=lambda: _sample_training_batch(
            replay,
            config,
            rng,
            device=None,
            pin_memory=True,
        ),
        count=config.steps,
        max_prefetch=config.prefetch_batches,
    )
    for batch in cpu_batches:
        yield _batch_to_device(batch, config.device, non_blocking=True)


def _sample_training_replay(
    replay: ReplayDataset,
    config: TrainingConfig,
    rng: random.Random,
) -> list[ReplaySample]:
    if config.priority_enabled:
        sampler = getattr(replay, "sample_priority_biased", None)
        if sampler is None:
            raise ValueError("replay dataset does not support priority-aware sampling")
        return cast(
            list[ReplaySample],
            sampler(
                config.batch_size,
                rng,
                priority_config=_priority_sampling_config(config),
                recent_fraction=config.recent_sample_fraction,
                recent_window=config.recent_sample_window,
            ),
        )
    if config.recent_sample_fraction <= 0.0:
        return replay.sample(config.batch_size, rng)
    if config.recent_sample_fraction > 1.0:
        raise ValueError("recent_sample_fraction must be in [0, 1]")
    if config.recent_sample_window <= 0:
        raise ValueError("recent_sample_window must be positive when recency sampling is enabled")
    sampler = getattr(replay, "sample_recency_biased", None)
    if sampler is None:
        raise ValueError("replay dataset does not support recency-biased sampling")
    return cast(
        list[ReplaySample],
        sampler(
            config.batch_size,
            rng,
            recent_fraction=config.recent_sample_fraction,
            recent_window=config.recent_sample_window,
        ),
    )


def _sample_training_batch(
    replay: ReplayDataset,
    config: TrainingConfig,
    rng: random.Random,
    *,
    device: torch.device | str | None,
    pin_memory: bool,
) -> TrainingBatch:
    array_sampler = getattr(replay, "sample_arrays", None)
    if array_sampler is not None:
        if config.priority_enabled:
            raw_arrays = array_sampler(
                config.batch_size,
                rng,
                recent_fraction=config.recent_sample_fraction,
                recent_window=config.recent_sample_window,
                priority_config=_priority_sampling_config(config),
            )
        else:
            raw_arrays = array_sampler(
                config.batch_size,
                rng,
                recent_fraction=config.recent_sample_fraction,
                recent_window=config.recent_sample_window,
            )
        arrays = TrainingArrays(
            features=np.asarray(raw_arrays.features, dtype=np.float32),
            policies=np.asarray(raw_arrays.policies, dtype=np.float32),
            values=np.asarray(raw_arrays.values, dtype=np.float32),
            sample_weights=np.asarray(raw_arrays.sample_weights, dtype=np.float32),
            legal_masks=(
                None
                if getattr(raw_arrays, "legal_masks", None) is None
                else np.asarray(raw_arrays.legal_masks, dtype=np.bool_)
            ),
        )
        if config.symmetry_augmentation:
            features, policies, legal_masks = augment_training_arrays_randomly(
                arrays.features,
                arrays.policies,
                arrays.legal_masks,
                rng,
            )
            arrays = TrainingArrays(
                features=features,
                policies=policies,
                values=arrays.values,
                sample_weights=arrays.sample_weights,
                legal_masks=legal_masks,
            )
        return arrays_to_batch(arrays, device=device, pin_memory=pin_memory)

    samples = _sample_training_replay(replay, config, rng)
    if config.symmetry_augmentation:
        samples = augment_samples_randomly(samples, rng)
    return samples_to_batch(samples, device=device, pin_memory=pin_memory)


def _priority_sampling_config(config: TrainingConfig) -> PrioritySamplingConfig:
    return PrioritySamplingConfig(
        enabled=config.priority_enabled,
        alpha=config.priority_alpha,
        beta=config.priority_beta,
        value_error_weight=config.priority_value_error_weight,
        policy_kl_weight=config.priority_policy_kl_weight,
        target_age_weight=config.priority_target_age_weight,
        search_reanalyzed_boost=config.priority_search_reanalyzed_boost,
        max_priority=config.priority_max_priority,
    )


def _batch_to_device(
    batch: TrainingBatch,
    device: torch.device | str | None,
    *,
    non_blocking: bool = False,
) -> TrainingBatch:
    if device is None:
        return batch
    return TrainingBatch(
        features=batch.features.to(device=device, non_blocking=non_blocking),
        policy=batch.policy.to(device=device, non_blocking=non_blocking),
        value=batch.value.to(device=device, non_blocking=non_blocking),
        legal_mask=batch.legal_mask.to(device=device, non_blocking=non_blocking),
        sample_weight=batch.sample_weight.to(device=device, non_blocking=non_blocking),
    )


def _tensor_from_numpy(
    torch: Any,
    array: np.ndarray,
    *,
    pin_memory: bool,
) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(array))
    if pin_memory:
        return cast("Tensor", tensor.pin_memory())
    return cast("Tensor", tensor)


def _use_cuda_prefetch(torch: Any, config: TrainingConfig) -> bool:
    return bool(
        config.prefetch_batches > 0
        and config.steps > 0
        and str(config.device).startswith("cuda")
        and torch.cuda.is_available()
    )


def load_training_config(path: str | Path) -> TrainingConfig:
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError("training config must be a JSON object")
    return TrainingConfig(**data)


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


def _legal_masks_from_features(features: np.ndarray) -> np.ndarray:
    legal_place = features[:, LEGAL_PLACE_FEATURE_CHANNEL].reshape(-1, BOARD_CELLS) > 0.5
    legal_mask = np.zeros((features.shape[0], BOARD_CELLS + 1), dtype=np.bool_)
    legal_mask[:, :BOARD_CELLS] = legal_place
    legal_mask[:, PASS_ACTION] = True
    return legal_mask


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
    torch = _import_torch()
    positive = policy > 0.0
    per_row = -(torch.where(positive, policy * torch.log(policy.clamp_min(1e-45)), 0.0)).sum(dim=1)
    return _weighted_mean(per_row, sample_weight)


def _weighted_mean(values: torch.Tensor, sample_weight: torch.Tensor) -> torch.Tensor:
    weights = sample_weight.to(dtype=values.dtype)
    return (values * weights).sum() / weights.sum()


def _create_grad_scaler(config: TrainingConfig) -> Any | None:
    torch = _import_torch()
    return _create_grad_scaler_for_device(torch, config.device, enabled=config.amp)


def _create_grad_scaler_for_device(
    torch: Any,
    device: torch.device | str | None,
    *,
    enabled: bool,
) -> Any | None:
    if not _cuda_amp_enabled(torch, device, enabled=enabled):
        return None
    return torch.amp.GradScaler("cuda", enabled=True)


def _amp_enabled(config: TrainingConfig) -> bool:
    torch = _import_torch()
    return _cuda_amp_enabled(torch, config.device, enabled=config.amp)


def _cuda_amp_enabled(
    torch: Any,
    device: torch.device | str | None,
    *,
    enabled: bool,
) -> bool:
    return bool(
        enabled
        and device is not None
        and str(device).startswith("cuda")
        and torch.cuda.is_available()
    )


def _autocast_context(torch: Any, *, enabled: bool) -> Any:
    if not enabled:
        return nullcontext()
    return torch.amp.autocast("cuda", enabled=True)


def _import_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required for training") from exc
    return torch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a Great Kingdom policy-value network")
    parser.add_argument("--replay", type=Path, required=True, help="Path to replay .npz file")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Output checkpoint path")
    parser.add_argument("--resume", type=Path, default=None, help="Checkpoint to resume from")
    parser.add_argument(
        "--bootstrap-weights",
        type=Path,
        default=None,
        help="Checkpoint to load model weights from without optimizer, scheduler, or step state",
    )
    parser.add_argument("--config", type=Path, default=None, help="JSON TrainingConfig override")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--model-preset",
        choices=[
            "small",
            "medium",
            "medium_plus",
            "strong",
            "large",
            "large_policy",
            "large_plus",
        ],
        default=None,
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=None,
        help="record loss every N training steps instead of only the final step",
    )
    parser.add_argument(
        "--no-symmetry-augmentation",
        action="store_true",
        help="disable random board symmetry augmentation during batch sampling",
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=None,
        help="enable exponential moving average weights with the given decay",
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> TrainingConfig:
    config = load_training_config(args.config) if args.config is not None else TrainingConfig()
    overrides = {
        "device": args.device,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "model_preset": args.model_preset,
        "symmetry_augmentation": False if args.no_symmetry_augmentation else None,
        "ema_decay": args.ema_decay,
    }
    data = asdict(config)
    data.update({key: value for key, value in overrides.items() if value is not None})
    return TrainingConfig(**data)


def _log_every_from_args(args: argparse.Namespace, config: TrainingConfig) -> int:
    log_every = args.log_every if args.log_every is not None else config.steps
    if log_every <= 0:
        raise ValueError("log_every must be positive")
    return max(1, log_every)


def print_training_startup_config(
    *,
    config: TrainingConfig,
    replay: ReplayDataset,
    replay_path: Path,
    checkpoint_path: Path,
    resume_path: Path | None,
    bootstrap_weights_path: Path | None = None,
) -> None:
    print(
        json.dumps(
            {
                "event": "train_config",
                "config": asdict(config),
                "replay": {
                    "path": str(replay_path),
                    "samples": len(replay),
                    "capacity": getattr(replay, "capacity", len(replay)),
                    "type": type(replay).__name__,
                },
                "checkpoint": str(checkpoint_path),
                "resume": str(resume_path) if resume_path is not None else None,
                "bootstrap_weights": (
                    str(bootstrap_weights_path) if bootstrap_weights_path is not None else None
                ),
            },
            sort_keys=True,
        )
    )


def load_training_replay(path: str | Path) -> ReplayDataset:
    from great_kingdom_ai.reanalyze import (
        ReanalyzeTargetSnapshot,
        is_reanalyze_target_snapshot,
    )
    from great_kingdom_ai.replay import TrajectoryReplayDataset, TrajectoryReplayStore

    if is_reanalyze_target_snapshot(path):
        return ReanalyzeTargetSnapshot.load(path)
    return TrajectoryReplayDataset(TrajectoryReplayStore.load(path))


def main() -> NoReturn:
    args = build_parser().parse_args()
    config = _config_from_args(args)
    replay = load_training_replay(args.replay)
    print_training_startup_config(
        config=config,
        replay=replay,
        replay_path=args.replay,
        checkpoint_path=args.checkpoint,
        resume_path=args.resume,
        bootstrap_weights_path=args.bootstrap_weights,
    )
    summary = train_from_replay(
        replay,
        config,
        checkpoint_path=args.checkpoint,
        resume_path=args.resume,
        bootstrap_weights_path=args.bootstrap_weights,
        log_every=_log_every_from_args(args, config),
    )
    print(
        json.dumps(
            {
                "event": "train_summary",
                "start_step": summary.start_step,
                "end_step": summary.end_step,
                "checkpoint": str(summary.checkpoint_path) if summary.checkpoint_path else None,
                "losses": summary.losses,
            },
            sort_keys=True,
        )
    )
    raise SystemExit(0)


if __name__ == "__main__":
    main()


__all__ = [
    "LossBreakdown",
    "TrainState",
    "TrainSummary",
    "TrainingArrays",
    "TrainingBatch",
    "TrainingConfig",
    "ReplayDataset",
    "arrays_to_batch",
    "compute_losses",
    "create_lr_scheduler",
    "create_train_state",
    "load_checkpoint",
    "load_checkpoint_weights",
    "load_training_config",
    "load_training_replay",
    "print_training_startup_config",
    "samples_to_batch",
    "save_checkpoint",
    "summarize_checkpoint_optimizer_state",
    "summarize_optimizer_state_dict",
    "train_from_replay",
    "train_step",
]
