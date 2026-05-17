"""Checkpoint, optimizer, scheduler, and EMA state helpers."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from great_kingdom_ai.training.config import TrainingConfig
from great_kingdom_ai.training.torch_utils import _cuda_amp_enabled, _import_torch

if TYPE_CHECKING:
    import torch
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler

    from great_kingdom_ai.model import PolicyValueNetwork

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
        return cast(
            "LRScheduler",
            torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config.lr_decay_steps,
                gamma=config.lr_decay_gamma,
            ),
        )
    if config.lr_schedule not in {"constant_with_warmup", "warmup_cosine"}:
        raise ValueError(
            "lr_schedule must be one of: step, constant_with_warmup, warmup_cosine"
        )
    if config.lr_warmup_steps < 0:
        raise ValueError("lr_warmup_steps must be non-negative")
    if config.lr_schedule == "constant_with_warmup":
        def warmup_lr_factor(step_index: int) -> float:
            if config.lr_warmup_steps > 0 and step_index < config.lr_warmup_steps:
                return (step_index + 1) / config.lr_warmup_steps
            return 1.0

        return cast(
            "LRScheduler",
            torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lr_factor),
        )

    if not math.isfinite(config.lr_min_factor) or not 0.0 <= config.lr_min_factor <= 1.0:
        raise ValueError("lr_min_factor must be in [0, 1]")

    total_steps = config.lr_cosine_steps if config.lr_cosine_steps > 0 else config.steps
    if total_steps <= 0:
        raise ValueError("lr_cosine_steps or steps must be positive")
    warmup_steps = min(config.lr_warmup_steps, total_steps)

    def cosine_lr_factor(step_index: int) -> float:
        if warmup_steps > 0 and step_index < warmup_steps:
            return (step_index + 1) / warmup_steps
        decay_steps = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, (step_index - warmup_steps) / decay_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return config.lr_min_factor + (1.0 - config.lr_min_factor) * cosine

    return cast(
        "LRScheduler",
        torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=cosine_lr_factor),
    )


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


__all__ = [
    "TrainState",
    "create_lr_scheduler",
    "create_train_state",
    "load_checkpoint",
    "load_checkpoint_weights",
    "save_checkpoint",
    "summarize_checkpoint_optimizer_state",
    "summarize_optimizer_state_dict",
]
