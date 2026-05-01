"""Training loop and checkpoint helpers for AlphaZero-lite models."""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

import numpy as np

from great_kingdom_ai.replay_buffer import ReplayBuffer, ReplaySample

if TYPE_CHECKING:
    import torch
    from torch import nn
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
    lr_decay_gamma: float = 0.99
    lr_decay_steps: int = 100
    seed: int = 0
    device: str = "cpu"
    model_preset: str = "small"


@dataclass(frozen=True)
class TrainingBatch:
    features: torch.Tensor
    policy: torch.Tensor
    value: torch.Tensor


@dataclass(frozen=True)
class LossBreakdown:
    policy: torch.Tensor
    value: torch.Tensor
    regularization: torch.Tensor
    total: torch.Tensor

    def to_float_dict(self) -> dict[str, float]:
        return {
            "policy": float(self.policy.detach().cpu()),
            "value": float(self.value.detach().cpu()),
            "regularization": float(self.regularization.detach().cpu()),
            "total": float(self.total.detach().cpu()),
        }


@dataclass(frozen=True)
class TrainState:
    model: PolicyValueNetwork
    optimizer: Optimizer
    scheduler: LRScheduler
    step: int = 0
    model_preset: str = "small"


@dataclass(frozen=True)
class TrainSummary:
    start_step: int
    end_step: int
    checkpoint_path: Path | None
    losses: list[dict[str, float]]


def samples_to_batch(
    samples: Sequence[ReplaySample],
    *,
    device: torch.device | str | None = None,
) -> TrainingBatch:
    """Convert replay samples into tensors shaped for the policy-value network."""
    torch = _import_torch()
    if not samples:
        raise ValueError("training batch must contain at least one sample")

    features = np.stack([sample.features for sample in samples], axis=0).astype(np.float32)
    policies = np.stack([sample.policy for sample in samples], axis=0).astype(np.float32)
    values = np.asarray([sample.value for sample in samples], dtype=np.float32)

    return TrainingBatch(
        features=torch.from_numpy(features).to(device=device),
        policy=torch.from_numpy(policies).to(device=device),
        value=torch.from_numpy(values).to(device=device),
    )


def compute_losses(
    model: nn.Module,
    batch: TrainingBatch,
    *,
    policy_loss_weight: float = 1.0,
    value_loss_weight: float = 1.0,
    l2_loss_weight: float = 0.0,
) -> LossBreakdown:
    """Compute AlphaZero policy cross-entropy, value MSE, and optional L2 loss."""
    torch = _import_torch()
    if policy_loss_weight < 0.0 or value_loss_weight < 0.0 or l2_loss_weight < 0.0:
        raise ValueError("loss weights must be non-negative")

    policy_logits, value = model(batch.features)
    log_policy = torch.log_softmax(policy_logits, dim=1)
    policy_loss = -(batch.policy * log_policy).sum(dim=1).mean()
    value_loss = torch.nn.functional.mse_loss(value, batch.value)
    regularization = _l2_regularization(model) * l2_loss_weight
    total = policy_loss_weight * policy_loss + value_loss_weight * value_loss + regularization
    return LossBreakdown(
        policy=policy_loss,
        value=value_loss,
        regularization=regularization,
        total=total,
    )


def create_train_state(config: TrainingConfig) -> TrainState:
    torch = _import_torch()
    from great_kingdom_ai.model import create_model

    model = create_model(config.model_preset).to(config.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.lr_decay_steps,
        gamma=config.lr_decay_gamma,
    )
    return TrainState(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=0,
        model_preset=config.model_preset,
    )


def train_step(state: TrainState, batch: TrainingBatch, config: TrainingConfig) -> LossBreakdown:
    state.model.train()
    state.optimizer.zero_grad(set_to_none=True)
    losses = compute_losses(
        state.model,
        batch,
        policy_loss_weight=config.policy_loss_weight,
        value_loss_weight=config.value_loss_weight,
        l2_loss_weight=config.l2_loss_weight,
    )
    losses.total.backward()
    state.optimizer.step()
    state.scheduler.step()
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
        },
        destination,
    )
    return destination


def load_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str | None = None,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    lr_decay_steps: int = 100,
    lr_decay_gamma: float = 0.99,
) -> TrainState:
    torch = _import_torch()
    from great_kingdom_ai.model import ModelConfig, PolicyValueNetwork

    checkpoint = torch.load(Path(path), map_location=device or "cpu", weights_only=False)
    config = ModelConfig(**checkpoint["model_config"])
    model = PolicyValueNetwork(config).to(device=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=lr_decay_steps,
        gamma=lr_decay_gamma,
    )
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    return TrainState(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=int(checkpoint["step"]),
        model_preset=str(checkpoint.get("model_preset", "custom")),
    )


def train_from_replay(
    replay: ReplayBuffer,
    config: TrainingConfig,
    *,
    checkpoint_path: str | Path | None = None,
    resume_path: str | Path | None = None,
    log_every: int = 0,
) -> TrainSummary:
    if len(replay) < config.batch_size:
        raise ValueError("replay buffer must contain at least batch_size samples")

    torch = _import_torch()
    torch.manual_seed(config.seed)
    rng = random.Random(config.seed)

    if resume_path is None:
        state = create_train_state(config)
    else:
        state = load_checkpoint(
            resume_path,
            device=config.device,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            lr_decay_steps=config.lr_decay_steps,
            lr_decay_gamma=config.lr_decay_gamma,
        )

    start_step = state.step
    losses: list[dict[str, float]] = []
    for step in range(start_step, start_step + config.steps):
        samples = replay.sample(config.batch_size, rng)
        batch = samples_to_batch(samples, device=config.device)
        loss = train_step(state, batch, config)
        state = TrainState(
            model=state.model,
            optimizer=state.optimizer,
            scheduler=state.scheduler,
            step=step + 1,
            model_preset=state.model_preset,
        )
        if log_every > 0:
            should_log = state.step == start_step + config.steps or state.step % log_every == 0
        else:
            should_log = False
        if should_log:
            losses.append({"step": float(state.step), **loss.to_float_dict()})

    saved_path = save_checkpoint(state, checkpoint_path) if checkpoint_path is not None else None
    return TrainSummary(
        start_step=start_step,
        end_step=state.step,
        checkpoint_path=saved_path,
        losses=losses,
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
        return torch.tensor(0.0)
    return torch.stack(parameters).sum()


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
    parser.add_argument("--config", type=Path, default=None, help="JSON TrainingConfig override")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--model-preset", choices=["small", "medium", "large"], default=None)
    return parser


def _config_from_args(args: argparse.Namespace) -> TrainingConfig:
    config = load_training_config(args.config) if args.config is not None else TrainingConfig()
    overrides = {
        "device": args.device,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "model_preset": args.model_preset,
    }
    data = asdict(config)
    data.update({key: value for key, value in overrides.items() if value is not None})
    return TrainingConfig(**data)


def main() -> NoReturn:
    args = build_parser().parse_args()
    config = _config_from_args(args)
    replay = ReplayBuffer.load(args.replay)
    summary = train_from_replay(
        replay,
        config,
        checkpoint_path=args.checkpoint,
        resume_path=args.resume,
        log_every=max(1, config.steps),
    )
    print(
        json.dumps(
            {
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
    "TrainingBatch",
    "TrainingConfig",
    "compute_losses",
    "create_train_state",
    "load_checkpoint",
    "load_training_config",
    "samples_to_batch",
    "save_checkpoint",
    "train_from_replay",
    "train_step",
]
