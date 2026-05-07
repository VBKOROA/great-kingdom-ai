"""Training loop and checkpoint helpers for AlphaZero-lite models."""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn, Protocol, cast

import numpy as np

from great_kingdom_ai.augmentation import augment_samples_randomly
from great_kingdom_ai.features import BOARD_CELLS, LEGAL_PLACE_FEATURE_CHANNEL, PASS_ACTION
from great_kingdom_ai.replay_buffer import ReplayBuffer, ReplaySample

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
    lr_decay_gamma: float = 0.99
    lr_decay_steps: int = 100
    seed: int = 0
    device: str = "cpu"
    model_preset: str = "small"
    symmetry_augmentation: bool = True
    mask_policy_loss: bool = True
    amp: bool = False
    recent_sample_fraction: float = 0.0
    recent_sample_window: int = 0


@dataclass(frozen=True)
class TrainingBatch:
    features: torch.Tensor
    policy: torch.Tensor
    value: torch.Tensor
    legal_mask: torch.Tensor
    sample_weight: torch.Tensor


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
        features=torch.from_numpy(features).to(device=device),
        policy=torch.from_numpy(policies).to(device=device),
        value=torch.from_numpy(values).to(device=device),
        legal_mask=torch.from_numpy(legal_masks).to(device=device),
        sample_weight=torch.from_numpy(sample_weights).to(device=device),
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
        scaler=_create_grad_scaler(config),
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
    if amp_enabled and state.scaler is not None:
        state.scaler.scale(losses.total).backward()  # type: ignore[no-untyped-call]
        state.scaler.step(state.optimizer)
        state.scaler.update()
    else:
        losses.total.backward()  # type: ignore[no-untyped-call]
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
            "scaler_state": None if state.scaler is None else state.scaler.state_dict(),
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
    amp: bool = False,
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
    scaler = _create_grad_scaler_for_device(torch, device, enabled=amp)
    scaler_state = checkpoint.get("scaler_state")
    if scaler is not None and scaler_state is not None:
        scaler.load_state_dict(scaler_state)
    return TrainState(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        step=int(checkpoint["step"]),
        model_preset=str(checkpoint.get("model_preset", "custom")),
    )


def load_checkpoint_weights(
    path: str | Path,
    config: TrainingConfig,
) -> TrainState:
    """Load only model weights from a checkpoint and create fresh training state."""
    torch = _import_torch()
    from great_kingdom_ai.model import ModelConfig, PolicyValueNetwork

    checkpoint = torch.load(Path(path), map_location=config.device, weights_only=False)
    model_config = ModelConfig(**checkpoint["model_config"])
    model = PolicyValueNetwork(model_config).to(device=config.device)
    model.load_state_dict(checkpoint["model_state"])
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
        scaler=_create_grad_scaler(config),
        step=0,
        model_preset=str(checkpoint.get("model_preset", "custom")),
    )


def train_from_replay(
    replay: ReplayDataset,
    config: TrainingConfig,
    *,
    checkpoint_path: str | Path | None = None,
    resume_path: str | Path | None = None,
    bootstrap_weights_path: str | Path | None = None,
    log_every: int = 0,
    progress_callback: Callable[[int, int, dict[str, float]], None] | None = None,
) -> TrainSummary:
    if len(replay) < config.batch_size:
        raise ValueError("replay buffer must contain at least batch_size samples")
    if resume_path is not None and bootstrap_weights_path is not None:
        raise ValueError("resume_path and bootstrap_weights_path are mutually exclusive")

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
            lr_decay_steps=config.lr_decay_steps,
            lr_decay_gamma=config.lr_decay_gamma,
            amp=config.amp,
        )

    start_step = state.step
    losses: list[dict[str, float]] = []
    for step in range(start_step, start_step + config.steps):
        samples = _sample_training_replay(replay, config, rng)
        if config.symmetry_augmentation:
            samples = augment_samples_randomly(samples, rng)
        batch = samples_to_batch(samples, device=config.device)
        loss = train_step(state, batch, config)
        state = TrainState(
            model=state.model,
            optimizer=state.optimizer,
            scheduler=state.scheduler,
            scaler=state.scaler,
            step=step + 1,
            model_preset=state.model_preset,
        )
        if log_every > 0:
            should_log = state.step == start_step + config.steps or state.step % log_every == 0
        else:
            should_log = False
        if should_log:
            loss_values = loss.to_float_dict()
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


def _sample_training_replay(
    replay: ReplayDataset,
    config: TrainingConfig,
    rng: random.Random,
) -> list[ReplaySample]:
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
        choices=["small", "medium", "medium_plus", "strong", "large"],
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
    return parser


def _config_from_args(args: argparse.Namespace) -> TrainingConfig:
    config = load_training_config(args.config) if args.config is not None else TrainingConfig()
    overrides = {
        "device": args.device,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "model_preset": args.model_preset,
        "symmetry_augmentation": False if args.no_symmetry_augmentation else None,
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
    replay: ReplayBuffer,
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
                    "capacity": replay.capacity,
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


def main() -> NoReturn:
    args = build_parser().parse_args()
    config = _config_from_args(args)
    replay = ReplayBuffer.load(args.replay)
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
    "TrainingBatch",
    "TrainingConfig",
    "ReplayDataset",
    "compute_losses",
    "create_train_state",
    "load_checkpoint",
    "load_checkpoint_weights",
    "load_training_config",
    "print_training_startup_config",
    "samples_to_batch",
    "save_checkpoint",
    "train_from_replay",
    "train_step",
]
