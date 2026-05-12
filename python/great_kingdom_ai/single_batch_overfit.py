"""Single-batch overfit diagnostic for Great Kingdom training data."""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

from great_kingdom_ai.replay_buffer import ReplayBuffer
from great_kingdom_ai.train import (
    TrainingBatch,
    TrainingConfig,
    compute_losses,
    create_train_state,
    load_checkpoint,
    load_training_config,
    samples_to_batch,
    save_checkpoint,
    train_step,
)

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class SingleBatchOverfitSummary:
    start_step: int
    end_step: int
    checkpoint_path: Path | None
    initial_loss: dict[str, float]
    final_loss: dict[str, float]
    losses: list[dict[str, float]]


def run_single_batch_overfit(
    replay: ReplayBuffer,
    config: TrainingConfig,
    *,
    checkpoint_path: str | Path | None = None,
    resume_path: str | Path | None = None,
    log_every: int = 50,
    progress_callback: Callable[[int, int, dict[str, float]], None] | None = None,
) -> SingleBatchOverfitSummary:
    """Train repeatedly on one fixed replay batch and report loss movement."""
    if len(replay) < config.batch_size:
        raise ValueError("replay buffer must contain at least batch_size samples")
    if config.steps <= 0:
        raise ValueError("steps must be positive")
    if log_every <= 0:
        raise ValueError("log_every must be positive")

    torch = _import_torch()
    torch.manual_seed(config.seed)
    rng = random.Random(config.seed)
    batch = samples_to_batch(replay.sample(config.batch_size, rng), device=config.device)

    if resume_path is None:
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
        )

    start_step = state.step
    initial_loss = _evaluate_loss(state.model, batch, config)
    losses: list[dict[str, float]] = [{"step": float(start_step), **initial_loss}]
    if progress_callback is not None:
        progress_callback(0, config.steps, initial_loss)

    for step in range(start_step, start_step + config.steps):
        loss = train_step(state, batch, config)
        state = type(state)(
            model=state.model,
            optimizer=state.optimizer,
            scheduler=state.scheduler,
            step=step + 1,
            model_preset=state.model_preset,
        )
        should_log = (
            state.step == start_step + config.steps
            or (state.step - start_step) % log_every == 0
        )
        if should_log:
            loss_values = loss.to_float_dict()
            losses.append({"step": float(state.step), **loss_values})
            if progress_callback is not None and state.step != start_step + config.steps:
                progress_callback(state.step - start_step, config.steps, loss_values)

    final_loss = _evaluate_loss(state.model, batch, config)
    if losses[-1]["step"] != float(state.step):
        losses.append({"step": float(state.step), **final_loss})
    else:
        losses[-1] = {"step": float(state.step), **final_loss}
    if progress_callback is not None:
        progress_callback(config.steps, config.steps, final_loss)

    saved_path = save_checkpoint(state, checkpoint_path) if checkpoint_path is not None else None
    return SingleBatchOverfitSummary(
        start_step=start_step,
        end_step=state.step,
        checkpoint_path=saved_path,
        initial_loss=initial_loss,
        final_loss=final_loss,
        losses=losses,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Overfit one fixed replay batch to diagnose training/loss wiring"
    )
    parser.add_argument("--replay", type=Path, required=True, help="Path to replay .npz file")
    parser.add_argument("--config", type=Path, default=None, help="JSON TrainingConfig override")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional output checkpoint path",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Optional checkpoint to resume from",
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
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
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--no-mask-policy-loss",
        action="store_true",
        help="disable legal-action masking in policy loss",
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> TrainingConfig:
    config = load_training_config(args.config) if args.config is not None else TrainingConfig()
    overrides = {
        "device": args.device,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "model_preset": args.model_preset,
        "seed": args.seed,
        "symmetry_augmentation": False,
        "mask_policy_loss": False if args.no_mask_policy_loss else None,
    }
    data = asdict(config)
    data.update({key: value for key, value in overrides.items() if value is not None})
    data["symmetry_augmentation"] = False
    return TrainingConfig(**data)


def _evaluate_loss(
    model: torch.nn.Module,
    batch: TrainingBatch,
    config: TrainingConfig,
) -> dict[str, float]:
    torch = _import_torch()
    model.train()
    with torch.no_grad():
        return compute_losses(
            model,
            batch,
            policy_loss_weight=config.policy_loss_weight,
            value_loss_weight=config.value_loss_weight,
            l2_loss_weight=config.l2_loss_weight,
            mask_policy_loss=config.mask_policy_loss,
        ).to_float_dict()


def _batch_diagnostics(batch: TrainingBatch) -> dict[str, float]:
    illegal_target_mass = batch.policy.masked_select(~batch.legal_mask).sum()
    legal_actions = batch.legal_mask.sum(dim=1).to(dtype=batch.value.dtype)
    return {
        "illegal_target_mass": float(illegal_target_mass.detach().cpu()),
        "legal_actions_mean": float(legal_actions.mean().detach().cpu()),
        "value_min": float(batch.value.min().detach().cpu()),
        "value_mean": float(batch.value.mean().detach().cpu()),
        "value_max": float(batch.value.max().detach().cpu()),
    }


def _print_loss_progress(done: int, total: int, losses: dict[str, float]) -> None:
    _print_event(
        "single_batch_overfit_loss",
        {
            "step": done,
            "total_steps": total,
            "loss": losses,
        },
    )


def _print_event(event: str, payload: dict[str, Any]) -> None:
    print(json.dumps({"event": event, **payload}, sort_keys=True))


def _import_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required for single-batch overfit") from exc
    return torch


def main() -> NoReturn:
    args = build_parser().parse_args()
    config = _config_from_args(args)
    replay = ReplayBuffer.load(args.replay)
    rng = random.Random(config.seed)
    diagnostic_batch = samples_to_batch(replay.sample(config.batch_size, rng), device=config.device)

    _print_event(
        "single_batch_overfit_config",
        {
            "config": asdict(config),
            "replay": {
                "path": str(args.replay),
                "samples": len(replay),
                "capacity": replay.capacity,
            },
            "checkpoint": str(args.checkpoint) if args.checkpoint is not None else None,
            "resume": str(args.resume) if args.resume is not None else None,
            "batch": _batch_diagnostics(diagnostic_batch),
        },
    )
    summary = run_single_batch_overfit(
        replay,
        config,
        checkpoint_path=args.checkpoint,
        resume_path=args.resume,
        log_every=args.log_every,
        progress_callback=_print_loss_progress,
    )
    _print_event(
        "single_batch_overfit_summary",
        {
            "start_step": summary.start_step,
            "end_step": summary.end_step,
            "checkpoint": str(summary.checkpoint_path) if summary.checkpoint_path else None,
            "initial_loss": summary.initial_loss,
            "final_loss": summary.final_loss,
            "losses": summary.losses,
        },
    )
    raise SystemExit(0)


if __name__ == "__main__":
    main()


__all__ = [
    "SingleBatchOverfitSummary",
    "build_parser",
    "run_single_batch_overfit",
]
