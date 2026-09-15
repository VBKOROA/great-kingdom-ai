"""KLENT checkpoint persistence with strict algorithm guards."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from great_kingdom_ai.klent.shards import KLENT_ALGORITHM
from great_kingdom_ai.klent.types import KlentConfig
from great_kingdom_ai.training.checkpoint import create_optimizer
from great_kingdom_ai.training.config import TrainingConfig
from great_kingdom_ai.training.torch_utils import _cuda_amp_enabled, _import_torch

if TYPE_CHECKING:
    from torch.optim import Optimizer

    from great_kingdom_ai.model import PolicyValueNetwork


@dataclass(frozen=True)
class KlentTrainState:
    model: PolicyValueNetwork
    optimizer: Optimizer
    iteration: int
    total_steps: int
    klent_config: KlentConfig
    model_preset: str
    last_shard: str | None = None
    scaler: Any | None = None


def save_klent_checkpoint(state: KlentTrainState, path: str | Path) -> Path:
    torch = _import_torch()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "algorithm": KLENT_ALGORITHM,
            "iteration": state.iteration,
            "total_steps": state.total_steps,
            "model_preset": state.model_preset,
            "model_config": asdict(state.model.config),
            "klent_config": asdict(state.klent_config),
            "model_state": state.model.state_dict(),
            "optimizer": _optimizer_type(state.optimizer),
            "optimizer_state": state.optimizer.state_dict(),
            "scaler_state": None if state.scaler is None else state.scaler.state_dict(),
            "last_shard": state.last_shard,
        },
        destination,
    )
    return destination


def load_klent_checkpoint(
    path: str | Path,
    *,
    device: str | None = None,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    optimizer: str = "adamw",
    amp: bool = False,
) -> KlentTrainState:
    torch = _import_torch()
    from great_kingdom_ai.model import ModelConfig, PolicyValueNetwork

    checkpoint = torch.load(Path(path), map_location=device or "cpu", weights_only=False)
    _validate_klent_checkpoint(checkpoint)
    model_config = ModelConfig(**checkpoint["model_config"])
    if not model_config.action_value_head:
        raise ValueError(
            "strict KLENT resume requires an action_value_head checkpoint; "
            "use warm_start_klent_model for state-value checkpoints"
        )
    model = PolicyValueNetwork(model_config).to(device=device)
    model.load_state_dict(checkpoint["model_state"])

    train_config = TrainingConfig(
        optimizer=optimizer,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=str(device or "cpu"),
    )
    optimizer_instance = create_optimizer(torch, model, train_config)
    if _optimizer_type(optimizer_instance) == str(checkpoint.get("optimizer", "")).lower():
        optimizer_instance.load_state_dict(checkpoint["optimizer_state"])

    scaler: Any | None = None
    if _cuda_amp_enabled(torch, device, enabled=amp):
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        scaler_state = checkpoint.get("scaler_state")
        if scaler_state is not None:
            scaler.load_state_dict(scaler_state)
    return KlentTrainState(
        model=model,
        optimizer=optimizer_instance,
        iteration=int(checkpoint["iteration"]),
        total_steps=int(checkpoint.get("total_steps", 0)),
        klent_config=KlentConfig(**checkpoint["klent_config"]),
        model_preset=str(checkpoint.get("model_preset", "custom")),
        last_shard=(
            None if checkpoint.get("last_shard") is None else str(checkpoint["last_shard"])
        ),
        scaler=scaler,
    )


def warm_start_klent_model(
    path: str | Path,
    *,
    model_preset: str | None = None,
    action_value_head: bool = True,
    device: str | None = None,
) -> PolicyValueNetwork:
    """Copy matching backbone/policy weights from a checkpoint into a new Q-head model.

    Action-value heads that are missing from the source are freshly initialized.
    State-value heads in the source are ignored.
    """
    torch = _import_torch()
    from great_kingdom_ai.model import MODEL_PRESETS, ModelConfig, PolicyValueNetwork

    checkpoint = torch.load(Path(path), map_location=device or "cpu", weights_only=False)
    base_config = dict(checkpoint["model_config"])
    if model_preset is not None:
        if model_preset not in MODEL_PRESETS:
            raise ValueError(f"unknown model preset {model_preset!r}")
        base_config = {**asdict(MODEL_PRESETS[model_preset]), **base_config}
    target_config = ModelConfig(
        **{**base_config, "action_value_head": action_value_head, "terminal_board_head": False}
    )
    model = PolicyValueNetwork(target_config).to(device=device)
    source_state = checkpoint["model_state"]
    if not isinstance(source_state, dict):
        raise ValueError("checkpoint model_state must be a mapping")

    target_state = model.state_dict()
    action_value_keys = {
        key for key in target_state if key.startswith(("q_spatial.", "q_pass."))
    }
    missing: list[str] = []
    copied: dict[str, Any] = {}
    for key, target_tensor in target_state.items():
        source_tensor = source_state.get(key)
        if source_tensor is None:
            if key not in action_value_keys:
                missing.append(key)
            continue
        if tuple(source_tensor.shape) != tuple(target_tensor.shape):
            missing.append(key)
            continue
        copied[key] = source_tensor
    if missing:
        raise ValueError(f"warm-start weights are missing or incompatible: {sorted(missing)}")

    model.load_state_dict(copied, strict=False)
    return model


def _validate_klent_checkpoint(checkpoint: Any) -> None:
    if not isinstance(checkpoint, dict):
        raise ValueError("KLENT checkpoint must contain a mapping")
    if checkpoint.get("algorithm") != KLENT_ALGORITHM:
        raise ValueError(
            f"checkpoint algorithm must be {KLENT_ALGORITHM!r}, "
            f"got {checkpoint.get('algorithm')!r}"
        )


def _optimizer_type(optimizer: Optimizer) -> str:
    name = optimizer.__class__.__name__.lower()
    if name == "sgd":
        return "sgd"
    if name == "adamw":
        return "adamw"
    return name


__all__ = [
    "KlentTrainState",
    "load_klent_checkpoint",
    "save_klent_checkpoint",
    "warm_start_klent_model",
]