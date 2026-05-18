from __future__ import annotations

import importlib
import importlib.util
import random

import pytest

_torch_spec = importlib.util.find_spec("torch")
pytestmark = pytest.mark.skipif(
    _torch_spec is None,
    reason="torch is not installed",
)
torch = importlib.import_module("torch") if _torch_spec is not None else None

from _training_helpers import make_replay  # noqa: E402
from great_kingdom_ai.training.batch import samples_to_batch  # noqa: E402
from great_kingdom_ai.training.checkpoint import (  # noqa: E402
    create_train_state,
    load_checkpoint,
    load_checkpoint_weights,
    save_checkpoint,
    summarize_checkpoint_optimizer_state,
)
from great_kingdom_ai.training.config import TrainingConfig  # noqa: E402
from great_kingdom_ai.training.loop import train_step  # noqa: E402


def test_warmup_cosine_scheduler_changes_learning_rate(tmp_path) -> None:
    config = TrainingConfig(
        batch_size=2,
        steps=4,
        learning_rate=1.0,
        lr_schedule="warmup_cosine",
        lr_warmup_steps=2,
        lr_min_factor=0.1,
        seed=3,
    )
    state = create_train_state(config)
    batch = samples_to_batch(make_replay().sample(2, random.Random(3)))

    assert state.optimizer.param_groups[0]["lr"] == pytest.approx(0.5)

    for _ in range(config.steps):
        train_step(state, batch, config)

    assert state.optimizer.param_groups[0]["lr"] == pytest.approx(0.1)
    checkpoint = save_checkpoint(state, tmp_path / "warmup-cosine.pt")
    loaded = load_checkpoint(
        checkpoint,
        lr_schedule=config.lr_schedule,
        lr_warmup_steps=config.lr_warmup_steps,
        lr_min_factor=config.lr_min_factor,
        steps=config.steps,
    )

    assert loaded.optimizer.param_groups[0]["lr"] == pytest.approx(0.1)


def test_constant_with_warmup_scheduler_keeps_learning_rate_after_warmup(tmp_path) -> None:
    config = TrainingConfig(
        batch_size=2,
        steps=4,
        learning_rate=1.0,
        lr_schedule="constant_with_warmup",
        lr_warmup_steps=2,
        seed=3,
    )
    state = create_train_state(config)
    batch = samples_to_batch(make_replay().sample(2, random.Random(3)))

    assert state.optimizer.param_groups[0]["lr"] == pytest.approx(0.5)

    for _ in range(config.steps):
        train_step(state, batch, config)

    assert state.optimizer.param_groups[0]["lr"] == pytest.approx(1.0)
    checkpoint = save_checkpoint(state, tmp_path / "constant-with-warmup.pt")
    loaded = load_checkpoint(
        checkpoint,
        lr_schedule=config.lr_schedule,
        lr_warmup_steps=config.lr_warmup_steps,
        steps=config.steps,
    )

    assert loaded.optimizer.param_groups[0]["lr"] == pytest.approx(1.0)



def test_checkpoint_round_trips_model_outputs_and_optimizer_state(tmp_path) -> None:
    config = TrainingConfig(batch_size=2, steps=1, seed=5)
    state = create_train_state(config)
    batch = samples_to_batch(make_replay().sample(2, random.Random(5)))
    train_step(state, batch, config)
    state = type(state)(
        model=state.model,
        optimizer=state.optimizer,
        scheduler=state.scheduler,
        step=1,
        model_preset=state.model_preset,
    )
    state.model.eval()
    inputs = batch.features
    with torch.no_grad():
        expected_policy, expected_value = state.model(inputs)

    checkpoint_path = save_checkpoint(state, tmp_path / "checkpoint.pt")
    loaded = load_checkpoint(checkpoint_path)
    loaded.model.eval()
    with torch.no_grad():
        actual_policy, actual_value = loaded.model(inputs)

    assert loaded.step == 1
    assert loaded.model_preset == "small"
    assert torch.allclose(actual_policy, expected_policy)
    assert torch.allclose(actual_value, expected_value)
    assert loaded.optimizer.state_dict()["state"]


def test_create_train_state_supports_sgd_optimizer() -> None:
    config = TrainingConfig(
        optimizer="sgd",
        learning_rate=0.02,
        momentum=0.9,
        weight_decay=1e-4,
    )
    state = create_train_state(config)

    assert isinstance(state.optimizer, torch.optim.SGD)
    assert state.optimizer.param_groups[0]["lr"] == pytest.approx(0.02)
    assert state.optimizer.param_groups[0]["momentum"] == pytest.approx(0.9)
    assert state.optimizer.param_groups[0]["weight_decay"] == pytest.approx(1e-4)


def test_load_checkpoint_resets_optimizer_state_when_optimizer_changes(tmp_path) -> None:
    save_config = TrainingConfig(batch_size=2, steps=1, seed=5)
    state = create_train_state(save_config)
    batch = samples_to_batch(make_replay().sample(2, random.Random(5)))
    train_step(state, batch, save_config)
    checkpoint_path = save_checkpoint(
        type(state)(
            model=state.model,
            optimizer=state.optimizer,
            scheduler=state.scheduler,
            step=4,
            model_preset=state.model_preset,
        ),
        tmp_path / "adamw.pt",
    )

    loaded = load_checkpoint(
        checkpoint_path,
        optimizer="sgd",
        learning_rate=0.02,
        momentum=0.9,
        weight_decay=1e-4,
    )

    assert loaded.step == 4
    assert isinstance(loaded.optimizer, torch.optim.SGD)
    assert loaded.optimizer.state_dict()["state"] == {}
    assert loaded.optimizer.param_groups[0]["lr"] == pytest.approx(0.02)


def test_summarize_checkpoint_optimizer_state_reports_adam_buffers(tmp_path) -> None:
    config = TrainingConfig(batch_size=2, steps=1, seed=5)
    state = create_train_state(config)
    batch = samples_to_batch(make_replay().sample(2, random.Random(5)))
    train_step(state, batch, config)
    state = type(state)(
        model=state.model,
        optimizer=state.optimizer,
        scheduler=state.scheduler,
        step=1,
        model_preset=state.model_preset,
    )

    checkpoint_path = save_checkpoint(state, tmp_path / "checkpoint.pt")
    summary = summarize_checkpoint_optimizer_state(checkpoint_path)

    assert summary["checkpoint_step"] == 1
    assert summary["state_entries"] > 0
    assert summary["step"]["max"] == pytest.approx(1.0)
    assert summary["param_groups"][0]["lr"] > 0.0
    assert summary["tensor_buffers"]["exp_avg"]["tensors"] > 0
    assert summary["tensor_buffers"]["exp_avg_sq"]["elements"] > 0


def test_checkpoint_preserves_ema_and_can_prefer_ema_weights(tmp_path) -> None:
    config = TrainingConfig(ema_decay=0.9)
    state = create_train_state(config)
    assert state.ema_model is not None
    with torch.no_grad():
        for tensor in state.model.state_dict().values():
            if tensor.is_floating_point():
                tensor.fill_(3.0)
        for tensor in state.ema_model.state_dict().values():
            if tensor.is_floating_point():
                tensor.fill_(1.0)

    checkpoint_path = save_checkpoint(state, tmp_path / "ema.pt")
    loaded = load_checkpoint(checkpoint_path)
    preferred = load_checkpoint(checkpoint_path, prefer_ema=True)

    assert loaded.ema_model is not None
    assert loaded.ema_decay == pytest.approx(0.9)
    loaded_first = next(
        tensor for tensor in loaded.model.state_dict().values() if tensor.is_floating_point()
    )
    preferred_first = next(
        tensor for tensor in preferred.model.state_dict().values() if tensor.is_floating_point()
    )
    assert torch.allclose(loaded_first, torch.full_like(loaded_first, 3.0))
    assert torch.allclose(preferred_first, torch.full_like(preferred_first, 1.0))


def test_checkpoint_weight_bootstrap_keeps_model_and_resets_training_state(tmp_path) -> None:
    save_config = TrainingConfig(batch_size=2, steps=1, seed=5)
    state = create_train_state(save_config)
    batch = samples_to_batch(make_replay().sample(2, random.Random(5)))
    train_step(state, batch, save_config)
    state = type(state)(
        model=state.model,
        optimizer=state.optimizer,
        scheduler=state.scheduler,
        step=7,
        model_preset=state.model_preset,
    )
    state.model.eval()
    with torch.no_grad():
        expected_policy, expected_value = state.model(batch.features)

    checkpoint_path = save_checkpoint(state, tmp_path / "checkpoint.pt")
    bootstrap_config = TrainingConfig(
        learning_rate=5e-4,
        weight_decay=2e-3,
        lr_decay_steps=17,
        lr_decay_gamma=0.8,
    )
    loaded = load_checkpoint_weights(checkpoint_path, bootstrap_config)
    loaded.model.eval()
    with torch.no_grad():
        actual_policy, actual_value = loaded.model(batch.features)

    assert loaded.step == 0
    assert loaded.model_preset == "small"
    assert torch.allclose(actual_policy, expected_policy)
    assert torch.allclose(actual_value, expected_value)
    assert loaded.optimizer.state_dict()["state"] == {}
    assert loaded.optimizer.param_groups[0]["lr"] == pytest.approx(5e-4)
    assert loaded.optimizer.param_groups[0]["weight_decay"] == pytest.approx(2e-3)
    assert loaded.scheduler.state_dict()["step_size"] == 17
    assert loaded.scheduler.state_dict()["gamma"] == pytest.approx(0.8)
