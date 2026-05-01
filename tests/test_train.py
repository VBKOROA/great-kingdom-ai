from __future__ import annotations

import importlib
import importlib.util
import random

import numpy as np
import pytest

_torch_spec = importlib.util.find_spec("torch")
pytestmark = pytest.mark.skipif(
    _torch_spec is None,
    reason="torch is not installed",
)
torch = importlib.import_module("torch") if _torch_spec is not None else None

from great_kingdom_ai.features import ACTION_SPACE, BOARD_SIZE, FEATURE_CHANNELS  # noqa: E402
from great_kingdom_ai.replay_buffer import ReplayBuffer, ReplaySample  # noqa: E402
from great_kingdom_ai.train import (  # noqa: E402
    TrainingConfig,
    compute_losses,
    create_train_state,
    load_checkpoint,
    samples_to_batch,
    save_checkpoint,
    train_from_replay,
    train_step,
)


def make_sample(index: int, value: float = 1.0) -> ReplaySample:
    features = np.zeros((FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    features[index % FEATURE_CHANNELS, index % BOARD_SIZE, (index * 3) % BOARD_SIZE] = 1.0
    policy = np.zeros(ACTION_SPACE, dtype=np.float32)
    policy[index % ACTION_SPACE] = 1.0
    return ReplaySample(features=features, policy=policy, value=value)


def make_replay(size: int = 6) -> ReplayBuffer:
    buffer = ReplayBuffer(capacity=size)
    for index in range(size):
        buffer.push(make_sample(index, value=1.0 if index % 2 else -1.0))
    return buffer


def test_compute_losses_returns_policy_value_and_regularization_terms() -> None:
    config = TrainingConfig(batch_size=2, l2_loss_weight=1e-6)
    state = create_train_state(config)
    samples = [make_sample(0), make_sample(1, value=-1.0)]
    batch = samples_to_batch(samples)

    losses = compute_losses(state.model, batch, l2_loss_weight=config.l2_loss_weight)

    assert losses.policy.item() > 0.0
    assert losses.value.item() >= 0.0
    assert losses.regularization.item() > 0.0
    assert losses.total.item() >= losses.policy.item()


def test_train_step_updates_model_parameters() -> None:
    config = TrainingConfig(batch_size=2, steps=1, seed=3)
    state = create_train_state(config)
    batch = samples_to_batch(make_replay().sample(2, random.Random(3)))
    before = next(state.model.parameters()).detach().clone()

    train_step(state, batch, config)
    after = next(state.model.parameters()).detach()

    assert not torch.equal(before, after)


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


def test_train_from_replay_saves_checkpoint_and_resume_advances_step(tmp_path) -> None:
    replay = make_replay(size=8)
    first_checkpoint = tmp_path / "first.pt"
    second_checkpoint = tmp_path / "second.pt"
    config = TrainingConfig(batch_size=4, steps=2, seed=11)

    first = train_from_replay(replay, config, checkpoint_path=first_checkpoint, log_every=1)
    resumed = train_from_replay(
        replay,
        TrainingConfig(batch_size=4, steps=1, seed=11),
        checkpoint_path=second_checkpoint,
        resume_path=first_checkpoint,
        log_every=1,
    )

    assert first.start_step == 0
    assert first.end_step == 2
    assert first_checkpoint.is_file()
    assert resumed.start_step == 2
    assert resumed.end_step == 3
    assert second_checkpoint.is_file()
    assert resumed.losses[-1]["total"] > 0.0
