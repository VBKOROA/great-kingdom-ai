from __future__ import annotations

import importlib
import importlib.util
import random
from contextlib import nullcontext

import numpy as np
import pytest

_torch_spec = importlib.util.find_spec("torch")
pytestmark = pytest.mark.skipif(
    _torch_spec is None,
    reason="torch is not installed",
)
torch = importlib.import_module("torch") if _torch_spec is not None else None

import great_kingdom_ai.training.loop as train_loop  # noqa: E402
from _training_helpers import (  # noqa: E402
    ArrayReplay,
    PriorityArrayReplay,
    RecencyReplay,
    make_replay,
    make_sample,
    make_weighted_sample,
)
from great_kingdom_ai.features import ACTION_SPACE, BOARD_SIZE, FEATURE_CHANNELS  # noqa: E402
from great_kingdom_ai.replay_buffer import ReplaySample  # noqa: E402
from great_kingdom_ai.training.batch import (  # noqa: E402
    TrainingArrays,
    arrays_to_batch,
    samples_to_batch,
)
from great_kingdom_ai.training.checkpoint import create_train_state, save_checkpoint  # noqa: E402
from great_kingdom_ai.training.config import TrainingConfig  # noqa: E402
from great_kingdom_ai.training.loop import (  # noqa: E402
    compute_losses,
    train_from_replay,
    train_step,
)


def test_compute_losses_returns_policy_value_and_regularization_terms() -> None:
    config = TrainingConfig(batch_size=2, l2_loss_weight=1e-6)
    state = create_train_state(config)
    samples = [make_sample(0), make_sample(1, value=-1.0)]
    batch = samples_to_batch(samples)

    losses = compute_losses(state.model, batch, l2_loss_weight=config.l2_loss_weight)

    assert losses.policy.item() > 0.0
    assert losses.policy_entropy.item() >= 0.0
    assert losses.policy_kl.item() >= 0.0
    assert losses.value.item() >= 0.0
    assert losses.regularization.item() > 0.0
    assert losses.total.item() >= losses.policy.item()


def test_compute_losses_uses_sample_weights() -> None:
    config = TrainingConfig(batch_size=2)
    state = create_train_state(config)
    samples = [
        make_weighted_sample(0, value=1.0, sample_weight=1.0),
        make_weighted_sample(1, value=-1.0, sample_weight=5.0),
    ]
    batch = samples_to_batch(samples)
    unweighted_batch = type(batch)(
        features=batch.features,
        policy=batch.policy,
        value=batch.value,
        legal_mask=batch.legal_mask,
        sample_weight=torch.ones_like(batch.sample_weight),
    )

    weighted = compute_losses(state.model, batch)
    unweighted = compute_losses(state.model, unweighted_batch)

    assert batch.sample_weight.tolist() == pytest.approx([1.0, 5.0])
    assert weighted.total.item() != pytest.approx(unweighted.total.item())


def test_arrays_to_batch_uses_precomputed_legal_masks() -> None:
    config = TrainingConfig(batch_size=1)
    state = create_train_state(config)
    features = np.zeros((1, FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    policies = np.zeros((1, ACTION_SPACE), dtype=np.float32)
    policies[0, 0] = 1.0
    legal_masks = np.zeros((1, ACTION_SPACE), dtype=np.bool_)
    legal_masks[0, 0] = True

    batch = arrays_to_batch(
        TrainingArrays(
            features=features,
            policies=policies,
            values=np.asarray([0.0], dtype=np.float32),
            sample_weights=np.asarray([1.0], dtype=np.float32),
            legal_masks=legal_masks,
        )
    )
    losses = compute_losses(state.model, batch, mask_policy_loss=True)

    assert losses.total.item() > 0.0


def test_train_step_updates_model_parameters() -> None:
    config = TrainingConfig(batch_size=2, steps=1, seed=3)
    state = create_train_state(config)
    batch = samples_to_batch(make_replay().sample(2, random.Random(3)))
    before = next(state.model.parameters()).detach().clone()

    train_step(state, batch, config)
    after = next(state.model.parameters()).detach()

    assert not torch.equal(before, after)


def test_train_step_updates_ema_model_after_optimizer_step() -> None:
    config = TrainingConfig(batch_size=2, steps=1, seed=3, ema_decay=0.5)
    state = create_train_state(config)
    assert state.ema_model is not None
    batch = samples_to_batch(make_replay().sample(2, random.Random(3)))
    before_model_state = {
        key: value.detach().clone()
        for key, value in state.model.state_dict().items()
        if value.is_floating_point()
    }

    train_step(state, batch, config)

    model_state = state.model.state_dict()
    ema_state = state.ema_model.state_dict()
    first_key = next(iter(before_model_state))
    expected = before_model_state[first_key] * 0.5 + model_state[first_key] * 0.5
    assert torch.allclose(ema_state[first_key], expected)



def test_amp_is_disabled_without_cuda_device() -> None:
    state = create_train_state(TrainingConfig(batch_size=2, amp=True, device="cpu"))

    assert state.scaler is None


def test_train_step_skips_scheduler_when_amp_optimizer_step_is_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeScaledLoss:
        def __init__(self, loss: torch.Tensor) -> None:
            self.loss = loss

        def backward(self) -> None:
            self.loss.backward()

    class SkippingScaler:
        def __init__(self) -> None:
            self.scale_value = 2.0

        def get_scale(self) -> float:
            return self.scale_value

        def scale(self, loss: torch.Tensor) -> FakeScaledLoss:
            return FakeScaledLoss(loss)

        def step(self, optimizer: torch.optim.Optimizer) -> None:
            del optimizer

        def update(self) -> None:
            self.scale_value = 1.0

    config = TrainingConfig(
        batch_size=2,
        amp=True,
        lr_schedule="step",
        lr_decay_steps=1,
        lr_decay_gamma=0.1,
    )
    state = create_train_state(config)
    state = type(state)(
        model=state.model,
        optimizer=state.optimizer,
        scheduler=state.scheduler,
        scaler=SkippingScaler(),
        step=state.step,
        model_preset=state.model_preset,
    )
    batch = samples_to_batch(make_replay().sample(2, random.Random(3)))
    monkeypatch.setattr(train_loop, "_amp_enabled", lambda config: True)
    monkeypatch.setattr(train_loop, "_autocast_context", lambda torch, *, enabled: nullcontext())
    learning_rate = state.optimizer.param_groups[0]["lr"]

    train_step(state, batch, config)

    assert state.optimizer.param_groups[0]["lr"] == pytest.approx(learning_rate)
    assert state.scheduler.state_dict()["last_epoch"] == 0


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
    assert "policy_kl" in resumed.losses[-1]
    assert resumed.losses[-1]["train_step_seconds"] > 0.0


def test_train_from_replay_can_override_resumed_optimizer_lr_once(tmp_path) -> None:
    replay = make_replay(size=8)
    first_checkpoint = tmp_path / "first.pt"
    second_checkpoint = tmp_path / "second.pt"

    train_from_replay(
        replay,
        TrainingConfig(batch_size=4, steps=2, learning_rate=1e-3, seed=11),
        checkpoint_path=first_checkpoint,
    )
    train_from_replay(
        replay,
        TrainingConfig(batch_size=4, steps=1, learning_rate=1e-3, seed=11),
        checkpoint_path=second_checkpoint,
        resume_path=first_checkpoint,
        resume_optimizer_lr_override=5e-5,
    )

    checkpoint = torch.load(second_checkpoint, map_location="cpu", weights_only=False)
    assert checkpoint["optimizer_state"]["param_groups"][0]["lr"] == pytest.approx(5e-5)
    assert checkpoint["scheduler_state"]["_last_lr"][0] == pytest.approx(5e-5)
    assert checkpoint["scheduler_state"]["base_lrs"][0] == pytest.approx(5e-5)


def test_train_from_replay_rejects_optimizer_lr_override_without_resume(tmp_path) -> None:
    replay = make_replay(size=8)

    with pytest.raises(ValueError, match="requires resume_path"):
        train_from_replay(
            replay,
            TrainingConfig(batch_size=4, steps=1),
            resume_optimizer_lr_override=5e-5,
        )


def test_train_from_replay_can_bootstrap_weights_without_resuming_step(tmp_path) -> None:
    replay = make_replay(size=8)
    first_checkpoint = tmp_path / "first.pt"
    second_checkpoint = tmp_path / "second.pt"
    config = TrainingConfig(batch_size=4, steps=2, seed=11)

    first = train_from_replay(replay, config, checkpoint_path=first_checkpoint)
    bootstrapped = train_from_replay(
        replay,
        TrainingConfig(batch_size=4, steps=1, seed=11),
        checkpoint_path=second_checkpoint,
        bootstrap_weights_path=first_checkpoint,
        log_every=1,
    )

    assert first.end_step == 2
    assert bootstrapped.start_step == 0
    assert bootstrapped.end_step == 1
    assert second_checkpoint.is_file()
    assert bootstrapped.losses[-1]["total"] > 0.0


def test_train_from_replay_rejects_resume_and_weight_bootstrap_together(tmp_path) -> None:
    replay = make_replay(size=8)
    checkpoint = tmp_path / "checkpoint.pt"
    save_checkpoint(create_train_state(TrainingConfig()), checkpoint)

    with pytest.raises(ValueError, match="mutually exclusive"):
        train_from_replay(
            replay,
            TrainingConfig(batch_size=4, steps=1),
            resume_path=checkpoint,
            bootstrap_weights_path=checkpoint,
        )


def test_train_from_replay_uses_recency_biased_sampler() -> None:
    replay = RecencyReplay()
    config = TrainingConfig(
        batch_size=2,
        steps=1,
        recent_sample_fraction=0.5,
        recent_sample_window=3,
    )

    train_from_replay(replay, config)

    assert replay.calls == [(2, 0.5, 3)]


def test_train_from_replay_uses_array_sampler_when_available() -> None:
    replay = ArrayReplay()
    config = TrainingConfig(
        batch_size=2,
        steps=1,
        recent_sample_fraction=0.5,
        recent_sample_window=3,
    )

    train_from_replay(replay, config)

    assert replay.calls == [(2, 0.5, 3)]


def test_train_from_replay_passes_priority_config_to_array_sampler() -> None:
    replay = PriorityArrayReplay()
    config = TrainingConfig(
        batch_size=2,
        steps=1,
        recent_sample_fraction=0.5,
        recent_sample_window=3,
        priority_enabled=True,
        priority_alpha=0.4,
        priority_beta=0.2,
    )

    train_from_replay(replay, config)

    assert replay.calls == [(2, 0.5, 3, True, 0.4, 0.2)]

def test_masked_policy_loss_rejects_illegal_target_mass() -> None:
    config = TrainingConfig(batch_size=1)
    state = create_train_state(config)
    features = np.zeros((FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    policy = np.zeros(ACTION_SPACE, dtype=np.float32)
    policy[0] = 1.0
    batch = samples_to_batch([ReplaySample(features=features, policy=policy, value=0.0)])

    with pytest.raises(ValueError, match="illegal actions"):
        compute_losses(state.model, batch, mask_policy_loss=True)
