from __future__ import annotations

import importlib
import importlib.util
import random
from pathlib import Path

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
    TrainingArrays,
    TrainingConfig,
    build_parser,
    compute_losses,
    create_train_state,
    load_checkpoint,
    load_checkpoint_weights,
    print_training_startup_config,
    samples_to_batch,
    save_checkpoint,
    train_from_replay,
    train_step,
)


def make_sample(index: int, value: float = 1.0) -> ReplaySample:
    features = np.zeros((FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    features[index % FEATURE_CHANNELS, index % BOARD_SIZE, (index * 3) % BOARD_SIZE] = 1.0
    features[4, :, :] = 1.0
    policy = np.zeros(ACTION_SPACE, dtype=np.float32)
    policy[index % ACTION_SPACE] = 1.0
    return ReplaySample(features=features, policy=policy, value=value)


def make_weighted_sample(index: int, value: float, sample_weight: float) -> ReplaySample:
    sample = make_sample(index, value=value)
    return ReplaySample(
        features=sample.features,
        policy=sample.policy,
        value=sample.value,
        sample_weight=sample_weight,
    )


def make_replay(size: int = 6) -> ReplayBuffer:
    buffer = ReplayBuffer(capacity=size)
    for index in range(size):
        buffer.push(make_sample(index, value=1.0 if index % 2 else -1.0))
    return buffer


class RecencyReplay:
    def __init__(self) -> None:
        self.calls: list[tuple[int, float, int]] = []

    def __len__(self) -> int:
        return 4

    def sample(self, batch_size: int, rng: random.Random) -> list[ReplaySample]:
        del batch_size, rng
        raise AssertionError("uniform sample should not be used")

    def sample_recency_biased(
        self,
        batch_size: int,
        rng: random.Random,
        *,
        recent_fraction: float,
        recent_window: int,
    ) -> list[ReplaySample]:
        del rng
        self.calls.append((batch_size, recent_fraction, recent_window))
        return [make_sample(index) for index in range(batch_size)]


class ArrayReplay:
    def __init__(self) -> None:
        self.calls: list[tuple[int, float, int]] = []

    def __len__(self) -> int:
        return 4

    def sample(self, batch_size: int, rng: random.Random) -> list[ReplaySample]:
        del batch_size, rng
        raise AssertionError("sample should not be used when sample_arrays exists")

    def sample_arrays(
        self,
        batch_size: int,
        rng: random.Random,
        *,
        recent_fraction: float = 0.0,
        recent_window: int = 0,
    ) -> TrainingArrays:
        del rng
        self.calls.append((batch_size, recent_fraction, recent_window))
        samples = [
            make_sample(index, value=1.0 if index % 2 else -1.0)
            for index in range(batch_size)
        ]
        return TrainingArrays(
            features=np.stack([sample.features for sample in samples], axis=0).astype(np.float32),
            policies=np.stack([sample.policy for sample in samples], axis=0).astype(np.float32),
            values=np.asarray([sample.value for sample in samples], dtype=np.float32),
            sample_weights=np.ones((batch_size,), dtype=np.float32),
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


def test_train_step_updates_model_parameters() -> None:
    config = TrainingConfig(batch_size=2, steps=1, seed=3)
    state = create_train_state(config)
    batch = samples_to_batch(make_replay().sample(2, random.Random(3)))
    before = next(state.model.parameters()).detach().clone()

    train_step(state, batch, config)
    after = next(state.model.parameters()).detach()

    assert not torch.equal(before, after)


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


def test_amp_is_disabled_without_cuda_device() -> None:
    state = create_train_state(TrainingConfig(batch_size=2, amp=True, device="cpu"))

    assert state.scaler is None


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


def test_train_parser_accepts_log_every_override() -> None:
    args = build_parser().parse_args(
        [
            "--replay",
            "replay.npz",
            "--checkpoint",
            "checkpoint.pt",
            "--log-every",
            "100",
        ]
    )

    assert args.log_every == 100


def test_train_parser_accepts_weight_bootstrap_checkpoint() -> None:
    args = build_parser().parse_args(
        [
            "--replay",
            "replay.npz",
            "--checkpoint",
            "checkpoint.pt",
            "--bootstrap-weights",
            "best.pt",
        ]
    )

    assert args.bootstrap_weights == Path("best.pt")


def test_masked_policy_loss_rejects_illegal_target_mass() -> None:
    config = TrainingConfig(batch_size=1)
    state = create_train_state(config)
    features = np.zeros((FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    policy = np.zeros(ACTION_SPACE, dtype=np.float32)
    policy[0] = 1.0
    batch = samples_to_batch([ReplaySample(features=features, policy=policy, value=0.0)])

    with pytest.raises(ValueError, match="illegal actions"):
        compute_losses(state.model, batch, mask_policy_loss=True)


def test_print_training_startup_config_outputs_effective_settings(tmp_path, capsys) -> None:
    replay = make_replay(size=3)
    config = TrainingConfig(
        batch_size=2,
        steps=5,
        device="cuda",
        symmetry_augmentation=False,
    )

    print_training_startup_config(
        config=config,
        replay=replay,
        replay_path=tmp_path / "replay.npz",
        checkpoint_path=tmp_path / "checkpoint.pt",
        resume_path=None,
    )

    output = capsys.readouterr().out
    assert '"event": "train_config"' in output
    assert '"batch_size": 2' in output
    assert '"device": "cuda"' in output
    assert '"symmetry_augmentation": false' in output
    assert '"samples": 3' in output
