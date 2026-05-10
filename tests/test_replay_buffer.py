import random

import numpy as np
import pytest
from great_kingdom_ai.features import ACTION_SPACE, BOARD_SIZE, FEATURE_CHANNELS
from great_kingdom_ai.priority_sampling import PrioritySamplingConfig
from great_kingdom_ai.replay_buffer import ReplayBuffer, ReplaySample


def make_sample(
    value: float = 1.0,
    action: int = 0,
    sample_weight: float = 1.0,
) -> ReplaySample:
    features = np.zeros((FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    features[0, 0, action % BOARD_SIZE] = 1.0
    policy = np.zeros(ACTION_SPACE, dtype=np.float32)
    policy[action] = 1.0
    return ReplaySample(
        features=features,
        policy=policy,
        value=value,
        sample_weight=sample_weight,
    )


def make_sample_with_root_logits(action: int = 0) -> ReplaySample:
    sample = make_sample(action=action)
    root_logits = np.full(ACTION_SPACE, -5.0, dtype=np.float32)
    root_logits[action] = 3.0
    return ReplaySample(
        features=sample.features,
        policy=sample.policy,
        value=sample.value,
        root_policy_logits=root_logits,
    )


def test_replay_buffer_push_sample_and_capacity() -> None:
    buffer = ReplayBuffer(capacity=2)
    buffer.push(make_sample(value=1.0, action=1))
    buffer.push(make_sample(value=-1.0, action=2))
    buffer.push(make_sample(value=1.0, action=3))

    batch = buffer.sample(2, random.Random(11))

    assert len(buffer) == 2
    assert {int(np.argmax(sample.policy)) for sample in batch} == {2, 3}


def test_replay_buffer_rejects_invalid_policy_target() -> None:
    sample = make_sample()
    sample.policy[1] = 1.0

    with pytest.raises(ValueError, match="sum to 1"):
        ReplayBuffer(capacity=1).push(sample)


def test_replay_buffer_save_and_load_round_trip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "replay.npz"
    buffer = ReplayBuffer(capacity=4)
    buffer.push(make_sample(value=1.0, action=5))
    buffer.push(make_sample(value=-1.0, action=81))

    buffer.save(path)
    loaded = ReplayBuffer.load(path)
    batch = loaded.sample(2, random.Random(1))

    assert len(loaded) == 2
    assert loaded.capacity == 4
    assert {sample.value for sample in batch} == {1.0, -1.0}
    assert {int(np.argmax(sample.policy)) for sample in batch} == {5, 81}


def test_replay_buffer_save_and_load_root_policy_logits(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "replay.npz"
    buffer = ReplayBuffer(capacity=4)
    buffer.push(make_sample_with_root_logits(action=5))

    buffer.save(path)
    loaded = ReplayBuffer.load(path)
    batch = loaded.sample(1, random.Random(1))

    assert batch[0].root_policy_logits is not None
    assert batch[0].root_policy_logits[5] == pytest.approx(3.0)


def test_replay_buffer_save_and_load_sample_weights(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "replay.npz"
    buffer = ReplayBuffer(capacity=4)
    buffer.push(make_sample(value=1.0, action=5, sample_weight=3.5))

    buffer.save(path)
    loaded = ReplayBuffer.load(path)
    batch = loaded.sample(1, random.Random(1))

    assert batch[0].sample_weight == pytest.approx(3.5)


def test_replay_buffer_rejects_invalid_sample_weight() -> None:
    with pytest.raises(ValueError, match="sample_weight"):
        ReplayBuffer(capacity=1).push(make_sample(sample_weight=0.0))


def test_replay_buffer_can_sample_priority_biased_rows() -> None:
    buffer = ReplayBuffer(capacity=3)
    buffer.push(make_sample(value=1.0, action=1, sample_weight=1.0))
    buffer.push(make_sample(value=1.0, action=2, sample_weight=1.0))
    buffer.push(make_sample(value=1.0, action=3, sample_weight=1000.0))

    batch = buffer.sample_priority_biased(
        1,
        random.Random(0),
        priority_config=PrioritySamplingConfig(enabled=True, alpha=1.0, beta=0.4),
    )

    assert int(np.argmax(batch[0].policy)) == 3
    assert 0.0 < batch[0].sample_weight <= 1000.0


def test_replay_buffer_samples_contiguous_training_arrays_with_legal_masks() -> None:
    buffer = ReplayBuffer(capacity=3)
    buffer.push(make_sample(value=1.0, action=1, sample_weight=1.0))
    buffer.push(make_sample(value=1.0, action=2, sample_weight=3.0))

    batch = buffer.sample_arrays(2, random.Random(0))

    assert batch.features.flags.c_contiguous
    assert batch.policies.flags.c_contiguous
    assert batch.values.flags.c_contiguous
    assert batch.sample_weights.flags.c_contiguous
    assert batch.legal_masks.shape == (2, ACTION_SPACE)
    assert batch.legal_masks[:, 81].all()


def test_replay_buffer_sample_arrays_supports_priority_weights() -> None:
    buffer = ReplayBuffer(capacity=3)
    buffer.push(make_sample(value=1.0, action=1, sample_weight=1.0))
    buffer.push(make_sample(value=1.0, action=2, sample_weight=1000.0))

    batch = buffer.sample_arrays(
        1,
        random.Random(0),
        priority_config=PrioritySamplingConfig(enabled=True, alpha=1.0, beta=0.4),
    )

    assert int(np.argmax(batch.policies[0])) == 2
    assert 0.0 < batch.sample_weights[0] <= 1000.0
