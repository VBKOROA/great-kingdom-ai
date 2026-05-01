import random

import numpy as np
import pytest
from great_kingdom_ai.features import ACTION_SPACE, BOARD_SIZE, FEATURE_CHANNELS
from great_kingdom_ai.replay_buffer import ReplayBuffer, ReplaySample


def make_sample(value: float = 1.0, action: int = 0) -> ReplaySample:
    features = np.zeros((FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    features[0, 0, action % BOARD_SIZE] = 1.0
    policy = np.zeros(ACTION_SPACE, dtype=np.float32)
    policy[action] = 1.0
    return ReplaySample(features=features, policy=policy, value=value)


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
