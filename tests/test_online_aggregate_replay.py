from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
from great_kingdom_ai.features import ACTION_SPACE, BOARD_SIZE, FEATURE_CHANNELS
from great_kingdom_ai.online_aggregate_replay import OnlineAggregateReplayBuffer
from great_kingdom_ai.replay_buffer import ReplayBuffer, ReplaySample


def make_sample(
    feature_index: int,
    action: int,
    *,
    value: float = 1.0,
) -> ReplaySample:
    features = np.zeros((FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    features[feature_index % FEATURE_CHANNELS, 0, 0] = 1.0
    policy = np.zeros(ACTION_SPACE, dtype=np.float32)
    policy[action] = 1.0
    root_policy_logits = np.full(ACTION_SPACE, -4.0, dtype=np.float32)
    root_policy_logits[action] = 4.0
    return ReplaySample(
        features=features,
        policy=policy,
        value=value,
        root_policy_logits=root_policy_logits,
    )


def test_online_aggregate_replay_averages_duplicate_targets() -> None:
    replay = OnlineAggregateReplayBuffer(
        capacity=4,
        sample_weight_mode="log_count",
        sample_weight_cap=None,
    )

    replay.push(make_sample(0, 0, value=1.0))
    replay.push(make_sample(0, 1, value=-1.0))
    replay.push(make_sample(1, 2, value=1.0))

    batch = replay.sample(2, random.Random(3))
    duplicate = next(sample for sample in batch if sample.policy[0] > 0.0)

    assert len(replay) == 2
    assert replay.raw_sample_count == 3
    assert duplicate.policy[0] == pytest.approx(0.5)
    assert duplicate.policy[1] == pytest.approx(0.5)
    assert duplicate.value == pytest.approx(0.0)
    assert duplicate.sample_weight == pytest.approx(np.log1p(2.0))
    assert duplicate.root_policy_logits is not None
    assert duplicate.root_policy_logits[0] == pytest.approx(0.0)
    assert duplicate.root_policy_logits[1] == pytest.approx(0.0)


def test_online_aggregate_replay_save_load_preserves_counts(tmp_path: Path) -> None:
    path = tmp_path / "replay-aggregated.npz"
    replay = OnlineAggregateReplayBuffer(capacity=8, sample_weight_mode="sqrt_count")
    replay.push(make_sample(0, 0, value=1.0))
    replay.push(make_sample(0, 1, value=-1.0))
    replay.save(path)

    loaded = OnlineAggregateReplayBuffer.load(
        path,
        sample_weight_mode="sqrt_count",
    )
    loaded.push(make_sample(0, 2, value=1.0))
    loaded.save(path)

    as_training_replay = ReplayBuffer.load(path)
    sample = as_training_replay.sample(1, random.Random(0))[0]

    assert len(loaded) == 1
    assert loaded.raw_sample_count == 3
    assert sample.policy[0] == pytest.approx(1.0 / 3.0)
    assert sample.policy[1] == pytest.approx(1.0 / 3.0)
    assert sample.policy[2] == pytest.approx(1.0 / 3.0)
    assert sample.value == pytest.approx(1.0 / 3.0)
    assert sample.sample_weight == pytest.approx(np.sqrt(3.0))


def test_online_aggregate_replay_atomic_save_preserves_counts(tmp_path: Path) -> None:
    path = tmp_path / "replay-aggregated.npz"
    replay = OnlineAggregateReplayBuffer(capacity=8, sample_weight_mode="sqrt_count")
    replay.push(make_sample(0, 0, value=1.0))
    replay.push(make_sample(0, 1, value=-1.0))

    replay.save_atomic(path, compressed=False)

    loaded = OnlineAggregateReplayBuffer.load(path, sample_weight_mode="sqrt_count")
    assert len(loaded) == 1
    assert loaded.raw_sample_count == 2
    assert not (tmp_path / "replay-aggregated.npz.tmp").exists()


def test_online_aggregate_replay_capacity_counts_unique_rows() -> None:
    replay = OnlineAggregateReplayBuffer(capacity=2)
    replay.push(make_sample(0, 0))
    replay.push(make_sample(1, 1))
    replay.push(make_sample(0, 2))
    replay.push(make_sample(2, 3))

    batch = replay.sample(2, random.Random(5))

    assert len(replay) == 2
    assert any(sample.policy[0] == pytest.approx(0.5) for sample in batch)
    assert any(sample.policy[3] == pytest.approx(1.0) for sample in batch)


def test_online_aggregate_replay_can_sample_recent_rows() -> None:
    replay = OnlineAggregateReplayBuffer(capacity=8)
    for index in range(6):
        replay.push(make_sample(index, index))

    batch = replay.sample_recency_biased(
        2,
        random.Random(0),
        recent_fraction=1.0,
        recent_window=2,
    )

    assert {int(np.argmax(sample.policy)) for sample in batch} == {4, 5}


def test_online_aggregate_replay_samples_training_arrays() -> None:
    replay = OnlineAggregateReplayBuffer(capacity=8, sample_weight_mode="count")
    replay.push(make_sample(0, 0, value=1.0))
    replay.push(make_sample(0, 1, value=-1.0))
    replay.push(make_sample(1, 2, value=1.0))

    arrays = replay.sample_arrays(2, random.Random(3))

    assert arrays.features.shape == (2, FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    assert arrays.policies.shape == (2, ACTION_SPACE)
    assert arrays.values.shape == (2,)
    assert arrays.sample_weights.shape == (2,)
    duplicate_index = int(np.argmax(arrays.sample_weights))
    assert arrays.policies[duplicate_index, 0] == pytest.approx(0.5)
    assert arrays.policies[duplicate_index, 1] == pytest.approx(0.5)
    assert arrays.values[duplicate_index] == pytest.approx(0.0)
    assert arrays.sample_weights[duplicate_index] == pytest.approx(2.0)


def test_online_aggregate_replay_recency_updates_duplicate_rows() -> None:
    replay = OnlineAggregateReplayBuffer(capacity=8)
    replay.push(make_sample(0, 0))
    replay.push(make_sample(1, 1))
    replay.push(make_sample(2, 2))
    replay.push(make_sample(0, 3))

    batch = replay.sample_recency_biased(
        1,
        random.Random(0),
        recent_fraction=1.0,
        recent_window=1,
    )

    assert batch[0].policy[0] == pytest.approx(0.5)
    assert batch[0].policy[3] == pytest.approx(0.5)
