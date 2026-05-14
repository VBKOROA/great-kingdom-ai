from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from great_kingdom_ai.features import ACTION_SPACE, BOARD_SIZE, FEATURE_CHANNELS, PASS_ACTION
from great_kingdom_ai.priority_sampling import PrioritySamplingConfig
from great_kingdom_ai.shard_indexed_dataset import ShardIndexedTrajectoryDataset
from great_kingdom_ai.shard_replay_index import ShardReplayIndex
from great_kingdom_ai.trajectory_dataset import TrajectoryReplayDataset
from great_kingdom_ai.trajectory_replay import (
    TrajectoryEpisode,
    TrajectoryReplayStore,
    TrajectoryTransition,
    legal_mask_from_features,
)


@dataclass(frozen=True)
class IndexSource:
    shard_id: str
    replay_path: Path
    log_path: Path
    model_version: str = "training-latest"
    model_path: Path = Path("model.onnx")
    seed_start: int = 0
    games: int = 1
    created_at: str = "2026-05-14T00:00:00+00:00"


def make_features(action: int = PASS_ACTION) -> np.ndarray:
    features = np.zeros((FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    if action != PASS_ACTION:
        features[4, action // BOARD_SIZE, action % BOARD_SIZE] = 1.0
    return features


def make_policy(action: int = PASS_ACTION) -> np.ndarray:
    policy = np.zeros((ACTION_SPACE,), dtype=np.float32)
    policy[action] = 1.0
    return policy


def make_transition(
    *,
    episode_id: int,
    timestep: int,
    player: int,
    action: int,
    winner: int,
) -> TrajectoryTransition:
    features = make_features(action)
    return TrajectoryTransition(
        episode_id=episode_id,
        timestep=timestep,
        player=player,
        features=features,
        legal_mask=legal_mask_from_features(features),
        action=action,
        policy_target=make_policy(action),
        winner=winner,
        terminal=False,
    )


def make_episode(episode_id: int, *, winner: int) -> TrajectoryEpisode:
    transitions = (
        make_transition(
            episode_id=episode_id,
            timestep=0,
            player=1,
            action=1 + episode_id,
            winner=winner,
        ),
        make_transition(
            episode_id=episode_id,
            timestep=1,
            player=2,
            action=PASS_ACTION,
            winner=winner,
        ),
    )
    return TrajectoryEpisode(
        episode_id=episode_id,
        seed=episode_id,
        transitions=transitions,
        winner=winner,
        end_reason=1,
        territory_scores=(1, 0),
    )


def write_shard(tmp_path: Path, shard_id: str, episode: TrajectoryEpisode) -> IndexSource:
    shard_dir = tmp_path / "shards" / shard_id
    shard_dir.mkdir(parents=True)
    replay_path = shard_dir / "trajectory-replay.npz"
    TrajectoryReplayStore.from_episodes(8, (episode,)).save(replay_path, compressed=False)
    log_path = shard_dir / "game_logs.json"
    log_path.write_text("[]", encoding="utf-8")
    return IndexSource(
        shard_id=shard_id,
        replay_path=replay_path,
        log_path=log_path,
        seed_start=episode.seed,
    )


def build_indexed_dataset(
    tmp_path: Path,
    episodes: tuple[TrajectoryEpisode, ...],
    *,
    cache_shards: int = 16,
) -> ShardIndexedTrajectoryDataset:
    index = ShardReplayIndex.load_or_create(tmp_path / "replay-index", capacity=32)
    for offset, episode in enumerate(episodes):
        index.add_shard(write_shard(tmp_path, f"shard-{offset}", episode))
    return ShardIndexedTrajectoryDataset(index, cache_shards=cache_shards)


def assert_batches_equal(left: object, right: object) -> None:
    assert left.indexes.tolist() == right.indexes.tolist()
    np.testing.assert_array_equal(left.features, right.features)
    np.testing.assert_array_equal(left.policies, right.policies)
    np.testing.assert_allclose(left.values, right.values)
    np.testing.assert_allclose(left.sample_weights, right.sample_weights)
    np.testing.assert_array_equal(left.legal_masks, right.legal_masks)


def test_shard_indexed_dataset_matches_monolithic_recency_sampling(tmp_path: Path) -> None:
    episodes = (make_episode(0, winner=1), make_episode(1, winner=2))
    monolithic = TrajectoryReplayDataset(TrajectoryReplayStore.from_episodes(8, episodes))
    sharded = build_indexed_dataset(tmp_path, episodes)

    monolithic_batch = monolithic.sample_arrays(
        2,
        random.Random(3),
        recent_fraction=1.0,
        recent_window=2,
    )
    sharded_batch = sharded.sample_arrays(
        2,
        random.Random(3),
        recent_fraction=1.0,
        recent_window=2,
    )

    assert sharded_batch.indexes.tolist() == [2, 3]
    assert sharded_batch.values.tolist() == pytest.approx([-1.0, 1.0])
    assert_batches_equal(sharded_batch, monolithic_batch)


def test_shard_indexed_dataset_gathers_batch_across_shards(tmp_path: Path) -> None:
    episodes = (make_episode(0, winner=1), make_episode(1, winner=2))
    monolithic = TrajectoryReplayDataset(TrajectoryReplayStore.from_episodes(8, episodes))
    sharded = build_indexed_dataset(tmp_path, episodes)

    monolithic_batch = monolithic.sample_arrays(4, random.Random(11))
    sharded_batch = sharded.sample_arrays(4, random.Random(11))

    assert sorted(sharded_batch.indexes.tolist()) == [0, 1, 2, 3]
    assert_batches_equal(sharded_batch, monolithic_batch)


def test_shard_indexed_dataset_supports_sample_weight_priority(tmp_path: Path) -> None:
    episodes = (make_episode(0, winner=1), make_episode(1, winner=2))
    mono_store = TrajectoryReplayStore.from_episodes(8, episodes)
    mono_store.sample_weights[:] = np.asarray([1.0, 1.0, 8.0, 8.0], dtype=np.float32)
    monolithic = TrajectoryReplayDataset(mono_store)

    first = write_shard(tmp_path, "first", episodes[0])
    second = write_shard(tmp_path, "second", episodes[1])
    first_store = TrajectoryReplayStore.load(first.replay_path)
    first_store.sample_weights[:] = np.asarray([1.0, 1.0], dtype=np.float32)
    first_store.save(first.replay_path, compressed=False)
    second_store = TrajectoryReplayStore.load(second.replay_path)
    second_store.sample_weights[:] = np.asarray([8.0, 8.0], dtype=np.float32)
    second_store.save(second.replay_path, compressed=False)
    index = ShardReplayIndex.load_or_create(tmp_path / "replay-index", capacity=32)
    index.add_shard(first)
    index.add_shard(second)
    sharded = ShardIndexedTrajectoryDataset(index)

    config = PrioritySamplingConfig(enabled=True, alpha=1.0, beta=0.4)
    monolithic_batch = monolithic.sample_arrays(2, random.Random(5), priority_config=config)
    sharded_batch = sharded.sample_arrays(2, random.Random(5), priority_config=config)

    assert_batches_equal(sharded_batch, monolithic_batch)
    assert np.all(sharded_batch.sample_weights > 0.0)


def test_shard_indexed_dataset_caches_loaded_shards(tmp_path: Path) -> None:
    episodes = (make_episode(0, winner=1), make_episode(1, winner=2))
    dataset = build_indexed_dataset(tmp_path, episodes, cache_shards=1)

    dataset.sample_arrays(2, random.Random(3), recent_fraction=1.0, recent_window=2)
    first_info = dataset.cache_info
    dataset.sample_arrays(2, random.Random(3), recent_fraction=1.0, recent_window=2)
    second_info = dataset.cache_info

    assert first_info["size"] == 1
    assert first_info["misses"] == 1
    assert second_info["hits"] > first_info["hits"]


def test_shard_indexed_dataset_can_limit_sampled_shards_per_batch(tmp_path: Path) -> None:
    episodes = tuple(make_episode(index, winner=1 if index % 2 == 0 else 2) for index in range(6))
    dataset = build_indexed_dataset(tmp_path, episodes, cache_shards=2)
    limited = ShardIndexedTrajectoryDataset(
        dataset._index,  # noqa: SLF001
        cache_shards=2,
        sample_shards_per_batch=2,
    )

    batch = limited.sample_arrays(4, random.Random(13))
    touched_spans = {index // 2 for index in batch.indexes.tolist()}

    assert len(touched_spans) <= 2
    assert limited.cache_info["misses"] <= 2


def test_shard_indexed_dataset_cache_local_sampling_preserves_recency_window(
    tmp_path: Path,
) -> None:
    episodes = tuple(make_episode(index, winner=1 if index % 2 == 0 else 2) for index in range(6))
    dataset = build_indexed_dataset(tmp_path, episodes, cache_shards=4)
    limited = ShardIndexedTrajectoryDataset(
        dataset._index,  # noqa: SLF001
        cache_shards=4,
        sample_shards_per_batch=2,
    )

    batch = limited.sample_arrays(
        4,
        random.Random(17),
        recent_fraction=1.0,
        recent_window=4,
    )

    assert min(batch.indexes.tolist()) >= 8
