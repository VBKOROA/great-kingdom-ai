from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from great_kingdom_ai.features import ACTION_SPACE, BOARD_SIZE, FEATURE_CHANNELS, PASS_ACTION
from great_kingdom_ai.shard_replay_index import ShardReplayIndex
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


def make_episode(episode_id: int, rows: int) -> TrajectoryEpisode:
    transitions = []
    for timestep in range(rows):
        action = 1 + timestep
        features = make_features(action)
        transitions.append(
            TrajectoryTransition(
                episode_id=episode_id,
                timestep=timestep,
                player=1 if timestep % 2 == 0 else 2,
                features=features,
                legal_mask=legal_mask_from_features(features),
                action=action,
                policy_target=make_policy(action),
                winner=1,
                terminal=timestep == rows - 1,
            )
        )
    return TrajectoryEpisode(
        episode_id=episode_id,
        seed=episode_id,
        transitions=tuple(transitions),
        winner=1,
        end_reason=1,
        territory_scores=(1, 0),
    )


def write_shard(tmp_path: Path, shard_id: str, rows: int) -> IndexSource:
    shard_dir = tmp_path / "shards" / shard_id
    shard_dir.mkdir(parents=True)
    replay_path = shard_dir / "trajectory-replay.npz"
    TrajectoryReplayStore.from_episodes(rows, (make_episode(rows, rows),)).save(
        replay_path,
        compressed=False,
    )
    log_path = shard_dir / "game_logs.json"
    log_path.write_text("[]", encoding="utf-8")
    return IndexSource(
        shard_id=shard_id,
        replay_path=replay_path,
        log_path=log_path,
        seed_start=rows,
    )


def test_shard_replay_index_add_load_and_state_rebuild(tmp_path: Path) -> None:
    first = write_shard(tmp_path, "first", 2)
    second = write_shard(tmp_path, "second", 3)
    index = ShardReplayIndex.load_or_create(tmp_path / "replay-index", capacity=10)

    index.add_shard(first)
    index.add_shard(second)
    index.add_shard(first)

    loaded = ShardReplayIndex.load_or_create(tmp_path / "replay-index", capacity=10)

    assert [record.shard_id for record in loaded.active_records] == ["first", "second"]
    assert [span.start for span in loaded.active_spans()] == [0, 2]
    assert [span.stop for span in loaded.active_spans()] == [2, 5]
    assert loaded.active_rows == 5
    assert '"active_rows": 5' in (tmp_path / "replay-index" / "state.json").read_text(
        encoding="utf-8"
    )
    assert sum(1 for _ in (tmp_path / "replay-index" / "index.jsonl").open()) == 2


def test_shard_replay_index_evicts_oldest_shards(tmp_path: Path) -> None:
    first = write_shard(tmp_path, "first", 2)
    second = write_shard(tmp_path, "second", 3)
    third = write_shard(tmp_path, "third", 4)
    index = ShardReplayIndex.load_or_create(tmp_path / "replay-index", capacity=5)

    index.add_shard(first)
    index.add_shard(second)
    index.add_shard(third)
    evicted = index.evict_to_capacity()

    assert [record.shard_id for record in evicted] == ["first", "second"]
    assert [record.shard_id for record in index.active_records] == ["third"]
    assert index.active_rows == 4

    loaded = ShardReplayIndex.load_or_create(tmp_path / "replay-index", capacity=5)
    assert [record.shard_id for record in loaded.evicted_records] == ["first", "second"]
    assert [record.shard_id for record in loaded.active_records] == ["third"]


def test_shard_replay_index_rejects_missing_active_shard(tmp_path: Path) -> None:
    source = write_shard(tmp_path, "missing", 2)
    index = ShardReplayIndex.load_or_create(tmp_path / "replay-index", capacity=10)
    index.add_shard(source)
    source.replay_path.unlink()

    with pytest.raises(FileNotFoundError, match="active indexed shard replay is missing"):
        ShardReplayIndex.load_or_create(tmp_path / "replay-index", capacity=10)


def test_shard_replay_index_allows_missing_evicted_shard(tmp_path: Path) -> None:
    first = write_shard(tmp_path, "first", 3)
    second = write_shard(tmp_path, "second", 3)
    index = ShardReplayIndex.load_or_create(tmp_path / "replay-index", capacity=3)
    index.add_shard(first)
    index.add_shard(second)
    index.evict_to_capacity()
    first.replay_path.unlink()

    loaded = ShardReplayIndex.load_or_create(tmp_path / "replay-index", capacity=3)

    assert [record.shard_id for record in loaded.active_records] == ["second"]
