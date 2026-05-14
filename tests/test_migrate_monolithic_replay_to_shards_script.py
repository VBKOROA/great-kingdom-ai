from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from great_kingdom_ai.features import ACTION_SPACE
from great_kingdom_ai.shard_indexed_dataset import ShardIndexedTrajectoryDataset
from great_kingdom_ai.shard_replay_index import ShardReplayIndex
from great_kingdom_ai.trajectory_replay import TrajectoryReplayStore

from tests.test_trajectory_replay import make_episode


def load_script_module() -> Any:
    path = Path("scripts/migrate_monolithic_replay_to_shards.py")
    spec = importlib.util.spec_from_file_location("migrate_monolithic_replay_to_shards", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_migrate_monolithic_replay_to_shards_splits_on_episode_boundaries(
    tmp_path: Path,
) -> None:
    module = load_script_module()
    source = tmp_path / "replay" / "trajectory-replay.npz"
    episodes = (
        make_episode(0, actions=[1, 2]),
        make_episode(1, actions=[3, 4]),
        make_episode(2, actions=[5, 6]),
    )
    source.parent.mkdir()
    TrajectoryReplayStore.from_episodes(16, episodes).save(source, compressed=False)

    summary = module.migrate_monolithic_replay_to_shards(
        source=source,
        output_shard_root=tmp_path / "shards",
        index_dir=tmp_path / "replay-index",
        target_rows_per_shard=4,
        drop_optional_arrays=True,
    )

    assert summary.rows == 6
    assert summary.episodes == 3
    assert [shard.rows for shard in summary.shards] == [4, 2]
    assert [shard.episodes for shard in summary.shards] == [2, 1]

    index = ShardReplayIndex.load_or_create(tmp_path / "replay-index", capacity=16)
    assert [record.shard_id for record in index.active_records] == [
        "migrated-000001",
        "migrated-000002",
    ]
    assert len(index) == 6

    dataset = ShardIndexedTrajectoryDataset(index)
    batch = dataset.sample_arrays(6, __import__("random").Random(7))
    assert sorted(batch.indexes.tolist()) == [0, 1, 2, 3, 4, 5]
    assert batch.features.shape[0] == 6


def test_migrate_monolithic_replay_to_shards_drops_optional_arrays(tmp_path: Path) -> None:
    module = load_script_module()
    source = tmp_path / "trajectory-replay.npz"
    episode = make_episode(0, actions=[1, 2])
    transitions = tuple(
        replace(
            transition,
            root_policy_logits=np.ones((ACTION_SPACE,), dtype=np.float32),
            next_features=transition.features.copy(),
        )
        for transition in episode.transitions
    )
    with_optional = replace(
        episode,
        transitions=transitions,
    )
    TrajectoryReplayStore.from_episodes(8, (with_optional,)).save(source, compressed=False)

    module.migrate_monolithic_replay_to_shards(
        source=source,
        output_shard_root=tmp_path / "shards",
        index_dir=tmp_path / "replay-index",
        target_rows_per_shard=8,
        drop_optional_arrays=True,
    )

    shard = TrajectoryReplayStore.load(
        tmp_path / "shards" / "migrated-000001" / "trajectory-replay.npz"
    )
    assert shard.root_policy_logits is None
    assert shard.next_features is None


def test_migrate_monolithic_replay_to_shards_dry_run_writes_nothing(tmp_path: Path) -> None:
    module = load_script_module()
    source = tmp_path / "trajectory-replay.npz"
    TrajectoryReplayStore.from_episodes(
        8,
        (make_episode(0, actions=[1, 2]), make_episode(1, actions=[3, 4])),
    ).save(source, compressed=False)

    summary = module.migrate_monolithic_replay_to_shards(
        source=source,
        output_shard_root=tmp_path / "shards",
        index_dir=tmp_path / "replay-index",
        target_rows_per_shard=2,
        dry_run=True,
    )

    assert summary.dry_run is True
    assert summary.rows == 4
    assert len(summary.shards) == 2
    assert not (tmp_path / "shards").exists()
    assert not (tmp_path / "replay-index").exists()


def test_migrate_monolithic_replay_to_shards_rejects_capacity_below_source_rows(
    tmp_path: Path,
) -> None:
    module = load_script_module()
    source = tmp_path / "trajectory-replay.npz"
    TrajectoryReplayStore.from_episodes(8, (make_episode(0, actions=[1, 2]),)).save(
        source,
        compressed=False,
    )

    with pytest.raises(ValueError, match="source transition count"):
        module.migrate_monolithic_replay_to_shards(
            source=source,
            output_shard_root=tmp_path / "shards",
            index_dir=tmp_path / "replay-index",
            capacity=1,
        )


def test_migrate_monolithic_replay_to_shards_parser_accepts_recommended_args() -> None:
    module = load_script_module()

    args = module.build_parser().parse_args(
        [
            "--source",
            "data/runpod/train-v2-gumbel-512k/replay/trajectory-replay.npz",
            "--output-shard-root",
            "data/runpod/train-v2-gumbel-512k/shards",
            "--index-dir",
            "data/runpod/train-v2-gumbel-512k/replay-index",
            "--target-rows-per-shard",
            "4096",
            "--drop-optional-arrays",
            "--dry-run",
        ]
    )

    assert args.target_rows_per_shard == 4096
    assert args.drop_optional_arrays is True
    assert args.dry_run is True
