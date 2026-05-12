from __future__ import annotations

from pathlib import Path

import pytest
from great_kingdom_ai.trajectory_replay import TrajectoryReplayStore
from great_kingdom_ai.trajectory_replay_migration import (
    build_parser,
    migrate_trajectory_replay_capacity,
)

from tests.test_trajectory_replay import make_episode


def test_migrate_trajectory_replay_capacity_preserves_replay_data(tmp_path: Path) -> None:
    source = tmp_path / "trajectory-replay.npz"
    output = tmp_path / "trajectory-replay-500k.npz"
    replay = TrajectoryReplayStore.empty(capacity=3)
    replay.extend_episodes([make_episode(0, actions=[1, 2])])
    replay.save(source, compressed=False)

    migrated_path = migrate_trajectory_replay_capacity(source, output, 8)
    migrated = TrajectoryReplayStore.load(migrated_path)

    assert migrated.capacity == 8
    assert len(migrated) == 2
    assert migrated.episode_ids.tolist() == [0]
    assert migrated.actions.tolist() == [1, 2]


def test_migrate_trajectory_replay_capacity_rejects_capacity_below_current_size(
    tmp_path: Path,
) -> None:
    source = tmp_path / "trajectory-replay.npz"
    replay = TrajectoryReplayStore.empty(capacity=3)
    replay.extend_episodes([make_episode(0, actions=[1, 2])])
    replay.save(source, compressed=False)

    with pytest.raises(ValueError, match="current transition count"):
        migrate_trajectory_replay_capacity(source, tmp_path / "too-small.npz", 1)


def test_migrate_trajectory_replay_capacity_can_rewrite_in_place_with_force(
    tmp_path: Path,
) -> None:
    source = tmp_path / "trajectory-replay.npz"
    replay = TrajectoryReplayStore.empty(capacity=3)
    replay.extend_episodes([make_episode(0, actions=[1, 2])])
    replay.save(source, compressed=False)

    migrated_path = migrate_trajectory_replay_capacity(source, source, 6, force=True)
    migrated = TrajectoryReplayStore.load(migrated_path)

    assert migrated_path == source
    assert migrated.capacity == 6
    assert len(migrated) == 2
    assert not (tmp_path / ".trajectory-replay.tmp.npz").exists()


def test_trajectory_replay_migration_parser_accepts_paths() -> None:
    args = build_parser().parse_args(
        [
            "--source",
            "trajectory-replay.npz",
            "--output",
            "trajectory-replay-500k.npz",
            "--capacity",
            "500000",
            "--force",
        ]
    )

    assert args.source == Path("trajectory-replay.npz")
    assert args.output == Path("trajectory-replay-500k.npz")
    assert args.capacity == 500000
    assert args.force is True
