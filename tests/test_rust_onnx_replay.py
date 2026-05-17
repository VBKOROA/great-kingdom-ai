from __future__ import annotations

from pathlib import Path

import numpy as np
from great_kingdom_ai.features import ACTION_SPACE, BOARD_SIZE, FEATURE_CHANNELS
from great_kingdom_ai.replay.sample import ReplaySample
from great_kingdom_ai.replay.trajectory import TrajectoryReplayStore
from great_kingdom_ai.rust_onnx_replay import import_rust_self_play_samples
from great_kingdom_ai.self_play import GameLog, MoveLog


def make_sample(index: int) -> ReplaySample:
    features = np.zeros((FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    features[index % FEATURE_CHANNELS, 0, 0] = 1.0
    features[4, :, :] = 1.0
    policy = np.zeros(ACTION_SPACE, dtype=np.float32)
    policy[index % ACTION_SPACE] = 1.0
    root_policy_logits = np.full(ACTION_SPACE, -2.0, dtype=np.float32)
    root_policy_logits[index % ACTION_SPACE] = 2.0
    return ReplaySample(
        features=features,
        policy=policy,
        value=1.0,
        root_policy_logits=root_policy_logits,
    )


def make_log(seed: int, move_count: int = 1) -> GameLog:
    return GameLog(
        seed=seed,
        moves=[
            MoveLog(turn=index, player=1 if index % 2 == 0 else 2, action=index)
            for index in range(move_count)
        ],
        winner=1,
        end_reason=1,
        territory_scores=(0, 0),
    )


def test_import_rust_self_play_samples_extends_replay_and_logs(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifact"

    summary = import_rust_self_play_samples(
        artifact_dir=artifact_dir,
        samples=[make_sample(0), make_sample(1)],
        logs=[make_log(10, move_count=2)],
        replay_path=tmp_path / "replay" / "trajectory-replay.npz",
        replay_capacity=8,
        game_log_path=tmp_path / "replay" / "game_logs.json",
    )

    replay = TrajectoryReplayStore.load(tmp_path / "replay" / "trajectory-replay.npz")
    assert summary.imported_samples == 2
    assert summary.imported_games == 1
    assert len(replay) == 2
    assert replay.root_policy_logits is not None
    assert '"seed": 10' in (tmp_path / "replay" / "game_logs.json").read_text()
    assert not artifact_dir.exists()


def test_import_rust_self_play_samples_appends_jsonl_logs(tmp_path: Path) -> None:
    log_path = tmp_path / "replay" / "game_logs.jsonl"

    import_rust_self_play_samples(
        artifact_dir=tmp_path / "artifact",
        samples=[],
        logs=[make_log(10), make_log(11)],
        replay_path=tmp_path / "replay" / "trajectory-replay.npz",
        replay_capacity=8,
        game_log_path=log_path,
    )

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert '"seed": 10' in lines[0]
    assert '"seed": 11' in lines[1]
    assert not (tmp_path / "replay" / "trajectory-replay.npz").exists()
