import numpy as np
from great_kingdom_ai.artifacts import (
    ArtifactPaths,
    load_artifact_paths,
    load_self_play_artifact,
    save_self_play_artifact,
)
from great_kingdom_ai.features import ACTION_SPACE, BOARD_SIZE, FEATURE_CHANNELS
from great_kingdom_ai.replay_buffer import ReplaySample
from great_kingdom_ai.self_play import GameLog, MoveLog


def make_sample() -> ReplaySample:
    features = np.zeros((FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    policy = np.zeros(ACTION_SPACE, dtype=np.float32)
    policy[81] = 1.0
    return ReplaySample(features=features, policy=policy, value=1.0)


def test_artifact_paths_load_from_config_and_create_directories(tmp_path) -> None:
    config_path = tmp_path / "artifacts.json"
    config_path.write_text(
        """
{
  "replay_dir": "replay",
  "checkpoint_dir": "checkpoints",
  "log_dir": "logs"
}
""".strip(),
        encoding="utf-8",
    )

    paths = load_artifact_paths(config_path)
    rooted = ArtifactPaths(
        replay_dir=tmp_path / paths.replay_dir,
        checkpoint_dir=tmp_path / paths.checkpoint_dir,
        log_dir=tmp_path / paths.log_dir,
    )
    rooted.ensure_dirs()

    assert rooted.replay_dir.is_dir()
    assert rooted.checkpoint_dir.is_dir()
    assert rooted.log_dir.is_dir()
    assert rooted.to_dict()["replay_dir"].endswith("replay")


def test_self_play_artifact_round_trips_replay_and_logs(tmp_path) -> None:
    log = GameLog(
        seed=1,
        moves=[MoveLog(turn=0, player=1, action=81)],
        winner=2,
        end_reason=3,
        territory_scores=(0, 0),
    )

    paths = save_self_play_artifact(tmp_path, logs=[log], samples=[make_sample()])
    buffer, logs = load_self_play_artifact(tmp_path)

    assert paths["replay"].name == "replay.npz"
    assert paths["logs"].name == "game_logs.json"
    assert len(buffer) == 1
    assert logs[0]["seed"] == 1
    assert logs[0]["moves"] == [{"turn": 0, "player": 1, "action": 81}]
