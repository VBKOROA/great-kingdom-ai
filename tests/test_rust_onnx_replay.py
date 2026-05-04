from __future__ import annotations

from pathlib import Path

import numpy as np
from great_kingdom_ai.features import ACTION_SPACE, BOARD_SIZE, FEATURE_CHANNELS
from great_kingdom_ai.replay_buffer import ReplayBuffer, ReplaySample
from great_kingdom_ai.rust_onnx_replay import (
    import_legacy_pipeline_data,
    import_rust_self_play_artifacts,
    write_rust_self_play_artifacts,
)
from great_kingdom_ai.self_play import GameLog, MoveLog


def make_sample(index: int) -> ReplaySample:
    features = np.zeros((FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    features[index % FEATURE_CHANNELS, 0, 0] = 1.0
    policy = np.zeros(ACTION_SPACE, dtype=np.float32)
    policy[index % ACTION_SPACE] = 1.0
    return ReplaySample(features=features, policy=policy, value=1.0)


def make_log(seed: int) -> GameLog:
    return GameLog(
        seed=seed,
        moves=[MoveLog(turn=0, player=1, action=seed % ACTION_SPACE)],
        winner=1,
        end_reason=1,
        territory_scores=(0, 0),
    )


def test_import_rust_self_play_artifacts_extends_replay_and_logs(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifact"
    write_rust_self_play_artifacts(
        output_dir=artifact_dir,
        samples=[make_sample(0), make_sample(1)],
        logs=[make_log(10)],
        manifest={"format_version": 1},
    )

    summary = import_rust_self_play_artifacts(
        artifact_dir=artifact_dir,
        replay_path=tmp_path / "replay" / "replay.npz",
        replay_capacity=8,
        game_log_path=tmp_path / "replay" / "game_logs.json",
    )

    replay = ReplayBuffer.load(tmp_path / "replay" / "replay.npz")
    assert summary.imported_samples == 2
    assert summary.imported_games == 1
    assert len(replay) == 2
    assert '"seed": 10' in (tmp_path / "replay" / "game_logs.json").read_text()


def test_import_legacy_pipeline_data_copies_replay_logs_and_checkpoints(
    tmp_path: Path,
) -> None:
    legacy = tmp_path / "legacy"
    onnx = tmp_path / "onnx"
    replay = ReplayBuffer(capacity=4)
    replay.extend([make_sample(0), make_sample(1), make_sample(2)])
    replay.save(legacy / "replay" / "replay.npz")
    (legacy / "replay" / "game_logs.json").write_text(
        '[{"seed": 7}, {"seed": 9}]',
        encoding="utf-8",
    )
    (legacy / "checkpoints").mkdir(parents=True)
    (legacy / "checkpoints" / "best.pt").write_text("best", encoding="utf-8")

    summary = import_legacy_pipeline_data(
        legacy_work_dir=legacy,
        onnx_work_dir=onnx,
        replay_capacity=2,
    )
    second_summary = import_legacy_pipeline_data(
        legacy_work_dir=legacy,
        onnx_work_dir=onnx,
        replay_capacity=2,
    )

    imported_replay = ReplayBuffer.load(onnx / "replay" / "replay.npz")
    assert len(imported_replay) == 2
    assert summary.imported is True
    assert summary.next_seed_start == 10
    assert second_summary.imported is False
    assert (onnx / "checkpoints" / "best.pt").read_text(encoding="utf-8") == "best"
    assert (onnx / "reports" / "legacy-import.json").is_file()
