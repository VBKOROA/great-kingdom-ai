from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from great_kingdom_ai.actor_learner_processes import (
    ActorProcessConfig,
    LearnerProcessConfig,
    load_shard_records,
    pending_shards,
    run_actor_process_once,
    run_learner_process_once,
)
from great_kingdom_ai.features import ACTION_SPACE, BOARD_SIZE, FEATURE_CHANNELS
from great_kingdom_ai.pipeline_printer import PipelinePrinter
from great_kingdom_ai.replay_buffer import ReplaySample
from great_kingdom_ai.rust_onnx_self_play import RustOnnxSelfPlayConfig, RustSelfPlayRunSummary
from great_kingdom_ai.self_play import GameLog, MoveLog
from great_kingdom_ai.train import TrainingConfig


def make_sample(index: int) -> ReplaySample:
    features = np.zeros((FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    features[index % FEATURE_CHANNELS, 0, 0] = 1.0
    policy = np.zeros(ACTION_SPACE, dtype=np.float32)
    policy[index % ACTION_SPACE] = 1.0
    return ReplaySample(features=features, policy=policy, value=1.0)


def fake_actor_runner(config: RustOnnxSelfPlayConfig) -> RustSelfPlayRunSummary:
    logs = [
        GameLog(
            seed=config.seed_start,
            moves=[MoveLog(turn=0, player=1, action=1)],
            winner=1,
            end_reason=1,
            territory_scores=(0, 0),
        )
    ]
    samples = [make_sample(config.seed_start), make_sample(config.seed_start + 1)]
    return RustSelfPlayRunSummary(
        artifact_dir=config.output_dir,
        games=len(logs),
        samples=len(samples),
        onnx_model_path=config.onnx_model_path,
        onnx_device=config.onnx_device,
        replay_samples=tuple(samples),
        game_logs=tuple(logs),
    )


class FakeTrainSummary:
    def __init__(self, checkpoint_path: Path) -> None:
        self.start_step = 0
        self.end_step = 3
        self.checkpoint_path = checkpoint_path
        self.losses: list[dict[str, float]] = []


def test_actor_process_writes_append_only_shard_metadata(tmp_path: Path) -> None:
    summary = run_actor_process_once(
        ActorProcessConfig(
            work_dir=tmp_path,
            onnx_model_path=tmp_path / "best.onnx",
            model_version="best-000007",
            games=1,
            seed_start=10,
        ),
        runner=fake_actor_runner,
        printer=PipelinePrinter(enabled=False),
    )

    shard = summary.shard
    assert shard.shard_id == "best-000007-seed-00000010-games-0001"
    assert shard.status == "completed"
    assert shard.replay_path.is_file()
    assert shard.log_path.is_file()

    records = load_shard_records(tmp_path / "shards" / "metadata.jsonl")
    assert len(records) == 1
    assert records[0].shard_id == shard.shard_id
    assert records[0].status == "completed"
    assert pending_shards(tmp_path / "shards" / "metadata.jsonl") == records


def test_learner_process_imports_pending_shards_and_trains(tmp_path: Path) -> None:
    run_actor_process_once(
        ActorProcessConfig(
            work_dir=tmp_path,
            onnx_model_path=tmp_path / "best.onnx",
            model_version="best-000001",
            games=1,
            seed_start=0,
        ),
        runner=fake_actor_runner,
        printer=PipelinePrinter(enabled=False),
    )
    train_calls: list[dict[str, Any]] = []

    def fake_train(
        replay: Any,
        config: TrainingConfig,
        *,
        checkpoint_path: str | Path,
        resume_path: str | Path | None,
        bootstrap_weights_path: str | Path | None = None,
        log_every: int,
        progress_callback: Any = None,
    ) -> FakeTrainSummary:
        del config, log_every
        train_calls.append(
            {
                "replay_samples": len(replay),
                "resume_path": resume_path,
                "bootstrap_weights_path": bootstrap_weights_path,
            }
        )
        if progress_callback is not None:
            progress_callback(3, 3, {"total": 0.25})
        destination = Path(checkpoint_path)
        destination.write_text("candidate", encoding="utf-8")
        return FakeTrainSummary(destination)

    summary = run_learner_process_once(
        LearnerProcessConfig(work_dir=tmp_path, replay_capacity=8),
        TrainingConfig(batch_size=1, steps=3, device="cpu"),
        trainer=fake_train,
        printer=PipelinePrinter(enabled=False),
    )

    assert summary.imported_samples == 2
    assert summary.imported_games == 1
    assert summary.replay_samples == 2
    assert summary.train_end_step == 3
    assert summary.candidate_checkpoint.read_text(encoding="utf-8") == "candidate"
    assert summary.training_latest_checkpoint.read_text(encoding="utf-8") == "candidate"
    assert train_calls == [
        {
            "replay_samples": 2,
            "resume_path": None,
            "bootstrap_weights_path": None,
        }
    ]

    records = load_shard_records(tmp_path / "shards" / "metadata.jsonl")
    assert [record.status for record in records] == ["imported"]
    assert pending_shards(tmp_path / "shards" / "metadata.jsonl") == []
    assert (tmp_path / "replay" / "replay.npz").is_file()
    assert (tmp_path / "replay" / "game_logs.jsonl").is_file()


def test_learner_process_uses_existing_training_latest_for_resume(tmp_path: Path) -> None:
    run_actor_process_once(
        ActorProcessConfig(
            work_dir=tmp_path,
            onnx_model_path=tmp_path / "best.onnx",
            model_version="best",
            games=1,
            seed_start=0,
        ),
        runner=fake_actor_runner,
        printer=PipelinePrinter(enabled=False),
    )
    training_latest = tmp_path / "checkpoints" / "training-latest.pt"
    training_latest.parent.mkdir(parents=True)
    training_latest.write_text("previous", encoding="utf-8")
    resume_paths: list[Path | None] = []

    def fake_train(
        replay: Any,
        config: TrainingConfig,
        *,
        checkpoint_path: str | Path,
        resume_path: str | Path | None,
        bootstrap_weights_path: str | Path | None = None,
        log_every: int,
        progress_callback: Any = None,
    ) -> FakeTrainSummary:
        del replay, config, bootstrap_weights_path, log_every, progress_callback
        resume_paths.append(Path(resume_path) if resume_path is not None else None)
        destination = Path(checkpoint_path)
        destination.write_text("candidate", encoding="utf-8")
        return FakeTrainSummary(destination)

    run_learner_process_once(
        LearnerProcessConfig(work_dir=tmp_path, replay_capacity=8),
        TrainingConfig(batch_size=1, steps=1, device="cpu"),
        trainer=fake_train,
        printer=PipelinePrinter(enabled=False),
    )

    assert resume_paths == [training_latest]
