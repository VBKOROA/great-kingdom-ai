from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import great_kingdom_ai.rust_onnx_pipeline as pipeline_module
import numpy as np
from great_kingdom_ai.evaluate import ArenaConfig, ArenaGameResult, ArenaReport, summarize_arena
from great_kingdom_ai.features import ACTION_SPACE, BOARD_SIZE, FEATURE_CHANNELS
from great_kingdom_ai.pipeline import PipelinePrinter
from great_kingdom_ai.replay_buffer import ReplaySample
from great_kingdom_ai.rust_onnx_pipeline import RustOnnxPipelineConfig, run_rust_onnx_pipeline
from great_kingdom_ai.rust_onnx_replay import write_rust_self_play_artifacts
from great_kingdom_ai.rust_onnx_self_play import RustOnnxSelfPlayConfig, RustSelfPlayRunSummary
from great_kingdom_ai.self_play import GameLog, MoveLog
from great_kingdom_ai.train import TrainingConfig


def make_sample(index: int) -> ReplaySample:
    features = np.zeros((FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    features[index % FEATURE_CHANNELS, 0, 0] = 1.0
    policy = np.zeros(ACTION_SPACE, dtype=np.float32)
    policy[index % ACTION_SPACE] = 1.0
    return ReplaySample(features=features, policy=policy, value=1.0)


def fake_runner(config: RustOnnxSelfPlayConfig) -> RustSelfPlayRunSummary:
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
    artifact_dir = write_rust_self_play_artifacts(
        output_dir=config.output_dir,
        samples=samples,
        logs=logs,
        manifest={"format_version": 1, "model_path": str(config.onnx_model_path)},
    )
    return RustSelfPlayRunSummary(
        artifact_dir=artifact_dir,
        games=len(logs),
        samples=len(samples),
        onnx_model_path=config.onnx_model_path,
        onnx_device=config.onnx_device,
    )


class FakeTrainSummary:
    def __init__(self, checkpoint_path: Path) -> None:
        self.start_step = 0
        self.end_step = 3
        self.checkpoint_path = checkpoint_path
        self.losses: list[dict[str, float]] = []


def test_rust_onnx_pipeline_dispatches_runner_and_imports_replay(
    tmp_path: Path,
    monkeypatch,
) -> None:
    exported: list[tuple[Path, Path]] = []

    def fake_save_checkpoint(state: object, path: str | Path) -> Path:
        del state
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("best", encoding="utf-8")
        return destination

    def fake_export(checkpoint_path: str | Path, output_path: str | Path, **kwargs: Any) -> object:
        del kwargs
        exported.append((Path(checkpoint_path), Path(output_path)))
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text("onnx", encoding="utf-8")
        return object()

    def fake_train_from_replay(
        replay: Any,
        config: TrainingConfig,
        *,
        checkpoint_path: str | Path,
        resume_path: str | Path | None,
        log_every: int,
        progress_callback: Any = None,
    ) -> FakeTrainSummary:
        del replay, config, resume_path, log_every
        if progress_callback is not None:
            progress_callback(3, 3, {"total": 0.5})
        destination = Path(checkpoint_path)
        destination.write_text("candidate", encoding="utf-8")
        return FakeTrainSummary(destination)

    monkeypatch.setattr(pipeline_module, "create_train_state", lambda config: object())
    monkeypatch.setattr(pipeline_module, "save_checkpoint", fake_save_checkpoint)
    monkeypatch.setattr(pipeline_module, "export_checkpoint_to_onnx", fake_export)
    monkeypatch.setattr(pipeline_module, "train_from_replay", fake_train_from_replay)

    summary = run_rust_onnx_pipeline(
        pipeline_config=RustOnnxPipelineConfig(
            work_dir=tmp_path,
            iterations=1,
            replay_capacity=8,
            self_play_games=1,
            skip_arena=True,
        ),
        train_config=TrainingConfig(batch_size=1, steps=3, device="cpu"),
        arena_config=ArenaConfig(games=1, device="cpu"),
        printer=PipelinePrinter(enabled=False),
        rust_self_play_runner=fake_runner,
    )

    assert len(summary.iterations) == 1
    assert summary.replay_samples == 2
    assert exported[0][0] == tmp_path / "checkpoints" / "best.pt"
    assert exported[0][1] == tmp_path / "checkpoints" / "onnx" / "best-000001.onnx"
    assert (tmp_path / "checkpoints" / "candidate.pt").read_text(encoding="utf-8") == "candidate"
    assert (tmp_path / "replay" / "game_logs.json").is_file()
    assert (tmp_path / "replay" / "replay-aggregated.npz").is_file()

    with np.load(tmp_path / "replay" / "replay-aggregated.npz") as data:
        assert data["features"].shape[0] == 2


def test_rust_onnx_arena_config_offsets_seed_start_by_iteration() -> None:
    config = pipeline_module._arena_config_for_pipeline(
        ArenaConfig(games=20, seed_start=100000),
        iteration=3,
    )

    assert config.seed_start == 100040


def test_rust_onnx_pipeline_uses_distinct_arena_seed_windows(
    tmp_path: Path,
    monkeypatch,
) -> None:
    arena_seed_starts: list[int] = []

    def fake_save_checkpoint(state: object, path: str | Path) -> Path:
        del state
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("best", encoding="utf-8")
        return destination

    def fake_export(checkpoint_path: str | Path, output_path: str | Path, **kwargs: Any) -> object:
        del checkpoint_path, kwargs
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text("onnx", encoding="utf-8")
        return object()

    def fake_train_from_replay(
        replay: Any,
        config: TrainingConfig,
        *,
        checkpoint_path: str | Path,
        resume_path: str | Path | None,
        log_every: int,
        progress_callback: Any = None,
    ) -> FakeTrainSummary:
        del replay, config, resume_path, log_every, progress_callback
        destination = Path(checkpoint_path)
        destination.write_text("candidate", encoding="utf-8")
        return FakeTrainSummary(destination)

    def fake_run_arena(
        candidate_model: object,
        best_model: object,
        config: ArenaConfig,
        progress_callback=None,
    ) -> ArenaReport:
        del candidate_model, best_model, progress_callback
        arena_seed_starts.append(config.seed_start)
        game = ArenaGameResult(
            seed=config.seed_start,
            candidate_player=1,
            best_player=2,
            winner=2,
            end_reason=1,
            moves=[MoveLog(turn=0, player=1, action=1)],
            territory_scores=(0, 0),
        )
        return ArenaReport(
            config=config,
            games=[game],
            summary=summarize_arena([game], promotion_threshold=config.promotion_threshold),
        )

    monkeypatch.setattr(pipeline_module, "create_train_state", lambda config: object())
    monkeypatch.setattr(pipeline_module, "save_checkpoint", fake_save_checkpoint)
    monkeypatch.setattr(pipeline_module, "export_checkpoint_to_onnx", fake_export)
    monkeypatch.setattr(pipeline_module, "train_from_replay", fake_train_from_replay)
    monkeypatch.setattr(
        pipeline_module,
        "load_model_from_checkpoint",
        lambda path, *, device: object(),
    )
    monkeypatch.setattr(pipeline_module, "run_arena", fake_run_arena)

    run_rust_onnx_pipeline(
        pipeline_config=RustOnnxPipelineConfig(
            work_dir=tmp_path,
            iterations=2,
            replay_capacity=8,
            self_play_games=1,
            skip_arena=False,
        ),
        train_config=TrainingConfig(batch_size=1, steps=3, device="cpu"),
        arena_config=ArenaConfig(games=20, seed_start=100000, device="cpu"),
        printer=PipelinePrinter(enabled=False),
        rust_self_play_runner=fake_runner,
    )

    assert arena_seed_starts == [100000, 100020]


def test_main_applies_device_to_onnx_device_when_not_explicit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_run_rust_onnx_pipeline(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return pipeline_module.RustOnnxPipelineSummary(
            iterations=[],
            replay_samples=0,
            best_checkpoint=tmp_path / "best.pt",
            replay_path=tmp_path / "replay.npz",
        )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "great-kingdom-rust-onnx-pipeline",
            "--device",
            "cuda",
            "--json",
        ],
    )
    monkeypatch.setattr(pipeline_module, "load_training_config", lambda path: TrainingConfig())
    monkeypatch.setattr(pipeline_module, "load_arena_config", lambda path: ArenaConfig())
    monkeypatch.setattr(pipeline_module, "run_rust_onnx_pipeline", fake_run_rust_onnx_pipeline)

    try:
        pipeline_module.main()
    except SystemExit as exc:
        assert exc.code == 0

    config = captured["pipeline_config"]
    assert config.onnx_device == "cuda"


def test_main_prefers_explicit_onnx_device_over_device(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_run_rust_onnx_pipeline(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return pipeline_module.RustOnnxPipelineSummary(
            iterations=[],
            replay_samples=0,
            best_checkpoint=tmp_path / "best.pt",
            replay_path=tmp_path / "replay.npz",
        )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "great-kingdom-rust-onnx-pipeline",
            "--device",
            "cuda",
            "--onnx-device",
            "cpu",
            "--json",
        ],
    )
    monkeypatch.setattr(pipeline_module, "load_training_config", lambda path: TrainingConfig())
    monkeypatch.setattr(pipeline_module, "load_arena_config", lambda path: ArenaConfig())
    monkeypatch.setattr(pipeline_module, "run_rust_onnx_pipeline", fake_run_rust_onnx_pipeline)

    try:
        pipeline_module.main()
    except SystemExit as exc:
        assert exc.code == 0

    config = captured["pipeline_config"]
    assert config.onnx_device == "cpu"
