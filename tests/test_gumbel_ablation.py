from __future__ import annotations

from pathlib import Path
from typing import Any

from great_kingdom_ai import gumbel_ablation as module
from great_kingdom_ai.evaluate import ArenaConfig, ArenaGameResult, ArenaReport, summarize_arena
from great_kingdom_ai.pipeline import PipelinePrinter
from great_kingdom_ai.rust_onnx_pipeline import (
    RustOnnxPipelineConfig,
    RustOnnxPipelineIterationSummary,
    RustOnnxPipelineSummary,
)
from great_kingdom_ai.rust_onnx_replay import RustReplayImportSummary
from great_kingdom_ai.rust_onnx_self_play import RustSelfPlayRunSummary
from great_kingdom_ai.self_play import MoveLog, SelfPlayConfig
from great_kingdom_ai.train import TrainingConfig


def test_ablation_configs_express_pure_and_log_count_gumbel() -> None:
    pure = module.load_rust_onnx_pipeline_config(module.DEFAULT_PURE_PIPELINE_CONFIG)
    modified = module.load_rust_onnx_pipeline_config(module.DEFAULT_MODIFIED_PIPELINE_CONFIG)

    assert pure.self_play.policy_target_c_visit == pure.self_play.gumbel_c_visit
    assert pure.self_play.policy_target_c_scale == pure.self_play.gumbel_c_scale
    assert pure.self_play.policy_target_temperature == 1.0
    assert pure.aggregate_replay is False

    assert modified.self_play.gumbel_c_visit == pure.self_play.gumbel_c_visit
    assert modified.self_play.gumbel_c_scale == pure.self_play.gumbel_c_scale
    assert modified.self_play.policy_target_c_visit == pure.self_play.policy_target_c_visit
    assert modified.self_play.policy_target_c_scale == pure.self_play.policy_target_c_scale
    assert modified.self_play.policy_target_temperature == pure.self_play.policy_target_temperature
    assert modified.aggregate_replay is True
    assert modified.aggregate_replay_weight_mode == "log_count"
    assert modified.aggregate_replay_weight_cap is None

    arena = module.load_arena_config(module.DEFAULT_ARENA_CONFIG)
    assert arena.games == 400
    assert arena.policy_target_c_visit == arena.gumbel_c_visit
    assert arena.policy_target_c_scale == arena.gumbel_c_scale
    assert arena.policy_target_temperature == 1.0


def test_apply_overrides_preserves_nested_self_play_config() -> None:
    pure = RustOnnxPipelineConfig(self_play=SelfPlayConfig(policy_target_c_visit=9.0))
    modified = RustOnnxPipelineConfig(self_play=SelfPlayConfig(policy_target_c_visit=3.0))

    pure, modified, train, arena = module._apply_overrides(
        pure=pure,
        modified=modified,
        train=TrainingConfig(),
        arena=ArenaConfig(),
        device="cuda",
        onnx_device=None,
        train_steps=7,
        batch_size=13,
        self_play_games=5,
        min_replay_samples=11,
        arena_games=4,
    )

    assert isinstance(pure.self_play, SelfPlayConfig)
    assert isinstance(modified.self_play, SelfPlayConfig)
    assert pure.onnx_device == "cuda"
    assert pure.min_replay_samples == 11
    assert modified.self_play.policy_target_c_visit == 3.0
    assert train.device == "cuda"
    assert train.steps == 7
    assert train.batch_size == 13
    assert arena.device == "cuda"
    assert arena.games == 4


def test_run_gumbel_ablation_uses_shared_initial_checkpoint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    seen_work_dirs: list[Path] = []

    def fake_save_initial_checkpoint(path: Path, config: TrainingConfig) -> Path:
        del config
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("initial", encoding="utf-8")
        return path

    def fake_run_pipeline(
        *,
        pipeline_config: RustOnnxPipelineConfig,
        train_config: TrainingConfig,
        arena_config: ArenaConfig,
        printer: PipelinePrinter,
    ) -> RustOnnxPipelineSummary:
        del train_config, arena_config, printer
        seen_work_dirs.append(pipeline_config.work_dir)
        best_path = pipeline_config.work_dir / "checkpoints" / "best.pt"
        assert best_path.read_text(encoding="utf-8") == "initial"
        candidate = pipeline_config.work_dir / "checkpoints" / "candidate.pt"
        candidate.parent.mkdir(parents=True, exist_ok=True)
        candidate.write_text("candidate", encoding="utf-8")
        iteration = RustOnnxPipelineIterationSummary(
            iteration=1,
            onnx_model_path=pipeline_config.work_dir / "checkpoints" / "onnx" / "best.onnx",
            self_play=RustSelfPlayRunSummary(
                artifact_dir=pipeline_config.work_dir / "self-play",
                games=1,
                samples=2,
                onnx_model_path=pipeline_config.work_dir / "best.onnx",
                onnx_device=pipeline_config.onnx_device,
            ),
            replay_import=RustReplayImportSummary(
                artifact_dir=pipeline_config.work_dir / "self-play",
                replay_path=pipeline_config.work_dir / "replay" / "replay.npz",
                imported_samples=2,
                replay_samples=2,
                imported_games=1,
            ),
            train_start_step=0,
            train_end_step=1,
            candidate_checkpoint=candidate,
            candidate_win_rate=None,
            promoted=False,
        )
        return RustOnnxPipelineSummary(
            iterations=[iteration],
            replay_samples=2,
            best_checkpoint=best_path,
            replay_path=pipeline_config.work_dir / "replay" / "replay.npz",
        )

    def fake_run_arena(**kwargs: Any) -> ArenaReport:
        config = kwargs["config"]
        game = ArenaGameResult(
            seed=config.seed_start,
            candidate_player=1,
            best_player=2,
            winner=1,
            end_reason=1,
            moves=[MoveLog(turn=0, player=1, action=1)],
            territory_scores=(0, 0),
        )
        return ArenaReport(
            config=config,
            games=[game],
            summary=summarize_arena([game], promotion_threshold=config.promotion_threshold),
        )

    monkeypatch.setattr(module, "_save_initial_checkpoint", fake_save_initial_checkpoint)
    monkeypatch.setattr(module, "run_rust_onnx_pipeline", fake_run_pipeline)
    monkeypatch.setattr(module, "load_model_from_checkpoint", lambda path, *, device: object())
    monkeypatch.setattr(module, "run_arena", fake_run_arena)

    summary = module.run_gumbel_ablation(
        pure_pipeline_config=RustOnnxPipelineConfig(self_play=SelfPlayConfig()),
        modified_pipeline_config=RustOnnxPipelineConfig(self_play=SelfPlayConfig()),
        train_config=TrainingConfig(batch_size=1, steps=1),
        arena_config=ArenaConfig(games=1),
        run_dir=tmp_path,
        printer=PipelinePrinter(enabled=False),
    )

    assert seen_work_dirs == [tmp_path / "pure", tmp_path / "sqrt-count"]
    assert summary.arena_summary["candidate_win_rate"] == 1.0
    assert (tmp_path / "reports" / "summary.json").is_file()
    assert (tmp_path / "reports" / "sqrt-count-vs-pure-arena.json").is_file()
