from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import great_kingdom_ai.pipeline as pipeline_module
import numpy as np
import pytest
from great_kingdom_ai.evaluate import ArenaConfig, ArenaReport, summarize_arena
from great_kingdom_ai.features import ACTION_SPACE, BOARD_SIZE, FEATURE_CHANNELS
from great_kingdom_ai.pipeline import (
    PipelineConfig,
    PipelinePrinter,
    generate_self_play_samples,
    load_pipeline_config,
    run_pipeline,
)
from great_kingdom_ai.replay_buffer import ReplaySample
from great_kingdom_ai.self_play import GameLog, MoveLog, SelfPlayConfig
from great_kingdom_ai.train import TrainingConfig


def make_pipeline_config(**overrides: object) -> PipelineConfig:
    data: dict[str, object] = {}
    data.update(overrides)
    return PipelineConfig(**data)


def make_sample(index: int) -> ReplaySample:
    features = np.zeros((FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    features[index % FEATURE_CHANNELS, 0, 0] = 1.0
    policy = np.zeros(ACTION_SPACE, dtype=np.float32)
    policy[index % ACTION_SPACE] = 1.0
    return ReplaySample(features=features, policy=policy, value=1.0)


def test_pipeline_config_defaults_policy_target_scale() -> None:
    config = PipelineConfig()

    assert config.policy_target_c_visit == pytest.approx(5.0)
    assert config.policy_target_c_scale == pytest.approx(0.25)


def fake_self_play_runner(
    seed: int,
    config: SelfPlayConfig,
) -> tuple[GameLog, list[ReplaySample]]:
    return (
        GameLog(
            seed=seed,
            moves=[MoveLog(turn=0, player=1, action=seed % ACTION_SPACE)],
            winner=1,
            end_reason=1,
            territory_scores=(0, 0),
        ),
        [make_sample(seed), make_sample(seed + 1)],
    )


@dataclass(frozen=True)
class FakeTrainSummary:
    start_step: int
    end_step: int
    checkpoint_path: Path
    losses: list[dict[str, float]]


def test_generate_self_play_samples_runs_until_game_and_sample_targets() -> None:
    config = make_pipeline_config(self_play_games=2, min_replay_samples=5, max_self_play_games=4)

    logs, samples = generate_self_play_samples(
        pipeline_config=config,
        runner=fake_self_play_runner,
        printer=PipelinePrinter(enabled=False),
    )

    assert len(logs) == 3
    assert len(samples) == 6
    assert [log.seed for log in logs] == [0, 1, 2]


def test_generate_self_play_samples_can_run_without_game_cap() -> None:
    config = make_pipeline_config(
        self_play_games=1,
        min_replay_samples=5,
        max_self_play_games=None,
    )

    logs, samples = generate_self_play_samples(
        pipeline_config=config,
        runner=fake_self_play_runner,
        printer=PipelinePrinter(enabled=False),
    )

    assert len(logs) == 3
    assert len(samples) == 6


def test_generate_self_play_samples_passes_playout_cap_config() -> None:
    seen_configs: list[SelfPlayConfig] = []

    def recording_runner(
        seed: int,
        config: SelfPlayConfig,
    ) -> tuple[GameLog, list[ReplaySample]]:
        seen_configs.append(config)
        return fake_self_play_runner(seed, config)

    generate_self_play_samples(
        pipeline_config=make_pipeline_config(
            self_play_games=1,
            min_replay_samples=1,
            gumbel_simulations=100,
            playout_cap_randomization=True,
            playout_cap_full_search_fraction=0.25,
            playout_cap_fast_simulations=16,
        ),
        runner=recording_runner,
        printer=PipelinePrinter(enabled=False),
    )

    assert seen_configs[0].playout_cap_randomization is True
    assert seen_configs[0].playout_cap_full_search_fraction == pytest.approx(0.25)
    assert seen_configs[0].playout_cap_full_simulations == 100
    assert seen_configs[0].playout_cap_fast_simulations == 16


def test_generate_self_play_samples_passes_gumbel_config() -> None:
    seen_configs: list[SelfPlayConfig] = []

    def recording_runner(
        seed: int,
        config: SelfPlayConfig,
    ) -> tuple[GameLog, list[ReplaySample]]:
        seen_configs.append(config)
        return fake_self_play_runner(seed, config)

    generate_self_play_samples(
        pipeline_config=make_pipeline_config(
            self_play_games=1,
            min_replay_samples=1,
            gumbel_simulations=32,
            gumbel_max_considered_actions=8,
            policy_target_c_visit=5.0,
            policy_target_c_scale=0.25,
            policy_target_temperature=2.0,
            gumbel_seed=7,
        ),
        runner=recording_runner,
        printer=PipelinePrinter(enabled=False),
    )

    assert seen_configs[0].gumbel_simulations == 32
    assert seen_configs[0].gumbel_max_considered_actions == 8
    assert seen_configs[0].policy_target_c_visit == pytest.approx(5.0)
    assert seen_configs[0].policy_target_c_scale == pytest.approx(0.25)
    assert seen_configs[0].policy_target_temperature == pytest.approx(2.0)
    assert seen_configs[0].gumbel_seed == 7


def test_generate_self_play_samples_prints_progress(capsys) -> None:
    generate_self_play_samples(
        pipeline_config=make_pipeline_config(
            self_play_games=1,
            min_replay_samples=3,
            max_self_play_games=2,
        ),
        runner=fake_self_play_runner,
        printer=PipelinePrinter(),
    )

    output = capsys.readouterr().out
    assert "self-play target" in output
    assert "self-play samples" in output
    assert "avg_samples/game" in output


def test_generate_self_play_samples_uses_batched_model_priors(monkeypatch) -> None:
    seen_batches: list[list[int]] = []

    def fake_batched_games(**kwargs) -> list[tuple[GameLog, list[ReplaySample]]]:
        seeds = list(kwargs["seeds"])
        seen_batches.append(seeds)
        return [fake_self_play_runner(seed, kwargs["config"]) for seed in seeds]

    monkeypatch.setattr(pipeline_module, "play_self_play_games_batched", fake_batched_games)

    logs, samples = generate_self_play_samples(
        pipeline_config=make_pipeline_config(
            self_play_games=2,
            min_replay_samples=1,
            self_play_batch_size=2,
        ),
        printer=PipelinePrinter(enabled=False),
        batch_prior_provider=lambda states: [[0.0] * ACTION_SPACE for _ in states],
    )

    assert [log.seed for log in logs] == [0, 1]
    assert len(samples) == 4
    assert seen_batches == [[0, 1]]


def test_arena_config_for_pipeline_offsets_seed_start_by_iteration() -> None:
    config = pipeline_module._arena_config_for_pipeline(
        ArenaConfig(games=20, seed_start=100000),
        make_pipeline_config(gumbel_seed=2026),
        iteration=3,
    )

    assert config.seed_start == 100040
    assert config.gumbel_seed == 2026


def test_arena_config_for_pipeline_rejects_non_positive_iteration() -> None:
    with pytest.raises(ValueError, match="iteration must be positive"):
        pipeline_module._arena_config_for_pipeline(
            ArenaConfig(games=20, seed_start=100000),
            make_pipeline_config(),
            iteration=0,
        )


def test_run_pipeline_saves_artifacts_and_promotes_candidate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    resume_paths: list[Path | None] = []
    arena_seed_starts: list[int] = []

    def fake_create_train_state(config: TrainingConfig) -> object:
        return object()

    def fake_save_checkpoint(state: object, path: str | Path) -> Path:
        del state
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("checkpoint", encoding="utf-8")
        return destination

    def fake_train_from_replay(
        replay: Any,
        config: TrainingConfig,
        *,
        checkpoint_path: str | Path,
        resume_path: str | Path | None,
        log_every: int,
        progress_callback: Any = None,
    ) -> FakeTrainSummary:
        del replay, config, log_every, progress_callback
        resume_paths.append(Path(resume_path) if resume_path is not None else None)
        destination = Path(checkpoint_path)
        destination.write_text("candidate", encoding="utf-8")
        return FakeTrainSummary(
            start_step=0,
            end_step=2,
            checkpoint_path=destination,
            losses=[],
        )

    def fake_load_model_from_checkpoint(path: str | Path, *, device: str) -> str:
        return f"{device}:{Path(path).name}"

    def fake_run_arena(
        candidate_model: str,
        best_model: str,
        config: ArenaConfig,
        progress_callback=None,
    ) -> ArenaReport:
        del candidate_model, best_model
        from great_kingdom_ai.evaluate import ArenaGameResult

        arena_seed_starts.append(config.seed_start)
        game = ArenaGameResult(
            seed=1,
            candidate_player=1,
            best_player=2,
            winner=1,
            end_reason=1,
            moves=[MoveLog(turn=0, player=1, action=1)],
            territory_scores=(0, 0),
        )
        if progress_callback is not None:
            progress_callback(1, config.games, game)
        return ArenaReport(
            config=config,
            games=[game],
            summary=summarize_arena([game], promotion_threshold=config.promotion_threshold),
        )

    monkeypatch.setattr(pipeline_module, "create_train_state", fake_create_train_state)
    monkeypatch.setattr(pipeline_module, "save_checkpoint", fake_save_checkpoint)
    monkeypatch.setattr(pipeline_module, "train_from_replay", fake_train_from_replay)
    monkeypatch.setattr(
        pipeline_module,
        "load_model_from_checkpoint",
        fake_load_model_from_checkpoint,
    )
    monkeypatch.setattr(pipeline_module, "run_arena", fake_run_arena)

    summary = run_pipeline(
        pipeline_config=make_pipeline_config(
            work_dir=tmp_path,
            iterations=2,
            self_play_games=1,
            min_replay_samples=1,
            replay_capacity=8,
        ),
        train_config=TrainingConfig(batch_size=1, steps=2, device="cpu"),
        arena_config=ArenaConfig(games=1, device="cpu", promotion_threshold=1.0),
        printer=PipelinePrinter(enabled=False),
        self_play_runner=fake_self_play_runner,
    )

    assert len(summary.iterations) == 2
    assert summary.self_play_games == 2
    assert summary.replay_samples == 4
    assert summary.train_end_step == 2
    assert summary.candidate_win_rate == 1.0
    assert summary.promoted is True
    assert summary.artifacts.replay_path.is_file()
    assert summary.artifacts.best_checkpoint.read_text(encoding="utf-8") == "candidate"
    assert (tmp_path / "checkpoints" / "candidates" / "candidate-000001.pt").is_file()
    assert (tmp_path / "checkpoints" / "candidates" / "candidate-000002.pt").is_file()
    assert summary.artifacts.arena_report_path is not None
    assert summary.artifacts.arena_report_path.is_file()
    assert summary.artifacts.metrics_path.is_file()
    assert resume_paths == [
        tmp_path / "checkpoints" / "best.pt",
        tmp_path / "checkpoints" / "best.pt",
    ]
    assert arena_seed_starts == [0, 1]


def test_run_pipeline_resume_continues_iteration_and_arena_seed_windows(
    tmp_path: Path,
    monkeypatch,
) -> None:
    arena_seed_starts: list[int] = []

    (tmp_path / "reports").mkdir(parents=True)
    (tmp_path / "reports" / "metrics.jsonl").write_text(
        '{"iteration": 2}\n',
        encoding="utf-8",
    )

    def fake_create_train_state(config: TrainingConfig) -> object:
        return object()

    def fake_save_checkpoint(state: object, path: str | Path) -> None:
        del state
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("checkpoint", encoding="utf-8")

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
        return FakeTrainSummary(
            start_step=0,
            end_step=1,
            checkpoint_path=destination,
            losses=[],
        )

    def fake_load_model_from_checkpoint(path: str | Path, *, device: str) -> str:
        return f"{device}:{Path(path).name}"

    def fake_run_arena(
        candidate_model: str,
        best_model: str,
        config: ArenaConfig,
        progress_callback=None,
    ) -> ArenaReport:
        del candidate_model, best_model, progress_callback
        from great_kingdom_ai.evaluate import ArenaGameResult

        arena_seed_starts.append(config.seed_start)
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

    monkeypatch.setattr(pipeline_module, "create_train_state", fake_create_train_state)
    monkeypatch.setattr(pipeline_module, "save_checkpoint", fake_save_checkpoint)
    monkeypatch.setattr(pipeline_module, "train_from_replay", fake_train_from_replay)
    monkeypatch.setattr(
        pipeline_module,
        "load_model_from_checkpoint",
        fake_load_model_from_checkpoint,
    )
    monkeypatch.setattr(pipeline_module, "run_arena", fake_run_arena)

    summary = run_pipeline(
        pipeline_config=make_pipeline_config(
            work_dir=tmp_path,
            iterations=2,
            self_play_games=1,
            min_replay_samples=1,
            replay_capacity=8,
            resume=True,
        ),
        train_config=TrainingConfig(batch_size=1, steps=1, device="cpu"),
        arena_config=ArenaConfig(
            games=20,
            seed_start=100000,
            device="cpu",
            promotion_threshold=1.0,
        ),
        printer=PipelinePrinter(enabled=False),
        self_play_runner=fake_self_play_runner,
    )

    assert [iteration.iteration for iteration in summary.iterations] == [3, 4]
    assert arena_seed_starts == [100040, 100060]
    assert (tmp_path / "checkpoints" / "candidates" / "candidate-000003.pt").is_file()
    assert (tmp_path / "checkpoints" / "candidates" / "candidate-000004.pt").is_file()
    assert (tmp_path / "reports" / "arena" / "arena-000003.json").is_file()
    assert (tmp_path / "reports" / "arena" / "arena-000004.json").is_file()


def test_load_pipeline_config_parses_work_dir(tmp_path: Path) -> None:
    path = tmp_path / "pipeline.json"
    path.write_text(
        (
            '{"work_dir": "data/x", "self_play_games": 3, '
            '"policy_target_c_visit": 5.0, "policy_target_c_scale": 0.25}'
        ),
        encoding="utf-8",
    )

    config = load_pipeline_config(path)

    assert config.work_dir == Path("data/x")
    assert config.self_play_games == 3


def test_load_pipeline_config_requires_policy_target_scale(tmp_path: Path) -> None:
    path = tmp_path / "pipeline.json"
    path.write_text('{"work_dir": "data/x", "self_play_games": 3}', encoding="utf-8")

    with pytest.raises(ValueError, match="policy_target_c_visit"):
        load_pipeline_config(path)
