from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import great_kingdom_ai.actor_learner_v2 as actor_learner_v2_module
import numpy as np
import pytest
from great_kingdom_ai.actor_learner_v2 import (
    ActorV2Config,
    FactoryInitV2Config,
    LearnerV2Config,
    _continuous_train_steps,
    _next_actor_seed_start,
    _run_learner_cli,
    load_v2_shard_records,
    pending_v2_shards,
    run_actor_v2_once,
    run_factory_init_v2_once,
    run_learner_v2_once,
)
from great_kingdom_ai.features import ACTION_SPACE, BOARD_SIZE, FEATURE_CHANNELS, PASS_ACTION
from great_kingdom_ai.pipeline_printer import PipelinePrinter
from great_kingdom_ai.rust_onnx_self_play import RustOnnxSelfPlayConfig, RustSelfPlayRunSummary
from great_kingdom_ai.self_play import GameLog, MoveLog
from great_kingdom_ai.train import TrainingConfig
from great_kingdom_ai.trajectory_replay import (
    TrajectoryEpisode,
    TrajectoryReplayStore,
    TrajectoryTransition,
    legal_mask_from_features,
)


def make_features(action: int = PASS_ACTION) -> np.ndarray:
    features = np.zeros((FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    if action != PASS_ACTION:
        features[4, action // BOARD_SIZE, action % BOARD_SIZE] = 1.0
    return features


def make_policy(action: int = PASS_ACTION) -> np.ndarray:
    policy = np.zeros((ACTION_SPACE,), dtype=np.float32)
    policy[action] = 1.0
    return policy


def make_episode(seed: int) -> TrajectoryEpisode:
    transitions = []
    for timestep, player in enumerate((1, 2)):
        action = 1 + timestep
        features = make_features(action)
        transitions.append(
            TrajectoryTransition(
                episode_id=seed,
                timestep=timestep,
                player=player,
                features=features,
                legal_mask=legal_mask_from_features(features),
                action=action,
                policy_target=make_policy(action),
                root_policy_logits=np.zeros((ACTION_SPACE,), dtype=np.float32),
                next_features=features.copy(),
                winner=1,
                terminal=timestep == 1,
            )
        )
    return TrajectoryEpisode(
        episode_id=seed,
        seed=seed,
        transitions=tuple(transitions),
        winner=1,
        end_reason=1,
        territory_scores=(2, 0),
    )


def fake_actor_runner(config: RustOnnxSelfPlayConfig) -> RustSelfPlayRunSummary:
    episodes = tuple(make_episode(config.seed_start + index) for index in range(config.games))
    logs = tuple(
        GameLog(
            seed=episode.seed,
            moves=[MoveLog(turn=0, player=1, action=1)],
            winner=1,
            end_reason=1,
            territory_scores=(2, 0),
        )
        for episode in episodes
    )
    transitions = sum(len(episode.transitions) for episode in episodes)
    return RustSelfPlayRunSummary(
        artifact_dir=config.output_dir,
        games=config.games,
        samples=transitions,
        onnx_model_path=config.onnx_model_path,
        onnx_device=config.onnx_device,
        replay_samples=(),
        game_logs=logs,
        trajectory_episodes=episodes,
    )


class FakeTrainSummary:
    def __init__(self, checkpoint_path: Path) -> None:
        self.start_step = 4
        self.end_step = 7
        self.checkpoint_path = checkpoint_path
        self.losses: list[dict[str, float]] = []


class FakeFactoryState:
    step = 0


def test_factory_init_v2_writes_initial_checkpoint_and_onnx(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    def fake_state_factory(config: TrainingConfig) -> FakeFactoryState:
        calls.append({"model_preset": config.model_preset, "device": config.device})
        return FakeFactoryState()

    def fake_save(state: Any, path: str | Path) -> Path:
        assert isinstance(state, FakeFactoryState)
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("checkpoint", encoding="utf-8")
        return destination

    def fake_export(checkpoint_path: str | Path, output_path: str | Path, **kwargs: Any) -> None:
        calls.append(
            {
                "checkpoint_path": Path(checkpoint_path),
                "output_path": Path(output_path),
                "kwargs": kwargs,
            }
        )
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text("onnx", encoding="utf-8")

    summary = run_factory_init_v2_once(
        FactoryInitV2Config(work_dir=tmp_path, onnx_device="cpu", onnx_precision="fp32"),
        TrainingConfig(model_preset="small", device="cpu"),
        state_factory=fake_state_factory,
        checkpoint_saver=fake_save,
        onnx_exporter=fake_export,
        printer=PipelinePrinter(enabled=False),
    )

    assert summary.checkpoint_path == tmp_path / "checkpoints" / "training-latest.pt"
    assert summary.onnx_output_path == (
        tmp_path / "checkpoints" / "onnx" / "training-latest.onnx"
    )
    assert summary.step == 0
    assert summary.overwritten is False
    assert summary.to_dict()["model_preset"] == "small"
    assert summary.checkpoint_path.read_text(encoding="utf-8") == "checkpoint"
    assert summary.onnx_output_path.read_text(encoding="utf-8") == "onnx"
    assert calls == [
        {"model_preset": "small", "device": "cpu"},
        {
            "checkpoint_path": summary.checkpoint_path,
            "output_path": summary.onnx_output_path.with_suffix(".onnx.tmp"),
            "kwargs": {
                "device": "cpu",
                "precision": "fp32",
                "dummy_batch_size": 2,
                "prefer_ema": True,
            },
        },
    ]


def test_factory_init_v2_refuses_to_overwrite_outputs(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoints" / "training-latest.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError, match="factory init output already exists"):
        run_factory_init_v2_once(
            FactoryInitV2Config(work_dir=tmp_path),
            TrainingConfig(),
            state_factory=lambda config: FakeFactoryState(),
            checkpoint_saver=lambda state, path: Path(path),
            onnx_exporter=lambda *args, **kwargs: None,
            printer=PipelinePrinter(enabled=False),
        )


def test_factory_init_v2_checks_onnx_dependencies_before_writing(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    import_calls: list[str] = []

    def fake_import_module(name: str) -> Any:
        import_calls.append(name)
        if name == "onnx":
            raise ModuleNotFoundError(name)
        return object()

    monkeypatch.setattr(actor_learner_v2_module.importlib, "import_module", fake_import_module)

    with pytest.raises(RuntimeError, match="factory ONNX export dependencies are missing"):
        run_factory_init_v2_once(
            FactoryInitV2Config(work_dir=tmp_path, onnx_precision="fp16"),
            TrainingConfig(),
            state_factory=lambda config: FakeFactoryState(),
            checkpoint_saver=lambda state, path: Path(path),
            onnx_exporter=lambda *args, **kwargs: None,
            printer=PipelinePrinter(enabled=False),
        )

    assert import_calls == ["onnx", "onnxconverter_common"]
    assert not (tmp_path / "checkpoints" / "training-latest.pt").exists()


def test_actor_v2_writes_trajectory_shard_metadata(tmp_path: Path) -> None:
    summary = run_actor_v2_once(
        ActorV2Config(
            work_dir=tmp_path,
            onnx_model_path=tmp_path / "model.onnx",
            model_version="ema",
            model_iteration=42,
            games=2,
            seed_start=10,
            onnx_device="cpu",
        ),
        runner=fake_actor_runner,
        printer=PipelinePrinter(enabled=False),
    )

    assert summary.shard.shard_id == "ema-seed-00000010-games-0002"
    assert summary.shard.transitions == 4
    assert summary.shard.replay_path.is_file()
    replay = TrajectoryReplayStore.load(summary.shard.replay_path)
    assert len(replay) == 4
    assert replay.root_policy_logits is not None
    assert replay.root_policy_logits.shape == (4, ACTION_SPACE)
    assert replay.model_versions.tolist() == [42, 42, 42, 42]
    assert replay.created_iterations.tolist() == [42, 42, 42, 42]
    assert replay.next_features is None
    assert pending_v2_shards(tmp_path / "shards" / "metadata.jsonl")[0].shard_id == (
        summary.shard.shard_id
    )


def test_actor_v2_next_seed_resumes_from_metadata(tmp_path: Path) -> None:
    first = run_actor_v2_once(
        ActorV2Config(
            work_dir=tmp_path,
            onnx_model_path=tmp_path / "model.onnx",
            model_version="training-latest",
            games=64,
            seed_start=1000000,
            onnx_device="cpu",
        ),
        runner=fake_actor_runner,
        printer=PipelinePrinter(enabled=False),
    )
    second = run_actor_v2_once(
        ActorV2Config(
            work_dir=tmp_path,
            onnx_model_path=tmp_path / "model.onnx",
            model_version="training-latest",
            games=64,
            seed_start=first.shard.seed_start + first.shard.games,
            onnx_device="cpu",
        ),
        runner=fake_actor_runner,
        printer=PipelinePrinter(enabled=False),
    )

    next_seed = _next_actor_seed_start(
        ActorV2Config(
            work_dir=tmp_path,
            model_version="training-latest",
            games=64,
            seed_start=1000000,
        )
    )

    assert second.shard.seed_start == 1000064
    assert next_seed == 1000128


def test_actor_v2_cli_reserves_seed_ranges_across_restarts(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(actor_learner_v2_module, "run_rust_onnx_self_play", fake_actor_runner)
    args = argparse.Namespace(loop=True, max_cycles=2, sleep_seconds=0.0, json=True)
    config = ActorV2Config(
        work_dir=tmp_path,
        onnx_model_path=tmp_path / "model.onnx",
        model_version="training-latest",
        games=2,
        seed_start=100,
        onnx_device="cpu",
    )

    first = actor_learner_v2_module._run_actor_cli(config, args)
    second = actor_learner_v2_module._run_actor_cli(config, args)

    assert [summary["shard"]["seed_start"] for summary in first] == [100, 102]
    assert [summary["shard"]["seed_start"] for summary in second] == [104, 106]
    assert json.loads((tmp_path / "shards" / "actor-seed-state.json").read_text()) == {
        "training-latest": 108
    }


def test_learner_v2_imports_pending_shards_trains_and_exports(tmp_path: Path) -> None:
    run_actor_v2_once(
        ActorV2Config(
            work_dir=tmp_path,
            onnx_model_path=tmp_path / "model.onnx",
            model_version="ema",
            games=2,
            seed_start=0,
            onnx_device="cpu",
        ),
        runner=fake_actor_runner,
        printer=PipelinePrinter(enabled=False),
    )
    train_calls: list[dict[str, Any]] = []
    export_calls: list[dict[str, Any]] = []

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
        batch = replay.sample_arrays(2, __import__("random").Random(0))
        train_calls.append(
            {
                "replay_rows": len(replay),
                "batch_shape": batch.features.shape,
                "resume_path": resume_path,
                "bootstrap_weights_path": bootstrap_weights_path,
            }
        )
        if progress_callback is not None:
            progress_callback(1, 1, {"total": 0.1})
        destination = Path(checkpoint_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("candidate", encoding="utf-8")
        return FakeTrainSummary(destination)

    def fake_export(
        checkpoint_path: str | Path,
        output_path: str | Path,
        **kwargs: Any,
    ) -> None:
        export_calls.append(
            {
                "checkpoint_path": Path(checkpoint_path),
                "output_path": Path(output_path),
                "kwargs": kwargs,
            }
        )
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text("onnx", encoding="utf-8")

    summary = run_learner_v2_once(
        LearnerV2Config(
            work_dir=tmp_path,
            replay_capacity=16,
            min_replay_transitions=1,
            onnx_device="cpu",
        ),
        TrainingConfig(batch_size=2, steps=1, device="cpu"),
        trainer=fake_train,
        onnx_exporter=fake_export,
        printer=PipelinePrinter(enabled=False),
    )

    assert summary.trained is True
    assert summary.imported_transitions == 4
    assert summary.replay_transitions == 4
    assert summary.cycle_seconds >= 0.0
    assert summary.to_dict()["cycle_seconds"] >= 0.0
    assert train_calls == [
        {
            "replay_rows": 4,
            "batch_shape": (2, FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE),
            "resume_path": None,
            "bootstrap_weights_path": None,
        }
    ]
    assert export_calls[0]["checkpoint_path"] == tmp_path / "checkpoints" / "training-latest.pt"
    assert export_calls[0]["output_path"] == (
        tmp_path / "checkpoints" / "onnx" / "training-latest.onnx.tmp"
    )
    assert (tmp_path / "checkpoints" / "onnx" / "training-latest.onnx").read_text(
        encoding="utf-8"
    ) == "onnx"
    records = load_v2_shard_records(tmp_path / "shards" / "metadata.jsonl")
    assert [record.status for record in records] == ["imported"]
    replay = TrajectoryReplayStore.load(tmp_path / "replay" / "trajectory-replay.npz")
    assert replay.root_policy_logits is not None
    assert replay.next_features is None
    assert (tmp_path / "replay" / "game_logs.jsonl").is_file()


def test_learner_v2_passes_one_shot_optimizer_lr_override(tmp_path: Path) -> None:
    run_actor_v2_once(
        ActorV2Config(
            work_dir=tmp_path,
            onnx_model_path=tmp_path / "model.onnx",
            model_version="ema",
            games=2,
            seed_start=0,
            onnx_device="cpu",
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
        resume_optimizer_lr_override: float | None = None,
        log_every: int,
        progress_callback: Any = None,
    ) -> FakeTrainSummary:
        del replay, config, resume_path, bootstrap_weights_path, log_every, progress_callback
        train_calls.append({"resume_optimizer_lr_override": resume_optimizer_lr_override})
        destination = Path(checkpoint_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("candidate", encoding="utf-8")
        return FakeTrainSummary(destination)

    run_learner_v2_once(
        LearnerV2Config(
            work_dir=tmp_path,
            replay_capacity=16,
            min_replay_transitions=1,
            export_onnx=False,
        ),
        TrainingConfig(batch_size=2, steps=1, device="cpu"),
        trainer=fake_train,
        resume_optimizer_lr_override=5e-5,
        printer=PipelinePrinter(enabled=False),
    )

    assert train_calls == [{"resume_optimizer_lr_override": 5e-5}]


def test_learner_v2_continuous_consumes_optimizer_lr_override_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_actor_v2_once(
        ActorV2Config(
            work_dir=tmp_path,
            onnx_model_path=tmp_path / "model.onnx",
            model_version="ema",
            games=2,
            seed_start=0,
            onnx_device="cpu",
        ),
        runner=fake_actor_runner,
        printer=PipelinePrinter(enabled=False),
    )
    train_calls: list[float | None] = []

    def fake_train(
        replay: Any,
        config: TrainingConfig,
        *,
        checkpoint_path: str | Path,
        resume_path: str | Path | None,
        bootstrap_weights_path: str | Path | None = None,
        resume_optimizer_lr_override: float | None = None,
        log_every: int,
        progress_callback: Any = None,
    ) -> FakeTrainSummary:
        del replay, config, resume_path, bootstrap_weights_path, log_every, progress_callback
        train_calls.append(resume_optimizer_lr_override)
        destination = Path(checkpoint_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("candidate", encoding="utf-8")
        return FakeTrainSummary(destination)

    def fake_export(
        checkpoint_path: str | Path,
        output_path: str | Path,
        **kwargs: Any,
    ) -> None:
        del checkpoint_path, kwargs
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text("onnx", encoding="utf-8")

    monkeypatch.setattr(actor_learner_v2_module, "train_from_replay", fake_train)
    monkeypatch.setattr(actor_learner_v2_module, "export_checkpoint_to_onnx", fake_export)

    args = argparse.Namespace(
        loop=True,
        max_cycles=2,
        json=True,
        sleep_seconds=0.0,
        override_optimizer_lr=5e-5,
    )
    actor_learner_v2_module._run_learner_continuous_cli(
        LearnerV2Config(
            work_dir=tmp_path,
            replay_capacity=32,
            min_replay_transitions=1,
            train_reuse_factor=2.0,
            onnx_device="cpu",
        ),
        TrainingConfig(batch_size=2, steps=1, device="cpu"),
        args,
    )

    assert train_calls == [5e-5, None]


def test_learner_v2_continuous_keeps_optimizer_lr_override_while_waiting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_calls: list[float | None] = []

    def fake_train(
        replay: Any,
        config: TrainingConfig,
        *,
        checkpoint_path: str | Path,
        resume_path: str | Path | None,
        bootstrap_weights_path: str | Path | None = None,
        resume_optimizer_lr_override: float | None = None,
        log_every: int,
        progress_callback: Any = None,
    ) -> FakeTrainSummary:
        del replay, config, resume_path, bootstrap_weights_path, log_every, progress_callback
        train_calls.append(resume_optimizer_lr_override)
        destination = Path(checkpoint_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("candidate", encoding="utf-8")
        return FakeTrainSummary(destination)

    def fake_export(
        checkpoint_path: str | Path,
        output_path: str | Path,
        **kwargs: Any,
    ) -> None:
        del checkpoint_path, kwargs
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text("onnx", encoding="utf-8")

    sleep_calls = 0

    def fake_sleep(seconds: float) -> None:
        nonlocal sleep_calls
        del seconds
        sleep_calls += 1
        if sleep_calls == 1:
            run_actor_v2_once(
                ActorV2Config(
                    work_dir=tmp_path,
                    onnx_model_path=tmp_path / "model.onnx",
                    model_version="ema",
                    games=2,
                    seed_start=0,
                    onnx_device="cpu",
                ),
                runner=fake_actor_runner,
                printer=PipelinePrinter(enabled=False),
            )

    monkeypatch.setattr(actor_learner_v2_module, "train_from_replay", fake_train)
    monkeypatch.setattr(actor_learner_v2_module, "export_checkpoint_to_onnx", fake_export)
    monkeypatch.setattr(actor_learner_v2_module.time, "sleep", fake_sleep)

    args = argparse.Namespace(
        loop=True,
        max_cycles=2,
        json=True,
        sleep_seconds=0.0,
        override_optimizer_lr=5e-5,
    )
    actor_learner_v2_module._run_learner_continuous_cli(
        LearnerV2Config(
            work_dir=tmp_path,
            replay_capacity=32,
            min_replay_transitions=1,
            train_reuse_factor=2.0,
            onnx_device="cpu",
        ),
        TrainingConfig(batch_size=2, steps=1, device="cpu"),
        args,
    )

    assert train_calls == [5e-5]


def test_learner_v2_prunes_imported_shards_after_successful_cycle(tmp_path: Path) -> None:
    first = run_actor_v2_once(
        ActorV2Config(
            work_dir=tmp_path,
            onnx_model_path=tmp_path / "model.onnx",
            model_version="ema",
            games=1,
            seed_start=0,
            onnx_device="cpu",
        ),
        runner=fake_actor_runner,
        printer=PipelinePrinter(enabled=False),
    )
    second = run_actor_v2_once(
        ActorV2Config(
            work_dir=tmp_path,
            onnx_model_path=tmp_path / "model.onnx",
            model_version="ema",
            games=1,
            seed_start=1,
            onnx_device="cpu",
        ),
        runner=fake_actor_runner,
        printer=PipelinePrinter(enabled=False),
    )

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
        del replay, config, resume_path, bootstrap_weights_path, log_every, progress_callback
        destination = Path(checkpoint_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("candidate", encoding="utf-8")
        return FakeTrainSummary(destination)

    summary = run_learner_v2_once(
        LearnerV2Config(
            work_dir=tmp_path,
            replay_capacity=16,
            min_replay_transitions=1,
            export_onnx=False,
            prune_artifacts=True,
            prune_keep_imported_shards=0,
        ),
        TrainingConfig(batch_size=2, steps=1, device="cpu"),
        trainer=fake_train,
        printer=PipelinePrinter(enabled=False),
    )

    assert summary.pruned_artifacts == 2
    assert summary.pruned_bytes > 0
    assert not first.shard.shard_dir.exists()
    assert not second.shard.shard_dir.exists()
    assert (tmp_path / "replay" / "trajectory-replay.npz").is_file()
    assert (tmp_path / "checkpoints" / "training-latest.pt").is_file()
    records = load_v2_shard_records(tmp_path / "shards" / "metadata.jsonl")
    assert [record.status for record in records] == [
        "imported",
        "imported",
    ]


def test_learner_v2_waits_until_min_replay_transitions(tmp_path: Path) -> None:
    summary = run_learner_v2_once(
        LearnerV2Config(
            work_dir=tmp_path,
            replay_capacity=16,
            min_replay_transitions=8,
            export_onnx=False,
        ),
        TrainingConfig(batch_size=2, steps=1, device="cpu"),
        trainer=lambda *args, **kwargs: None,
        printer=PipelinePrinter(enabled=False),
    )

    assert summary.trained is False
    assert summary.replay_transitions is None
    assert summary.cycle_seconds >= 0.0


def test_learner_v2_does_not_load_replay_when_no_pending_shards(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    def fail_load(path: Path) -> TrajectoryReplayStore:
        raise AssertionError(f"unexpected replay load: {path}")

    monkeypatch.setattr(actor_learner_v2_module.TrajectoryReplayStore, "load", fail_load)

    summary = run_learner_v2_once(
        LearnerV2Config(
            work_dir=tmp_path,
            replay_capacity=16,
            min_replay_transitions=1,
            export_onnx=False,
        ),
        TrainingConfig(batch_size=2, steps=1, device="cpu"),
        trainer=lambda *args, **kwargs: None,
        printer=PipelinePrinter(enabled=False),
    )

    assert summary.trained is False
    assert summary.replay_transitions is None


def test_continuous_train_steps_follow_reuse_budget() -> None:
    assert (
        _continuous_train_steps(
            train_budget_samples=3000 * 16,
            replay_transitions=8192,
            train_config=TrainingConfig(batch_size=1024, steps=64),
            min_replay_transitions=8192,
        )
        == 46
    )
    assert (
        _continuous_train_steps(
            train_budget_samples=3000 * 32,
            replay_transitions=8192,
            train_config=TrainingConfig(batch_size=1024, steps=64),
            min_replay_transitions=8192,
        )
        == 64
    )
    assert (
        _continuous_train_steps(
            train_budget_samples=1023,
            replay_transitions=8192,
            train_config=TrainingConfig(batch_size=1024, steps=64),
            min_replay_transitions=8192,
        )
        == 0
    )


def test_learner_v2_loop_trains_from_imported_transition_budget(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    run_actor_v2_once(
        ActorV2Config(
            work_dir=tmp_path,
            onnx_model_path=tmp_path / "model.onnx",
            model_version="ema",
            games=2,
            seed_start=0,
            onnx_device="cpu",
        ),
        runner=fake_actor_runner,
        printer=PipelinePrinter(enabled=False),
    )
    train_steps: list[int] = []

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
        del replay, resume_path, bootstrap_weights_path, log_every, progress_callback
        train_steps.append(config.steps)
        destination = Path(checkpoint_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("candidate", encoding="utf-8")
        return FakeTrainSummary(destination)

    monkeypatch.setattr(actor_learner_v2_module, "train_from_replay", fake_train)
    summaries = _run_learner_cli(
        LearnerV2Config(
            work_dir=tmp_path,
            replay_capacity=16,
            min_replay_transitions=1,
            export_onnx=False,
            train_reuse_factor=16.0,
        ),
        TrainingConfig(batch_size=2, steps=64, device="cpu"),
        argparse.Namespace(loop=True, max_cycles=1, sleep_seconds=0.0, json=True),
    )

    assert train_steps == [32]
    assert summaries[0]["trained"] is True
    assert summaries[0]["imported_transitions"] == 4
