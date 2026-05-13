from __future__ import annotations

from pathlib import Path
from typing import Any

import great_kingdom_ai.train_v2_pipeline as pipeline_module
import numpy as np
import pytest
from great_kingdom_ai.evaluate import ArenaConfig
from great_kingdom_ai.features import ACTION_SPACE, BOARD_SIZE, FEATURE_CHANNELS, PASS_ACTION
from great_kingdom_ai.pipeline_printer import PipelinePrinter
from great_kingdom_ai.reanalyze import ReanalyzeTargetSnapshot
from great_kingdom_ai.rust_onnx_self_play import RustOnnxSelfPlayConfig, RustSelfPlayRunSummary
from great_kingdom_ai.self_play import GameLog, MoveLog, SelfPlayConfig
from great_kingdom_ai.train import TrainingConfig
from great_kingdom_ai.trajectory_replay import (
    TrajectoryEpisode,
    TrajectoryReplayBuffer,
    TrajectoryTransition,
    legal_mask_from_features,
)


def make_features(action: int) -> np.ndarray:
    features = np.zeros((FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    if action != PASS_ACTION:
        features[4, action // BOARD_SIZE, action % BOARD_SIZE] = 1.0
    return features


def make_policy(action: int) -> np.ndarray:
    policy = np.zeros(ACTION_SPACE, dtype=np.float32)
    policy[action] = 1.0
    return policy


def write_text(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def make_episode(seed: int) -> TrajectoryEpisode:
    actions = [seed % 10, PASS_ACTION]
    transitions = tuple(
        TrajectoryTransition(
            episode_id=seed,
            timestep=index,
            player=1 if index == 0 else 2,
            features=make_features(action),
            legal_mask=legal_mask_from_features(make_features(action)),
            action=action,
            policy_target=make_policy(action),
            root_policy_logits=np.zeros(ACTION_SPACE, dtype=np.float32),
            next_features=make_features(actions[index + 1]) if index + 1 < len(actions) else None,
            winner=1,
            terminal=index == len(actions) - 1,
        )
        for index, action in enumerate(actions)
    )
    return TrajectoryEpisode(
        episode_id=seed,
        seed=seed,
        transitions=transitions,
        winner=1,
        end_reason=1,
        territory_scores=(0, 0),
    )


def fake_runner(config: RustOnnxSelfPlayConfig) -> RustSelfPlayRunSummary:
    episodes = tuple(
        make_episode(seed) for seed in range(config.seed_start, config.seed_start + config.games)
    )
    logs = tuple(
        GameLog(
            seed=episode.seed,
            moves=[
                MoveLog(
                    turn=transition.timestep,
                    player=transition.player,
                    action=transition.action,
                )
                for transition in episode.transitions
            ],
            winner=episode.winner,
            end_reason=episode.end_reason,
            territory_scores=episode.territory_scores,
        )
        for episode in episodes
    )
    return RustSelfPlayRunSummary(
        artifact_dir=config.output_dir,
        games=len(episodes),
        samples=sum(len(episode.transitions) for episode in episodes),
        onnx_model_path=config.onnx_model_path,
        onnx_device=config.onnx_device,
        game_logs=logs,
        trajectory_episodes=episodes,
    )


class FakeTrainSummary:
    def __init__(self, checkpoint_path: Path) -> None:
        self.start_step = 0
        self.end_step = 3
        self.checkpoint_path = checkpoint_path
        self.losses: list[dict[str, float]] = []


def test_train_v2_pipeline_wires_trajectory_reanalyze_and_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exported: list[tuple[Path, Path, str]] = []
    reanalyze_calls: list[dict[str, Any]] = []
    trained_replay_types: list[str] = []
    train_steps: list[int] = []

    def fake_save_checkpoint(state: object, path: str | Path) -> Path:
        del state
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("best", encoding="utf-8")
        return destination

    def fake_export(checkpoint_path: str | Path, output_path: str | Path, **kwargs: Any) -> object:
        exported.append(
            (Path(checkpoint_path), Path(output_path), str(kwargs.get("precision", "fp32")))
        )
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text("onnx", encoding="utf-8")
        return object()

    def fake_build_reanalyze_snapshot_from_store(
        replay: Any,
        checkpoint_path: str | Path,
        config: Any,
        progress_callback: Any = None,
    ) -> ReanalyzeTargetSnapshot:
        if progress_callback is not None:
            progress_callback("fake", 1, 1, "done")
        features = replay.features.copy()
        policies = replay.policy_targets.copy()
        values = np.ones((len(replay),), dtype=np.float32)
        snapshot = ReanalyzeTargetSnapshot(
            features=features,
            policies=policies,
            values=values,
            refreshed_values=np.zeros_like(values),
            sample_weights=np.ones_like(values),
            episode_ids=np.repeat(replay.episode_ids, np.diff(replay.episode_offsets)).astype(
                np.int64
            ),
            timesteps=replay.timesteps.copy(),
            players=replay.players.copy(),
            source_model_versions=replay.model_versions.copy(),
            created_iterations=replay.created_iterations.copy(),
            target_ages=np.zeros((len(replay),), dtype=np.int64),
            model_version=config.model_version or 0,
            bootstrap_td_steps=config.bootstrap_td_steps,
            gamma=config.gamma,
            checkpoint_path=str(checkpoint_path),
            policy_logits=np.zeros((len(replay), ACTION_SPACE), dtype=np.float32),
        )
        reanalyze_calls.append(
            {
                "checkpoint_path": Path(checkpoint_path),
                "onnx_model_path": config.onnx_model_path,
                "onnx_device": config.onnx_device,
                "onnx_max_batch_size": config.onnx_max_batch_size,
                "bootstrap_td_steps": config.bootstrap_td_steps,
                "search_policy_target_c_visit": config.search.policy_target_c_visit,
                "search_policy_target_c_scale": config.search.policy_target_c_scale,
            }
        )
        return snapshot

    def fake_train_from_replay(
        replay: Any,
        config: TrainingConfig,
        *,
        checkpoint_path: str | Path,
        resume_path: str | Path | None,
        bootstrap_weights_path: str | Path | None = None,
        log_every: int,
        progress_callback: Any = None,
    ) -> FakeTrainSummary:
        del resume_path, bootstrap_weights_path, log_every
        train_steps.append(config.steps)
        trained_replay_types.append(type(replay).__name__)
        if progress_callback is not None:
            progress_callback(3, 3, {"total": 0.5})
        destination = Path(checkpoint_path)
        destination.write_text("candidate", encoding="utf-8")
        return FakeTrainSummary(destination)

    monkeypatch.setattr(pipeline_module, "create_train_state", lambda config: object())
    monkeypatch.setattr(pipeline_module, "save_checkpoint", fake_save_checkpoint)
    monkeypatch.setattr(pipeline_module, "export_checkpoint_to_onnx", fake_export)
    monkeypatch.setattr(
        pipeline_module,
        "build_reanalyze_snapshot_from_store",
        fake_build_reanalyze_snapshot_from_store,
    )
    monkeypatch.setattr(pipeline_module, "train_from_replay", fake_train_from_replay)
    old_target = write_text(tmp_path / "targets" / "targets-000000.npz")
    old_candidate = write_text(tmp_path / "checkpoints" / "candidates" / "candidate-000000.pt")
    old_onnx = write_text(tmp_path / "checkpoints" / "onnx" / "best-000000.onnx")
    old_self_play = write_text(tmp_path / "self-play" / "iteration-000000" / "marker.txt")

    summary = pipeline_module.run_train_v2_pipeline(
        pipeline_config=pipeline_module.TrainV2PipelineConfig(
            work_dir=tmp_path,
            iterations=1,
            replay_capacity=8,
            self_play_games=1,
            min_replay_transitions=1,
            onnx_precision="fp16",
            train_reuse_factor=1.5,
            min_train_steps=1,
            max_train_steps=None,
            skip_arena=True,
            always_promote=True,
            prune_artifacts=True,
            prune_keep_targets=1,
            prune_keep_candidates=1,
            prune_keep_onnx=1,
            self_play=SelfPlayConfig(policy_target_c_visit=50.0, policy_target_c_scale=1.0),
        ),
        train_config=TrainingConfig(batch_size=1, steps=3, device="cpu"),
        arena_config=ArenaConfig(games=1, device="cpu"),
        printer=PipelinePrinter(enabled=False),
        rust_self_play_runner=fake_runner,
    )

    replay = TrajectoryReplayBuffer.load(tmp_path / "replay" / "trajectory-replay.npz")
    assert len(replay) == 2
    assert summary.replay_transitions == 2
    assert summary.latest_target_snapshot_path == tmp_path / "targets" / "latest.npz"
    assert exported[0] == (
        tmp_path / "checkpoints" / "best.pt",
        tmp_path / "checkpoints" / "onnx" / "best-000001.onnx",
        "fp16",
    )
    assert reanalyze_calls == [
        {
            "checkpoint_path": tmp_path / "checkpoints" / "best.pt",
            "onnx_model_path": str(tmp_path / "checkpoints" / "onnx" / "best-000001.onnx"),
            "onnx_device": "cpu",
            "onnx_max_batch_size": 128,
            "bootstrap_td_steps": 4,
            "search_policy_target_c_visit": 50.0,
            "search_policy_target_c_scale": 1.0,
        }
    ]
    assert trained_replay_types == ["ReanalyzeTargetSnapshot"]
    assert train_steps == [3]
    assert (tmp_path / "checkpoints" / "best.pt").read_text(encoding="utf-8") == "candidate"
    assert (tmp_path / "replay" / "game_logs.jsonl").is_file()
    assert not old_target.exists()
    assert not old_candidate.exists()
    assert not old_onnx.exists()
    assert not old_self_play.parent.exists()
    assert (tmp_path / "targets" / "targets-000001.npz").exists()
    assert (tmp_path / "checkpoints" / "candidates" / "candidate-000001.pt").exists()
    assert (tmp_path / "checkpoints" / "onnx" / "best-000001.onnx").exists()


def test_train_v2_pipeline_uses_on_sample_reanalyze_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_calls: list[dict[str, Any]] = []
    trained_replay_types: list[str] = []

    class FakeOnSampleDataset:
        def __init__(
            self,
            replay: Any,
            *,
            checkpoint_path: str | Path,
            config: Any,
        ) -> None:
            self._size = len(replay)
            self.checkpoint_path = Path(checkpoint_path)
            self.model_version = int(config.model_version or 0)
            self.bootstrap_td_steps = config.bootstrap_td_steps
            self.gamma = config.gamma
            dataset_calls.append(
                {
                    "rows": len(replay),
                    "checkpoint_path": Path(checkpoint_path),
                    "bootstrap_td_steps": config.bootstrap_td_steps,
                    "policy_reanalyze_ratio": config.policy_reanalyze_ratio,
                }
            )

        def __len__(self) -> int:
            return self._size

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
        bootstrap_weights_path: str | Path | None = None,
        log_every: int,
        progress_callback: Any = None,
    ) -> FakeTrainSummary:
        del config, resume_path, bootstrap_weights_path, log_every, progress_callback
        trained_replay_types.append(type(replay).__name__)
        destination = Path(checkpoint_path)
        destination.write_text("candidate", encoding="utf-8")
        return FakeTrainSummary(destination)

    monkeypatch.setattr(pipeline_module, "create_train_state", lambda config: object())
    monkeypatch.setattr(pipeline_module, "save_checkpoint", fake_save_checkpoint)
    monkeypatch.setattr(pipeline_module, "export_checkpoint_to_onnx", fake_export)
    monkeypatch.setattr(pipeline_module, "OnSampleReanalyzeDataset", FakeOnSampleDataset)
    monkeypatch.setattr(pipeline_module, "train_from_replay", fake_train_from_replay)
    write_text(tmp_path / "targets" / "latest.npz")

    summary = pipeline_module.run_train_v2_pipeline(
        pipeline_config=pipeline_module.TrainV2PipelineConfig(
            work_dir=tmp_path,
            iterations=1,
            replay_capacity=8,
            self_play_games=1,
            min_replay_transitions=1,
            reanalyze_mode="on_sample",
            policy_reanalyze_ratio=0.99,
            save_target_snapshots=False,
            skip_arena=True,
            always_promote=True,
            self_play=SelfPlayConfig(policy_target_c_visit=5.0, policy_target_c_scale=0.25),
        ),
        train_config=TrainingConfig(batch_size=1, steps=1, device="cpu"),
        arena_config=ArenaConfig(games=1, device="cpu"),
        printer=PipelinePrinter(enabled=False),
        rust_self_play_runner=fake_runner,
    )

    assert dataset_calls == [
        {
            "rows": 2,
            "checkpoint_path": tmp_path / "checkpoints" / "best.pt",
            "bootstrap_td_steps": 4,
            "policy_reanalyze_ratio": 0.99,
        }
    ]
    assert trained_replay_types == ["FakeOnSampleDataset"]
    assert summary.latest_target_snapshot_path is None
    assert not (tmp_path / "targets" / "targets-000001.npz").exists()


def test_train_config_for_iteration_scales_steps_from_new_transitions() -> None:
    config = pipeline_module._train_config_for_iteration(
        TrainingConfig(batch_size=512, steps=999),
        new_transitions=52120,
        pipeline_config=pipeline_module.TrainV2PipelineConfig(
            train_reuse_factor=1.5,
            min_train_steps=64,
            max_train_steps=192,
            self_play=SelfPlayConfig(policy_target_c_visit=5.0, policy_target_c_scale=0.25),
        ),
    )

    assert config.steps == 153
    assert pipeline_module._effective_reuse_factor(config, 52120) == pytest.approx(
        153 * 512 / 52120
    )


def test_train_v2_parser_exposes_reanalyze_mode() -> None:
    args = pipeline_module.build_parser().parse_args(
        [
            "--reanalyze-mode",
            "on_sample",
            "--policy-reanalyze-ratio",
            "0.99",
            "--dynamic-horizon-enabled",
            "--dynamic-horizon-tau",
            "0.25",
            "--dynamic-horizon-total-steps",
            "50",
        ]
    )

    assert args.reanalyze_mode == "on_sample"
    assert args.policy_reanalyze_ratio == pytest.approx(0.99)
    assert args.dynamic_horizon_enabled is True
    assert args.dynamic_horizon_tau == pytest.approx(0.25)
    assert args.dynamic_horizon_total_steps == 50


def test_train_v2_pipeline_rejects_unknown_reanalyze_mode() -> None:
    with pytest.raises(ValueError, match="reanalyze_mode"):
        pipeline_module._validate_config(
            pipeline_module.TrainV2PipelineConfig(
                reanalyze_mode="invalid",
                self_play=SelfPlayConfig(policy_target_c_visit=5.0, policy_target_c_scale=0.25),
            )
        )


def test_train_v2_pipeline_rejects_invalid_policy_reanalyze_ratio() -> None:
    with pytest.raises(ValueError, match="policy_reanalyze_ratio"):
        pipeline_module._validate_config(
            pipeline_module.TrainV2PipelineConfig(
                policy_reanalyze_ratio=1.1,
                self_play=SelfPlayConfig(policy_target_c_visit=5.0, policy_target_c_scale=0.25),
            )
        )


def test_train_v2_pipeline_requires_dynamic_horizon_total_steps() -> None:
    with pytest.raises(ValueError, match="dynamic_horizon_total_steps"):
        pipeline_module._validate_config(
            pipeline_module.TrainV2PipelineConfig(
                dynamic_horizon_enabled=True,
                self_play=SelfPlayConfig(policy_target_c_visit=5.0, policy_target_c_scale=0.25),
            )
        )


def test_train_v2_pipeline_requires_trajectory_episodes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def runner_without_trajectories(config: RustOnnxSelfPlayConfig) -> RustSelfPlayRunSummary:
        return RustSelfPlayRunSummary(
            artifact_dir=config.output_dir,
            games=1,
            samples=0,
            onnx_model_path=config.onnx_model_path,
            onnx_device=config.onnx_device,
        )

    monkeypatch.setattr(pipeline_module, "create_train_state", lambda config: object())
    monkeypatch.setattr(
        pipeline_module,
        "save_checkpoint",
        lambda state, path: Path(path).write_text("best", encoding="utf-8") or Path(path),
    )
    monkeypatch.setattr(
        pipeline_module,
        "export_checkpoint_to_onnx",
        lambda checkpoint_path, output_path, **kwargs: Path(output_path).write_text("onnx"),
    )

    with pytest.raises(RuntimeError, match="trajectory_episodes"):
        pipeline_module.run_train_v2_pipeline(
            pipeline_config=pipeline_module.TrainV2PipelineConfig(
                work_dir=tmp_path,
                iterations=1,
                replay_capacity=8,
                self_play_games=1,
                self_play=SelfPlayConfig(policy_target_c_visit=5.0, policy_target_c_scale=0.25),
            ),
            train_config=TrainingConfig(batch_size=1, steps=1, device="cpu"),
            arena_config=ArenaConfig(games=1, device="cpu"),
            printer=PipelinePrinter(enabled=False),
            rust_self_play_runner=runner_without_trajectories,
        )
