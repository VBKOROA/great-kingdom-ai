from __future__ import annotations

import importlib
import importlib.util
import random
from pathlib import Path

import numpy as np
import pytest
from great_kingdom_ai.features import ACTION_SPACE, BOARD_SIZE, FEATURE_CHANNELS, PASS_ACTION
from great_kingdom_ai.reanalyze import (
    ReanalyzeConfig,
    ReanalyzeTargetSnapshot,
    build_parser,
    build_reanalyze_snapshot,
    is_reanalyze_target_snapshot,
)
from great_kingdom_ai.train import (
    TrainingConfig,
    create_train_state,
    load_training_replay,
    save_checkpoint,
)
from great_kingdom_ai.trajectory_replay import (
    TrajectoryEpisode,
    TrajectoryReplayBuffer,
    TrajectoryTransition,
    legal_mask_from_features,
)

_torch_spec = importlib.util.find_spec("torch")
torch = importlib.import_module("torch") if _torch_spec is not None else None


def make_features(action: int) -> np.ndarray:
    features = np.zeros((FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    if action != PASS_ACTION:
        features[4, action // BOARD_SIZE, action % BOARD_SIZE] = 1.0
    return features


def make_policy(action: int) -> np.ndarray:
    policy = np.zeros(ACTION_SPACE, dtype=np.float32)
    policy[action] = 1.0
    return policy


def make_episode(*, episode_id: int = 0, winner: int = 1) -> TrajectoryEpisode:
    actions = [1, 2, PASS_ACTION]
    transitions = tuple(
        TrajectoryTransition(
            episode_id=episode_id,
            timestep=index,
            player=1 if index % 2 == 0 else 2,
            features=make_features(action),
            legal_mask=legal_mask_from_features(make_features(action)),
            action=action,
            policy_target=make_policy(action),
            winner=winner,
            terminal=index == len(actions) - 1,
            model_version=4,
            created_iteration=3,
        )
        for index, action in enumerate(actions)
    )
    return TrajectoryEpisode(
        episode_id=episode_id,
        seed=100,
        transitions=transitions,
        winner=winner,
        end_reason=1,
        territory_scores=(4, 1),
    )


def test_reanalyze_target_snapshot_round_trips_and_samples_arrays(tmp_path: Path) -> None:
    features = np.stack([make_features(1), make_features(PASS_ACTION)], axis=0)
    policies = np.stack([make_policy(1), make_policy(PASS_ACTION)], axis=0)
    snapshot = ReanalyzeTargetSnapshot(
        features=features,
        policies=policies,
        values=np.asarray([0.25, 1.0], dtype=np.float32),
        refreshed_values=np.asarray([0.25, 0.5], dtype=np.float32),
        sample_weights=np.asarray([1.0, 2.0], dtype=np.float32),
        episode_ids=np.asarray([7, 7], dtype=np.int64),
        timesteps=np.asarray([0, 1], dtype=np.int64),
        players=np.asarray([1, 2], dtype=np.int64),
        source_model_versions=np.asarray([3, 3], dtype=np.int64),
        created_iterations=np.asarray([2, 2], dtype=np.int64),
        target_ages=np.asarray([5, 5], dtype=np.int64),
        model_version=8,
        bootstrap_td_steps=2,
        gamma=0.5,
        checkpoint_path="checkpoint.pt",
    )
    path = tmp_path / "targets.npz"

    snapshot.save(path)
    loaded = ReanalyzeTargetSnapshot.load(path)
    batch = loaded.sample_arrays(2, random.Random(0))

    assert is_reanalyze_target_snapshot(path)
    assert len(loaded) == 2
    assert loaded.capacity == 2
    assert loaded.model_version == 8
    assert loaded.target_ages.tolist() == [5, 5]
    assert batch.features.shape == (2, FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    assert sorted(batch.sample_weights.tolist()) == pytest.approx([1.0, 2.0])
    assert load_training_replay(path).__class__ is ReanalyzeTargetSnapshot


@pytest.mark.skipif(_torch_spec is None, reason="torch is not installed")
def test_build_reanalyze_snapshot_refreshes_values_and_bootstrap_targets(
    tmp_path: Path,
) -> None:
    replay = TrajectoryReplayBuffer(capacity=8)
    replay.push_episode(make_episode())
    state = create_train_state(TrainingConfig(batch_size=2, seed=9))
    for parameter in state.model.parameters():
        parameter.data.zero_()
    state = type(state)(
        model=state.model,
        optimizer=state.optimizer,
        scheduler=state.scheduler,
        scaler=state.scaler,
        step=9,
        model_preset=state.model_preset,
    )
    checkpoint = save_checkpoint(state, tmp_path / "checkpoint.pt")

    snapshot = build_reanalyze_snapshot(
        replay,
        checkpoint_path=checkpoint,
        config=ReanalyzeConfig(batch_size=2, bootstrap_td_steps=1, gamma=1.0),
    )

    assert snapshot.model_version == 9
    assert snapshot.target_ages.tolist() == [5, 5, 5]
    assert snapshot.refreshed_values.tolist() == pytest.approx([0.0, 0.0, 0.0])
    assert snapshot.values.tolist() == pytest.approx([-0.0, -1.0, 1.0])


def test_reanalyze_parser_exposes_phase3_cli_options() -> None:
    args = build_parser().parse_args(
        [
            "--replay",
            "trajectory.npz",
            "--checkpoint",
            "checkpoint.pt",
            "--output",
            "targets.npz",
            "--batch-size",
            "16",
            "--bootstrap-td-steps",
            "4",
        ]
    )

    assert args.batch_size == 16
    assert args.bootstrap_td_steps == 4
