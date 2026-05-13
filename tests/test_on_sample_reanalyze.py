from __future__ import annotations

import importlib
import importlib.util
import random
from pathlib import Path

import great_kingdom_ai.on_sample_reanalyze as on_sample_reanalyze_module
import great_kingdom_ai.reanalyze as reanalyze_module
import numpy as np
import pytest
from great_kingdom_ai.features import ACTION_SPACE, BOARD_SIZE, FEATURE_CHANNELS, PASS_ACTION
from great_kingdom_ai.on_sample_reanalyze import OnSampleReanalyzeDataset
from great_kingdom_ai.priority_sampling import PrioritySampleResult, PrioritySamplingConfig
from great_kingdom_ai.reanalyze import ReanalyzeConfig, build_reanalyze_snapshot_from_store
from great_kingdom_ai.search_reanalyze import SearchReanalyzeResult
from great_kingdom_ai.train import TrainingConfig, create_train_state, save_checkpoint
from great_kingdom_ai.trajectory_replay import (
    TrajectoryEpisode,
    TrajectoryReplayBuffer,
    TrajectoryReplayStore,
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


def make_episode(
    *,
    sample_weights: tuple[float, ...] = (1.0, 1.0, 1.0),
    model_versions: tuple[int, ...] = (10, 10, 10),
    created_iterations: tuple[int, ...] = (3, 3, 3),
    winner: int = 1,
) -> TrajectoryEpisode:
    actions = (1, 2, PASS_ACTION)
    transitions = tuple(
        TrajectoryTransition(
            episode_id=0,
            timestep=index,
            player=1 if index % 2 == 0 else 2,
            features=make_features(action),
            legal_mask=legal_mask_from_features(make_features(action)),
            action=action,
            policy_target=make_policy(action),
            winner=winner,
            terminal=index == len(actions) - 1,
            model_version=model_versions[index],
            created_iteration=created_iterations[index],
            sample_weight=sample_weights[index],
        )
        for index, action in enumerate(actions)
    )
    return TrajectoryEpisode(
        episode_id=0,
        seed=100,
        transitions=transitions,
        winner=winner,
        end_reason=1,
        territory_scores=(4, 1),
    )


def make_store(
    *,
    sample_weights: tuple[float, ...] = (1.0, 1.0, 1.0),
    model_versions: tuple[int, ...] = (10, 10, 10),
    created_iterations: tuple[int, ...] = (3, 3, 3),
) -> TrajectoryReplayStore:
    replay = TrajectoryReplayBuffer(capacity=8)
    replay.push_episode(
        make_episode(
            sample_weights=sample_weights,
            model_versions=model_versions,
            created_iterations=created_iterations,
        )
    )
    return TrajectoryReplayStore.from_episodes(replay.capacity, replay.episodes)


def make_checkpoint(tmp_path: Path) -> Path:
    state = create_train_state(TrainingConfig(batch_size=2, seed=9))
    return save_checkpoint(state, tmp_path / "checkpoint.pt")


@pytest.mark.skipif(_torch_spec is None, reason="torch is not installed")
def test_on_sample_priority_update_uses_config_and_next_priority_sample(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = make_store(model_versions=(10, 5, 10))
    dataset = OnSampleReanalyzeDataset(
        store,
        checkpoint_path=make_checkpoint(tmp_path),
        config=ReanalyzeConfig(batch_size=3, model_version=10),
    )

    def fake_evaluate_logits_values(
        features: np.ndarray,
        legal_masks: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        del legal_masks
        return (
            np.stack([make_policy(PASS_ACTION) for _ in range(features.shape[0])], axis=0),
            np.zeros((features.shape[0],), dtype=np.float32),
        )

    monkeypatch.setattr(dataset, "_evaluate_logits_values", fake_evaluate_logits_values)
    priority_config = PrioritySamplingConfig(
        enabled=True,
        alpha=1.0,
        beta=0.0,
        value_error_weight=1.0,
        policy_kl_weight=0.0,
        target_age_weight=10.0,
    )

    dataset.sample_arrays(3, random.Random(0), priority_config=priority_config)

    assert dataset.priorities[1] > dataset.priorities[0]
    assert dataset.priorities[1] > dataset.priorities[2]

    def choose_largest_priority(
        *,
        priorities: np.ndarray,
        batch_size: int,
        rng: random.Random,
        beta: float = 0.0,
        recent_fraction: float = 0.0,
        recent_window: int = 0,
    ) -> PrioritySampleResult:
        del rng, beta, recent_fraction, recent_window
        assert batch_size == 1
        return PrioritySampleResult(
            indexes=[int(np.argmax(priorities))],
            importance_weights=np.ones((1,), dtype=np.float32),
        )

    monkeypatch.setattr(
        on_sample_reanalyze_module,
        "sample_priority_indexes",
        choose_largest_priority,
    )

    next_batch = dataset.sample_arrays(1, random.Random(1), priority_config=priority_config)

    assert next_batch.indexes.tolist() == [1]


@pytest.mark.skipif(_torch_spec is None, reason="torch is not installed")
def test_on_sample_priority_zero_weights_keep_initial_replay_weights(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = make_store(sample_weights=(1.0, 5.0, 2.0), model_versions=(10, 5, 10))
    dataset = OnSampleReanalyzeDataset(
        store,
        checkpoint_path=make_checkpoint(tmp_path),
        config=ReanalyzeConfig(batch_size=3, model_version=10),
    )
    seen_priorities: list[np.ndarray] = []

    def fake_sample_priority_indexes(
        *,
        priorities: np.ndarray,
        batch_size: int,
        rng: random.Random,
        beta: float = 0.0,
        recent_fraction: float = 0.0,
        recent_window: int = 0,
    ) -> PrioritySampleResult:
        del priorities, rng, beta, recent_fraction, recent_window
        seen_priorities.append(dataset.priorities)
        return PrioritySampleResult(
            indexes=list(range(batch_size)),
            importance_weights=np.ones((batch_size,), dtype=np.float32),
        )

    monkeypatch.setattr(
        on_sample_reanalyze_module,
        "sample_priority_indexes",
        fake_sample_priority_indexes,
    )
    priority_config = PrioritySamplingConfig(
        enabled=True,
        alpha=1.0,
        beta=0.0,
        value_error_weight=0.0,
        policy_kl_weight=0.0,
        target_age_weight=0.0,
    )

    dataset.sample_arrays(3, random.Random(0), priority_config=priority_config)

    assert seen_priorities[0].tolist() == pytest.approx([1.0, 5.0, 2.0])
    assert dataset.priorities.tolist() == pytest.approx([1.0, 5.0, 2.0])


@pytest.mark.skipif(_torch_spec is None, reason="torch is not installed")
def test_on_sample_priority_sampling_combines_base_and_importance_weights(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = make_store(sample_weights=(2.0, 3.0, 4.0))
    dataset = OnSampleReanalyzeDataset(
        store,
        checkpoint_path=make_checkpoint(tmp_path),
        config=ReanalyzeConfig(batch_size=1, model_version=10),
    )

    def fake_sample_priority_indexes(
        *,
        priorities: np.ndarray,
        batch_size: int,
        rng: random.Random,
        beta: float = 0.0,
        recent_fraction: float = 0.0,
        recent_window: int = 0,
    ) -> PrioritySampleResult:
        del priorities, batch_size, rng, beta, recent_fraction, recent_window
        return PrioritySampleResult(
            indexes=[1],
            importance_weights=np.asarray([0.25], dtype=np.float32),
        )

    monkeypatch.setattr(
        on_sample_reanalyze_module,
        "sample_priority_indexes",
        fake_sample_priority_indexes,
    )

    batch = dataset.sample_arrays(
        1,
        random.Random(0),
        priority_config=PrioritySamplingConfig(enabled=True, alpha=1.0, beta=0.4),
    )

    assert batch.indexes.tolist() == [1]
    assert batch.sample_weights.tolist() == pytest.approx([0.75])


@pytest.mark.skipif(_torch_spec is None, reason="torch is not installed")
def test_on_sample_dynamic_horizon_matches_snapshot_row_for_row(tmp_path: Path) -> None:
    store = make_store(created_iterations=(10, 0, 10))
    checkpoint = make_checkpoint(tmp_path)
    config = ReanalyzeConfig(
        batch_size=3,
        bootstrap_td_steps=2,
        gamma=0.5,
        dynamic_horizon_enabled=True,
        dynamic_horizon_tau=0.5,
        dynamic_horizon_total_steps=10,
        model_version=10,
    )
    snapshot = build_reanalyze_snapshot_from_store(
        store,
        checkpoint_path=checkpoint,
        config=config,
    )
    dataset = OnSampleReanalyzeDataset(store, checkpoint_path=checkpoint, config=config)

    batch = dataset.sample_arrays(len(store), random.Random(3))

    for replay_index, value in zip(batch.indexes.tolist(), batch.values.tolist(), strict=True):
        assert value == pytest.approx(float(snapshot.values[replay_index]))


@pytest.mark.skipif(_torch_spec is None, reason="torch is not installed")
def test_on_sample_mcts_root_bootstrap_matches_snapshot_row_for_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = make_store()
    checkpoint = make_checkpoint(tmp_path)
    config = ReanalyzeConfig(
        batch_size=3,
        bootstrap_td_steps=1,
        gamma=0.5,
        value_bootstrap_source="mcts_root",
        model_version=10,
    )

    def fake_refresh_sampled_policies_with_search(**kwargs: object) -> SearchReanalyzeResult:
        policies = np.asarray(kwargs["policies"], dtype=np.float32)
        transitions = kwargs["transitions"]
        root_values = np.asarray(
            [
                0.25 + 0.1 * ref[0].transitions[ref[1]].timestep
                for ref in transitions
            ],
            dtype=np.float32,
        )
        return SearchReanalyzeResult(
            policies=policies.copy(),
            search_reanalyzed=np.ones((policies.shape[0],), dtype=np.bool_),
            selected_indexes=tuple(range(policies.shape[0])),
            root_values=root_values,
        )

    monkeypatch.setattr(
        reanalyze_module,
        "refresh_sampled_policies_with_search",
        fake_refresh_sampled_policies_with_search,
    )
    monkeypatch.setattr(
        on_sample_reanalyze_module,
        "refresh_sampled_policies_with_search",
        fake_refresh_sampled_policies_with_search,
    )

    snapshot = build_reanalyze_snapshot_from_store(
        store,
        checkpoint_path=checkpoint,
        config=config,
    )
    dataset = OnSampleReanalyzeDataset(store, checkpoint_path=checkpoint, config=config)
    batch = dataset.sample_arrays(len(store), random.Random(3))

    for replay_index, value in zip(batch.indexes.tolist(), batch.values.tolist(), strict=True):
        assert value == pytest.approx(float(snapshot.values[replay_index]))
