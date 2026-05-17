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
from great_kingdom_ai.priority_sampling import PrioritySamplingConfig
from great_kingdom_ai.reanalyze import (
    ReanalyzeConfig,
    ReanalyzeTargetSnapshot,
    SearchReanalyzeConfig,
    build_parser,
    build_reanalyze_snapshot,
    build_reanalyze_snapshot_from_store,
    is_reanalyze_target_snapshot,
)
from great_kingdom_ai.replay import (
    TrajectoryEpisode,
    TrajectoryReplayStore,
    TrajectoryTransition,
    legal_mask_from_features,
)
from great_kingdom_ai.search_reanalyze import select_search_reanalyze_indexes
from great_kingdom_ai.training import (
    TrainingConfig,
    create_train_state,
    save_checkpoint,
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
        policy_logits=np.stack([make_policy(1), make_policy(PASS_ACTION)], axis=0),
        search_reanalyzed=np.asarray([True, False], dtype=np.bool_),
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
    assert loaded.policy_logits is not None
    assert loaded.search_reanalyzed is not None
    assert loaded.search_reanalyzed.tolist() == [True, False]
    assert batch.features.shape == (2, FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    assert sorted(batch.sample_weights.tolist()) == pytest.approx([1.0, 2.0])


def test_dynamic_horizon_shrinks_td_steps_for_older_rows() -> None:
    assert (
        reanalyze_module._effective_bootstrap_td_steps(
            td_steps=5,
            model_version=100,
            created_iteration=100,
            dynamic_horizon_enabled=True,
            dynamic_horizon_tau=0.3,
            dynamic_horizon_total_steps=100,
        )
        == 5
    )
    assert (
        reanalyze_module._effective_bootstrap_td_steps(
            td_steps=5,
            model_version=100,
            created_iteration=40,
            dynamic_horizon_enabled=True,
            dynamic_horizon_tau=0.3,
            dynamic_horizon_total_steps=100,
        )
        == 3
    )
    assert (
        reanalyze_module._effective_bootstrap_td_steps(
            td_steps=5,
            model_version=100,
            created_iteration=0,
            dynamic_horizon_enabled=True,
            dynamic_horizon_tau=0.3,
            dynamic_horizon_total_steps=100,
        )
        == 2
    )


def test_dynamic_horizon_changes_bootstrap_target_for_stale_store_rows() -> None:
    store = TrajectoryReplayStore.from_episodes(8, (make_episode(),))
    refreshed_values = np.asarray([0.1, 0.4, 0.9], dtype=np.float32)

    targets = reanalyze_module._bootstrap_targets_from_store(
        store,
        refreshed_values,
        td_steps=2,
        gamma=1.0,
        model_version=10,
        dynamic_horizon_enabled=True,
        dynamic_horizon_tau=0.5,
        dynamic_horizon_total_steps=10,
    )

    assert targets.tolist() == pytest.approx([-0.4, -1.0, 1.0])


@pytest.mark.skipif(_torch_spec is None, reason="torch is not installed")
def test_on_sample_reanalyze_matches_snapshot_bootstrap_values(tmp_path: Path) -> None:
    store = TrajectoryReplayStore.from_episodes(8, (make_episode(),))
    state = create_train_state(TrainingConfig(batch_size=2, seed=9))
    checkpoint = save_checkpoint(state, tmp_path / "checkpoint.pt")
    config = ReanalyzeConfig(batch_size=2, bootstrap_td_steps=1, gamma=0.5)

    snapshot = build_reanalyze_snapshot_from_store(
        store,
        checkpoint_path=checkpoint,
        config=config,
    )
    dataset = OnSampleReanalyzeDataset(
        store,
        checkpoint_path=checkpoint,
        config=config,
    )
    batch = dataset.sample_arrays(len(store), random.Random(3))

    assert sorted(batch.indexes.tolist()) == [0, 1, 2]
    sampled_rows = zip(batch.indexes.tolist(), batch.values.tolist(), strict=True)
    for replay_index, sampled_value in sampled_rows:
        assert sampled_value == pytest.approx(float(snapshot.values[replay_index]))


@pytest.mark.skipif(_torch_spec is None, reason="torch is not installed")
def test_on_sample_reanalyze_refreshes_policy_targets_by_batch_ratio(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from great_kingdom_ai.search_reanalyze import SearchReanalyzeResult

    store = TrajectoryReplayStore.from_episodes(8, (make_episode(),))
    state = create_train_state(TrainingConfig(batch_size=2, seed=9))
    checkpoint = save_checkpoint(state, tmp_path / "checkpoint.pt")

    def fake_refresh_sampled_policies_with_search(**kwargs: object) -> SearchReanalyzeResult:
        policies = np.asarray(kwargs["policies"], dtype=np.float32)
        refreshed = np.stack([make_policy(PASS_ACTION) for _ in range(policies.shape[0])], axis=0)
        return SearchReanalyzeResult(
            policies=refreshed,
            search_reanalyzed=np.ones((policies.shape[0],), dtype=np.bool_),
            selected_indexes=tuple(range(policies.shape[0])),
        )

    monkeypatch.setattr(
        on_sample_reanalyze_module,
        "refresh_sampled_policies_with_search",
        fake_refresh_sampled_policies_with_search,
    )
    dataset = OnSampleReanalyzeDataset(
        store,
        checkpoint_path=checkpoint,
        config=ReanalyzeConfig(batch_size=2, policy_reanalyze_ratio=0.5),
    )
    batch = dataset.sample_arrays(len(store), random.Random(3))

    assert int(batch.search_reanalyzed.sum()) == 2
    stats = dataset.target_stats()
    assert stats.sampled_batches == 1
    assert stats.sampled_rows == 3
    assert stats.policy_reanalyzed == 2
    assert sum(stats.bootstrap_source_counts.values()) == 3
    for position, was_refreshed in enumerate(batch.search_reanalyzed.tolist()):
        if was_refreshed:
            assert batch.policies[position].tolist() == pytest.approx(
                make_policy(PASS_ACTION).tolist()
            )


@pytest.mark.skipif(_torch_spec is None, reason="torch is not installed")
def test_on_sample_reanalyze_can_bootstrap_from_mcts_root_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from great_kingdom_ai.search_reanalyze import SearchReanalyzeResult

    store = TrajectoryReplayStore.from_episodes(8, (make_episode(),))
    state = create_train_state(TrainingConfig(batch_size=2, seed=9))
    checkpoint = save_checkpoint(state, tmp_path / "checkpoint.pt")

    def fake_refresh_sampled_policies_with_search(**kwargs: object) -> SearchReanalyzeResult:
        policies = np.asarray(kwargs["policies"], dtype=np.float32)
        row_count = policies.shape[0]
        return SearchReanalyzeResult(
            policies=policies.copy(),
            search_reanalyzed=np.ones((row_count,), dtype=np.bool_),
            selected_indexes=tuple(range(row_count)),
            root_values=np.full((row_count,), 0.75, dtype=np.float32),
        )

    monkeypatch.setattr(
        on_sample_reanalyze_module,
        "refresh_sampled_policies_with_search",
        fake_refresh_sampled_policies_with_search,
    )
    dataset = OnSampleReanalyzeDataset(
        store,
        checkpoint_path=checkpoint,
        config=ReanalyzeConfig(
            batch_size=2,
            bootstrap_td_steps=1,
            value_bootstrap_source="mcts_root",
        ),
    )

    values = dataset._sampled_bootstrap_targets([0, 1, 2])

    assert values.tolist() == pytest.approx([-0.75, -1.0, 1.0])


def test_reanalyze_target_snapshot_priority_sampling_uses_importance_weights() -> None:
    features = np.stack([make_features(1), make_features(PASS_ACTION)], axis=0)
    policies = np.stack([make_policy(1), make_policy(PASS_ACTION)], axis=0)
    snapshot = ReanalyzeTargetSnapshot(
        features=features,
        policies=policies,
        values=np.asarray([1.0, 0.0], dtype=np.float32),
        refreshed_values=np.asarray([1.0, -1.0], dtype=np.float32),
        sample_weights=np.asarray([1.0, 2.0], dtype=np.float32),
        episode_ids=np.asarray([7, 7], dtype=np.int64),
        timesteps=np.asarray([0, 1], dtype=np.int64),
        players=np.asarray([1, 2], dtype=np.int64),
        source_model_versions=np.asarray([3, 3], dtype=np.int64),
        created_iterations=np.asarray([2, 2], dtype=np.int64),
        target_ages=np.asarray([0, 5], dtype=np.int64),
        model_version=8,
        bootstrap_td_steps=0,
        gamma=1.0,
        policy_logits=np.stack([make_policy(1), make_policy(1)], axis=0),
    )

    scores = snapshot.priority_scores(
        PrioritySamplingConfig(
            enabled=True,
            alpha=1.0,
            beta=0.4,
            value_error_weight=1.0,
            policy_kl_weight=0.0,
            target_age_weight=1.0,
        )
    )
    batch = snapshot.sample_arrays(
        1,
        random.Random(0),
        priority_config=PrioritySamplingConfig(enabled=True, alpha=1.0, beta=0.4),
    )

    assert scores[1] > scores[0]
    assert batch.sample_weights.shape == (1,)
    assert 0.0 < batch.sample_weights[0] <= 2.0


def test_reanalyze_target_snapshot_priority_boosts_search_reanalyzed_rows() -> None:
    features = np.stack(
        [make_features(1), make_features(2), make_features(PASS_ACTION)],
        axis=0,
    )
    policies = np.stack([make_policy(1), make_policy(2), make_policy(PASS_ACTION)], axis=0)
    snapshot = ReanalyzeTargetSnapshot(
        features=features,
        policies=policies,
        values=np.zeros((3,), dtype=np.float32),
        refreshed_values=np.zeros((3,), dtype=np.float32),
        sample_weights=np.ones((3,), dtype=np.float32),
        episode_ids=np.asarray([7, 7, 7], dtype=np.int64),
        timesteps=np.asarray([0, 1, 2], dtype=np.int64),
        players=np.asarray([1, 2, 1], dtype=np.int64),
        source_model_versions=np.asarray([3, 3, 3], dtype=np.int64),
        created_iterations=np.asarray([2, 2, 2], dtype=np.int64),
        target_ages=np.zeros((3,), dtype=np.int64),
        model_version=8,
        bootstrap_td_steps=0,
        gamma=1.0,
        policy_logits=policies,
        search_reanalyzed=np.asarray([False, True, False], dtype=np.bool_),
    )

    scores = snapshot.priority_scores(
        PrioritySamplingConfig(
            enabled=True,
            alpha=1.0,
            beta=0.4,
            value_error_weight=0.0,
            policy_kl_weight=0.0,
            target_age_weight=0.0,
            search_reanalyzed_boost=2.0,
        )
    )

    assert scores.tolist() == pytest.approx([1.0, 2.0, 1.0])


def test_reanalyze_target_snapshot_caches_priority_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features = np.stack(
        [make_features(1), make_features(2), make_features(PASS_ACTION)],
        axis=0,
    )
    policies = np.stack([make_policy(1), make_policy(2), make_policy(PASS_ACTION)], axis=0)
    snapshot = ReanalyzeTargetSnapshot(
        features=features,
        policies=policies,
        values=np.asarray([1.0, 0.0, -1.0], dtype=np.float32),
        refreshed_values=np.asarray([1.0, -1.0, -1.0], dtype=np.float32),
        sample_weights=np.ones((3,), dtype=np.float32),
        episode_ids=np.asarray([7, 7, 7], dtype=np.int64),
        timesteps=np.asarray([0, 1, 2], dtype=np.int64),
        players=np.asarray([1, 2, 1], dtype=np.int64),
        source_model_versions=np.asarray([3, 3, 3], dtype=np.int64),
        created_iterations=np.asarray([2, 2, 2], dtype=np.int64),
        target_ages=np.asarray([0, 5, 1], dtype=np.int64),
        model_version=8,
        bootstrap_td_steps=0,
        gamma=1.0,
        policy_logits=policies,
    )
    calls = 0

    def fake_priority_scores(**kwargs: object) -> np.ndarray:
        nonlocal calls
        calls += 1
        del kwargs
        return np.asarray([1.0, 4.0, 2.0], dtype=np.float32)

    monkeypatch.setattr(reanalyze_module, "priority_scores", fake_priority_scores)
    config = PrioritySamplingConfig(enabled=True, alpha=1.0, beta=0.4)

    snapshot.sample_arrays(1, random.Random(0), priority_config=config)
    snapshot.sample_arrays(1, random.Random(1), priority_config=config)
    scores = snapshot.priority_scores(config)

    assert calls == 1
    assert scores.tolist() == pytest.approx([1.0, 4.0, 2.0])


@pytest.mark.skipif(_torch_spec is None, reason="torch is not installed")
def test_build_reanalyze_snapshot_refreshes_values_and_bootstrap_targets(
    tmp_path: Path,
) -> None:
    replay = TrajectoryReplayStore.from_episodes(8, (make_episode(),))
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
    store_snapshot = build_reanalyze_snapshot_from_store(
        replay,
        checkpoint_path=checkpoint,
        config=ReanalyzeConfig(batch_size=2, bootstrap_td_steps=1, gamma=1.0),
    )

    assert snapshot.model_version == 9
    assert snapshot.target_ages.tolist() == [5, 5, 5]
    assert snapshot.policy_logits is not None
    assert snapshot.policy_logits.shape == (3, ACTION_SPACE)
    assert snapshot.refreshed_values.tolist() == pytest.approx([0.0, 0.0, 0.0])
    assert snapshot.values.tolist() == pytest.approx([-0.0, -1.0, 1.0])
    assert store_snapshot.values.tolist() == pytest.approx(snapshot.values.tolist())
    assert store_snapshot.episode_ids.tolist() == snapshot.episode_ids.tolist()


@pytest.mark.skipif(_torch_spec is None, reason="torch is not installed")
def test_build_reanalyze_snapshot_can_refresh_policy_targets_with_search(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import great_kingdom_ai.search_reanalyze as search_reanalyze

    class FakeResult:
        def policy_target(self) -> list[float]:
            return make_policy(PASS_ACTION).tolist()

        def root_value(self) -> float:
            return 0.25

    class FakeGameState:
        def __init__(self) -> None:
            self.actions: list[int] = []

        def current_player(self) -> int:
            return 1 if len(self.actions) % 2 == 0 else 2

        def is_terminal(self) -> bool:
            return False

        def apply_action(self, action: int) -> None:
            self.actions.append(action)

        def feature_planes(self) -> list[float]:
            action = [1, 2, PASS_ACTION][len(self.actions)]
            return make_features(action).reshape(-1).tolist()

    class FakeRequest:
        def __init__(self, states: list[FakeGameState]) -> None:
            self.states = states

        def feature_planes(self) -> list[list[float]]:
            return [state.feature_planes() for state in self.states]

    class FakeBatch:
        def __init__(self, game_count: int, **kwargs: object) -> None:
            self.states = [FakeGameState() for _ in range(game_count)]
            self.kwargs = kwargs

        def apply_actions(self, actions: list[int | None]) -> None:
            for state, action in zip(self.states, actions, strict=True):
                if action is not None:
                    state.apply_action(action)

        def active_game_indexes(self) -> list[int]:
            return list(range(len(self.states)))

        def active_eval_request(self) -> FakeRequest:
            return FakeRequest(self.states)

        def search_active_with_logits_and_evaluator(
            self,
            policy_logits: list[list[float]],
            evaluator: object,
            root_values: list[float],
            leaf_batch_size: int,
        ) -> list[FakeResult]:
            del evaluator
            assert len(policy_logits) == len(self.states)
            assert len(root_values) == len(self.states)
            assert all(len(row) == ACTION_SPACE for row in policy_logits)
            assert all(-1.0 <= value <= 1.0 for value in root_values)
            assert leaf_batch_size == 3
            return [FakeResult() for _ in self.states]

    class FakeCore:
        GameState = FakeGameState
        GumbelSelfPlayBatch = FakeBatch

    monkeypatch.setattr(search_reanalyze, "_import_core", lambda: FakeCore)
    replay = TrajectoryReplayStore.from_episodes(8, (make_episode(),))
    state = create_train_state(TrainingConfig(batch_size=2, seed=9))
    checkpoint = save_checkpoint(state, tmp_path / "checkpoint.pt")

    snapshot = build_reanalyze_snapshot(
        replay,
        checkpoint_path=checkpoint,
        config=ReanalyzeConfig(
            batch_size=2,
            search=SearchReanalyzeConfig(fraction=1.0, simulations=4, leaf_batch_size=3),
        ),
    )

    assert snapshot.search_reanalyzed is not None
    assert snapshot.search_reanalyzed.tolist() == [True, True, True]
    assert snapshot.policies[0].tolist() == pytest.approx(make_policy(PASS_ACTION).tolist())


def test_refresh_policies_with_search_can_use_onnx_evaluator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import great_kingdom_ai.search_reanalyze as search_reanalyze

    class FakeResult:
        def policy_target(self) -> list[float]:
            return make_policy(PASS_ACTION).tolist()

        def root_value(self) -> float:
            return 0.25

    class FakeGameState:
        def __init__(self) -> None:
            self.actions: list[int] = []

        def current_player(self) -> int:
            return 1 if len(self.actions) % 2 == 0 else 2

        def is_terminal(self) -> bool:
            return False

        def apply_action(self, action: int) -> None:
            self.actions.append(action)

        def feature_planes(self) -> list[float]:
            action = [1, 2, PASS_ACTION][len(self.actions)]
            return make_features(action).reshape(-1).tolist()

    class FakeRequest:
        def __init__(self, states: list[FakeGameState]) -> None:
            self.states = states

        def feature_planes(self) -> list[list[float]]:
            return [state.feature_planes() for state in self.states]

    class FakeBatch:
        def __init__(self, game_count: int, **kwargs: object) -> None:
            self.states = [FakeGameState() for _ in range(game_count)]
            self.kwargs = kwargs

        def apply_actions(self, actions: list[int | None]) -> None:
            for state, action in zip(self.states, actions, strict=True):
                if action is not None:
                    state.apply_action(action)

        def active_game_indexes(self) -> list[int]:
            return list(range(len(self.states)))

        def active_eval_request(self) -> FakeRequest:
            return FakeRequest(self.states)

        def search_active_with_onnx_evaluator(
            self,
            evaluator: object,
            *,
            leaf_batch_size: int,
        ) -> list[FakeResult]:
            assert evaluator == "onnx-evaluator"
            assert leaf_batch_size == 5
            return [FakeResult() for _ in self.states]

    class FakeCore:
        GameState = FakeGameState
        GumbelSelfPlayBatch = FakeBatch

    monkeypatch.setattr(search_reanalyze, "_import_core", lambda: FakeCore)
    episode = make_episode()
    policies = np.stack([transition.policy_target for transition in episode.transitions], axis=0)

    result = search_reanalyze.refresh_policies_with_search(
        episodes=(episode,),
        policies=policies,
        values=np.zeros((3,), dtype=np.float32),
        policy_logits=np.zeros_like(policies),
        refreshed_values=np.zeros((3,), dtype=np.float32),
        target_ages=np.zeros((3,), dtype=np.int64),
        model=None,
        device="cpu",
        onnx_evaluator="onnx-evaluator",
        config=SearchReanalyzeConfig(fraction=1.0, leaf_batch_size=5),
    )

    assert result.search_reanalyzed.tolist() == [True, True, True]
    assert result.policies[0].tolist() == pytest.approx(make_policy(PASS_ACTION).tolist())
    assert result.root_values is not None
    assert result.root_values.tolist() == pytest.approx([0.25, 0.25, 0.25])

    sampled_result = search_reanalyze.refresh_sampled_policies_with_search(
        transitions=((episode, 0), (episode, 2)),
        policies=policies[[0, 2]],
        policy_logits=np.zeros_like(policies[[0, 2]]),
        refreshed_values=np.zeros((2,), dtype=np.float32),
        model=None,
        device="cpu",
        onnx_evaluator="onnx-evaluator",
        config=SearchReanalyzeConfig(fraction=1.0, leaf_batch_size=5),
    )

    assert sampled_result.search_reanalyzed.tolist() == [True, True]
    assert sampled_result.policies[1].tolist() == pytest.approx(
        make_policy(PASS_ACTION).tolist()
    )
    assert sampled_result.root_values is not None
    assert sampled_result.root_values.tolist() == pytest.approx([0.25, 0.25])


def test_search_reanalyze_clips_finite_out_of_range_root_value() -> None:
    import great_kingdom_ai.search_reanalyze as search_reanalyze

    class FakeResult:
        def root_value(self) -> float:
            return 1.5

    assert search_reanalyze._root_value_from_result(FakeResult()) == pytest.approx(1.0)


def test_search_reanalyze_treats_nonfinite_root_value_as_unavailable() -> None:
    import great_kingdom_ai.search_reanalyze as search_reanalyze

    class FakeResult:
        def root_value(self) -> float:
            return float("nan")

    assert search_reanalyze._root_value_from_result(FakeResult()) is None


def test_search_reanalyze_reconstruction_validation_is_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import great_kingdom_ai.search_reanalyze as search_reanalyze

    calls = 0

    def fake_validate(batch: object, refs: object) -> None:
        nonlocal calls
        del batch, refs
        calls += 1

    monkeypatch.setattr(search_reanalyze, "_validate_reconstructed_batch", fake_validate)
    monkeypatch.delenv("GKA_SEARCH_REANALYZE_VALIDATE", raising=False)
    assert not search_reanalyze._validate_reconstructed_batches_enabled()

    monkeypatch.setenv("GKA_SEARCH_REANALYZE_VALIDATE", "1")
    assert search_reanalyze._validate_reconstructed_batches_enabled()
    search_reanalyze._validate_reconstructed_batch(object(), object())

    assert calls == 1


def test_reconstruct_batch_uses_rust_action_history_fast_path() -> None:
    import great_kingdom_ai.search_reanalyze as search_reanalyze

    class FakeBatch:
        captured_histories: list[list[int]] | None = None

        def __init__(self, game_count: int, **kwargs: object) -> None:
            raise AssertionError("slow constructor should not be used")

        @staticmethod
        def from_action_histories(
            action_histories: list[list[int]],
            **kwargs: object,
        ) -> str:
            FakeBatch.captured_histories = action_histories
            assert kwargs["simulations"] == 7
            assert kwargs["max_considered_actions"] == 3
            return "fast-batch"

    class FakeCore:
        GumbelSelfPlayBatch = FakeBatch

    episode = make_episode()
    refs = [
        search_reanalyze._TransitionRef(episode, 0, 10),
        search_reanalyze._TransitionRef(episode, 2, 12),
    ]

    batch = search_reanalyze._reconstruct_batch(
        FakeCore,
        refs,
        config=SearchReanalyzeConfig(
            simulations=7,
            max_considered_actions=3,
            policy_target_c_visit=5.0,
            policy_target_c_scale=0.25,
        ),
    )

    assert batch == "fast-batch"
    assert FakeBatch.captured_histories == [[], [1, 2]]


def test_select_search_reanalyze_indexes_uses_fraction_budget_and_priority() -> None:
    episode = make_episode()
    policies = np.stack([transition.policy_target for transition in episode.transitions], axis=0)
    logits = np.zeros_like(policies)
    logits[2, 0] = 10.0

    indexes = select_search_reanalyze_indexes(
        episodes=(episode,),
        policies=policies,
        values=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        policy_logits=logits,
        refreshed_values=np.asarray([0.0, 0.0, -1.0], dtype=np.float32),
        target_ages=np.asarray([0, 1, 9], dtype=np.int64),
        config=SearchReanalyzeConfig(
            fraction=1.0,
            budget=1,
            opening_weight=0.0,
            value_error_weight=1.0,
            policy_kl_weight=1.0,
            target_age_weight=1.0,
        ),
    )

    assert indexes == (2,)


def test_reanalyze_parser_exposes_phase7_cli_options() -> None:
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
            "--onnx-model",
            "model.onnx",
            "--onnx-device",
            "cuda",
            "--onnx-max-batch-size",
            "32",
            "--bootstrap-td-steps",
            "4",
            "--dynamic-horizon-enabled",
            "--dynamic-horizon-tau",
            "0.25",
            "--dynamic-horizon-total-steps",
            "120",
            "--value-bootstrap-source",
            "mcts_root",
            "--search-reanalyze-fraction",
            "0.25",
            "--search-reanalyze-budget",
            "7",
            "--search-reanalyze-simulations",
            "8",
            "--search-reanalyze-leaf-batch-size",
            "3",
            "--search-reanalyze-root-batch-size",
            "5",
        ]
    )

    assert args.batch_size == 16
    assert args.onnx_model == Path("model.onnx")
    assert args.onnx_device == "cuda"
    assert args.onnx_max_batch_size == 32
    assert args.bootstrap_td_steps == 4
    assert args.dynamic_horizon_enabled is True
    assert args.dynamic_horizon_tau == pytest.approx(0.25)
    assert args.dynamic_horizon_total_steps == 120
    assert args.value_bootstrap_source == "mcts_root"
    assert args.search_reanalyze_fraction == pytest.approx(0.25)
    assert args.search_reanalyze_budget == 7
    assert args.search_reanalyze_simulations == 8
    assert args.search_reanalyze_leaf_batch_size == 3
    assert args.search_reanalyze_root_batch_size == 5


def test_evaluate_policy_logits_values_with_onnx_uses_core_evaluator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[bytes] = []

    class FakeEvalRequest:
        def __init__(self, row_count: int, payload: bytes) -> None:
            self.row_count = row_count
            self.payload = payload

        @staticmethod
        def from_feature_plane_bytes(row_count: int, payload: bytes) -> FakeEvalRequest:
            requests.append(payload)
            return FakeEvalRequest(row_count, payload)

    class FakeOnnxEvaluator:
        def __init__(self, path: str, *, device: str, max_batch_size: int) -> None:
            assert path == "model.onnx"
            assert device == "cuda"
            assert max_batch_size == 4

        def evaluate(self, request: FakeEvalRequest) -> tuple[list[list[float]], list[float]]:
            logits = np.zeros((request.row_count, ACTION_SPACE), dtype=np.float32)
            logits[:, PASS_ACTION] = 1.0
            values = np.arange(request.row_count, dtype=np.float32)
            return logits.tolist(), values.tolist()

    class FakeCore:
        EvalRequest = FakeEvalRequest
        OnnxEvaluator = FakeOnnxEvaluator

    monkeypatch.setattr(reanalyze_module, "_import_core", lambda: FakeCore)
    features = np.stack(
        [make_features(1), make_features(2), make_features(PASS_ACTION)],
        axis=0,
    )
    evaluator = reanalyze_module._create_onnx_evaluator(
        "model.onnx",
        device="cuda",
        max_batch_size=4,
    )

    logits, values = reanalyze_module._evaluate_policy_logits_values_with_onnx(
        evaluator,
        features,
        batch_size=2,
        device="cuda",
    )

    assert len(requests) == 2
    assert logits.shape == (3, ACTION_SPACE)
    assert logits[:, PASS_ACTION].tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert values.tolist() == pytest.approx([0.0, 1.0, 0.0])
