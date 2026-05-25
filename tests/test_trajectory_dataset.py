from __future__ import annotations

import random

import great_kingdom_ai.replay.dataset as dataset_module
import numpy as np
import pytest
from great_kingdom_ai.features import ACTION_SPACE, BOARD_SIZE, FEATURE_CHANNELS, PASS_ACTION
from great_kingdom_ai.priority_sampling import PrioritySamplingConfig
from great_kingdom_ai.replay import (
    TrajectoryEpisode,
    TrajectoryReplayDataset,
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


def make_transition(
    *,
    episode_id: int,
    timestep: int,
    player: int,
    action: int,
    winner: int,
) -> TrajectoryTransition:
    features = make_features(action)
    return TrajectoryTransition(
        episode_id=episode_id,
        timestep=timestep,
        player=player,
        features=features,
        legal_mask=legal_mask_from_features(features),
        action=action,
        policy_target=make_policy(action),
        winner=winner,
        terminal=False,
    )


def make_episode(episode_id: int, *, winner: int) -> TrajectoryEpisode:
    transitions = (
        make_transition(
            episode_id=episode_id,
            timestep=0,
            player=1,
            action=1,
            winner=winner,
        ),
        make_transition(
            episode_id=episode_id,
            timestep=1,
            player=2,
            action=PASS_ACTION,
            winner=winner,
        ),
    )
    return TrajectoryEpisode(
        episode_id=episode_id,
        seed=episode_id,
        transitions=transitions,
        winner=winner,
        end_reason=1,
        territory_scores=(1, 0),
    )


def test_trajectory_replay_dataset_samples_terminal_targets_with_recency() -> None:
    store = TrajectoryReplayStore.from_episodes(
        8,
        (make_episode(0, winner=1), make_episode(1, winner=2)),
    )
    dataset = TrajectoryReplayDataset(store)

    batch = dataset.sample_arrays(
        2,
        random.Random(3),
        recent_fraction=1.0,
        recent_window=2,
    )

    assert batch.indexes.tolist() == [2, 3]
    assert batch.features.shape == (2, FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    assert batch.policies.shape == (2, ACTION_SPACE)
    assert batch.values.tolist() == pytest.approx([-1.0, 1.0])
    assert batch.legal_masks.shape == (2, ACTION_SPACE)


def test_trajectory_replay_dataset_bootstraps_from_episode_turn_root_values() -> None:
    transition = make_transition(
        episode_id=0,
        timestep=1,
        player=2,
        action=PASS_ACTION,
        winner=1,
    )
    episode = TrajectoryEpisode(
        episode_id=0,
        seed=0,
        transitions=(transition,),
        winner=1,
        end_reason=1,
        territory_scores=(1, 0),
        turn_players=np.asarray([1, 2, 1, 2], dtype=np.int64),
        turn_actions=np.asarray([1, PASS_ACTION, 2, PASS_ACTION], dtype=np.int64),
        turn_root_values=np.asarray([0.1, 0.2, 0.7, -0.4], dtype=np.float32),
        turn_full_search=np.asarray([False, True, False, False], dtype=np.bool_),
    )
    store = TrajectoryReplayStore.from_episodes(8, (episode,))
    dataset = TrajectoryReplayDataset(
        store,
        bootstrap_td_steps=2,
        value_bootstrap_source="mcts_root",
    )

    batch = dataset.sample_arrays(1, random.Random(0))

    assert batch.indexes.tolist() == [0]
    assert batch.values.tolist() == pytest.approx([-0.4])


def test_trajectory_replay_dataset_reconstructs_missing_features_from_turn_actions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeState:
        def __init__(self) -> None:
            self.actions: list[int] = []

        def is_terminal(self) -> bool:
            return False

        def apply_action(self, action: int) -> None:
            self.actions.append(action)

        def feature_planes(self) -> list[float]:
            row = [0.0] * (FEATURE_CHANNELS * BOARD_SIZE * BOARD_SIZE)
            row[0] = float(len(self.actions))
            return row

        def legal_mask(self) -> list[bool]:
            mask = [False] * ACTION_SPACE
            mask[PASS_ACTION] = True
            return mask

        def current_player(self) -> int:
            return 1 if len(self.actions) % 2 == 0 else 2

    class FakeCore:
        GameState = FakeState

    monkeypatch.setattr(dataset_module, "import_core", lambda _context: FakeCore)
    policy = make_policy(PASS_ACTION)
    transition = TrajectoryTransition(
        episode_id=0,
        timestep=1,
        player=2,
        features=None,
        legal_mask=None,
        action=PASS_ACTION,
        policy_target=policy,
        root_value=0.2,
        winner=1,
        terminal=True,
    )
    episode = TrajectoryEpisode(
        episode_id=0,
        seed=0,
        transitions=(transition,),
        winner=1,
        end_reason=1,
        territory_scores=(1, 0),
        turn_players=np.asarray([1, 2], dtype=np.int64),
        turn_actions=np.asarray([1, PASS_ACTION], dtype=np.int64),
        turn_root_values=np.asarray([0.1, 0.2], dtype=np.float32),
        turn_full_search=np.asarray([False, True], dtype=np.bool_),
    )
    store = TrajectoryReplayStore.from_episodes(8, (episode,))
    assert store.features is None
    dataset = TrajectoryReplayDataset(store)

    batch = dataset.sample_arrays(1, random.Random(0))

    assert batch.features[0, 0, 0, 0] == np.float32(1.0)
    assert batch.legal_masks[0, PASS_ACTION]


def test_trajectory_replay_dataset_uses_terminal_when_bootstrap_exceeds_game() -> None:
    transition = make_transition(
        episode_id=0,
        timestep=2,
        player=1,
        action=PASS_ACTION,
        winner=1,
    )
    episode = TrajectoryEpisode(
        episode_id=0,
        seed=0,
        transitions=(transition,),
        winner=1,
        end_reason=1,
        territory_scores=(1, 0),
        turn_players=np.asarray([1, 2, 1], dtype=np.int64),
        turn_actions=np.asarray([1, 2, PASS_ACTION], dtype=np.int64),
        turn_root_values=np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
        turn_full_search=np.asarray([False, False, True], dtype=np.bool_),
    )
    store = TrajectoryReplayStore.from_episodes(8, (episode,))
    dataset = TrajectoryReplayDataset(
        store,
        bootstrap_td_steps=5,
        value_bootstrap_source="mcts_root",
    )

    batch = dataset.sample_arrays(1, random.Random(0))

    assert batch.values.tolist() == pytest.approx([1.0])


def test_trajectory_replay_dataset_supports_priority_sampling() -> None:
    store = TrajectoryReplayStore.from_episodes(
        8,
        (make_episode(0, winner=1), make_episode(1, winner=2)),
    )
    store.sample_weights[:] = np.asarray([2.0, 2.0, 2.0, 2.0], dtype=np.float32)
    store.sampling_priorities[:] = np.asarray([1.0, 1.0, 8.0, 8.0], dtype=np.float32)
    dataset = TrajectoryReplayDataset(store)

    batch = dataset.sample_arrays(
        2,
        random.Random(5),
        priority_config=PrioritySamplingConfig(enabled=True, alpha=1.0, beta=0.4),
    )

    assert batch.indexes.shape == (2,)
    assert batch.sample_weights.shape == (2,)
    assert np.all(batch.sample_weights > 0.0)
    assert np.all(batch.sample_weights <= 2.0)


def test_trajectory_replay_dataset_updates_sampling_priorities_separately() -> None:
    store = TrajectoryReplayStore.from_episodes(
        8,
        (make_episode(0, winner=1), make_episode(1, winner=2)),
    )
    dataset = TrajectoryReplayDataset(store)

    dataset.update_sampling_priorities(
        np.asarray([1, 3], dtype=np.int64),
        np.asarray([5.0, 9.0], dtype=np.float32),
        ema=0.0,
        epsilon=0.001,
        max_priority=8.0,
    )

    assert store.sample_weights.tolist() == pytest.approx([1.0, 1.0, 1.0, 1.0])
    assert store.sampling_priorities.tolist() == pytest.approx([1.0, 5.0, 1.0, 8.0])
