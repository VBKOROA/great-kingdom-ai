from __future__ import annotations

import random

import numpy as np
import pytest
from great_kingdom_ai.features import ACTION_SPACE, BOARD_SIZE, FEATURE_CHANNELS, PASS_ACTION
from great_kingdom_ai.priority_sampling import PrioritySamplingConfig
from great_kingdom_ai.trajectory_dataset import TrajectoryReplayDataset
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


def test_trajectory_replay_dataset_supports_priority_sampling() -> None:
    store = TrajectoryReplayStore.from_episodes(
        8,
        (make_episode(0, winner=1), make_episode(1, winner=2)),
    )
    store.sample_weights[:] = np.asarray([1.0, 1.0, 8.0, 8.0], dtype=np.float32)
    dataset = TrajectoryReplayDataset(store)

    batch = dataset.sample_arrays(
        2,
        random.Random(5),
        priority_config=PrioritySamplingConfig(enabled=True, alpha=1.0, beta=0.4),
    )

    assert batch.indexes.shape == (2,)
    assert batch.sample_weights.shape == (2,)
    assert np.all(batch.sample_weights > 0.0)
