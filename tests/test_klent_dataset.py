from __future__ import annotations

import json
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from great_kingdom_ai.features import ACTION_SPACE, BOARD_SIZE, FEATURE_CHANNELS, PASS_ACTION
from great_kingdom_ai.klent.dataset import KlentReplayDataset
from great_kingdom_ai.klent.shards import (
    KLENT_ALGORITHM,
    KlentShardMetadata,
    load_klent_shard,
    save_klent_shard,
    shard_metadata_path,
)
from great_kingdom_ai.klent.types import KlentConfig
from great_kingdom_ai.replay import (
    TrajectoryEpisode,
    TrajectoryReplayStore,
    TrajectoryTransition,
    legal_mask_from_features,
)


def make_features(action: int) -> np.ndarray:
    features = np.zeros((FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    if action != PASS_ACTION:
        features[4, action // BOARD_SIZE, action % BOARD_SIZE] = 1.0
    return features


def make_policy(action: int) -> np.ndarray:
    policy = np.zeros((ACTION_SPACE,), dtype=np.float32)
    policy[action] = 1.0
    return policy


def make_klent_transition(
    *,
    episode_id: int = 0,
    timestep: int = 0,
    player: int = 1,
    action: int = 1,
    winner: int = 1,
    terminal: bool = False,
    lambda_return: float | None = 0.5,
    policy_target: np.ndarray | None = None,
) -> TrajectoryTransition:
    features = make_features(action)
    return TrajectoryTransition(
        episode_id=episode_id,
        timestep=timestep,
        player=player,
        features=features,
        legal_mask=legal_mask_from_features(features),
        action=action,
        policy_target=make_policy(action) if policy_target is None else policy_target,
        winner=winner,
        terminal=terminal,
        model_version=3,
        created_iteration=1,
        lambda_return=lambda_return,
    )


def make_klent_episode(
    episode_id: int = 0,
    *,
    winner: int = 1,
    lambda_returns: tuple[float, float] = (0.5, 1.0),
) -> TrajectoryEpisode:
    return TrajectoryEpisode(
        episode_id=episode_id,
        seed=episode_id,
        transitions=(
            make_klent_transition(
                episode_id=episode_id,
                timestep=0,
                player=1,
                action=1,
                winner=winner,
                lambda_return=lambda_returns[0],
            ),
            make_klent_transition(
                episode_id=episode_id,
                timestep=1,
                player=2,
                action=PASS_ACTION,
                winner=winner,
                terminal=True,
                lambda_return=lambda_returns[1],
            ),
        ),
        winner=winner,
        end_reason=1,
        territory_scores=(1, 0),
    )


def make_store(*episodes: TrajectoryEpisode) -> TrajectoryReplayStore:
    return TrajectoryReplayStore.from_episodes(16, episodes)


def make_metadata(store: TrajectoryReplayStore, config: KlentConfig) -> KlentShardMetadata:
    return KlentShardMetadata.from_config(
        config,
        iteration=0,
        model_version=3,
        transitions=len(store),
        games=1,
    )


def test_klent_dataset_uses_stored_lambda_returns_and_actions() -> None:
    config = KlentConfig()
    store = make_store(make_klent_episode())
    dataset = KlentReplayDataset(store, config=config, metadata=make_metadata(store, config))

    batch = dataset.sample_arrays(len(store), random.Random(0))
    samples = dataset.sample(2, random.Random(1))

    assert store.lambda_returns is not None
    assert batch.values.tolist() == pytest.approx(
        store.lambda_returns[batch.indexes].tolist()
    )
    assert set(batch.values.tolist()) == {0.5, 1.0}
    assert batch.actions is not None
    assert sorted(batch.actions.tolist()) == [1, PASS_ACTION]
    assert all(sample.action is not None for sample in samples)
    assert all(sample.value in {0.5, 1.0} for sample in samples)


def test_klent_dataset_rejects_missing_lambda_returns() -> None:
    store = make_store(
        make_klent_episode(),
    )
    payload = store.to_payload()
    del payload["lambda_returns"]
    del payload["lambda_returns_present"]
    legacy = TrajectoryReplayStore.from_payload(payload)

    with pytest.raises(ValueError, match="lambda returns"):
        KlentReplayDataset(legacy, config=KlentConfig())


def test_klent_dataset_rejects_partially_missing_lambda_returns() -> None:
    store = make_store(make_klent_episode())
    assert store.lambda_returns_present is not None
    store.lambda_returns_present = np.asarray([True, False])
    store.validate()

    with pytest.raises(ValueError, match="present for every"):
        KlentReplayDataset(store, config=KlentConfig())


def test_klent_dataset_rejects_illegal_taken_action() -> None:
    store = make_store(make_klent_episode())
    assert store.legal_masks is not None
    store.legal_masks[0, 1] = False

    with pytest.raises(ValueError, match="illegal taken actions"):
        KlentReplayDataset(store, config=KlentConfig())


def test_klent_dataset_rejects_illegal_policy_mass() -> None:
    episode = make_klent_episode()
    policy = np.zeros((ACTION_SPACE,), dtype=np.float32)
    policy[1] = 0.5
    policy[2] = 0.5
    transitions = (
        replace(episode.transitions[0], policy_target=policy),
        episode.transitions[1],
    )
    store = make_store(replace(episode, transitions=transitions))

    with pytest.raises(ValueError, match="illegal actions"):
        KlentReplayDataset(store, config=KlentConfig())


def test_klent_dataset_requires_stored_features() -> None:
    store = make_store(make_klent_episode())
    payload = store.to_payload()
    del payload["features"]
    rebuilt = TrajectoryReplayStore.from_payload(payload)

    with pytest.raises(ValueError, match="stored features"):
        KlentReplayDataset(rebuilt, config=KlentConfig())


def test_klent_dataset_rejects_config_mismatch() -> None:
    store = make_store(make_klent_episode())
    metadata = make_metadata(store, KlentConfig())

    with pytest.raises(ValueError, match="does not match"):
        KlentReplayDataset(store, config=KlentConfig(alpha=0.5), metadata=metadata)


def test_klent_shard_round_trip_preserves_metadata_and_lambda_returns(tmp_path: Path) -> None:
    config = KlentConfig()
    store = make_store(make_klent_episode(0), make_klent_episode(1, winner=2))
    metadata = KlentShardMetadata.from_config(
        config,
        iteration=4,
        model_version=7,
        transitions=len(store),
        games=2,
        created_at="2026-01-01T00:00:00+00:00",
    )

    replay_path = tmp_path / "shard.npz"
    save_klent_shard(store, replay_path, metadata)
    loaded_store, loaded_metadata = load_klent_shard(replay_path)

    assert loaded_metadata == metadata
    assert loaded_store.lambda_returns is not None
    assert loaded_store.lambda_returns.tolist() == pytest.approx([0.5, 1.0, 0.5, 1.0])
    dataset = KlentReplayDataset(
        loaded_store,
        config=config,
        metadata=loaded_metadata,
    )
    assert len(dataset) == 4


def test_klent_shard_rejects_missing_metadata(tmp_path: Path) -> None:
    store = make_store(make_klent_episode())
    replay_path = tmp_path / "shard.npz"
    store.save(replay_path)

    with pytest.raises(ValueError, match="metadata is missing"):
        load_klent_shard(replay_path)


def test_klent_shard_rejects_foreign_algorithm(tmp_path: Path) -> None:
    store = make_store(make_klent_episode())
    replay_path = tmp_path / "shard.npz"
    store.save(replay_path)
    metadata_path = shard_metadata_path(replay_path)
    metadata_path.write_text(
        json.dumps(
            {
                "algorithm": "gumbel",
                "schema_version": 1,
                "iteration": 0,
                "model_version": 0,
                "alpha": 0.03,
                "beta": 0.1,
                "lambda_param": 0.8825,
                "gamma": 1.0,
                "transitions": len(store),
                "games": 1,
                "created_at": "",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=KLENT_ALGORITHM):
        load_klent_shard(replay_path)