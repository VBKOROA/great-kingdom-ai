from dataclasses import dataclass
from pathlib import Path

import great_kingdom_ai.replay.trajectory as trajectory_module
import numpy as np
import pytest
from great_kingdom_ai.features import ACTION_SPACE, BOARD_SIZE, FEATURE_CHANNELS, PASS_ACTION
from great_kingdom_ai.replay import (
    TrajectoryEpisode,
    TrajectoryReplayStore,
    TrajectoryTransition,
    legal_mask_from_features,
    trajectory_episode_from_self_play_result,
)
from great_kingdom_ai.replay.sample import ReplaySample
from great_kingdom_ai.trajectory_targets import (
    BootstrapValueTargetConfig,
    bootstrap_value_target_for_transition,
)


@dataclass(frozen=True)
class FakeMove:
    turn: int
    player: int
    action: int


@dataclass(frozen=True)
class FakeLog:
    seed: int
    moves: list[FakeMove]
    winner: int
    end_reason: int
    territory_scores: tuple[int, int]


def make_features(action: int) -> np.ndarray:
    features = np.zeros((FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    if action != PASS_ACTION:
        features[4, action // BOARD_SIZE, action % BOARD_SIZE] = 1.0
    return features


def make_policy(action: int) -> np.ndarray:
    policy = np.zeros(ACTION_SPACE, dtype=np.float32)
    policy[action] = 1.0
    return policy


def make_transition(
    *,
    episode_id: int = 0,
    timestep: int = 0,
    player: int = 1,
    action: int = PASS_ACTION,
    winner: int = 1,
    terminal: bool = True,
    with_metadata: bool = False,
    root_value: float | None = None,
) -> TrajectoryTransition:
    features = make_features(action)
    root_policy_logits = np.linspace(-1.0, 1.0, ACTION_SPACE, dtype=np.float32)
    next_features = make_features(PASS_ACTION)
    return TrajectoryTransition(
        episode_id=episode_id,
        timestep=timestep,
        player=player,
        features=features,
        legal_mask=legal_mask_from_features(features),
        action=action,
        policy_target=make_policy(action),
        root_policy_logits=root_policy_logits if with_metadata else None,
        root_value=root_value if root_value is not None else (0.25 if with_metadata else None),
        next_features=next_features if with_metadata else None,
        winner=winner,
        terminal=terminal,
        model_version=7 if with_metadata else 0,
        search_config_hash="search-v1" if with_metadata else "",
        created_iteration=11 if with_metadata else 0,
        sample_weight=1.5 if with_metadata else 1.0,
    )


def make_episode(
    episode_id: int,
    *,
    actions: list[int],
    winner: int = 1,
    root_values: list[float | None] | None = None,
) -> TrajectoryEpisode:
    values = [None] * len(actions) if root_values is None else root_values
    if len(values) != len(actions):
        raise ValueError("root_values length must match actions length")
    transitions = tuple(
        make_transition(
            episode_id=episode_id,
            timestep=index,
            player=1 if index % 2 == 0 else 2,
            action=action,
            winner=winner,
            terminal=index == len(actions) - 1,
            root_value=values[index],
        )
        for index, action in enumerate(actions)
    )
    return TrajectoryEpisode(
        episode_id=episode_id,
        seed=100 + episode_id,
        transitions=transitions,
        winner=winner,
        end_reason=1,
        territory_scores=(3, 1),
    )


def test_trajectory_replay_store_saves_loads_and_refs_rows(tmp_path: Path) -> None:
    path = tmp_path / "trajectory-replay.npz"
    replay = TrajectoryReplayStore.from_episodes(
        8,
        (
            make_episode(0, actions=[1, 2]),
            make_episode(1, actions=[3, PASS_ACTION], winner=2),
        ),
    )

    replay.save(path)
    loaded = TrajectoryReplayStore.load(path)
    refs = loaded.transition_refs([2, 0, 3])

    assert len(loaded) == 4
    assert loaded.episode_count == 2
    assert [(episode.episode_id, index) for episode, index in refs] == [(1, 0), (0, 0), (1, 1)]


def test_trajectory_replay_save_load_preserves_optional_transition_metadata(
    tmp_path: Path,
) -> None:
    path = tmp_path / "trajectory-replay.npz"
    transition = make_transition(with_metadata=True)
    replay = TrajectoryReplayStore.from_episodes(
        4,
        (
            TrajectoryEpisode(
                episode_id=0,
                seed=10,
                transitions=(transition,),
                winner=1,
                end_reason=1,
                territory_scores=(3, 1),
            ),
        ),
    )

    replay.save(path)
    loaded = TrajectoryReplayStore.load(path)
    loaded_transition = loaded.transitions()[0]

    with np.load(path) as data:
        assert "search_config_hashes" not in data
        assert data["search_config_hash_table"].tolist() == ["search-v1"]
        assert data["search_config_hash_ids"].tolist() == [0]

    assert loaded_transition.root_policy_logits is not None
    assert loaded_transition.root_value == pytest.approx(0.25)
    assert loaded_transition.next_features is not None
    assert loaded_transition.model_version == 7
    assert loaded_transition.search_config_hash == "search-v1"
    assert loaded_transition.created_iteration == 11
    assert loaded_transition.sample_weight == pytest.approx(1.5)


def test_trajectory_replay_payload_loader_reads_arrays_once_per_key() -> None:
    replay = TrajectoryReplayStore.from_episodes(
        8,
        (
            make_episode(0, actions=[1, 2]),
            make_episode(1, actions=[3, PASS_ACTION], winner=2),
        ),
    )
    payload = trajectory_module._episodes_to_payload(replay.capacity, replay.episodes)

    class CountingPayload:
        def __init__(self, arrays: dict[str, np.ndarray]) -> None:
            self.arrays = arrays
            self.counts = {key: 0 for key in arrays}

        def __contains__(self, key: object) -> bool:
            return key in self.arrays

        def __getitem__(self, key: str) -> np.ndarray:
            self.counts[key] += 1
            return self.arrays[key]

    data = CountingPayload(payload)
    episodes = trajectory_module._episodes_from_payload(data)

    assert sum(len(episode.transitions) for episode in episodes) == 4
    assert data.counts["policy_targets"] == 2
    assert data.counts["legal_masks"] == 2
    assert data.counts["timesteps"] == 2


def test_trajectory_replay_store_extends_evicts_and_preserves_schema(
    tmp_path: Path,
) -> None:
    path = tmp_path / "trajectory-replay.npz"
    store = TrajectoryReplayStore.empty(capacity=3)

    store.extend_episodes([make_episode(0, actions=[1, 2])])
    store.extend_episodes([make_episode(1, actions=[3, PASS_ACTION], winner=2)])
    store.save(path, compressed=False)
    loaded_store = TrajectoryReplayStore.load(path)

    assert len(loaded_store) == 2
    assert loaded_store.episode_ids.tolist() == [1]
    assert loaded_store.features.shape == (2, FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    assert loaded_store.episode_count == 1
    assert [episode.episode_id for episode in loaded_store.episodes] == [1]


def test_trajectory_replay_capacity_evicts_whole_old_episodes() -> None:
    replay = TrajectoryReplayStore.empty(capacity=3)
    replay.extend_episodes([make_episode(0, actions=[1, 2])])
    replay.extend_episodes([make_episode(1, actions=[3, PASS_ACTION])])

    assert len(replay) == 2
    assert [episode.episode_id for episode in replay.episodes] == [1]


def test_trajectory_replay_extends_from_store_without_episode_roundtrip() -> None:
    replay = TrajectoryReplayStore.from_episodes(
        8,
        (make_episode(0, actions=[1, 2]),),
    )
    replay.update_sampling_priorities(
        np.asarray([0, 1], dtype=np.int64),
        np.asarray([4.0, 7.0], dtype=np.float32),
        ema=0.0,
        epsilon=1e-6,
        max_priority=None,
    )
    incoming = TrajectoryReplayStore.from_episodes(
        8,
        (
            TrajectoryEpisode(
                episode_id=1,
                seed=101,
                transitions=(
                    make_transition(
                        episode_id=1,
                        timestep=0,
                        action=3,
                        with_metadata=True,
                    ),
                ),
                winner=1,
                end_reason=1,
                territory_scores=(3, 1),
            ),
        ),
    )

    replay.extend_store(incoming)

    assert len(replay) == 3
    assert replay.episode_ids.tolist() == [0, 1]
    assert replay.sampling_priorities.tolist() == pytest.approx([4.0, 7.0, 7.0])
    assert replay.root_policy_logits is not None
    assert replay.root_policy_logits_present is not None
    assert replay.root_policy_logits_present.tolist() == [False, False, True]
    assert replay.search_config_hash_table.tolist() == ["", "search-v1"]


def test_bootstrap_value_target_uses_future_root_value_from_sample_player_perspective() -> None:
    episode = make_episode(
        0,
        actions=[1, 2, 3, PASS_ACTION],
        root_values=[None, None, 0.4, None],
    )
    config = BootstrapValueTargetConfig(bootstrap_td_steps=2, gamma=0.5)

    target = bootstrap_value_target_for_transition(episode, 0, config)

    assert target == pytest.approx(0.1)


def test_bootstrap_value_target_flips_future_opponent_perspective() -> None:
    episode = make_episode(
        0,
        actions=[1, 2, 3, PASS_ACTION],
        root_values=[None, 0.25, None, None],
    )
    config = BootstrapValueTargetConfig(bootstrap_td_steps=1)

    target = bootstrap_value_target_for_transition(episode, 0, config)

    assert target == pytest.approx(-0.25)


def test_bootstrap_value_target_uses_terminal_result_within_td_window() -> None:
    episode = make_episode(
        0,
        actions=[1, 2, PASS_ACTION],
        winner=2,
        root_values=[None, 1.0, None],
    )
    config = BootstrapValueTargetConfig(bootstrap_td_steps=4)

    target = bootstrap_value_target_for_transition(episode, 0, config)

    assert target == pytest.approx(-1.0)


def test_bootstrap_value_target_requires_future_root_value() -> None:
    episode = make_episode(0, actions=[1, 2, 3, PASS_ACTION])
    config = BootstrapValueTargetConfig(bootstrap_td_steps=2)

    with pytest.raises(ValueError, match="root_value"):
        bootstrap_value_target_for_transition(episode, 0, config)


def test_build_trajectory_episode_from_self_play_result() -> None:
    root_logits = np.linspace(-1.0, 1.0, ACTION_SPACE, dtype=np.float32)
    samples = [
        ReplaySample(
            features=make_features(7),
            policy=make_policy(7),
            value=1.0,
            root_policy_logits=root_logits,
            sample_weight=2.0,
        ),
        ReplaySample(
            features=make_features(PASS_ACTION),
            policy=make_policy(PASS_ACTION),
            value=-1.0,
        ),
    ]
    log = FakeLog(
        seed=42,
        moves=[
            FakeMove(turn=0, player=1, action=7),
            FakeMove(turn=1, player=2, action=PASS_ACTION),
        ],
        winner=1,
        end_reason=2,
        territory_scores=(5, 3),
    )

    episode = trajectory_episode_from_self_play_result(
        log,
        samples,
        episode_id=9,
        root_values=[0.6, -0.2],
        model_version=3,
        search_config_hash="abc",
        created_iteration=4,
    )

    assert episode.episode_id == 9
    assert episode.transitions[0].next_features is not None
    assert episode.transitions[0].root_policy_logits is not None
    assert episode.transitions[0].root_value == pytest.approx(0.6)
    assert episode.transitions[0].sample_weight == pytest.approx(2.0)
    assert episode.transitions[0].model_version == 3
    assert episode.transitions[0].search_config_hash == "abc"
    assert episode.transitions[0].created_iteration == 4
    assert episode.transitions[-1].terminal is True


def test_build_trajectory_episode_requires_unfiltered_self_play_samples() -> None:
    log = FakeLog(
        seed=42,
        moves=[
            FakeMove(turn=0, player=1, action=7),
            FakeMove(turn=1, player=2, action=PASS_ACTION),
        ],
        winner=1,
        end_reason=2,
        territory_scores=(5, 3),
    )
    samples = [ReplaySample(features=make_features(7), policy=make_policy(7), value=1.0)]

    with pytest.raises(ValueError, match="same length"):
        trajectory_episode_from_self_play_result(log, samples, episode_id=0)
