from __future__ import annotations

import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from great_kingdom_ai.augmentation import (
    ALL_SYMMETRIES,
    augment_sample,
    augment_training_arrays_randomly,
)
from great_kingdom_ai.features import ACTION_SPACE, BOARD_SIZE, FEATURE_CHANNELS, PASS_ACTION
from great_kingdom_ai.replay import (
    TrajectoryEpisode,
    TrajectoryReplayDataset,
    TrajectoryReplayStore,
    TrajectoryTransition,
    legal_mask_from_features,
)
from great_kingdom_ai.replay.sample import ReplaySample
from great_kingdom_ai.replay.terminal_board import (
    TERMINAL_BOARD_ENCODING,
    absolute_terminal_boards_to_perspective,
    validate_absolute_terminal_board,
)


def make_board() -> np.ndarray:
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.uint8)
    board[0, 0] = 1
    board[0, 1] = 2
    board[1, 0] = 3
    return board


def make_features(action: int = PASS_ACTION) -> np.ndarray:
    features = np.zeros((FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    features[4, :, :] = 1.0
    if action != PASS_ACTION:
        features[4, action // BOARD_SIZE, action % BOARD_SIZE] = 1.0
    return features


def make_policy(action: int = PASS_ACTION) -> np.ndarray:
    policy = np.zeros(ACTION_SPACE, dtype=np.float32)
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
        terminal=timestep == 1,
    )


def make_episode(
    episode_id: int,
    *,
    winner: int = 1,
    terminal_board: np.ndarray | None = None,
) -> TrajectoryEpisode:
    return TrajectoryEpisode(
        episode_id=episode_id,
        seed=episode_id,
        transitions=(
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
        ),
        winner=winner,
        end_reason=1,
        territory_scores=(1, 0),
        terminal_board=terminal_board,
    )


def test_absolute_to_perspective_swaps_colors_for_orange() -> None:
    board = make_board()
    blue = absolute_terminal_boards_to_perspective(board[None], np.asarray([1]))
    orange = absolute_terminal_boards_to_perspective(board[None], np.asarray([2]))

    assert blue[0, 0, 0] == 1
    assert blue[0, 0, 1] == 2
    assert orange[0, 0, 0] == 2
    assert orange[0, 0, 1] == 1
    assert blue[0, 1, 0] == orange[0, 1, 0] == 3
    assert blue[0, 2, 2] == orange[0, 2, 2] == 0


def test_terminal_board_validation_rejects_bad_shape_and_values() -> None:
    with pytest.raises(ValueError, match="shape"):
        validate_absolute_terminal_board(np.zeros((8, 9), dtype=np.uint8))
    bad = np.zeros((9, 9), dtype=np.uint8)
    bad[0, 0] = 4
    with pytest.raises(ValueError, match=r"\[0, 3\]"):
        validate_absolute_terminal_board(bad)


def test_store_round_trip_preserves_terminal_boards(tmp_path: Path) -> None:
    board = make_board()
    store = TrajectoryReplayStore.from_episodes(
        8,
        (make_episode(0, terminal_board=board), make_episode(1)),
    )
    path = tmp_path / "replay.npz"
    store.save(path, compressed=False)
    loaded = TrajectoryReplayStore.load(path)

    assert loaded.terminal_board_encoding == TERMINAL_BOARD_ENCODING
    assert loaded.terminal_board_present is not None
    assert loaded.terminal_board_present.tolist() == [True, False]
    assert np.array_equal(loaded.episodes[0].terminal_board, board)
    assert loaded.episodes[1].terminal_board is None


def test_legacy_replay_without_terminal_boards_loads(tmp_path: Path) -> None:
    store = TrajectoryReplayStore.from_episodes(8, (make_episode(0),))
    assert store.terminal_boards is None
    path = tmp_path / "legacy.npz"
    store.save(path, compressed=False)
    loaded = TrajectoryReplayStore.load(path)
    assert loaded.terminal_boards is None
    assert loaded.episodes[0].terminal_board is None


def test_mixed_append_marks_missing_boards_invalid() -> None:
    with_board = TrajectoryReplayStore.from_episodes(
        16,
        (make_episode(0, terminal_board=make_board()),),
    )
    without_board = TrajectoryReplayStore.from_episodes(16, (make_episode(1),))
    with_board.extend_store(without_board)

    present = with_board.terminal_board_present
    assert present is not None
    assert present.tolist() == [True, False]
    assert with_board.terminal_boards is not None
    assert np.array_equal(with_board.terminal_boards[0], make_board())


def test_capacity_trim_keeps_boards_aligned() -> None:
    store = TrajectoryReplayStore.from_episodes(
        2,
        (
            make_episode(0, terminal_board=make_board()),
            make_episode(1, terminal_board=make_board()),
        ),
    )
    assert len(store) == 4
    store.compact_to_capacity(2)

    assert store.episode_count == 1
    assert store.terminal_boards is not None
    assert store.terminal_board_present.tolist() == [True]
    assert store.episodes[0].episode_id == 1


def test_unknown_terminal_board_encoding_is_rejected() -> None:
    store = TrajectoryReplayStore.from_episodes(
        8,
        (make_episode(0, terminal_board=make_board()),),
    )
    payload = store.to_payload()
    payload["terminal_board_encoding"] = np.asarray("future_v9", dtype=np.str_)
    with pytest.raises(ValueError, match="terminal_board_encoding"):
        TrajectoryReplayStore.from_payload(payload)


def test_dataset_gathers_player_perspective_targets() -> None:
    board = make_board()
    store = TrajectoryReplayStore.from_episodes(
        8,
        (
            make_episode(0, terminal_board=board),
            make_episode(1, terminal_board=board),
        ),
    )
    dataset = TrajectoryReplayDataset(store)
    batch = dataset.sample_arrays(4, random.Random(0))

    assert batch.terminal_board_targets is not None
    assert batch.terminal_board_valid is not None
    assert batch.terminal_board_valid.tolist() == [True, True, True, True]
    for row in range(batch.terminal_board_targets.shape[0]):
        player = int(store.players[batch.indexes[row]])
        expected = absolute_terminal_boards_to_perspective(
            board[None],
            np.asarray([player]),
        )[0]
        assert np.array_equal(batch.terminal_board_targets[row], expected)


def test_self_play_collects_terminal_boards_single_and_batched() -> None:
    from great_kingdom_ai.self_play import (
        SelfPlayConfig,
        play_self_play_game,
        play_self_play_games_batched,
    )

    config = SelfPlayConfig(
        gumbel_simulations=2,
        gumbel_max_considered_actions=2,
        max_turns=400,
    )
    log, _samples = play_self_play_game(
        seed=0,
        config=config,
        prior_provider=lambda _state: [0.0] * ACTION_SPACE,
    )
    assert log.terminal_board is not None
    board = np.asarray(log.terminal_board, dtype=np.uint8).reshape(BOARD_SIZE, BOARD_SIZE)
    assert set(np.unique(board)).issubset({0, 1, 2, 3})

    batched = play_self_play_games_batched(
        seeds=[1],
        search_factory=lambda: None,
        config=config,
        prior_provider=None,
        evaluator_provider=None,
        feature_batch_prior_provider=lambda features, masks: [
            [0.0] * ACTION_SPACE for _ in features
        ],
        request_evaluator_provider=None,
    )
    assert len(batched) == 1
    batched_log, _ = batched[0]
    assert batched_log.terminal_board is not None


def test_augmentation_applies_same_symmetry_to_terminal_target() -> None:
    for symmetry in ALL_SYMMETRIES:
        board = make_board()
        sample = ReplaySample(
            features=make_features(10),
            policy=make_policy(10),
            value=1.0,
            terminal_board_target=board,
        )
        augmented = augment_sample(sample, symmetry)
        assert augmented.terminal_board_target is not None
        assert augmented.policy[81] == sample.policy[81]

    board = make_board()
    features = make_features(10)[None]
    policies = make_policy(10)[None]
    transformed_features, _policies, _masks, transformed_boards, _actions = (
        augment_training_arrays_randomly(
            features,
            policies,
            None,
            random.Random(3),
            symmetries=("rot90",),
            terminal_board_targets=board[None].astype(np.int64),
        )
    )
    assert transformed_boards is not None
    assert np.array_equal(
        transformed_boards[0],
        np.rot90(board.astype(np.int64), k=1).reshape(BOARD_SIZE, BOARD_SIZE),
    )
    assert transformed_features[0, 0, 0, 0] == features[0, 0, 0, 0]


def test_dataset_sample_drops_invalid_terminal_targets(monkeypatch) -> None:
    store = TrajectoryReplayStore.from_episodes(
        8,
        (make_episode(0, terminal_board=make_board()),),
    )
    dataset = TrajectoryReplayDataset(store)
    batch = dataset.sample_arrays(2, random.Random(0))
    invalid_batch = replace(
        batch,
        terminal_board_valid=np.asarray([True, False]),
    )
    monkeypatch.setattr(dataset, "sample_arrays", lambda *args, **kwargs: invalid_batch)

    samples = dataset.sample(2, random.Random(0))

    assert samples[0].terminal_board_target is not None
    assert samples[1].terminal_board_target is None
