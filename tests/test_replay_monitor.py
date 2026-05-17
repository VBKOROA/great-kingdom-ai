from __future__ import annotations

from pathlib import Path

import numpy as np
from great_kingdom_ai.features import ACTION_SPACE, BOARD_SIZE, FEATURE_CHANNELS, PASS_ACTION
from great_kingdom_ai.replay import (
    TrajectoryEpisode,
    TrajectoryReplayStore,
    TrajectoryTransition,
    legal_mask_from_features,
)
from great_kingdom_ai.replay_monitor import (
    ReplayMonitorConfig,
    ReplayMonitorState,
    check_replay_once,
    run_replay_monitor,
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


def make_episode(*, root_policy_logits: np.ndarray | None) -> TrajectoryEpisode:
    features = make_features(1)
    transition = TrajectoryTransition(
        episode_id=0,
        timestep=0,
        player=1,
        features=features,
        legal_mask=legal_mask_from_features(features),
        action=1,
        policy_target=make_policy(1),
        root_policy_logits=root_policy_logits,
        winner=1,
        terminal=True,
    )
    return TrajectoryEpisode(
        episode_id=0,
        seed=0,
        transitions=(transition,),
        winner=1,
        end_reason=1,
        territory_scores=(1, 0),
    )


def test_replay_monitor_reports_missing_replay_as_warning(tmp_path: Path) -> None:
    report = check_replay_once(ReplayMonitorConfig(replay_path=tmp_path / "missing.npz"))

    assert report.ok is True
    assert report.exists is False
    assert [alert.code for alert in report.alerts] == ["replay_missing"]


def test_replay_monitor_accepts_root_logits_when_thresholds_allow_sharp_targets(
    tmp_path: Path,
) -> None:
    path = tmp_path / "trajectory-replay.npz"
    root_logits = np.zeros((ACTION_SPACE,), dtype=np.float32)
    root_logits[1] = 3.0
    TrajectoryReplayStore.from_episodes(
        4,
        [make_episode(root_policy_logits=root_logits)],
    ).save(path, compressed=False)

    report = check_replay_once(
        ReplayMonitorConfig(
            replay_path=path,
            sharp_target_max_probability_mean=1.1,
            sharp_target_max_probability_p50=1.1,
        )
    )

    assert report.ok is True
    assert report.rows == 1
    assert report.metrics["root_prior_available_fraction"] == 1.0
    assert report.alerts == []


def test_replay_monitor_alerts_when_root_logits_are_missing(tmp_path: Path) -> None:
    path = tmp_path / "trajectory-replay.npz"
    TrajectoryReplayStore.from_episodes(
        4,
        [make_episode(root_policy_logits=None)],
    ).save(path, compressed=False)

    report = check_replay_once(
        ReplayMonitorConfig(
            replay_path=path,
            sharp_target_max_probability_mean=1.1,
            sharp_target_max_probability_p50=1.1,
        )
    )

    assert report.ok is False
    assert "root_policy_logits_missing" in [alert.code for alert in report.alerts]


def test_replay_monitor_alerts_when_replay_stops_growing(tmp_path: Path) -> None:
    path = tmp_path / "trajectory-replay.npz"
    root_logits = np.zeros((ACTION_SPACE,), dtype=np.float32)
    TrajectoryReplayStore.from_episodes(
        4,
        [make_episode(root_policy_logits=root_logits)],
    ).save(path, compressed=False)
    config = ReplayMonitorConfig(
        replay_path=path,
        stale_checks=1,
        sharp_target_max_probability_mean=1.1,
        sharp_target_max_probability_p50=1.1,
    )
    state = ReplayMonitorState()

    first = check_replay_once(config, state=state)
    second = check_replay_once(config, state=state)

    assert "replay_rows_not_growing" not in [alert.code for alert in first.alerts]
    assert "replay_rows_not_growing" in [alert.code for alert in second.alerts]


def test_replay_monitor_once_returns_report(tmp_path: Path) -> None:
    report = run_replay_monitor(
        ReplayMonitorConfig(replay_path=tmp_path / "missing.npz"),
        once=True,
        json_output=True,
    )[0]

    assert report.exists is False
