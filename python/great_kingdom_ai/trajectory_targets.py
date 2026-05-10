"""Compatibility targets for trajectory replay."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from great_kingdom_ai.replay_buffer import ReplaySample
from great_kingdom_ai.self_play_data import value_target_for_player

if TYPE_CHECKING:
    from great_kingdom_ai.trajectory_replay import TrajectoryEpisode, TrajectoryTransition


@dataclass(frozen=True)
class BootstrapValueTargetConfig:
    """Configuration for n-step value targets over trajectory replay episodes."""

    bootstrap_td_steps: int = 0
    gamma: float = 1.0

    def __post_init__(self) -> None:
        if self.bootstrap_td_steps < 0:
            raise ValueError("bootstrap_td_steps must be non-negative")
        if not math.isfinite(self.gamma) or self.gamma < 0.0 or self.gamma > 1.0:
            raise ValueError("gamma must be finite and in [0, 1]")


def value_target_for_transition(transition: TrajectoryTransition) -> float:
    """Return the final-outcome value from a trajectory transition player's view."""
    if transition.winner is None:
        raise ValueError("transition winner is required for replay-sample compatibility")
    return value_target_for_player(player=transition.player, winner=transition.winner)


def bootstrap_value_target_for_transition(
    episode: TrajectoryEpisode,
    transition_index: int,
    config: BootstrapValueTargetConfig,
) -> float:
    """Return a terminal-or-n-step bootstrap value target for one episode transition.

    Future ``root_value`` predictions are stored from the future state's player
    perspective, so they are converted back to the sampled transition player's
    perspective before being used as a target.
    """
    transition = _episode_transition(episode, transition_index)
    if config.bootstrap_td_steps == 0 or transition.terminal:
        return value_target_for_transition(transition)

    target_index = transition_index + config.bootstrap_td_steps
    terminal_index = len(episode.transitions) - 1
    if target_index >= terminal_index:
        return value_target_for_transition(transition)

    bootstrap_transition = episode.transitions[target_index]
    bootstrap_value = _required_root_value(bootstrap_transition)
    if bootstrap_transition.player != transition.player:
        bootstrap_value = -bootstrap_value
    return (config.gamma**config.bootstrap_td_steps) * bootstrap_value


def value_target_for_episode_transition(
    episode: TrajectoryEpisode,
    transition_index: int,
    *,
    config: BootstrapValueTargetConfig | None = None,
) -> float:
    """Return the configured training value target for one episode transition."""
    if config is None:
        transition = _episode_transition(episode, transition_index)
        return value_target_for_transition(transition)
    return bootstrap_value_target_for_transition(episode, transition_index, config)


def replay_sample_from_transition(transition: TrajectoryTransition) -> ReplaySample:
    """Convert one trajectory transition into the legacy independent sample format."""
    return ReplaySample(
        features=transition.features,
        policy=transition.policy_target,
        value=value_target_for_transition(transition),
        root_policy_logits=transition.root_policy_logits,
        sample_weight=transition.sample_weight,
    )


def replay_sample_from_episode_transition(
    episode: TrajectoryEpisode,
    transition_index: int,
    *,
    value_target_config: BootstrapValueTargetConfig | None = None,
) -> ReplaySample:
    """Convert one episode transition into a legacy sample with configured value target."""
    transition = _episode_transition(episode, transition_index)
    return ReplaySample(
        features=transition.features,
        policy=transition.policy_target,
        value=value_target_for_episode_transition(
            episode,
            transition_index,
            config=value_target_config,
        ),
        root_policy_logits=transition.root_policy_logits,
        sample_weight=transition.sample_weight,
    )


def replay_samples_from_episode(
    episode: TrajectoryEpisode,
    *,
    value_target_config: BootstrapValueTargetConfig | None = None,
) -> list[ReplaySample]:
    """Convert all transitions in one episode into legacy replay samples."""
    return [
        replay_sample_from_episode_transition(
            episode,
            index,
            value_target_config=value_target_config,
        )
        for index in range(len(episode.transitions))
    ]


def replay_samples_from_transitions(
    transitions: Iterable[TrajectoryTransition],
) -> list[ReplaySample]:
    """Convert sampled trajectory transitions into legacy replay samples."""
    return [replay_sample_from_transition(transition) for transition in transitions]


def _episode_transition(
    episode: TrajectoryEpisode,
    transition_index: int,
) -> TrajectoryTransition:
    if transition_index < 0 or transition_index >= len(episode.transitions):
        raise IndexError("transition_index is out of range")
    return episode.transitions[transition_index]


def _required_root_value(transition: TrajectoryTransition) -> float:
    if transition.root_value is None:
        raise ValueError("bootstrap transition root_value is required")
    root_value = float(transition.root_value)
    if not math.isfinite(root_value) or root_value < -1.0 or root_value > 1.0:
        raise ValueError("bootstrap transition root_value must be finite and in [-1, 1]")
    return root_value
