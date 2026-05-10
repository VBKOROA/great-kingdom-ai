"""Compatibility targets for trajectory replay."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from great_kingdom_ai.replay_buffer import ReplaySample
from great_kingdom_ai.self_play_data import value_target_for_player

if TYPE_CHECKING:
    from great_kingdom_ai.trajectory_replay import TrajectoryEpisode, TrajectoryTransition


def value_target_for_transition(transition: TrajectoryTransition) -> float:
    """Return the final-outcome value from a trajectory transition player's view."""
    if transition.winner is None:
        raise ValueError("transition winner is required for replay-sample compatibility")
    return value_target_for_player(player=transition.player, winner=transition.winner)


def replay_sample_from_transition(transition: TrajectoryTransition) -> ReplaySample:
    """Convert one trajectory transition into the legacy independent sample format."""
    return ReplaySample(
        features=transition.features,
        policy=transition.policy_target,
        value=value_target_for_transition(transition),
        root_policy_logits=transition.root_policy_logits,
        sample_weight=transition.sample_weight,
    )


def replay_samples_from_episode(episode: TrajectoryEpisode) -> list[ReplaySample]:
    """Convert all transitions in one episode into legacy replay samples."""
    return [replay_sample_from_transition(transition) for transition in episode.transitions]


def replay_samples_from_transitions(
    transitions: Iterable[TrajectoryTransition],
) -> list[ReplaySample]:
    """Convert sampled trajectory transitions into legacy replay samples."""
    return [replay_sample_from_transition(transition) for transition in transitions]
