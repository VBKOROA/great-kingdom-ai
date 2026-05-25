"""Trajectory replay schema objects."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from great_kingdom_ai.features import BOARD_SIZE, FEATURE_CHANNELS

FEATURE_SHAPE = (FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE)


class MoveLike(Protocol):
    turn: int
    player: int
    action: int


class GameLogLike(Protocol):
    seed: int
    moves: Sequence[MoveLike]
    winner: int
    end_reason: int
    territory_scores: tuple[int, int]


@dataclass(frozen=True)
class TrajectoryTransition:
    episode_id: int
    timestep: int
    player: int
    features: np.ndarray | None
    legal_mask: np.ndarray | None
    action: int
    policy_target: np.ndarray
    root_policy_logits: np.ndarray | None = None
    root_value: float | None = None
    next_features: np.ndarray | None = None
    winner: int | None = None
    terminal: bool = False
    model_version: int = 0
    search_config_hash: str = ""
    created_iteration: int = 0
    sample_weight: float = 1.0


@dataclass(frozen=True)
class TrajectoryEpisode:
    episode_id: int
    seed: int
    transitions: tuple[TrajectoryTransition, ...]
    winner: int
    end_reason: int
    territory_scores: tuple[int, int]
    turn_players: np.ndarray | None = None
    turn_actions: np.ndarray | None = None
    turn_root_values: np.ndarray | None = None
    turn_full_search: np.ndarray | None = None
