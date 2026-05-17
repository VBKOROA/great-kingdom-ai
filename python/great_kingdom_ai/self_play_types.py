"""Self-play data types and Rust batch/search protocols."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, Protocol

import numpy as np

from great_kingdom_ai.game_core import SearchResultLike, SelfPlayState

PASS_ACTION = 81


class SearchLike(Protocol):
    def search_with_logits(
        self,
        state: SelfPlayState,
        policy_logits: list[float],
    ) -> SearchResultLike: ...

    def search_with_logits_and_evaluator(
        self,
        state: SelfPlayState,
        policy_logits: list[float],
        evaluator: Callable[[Any], tuple[list[list[float]], list[float]]],
        root_value: float,
        leaf_batch_size: int = 8,
    ) -> SearchResultLike: ...

    def set_simulations(self, simulations: int) -> None: ...

    def set_max_considered_actions(self, max_considered_actions: int) -> None: ...


class SelfPlayBatchLike(Protocol):
    def len(self) -> int: ...

    def active_count(self) -> int: ...

    def active_game_indexes(self) -> list[int]: ...

    def active_eval_request(self) -> Any: ...

    def current_players(self) -> list[int] | bytes: ...

    def is_terminal(self) -> list[bool]: ...

    def winners(self) -> list[int | None]: ...

    def end_reasons(self) -> list[int | None]: ...

    def territory_scores(self) -> list[tuple[int, int]]: ...

    def search_active_with_logits(
        self,
        policy_logits: list[list[float]],
    ) -> list[SearchResultLike | None]: ...

    def search_active_with_logits_and_evaluator(
        self,
        policy_logits: list[list[float]],
        evaluator: Callable[[Any], tuple[list[list[float]], list[float]]],
        root_values: list[float],
        leaf_batch_size: int = 8,
    ) -> list[SearchResultLike | None]: ...

    def apply_actions(self, actions: list[int | None]) -> list[int | None]: ...

    def set_simulations(self, simulations: list[int | None]) -> None: ...

    def set_max_considered_actions(
        self,
        max_considered_actions: list[int | None],
    ) -> None: ...


@dataclass(frozen=True)
class MoveLog:
    turn: int
    player: int
    action: int


@dataclass(frozen=True)
class GameLog:
    seed: int
    moves: list[MoveLog]
    winner: int
    end_reason: int
    territory_scores: tuple[int, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SmokeSummary:
    games: int
    total_moves: int
    max_moves: int
    blue_wins: int
    orange_wins: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class SelfPlayConfig:
    max_turns: int = 200
    gumbel_simulations: int = 128
    gumbel_max_considered_actions: int = 16
    gumbel_c_visit: float = 50.0
    gumbel_c_scale: float = 1.0
    gumbel_scale: float = 1.0
    policy_target_c_visit: float = 5.0
    policy_target_c_scale: float = 0.25
    policy_target_temperature: float = 1.0
    gumbel_seed: int = 0
    temperature_turns: int = 10
    sampling_temperature: float = 1.0
    playout_cap_randomization: bool = False
    playout_cap_full_search_fraction: float = 0.25
    playout_cap_full_simulations: int = 128
    playout_cap_fast_simulations: int = 16
    playout_cap_full_max_considered_actions: int | None = None
    playout_cap_fast_max_considered_actions: int | None = None
    leaf_batch_size: int = 8

    def __post_init__(self) -> None:
        if self.max_turns <= 0:
            raise ValueError("max_turns must be positive")
        if self.gumbel_simulations <= 0:
            raise ValueError("gumbel_simulations must be positive")
        if self.gumbel_max_considered_actions <= 0:
            raise ValueError("gumbel_max_considered_actions must be positive")
        if self.gumbel_c_visit <= 0.0:
            raise ValueError("gumbel_c_visit must be positive")
        if self.gumbel_c_scale <= 0.0:
            raise ValueError("gumbel_c_scale must be positive")
        if not np.isfinite(self.gumbel_scale) or self.gumbel_scale < 0.0:
            raise ValueError("gumbel_scale must be finite and non-negative")
        if not np.isfinite(self.policy_target_c_visit) or self.policy_target_c_visit <= 0.0:
            raise ValueError("policy_target_c_visit must be finite and positive")
        if not np.isfinite(self.policy_target_c_scale) or self.policy_target_c_scale <= 0.0:
            raise ValueError("policy_target_c_scale must be finite and positive")
        if (
            not np.isfinite(self.policy_target_temperature)
            or self.policy_target_temperature <= 0.0
        ):
            raise ValueError("policy_target_temperature must be finite and positive")
        if self.temperature_turns < 0:
            raise ValueError("temperature_turns must be non-negative")
        if self.sampling_temperature < 0.0:
            raise ValueError("sampling_temperature must be non-negative")
        if not 0.0 < self.playout_cap_full_search_fraction <= 1.0:
            raise ValueError("playout_cap_full_search_fraction must be in (0, 1]")
        if self.playout_cap_full_simulations <= 0:
            raise ValueError("playout_cap_full_simulations must be positive")
        if self.playout_cap_fast_simulations <= 0:
            raise ValueError("playout_cap_fast_simulations must be positive")
        if (
            self.playout_cap_full_max_considered_actions is not None
            and self.playout_cap_full_max_considered_actions <= 0
        ):
            raise ValueError("playout_cap_full_max_considered_actions must be positive")
        if (
            self.playout_cap_fast_max_considered_actions is not None
            and self.playout_cap_fast_max_considered_actions <= 0
        ):
            raise ValueError("playout_cap_fast_max_considered_actions must be positive")
        if self.leaf_batch_size <= 0:
            raise ValueError("leaf_batch_size must be positive")


__all__ = [
    "PASS_ACTION",
    "GameLog",
    "MoveLog",
    "SearchLike",
    "SearchResultLike",
    "SelfPlayBatchLike",
    "SelfPlayConfig",
    "SelfPlayState",
    "SmokeSummary",
]
