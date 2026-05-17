"""Arena evaluation data types and Rust batch/search protocols."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from great_kingdom_ai.self_play_types import MoveLog, SelfPlayState

BLUE = 1
ORANGE = 2


class ArenaSearchResultLike(Protocol):
    def selected_action(self) -> int | None: ...

    def visit_counts(self) -> list[int]: ...


class ArenaSearchLike(Protocol):
    def search_with_logits_and_evaluator(
        self,
        state: SelfPlayState,
        policy_logits: list[float],
        evaluator: Callable[[Any], tuple[list[list[float]], list[float]]],
        root_value: float,
        leaf_batch_size: int = 8,
    ) -> ArenaSearchResultLike: ...


class ArenaBatchLike(Protocol):
    def len(self) -> int: ...

    def active_game_indexes(self) -> list[int]: ...

    def active_eval_request(self) -> Any: ...

    def active_legal_masks(self) -> list[list[bool]]: ...

    def current_players(self) -> list[int] | bytes: ...

    def candidate_players(self) -> list[int] | bytes: ...

    def seeds(self) -> list[int] | bytes: ...

    def search_active_with_logits_and_evaluator(
        self,
        policy_logits: list[list[float]],
        evaluator: Callable[[Any], tuple[list[list[float]], list[float]]],
        root_values: list[float],
        leaf_batch_size: int = 8,
    ) -> list[ArenaSearchResultLike | None]: ...

    def search_active_with_onnx_evaluators(
        self,
        candidate_evaluator: Any,
        best_evaluator: Any,
        leaf_batch_size: int = 8,
    ) -> tuple[list[ArenaSearchResultLike | None], list[list[float]]]: ...

    def apply_actions(self, actions: list[int | None]) -> list[int | None]: ...

    def is_terminal(self) -> list[bool]: ...

    def winners(self) -> list[int | None]: ...

    def end_reasons(self) -> list[int | None]: ...

    def territory_scores(self) -> list[tuple[int, int]]: ...


@dataclass
class ArenaOnnxEvaluators:
    candidate: Any
    best: Any


@dataclass(frozen=True)
class ArenaConfig:
    games: int = 20
    batch_size: int = 1
    seed_start: int = 0
    max_turns: int = 200
    gumbel_simulations: int = 128
    gumbel_max_considered_actions: int = 16
    gumbel_c_visit: float = 50.0
    gumbel_c_scale: float = 1.0
    gumbel_scale: float = 0.0
    opening_gumbel_turns: int = 0
    opening_gumbel_scale: float = 1.0
    policy_target_c_visit: float = 5.0
    policy_target_c_scale: float = 0.25
    policy_target_temperature: float = 1.0
    gumbel_seed: int = 0
    paired_seeds: bool = False
    leaf_batch_size: int = 8
    device: str = "cpu"
    promotion_threshold: float = 0.55

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")


@dataclass(frozen=True)
class ArenaGameResult:
    seed: int
    candidate_player: int
    best_player: int
    winner: int
    end_reason: int
    moves: list[MoveLog]
    territory_scores: tuple[int, int]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["moves"] = [move.__dict__ for move in self.moves]
        return data


@dataclass(frozen=True)
class ArenaSummary:
    games: int
    candidate_wins: int
    best_wins: int
    candidate_win_rate: float
    best_win_rate: float
    candidate_blue_games: int
    candidate_blue_wins: int
    candidate_orange_games: int
    candidate_orange_wins: int
    average_game_length: float
    promoted: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ArenaReport:
    config: ArenaConfig
    games: list[ArenaGameResult]
    summary: ArenaSummary

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "games": [game.to_dict() for game in self.games],
            "summary": self.summary.to_dict(),
        }


def validate_arena_config(config: ArenaConfig) -> None:
    if config.games < 0:
        raise ValueError("games must be non-negative")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.max_turns <= 0:
        raise ValueError("max_turns must be positive")
    if not 0.0 <= config.promotion_threshold <= 1.0:
        raise ValueError("promotion_threshold must be between 0 and 1")
    if config.gumbel_simulations <= 0:
        raise ValueError("gumbel_simulations must be positive")
    if config.gumbel_max_considered_actions <= 0:
        raise ValueError("gumbel_max_considered_actions must be positive")
    if config.gumbel_c_visit <= 0.0:
        raise ValueError("gumbel_c_visit must be positive")
    if config.gumbel_c_scale <= 0.0:
        raise ValueError("gumbel_c_scale must be positive")
    if not math.isfinite(config.gumbel_scale) or config.gumbel_scale < 0.0:
        raise ValueError("gumbel_scale must be finite and non-negative")
    if config.opening_gumbel_turns < 0:
        raise ValueError("opening_gumbel_turns must be non-negative")
    if (
        not math.isfinite(config.opening_gumbel_scale)
        or config.opening_gumbel_scale < 0.0
    ):
        raise ValueError("opening_gumbel_scale must be finite and non-negative")
    if not math.isfinite(config.policy_target_c_visit) or config.policy_target_c_visit <= 0.0:
        raise ValueError("policy_target_c_visit must be finite and positive")
    if not math.isfinite(config.policy_target_c_scale) or config.policy_target_c_scale <= 0.0:
        raise ValueError("policy_target_c_scale must be finite and positive")
    if (
        not math.isfinite(config.policy_target_temperature)
        or config.policy_target_temperature <= 0.0
    ):
        raise ValueError("policy_target_temperature must be finite and positive")
    if config.leaf_batch_size <= 0:
        raise ValueError("leaf_batch_size must be positive")


def other_player(player: int) -> int:
    return ORANGE if player == BLUE else BLUE


def arena_game_seed(config: ArenaConfig, game_index: int) -> int:
    seed_offset = game_index // 2 if config.paired_seeds else game_index
    return config.seed_start + seed_offset


__all__ = [
    "BLUE",
    "ORANGE",
    "ArenaBatchLike",
    "ArenaConfig",
    "ArenaGameResult",
    "ArenaOnnxEvaluators",
    "ArenaReport",
    "ArenaSearchLike",
    "ArenaSearchResultLike",
    "ArenaSummary",
    "arena_game_seed",
    "other_player",
    "validate_arena_config",
]
