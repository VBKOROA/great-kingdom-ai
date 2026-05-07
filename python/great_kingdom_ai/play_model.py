"""Model-backed move selection for the interactive play CLI."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from great_kingdom_ai.evaluator import evaluate_feature_batch_logits_values
from great_kingdom_ai.features import ACTION_SPACE


class ModelPlayState(Protocol):
    def feature_planes(self) -> list[float]: ...

    def legal_mask(self) -> list[bool]: ...

    def legal_actions(self) -> list[int]: ...


class SearchResultLike(Protocol):
    def selected_action(self) -> int | None: ...

    def visit_counts(self) -> list[int]: ...


class SearchLike(Protocol):
    def search_with_logits_and_evaluator(
        self,
        state: ModelPlayState,
        policy_logits: list[float],
        evaluator: Callable[[Any], tuple[list[list[float]], list[float]]],
        root_value: float,
        leaf_batch_size: int = 8,
    ) -> SearchResultLike: ...


@dataclass(frozen=True)
class ModelPlayConfig:
    device: str = "cpu"
    gumbel_simulations: int = 64
    gumbel_max_considered_actions: int = 16
    gumbel_c_visit: float = 50.0
    gumbel_c_scale: float = 1.0
    policy_target_c_visit: float = 5.0
    policy_target_c_scale: float = 0.25
    policy_target_temperature: float = 1.0
    gumbel_seed: int = 0
    leaf_batch_size: int = 8

    def __post_init__(self) -> None:
        if self.gumbel_simulations <= 0:
            raise ValueError("gumbel_simulations must be positive")
        if self.gumbel_max_considered_actions <= 0:
            raise ValueError("gumbel_max_considered_actions must be positive")
        if self.gumbel_c_visit <= 0.0:
            raise ValueError("gumbel_c_visit must be positive")
        if self.gumbel_c_scale <= 0.0:
            raise ValueError("gumbel_c_scale must be positive")
        if self.policy_target_c_visit <= 0.0:
            raise ValueError("policy_target_c_visit must be positive")
        if self.policy_target_c_scale <= 0.0:
            raise ValueError("policy_target_c_scale must be positive")
        if self.policy_target_temperature <= 0.0:
            raise ValueError("policy_target_temperature must be positive")
        if self.leaf_batch_size <= 0:
            raise ValueError("leaf_batch_size must be positive")


@dataclass
class ModelPlayer:
    model: Any
    search: SearchLike
    config: ModelPlayConfig

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        config: ModelPlayConfig | None = None,
    ) -> ModelPlayer:
        resolved_config = config if config is not None else ModelPlayConfig()
        from great_kingdom_ai.evaluate import load_model_from_checkpoint

        model = load_model_from_checkpoint(checkpoint, device=resolved_config.device)
        return cls(
            model=model,
            search=create_core_search_engine(resolved_config),
            config=resolved_config,
        )

    def select_action(self, state: ModelPlayState) -> int:
        return select_model_action(
            state=state,
            model=self.model,
            search=self.search,
            config=self.config,
        )


def select_model_action(
    *,
    state: ModelPlayState,
    model: Any,
    search: SearchLike,
    config: ModelPlayConfig,
) -> int:
    root_evaluation = evaluate_feature_batch_logits_values(
        model,
        [state.feature_planes()],
        [state.legal_mask()],
        device=config.device,
    )
    root_logits = [float(value) for value in root_evaluation.policy_logits[0]]
    root_value = float(root_evaluation.value[0])

    def evaluator(request: Any) -> tuple[list[list[float]], list[float]]:
        evaluation = evaluate_feature_batch_logits_values(
            model,
            request.feature_planes(),
            request.legal_masks(),
            device=config.device,
        )
        return (
            [[float(value) for value in policy] for policy in evaluation.policy_logits],
            [float(value) for value in evaluation.value],
        )

    result = search.search_with_logits_and_evaluator(
        state,
        root_logits,
        evaluator,
        root_value,
        config.leaf_batch_size,
    )
    return select_search_action(result, root_logits, state.legal_actions())


def select_search_action(
    result: SearchResultLike,
    priors: Sequence[float],
    legal_actions: Sequence[int],
) -> int:
    selected = result.selected_action()
    legal_set = set(legal_actions)
    if selected in legal_set:
        return int(selected)

    if not legal_actions:
        raise ValueError("state has no legal actions")

    visits = result.visit_counts()
    if len(visits) != len(priors):
        raise ValueError("visit count and prior lengths must match")
    if len(priors) != ACTION_SPACE:
        raise ValueError(f"expected {ACTION_SPACE} policy logits, got {len(priors)}")
    return max(legal_actions, key=lambda action: (visits[action], priors[action], -action))


def create_core_search_engine(config: ModelPlayConfig) -> SearchLike:
    try:
        import great_kingdom_core as core  # type: ignore[import-untyped]
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "great_kingdom_core is not installed. Build it with maturin before model play."
        ) from exc

    return cast(
        SearchLike,
        core.GumbelSearch(
            simulations=config.gumbel_simulations,
            max_considered_actions=config.gumbel_max_considered_actions,
            c_visit=config.gumbel_c_visit,
            c_scale=config.gumbel_c_scale,
            seed=config.gumbel_seed,
            policy_target_temperature=config.policy_target_temperature,
            policy_target_c_visit=config.policy_target_c_visit,
            policy_target_c_scale=config.policy_target_c_scale,
        ),
    )


__all__ = [
    "ModelPlayConfig",
    "ModelPlayState",
    "ModelPlayer",
    "SearchLike",
    "SearchResultLike",
    "create_core_search_engine",
    "select_model_action",
    "select_search_action",
]
