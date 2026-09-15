"""KLENT (regularized policy optimization) targets and losses."""

from great_kingdom_ai.klent.loss import KlentLossBreakdown, compute_klent_losses
from great_kingdom_ai.klent.targets import (
    compute_analytical_policy,
    compute_lambda_returns,
    legal_mask_from_features,
    masked_state_value,
)
from great_kingdom_ai.klent.types import KlentConfig

__all__ = [
    "KlentConfig",
    "KlentLossBreakdown",
    "compute_analytical_policy",
    "compute_klent_losses",
    "compute_lambda_returns",
    "legal_mask_from_features",
    "masked_state_value",
]