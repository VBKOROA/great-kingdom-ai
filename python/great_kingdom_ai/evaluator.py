"""Batch neural-network evaluation helpers for Rust MCTS requests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from great_kingdom_ai.features import ACTION_SPACE, BOARD_SIZE, FEATURE_CHANNELS

if TYPE_CHECKING:
    import torch
    from torch import nn


class EvalRequestLike(Protocol):
    def feature_planes(self) -> list[list[float]]: ...

    def legal_masks(self) -> list[list[bool]]: ...


@dataclass(frozen=True)
class NetworkEvaluation:
    policy: np.ndarray
    value: np.ndarray


def evaluate_request(
    model: nn.Module,
    request: EvalRequestLike,
    *,
    device: torch.device | str | None = None,
) -> NetworkEvaluation:
    features = request.feature_planes()
    masks = request.legal_masks()
    return evaluate_feature_batch(model, features, masks, device=device)


def evaluate_feature_batch(
    model: nn.Module,
    feature_planes: list[list[float]],
    legal_masks: list[list[bool]],
    *,
    device: torch.device | str | None = None,
) -> NetworkEvaluation:
    torch = _import_torch()
    if len(feature_planes) != len(legal_masks):
        raise ValueError("feature batch and legal mask batch must have the same length")

    batch_size = len(feature_planes)
    features = _feature_array(feature_planes, batch_size)
    masks = _legal_mask_array(legal_masks, batch_size)

    model_device = _model_device(model)
    target_device = torch.device(device) if device is not None else model_device
    inputs = torch.from_numpy(features).to(device=target_device)
    mask_tensor = torch.from_numpy(masks).to(device=target_device)
    model.to(target_device)
    model.eval()

    with torch.no_grad():
        policy_logits, value = model(inputs)
        if policy_logits.shape != (batch_size, ACTION_SPACE):
            raise ValueError(
                f"expected policy logits shape {(batch_size, ACTION_SPACE)}, "
                f"got {tuple(policy_logits.shape)}"
            )
        if value.shape != (batch_size,):
            raise ValueError(f"expected value shape {(batch_size,)}, got {tuple(value.shape)}")
        masked_logits = policy_logits.masked_fill(
            ~mask_tensor,
            torch.finfo(policy_logits.dtype).min,
        )
        policy = torch.softmax(masked_logits, dim=1)

    return NetworkEvaluation(
        policy=policy.cpu().numpy().astype(np.float32, copy=False),
        value=value.cpu().numpy().astype(np.float32, copy=False),
    )


def _feature_array(feature_planes: list[list[float]], batch_size: int) -> np.ndarray:
    features = np.asarray(feature_planes, dtype=np.float32)
    expected = FEATURE_CHANNELS * BOARD_SIZE * BOARD_SIZE
    if features.shape != (batch_size, expected):
        raise ValueError(f"expected feature shape {(batch_size, expected)}, got {features.shape}")
    return features.reshape(batch_size, FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE)


def _legal_mask_array(legal_masks: list[list[bool]], batch_size: int) -> np.ndarray:
    masks = np.asarray(legal_masks, dtype=np.bool_)
    if masks.shape != (batch_size, ACTION_SPACE):
        raise ValueError(
            f"expected legal mask shape {(batch_size, ACTION_SPACE)}, got {masks.shape}"
        )
    if np.any(~masks.any(axis=1)):
        raise ValueError("each legal mask must contain at least one legal action")
    return masks


def _model_device(model: nn.Module) -> torch.device:
    torch = _import_torch()
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _import_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required for neural-network evaluation") from exc
    return torch
