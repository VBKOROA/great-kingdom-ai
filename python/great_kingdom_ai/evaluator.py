"""Batch neural-network evaluation helpers for Rust MCTS requests."""

from __future__ import annotations

import os
import sys
import time
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


@dataclass
class _EvalProfileStats:
    calls: int = 0
    samples: int = 0
    max_batch: int = 0
    feature_seconds: float = 0.0
    transfer_seconds: float = 0.0
    model_seconds: float = 0.0
    output_seconds: float = 0.0


_PROFILE_STATS = _EvalProfileStats()


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
    profile = _profile_enabled()
    start = time.perf_counter() if profile else 0.0
    features = _feature_array(feature_planes, batch_size)
    masks = _legal_mask_array(legal_masks, batch_size)
    feature_done = time.perf_counter() if profile else 0.0

    model_device = _model_device(model)
    target_device = torch.device(device) if device is not None else model_device
    inputs = torch.from_numpy(features).to(device=target_device)
    mask_tensor = torch.from_numpy(masks).to(device=target_device)
    model.to(target_device)
    model.eval()
    transfer_done = time.perf_counter() if profile else 0.0

    with torch.no_grad():
        policy_logits, value = model(inputs)
        if profile and target_device.type == "cuda":
            torch.cuda.synchronize(target_device)
        model_done = time.perf_counter() if profile else 0.0
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

    policy_array = policy.cpu().numpy().astype(np.float32, copy=False)
    value_array = value.cpu().numpy().astype(np.float32, copy=False)
    output_done = time.perf_counter() if profile else 0.0
    if profile:
        _record_profile(
            batch_size=batch_size,
            feature_seconds=feature_done - start,
            transfer_seconds=transfer_done - feature_done,
            model_seconds=model_done - transfer_done,
            output_seconds=output_done - model_done,
        )

    return NetworkEvaluation(policy=policy_array, value=value_array)


def _profile_enabled() -> bool:
    value = os.environ.get("GKA_EVAL_PROFILE", "")
    return value not in {"", "0", "false", "False", "no", "No"}


def _record_profile(
    *,
    batch_size: int,
    feature_seconds: float,
    transfer_seconds: float,
    model_seconds: float,
    output_seconds: float,
) -> None:
    _PROFILE_STATS.calls += 1
    _PROFILE_STATS.samples += batch_size
    _PROFILE_STATS.max_batch = max(_PROFILE_STATS.max_batch, batch_size)
    _PROFILE_STATS.feature_seconds += feature_seconds
    _PROFILE_STATS.transfer_seconds += transfer_seconds
    _PROFILE_STATS.model_seconds += model_seconds
    _PROFILE_STATS.output_seconds += output_seconds
    interval = int(os.environ.get("GKA_EVAL_PROFILE_INTERVAL", "100"))
    if interval <= 0 or _PROFILE_STATS.calls % interval != 0:
        return

    calls = _PROFILE_STATS.calls
    total_seconds = (
        _PROFILE_STATS.feature_seconds
        + _PROFILE_STATS.transfer_seconds
        + _PROFILE_STATS.model_seconds
        + _PROFILE_STATS.output_seconds
    )
    avg_batch = _PROFILE_STATS.samples / calls
    print(
        "[gka-eval-profile] "
        f"calls={calls} samples={_PROFILE_STATS.samples} "
        f"avg_batch={avg_batch:.1f} max_batch={_PROFILE_STATS.max_batch} "
        f"feature={_PROFILE_STATS.feature_seconds:.3f}s "
        f"transfer={_PROFILE_STATS.transfer_seconds:.3f}s "
        f"model={_PROFILE_STATS.model_seconds:.3f}s "
        f"output={_PROFILE_STATS.output_seconds:.3f}s "
        f"total={total_seconds:.3f}s",
        file=sys.stderr,
        flush=True,
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
