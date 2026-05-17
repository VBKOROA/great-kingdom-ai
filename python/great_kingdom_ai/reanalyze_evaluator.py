"""Public model evaluator API for trajectory reanalyze."""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable
from typing import Any

import numpy as np

from great_kingdom_ai.evaluator import evaluate_feature_arrays_logits_values

ReanalyzeProgressCallback = Callable[[str, int, int, str], None]


def evaluate_policy_logits_values(
    model: Any,
    features: np.ndarray,
    legal_masks: np.ndarray,
    *,
    batch_size: int,
    device: str,
    progress_callback: ReanalyzeProgressCallback | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    policy_logits: list[np.ndarray] = []
    values: list[np.ndarray] = []
    total_batches = math.ceil(features.shape[0] / batch_size)
    _report_progress(
        progress_callback,
        "eval",
        0,
        total_batches,
        f"rows={features.shape[0]}, batch_size={batch_size}, device={device}",
    )
    for start in range(0, features.shape[0], batch_size):
        end = min(start + batch_size, features.shape[0])
        batch_number = start // batch_size + 1
        evaluation = evaluate_feature_arrays_logits_values(
            model,
            features[start:end],
            legal_masks[start:end],
            device=device,
        )
        policy_logits.append(evaluation.policy_logits)
        values.append(evaluation.value)
        _report_progress(
            progress_callback,
            "eval",
            batch_number,
            total_batches,
            f"rows={start}:{end}",
        )
    return (
        np.concatenate(policy_logits, axis=0).astype(np.float32),
        np.concatenate(values, axis=0).astype(np.float32),
    )


def create_onnx_evaluator(
    onnx_model_path: str,
    *,
    device: str,
    max_batch_size: int,
) -> Any:
    core = import_core()
    return core.OnnxEvaluator(
        str(onnx_model_path),
        device=device,
        max_batch_size=max_batch_size,
    )


def evaluate_policy_logits_values_with_onnx(
    evaluator: Any,
    features: np.ndarray,
    *,
    batch_size: int,
    device: str,
    progress_callback: ReanalyzeProgressCallback | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    core = import_core()
    policy_logits: list[np.ndarray] = []
    values: list[np.ndarray] = []
    total_batches = math.ceil(features.shape[0] / batch_size)
    _report_progress(
        progress_callback,
        "eval",
        0,
        total_batches,
        (
            f"rows={features.shape[0]}, batch_size={batch_size}, "
            f"device={device}, backend=onnx"
        ),
    )
    for start in range(0, features.shape[0], batch_size):
        end = min(start + batch_size, features.shape[0])
        batch_number = start // batch_size + 1
        request = eval_request_from_feature_array(core, features[start:end])
        batch_logits, batch_values = evaluator.evaluate(request)
        policy_logits.append(np.asarray(batch_logits, dtype=np.float32))
        values.append(np.asarray(batch_values, dtype=np.float32))
        _report_progress(
            progress_callback,
            "eval",
            batch_number,
            total_batches,
            f"rows={start}:{end}",
        )
    return (
        np.concatenate(policy_logits, axis=0).astype(np.float32),
        np.concatenate(values, axis=0).astype(np.float32),
    )


def eval_request_from_feature_array(core: Any, features: np.ndarray) -> Any:
    rows = np.ascontiguousarray(features.reshape(features.shape[0], -1), dtype=np.float32)
    if hasattr(core.EvalRequest, "from_feature_plane_bytes"):
        return core.EvalRequest.from_feature_plane_bytes(rows.shape[0], rows.tobytes())
    return core.EvalRequest.from_feature_rows(rows.tolist())


def import_core() -> Any:
    try:
        return importlib.import_module("great_kingdom_core")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "great_kingdom_core is not installed. Build it with maturin before ONNX reanalyze."
        ) from exc


def _report_progress(
    progress_callback: ReanalyzeProgressCallback | None,
    stage: str,
    current: int,
    total: int,
    detail: str,
) -> None:
    if progress_callback is not None:
        progress_callback(stage, current, total, detail)


__all__ = [
    "ReanalyzeProgressCallback",
    "create_onnx_evaluator",
    "eval_request_from_feature_array",
    "evaluate_policy_logits_values",
    "evaluate_policy_logits_values_with_onnx",
    "import_core",
]
