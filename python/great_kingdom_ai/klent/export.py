"""KLENT ONNX export wrappers and CPU parity checks.

Two output contracts are produced from one Q-head checkpoint:

* ``eval``: ``(policy_logits, value)`` for the existing Rust Gumbel evaluator,
  with ``value = sum_a softmax(pi_masked)_a * Q(s, a)``.
* ``actor``: ``(policy_logits, value, q_values)`` for the zero-search actor,
  which computes ``pi'`` and ``E_pi'[Q]`` in Rust FP32.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from torch import nn

from great_kingdom_ai.features import ACTION_SPACE, BOARD_SIZE, FEATURE_CHANNELS
from great_kingdom_ai.klent._torch import _import_torch
from great_kingdom_ai.klent.checkpoint import load_klent_checkpoint
from great_kingdom_ai.klent.targets import legal_mask_from_features, masked_state_value
from great_kingdom_ai.klent.types import KlentPolicyValueModel
from great_kingdom_ai.onnx_export import (
    DEFAULT_OPSET_VERSION,
    DEFAULT_PARITY_TOLERANCE,
    ONNX_PRECISIONS,
    _convert_onnx_to_fp16_keep_io,
    _import_onnxruntime,
)

if TYPE_CHECKING:
    from torch import Tensor

KlentExportKind = Literal["eval", "actor"]
KLENT_EXPORT_KINDS: tuple[KlentExportKind, ...] = ("eval", "actor")
KLENT_ACTOR_OUTPUT_NAMES = ("policy_logits", "value", "q_values")
KLENT_EVAL_OUTPUT_NAMES = ("policy_logits", "value")


@dataclass(frozen=True)
class KlentOnnxExportSummary:
    checkpoint_path: Path
    output_path: Path
    kind: KlentExportKind
    opset_version: int
    precision: str
    dummy_batch_size: int
    output_names: tuple[str, ...]
    input_name: str = "features"

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_path": str(self.checkpoint_path),
            "output_path": str(self.output_path),
            "kind": self.kind,
            "opset_version": self.opset_version,
            "precision": self.precision,
            "dummy_batch_size": self.dummy_batch_size,
            "output_names": list(self.output_names),
            "input_name": self.input_name,
        }


@dataclass(frozen=True)
class KlentOnnxParitySummary:
    kind: KlentExportKind
    policy_shape: tuple[int, int]
    value_shape: tuple[int, ...]
    q_values_shape: tuple[int, ...] | None
    max_policy_abs_diff: float
    max_value_abs_diff: float
    max_q_values_abs_diff: float | None
    tolerance: float

    @property
    def passed(self) -> bool:
        policy_ok = self.policy_shape[1] == ACTION_SPACE
        value_ok = self.value_shape == (self.policy_shape[0],)
        q_ok = self.q_values_shape is None or (
            self.q_values_shape[0] == self.policy_shape[0]
            and self.q_values_shape[1] == ACTION_SPACE
        )
        diffs_ok = (
            self.max_policy_abs_diff <= self.tolerance
            and self.max_value_abs_diff <= self.tolerance
            and (
                self.max_q_values_abs_diff is None
                or self.max_q_values_abs_diff <= self.tolerance
            )
        )
        return policy_ok and value_ok and q_ok and diffs_ok

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "policy_shape": list(self.policy_shape),
            "value_shape": list(self.value_shape),
            "q_values_shape": (
                None if self.q_values_shape is None else list(self.q_values_shape)
            ),
            "max_policy_abs_diff": self.max_policy_abs_diff,
            "max_value_abs_diff": self.max_value_abs_diff,
            "max_q_values_abs_diff": self.max_q_values_abs_diff,
            "tolerance": self.tolerance,
            "passed": self.passed,
        }


class KlentGumbelExportWrapper(nn.Module):
    """Two-output wrapper that keeps the Rust Gumbel evaluator contract."""

    output_names = KLENT_EVAL_OUTPUT_NAMES

    def __init__(self, model: KlentPolicyValueModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        policy_logits, q_values = self.model.forward_q(features)
        legal_mask = legal_mask_from_features(features)
        value = masked_state_value(policy_logits, q_values, legal_mask)
        return policy_logits, value


class KlentActorExportWrapper(nn.Module):
    """Three-output wrapper for the Rust zero-search actor."""

    output_names = KLENT_ACTOR_OUTPUT_NAMES

    def __init__(self, model: KlentPolicyValueModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        policy_logits, q_values = self.model.forward_q(features)
        legal_mask = legal_mask_from_features(features)
        value = masked_state_value(policy_logits, q_values, legal_mask)
        return policy_logits, value, q_values


def export_klent_checkpoint_to_onnx(
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    kind: KlentExportKind = "eval",
    device: str = "cpu",
    opset_version: int = DEFAULT_OPSET_VERSION,
    dummy_batch_size: int = 2,
    precision: str = "fp32",
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    optimizer: str = "adamw",
) -> KlentOnnxExportSummary:
    """Export a Q-head KLENT checkpoint with the requested output contract."""
    if dummy_batch_size < 1:
        raise ValueError("dummy_batch_size must be at least 1")
    if precision not in ONNX_PRECISIONS:
        raise ValueError(f"precision must be one of: {', '.join(ONNX_PRECISIONS)}")
    wrapper = _wrapper_for_kind(kind)

    torch = _import_torch()
    checkpoint = Path(checkpoint_path)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    state = load_klent_checkpoint(
        checkpoint,
        device=device,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        optimizer=optimizer,
    )
    model = state.model.float()
    model.eval()
    export_module = wrapper(model)

    dummy_features = torch.zeros(
        (dummy_batch_size, FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE),
        dtype=torch.float32,
        device=device,
    )
    output_names = list(export_module.output_names)
    with torch.inference_mode():
        torch.onnx.export(
            export_module,
            dummy_features,
            destination,
            input_names=["features"],
            output_names=output_names,
            dynamic_axes={
                "features": {0: "batch"},
                **{name: {0: "batch"} for name in output_names},
            },
            opset_version=opset_version,
            dynamo=False,
        )

    if precision == "fp16":
        _convert_onnx_to_fp16_keep_io(destination)

    return KlentOnnxExportSummary(
        checkpoint_path=checkpoint,
        output_path=destination,
        kind=kind,
        opset_version=opset_version,
        precision=precision,
        dummy_batch_size=dummy_batch_size,
        output_names=export_module.output_names,
    )


def compare_klent_checkpoint_to_onnx(
    checkpoint_path: str | Path,
    onnx_path: str | Path,
    *,
    kind: KlentExportKind = "eval",
    batch_size: int = 3,
    seed: int = 0,
    tolerance: float = DEFAULT_PARITY_TOLERANCE,
    device: str = "cpu",
) -> KlentOnnxParitySummary:
    """Compare PyTorch wrapper inference against ONNX Runtime CPU inference."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")
    wrapper = _wrapper_for_kind(kind)

    torch = _import_torch()
    ort = _import_onnxruntime()
    state = load_klent_checkpoint(checkpoint_path, device=device)
    model = state.model.float()
    model.eval()
    export_module = wrapper(model)

    rng = np.random.default_rng(seed)
    features = rng.normal(
        size=(batch_size, FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE),
    ).astype(np.float32)
    with torch.inference_mode():
        torch_outputs = export_module(torch.from_numpy(features))
        torch_outputs = tuple(
            tensor.detach().cpu().numpy() for tensor in torch_outputs
        )

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_outputs = session.run(None, {"features": features})
    output_names = list(export_module.output_names)
    if len(onnx_outputs) != len(torch_outputs):
        raise ValueError(
            f"expected {len(torch_outputs)} ONNX outputs, got {len(onnx_outputs)}"
        )

    by_name = dict(zip(output_names, onnx_outputs, strict=True))
    policy = by_name["policy_logits"]
    value = by_name["value"]
    q_values = by_name.get("q_values")
    if policy.ndim != 2:
        raise ValueError(f"expected policy_logits rank 2, got {policy.shape}")
    if value.ndim != 1:
        raise ValueError(f"expected value rank 1, got {value.shape}")
    if q_values is not None and q_values.ndim != 2:
        raise ValueError(f"expected q_values rank 2, got {q_values.shape}")
    for name, array in by_name.items():
        if not np.isfinite(array).all():
            raise ValueError(f"ONNX output {name} contains non-finite values")

    torch_by_name = dict(zip(output_names, torch_outputs, strict=True))
    max_policy_diff = float(np.max(np.abs(torch_by_name["policy_logits"] - policy)))
    max_value_diff = float(np.max(np.abs(torch_by_name["value"] - value)))
    max_q_values_diff = (
        None
        if q_values is None
        else float(np.max(np.abs(torch_by_name["q_values"] - q_values)))
    )
    return KlentOnnxParitySummary(
        kind=kind,
        policy_shape=(int(policy.shape[0]), int(policy.shape[1])),
        value_shape=tuple(int(dim) for dim in value.shape),
        q_values_shape=(
            None if q_values is None else tuple(int(dim) for dim in q_values.shape)
        ),
        max_policy_abs_diff=max_policy_diff,
        max_value_abs_diff=max_value_diff,
        max_q_values_abs_diff=max_q_values_diff,
        tolerance=tolerance,
    )


def onnx_output_names(onnx_path: str | Path) -> list[str]:
    """Load an ONNX file and return its graph output names."""
    onnx = importlib.import_module("onnx")
    model = onnx.load(str(onnx_path))
    return [str(output.name) for output in model.graph.output]


def _wrapper_for_kind(
    kind: KlentExportKind,
) -> type[KlentGumbelExportWrapper] | type[KlentActorExportWrapper]:
    if kind == "eval":
        return KlentGumbelExportWrapper
    if kind == "actor":
        return KlentActorExportWrapper
    raise ValueError(f"kind must be one of: {', '.join(KLENT_EXPORT_KINDS)}")


__all__ = [
    "KLENT_ACTOR_OUTPUT_NAMES",
    "KLENT_EVAL_OUTPUT_NAMES",
    "KLENT_EXPORT_KINDS",
    "KlentActorExportWrapper",
    "KlentExportKind",
    "KlentGumbelExportWrapper",
    "KlentOnnxExportSummary",
    "KlentOnnxParitySummary",
    "compare_klent_checkpoint_to_onnx",
    "export_klent_checkpoint_to_onnx",
    "onnx_output_names",
]