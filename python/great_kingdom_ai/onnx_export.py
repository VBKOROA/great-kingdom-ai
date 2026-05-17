"""ONNX export and CPU parity helpers for policy-value checkpoints."""

from __future__ import annotations

import argparse
import importlib
import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NoReturn

import numpy as np

from great_kingdom_ai.features import ACTION_SPACE, BOARD_SIZE, FEATURE_CHANNELS
from great_kingdom_ai.training import load_checkpoint

DEFAULT_OPSET_VERSION = 17
DEFAULT_PARITY_TOLERANCE = 1e-5
ONNX_PRECISIONS = ("fp32", "fp16")


@dataclass(frozen=True)
class OnnxExportSummary:
    checkpoint_path: Path
    output_path: Path
    opset_version: int
    dummy_batch_size: int
    precision: str = "fp32"
    input_name: str = "features"
    output_names: tuple[str, str] = ("policy_logits", "value")

    def to_json_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["checkpoint_path"] = str(self.checkpoint_path)
        data["output_path"] = str(self.output_path)
        return data


@dataclass(frozen=True)
class OnnxParitySummary:
    policy_shape: tuple[int, int]
    value_shape: tuple[int]
    max_policy_abs_diff: float
    max_value_abs_diff: float
    tolerance: float

    @property
    def passed(self) -> bool:
        return (
            self.policy_shape[1] == ACTION_SPACE
            and self.max_policy_abs_diff <= self.tolerance
            and self.max_value_abs_diff <= self.tolerance
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "policy_shape": list(self.policy_shape),
            "value_shape": list(self.value_shape),
            "max_policy_abs_diff": self.max_policy_abs_diff,
            "max_value_abs_diff": self.max_value_abs_diff,
            "tolerance": self.tolerance,
            "passed": self.passed,
        }


def export_checkpoint_to_onnx(
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    device: str = "cpu",
    opset_version: int = DEFAULT_OPSET_VERSION,
    dummy_batch_size: int = 2,
    prefer_ema: bool = True,
    precision: str = "fp32",
) -> OnnxExportSummary:
    """Export a training checkpoint to ONNX with dynamic batch and optional FP16 internals."""
    if dummy_batch_size < 1:
        raise ValueError("dummy_batch_size must be at least 1")
    _validate_precision(precision)

    torch = _import_torch()
    checkpoint = Path(checkpoint_path)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    state = load_checkpoint(checkpoint, device=device, prefer_ema=prefer_ema)
    model = state.model.float()
    model.eval()

    dummy_features = torch.zeros(
        (dummy_batch_size, FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE),
        dtype=torch.float32,
        device=device,
    )
    with torch.inference_mode():
        torch.onnx.export(
            model,
            dummy_features,
            destination,
            input_names=["features"],
            output_names=["policy_logits", "value"],
            dynamic_axes={
                "features": {0: "batch"},
                "policy_logits": {0: "batch"},
                "value": {0: "batch"},
            },
            opset_version=opset_version,
            dynamo=False,
        )

    if precision == "fp16":
        _convert_onnx_to_fp16_keep_io(destination)

    return OnnxExportSummary(
        checkpoint_path=checkpoint,
        output_path=destination,
        opset_version=opset_version,
        dummy_batch_size=dummy_batch_size,
        precision=precision,
    )


def compare_checkpoint_to_onnx(
    checkpoint_path: str | Path,
    onnx_path: str | Path,
    *,
    batch_size: int = 3,
    seed: int = 0,
    tolerance: float = DEFAULT_PARITY_TOLERANCE,
    prefer_ema: bool = True,
) -> OnnxParitySummary:
    """Compare PyTorch checkpoint inference against ONNX Runtime CPU inference."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")

    torch = _import_torch()
    ort = _import_onnxruntime()

    state = load_checkpoint(checkpoint_path, device="cpu", prefer_ema=prefer_ema)
    model = state.model.float()
    model.eval()

    rng = np.random.default_rng(seed)
    features = rng.normal(
        size=(batch_size, FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE),
    ).astype(np.float32)

    with torch.inference_mode():
        torch_features = torch.from_numpy(features)
        torch_policy, torch_value = model(torch_features)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_policy, onnx_value = session.run(None, {"features": features})
    policy_shape_raw = tuple(int(dim) for dim in onnx_policy.shape)
    value_shape_raw = tuple(int(dim) for dim in onnx_value.shape)
    if len(policy_shape_raw) != 2:
        raise ValueError(f"expected ONNX policy output rank 2, got {policy_shape_raw}")
    if len(value_shape_raw) != 1:
        raise ValueError(f"expected ONNX value output rank 1, got {value_shape_raw}")

    policy_diff = np.max(np.abs(torch_policy.detach().cpu().numpy() - onnx_policy))
    value_diff = np.max(np.abs(torch_value.detach().cpu().numpy() - onnx_value))
    return OnnxParitySummary(
        policy_shape=(policy_shape_raw[0], policy_shape_raw[1]),
        value_shape=(value_shape_raw[0],),
        max_policy_abs_diff=float(policy_diff),
        max_value_abs_diff=float(value_diff),
        tolerance=tolerance,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a Great Kingdom checkpoint to ONNX")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Input PyTorch checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="Output ONNX model path")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--opset-version", type=int, default=DEFAULT_OPSET_VERSION)
    parser.add_argument("--dummy-batch-size", type=int, default=2)
    parser.add_argument("--precision", choices=ONNX_PRECISIONS, default="fp32")
    parser.add_argument(
        "--check-parity",
        action="store_true",
        help="compare exported ONNX CPU output against the PyTorch checkpoint",
    )
    parser.add_argument(
        "--no-ema",
        action="store_true",
        help="export raw training weights even when EMA weights are available",
    )
    parser.add_argument("--parity-batch-size", type=int, default=3)
    parser.add_argument("--parity-seed", type=int, default=0)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_PARITY_TOLERANCE)
    return parser


def main() -> NoReturn:
    args = build_parser().parse_args()
    export_summary = export_checkpoint_to_onnx(
        args.checkpoint,
        args.output,
        device=args.device,
        opset_version=args.opset_version,
        dummy_batch_size=args.dummy_batch_size,
        prefer_ema=not args.no_ema,
        precision=args.precision,
    )
    print(json.dumps({"event": "onnx_export", **export_summary.to_json_dict()}, sort_keys=True))

    if args.check_parity:
        parity_summary = compare_checkpoint_to_onnx(
            args.checkpoint,
            args.output,
            batch_size=args.parity_batch_size,
            seed=args.parity_seed,
            tolerance=args.tolerance,
            prefer_ema=not args.no_ema,
        )
        print(json.dumps({"event": "onnx_parity", **parity_summary.to_json_dict()}, sort_keys=True))
        if not parity_summary.passed:
            raise SystemExit(1)

    raise SystemExit(0)


def _import_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required for ONNX export") from exc
    return torch


def _import_onnxruntime() -> Any:
    try:
        return importlib.import_module("onnxruntime")
    except ModuleNotFoundError as exc:
        raise RuntimeError("onnxruntime is required for ONNX parity checks") from exc


def _validate_precision(precision: str) -> None:
    if precision not in ONNX_PRECISIONS:
        choices = ", ".join(ONNX_PRECISIONS)
        raise ValueError(f"precision must be one of: {choices}")


def _convert_onnx_to_fp16_keep_io(path: Path) -> None:
    try:
        import onnx
        from onnxconverter_common import float16
    except ModuleNotFoundError as exc:
        raise RuntimeError("onnxconverter-common is required for FP16 ONNX export") from exc

    model = onnx.load(path)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"the float32 number .* will be truncated to .*",
            category=UserWarning,
            module=r"onnxconverter_common\.float16",
        )
        converted = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.save(converted, path)


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_OPSET_VERSION",
    "DEFAULT_PARITY_TOLERANCE",
    "ONNX_PRECISIONS",
    "OnnxExportSummary",
    "OnnxParitySummary",
    "build_parser",
    "compare_checkpoint_to_onnx",
    "export_checkpoint_to_onnx",
]
