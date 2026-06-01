"""Export EMA weights from a Great Kingdom checkpoint to ONNX.

This is a small personal utility. It does not write a PyTorch checkpoint; it
exports the source checkpoint's EMA weights directly to FP32 ONNX and/or a
static S8S8 QDQ INT8 ONNX model for ONNX Runtime inference.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, NoReturn

from great_kingdom_ai.onnx_export import DEFAULT_OPSET_VERSION, export_checkpoint_to_onnx
from great_kingdom_ai.onnx_quantization import (
    DEFAULT_QDQ_CALIBRATION_BATCH_SIZE,
    DEFAULT_QDQ_CALIBRATION_SAMPLES,
    DEFAULT_QDQ_CALIBRATION_SEED,
    DEFAULT_QDQ_QUANTIZATION_MODE,
    QDQ_CALIBRATION_METHODS,
    QDQ_QUANTIZATION_MODES,
    CalibrationFeatureSummary,
    quantize_onnx_s8s8_qdq,
)

QUANTIZATION_FORMAT_QDQ_S8S8 = "qdq-s8s8"
QUANTIZATION_FORMAT_DYNAMIC = "dynamic"


def export_ema_onnx(
    checkpoint_path: str | Path,
    *,
    onnx_output_path: str | Path | None = None,
    quantized_onnx_output_path: str | Path | None = None,
    allow_raw_fallback: bool = False,
    opset_version: int = DEFAULT_OPSET_VERSION,
    dummy_batch_size: int = 1,
    quantize_per_channel: bool = True,
    quantize_reduce_range: bool = False,
    quantize_preprocess: bool = True,
    quantization_format: str = QUANTIZATION_FORMAT_QDQ_S8S8,
    calibration_features_path: str | Path | None = None,
    calibration_sample_count: int = DEFAULT_QDQ_CALIBRATION_SAMPLES,
    calibration_batch_size: int = DEFAULT_QDQ_CALIBRATION_BATCH_SIZE,
    calibration_seed: int = DEFAULT_QDQ_CALIBRATION_SEED,
    calibration_method: str = "minmax",
    quantization_mode: str = DEFAULT_QDQ_QUANTIZATION_MODE,
) -> dict[str, Any]:
    if onnx_output_path is None and quantized_onnx_output_path is None:
        raise ValueError("onnx_output_path or quantized_onnx_output_path is required")
    if dummy_batch_size < 1:
        raise ValueError("dummy_batch_size must be at least 1")

    source = Path(checkpoint_path)
    _validate_ema_available(source, allow_raw_fallback=allow_raw_fallback)

    summary: dict[str, Any] = {
        "source": str(source),
        "used_raw_fallback": _uses_raw_fallback(source),
    }

    if onnx_output_path is None:
        with tempfile.TemporaryDirectory(prefix="gka-ema-onnx-") as temp_dir:
            temp_onnx = Path(temp_dir) / "ema-fp32.onnx"
            _export_fp32_onnx(
                source,
                temp_onnx,
                opset_version=opset_version,
                dummy_batch_size=dummy_batch_size,
            )
            if quantized_onnx_output_path is None:
                return summary
            quantization_summary = _quantize_onnx(
                temp_onnx,
                quantized_onnx_output_path,
                per_channel=quantize_per_channel,
                reduce_range=quantize_reduce_range,
                preprocess=quantize_preprocess,
                quantization_format=quantization_format,
                calibration_features_path=calibration_features_path,
                calibration_sample_count=calibration_sample_count,
                calibration_batch_size=calibration_batch_size,
                calibration_seed=calibration_seed,
                calibration_method=calibration_method,
                quantization_mode=quantization_mode,
            )
            summary["quantized_onnx_output"] = str(quantized_onnx_output_path)
            summary["quantization"] = quantization_summary
            return summary

    _export_fp32_onnx(
        source,
        onnx_output_path,
        opset_version=opset_version,
        dummy_batch_size=dummy_batch_size,
    )
    summary["onnx_output"] = str(onnx_output_path)

    if quantized_onnx_output_path is not None:
        quantization_summary = _quantize_onnx(
            onnx_output_path,
            quantized_onnx_output_path,
            per_channel=quantize_per_channel,
            reduce_range=quantize_reduce_range,
            preprocess=quantize_preprocess,
            quantization_format=quantization_format,
            calibration_features_path=calibration_features_path,
            calibration_sample_count=calibration_sample_count,
            calibration_batch_size=calibration_batch_size,
            calibration_seed=calibration_seed,
            calibration_method=calibration_method,
            quantization_mode=quantization_mode,
        )
        summary["quantized_onnx_output"] = str(quantized_onnx_output_path)
        summary["quantization"] = quantization_summary

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export checkpoint EMA weights to ONNX")
    parser.add_argument("--checkpoint", type=Path, required=True, help="input checkpoint")
    parser.add_argument("--onnx-output", type=Path, default=None, help="optional FP32 ONNX output")
    parser.add_argument(
        "--quantized-onnx-output",
        type=Path,
        default=None,
        help="optional INT8 ONNX output",
    )
    parser.add_argument(
        "--quantization-format",
        choices=[QUANTIZATION_FORMAT_QDQ_S8S8, QUANTIZATION_FORMAT_DYNAMIC],
        default=QUANTIZATION_FORMAT_QDQ_S8S8,
        help="INT8 ONNX quantization representation; qdq-s8s8 is the CPU serving default",
    )
    parser.add_argument(
        "--quantization-mode",
        choices=QDQ_QUANTIZATION_MODES,
        default=DEFAULT_QDQ_QUANTIZATION_MODE,
        help=(
            "QDQ quantization scope; selective-attention quantizes Conv/Gemm while keeping "
            "attention-sensitive nodes in FP32"
        ),
    )
    parser.add_argument(
        "--calibration-method",
        choices=QDQ_CALIBRATION_METHODS,
        default="minmax",
        help="QDQ static calibration method",
    )
    parser.add_argument(
        "--allow-raw-fallback",
        action="store_true",
        help="use model_state if ema_model_state is missing",
    )
    parser.add_argument(
        "--no-quantize-per-channel",
        action="store_true",
        help="disable per-channel weight quantization for the INT8 ONNX output",
    )
    parser.add_argument(
        "--quantize-reduce-range",
        action="store_true",
        help="use a reduced INT8 range for older CPU compatibility",
    )
    parser.add_argument(
        "--skip-quantize-preprocess",
        action="store_true",
        help="skip ONNX Runtime quantization pre-processing",
    )
    parser.add_argument(
        "--calibration-features",
        type=Path,
        default=None,
        help="optional .npz/.npy with a [N, 11, 9, 9] features array for QDQ calibration",
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=DEFAULT_QDQ_CALIBRATION_SAMPLES,
        help="maximum number of calibration feature rows for QDQ quantization",
    )
    parser.add_argument(
        "--calibration-batch-size",
        type=int,
        default=DEFAULT_QDQ_CALIBRATION_BATCH_SIZE,
        help="calibration batch size for QDQ quantization",
    )
    parser.add_argument(
        "--calibration-seed",
        type=int,
        default=DEFAULT_QDQ_CALIBRATION_SEED,
        help="seed for selecting/generating QDQ calibration features",
    )
    parser.add_argument("--opset-version", type=int, default=DEFAULT_OPSET_VERSION)
    parser.add_argument("--dummy-batch-size", type=int, default=1)
    return parser


def main() -> NoReturn:
    args = build_parser().parse_args()
    summary = export_ema_onnx(
        args.checkpoint,
        onnx_output_path=args.onnx_output,
        quantized_onnx_output_path=args.quantized_onnx_output,
        allow_raw_fallback=args.allow_raw_fallback,
        opset_version=args.opset_version,
        dummy_batch_size=args.dummy_batch_size,
        quantize_per_channel=not args.no_quantize_per_channel,
        quantize_reduce_range=args.quantize_reduce_range,
        quantize_preprocess=not args.skip_quantize_preprocess,
        quantization_format=args.quantization_format,
        calibration_features_path=args.calibration_features,
        calibration_sample_count=args.calibration_samples,
        calibration_batch_size=args.calibration_batch_size,
        calibration_seed=args.calibration_seed,
        calibration_method=args.calibration_method,
        quantization_mode=args.quantization_mode,
    )
    print(json.dumps(summary, sort_keys=True))
    raise SystemExit(0)


def _export_fp32_onnx(
    checkpoint_path: Path,
    output_path: str | Path,
    *,
    opset_version: int,
    dummy_batch_size: int,
) -> None:
    export_checkpoint_to_onnx(
        checkpoint_path,
        output_path,
        device="cpu",
        opset_version=opset_version,
        dummy_batch_size=dummy_batch_size,
        prefer_ema=True,
    )


def _validate_ema_available(checkpoint_path: Path, *, allow_raw_fallback: bool) -> None:
    if allow_raw_fallback:
        return
    checkpoint = _load_checkpoint_header(checkpoint_path)
    if checkpoint.get("ema_model_state") is None:
        raise ValueError(f"{checkpoint_path} does not contain ema_model_state")


def _uses_raw_fallback(checkpoint_path: Path) -> bool:
    checkpoint = _load_checkpoint_header(checkpoint_path)
    return checkpoint.get("ema_model_state") is None


def _load_checkpoint_header(checkpoint_path: Path) -> dict[str, Any]:
    torch = _import_torch()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"{checkpoint_path} must contain a checkpoint dictionary")
    return checkpoint


def _quantize_onnx(
    input_path: str | Path,
    output_path: str | Path,
    *,
    per_channel: bool,
    reduce_range: bool,
    preprocess: bool,
    quantization_format: str,
    calibration_features_path: str | Path | None,
    calibration_sample_count: int,
    calibration_batch_size: int,
    calibration_seed: int,
    calibration_method: str,
    quantization_mode: str,
) -> dict[str, Any]:
    if quantization_format == QUANTIZATION_FORMAT_QDQ_S8S8:
        calibration = _quantize_onnx_qdq_s8s8(
            input_path,
            output_path,
            per_channel=per_channel,
            reduce_range=reduce_range,
            preprocess=preprocess,
            calibration_features_path=calibration_features_path,
            calibration_sample_count=calibration_sample_count,
            calibration_batch_size=calibration_batch_size,
            calibration_seed=calibration_seed,
            calibration_method=calibration_method,
            quantization_mode=quantization_mode,
        )
        return {
            "format": QUANTIZATION_FORMAT_QDQ_S8S8,
            "activation_type": "QInt8",
            "weight_type": "QInt8",
            "quant_format": "QDQ",
            "calibration": calibration.to_json_dict(),
        }
    if quantization_format == QUANTIZATION_FORMAT_DYNAMIC:
        _quantize_onnx_dynamic(
            input_path,
            output_path,
            per_channel=per_channel,
            reduce_range=reduce_range,
            preprocess=preprocess,
        )
        return {
            "format": QUANTIZATION_FORMAT_DYNAMIC,
            "weight_type": "QInt8",
        }
    raise ValueError(f"unknown quantization format: {quantization_format}")


def _quantize_onnx_qdq_s8s8(
    input_path: str | Path,
    output_path: str | Path,
    *,
    per_channel: bool,
    reduce_range: bool,
    preprocess: bool,
    calibration_features_path: str | Path | None,
    calibration_sample_count: int,
    calibration_batch_size: int,
    calibration_seed: int,
    calibration_method: str,
    quantization_mode: str,
) -> CalibrationFeatureSummary:
    return quantize_onnx_s8s8_qdq(
        input_path,
        output_path,
        calibration_features_path=calibration_features_path,
        calibration_sample_count=calibration_sample_count,
        calibration_batch_size=calibration_batch_size,
        calibration_seed=calibration_seed,
        calibration_method=calibration_method,
        quantization_mode=quantization_mode,
        per_channel=per_channel,
        reduce_range=reduce_range,
        preprocess=preprocess,
    )


def _quantize_onnx_dynamic(
    input_path: str | Path,
    output_path: str | Path,
    *,
    per_channel: bool,
    reduce_range: bool,
    preprocess: bool,
) -> None:
    try:
        from onnxruntime.quantization import QuantType, quant_pre_process, quantize_dynamic
    except ModuleNotFoundError as exc:
        raise RuntimeError("onnxruntime is required for ONNX quantization") from exc

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    source = Path(input_path)
    if preprocess:
        with tempfile.TemporaryDirectory(prefix="gka-onnx-quant-pre-") as temp_dir:
            preprocessed = Path(temp_dir) / "preprocessed.onnx"
            quant_pre_process(source, preprocessed)
            quantize_dynamic(
                preprocessed,
                destination,
                per_channel=per_channel,
                reduce_range=reduce_range,
                weight_type=QuantType.QInt8,
            )
            return

    quantize_dynamic(
        source,
        destination,
        per_channel=per_channel,
        reduce_range=reduce_range,
        weight_type=QuantType.QInt8,
    )


def _import_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("torch is required to inspect checkpoint EMA weights") from exc
    return torch


if __name__ == "__main__":
    main()
