"""ONNX Runtime quantization helpers for exported policy-value models."""

from __future__ import annotations

import importlib
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from great_kingdom_ai.features import BOARD_CELLS, BOARD_SIZE, FEATURE_CHANNELS

FEATURE_SHAPE = (FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE)
DEFAULT_QDQ_CALIBRATION_SAMPLES = 64
DEFAULT_QDQ_CALIBRATION_BATCH_SIZE = 8
DEFAULT_QDQ_CALIBRATION_SEED = 0
QDQ_CALIBRATION_METHODS = ("minmax", "entropy", "percentile")
QDQ_QUANTIZATION_MODES = ("full", "selective-attention")
DEFAULT_QDQ_QUANTIZATION_MODE = "full"
SELECTIVE_ATTENTION_QUANTIZED_OP_TYPES = ("Conv", "Gemm")
SELECTIVE_ATTENTION_EXCLUDED_OP_TYPES = ("LayerNormalization", "MatMul", "Softmax")
ATTENTION_NODE_NAME_MARKERS = ("/attention/", ".attention.", "attention/")


@dataclass(frozen=True)
class CalibrationFeatureSummary:
    source: str
    sample_count: int
    batch_size: int
    calibration_method: str = "minmax"
    quantization_mode: str = DEFAULT_QDQ_QUANTIZATION_MODE
    quantized_op_types: tuple[str, ...] | None = None
    excluded_node_count: int = 0
    excluded_node_samples: tuple[str, ...] = ()

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "sample_count": self.sample_count,
            "batch_size": self.batch_size,
            "calibration_method": self.calibration_method,
            "quantization_mode": self.quantization_mode,
            "quantized_op_types": (
                None if self.quantized_op_types is None else list(self.quantized_op_types)
            ),
            "excluded_node_count": self.excluded_node_count,
            "excluded_node_samples": list(self.excluded_node_samples),
        }


class FeatureCalibrationDataReader:
    """Small ONNX Runtime calibration reader over an in-memory feature array."""

    def __init__(
        self,
        features: np.ndarray,
        *,
        input_name: str = "features",
        batch_size: int = DEFAULT_QDQ_CALIBRATION_BATCH_SIZE,
    ) -> None:
        if batch_size < 1:
            raise ValueError("calibration batch size must be at least 1")
        feature_array = _validate_feature_array(features)
        if feature_array.shape[0] == 0:
            raise ValueError("calibration features must contain at least one sample")
        self._features = feature_array
        self._input_name = input_name
        self._batch_size = batch_size
        self._offset = 0

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self._offset >= self._features.shape[0]:
            return None
        end = min(self._offset + self._batch_size, self._features.shape[0])
        batch = self._features[self._offset : end]
        self._offset = end
        return {self._input_name: np.ascontiguousarray(batch, dtype=np.float32)}

    def rewind(self) -> None:
        self._offset = 0


def quantize_onnx_s8s8_qdq(
    input_path: str | Path,
    output_path: str | Path,
    *,
    calibration_features_path: str | Path | None = None,
    calibration_sample_count: int = DEFAULT_QDQ_CALIBRATION_SAMPLES,
    calibration_batch_size: int = DEFAULT_QDQ_CALIBRATION_BATCH_SIZE,
    calibration_seed: int = DEFAULT_QDQ_CALIBRATION_SEED,
    per_channel: bool = True,
    reduce_range: bool = False,
    preprocess: bool = True,
    calibration_method: str = "minmax",
    quantization_mode: str = DEFAULT_QDQ_QUANTIZATION_MODE,
) -> CalibrationFeatureSummary:
    """Quantize an FP32 ONNX model as static S8S8 QDQ for ONNX Runtime."""
    if calibration_sample_count < 1:
        raise ValueError("calibration sample count must be at least 1")
    calibration_method = _normalize_choice(
        calibration_method,
        choices=QDQ_CALIBRATION_METHODS,
        name="calibration_method",
    )
    quantization_mode = _normalize_choice(
        quantization_mode,
        choices=QDQ_QUANTIZATION_MODES,
        name="quantization_mode",
    )

    quantization = _import_onnxruntime_quantization()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    features, calibration_source = _calibration_features(
        calibration_features_path=calibration_features_path,
        sample_count=calibration_sample_count,
        seed=calibration_seed,
    )
    reader = FeatureCalibrationDataReader(features, batch_size=calibration_batch_size)

    source = Path(input_path)
    model_input = source
    quantized_op_types, nodes_to_exclude = _quantization_selection(
        source,
        quantization_mode=quantization_mode,
    )
    if preprocess:
        with tempfile.TemporaryDirectory(prefix="gka-onnx-quant-pre-") as temp_dir:
            preprocessed = Path(temp_dir) / "preprocessed.onnx"
            quantization.quant_pre_process(source, preprocessed)
            _, nodes_to_exclude = _quantization_selection(
                preprocessed,
                quantization_mode=quantization_mode,
            )
            _quantize_static_s8s8_qdq(
                quantization,
                preprocessed,
                destination,
                reader=reader,
                per_channel=per_channel,
                reduce_range=reduce_range,
                calibration_method=calibration_method,
                quantized_op_types=quantized_op_types,
                nodes_to_exclude=nodes_to_exclude,
            )
    else:
        _quantize_static_s8s8_qdq(
            quantization,
            model_input,
            destination,
            reader=reader,
            per_channel=per_channel,
            reduce_range=reduce_range,
            calibration_method=calibration_method,
            quantized_op_types=quantized_op_types,
            nodes_to_exclude=nodes_to_exclude,
        )

    return CalibrationFeatureSummary(
        source=calibration_source,
        sample_count=int(features.shape[0]),
        batch_size=calibration_batch_size,
        calibration_method=calibration_method,
        quantization_mode=quantization_mode,
        quantized_op_types=quantized_op_types,
        excluded_node_count=len(nodes_to_exclude),
        excluded_node_samples=tuple(nodes_to_exclude[:8]),
    )


def _quantize_static_s8s8_qdq(
    quantization: Any,
    model_input: Path,
    model_output: Path,
    *,
    reader: FeatureCalibrationDataReader,
    per_channel: bool,
    reduce_range: bool,
    calibration_method: str,
    quantized_op_types: tuple[str, ...] | None,
    nodes_to_exclude: tuple[str, ...],
) -> None:
    reader.rewind()
    kwargs: dict[str, Any] = {
        "quant_format": quantization.QuantFormat.QDQ,
        "activation_type": quantization.QuantType.QInt8,
        "weight_type": quantization.QuantType.QInt8,
        "per_channel": per_channel,
        "reduce_range": reduce_range,
        "calibrate_method": _calibration_method_enum(quantization, calibration_method),
    }
    if quantized_op_types is not None:
        kwargs["op_types_to_quantize"] = list(quantized_op_types)
    if nodes_to_exclude:
        kwargs["nodes_to_exclude"] = list(nodes_to_exclude)
    quantization.quantize_static(model_input, model_output, reader, **kwargs)


def _quantization_selection(
    model_path: Path,
    *,
    quantization_mode: str,
) -> tuple[tuple[str, ...] | None, tuple[str, ...]]:
    if quantization_mode == "full":
        return None, ()
    if quantization_mode != "selective-attention":
        raise ValueError(f"unknown quantization mode: {quantization_mode}")

    excluded_nodes = _selective_attention_excluded_nodes(model_path)
    if not excluded_nodes:
        raise ValueError(
            "selective-attention quantization did not find any attention-sensitive ONNX nodes"
        )
    return SELECTIVE_ATTENTION_QUANTIZED_OP_TYPES, excluded_nodes


def _selective_attention_excluded_nodes(model_path: Path) -> tuple[str, ...]:
    try:
        onnx = importlib.import_module("onnx")
    except ModuleNotFoundError as exc:
        raise RuntimeError("onnx is required for selective attention quantization") from exc

    model = onnx.load(model_path)
    selected: list[str] = []
    for node in model.graph.node:
        name = str(node.name)
        if not name:
            continue
        if node.op_type in SELECTIVE_ATTENTION_EXCLUDED_OP_TYPES or _is_attention_node_name(name):
            selected.append(name)
    selected.sort()
    return tuple(selected)


def _is_attention_node_name(name: str) -> bool:
    normalized = name.lower()
    return any(marker in normalized for marker in ATTENTION_NODE_NAME_MARKERS)


def _calibration_method_enum(quantization: Any, calibration_method: str) -> Any:
    if calibration_method == "minmax":
        return quantization.CalibrationMethod.MinMax
    if calibration_method == "entropy":
        return quantization.CalibrationMethod.Entropy
    if calibration_method == "percentile":
        return quantization.CalibrationMethod.Percentile
    raise ValueError(f"unknown calibration method: {calibration_method}")


def _normalize_choice(value: str, *, choices: Sequence[str], name: str) -> str:
    normalized = value.lower().replace("_", "-")
    if normalized not in choices:
        joined = ", ".join(choices)
        raise ValueError(f"{name} must be one of: {joined}")
    return normalized


def _calibration_features(
    *,
    calibration_features_path: str | Path | None,
    sample_count: int,
    seed: int,
) -> tuple[np.ndarray, str]:
    if calibration_features_path is None:
        return _synthetic_calibration_features(sample_count, seed=seed), "synthetic"

    path = Path(calibration_features_path)
    selected = _load_selected_feature_array(path, sample_count=sample_count, seed=seed)
    return selected, str(path)


def _load_selected_feature_array(path: Path, *, sample_count: int, seed: int) -> np.ndarray:
    data = np.load(path)
    if isinstance(data, np.lib.npyio.NpzFile):
        with data:
            if "features" in data:
                features = _validate_feature_array(data["features"])
                return _select_samples(features, sample_count=sample_count, seed=seed)
        return _load_selected_trajectory_replay_features(
            path,
            sample_count=sample_count,
            seed=seed,
        )
    features = _validate_feature_array(data)
    return _select_samples(features, sample_count=sample_count, seed=seed)


def _load_feature_array(path: Path) -> np.ndarray:
    data = np.load(path)
    if isinstance(data, np.lib.npyio.NpzFile):
        with data:
            if "features" not in data:
                raise ValueError(f"{path} must contain a 'features' array")
            return _validate_feature_array(data["features"])
    return _validate_feature_array(data)


def _select_samples(features: np.ndarray, *, sample_count: int, seed: int) -> np.ndarray:
    feature_array = _validate_feature_array(features)
    if feature_array.shape[0] <= sample_count:
        return feature_array
    rng = np.random.default_rng(seed)
    indexes = rng.choice(feature_array.shape[0], size=sample_count, replace=False)
    indexes.sort()
    return np.ascontiguousarray(feature_array[indexes], dtype=np.float32)


def _load_selected_trajectory_replay_features(
    path: Path,
    *,
    sample_count: int,
    seed: int,
) -> np.ndarray:
    from great_kingdom_ai.replay import TrajectoryReplayStore
    from great_kingdom_ai.replay.dataset import _features_and_masks_for_rows

    replay = TrajectoryReplayStore.load(path)
    if len(replay) == 0:
        raise ValueError(f"{path} trajectory replay must contain at least one transition")

    rng = np.random.default_rng(seed)
    row_count = min(sample_count, len(replay))
    indexes = rng.choice(len(replay), size=row_count, replace=False)
    indexes.sort()
    features, _legal_masks = _features_and_masks_for_rows(
        replay,
        np.asarray(indexes, dtype=np.int64),
    )
    return _validate_feature_array(features)


def _synthetic_calibration_features(sample_count: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    features = np.zeros((sample_count, *FEATURE_SHAPE), dtype=np.float32)
    cell_indexes = np.arange(BOARD_CELLS)

    for row in range(sample_count):
        planes = features[row].reshape(FEATURE_CHANNELS, BOARD_CELLS)
        own_count = int(rng.integers(0, 5))
        opponent_count = int(rng.integers(0, 5))
        neutral_count = int(rng.integers(1, 4))
        occupied_count = min(BOARD_CELLS, own_count + opponent_count + neutral_count)
        occupied = rng.choice(cell_indexes, size=occupied_count, replace=False)

        own_end = own_count
        opponent_end = own_end + opponent_count
        own_cells = occupied[:own_end]
        opponent_cells = occupied[own_end:opponent_end]
        neutral_cells = occupied[opponent_end:]

        planes[0, own_cells] = 1.0
        planes[1, opponent_cells] = 1.0
        planes[2, neutral_cells] = 1.0
        empty_mask = planes[0] + planes[1] + planes[2] == 0.0
        planes[3, empty_mask] = 1.0

        empty_cells = cell_indexes[empty_mask]
        own_territory = empty_cells[rng.random(empty_cells.shape[0]) < 0.12]
        opponent_territory = empty_cells[rng.random(empty_cells.shape[0]) < 0.12]
        planes[5, own_territory] = 1.0
        planes[6, opponent_territory] = 1.0
        legal_mask = empty_mask & (planes[6] == 0.0)
        planes[4, legal_mask] = 1.0

        planes[7, :] = np.float32(rng.integers(0, 5) / 4.0)
        planes[8, :] = np.float32(rng.integers(0, 5) / 4.0)
        planes[9, :] = np.float32(rng.integers(0, 2))
        planes[10, :] = np.float32(rng.integers(0, 2))

    return features


def _validate_feature_array(features: np.ndarray) -> np.ndarray:
    feature_array = np.asarray(features, dtype=np.float32)
    if feature_array.ndim != 4 or tuple(feature_array.shape[1:]) != FEATURE_SHAPE:
        raise ValueError(
            f"expected calibration features shape [N, {FEATURE_SHAPE}], got {features.shape}"
        )
    if not np.isfinite(feature_array).all():
        raise ValueError("calibration features must be finite")
    return np.ascontiguousarray(feature_array, dtype=np.float32)


def _import_onnxruntime_quantization() -> Any:
    try:
        return importlib.import_module("onnxruntime.quantization")
    except ModuleNotFoundError as exc:
        raise RuntimeError("onnxruntime is required for ONNX quantization") from exc


__all__ = [
    "DEFAULT_QDQ_CALIBRATION_BATCH_SIZE",
    "DEFAULT_QDQ_CALIBRATION_SAMPLES",
    "DEFAULT_QDQ_CALIBRATION_SEED",
    "DEFAULT_QDQ_QUANTIZATION_MODE",
    "QDQ_CALIBRATION_METHODS",
    "QDQ_QUANTIZATION_MODES",
    "CalibrationFeatureSummary",
    "FeatureCalibrationDataReader",
    "quantize_onnx_s8s8_qdq",
]
