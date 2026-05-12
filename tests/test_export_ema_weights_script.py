from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_torch_spec = importlib.util.find_spec("torch")
pytestmark = pytest.mark.skipif(
    _torch_spec is None,
    reason="torch is required for EMA ONNX export tests",
)

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "export_ema_weights.py"
SPEC = importlib.util.spec_from_file_location("export_ema_weights", SCRIPT_PATH)
assert SPEC is not None
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)

from great_kingdom_ai.train import (  # noqa: E402
    TrainingConfig,
    create_train_state,
    save_checkpoint,
)


def _torch():
    return importlib.import_module("torch")


def _save_ema_checkpoint(path: Path) -> Path:
    torch = _torch()
    state = create_train_state(TrainingConfig(ema_decay=0.9))
    assert state.ema_model is not None
    with torch.no_grad():
        for tensor in state.model.state_dict().values():
            if tensor.is_floating_point():
                tensor.fill_(3.0)
        for tensor in state.ema_model.state_dict().values():
            if tensor.is_floating_point():
                tensor.fill_(1.0)
    return save_checkpoint(state, path)


def test_export_ema_onnx_writes_fp32_onnx(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _save_ema_checkpoint(tmp_path / "source.pt")
    output = tmp_path / "ema.onnx"
    exported: list[tuple[Path, Path, dict[str, object]]] = []

    def fake_export(checkpoint_path: Path, output_path: Path, **kwargs: object) -> None:
        exported.append((Path(checkpoint_path), Path(output_path), kwargs))
        Path(output_path).write_text("fp32 onnx", encoding="utf-8")

    monkeypatch.setattr(module, "export_checkpoint_to_onnx", fake_export)

    summary = module.export_ema_onnx(source, onnx_output_path=output)

    assert output.read_text(encoding="utf-8") == "fp32 onnx"
    assert summary["onnx_output"] == str(output)
    assert summary["used_raw_fallback"] is False
    assert exported == [
        (
            source,
            output,
            {
                "device": "cpu",
                "opset_version": module.DEFAULT_OPSET_VERSION,
                "dummy_batch_size": 1,
                "prefer_ema": True,
            },
        )
    ]


def test_export_ema_onnx_can_write_only_quantized_onnx(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _save_ema_checkpoint(tmp_path / "source.pt")
    quantized = tmp_path / "ema-int8.onnx"

    def fake_export(checkpoint_path: Path, output_path: Path, **kwargs: object) -> None:
        assert Path(checkpoint_path) == source
        Path(output_path).write_text("fp32 onnx", encoding="utf-8")

    def fake_quantize(
        input_path: Path,
        output_path: Path,
        *,
        per_channel: bool,
        reduce_range: bool,
        preprocess: bool,
        calibration_features_path: Path | None,
        calibration_sample_count: int,
        calibration_batch_size: int,
        calibration_seed: int,
    ) -> module.CalibrationFeatureSummary:
        assert Path(input_path).read_text(encoding="utf-8") == "fp32 onnx"
        assert per_channel is True
        assert reduce_range is False
        assert preprocess is True
        assert calibration_features_path is None
        assert calibration_sample_count == module.DEFAULT_QDQ_CALIBRATION_SAMPLES
        assert calibration_batch_size == module.DEFAULT_QDQ_CALIBRATION_BATCH_SIZE
        assert calibration_seed == module.DEFAULT_QDQ_CALIBRATION_SEED
        Path(output_path).write_text("int8 onnx", encoding="utf-8")
        return module.CalibrationFeatureSummary(
            source="synthetic",
            sample_count=calibration_sample_count,
            batch_size=calibration_batch_size,
        )

    monkeypatch.setattr(module, "export_checkpoint_to_onnx", fake_export)
    monkeypatch.setattr(module, "_quantize_onnx_qdq_s8s8", fake_quantize)

    summary = module.export_ema_onnx(source, quantized_onnx_output_path=quantized)

    assert quantized.read_text(encoding="utf-8") == "int8 onnx"
    assert summary["quantized_onnx_output"] == str(quantized)
    assert summary["quantization"]["format"] == module.QUANTIZATION_FORMAT_QDQ_S8S8
    assert summary["quantization"]["activation_type"] == "QInt8"
    assert summary["quantization"]["weight_type"] == "QInt8"
    assert summary["quantization"]["quant_format"] == "QDQ"
    assert "onnx_output" not in summary


def test_export_ema_onnx_can_use_legacy_dynamic_quantization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _save_ema_checkpoint(tmp_path / "source.pt")
    quantized = tmp_path / "ema-dynamic-int8.onnx"
    calls: list[tuple[Path, Path]] = []

    def fake_export(checkpoint_path: Path, output_path: Path, **kwargs: object) -> None:
        assert Path(checkpoint_path) == source
        Path(output_path).write_text("fp32 onnx", encoding="utf-8")

    def fake_quantize(
        input_path: Path,
        output_path: Path,
        *,
        per_channel: bool,
        reduce_range: bool,
        preprocess: bool,
    ) -> None:
        assert per_channel is True
        assert reduce_range is False
        assert preprocess is True
        calls.append((Path(input_path), Path(output_path)))
        Path(output_path).write_text("dynamic int8 onnx", encoding="utf-8")

    monkeypatch.setattr(module, "export_checkpoint_to_onnx", fake_export)
    monkeypatch.setattr(module, "_quantize_onnx_dynamic", fake_quantize)

    summary = module.export_ema_onnx(
        source,
        quantized_onnx_output_path=quantized,
        quantization_format=module.QUANTIZATION_FORMAT_DYNAMIC,
    )

    assert quantized.read_text(encoding="utf-8") == "dynamic int8 onnx"
    assert calls[0][1] == quantized
    assert summary["quantization"] == {
        "format": module.QUANTIZATION_FORMAT_DYNAMIC,
        "weight_type": "QInt8",
    }


def test_export_ema_onnx_rejects_missing_ema_by_default(tmp_path: Path) -> None:
    state = create_train_state(TrainingConfig())
    source = save_checkpoint(state, tmp_path / "source.pt")

    with pytest.raises(ValueError, match="ema_model_state"):
        module.export_ema_onnx(source, onnx_output_path=tmp_path / "ema.onnx")
