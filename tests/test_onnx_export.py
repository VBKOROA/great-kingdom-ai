from __future__ import annotations

import importlib
import importlib.util

import pytest

_torch_spec = importlib.util.find_spec("torch")
_onnx_spec = importlib.util.find_spec("onnx")
_onnxruntime_spec = importlib.util.find_spec("onnxruntime")
pytestmark = pytest.mark.skipif(
    _torch_spec is None or _onnx_spec is None or _onnxruntime_spec is None,
    reason="torch, onnx, and onnxruntime are required for ONNX export tests",
)
onnx = importlib.import_module("onnx") if _onnx_spec is not None else None

from great_kingdom_ai.features import ACTION_SPACE  # noqa: E402
from great_kingdom_ai.onnx_export import (  # noqa: E402
    DEFAULT_OPSET_VERSION,
    compare_checkpoint_to_onnx,
    export_checkpoint_to_onnx,
)
from great_kingdom_ai.train import TrainingConfig, create_train_state, save_checkpoint  # noqa: E402


def _save_test_checkpoint(tmp_path) -> object:
    state = create_train_state(TrainingConfig(model_preset="small"))
    state.model.eval()
    return save_checkpoint(state, tmp_path / "checkpoint.pt")


def test_export_checkpoint_to_onnx_uses_opset_17_and_dynamic_batch_axis(tmp_path) -> None:
    checkpoint_path = _save_test_checkpoint(tmp_path)
    onnx_path = tmp_path / "model.onnx"

    summary = export_checkpoint_to_onnx(checkpoint_path, onnx_path, dummy_batch_size=2)

    model = onnx.load(onnx_path)
    input_batch_dim = model.graph.input[0].type.tensor_type.shape.dim[0]
    policy_batch_dim = model.graph.output[0].type.tensor_type.shape.dim[0]
    value_batch_dim = model.graph.output[1].type.tensor_type.shape.dim[0]

    assert summary.output_path == onnx_path
    assert model.opset_import[0].version == DEFAULT_OPSET_VERSION
    assert model.graph.input[0].name == "features"
    assert [output.name for output in model.graph.output] == ["policy_logits", "value"]
    assert input_batch_dim.dim_param == "batch"
    assert policy_batch_dim.dim_param == "batch"
    assert value_batch_dim.dim_param == "batch"


def test_onnx_runtime_outputs_match_pytorch_checkpoint(tmp_path) -> None:
    checkpoint_path = _save_test_checkpoint(tmp_path)
    onnx_path = tmp_path / "model.onnx"
    export_checkpoint_to_onnx(checkpoint_path, onnx_path)

    summary = compare_checkpoint_to_onnx(checkpoint_path, onnx_path, batch_size=4, seed=17)

    assert summary.policy_shape == (4, ACTION_SPACE)
    assert summary.value_shape == (4,)
    assert summary.max_policy_abs_diff <= 1e-5
    assert summary.max_value_abs_diff <= 1e-5
    assert summary.passed
