from __future__ import annotations

import importlib
import importlib.util

import numpy as np
import pytest

_torch_spec = importlib.util.find_spec("torch")
_onnx_spec = importlib.util.find_spec("onnx")
_onnxruntime_spec = importlib.util.find_spec("onnxruntime")
_onnxconverter_spec = importlib.util.find_spec("onnxconverter_common")
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


@pytest.mark.skipif(
    _onnxconverter_spec is None,
    reason="onnxconverter-common is required for FP16 ONNX export",
)
def test_export_checkpoint_to_onnx_fp16_keeps_float32_io(tmp_path) -> None:
    checkpoint_path = _save_test_checkpoint(tmp_path)
    onnx_path = tmp_path / "model-fp16.onnx"

    summary = export_checkpoint_to_onnx(checkpoint_path, onnx_path, precision="fp16")

    model = onnx.load(onnx_path)
    input_type = model.graph.input[0].type.tensor_type.elem_type
    output_types = [output.type.tensor_type.elem_type for output in model.graph.output]

    assert summary.precision == "fp16"
    assert input_type == onnx.TensorProto.FLOAT
    assert output_types == [onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT]
    assert any(node.op_type == "Cast" for node in model.graph.node)


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


@pytest.mark.skipif(
    importlib.util.find_spec("great_kingdom_core") is None,
    reason="great_kingdom_core extension is not installed",
)
def test_core_onnx_self_play_smoke_reaches_terminal_game(tmp_path) -> None:
    core = importlib.import_module("great_kingdom_core")
    checkpoint_path = _save_test_checkpoint(tmp_path)
    onnx_path = tmp_path / "model.onnx"
    export_checkpoint_to_onnx(checkpoint_path, onnx_path)

    evaluator = core.OnnxEvaluator(str(onnx_path), max_batch_size=2)
    batch = core.GumbelSelfPlayBatch(
        game_count=1,
        simulations=2,
        max_considered_actions=2,
        seed=11,
        policy_target_c_visit=5.0,
        policy_target_c_scale=0.25,
    )
    samples: list[tuple[np.ndarray, np.ndarray]] = []

    for _ in range(200):
        request = batch.active_eval_request()
        features = np.frombuffer(request.feature_plane_bytes(), dtype=np.float32).copy()
        results = batch.search_active_with_onnx_evaluator(evaluator, leaf_batch_size=2)
        result = results[0]
        assert result is not None
        policy = np.asarray(result.policy_target(), dtype=np.float32)
        samples.append((features, policy))
        batch.apply_actions([result.selected_action()])
        if batch.is_terminal()[0]:
            break

    assert batch.is_terminal()[0]
    assert samples
    feature_row, policy_row = samples[0]
    assert feature_row.shape == (core.FEATURE_CHANNELS * core.BOARD_CELLS,)
    assert policy_row.shape == (ACTION_SPACE,)
    assert np.isfinite(feature_row).all()
    assert np.isfinite(policy_row).all()
