from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

_torch_spec = importlib.util.find_spec("torch")
_onnx_spec = importlib.util.find_spec("onnx")
_onnxruntime_spec = importlib.util.find_spec("onnxruntime")
pytestmark = pytest.mark.skipif(
    _torch_spec is None or _onnx_spec is None or _onnxruntime_spec is None,
    reason="torch, onnx, and onnxruntime are required for ONNX export tests",
)
torch = importlib.import_module("torch") if _torch_spec is not None else None
onnx = importlib.import_module("onnx") if _onnx_spec is not None else None

from great_kingdom_ai.features import (  # noqa: E402
    ACTION_SPACE,
    BOARD_SIZE,
    FEATURE_CHANNELS,
    LEGAL_PLACE_FEATURE_CHANNEL,
)
from great_kingdom_ai.klent.checkpoint import (  # noqa: E402
    KlentTrainState,
    save_klent_checkpoint,
)
from great_kingdom_ai.klent.export import (  # noqa: E402
    KLENT_ACTOR_OUTPUT_NAMES,
    KLENT_EVAL_OUTPUT_NAMES,
    KlentActorExportWrapper,
    KlentGumbelExportWrapper,
    compare_klent_checkpoint_to_onnx,
    export_klent_checkpoint_to_onnx,
    onnx_output_names,
)
from great_kingdom_ai.klent.targets import (  # noqa: E402
    legal_mask_from_features,
    masked_state_value,
)
from great_kingdom_ai.klent.types import KlentConfig  # noqa: E402
from great_kingdom_ai.model import create_model  # noqa: E402
from great_kingdom_ai.training.checkpoint import create_optimizer  # noqa: E402
from great_kingdom_ai.training.config import TrainingConfig  # noqa: E402

_rust_core_available = importlib.util.find_spec("great_kingdom_core") is not None
requires_core = pytest.mark.skipif(
    not _rust_core_available,
    reason="great_kingdom_core extension is not installed",
)


def _save_klent_checkpoint(tmp_path: Path, preset: str = "small_klent") -> Path:
    model = create_model(preset)
    optimizer = create_optimizer(
        torch,
        model,
        TrainingConfig(model_preset=preset, learning_rate=1e-3, device="cpu"),
    )
    state = KlentTrainState(
        model=model,
        optimizer=optimizer,
        iteration=1,
        total_steps=7,
        klent_config=KlentConfig(),
        model_preset=preset,
    )
    return save_klent_checkpoint(state, tmp_path / "klent-checkpoint.pt")


class _FakeQModel(torch.nn.Module):
    def __init__(self, logits: torch.Tensor, q_values: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("_logits", logits.reshape(-1).clone())
        self.register_buffer("_q_values", q_values.reshape(-1).clone())

    def forward_q(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = features.shape[0]
        return (
            self._logits.unsqueeze(0).expand(batch, -1).clone(),
            self._q_values.unsqueeze(0).expand(batch, -1).clone(),
        )


def test_export_eval_onnx_keeps_two_output_contract(tmp_path: Path) -> None:
    checkpoint = _save_klent_checkpoint(tmp_path)
    onnx_path = tmp_path / "eval.onnx"

    summary = export_klent_checkpoint_to_onnx(checkpoint, onnx_path, kind="eval")

    assert summary.kind == "eval"
    assert summary.output_names == KLENT_EVAL_OUTPUT_NAMES
    model = onnx.load(str(onnx_path))
    output_names = [output.name for output in model.graph.output]
    assert output_names == list(KLENT_EVAL_OUTPUT_NAMES)
    value_type = model.graph.output[1].type.tensor_type
    assert len(value_type.shape.dim) == 1
    assert value_type.shape.dim[0].dim_param == "batch"
    assert onnx_output_names(onnx_path) == list(KLENT_EVAL_OUTPUT_NAMES)


def test_export_actor_onnx_keeps_three_output_contract(tmp_path: Path) -> None:
    checkpoint = _save_klent_checkpoint(tmp_path)
    onnx_path = tmp_path / "actor.onnx"

    summary = export_klent_checkpoint_to_onnx(checkpoint, onnx_path, kind="actor")

    assert summary.output_names == KLENT_ACTOR_OUTPUT_NAMES
    assert onnx_output_names(onnx_path) == list(KLENT_ACTOR_OUTPUT_NAMES)


@pytest.mark.parametrize("batch_size", [1, 5])
def test_klent_onnx_outputs_are_finite_for_multiple_batches(
    tmp_path: Path,
    batch_size: int,
) -> None:
    checkpoint = _save_klent_checkpoint(tmp_path)
    onnx_path = tmp_path / "actor.onnx"
    export_klent_checkpoint_to_onnx(checkpoint, onnx_path, kind="actor")

    ort = importlib.import_module("onnxruntime")
    rng = np.random.default_rng(batch_size)
    features = rng.normal(
        size=(batch_size, FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE),
    ).astype(np.float32)
    if batch_size > 1:
        features[0] = 0.0

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    policy, value, q_values = session.run(None, {"features": features})

    assert policy.shape == (batch_size, ACTION_SPACE)
    assert value.shape == (batch_size,)
    assert q_values.shape == (batch_size, ACTION_SPACE)
    assert np.isfinite(policy).all()
    assert np.isfinite(value).all()
    assert np.isfinite(q_values).all()
    assert np.all(np.abs(q_values) <= 1.0 + 1e-5)
    assert np.all(np.abs(value) <= 1.0 + 1e-5)


def test_klent_eval_onnx_matches_pytorch_parity(tmp_path: Path) -> None:
    checkpoint = _save_klent_checkpoint(tmp_path)
    onnx_path = tmp_path / "eval.onnx"
    export_klent_checkpoint_to_onnx(checkpoint, onnx_path, kind="eval")

    parity = compare_klent_checkpoint_to_onnx(checkpoint, onnx_path, kind="eval")

    assert parity.passed
    assert parity.value_shape == (parity.policy_shape[0],)
    assert parity.q_values_shape is None


def test_klent_actor_onnx_matches_pytorch_parity(tmp_path: Path) -> None:
    checkpoint = _save_klent_checkpoint(tmp_path)
    onnx_path = tmp_path / "actor.onnx"
    export_klent_checkpoint_to_onnx(checkpoint, onnx_path, kind="actor")

    parity = compare_klent_checkpoint_to_onnx(
        checkpoint,
        onnx_path,
        kind="actor",
        batch_size=4,
    )

    assert parity.passed
    assert parity.q_values_shape is not None
    assert parity.q_values_shape[1] == ACTION_SPACE


def test_exported_eval_value_ignores_illegal_q_values(tmp_path: Path) -> None:
    logits = torch.zeros((1, ACTION_SPACE), dtype=torch.float32)
    logits[0, 0] = 1.0
    logits[0, 1] = 5.0
    logits[0, 2] = -1.0
    logits[0, 3] = 0.5
    logits[0, 4] = 2.0
    base_q = torch.zeros((1, ACTION_SPACE), dtype=torch.float32)
    base_q[0, 0] = 0.2
    base_q[0, 1] = 0.9
    base_q[0, 2] = -0.4
    base_q[0, 3] = 0.1
    base_q[0, 4] = 0.3
    extreme_q = base_q.clone()
    extreme_q[0, 1] = 1e6
    extreme_q[0, 2] = -1e6
    features = torch.zeros((1, FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=torch.float32)
    features[0, LEGAL_PLACE_FEATURE_CHANNEL, 0, 0] = 1.0
    features[0, LEGAL_PLACE_FEATURE_CHANNEL, 0, 3] = 1.0
    features[0, LEGAL_PLACE_FEATURE_CHANNEL, 0, 4] = 1.0

    base_value = KlentGumbelExportWrapper(_FakeQModel(logits, base_q))(features)[1]
    extreme_value = KlentGumbelExportWrapper(_FakeQModel(logits, extreme_q))(features)[1]

    legal_mask = legal_mask_from_features(features)
    expected = masked_state_value(logits, base_q, legal_mask)
    assert torch.allclose(base_value, expected, atol=1e-6)
    assert torch.allclose(base_value, extreme_value, atol=1e-6)


def test_actor_wrapper_returns_q_values_for_rust_pi_prime(tmp_path: Path) -> None:
    logits = torch.zeros((1, ACTION_SPACE), dtype=torch.float32)
    q_values = torch.linspace(-1.0, 1.0, ACTION_SPACE).unsqueeze(0)
    features = torch.zeros((2, FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=torch.float32)

    policy, value, exported_q = KlentActorExportWrapper(_FakeQModel(logits, q_values))(features)

    assert policy.shape == (2, ACTION_SPACE)
    assert value.shape == (2,)
    assert exported_q.shape == (2, ACTION_SPACE)
    assert torch.allclose(exported_q, q_values.expand(2, -1))


@requires_core
def test_rust_onnx_evaluator_loads_exported_eval_model(tmp_path: Path) -> None:
    import great_kingdom_core as core  # type: ignore[import-untyped]

    checkpoint = _save_klent_checkpoint(tmp_path)
    onnx_path = tmp_path / "eval.onnx"
    export_klent_checkpoint_to_onnx(checkpoint, onnx_path, kind="eval")

    evaluator = core.OnnxEvaluator(str(onnx_path), device="cpu", max_batch_size=8)

    assert evaluator is not None