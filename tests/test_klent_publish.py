from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path

import pytest

_torch_spec = importlib.util.find_spec("torch")
_onnx_spec = importlib.util.find_spec("onnx")
_onnxruntime_spec = importlib.util.find_spec("onnxruntime")
pytestmark = pytest.mark.skipif(
    _torch_spec is None or _onnx_spec is None or _onnxruntime_spec is None,
    reason="torch, onnx, and onnxruntime are required for ONNX publication tests",
)
torch = importlib.import_module("torch") if _torch_spec is not None else None

from great_kingdom_ai.klent.checkpoint import (  # noqa: E402
    KlentTrainState,
    save_klent_checkpoint,
)
from great_kingdom_ai.klent.publish import (  # noqa: E402
    KLENT_ONNX_MANIFEST_NAME,
    KLENT_ONNX_POINTER_NAME,
    load_klent_onnx_manifest,
    load_klent_onnx_pointer,
    load_published_klent_onnx,
    publish_klent_onnx_artifacts,
)
from great_kingdom_ai.klent.types import KlentConfig  # noqa: E402
from great_kingdom_ai.model import create_model  # noqa: E402
from great_kingdom_ai.training.checkpoint import create_optimizer  # noqa: E402
from great_kingdom_ai.training.config import TrainingConfig  # noqa: E402


def _save_checkpoint(tmp_path: Path, name: str = "klent.pt") -> Path:
    model = create_model("small_klent")
    optimizer = create_optimizer(
        torch,
        model,
        TrainingConfig(model_preset="small_klent", learning_rate=1e-3, device="cpu"),
    )
    state = KlentTrainState(
        model=model,
        optimizer=optimizer,
        iteration=1,
        total_steps=3,
        klent_config=KlentConfig(),
        model_preset="small_klent",
    )
    return save_klent_checkpoint(state, tmp_path / name)


def _publish(tmp_path: Path, checkpoint: Path, model_version: int = 1):
    return publish_klent_onnx_artifacts(
        checkpoint,
        tmp_path / "work",
        model_version=model_version,
        iteration=model_version - 1,
        klent_config=KlentConfig(),
        model_preset="small_klent",
    )


def test_publish_writes_version_dir_manifest_and_pointer(tmp_path: Path) -> None:
    checkpoint = _save_checkpoint(tmp_path)

    manifest = _publish(tmp_path, checkpoint)
    version_dir = tmp_path / "work" / "onnx" / "version-00001"
    pointer = load_klent_onnx_pointer(tmp_path / "work")

    assert manifest.model_version == 1
    assert manifest.iteration == 0
    assert (version_dir / "actor.onnx").exists()
    assert (version_dir / "eval.onnx").exists()
    assert (version_dir / KLENT_ONNX_MANIFEST_NAME).exists()
    assert (tmp_path / "work" / "onnx" / KLENT_ONNX_POINTER_NAME).exists()
    assert pointer.model_version == 1
    assert pointer.actor_path.endswith("actor.onnx")
    assert pointer.eval_path.endswith("eval.onnx")

    loaded, loaded_manifest = load_published_klent_onnx(tmp_path / "work")
    assert loaded == pointer
    assert loaded_manifest == manifest
    assert loaded_manifest.export_for_kind("eval").output_names == ("policy_logits", "value")
    assert loaded_manifest.export_for_kind("actor").output_names == (
        "policy_logits",
        "value",
        "q_values",
    )
    assert all(record.parity_passed for record in manifest.exports)
    assert loaded_manifest.klent_config == KlentConfig()


def test_publish_requires_new_version_or_overwrite(tmp_path: Path) -> None:
    checkpoint = _save_checkpoint(tmp_path)
    _publish(tmp_path, checkpoint, model_version=1)

    with pytest.raises(FileExistsError, match="already exists"):
        _publish(tmp_path, checkpoint, model_version=1)

    _publish(tmp_path, checkpoint, model_version=2)
    pointer = load_klent_onnx_pointer(tmp_path / "work")
    assert pointer.model_version == 2


def test_publish_pointer_is_last_commit_marker(tmp_path: Path) -> None:
    checkpoint = _save_checkpoint(tmp_path)
    _publish(tmp_path, checkpoint)

    pointer_path = tmp_path / "work" / "onnx" / KLENT_ONNX_POINTER_NAME
    payload = json.loads(pointer_path.read_text(encoding="utf-8"))
    assert payload["model_version"] == 1
    assert not any(path.name.endswith(".tmp") for path in pointer_path.parent.iterdir())


def test_missing_pointer_is_reported(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="pointer is missing"):
        load_klent_onnx_pointer(tmp_path / "work")


def test_manifest_rejects_unknown_algorithm(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "algorithm": "gumbel",
                "schema_version": 1,
                "model_version": 1,
                "iteration": 0,
                "model_preset": "small",
                "checkpoint_path": "",
                "klent_config": {
                    "alpha": 0.03,
                    "beta": 0.1,
                    "lambda_param": 0.8825,
                    "gamma": 1.0,
                },
                "created_at": "",
                "exports": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="algorithm"):
        load_klent_onnx_manifest(manifest_path)