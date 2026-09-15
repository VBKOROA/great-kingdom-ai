from __future__ import annotations

import importlib
import importlib.util
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

_torch_spec = importlib.util.find_spec("torch")
pytestmark = pytest.mark.skipif(
    _torch_spec is None,
    reason="torch is not installed",
)
torch = importlib.import_module("torch") if _torch_spec is not None else None

import great_kingdom_ai.klent.trainer as trainer_module  # noqa: E402
from great_kingdom_ai.features import ACTION_SPACE, BOARD_SIZE, FEATURE_CHANNELS  # noqa: E402
from great_kingdom_ai.klent.checkpoint import (  # noqa: E402
    KlentTrainState,
    load_klent_checkpoint,
    save_klent_checkpoint,
    warm_start_klent_model,
)
from great_kingdom_ai.klent.publish import (  # noqa: E402
    load_klent_onnx_manifest,
    load_klent_onnx_pointer,
)
from great_kingdom_ai.klent.shards import load_klent_shard  # noqa: E402
from great_kingdom_ai.klent.trainer import (  # noqa: E402
    KlentTrainConfig,
    _actor_onnx_for_iteration,
    _collect_with_python_actor,
    fit_klent_model,
    run_klent_training,
)
from great_kingdom_ai.klent.types import KlentConfig  # noqa: E402
from great_kingdom_ai.model import create_model  # noqa: E402
from great_kingdom_ai.replay.dataset import TrajectoryArrayBatch  # noqa: E402
from great_kingdom_ai.training.checkpoint import create_optimizer  # noqa: E402
from great_kingdom_ai.training.config import TrainingConfig  # noqa: E402

_rust_core_available = importlib.util.find_spec("great_kingdom_core") is not None
requires_core = pytest.mark.skipif(
    not _rust_core_available,
    reason="great_kingdom_core extension is not installed",
)
_onnx_ready = (
    importlib.util.find_spec("onnx") is not None
    and importlib.util.find_spec("onnxruntime") is not None
)
requires_onnx = pytest.mark.skipif(
    not _onnx_ready,
    reason="onnx and onnxruntime are required for ONNX publication",
)


def make_config(tmp_path: Path, **overrides: object) -> KlentTrainConfig:
    defaults: dict[str, object] = {
        "work_dir": tmp_path / "klent",
        "model_preset": "small_klent",
        "device": "cpu",
        "seed": 0,
        "min_transitions": 1,
        "max_games_per_iteration": 64,
        "fit_epochs": 2,
        "batch_size": 4,
        "max_turns": 200,
        "symmetry_augmentation": False,
        "learning_rate": 1e-2,
        "export_onnx": False,
    }
    return KlentTrainConfig(**{**defaults, **overrides})  # type: ignore[arg-type]


@requires_core
def test_run_klent_training_publishes_iteration_checkpoints_and_shards(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)

    summaries = run_klent_training(config, iterations=2)

    assert [summary.iteration for summary in summaries] == [0, 1]
    assert all(summary.transitions >= config.min_transitions for summary in summaries)
    assert all(summary.epoch_losses for summary in summaries)
    latest = load_klent_checkpoint(config.work_dir / "checkpoints" / "latest.pt")
    assert latest.iteration == 2

    for iteration in (0, 1):
        shard_path = config.work_dir / "iterations" / f"iteration-{iteration:04d}.npz"
        checkpoint_path = (
            config.work_dir / "checkpoints" / f"iteration-{iteration + 1:04d}.pt"
        )
        assert shard_path.exists()
        assert checkpoint_path.exists()
        store, metadata = load_klent_shard(shard_path)
        assert metadata.iteration == iteration
        assert metadata.model_version == iteration
        assert metadata.transitions == len(store)
        assert store.model_versions.tolist() == [iteration] * len(store)
        assert store.lambda_returns_present is not None
        assert bool(store.lambda_returns_present.all())


@requires_core
def test_run_klent_training_resumes_from_latest_checkpoint(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    run_klent_training(config, iterations=1)

    resumed = run_klent_training(config, iterations=2)

    assert [summary.iteration for summary in resumed] == [1]
    latest = load_klent_checkpoint(config.work_dir / "checkpoints" / "latest.pt")
    assert latest.iteration == 2
    assert latest.total_steps > 0
    assert latest.last_shard is not None


@requires_core
def test_fresh_run_does_not_adopt_previous_run_checkpoints(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    run_klent_training(config, iterations=2)
    latest_path = config.work_dir / "checkpoints" / "latest.pt"

    run_klent_training(config, iterations=1, resume=False)
    fresh = load_klent_checkpoint(latest_path)
    assert fresh.iteration == 1

    resumed = run_klent_training(config, iterations=2)

    assert [summary.iteration for summary in resumed] == [1]
    trained = load_klent_checkpoint(latest_path)
    assert trained.iteration == 2
    assert trained.total_steps > fresh.total_steps


@requires_core
def test_resume_skips_corrupt_iteration_checkpoint(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    run_klent_training(config, iterations=1)
    corrupt = config.work_dir / "checkpoints" / "iteration-0002.pt"
    corrupt.write_bytes(b"truncated checkpoint")

    resumed = run_klent_training(config, iterations=2)

    assert [summary.iteration for summary in resumed] == [1]
    repaired = load_klent_checkpoint(corrupt)
    assert repaired.iteration == 2


@requires_core
def test_resume_falls_back_to_latest_checkpoint(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    run_klent_training(config, iterations=1)
    (config.work_dir / "checkpoints" / "iteration-0001.pt").unlink()

    resumed = run_klent_training(config, iterations=2)

    assert [summary.iteration for summary in resumed] == [1]
    latest = load_klent_checkpoint(config.work_dir / "checkpoints" / "latest.pt")
    assert latest.iteration == 2


@requires_core
@requires_onnx
def test_resume_republishes_from_latest_checkpoint(tmp_path: Path) -> None:
    config = make_config(tmp_path, export_onnx=True, onnx_precision="fp32")
    run_klent_training(config, iterations=1)
    (config.work_dir / "checkpoints" / "iteration-0001.pt").unlink()
    (config.work_dir / "onnx" / "current.json").unlink()

    resumed = run_klent_training(config, iterations=1)

    assert resumed == []
    pointer = load_klent_onnx_pointer(config.work_dir)
    assert pointer.model_version == 1


@requires_core
def test_run_klent_training_resume_rejects_config_mismatch(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    run_klent_training(config, iterations=1)

    mismatched = make_config(tmp_path, klent=KlentConfig(alpha=0.5))

    with pytest.raises(ValueError, match="does not match"):
        run_klent_training(mismatched, iterations=2)


@requires_core
def test_klent_fit_reduces_loss_over_epochs(tmp_path: Path) -> None:
    config = make_config(
        tmp_path,
        fit_epochs=10,
        min_transitions=8,
        max_games_per_iteration=64,
        learning_rate=1e-3,
    )
    rng_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(20260522)
        summaries = run_klent_training(config, iterations=1)
    finally:
        torch.random.set_rng_state(rng_state)

    epoch_losses = summaries[0].epoch_losses
    assert len(epoch_losses) == 10
    assert min(epoch_losses[1:]) < epoch_losses[0]


@requires_core
@requires_onnx
def test_run_klent_training_publishes_onnx_pointer(tmp_path: Path) -> None:
    config = make_config(tmp_path, export_onnx=True, onnx_precision="fp32")

    summaries = run_klent_training(config, iterations=1)

    pointer = load_klent_onnx_pointer(config.work_dir)
    manifest = load_klent_onnx_manifest(pointer.manifest_path)
    assert pointer.model_version == 1
    assert summaries[0].onnx_version_dir is not None
    assert Path(pointer.actor_path).exists()
    assert Path(pointer.eval_path).exists()
    assert {record.kind for record in manifest.exports} == {"eval", "actor"}
    assert all(record.parity_passed for record in manifest.exports)


@requires_core
@requires_onnx
def test_run_klent_training_with_rust_actor_two_iterations(tmp_path: Path) -> None:
    config = make_config(
        tmp_path,
        use_rust_actor=True,
        export_onnx=False,
        min_transitions=1,
        max_games_per_iteration=1,
        rust_self_play_batch_size=1,
        fit_epochs=1,
        batch_size=8,
    )

    summaries = run_klent_training(config, iterations=2)

    assert [summary.iteration for summary in summaries] == [0, 1]
    for summary in summaries:
        store, metadata = load_klent_shard(summary.shard_path)
        assert metadata.iteration == summary.iteration
        assert metadata.model_version == summary.iteration
        assert store.lambda_returns_present is not None
        assert bool(store.lambda_returns_present.all())
        assert store.model_versions.tolist() == [summary.iteration] * len(store)
    latest = load_klent_checkpoint(config.work_dir / "checkpoints" / "latest.pt")
    assert latest.iteration == 2


def test_load_klent_checkpoint_rejects_state_value_checkpoint(tmp_path: Path) -> None:
    from great_kingdom_ai.klent.shards import KLENT_ALGORITHM
    from great_kingdom_ai.model import ModelConfig

    path = tmp_path / "state-value.pt"
    torch.save(
        {
            "algorithm": KLENT_ALGORITHM,
            "iteration": 1,
            "total_steps": 0,
            "model_preset": "small",
            "model_config": asdict(ModelConfig()),
            "klent_config": asdict(KlentConfig()),
            "model_state": {},
            "optimizer": "adamw",
            "optimizer_state": {},
            "last_shard": None,
        },
        path,
    )

    with pytest.raises(ValueError, match="action_value_head"):
        load_klent_checkpoint(path)


def test_load_klent_checkpoint_rejects_foreign_algorithm(tmp_path: Path) -> None:
    path = tmp_path / "gumbel.pt"
    torch.save({"algorithm": "gumbel"}, path)

    with pytest.raises(ValueError, match="algorithm"):
        load_klent_checkpoint(path)


def test_save_klent_checkpoint_keeps_previous_file_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = create_model("small_klent")
    optimizer = create_optimizer(
        torch,
        model,
        TrainingConfig(model_preset="small_klent", learning_rate=1e-2, device="cpu"),
    )
    state = KlentTrainState(
        model=model,
        optimizer=optimizer,
        iteration=1,
        total_steps=0,
        klent_config=KlentConfig(),
        model_preset="small_klent",
    )
    path = tmp_path / "iteration-0001.pt"
    save_klent_checkpoint(state, path)

    def failing_save(obj: object, f: object, *args: object, **kwargs: object) -> None:
        Path(f).write_bytes(b"partial checkpoint")  # type: ignore[arg-type]
        raise RuntimeError("disk full")

    monkeypatch.setattr(torch, "save", failing_save)

    with pytest.raises(RuntimeError, match="disk full"):
        save_klent_checkpoint(state, path)

    assert not (tmp_path / ".iteration-0001.pt.tmp").exists()
    assert load_klent_checkpoint(path).iteration == 1


def test_warm_start_klent_model_copies_backbone_and_initializes_q_head(
    tmp_path: Path,
) -> None:
    from great_kingdom_ai.features import BOARD_SIZE, FEATURE_CHANNELS
    from great_kingdom_ai.model import create_model

    source = create_model("small")
    path = tmp_path / "source.pt"
    torch.save(
        {
            "model_config": asdict(source.config),
            "model_state": source.state_dict(),
        },
        path,
    )

    warm = warm_start_klent_model(path, model_preset="small")
    warm.eval()

    assert warm.has_action_value_head
    source_state = source.state_dict()
    warm_state = warm.state_dict()
    assert torch.equal(source_state["stem.0.weight"], warm_state["stem.0.weight"])
    assert torch.equal(source_state["policy_pass.2.weight"], warm_state["policy_pass.2.weight"])

    with torch.no_grad():
        logits, q_values = warm.forward_q(
            torch.zeros((2, FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE))
        )
    assert logits.shape[0] == 2
    assert q_values.shape[0] == 2


class _FixedDataset:
    """Minimal KLENT dataset stand-in for direct fit tests."""

    def __init__(self, size: int) -> None:
        rng = np.random.default_rng(0)
        self._features = rng.normal(size=(size, FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE))
        self._features = self._features.astype(np.float32)
        self._features[:, 4, :, :] = 1.0
        self._policies = np.full((size, ACTION_SPACE), 1.0 / ACTION_SPACE, dtype=np.float32)
        self._values = np.linspace(-1.0, 1.0, size).astype(np.float32)
        self._weights = np.ones((size,), dtype=np.float32)
        self._masks = np.ones((size, ACTION_SPACE), dtype=np.bool_)
        self._actions = np.arange(size, dtype=np.int64) % ACTION_SPACE

    def __len__(self) -> int:
        return self._features.shape[0]

    def arrays_for_indexes(self, indexes: np.ndarray) -> TrajectoryArrayBatch:
        return TrajectoryArrayBatch(
            indexes=np.asarray(indexes, dtype=np.int64),
            features=self._features[indexes],
            policies=self._policies[indexes],
            values=self._values[indexes],
            sample_weights=self._weights[indexes],
            legal_masks=self._masks[indexes],
            actions=self._actions[indexes],
        )


def test_fit_klent_model_applies_amp_autocast_when_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from contextlib import nullcontext

    recorded: list[bool] = []
    monkeypatch.setattr(
        trainer_module,
        "_cuda_amp_enabled",
        lambda torch_, device, enabled: bool(enabled),
    )

    def fake_autocast(torch_: object, *, enabled: bool) -> object:
        recorded.append(enabled)
        return nullcontext()

    monkeypatch.setattr(trainer_module, "_autocast_context", fake_autocast)
    model = create_model("small_klent")
    optimizer = create_optimizer(
        torch,
        model,
        TrainingConfig(model_preset="small_klent", learning_rate=1e-2, device="cpu"),
    )
    dataset = _FixedDataset(4)
    config = make_config(tmp_path, amp=True, batch_size=2, fit_epochs=1)

    losses, steps = fit_klent_model(
        model,
        dataset,  # type: ignore[arg-type]
        config,
        optimizer,
        start_steps=0,
        iteration=0,
    )

    assert recorded == [True, True]
    assert steps == 2
    assert all(np.isfinite(loss) for loss in losses)


@requires_core
def test_collection_fails_when_min_transitions_is_not_reached(tmp_path: Path) -> None:
    model = create_model("small_klent")
    config = make_config(
        tmp_path,
        min_transitions=10**6,
        max_games_per_iteration=1,
        fit_epochs=1,
    )

    with pytest.raises(RuntimeError, match="min_transitions"):
        _collect_with_python_actor(model, config, 0)


@requires_core
@requires_onnx
def test_rust_collection_fails_when_min_transitions_is_not_reached(tmp_path: Path) -> None:
    from great_kingdom_ai.klent.export import export_klent_checkpoint_to_onnx
    from great_kingdom_ai.klent.trainer import _collect_with_rust_actor

    model = create_model("small_klent")
    optimizer = create_optimizer(
        torch,
        model,
        TrainingConfig(model_preset="small_klent", learning_rate=1e-2, device="cpu"),
    )
    state = KlentTrainState(
        model=model,
        optimizer=optimizer,
        iteration=0,
        total_steps=0,
        klent_config=KlentConfig(),
        model_preset="small_klent",
    )
    checkpoint = trainer_module.save_klent_checkpoint(state, tmp_path / "actor-source.pt")
    actor_path = tmp_path / "actor.onnx"
    export_klent_checkpoint_to_onnx(checkpoint, actor_path, kind="actor")
    config = make_config(
        tmp_path,
        actor_onnx_path=actor_path,
        min_transitions=10**6,
        max_games_per_iteration=1,
        rust_self_play_batch_size=1,
        fit_epochs=1,
    )

    with pytest.raises(RuntimeError, match="min_transitions"):
        _collect_with_rust_actor(state, config, 0)


@requires_onnx
def test_actor_path_override_only_applies_to_first_iteration(tmp_path: Path) -> None:
    from great_kingdom_ai.klent.export import export_klent_checkpoint_to_onnx

    model = create_model("small_klent")
    optimizer = create_optimizer(
        torch,
        model,
        TrainingConfig(model_preset="small_klent", learning_rate=1e-2, device="cpu"),
    )
    state = KlentTrainState(
        model=model,
        optimizer=optimizer,
        iteration=0,
        total_steps=0,
        klent_config=KlentConfig(),
        model_preset="small_klent",
    )
    provided_actor = tmp_path / "provided-actor.onnx"
    source = save_klent_checkpoint(state, tmp_path / "provided-source.pt")
    export_klent_checkpoint_to_onnx(source, provided_actor, kind="actor")
    config = make_config(tmp_path, actor_onnx_path=provided_actor)
    (config.work_dir / "onnx").mkdir(parents=True, exist_ok=True)
    exported: list[Path] = []

    def fake_export(checkpoint: object, output: Path, **kwargs: object) -> None:
        exported.append(Path(output))
        Path(output).write_bytes(b"placeholder")

    first = _actor_onnx_for_iteration(state, config, 0, fake_export)
    second = _actor_onnx_for_iteration(state, config, 1, fake_export)

    assert first == provided_actor
    assert second.name == "actor-source-0001.onnx"
    assert second != provided_actor
    assert exported == [second]


@requires_onnx
def test_actor_path_override_mismatch_is_rejected(tmp_path: Path) -> None:
    from great_kingdom_ai.klent.export import export_klent_checkpoint_to_onnx

    model = create_model("small_klent")
    optimizer = create_optimizer(
        torch,
        model,
        TrainingConfig(model_preset="small_klent", learning_rate=1e-2, device="cpu"),
    )
    state = KlentTrainState(
        model=model,
        optimizer=optimizer,
        iteration=0,
        total_steps=0,
        klent_config=KlentConfig(),
        model_preset="small_klent",
    )
    other = create_model("small_klent")
    other_optimizer = create_optimizer(
        torch,
        other,
        TrainingConfig(model_preset="small_klent", learning_rate=1e-2, device="cpu"),
    )
    other_state = KlentTrainState(
        model=other,
        optimizer=other_optimizer,
        iteration=0,
        total_steps=0,
        klent_config=KlentConfig(),
        model_preset="small_klent",
    )
    other_checkpoint = save_klent_checkpoint(other_state, tmp_path / "other.pt")
    mismatched_actor = tmp_path / "mismatched-actor.onnx"
    export_klent_checkpoint_to_onnx(other_checkpoint, mismatched_actor, kind="actor")
    config = make_config(tmp_path, actor_onnx_path=mismatched_actor)
    exported: list[Path] = []

    def fake_export(checkpoint: object, output: Path, **kwargs: object) -> None:
        exported.append(Path(output))

    with pytest.raises(RuntimeError, match="actor_onnx_path"):
        _actor_onnx_for_iteration(state, config, 0, fake_export)

    assert exported == []


@requires_core
@requires_onnx
def test_resume_republishes_after_export_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_config(tmp_path, export_onnx=True, onnx_precision="fp32")
    real_publish = trainer_module.publish_klent_onnx_artifacts
    calls = {"count": 0}

    def flaky_publish(*args: object, **kwargs: object) -> object:
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("injected export failure")
        return real_publish(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(trainer_module, "publish_klent_onnx_artifacts", flaky_publish)

    with pytest.raises(RuntimeError, match="injected export failure"):
        run_klent_training(config, iterations=1)

    latest = config.work_dir / "checkpoints" / "latest.pt"
    assert not latest.exists()
    with pytest.raises(ValueError, match="pointer is missing"):
        load_klent_onnx_pointer(config.work_dir)

    resumed = run_klent_training(config, iterations=1)

    assert resumed == []
    pointer = load_klent_onnx_pointer(config.work_dir)
    assert pointer.model_version == 1
    assert latest.exists()
    assert calls["count"] == 2