from __future__ import annotations

import importlib.util
from dataclasses import asdict
from pathlib import Path

import pytest

_torch_spec = importlib.util.find_spec("torch")
pytestmark = pytest.mark.skipif(
    _torch_spec is None,
    reason="torch is not installed",
)
torch = importlib.import_module("torch") if _torch_spec is not None else None

from great_kingdom_ai.klent.checkpoint import (  # noqa: E402
    load_klent_checkpoint,
    warm_start_klent_model,
)
from great_kingdom_ai.klent.shards import load_klent_shard  # noqa: E402
from great_kingdom_ai.klent.trainer import (  # noqa: E402
    KlentTrainConfig,
    run_klent_training,
)
from great_kingdom_ai.klent.types import KlentConfig  # noqa: E402

_rust_core_available = importlib.util.find_spec("great_kingdom_core") is not None
requires_core = pytest.mark.skipif(
    not _rust_core_available,
    reason="great_kingdom_core extension is not installed",
)


def make_config(tmp_path: Path, **overrides: object) -> KlentTrainConfig:
    defaults: dict[str, object] = {
        "work_dir": tmp_path / "klent",
        "model_preset": "small_klent",
        "device": "cpu",
        "seed": 0,
        "min_transitions": 4,
        "max_games_per_iteration": 2,
        "fit_epochs": 2,
        "batch_size": 4,
        "max_turns": 200,
        "symmetry_augmentation": False,
        "learning_rate": 1e-2,
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
        fit_epochs=8,
        min_transitions=8,
        max_games_per_iteration=3,
        learning_rate=5e-3,
    )

    summaries = run_klent_training(config, iterations=1)

    epoch_losses = summaries[0].epoch_losses
    assert len(epoch_losses) == 8
    assert epoch_losses[-1] < epoch_losses[0]


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