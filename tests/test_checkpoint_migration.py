from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

import pytest

_torch_spec = importlib.util.find_spec("torch")
pytestmark = pytest.mark.skipif(
    _torch_spec is None,
    reason="torch is not installed",
)
torch = importlib.import_module("torch") if _torch_spec is not None else None

from great_kingdom_ai.checkpoint_migration import (  # noqa: E402
    build_parser,
    migrate_checkpoint_training_state,
)
from great_kingdom_ai.features import BOARD_SIZE, FEATURE_CHANNELS  # noqa: E402
from great_kingdom_ai.train import (  # noqa: E402
    TrainingConfig,
    create_train_state,
    load_checkpoint,
    save_checkpoint,
)


def test_migrate_checkpoint_training_state_keeps_model_and_resets_state(tmp_path: Path) -> None:
    source_config = TrainingConfig(batch_size=2, steps=1, seed=5)
    source_state = create_train_state(source_config)
    source_state = type(source_state)(
        model=source_state.model,
        optimizer=source_state.optimizer,
        scheduler=source_state.scheduler,
        step=17,
        model_preset=source_state.model_preset,
    )
    inputs = torch.zeros(1, FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    source_state.model.eval()
    with torch.no_grad():
        expected_policy, expected_value = source_state.model(inputs)

    source = save_checkpoint(source_state, tmp_path / "source.pt")
    output = tmp_path / "migrated.pt"
    migrated_config = TrainingConfig(
        learning_rate=1e-4,
        lr_schedule="constant_with_warmup",
        lr_warmup_steps=10,
    )

    migrated_path = migrate_checkpoint_training_state(source, output, migrated_config)
    migrated = load_checkpoint(
        migrated_path,
        learning_rate=migrated_config.learning_rate,
        lr_schedule=migrated_config.lr_schedule,
        lr_warmup_steps=migrated_config.lr_warmup_steps,
    )
    migrated.model.eval()
    with torch.no_grad():
        actual_policy, actual_value = migrated.model(inputs)

    assert migrated.step == 0
    assert migrated.optimizer.state_dict()["state"] == {}
    assert migrated.optimizer.param_groups[0]["lr"] == pytest.approx(1e-5)
    assert torch.allclose(actual_policy, expected_policy)
    assert torch.allclose(actual_value, expected_value)


def test_migrate_checkpoint_training_state_rejects_existing_output(tmp_path: Path) -> None:
    source = save_checkpoint(create_train_state(TrainingConfig()), tmp_path / "source.pt")
    output = save_checkpoint(create_train_state(TrainingConfig()), tmp_path / "output.pt")

    with pytest.raises(FileExistsError, match="already exists"):
        migrate_checkpoint_training_state(source, output, TrainingConfig())


def test_migrate_checkpoint_training_state_can_migrate_in_place_with_force(
    tmp_path: Path,
) -> None:
    source = save_checkpoint(create_train_state(TrainingConfig()), tmp_path / "source.pt")

    migrated_path = migrate_checkpoint_training_state(
        source,
        source,
        TrainingConfig(lr_schedule="constant_with_warmup", lr_warmup_steps=5),
        force=True,
    )
    migrated = load_checkpoint(
        migrated_path,
        lr_schedule="constant_with_warmup",
        lr_warmup_steps=5,
    )

    assert migrated_path == source
    assert migrated.step == 0
    assert not (tmp_path / ".source.pt.tmp").exists()


def test_checkpoint_migration_parser_accepts_paths() -> None:
    args = build_parser().parse_args(
        [
            "--source",
            "best.pt",
            "--output",
            "best-migrated.pt",
            "--config",
            "configs/runpod/train.json",
            "--force",
        ]
    )

    assert args.source == Path("best.pt")
    assert args.output == Path("best-migrated.pt")
    assert args.config == Path("configs/runpod/train.json")
    assert args.force is True
