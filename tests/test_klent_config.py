from __future__ import annotations

from pathlib import Path

import pytest
from great_kingdom_ai.klent.cli import build_parser
from great_kingdom_ai.klent.config import load_klent_train_config
from great_kingdom_ai.klent.types import KlentConfig

CONFIG_PATH = Path("configs/runpod/klent-train.yaml")


def test_runpod_klent_config_loads_plan_defaults() -> None:
    config = load_klent_train_config(CONFIG_PATH)

    assert config.work_dir == Path("data/runpod/klent-strong-attn")
    assert config.model_preset == "strong_attn_klent"
    assert config.klent == KlentConfig()
    assert config.fit_epochs == 1
    assert config.use_rust_actor is True
    assert config.export_onnx is True
    assert config.amp is True
    assert config.onnx_precision == "fp16"
    assert config.check_onnx_parity is False


def test_load_klent_train_config_converts_klent_mapping_and_paths(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "klent.yaml"
    config_path.write_text(
        "\n".join(
            [
                "work_dir: data/experiments/klent",
                "klent:",
                "  alpha: 0.5",
                "  beta: 0.25",
                "  lambda_param: 0.9",
                "  gamma: 0.99",
                "actor_onnx_path: models/actor.onnx",
                "min_transitions: 8",
                "max_games_per_iteration: 2",
                "fit_epochs: 3",
                "batch_size: 4",
            ]
        ),
        encoding="utf-8",
    )

    config = load_klent_train_config(config_path)

    assert config.work_dir == Path("data/experiments/klent")
    assert config.actor_onnx_path == Path("models/actor.onnx")
    assert config.klent == KlentConfig(
        alpha=0.5,
        beta=0.25,
        lambda_param=0.9,
        gamma=0.99,
    )
    assert config.fit_epochs == 3


def test_load_klent_train_config_rejects_bad_klent_mapping(tmp_path: Path) -> None:
    config_path = tmp_path / "klent.yaml"
    config_path.write_text("klent: [0.03, 0.1]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="klent must be a mapping"):
        load_klent_train_config(config_path)


def test_load_klent_train_config_rejects_unknown_keys(tmp_path: Path) -> None:
    config_path = tmp_path / "klent.yaml"
    config_path.write_text("unknown_key: 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid KLENT train config"):
        load_klent_train_config(config_path)


def test_klent_cli_parser_requires_config_and_defaults_to_resume() -> None:
    parser = build_parser()
    args = parser.parse_args(["--config", "configs/runpod/klent-train.yaml"])

    assert args.iterations == 1
    assert args.no_resume is False

    with pytest.raises(SystemExit):
        parser.parse_args([])