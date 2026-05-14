from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from great_kingdom_ai.train import TrainingConfig

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "compare_ema_decay_branches.py"
SPEC = importlib.util.spec_from_file_location("compare_ema_decay_branches", SCRIPT_PATH)
assert SPEC is not None
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_safe_decay_id_is_path_friendly() -> None:
    assert module._safe_decay_id(0.995) == "0p995"
    assert module._safe_decay_id(-0.1) == "m0p1"


def test_branch_train_config_overrides_selected_fields() -> None:
    config = module._branch_train_config(
        TrainingConfig(
            steps=10,
            ema_decay=0.995,
            device="cpu",
            seed=1,
            learning_rate=1e-4,
            recent_sample_fraction=0.35,
            recent_sample_window=64,
        ),
        steps=3,
        ema_decay=0.99,
        device="cuda",
        seed=5,
        learning_rate=5e-5,
        recent_sample_fraction=0.15,
        recent_sample_window=32,
    )

    assert config.steps == 3
    assert config.ema_decay == pytest.approx(0.99)
    assert config.device == "cuda"
    assert config.seed == 5
    assert config.learning_rate == pytest.approx(5e-5)
    assert config.recent_sample_fraction == pytest.approx(0.15)
    assert config.recent_sample_window == 32


def test_extract_probe_metric_summary_keeps_key_direction_metrics() -> None:
    payload = {
        "probes": [
            {
                "name": "all",
                "rows": 128,
                "target_kl_delta_mean": -0.1,
                "value_mse_delta": 0.2,
                "before_to_after_kl": {"mean": 0.3},
                "top1_flip_rate": 0.4,
                "value_prediction_delta_abs": {"mean": 0.5},
            }
        ]
    }

    summary = module._extract_probe_metric_summary(payload)

    assert summary == {
        "all": {
            "rows": 128,
            "target_kl_delta_mean": pytest.approx(-0.1),
            "value_mse_delta": pytest.approx(0.2),
            "before_to_after_kl_mean": pytest.approx(0.3),
            "top1_flip_rate": pytest.approx(0.4),
            "value_prediction_delta_abs_mean": pytest.approx(0.5),
        }
    }


def test_parser_defaults_include_current_branch_decays() -> None:
    args = module.build_parser().parse_args([])

    assert args.ema_decays == [0.99, 0.995]
    assert args.ema_init == "raw"
    assert args.steps == 256
