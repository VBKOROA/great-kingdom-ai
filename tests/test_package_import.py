from __future__ import annotations

import tomllib
from pathlib import Path

from great_kingdom_ai import __version__


def test_package_imports() -> None:
    assert __version__ == "0.1.0"


def test_console_scripts_are_current_entrypoints() -> None:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text())

    assert pyproject["project"]["scripts"] == {
        "great-kingdom-play": "great_kingdom_ai.cli:main",
        "great-kingdom-train": "great_kingdom_ai.training.cli:main",
        "great-kingdom-single-batch-overfit": "great_kingdom_ai.single_batch_overfit:main",
        "great-kingdom-evaluate": "great_kingdom_ai.evaluate:main",
        "great-kingdom-export-onnx": "great_kingdom_ai.onnx_export:main",
        "great-kingdom-onnx-value-trace": "great_kingdom_ai.onnx_value_trace:main",
        "great-kingdom-actor-v2": "great_kingdom_ai.async_v2.cli:actor_v2_main",
        "great-kingdom-learner-v2": "great_kingdom_ai.async_v2.cli:learner_v2_main",
        "great-kingdom-init-async-v2": "great_kingdom_ai.async_v2.cli:factory_init_v2_main",
        "great-kingdom-replay-monitor-v2": "great_kingdom_ai.replay_monitor:main",
    }
