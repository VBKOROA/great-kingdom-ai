from __future__ import annotations

import json
from pathlib import Path


def test_m6_smoke_config_runs_small_and_medium_on_cuda() -> None:
    config = json.loads(Path("configs/m6-smoke.json").read_text(encoding="utf-8"))

    assert config["device"] == "cuda"
    assert config["require_cuda"] is True
    assert config["batch_size"] > 0
    assert config["repeat"] > 0
    assert config["presets"] == ["small", "medium"]
