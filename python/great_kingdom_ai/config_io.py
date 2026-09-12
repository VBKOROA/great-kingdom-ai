"""Configuration file loading shared by runtime components."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]


def load_config_object(path: str | Path, label: str) -> dict[str, Any]:
    """Load a YAML config, retaining JSON support for development tests."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        if config_path.suffix.lower() == ".json":
            data = json.load(file)
        elif config_path.suffix.lower() in {".yaml", ".yml"}:
            data = yaml.safe_load(file)
        else:
            raise ValueError(f"{label} must use a .yaml, .yml, or .json extension")
    if not isinstance(data, dict):
        raise ValueError(f"{label} must be a mapping")
    return data


__all__ = ["load_config_object"]
