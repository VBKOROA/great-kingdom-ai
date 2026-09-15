"""KLENT training configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from great_kingdom_ai.config_io import load_config_object
from great_kingdom_ai.klent.trainer import KlentTrainConfig
from great_kingdom_ai.klent.types import KlentConfig

_PATH_FIELDS = ("work_dir", "warm_start_checkpoint", "actor_onnx_path")


def load_klent_train_config(path: str | Path) -> KlentTrainConfig:
    """Load a ``KlentTrainConfig`` from YAML or JSON."""
    data = load_config_object(path, "KLENT train config")
    klent_data = data.pop("klent", None)
    if klent_data is not None:
        if not isinstance(klent_data, dict):
            raise ValueError("klent must be a mapping of alpha/beta/lambda_param/gamma")
        data["klent"] = KlentConfig(**klent_data)
    for key in _PATH_FIELDS:
        value: Any = data.get(key)
        if value is not None:
            data[key] = Path(value)
    try:
        return KlentTrainConfig(**data)
    except TypeError as exc:
        raise ValueError(f"invalid KLENT train config: {exc}") from exc


__all__ = ["load_klent_train_config"]