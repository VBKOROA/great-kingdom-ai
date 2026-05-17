"""Persistence helpers for replay array payloads."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def save_npz_atomic(
    destination: Path,
    payload: dict[str, Any],
    *,
    compressed: bool,
) -> None:
    temporary = destination.with_name(f"{destination.name}.tmp")
    save = np.savez_compressed if compressed else np.savez
    try:
        with temporary.open("wb") as file:
            save(file, **payload)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
