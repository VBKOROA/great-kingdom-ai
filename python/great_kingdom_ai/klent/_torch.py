"""Lazy torch import for the standalone KLENT package."""

from __future__ import annotations

from typing import Any


def _import_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required for KLENT target and loss computation") from exc
    return torch


__all__ = ["_import_torch"]