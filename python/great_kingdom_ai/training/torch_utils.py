"""Small torch import and AMP helpers used by training modules."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any


def _cuda_amp_enabled(
    torch: Any,
    device: Any | str | None,
    *,
    enabled: bool,
) -> bool:
    return bool(
        enabled
        and device is not None
        and str(device).startswith("cuda")
        and torch.cuda.is_available()
    )


def _autocast_context(torch: Any, *, enabled: bool) -> Any:
    if not enabled:
        return nullcontext()
    return torch.amp.autocast("cuda", enabled=True)


def _import_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required for training") from exc
    return torch

__all__ = ["_autocast_context", "_cuda_amp_enabled", "_import_torch"]
