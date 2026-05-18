"""Persistence helpers for replay array payloads."""

from __future__ import annotations

import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class NpzArrayWriteStat:
    key: str
    bytes: int
    seconds: float


@dataclass(frozen=True)
class NpzSaveStats:
    array_stats: tuple[NpzArrayWriteStat, ...]
    write_seconds: float
    close_seconds: float
    replace_seconds: float
    total_seconds: float


def save_npz_atomic(
    destination: Path,
    payload: dict[str, Any],
    *,
    compressed: bool,
) -> NpzSaveStats:
    temporary = destination.with_name(f"{destination.name}.tmp")
    started_at = time.monotonic()
    array_stats: list[NpzArrayWriteStat] = []
    compression = zipfile.ZIP_DEFLATED if compressed else zipfile.ZIP_STORED
    try:
        archive = zipfile.ZipFile(temporary, mode="w", compression=compression)
        close_started_at: float | None = None
        try:
            for key, value in payload.items():
                array = np.asanyarray(value)
                array_started_at = time.monotonic()
                with archive.open(f"{key}.npy", mode="w", force_zip64=True) as member:
                    np.lib.format.write_array(member, array, allow_pickle=False)
                array_stats.append(
                    NpzArrayWriteStat(
                        key=key,
                        bytes=int(array.nbytes),
                        seconds=time.monotonic() - array_started_at,
                    )
                )
        finally:
            close_started_at = time.monotonic()
            archive.close()
        close_seconds = time.monotonic() - close_started_at
        write_seconds = time.monotonic() - started_at
        replace_started_at = time.monotonic()
        temporary.replace(destination)
        replace_seconds = time.monotonic() - replace_started_at
        return NpzSaveStats(
            array_stats=tuple(array_stats),
            write_seconds=write_seconds,
            close_seconds=close_seconds,
            replace_seconds=replace_seconds,
            total_seconds=time.monotonic() - started_at,
        )
    finally:
        if temporary.exists():
            temporary.unlink()
