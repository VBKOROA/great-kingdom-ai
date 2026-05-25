"""Persistence helpers for replay array payloads."""

from __future__ import annotations

import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

COPY_BUFFER_BYTES = 64 * 1024 * 1024


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
    copy_seconds: float
    replace_seconds: float
    total_seconds: float


def save_npz_atomic(
    destination: Path,
    payload: dict[str, Any],
    *,
    compressed: bool,
    temp_dir: Path | None = None,
) -> NpzSaveStats:
    temporary = destination.with_name(f"{destination.name}.tmp")
    archive_path = temporary
    if temp_dir is not None:
        temp_dir.mkdir(parents=True, exist_ok=True)
        archive_path = temp_dir / f"{destination.name}.{time.monotonic_ns()}.tmp"
    started_at = time.monotonic()
    array_stats: list[NpzArrayWriteStat] = []
    compression = zipfile.ZIP_DEFLATED if compressed else zipfile.ZIP_STORED
    try:
        archive = zipfile.ZipFile(archive_path, mode="w", compression=compression)
        close_started_at: float | None = None
        write_seconds = 0.0
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
            write_seconds = time.monotonic() - started_at
            close_started_at = time.monotonic()
            archive.close()
        close_seconds = time.monotonic() - close_started_at
        copy_seconds = 0.0
        if archive_path != temporary:
            copy_started_at = time.monotonic()
            _copy_file_buffered(archive_path, temporary)
            copy_seconds = time.monotonic() - copy_started_at
        replace_started_at = time.monotonic()
        temporary.replace(destination)
        replace_seconds = time.monotonic() - replace_started_at
        return NpzSaveStats(
            array_stats=tuple(array_stats),
            write_seconds=write_seconds,
            close_seconds=close_seconds,
            copy_seconds=copy_seconds,
            replace_seconds=replace_seconds,
            total_seconds=time.monotonic() - started_at,
        )
    finally:
        if temporary.exists():
            temporary.unlink()
        if archive_path != temporary and archive_path.exists():
            archive_path.unlink()


def _copy_file_buffered(source: Path, destination: Path) -> None:
    buffer = bytearray(COPY_BUFFER_BYTES)
    view = memoryview(buffer)
    with source.open("rb", buffering=0) as source_file:
        with destination.open("wb", buffering=0) as destination_file:
            while True:
                read = source_file.readinto(buffer)
                if not read:
                    break
                destination_file.write(view[:read])


def copy_file_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp")
    try:
        _copy_file_buffered(source, temporary)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
