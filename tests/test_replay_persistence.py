from pathlib import Path
from typing import Any

import numpy as np

from great_kingdom_ai.replay import persistence


class _FakeNpzMember:
    def __enter__(self) -> "_FakeNpzMember":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def write(self, data: bytes) -> int:
        return len(data)


class _FakeZipFile:
    def __init__(self, path: Path, **_kwargs: Any) -> None:
        self.path = path
        path.write_bytes(b"fake-npz")

    def open(self, *_args: object, **_kwargs: Any) -> _FakeNpzMember:
        return _FakeNpzMember()

    def close(self) -> None:
        return None


def test_save_npz_atomic_write_seconds_excludes_archive_close(
    tmp_path: Path, monkeypatch: Any
) -> None:
    timestamps = iter((10.0, 11.0, 13.0, 16.0, 17.0, 23.0, 24.0, 27.0, 28.0))
    monkeypatch.setattr(persistence.time, "monotonic", lambda: next(timestamps))
    monkeypatch.setattr(persistence.zipfile, "ZipFile", _FakeZipFile)

    stats = persistence.save_npz_atomic(
        tmp_path / "replay.npz",
        {"features": np.arange(4, dtype=np.float32)},
        compressed=False,
    )

    assert stats.array_stats[0].seconds == 2.0
    assert stats.write_seconds == 6.0
    assert stats.close_seconds == 6.0
    assert stats.copy_seconds == 0.0
    assert stats.replace_seconds == 3.0
    assert stats.total_seconds == 18.0


def test_save_npz_atomic_can_stage_archive_in_temp_dir(tmp_path: Path) -> None:
    destination = tmp_path / "network" / "replay.npz"
    temp_dir = tmp_path / "local-tmp"
    destination.parent.mkdir()

    stats = persistence.save_npz_atomic(
        destination,
        {"features": np.arange(4, dtype=np.float32)},
        compressed=False,
        temp_dir=temp_dir,
    )

    assert destination.is_file()
    assert stats.copy_seconds >= 0.0
    assert not list(temp_dir.iterdir())
    with np.load(destination) as data:
        assert data["features"].tolist() == [0.0, 1.0, 2.0, 3.0]
