"""Prune regenerable Runpod training artifacts.

The script is dry-run by default. Pass ``--delete`` to actually remove files.
It deliberately keeps trajectory replay and active checkpoints because those are
the expensive state needed to resume v2 training.
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

DEFAULT_WORK_DIR = Path("data/runpod/train-v2-recommended-medium-plus")


@dataclass(frozen=True)
class PruneItem:
    path: Path
    reason: str
    size_bytes: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prune regenerable Runpod artifacts")
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--delete", action="store_true", help="actually delete selected paths")
    parser.add_argument("--keep-targets", type=int, default=2)
    parser.add_argument("--keep-candidates", type=int, default=3)
    parser.add_argument("--keep-onnx", type=int, default=1)
    parser.add_argument(
        "--include-build-cache",
        action="store_true",
        help="also prune local test/type/build caches such as .pytest_cache and rust target",
    )
    return parser


def collect_prune_items(
    work_dir: Path,
    *,
    keep_targets: int = 2,
    keep_candidates: int = 3,
    keep_onnx: int = 1,
    include_build_cache: bool = False,
    repo_root: Path | None = None,
) -> list[PruneItem]:
    work_dir = work_dir.resolve()
    if not _is_safe_work_dir(work_dir):
        raise ValueError(f"refusing unsafe work-dir: {work_dir}")

    items: list[PruneItem] = []
    items.extend(
        _old_numbered_files(
            work_dir / "targets",
            "targets-*.npz",
            keep_targets,
            "old target snapshot",
        )
    )
    items.extend(
        _old_numbered_files(
            work_dir / "checkpoints" / "candidates",
            "candidate-*.pt",
            keep_candidates,
            "old candidate checkpoint",
        )
    )
    items.extend(
        _old_numbered_files(
            work_dir / "checkpoints" / "onnx",
            "best-*.onnx",
            keep_onnx,
            "regenerable ONNX export",
        )
    )
    items.extend(_all_children(work_dir / "self-play", "raw self-play artifact directory"))

    if include_build_cache:
        root = (repo_root or Path.cwd()).resolve()
        cache_paths = [
            root / ".pytest_cache",
            root / ".mypy_cache",
            root / ".ruff_cache",
            root / "python" / "great_kingdom_ai" / "__pycache__",
            root / "tests" / "__pycache__",
            root / "rust" / "great_kingdom_core" / "target",
        ]
        items.extend(
            PruneItem(path=path, reason="local build/test cache", size_bytes=_path_size(path))
            for path in cache_paths
            if path.exists()
        )

    return _dedupe_items(items)


def prune_items(items: list[PruneItem], *, delete: bool) -> None:
    for item in items:
        if not delete:
            continue
        _remove_path(item.path)


def format_summary(items: list[PruneItem], *, delete: bool) -> str:
    total = sum(item.size_bytes for item in items)
    mode = "deleted" if delete else "would_delete"
    lines = [f"{mode}={len(items)} bytes={total} human={_human_size(total)}"]
    for item in sorted(items, key=lambda entry: str(entry.path)):
        lines.append(f"{_human_size(item.size_bytes):>9}  {item.reason:<32}  {item.path}")
    return "\n".join(lines)


def main() -> NoReturn:
    args = build_parser().parse_args()
    items = collect_prune_items(
        args.work_dir,
        keep_targets=args.keep_targets,
        keep_candidates=args.keep_candidates,
        keep_onnx=args.keep_onnx,
        include_build_cache=args.include_build_cache,
    )
    print(format_summary(items, delete=args.delete))
    prune_items(items, delete=args.delete)
    raise SystemExit(0)


def _old_numbered_files(directory: Path, glob: str, keep: int, reason: str) -> list[PruneItem]:
    if keep < 0:
        raise ValueError("keep counts must be non-negative")
    if not directory.exists():
        return []
    paths = sorted((path for path in directory.glob(glob) if path.is_file()), key=_path_sort_key)
    selected = paths[: max(0, len(paths) - keep)]
    return [PruneItem(path=path, reason=reason, size_bytes=_path_size(path)) for path in selected]


def _all_children(directory: Path, reason: str) -> list[PruneItem]:
    if not directory.exists():
        return []
    return [
        PruneItem(path=path, reason=reason, size_bytes=_path_size(path))
        for path in sorted(directory.iterdir())
    ]


def _path_sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    suffix = stem.rsplit("-", 1)[-1]
    return (int(suffix) if suffix.isdecimal() else -1, path.name)


def _path_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file() or path.is_symlink():
        return path.lstat().st_size
    return sum(child.lstat().st_size for child in path.rglob("*") if child.exists())


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    if path.is_dir():
        shutil.rmtree(path)


def _is_safe_work_dir(path: Path) -> bool:
    parts = path.parts
    return len(parts) >= 3 and path.name not in {"", ".", ".."} and path != path.anchor


def _dedupe_items(items: list[PruneItem]) -> list[PruneItem]:
    seen: set[Path] = set()
    deduped: list[PruneItem] = []
    for item in items:
        resolved = item.path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(item)
    return deduped


def _human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.1f}{unit}"
        value /= 1024.0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    main()
