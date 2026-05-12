from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "prune_runpod_artifacts.py"
SPEC = importlib.util.spec_from_file_location("prune_runpod_artifacts", SCRIPT_PATH)
assert SPEC is not None
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def _write(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_collect_prune_items_keeps_active_v2_training_state(tmp_path: Path) -> None:
    work_dir = tmp_path / "runpod" / "train-v2"
    trajectory = _write(work_dir / "replay" / "trajectory-replay.npz")
    latest = _write(work_dir / "targets" / "latest.npz")
    best = _write(work_dir / "checkpoints" / "best.pt")
    training_latest = _write(work_dir / "checkpoints" / "training-latest.pt")
    _write(work_dir / "targets" / "targets-000001.npz")
    _write(work_dir / "targets" / "targets-000002.npz")
    _write(work_dir / "targets" / "targets-000003.npz")
    _write(work_dir / "checkpoints" / "candidates" / "candidate-000001.pt")
    _write(work_dir / "checkpoints" / "candidates" / "candidate-000002.pt")
    _write(work_dir / "checkpoints" / "onnx" / "best-000001.onnx")
    _write(work_dir / "checkpoints" / "onnx" / "best-000002.onnx")
    _write(work_dir / "self-play" / "iteration-000001" / "marker.txt")

    items = module.collect_prune_items(
        work_dir,
        keep_targets=1,
        keep_candidates=1,
        keep_onnx=1,
    )
    selected = {item.path for item in items}

    assert trajectory not in selected
    assert latest not in selected
    assert best not in selected
    assert training_latest not in selected
    assert work_dir / "targets" / "targets-000001.npz" in selected
    assert work_dir / "targets" / "targets-000002.npz" in selected
    assert work_dir / "targets" / "targets-000003.npz" not in selected
    assert work_dir / "checkpoints" / "candidates" / "candidate-000001.pt" in selected
    assert work_dir / "checkpoints" / "candidates" / "candidate-000002.pt" not in selected
    assert work_dir / "checkpoints" / "onnx" / "best-000001.onnx" in selected
    assert work_dir / "checkpoints" / "onnx" / "best-000002.onnx" not in selected
    assert work_dir / "self-play" / "iteration-000001" in selected


def test_prune_items_is_dry_run_unless_delete_is_true(tmp_path: Path) -> None:
    work_dir = tmp_path / "runpod" / "train-v2"
    old_target = _write(work_dir / "targets" / "targets-000001.npz")
    _write(work_dir / "targets" / "targets-000002.npz")
    items = module.collect_prune_items(work_dir, keep_targets=1)

    module.prune_items(items, delete=False)
    assert old_target.exists()

    module.prune_items(items, delete=True)
    assert not old_target.exists()


def test_collect_prune_items_can_include_build_caches(tmp_path: Path) -> None:
    work_dir = tmp_path / "runpod" / "train-v2"
    _write(work_dir / "targets" / "targets-000001.npz")
    repo_root = tmp_path / "repo"
    cache = _write(repo_root / ".pytest_cache" / "CACHEDIR.TAG")
    rust_target = _write(repo_root / "rust" / "great_kingdom_core" / "target" / "marker")

    items = module.collect_prune_items(
        work_dir,
        keep_targets=0,
        include_build_cache=True,
        repo_root=repo_root,
    )
    selected = {item.path for item in items}

    assert cache.parent in selected
    assert rust_target.parent in selected
