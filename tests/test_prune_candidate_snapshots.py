from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "prune_candidate_snapshots.py"
SPEC = importlib.util.spec_from_file_location("prune_candidate_snapshots", SCRIPT_PATH)
assert SPEC is not None
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def _touch_snapshot(path: Path, *, mtime: float) -> Path:
    path.write_text("checkpoint", encoding="utf-8")
    os.utime(path, (mtime, mtime))
    return path


def test_snapshot_sort_key_uses_mtime_then_name(tmp_path: Path) -> None:
    newer = _touch_snapshot(tmp_path / "z.pt", mtime=200.0)
    older = _touch_snapshot(tmp_path / "b.pt", mtime=100.0)
    same_time_name_first = _touch_snapshot(tmp_path / "a.pt", mtime=100.0)

    sorted_paths = sorted(
        [newer, older, same_time_name_first],
        key=module.snapshot_sort_key,
    )

    assert [path.name for path in sorted_paths] == ["a.pt", "b.pt", "z.pt"]


def test_prune_candidates_basic(tmp_path: Path) -> None:
    snapshots = [
        _touch_snapshot(tmp_path / f"snapshot-{i}.pt", mtime=float(i)) for i in range(100, 600, 100)
    ]

    def mock_evaluate(candidate: Path, pivot: Path, round_idx: int) -> dict[str, object]:
        win_rate = 0.5 + (candidate.stat().st_mtime - pivot.stat().st_mtime) / 1000.0
        return {"win_rate": win_rate, "report_path": f"report-{candidate.stem}.json"}

    result = module.prune_candidates(
        all_snapshots=snapshots,
        top_k=2,
        evaluate_fn=mock_evaluate,
        arena_config_path=Path("arena.json"),
        snapshot_dir=tmp_path,
    )

    assert result["initial_count"] == 5
    assert result["final_count"] == 2
    assert result["survivors"] == ["snapshot-400.pt", "snapshot-500.pt"]
    assert len(result["rounds"]) == 2


def test_prune_candidates_no_shrink_fallback_keeps_pivot_eligible(tmp_path: Path) -> None:
    snapshots = [
        _touch_snapshot(tmp_path / f"snapshot-{i}.pt", mtime=float(i)) for i in range(100, 600, 100)
    ]

    def mock_evaluate(candidate: Path, pivot: Path, round_idx: int) -> dict[str, object]:
        return {"win_rate": 0.6, "report_path": "report.json"}

    result = module.prune_candidates(
        all_snapshots=snapshots,
        top_k=3,
        evaluate_fn=mock_evaluate,
        arena_config_path=Path("arena.json"),
        snapshot_dir=tmp_path,
    )

    assert result["final_count"] == 3
    assert result["survivors"] == ["snapshot-300.pt", "snapshot-400.pt", "snapshot-500.pt"]
    assert len(result["rounds"]) == 2
    assert result["rounds"][0]["pivot"] == "snapshot-300.pt"
    assert result["rounds"][0]["round"] == 1
    assert result["rounds"][1]["pivot"] == "snapshot-400.pt"
    assert result["rounds"][1]["round"] == 2


def test_prune_candidates_already_small(tmp_path: Path) -> None:
    snapshots = [
        _touch_snapshot(tmp_path / "old.pt", mtime=100.0),
        _touch_snapshot(tmp_path / "new.pt", mtime=200.0),
    ]

    def fail_evaluate(candidate: Path, pivot: Path, round_idx: int) -> dict[str, object]:
        raise AssertionError("evaluate_fn should not be called")

    result = module.prune_candidates(
        all_snapshots=snapshots,
        top_k=3,
        evaluate_fn=fail_evaluate,
        arena_config_path=Path("arena.json"),
        snapshot_dir=tmp_path,
    )

    assert result["final_count"] == 2
    assert len(result["rounds"]) == 0
    assert result["survivors"] == ["old.pt", "new.pt"]


def test_default_report_dir_matches_runpod_candidate_layout() -> None:
    snapshot_dir = Path("work") / "checkpoints" / "candidates"

    assert module.default_report_dir(snapshot_dir) == Path("work") / "reports" / "pruning"
