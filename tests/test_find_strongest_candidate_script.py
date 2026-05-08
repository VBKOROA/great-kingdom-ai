from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "find_strongest_candidate.py"
SPEC = importlib.util.spec_from_file_location("find_strongest_candidate", SCRIPT_PATH)
assert SPEC is not None
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def _touch_candidates(directory: Path, iterations: list[int]) -> list[Path]:
    directory.mkdir(parents=True)
    paths = []
    for iteration in iterations:
        path = directory / f"candidate-{iteration:06d}.pt"
        path.write_text("checkpoint", encoding="utf-8")
        paths.append(path)
    return paths


def test_select_candidate_checkpoints_applies_iteration_interval(tmp_path: Path) -> None:
    candidate_dir = tmp_path / "candidates"
    _touch_candidates(candidate_dir, [1, 2, 3, 4, 5])

    selected = module.select_candidate_checkpoints(
        candidate_dir,
        glob="candidate-*.pt",
        interval=2,
        start_iteration=2,
        end_iteration=5,
        include_latest=True,
    )

    assert [path.name for path in selected] == [
        "candidate-000002.pt",
        "candidate-000004.pt",
        "candidate-000005.pt",
    ]


def test_select_candidate_checkpoints_can_skip_latest_append(tmp_path: Path) -> None:
    candidate_dir = tmp_path / "candidates"
    _touch_candidates(candidate_dir, [1, 2, 3, 4, 5])

    selected = module.select_candidate_checkpoints(
        candidate_dir,
        glob="candidate-*.pt",
        interval=2,
        start_iteration=None,
        end_iteration=None,
        include_latest=False,
    )

    assert [path.name for path in selected] == [
        "candidate-000001.pt",
        "candidate-000003.pt",
        "candidate-000005.pt",
    ]


def test_select_candidate_checkpoints_rejects_non_positive_interval(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="interval"):
        module.select_candidate_checkpoints(
            tmp_path,
            glob="candidate-*.pt",
            interval=0,
            start_iteration=None,
            end_iteration=None,
            include_latest=True,
        )
