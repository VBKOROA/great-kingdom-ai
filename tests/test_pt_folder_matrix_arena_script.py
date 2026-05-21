from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_pt_folder_matrix_arena.py"
SCRIPT_DIR = SCRIPT_PATH.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
SPEC = importlib.util.spec_from_file_location("run_pt_folder_matrix_arena", SCRIPT_PATH)
assert SPEC is not None
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_select_pt_checkpoints_uses_only_top_level_by_default(tmp_path: Path) -> None:
    (tmp_path / "b.pt").write_text("checkpoint", encoding="utf-8")
    (tmp_path / "a.pt").write_text("checkpoint", encoding="utf-8")
    (tmp_path / "skip.onnx").write_text("onnx", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "c.pt").write_text("checkpoint", encoding="utf-8")

    selected = module.select_pt_checkpoints(tmp_path, glob="*.pt", recursive=False)

    assert [path.name for path in selected] == ["a.pt", "b.pt"]


def test_select_pt_checkpoints_can_search_recursively(tmp_path: Path) -> None:
    (tmp_path / "b.pt").write_text("checkpoint", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "a.pt").write_text("checkpoint", encoding="utf-8")

    selected = module.select_pt_checkpoints(tmp_path, glob="*.pt", recursive=True)

    assert [path.relative_to(tmp_path).as_posix() for path in selected] == [
        "b.pt",
        "nested/a.pt",
    ]


def test_auto_games_total_is_mutually_exclusive_with_games(tmp_path: Path) -> None:
    parser = module.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([str(tmp_path), "--games", "20", "--auto-games-total", "100"])


def test_select_pt_checkpoints_requires_existing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="checkpoint directory"):
        module.select_pt_checkpoints(tmp_path / "missing", glob="*.pt", recursive=False)
