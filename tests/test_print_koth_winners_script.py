from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "print_koth_winners.py"
SPEC = importlib.util.spec_from_file_location("print_koth_winners", SCRIPT_PATH)
assert SPEC is not None
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_extract_winners_sorts_by_match_index_and_removes_duplicates() -> None:
    summary = {
        "matches": [
            {
                "match_index": 3,
                "champion_before": "b.pt",
                "challenger": "c.pt",
                "winner": "c.pt",
            },
            {
                "match_index": 1,
                "champion_before": "a.pt",
                "challenger": "b.pt",
                "winner": "b.pt",
            },
            {
                "match_index": 2,
                "champion_before": "b.pt",
                "challenger": "keep-b.pt",
                "winner": "b.pt",
            },
        ]
    }

    winners = module.extract_winners(summary, include_initial_champion=False)

    assert winners == ["b.pt", "c.pt"]


def test_extract_winners_can_keep_duplicates() -> None:
    summary = {
        "matches": [
            {"match_index": 1, "champion_before": "a.pt", "winner": "a.pt"},
            {"match_index": 2, "champion_before": "a.pt", "winner": "a.pt"},
            {"match_index": 3, "champion_before": "a.pt", "winner": "b.pt"},
        ]
    }

    winners = module.extract_winners(
        summary,
        include_initial_champion=False,
        dedupe=False,
    )

    assert winners == ["a.pt", "a.pt", "b.pt"]


def test_extract_winners_can_include_initial_champion() -> None:
    summary = {
        "matches": [
            {
                "match_index": 1,
                "champion_before": "a.pt",
                "challenger": "b.pt",
                "winner": "b.pt",
            }
        ]
    }

    winners = module.extract_winners(summary, include_initial_champion=True)

    assert winners == ["a.pt", "b.pt"]


def test_load_summary_requires_json_object(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps([]), encoding="utf-8")

    with pytest.raises(ValueError, match="JSON object"):
        module.load_summary(summary_path)


def test_extract_winners_requires_winner() -> None:
    with pytest.raises(ValueError, match="missing winner"):
        module.extract_winners({"matches": [{"match_index": 1}]}, include_initial_champion=False)
