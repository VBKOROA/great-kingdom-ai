from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_snapshot_ordo_ranking.py"
SCRIPT_DIR = SCRIPT_PATH.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
SPEC = importlib.util.spec_from_file_location("run_snapshot_ordo_ranking", SCRIPT_PATH)
assert SPEC is not None
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_select_snapshots_filename_sort(tmp_path: Path) -> None:
    (tmp_path / "snapshot-000100.pt").write_text("ck", encoding="utf-8")
    (tmp_path / "snapshot-000200.pt").write_text("ck", encoding="utf-8")
    (tmp_path / "snapshot-000150.pt").write_text("ck", encoding="utf-8")
    (tmp_path / "other.txt").write_text("other", encoding="utf-8")

    # Limit to latest 2
    selected = module.select_snapshots(
        tmp_path,
        glob="snapshot-*.pt",
        recursive=False,
        max_snapshots=2,
        sort_by_mtime=False,
    )

    # Sorted order of matches: 100, 150, 200. Latest 2 are 150, 200.
    assert [p.name for p in selected] == ["snapshot-000150.pt", "snapshot-000200.pt"]


def test_select_snapshots_sort_by_mtime(tmp_path: Path) -> None:
    p1 = tmp_path / "snapshot-000100.pt"
    p2 = tmp_path / "snapshot-000150.pt"
    p3 = tmp_path / "snapshot-000200.pt"

    p1.write_text("ck", encoding="utf-8")
    p2.write_text("ck", encoding="utf-8")
    p3.write_text("ck", encoding="utf-8")

    # Manually adjust mtime
    # p3 is oldest mtime, p1 is newest mtime
    os.utime(p3, (100, 100))
    os.utime(p2, (200, 200))
    os.utime(p1, (300, 300))

    # sort_by_mtime=True -> p3, p2, p1
    selected = module.select_snapshots(
        tmp_path,
        glob="snapshot-*.pt",
        recursive=False,
        max_snapshots=3,
        sort_by_mtime=True,
    )

    assert [p.name for p in selected] == ["snapshot-000200.pt", "snapshot-000150.pt", "snapshot-000100.pt"]


def test_generate_sparse_pairs_neighbor_offsets() -> None:
    snapshots = [
        Path("s1.pt"),
        Path("s2.pt"),
        Path("s3.pt"),
        Path("s4.pt"),
        Path("s5.pt"),
    ]
    anchors: list[Path] = []
    
    # Offsets: 1, 2. games=8
    pairs = module.generate_sparse_pairs(
        snapshots,
        anchors,
        pair_offsets=[1, 2],
        games=8,
        anchor_games=16,
    )

    # For offset 1:
    # s2 vs s1
    # s3 vs s2
    # s4 vs s3
    # s5 vs s4
    # For offset 2:
    # s3 vs s1
    # s4 vs s2
    # s5 vs s3
    
    # 7 pairs total
    assert len(pairs) == 7
    
    # Verify candidate / baseline order
    # Candidate should be the later snapshot
    neighbor_pairs_dict = {(p.candidate.name, p.baseline.name) for p in pairs}
    expected = {
        ("s2.pt", "s1.pt"),
        ("s3.pt", "s2.pt"),
        ("s4.pt", "s3.pt"),
        ("s5.pt", "s4.pt"),
        ("s3.pt", "s1.pt"),
        ("s4.pt", "s2.pt"),
        ("s5.pt", "s3.pt"),
    }
    assert neighbor_pairs_dict == expected


def test_generate_sparse_pairs_with_anchors_and_deduplication() -> None:
    s1 = Path("s1.pt")
    s2 = Path("s2.pt")
    s3 = Path("s3.pt")
    anchor = Path("s1.pt")  # Anchor overlaps with snapshot
    
    # If s1 is an anchor, s1 should be deduped from snapshots
    # Filtered snapshots: [s2.pt, s3.pt]
    # Offsets: 1. games=8, anchor_games=16
    pairs = module.generate_sparse_pairs(
        [s1, s2, s3],
        [anchor],
        pair_offsets=[1],
        games=8,
        anchor_games=16,
    )

    # Filtered snapshots: [s2.pt, s3.pt]
    # Neighbor: s3 vs s2 (games=8)
    # Anchor: s2 vs s1 (games=16), s3 vs s1 (games=16)
    assert len(pairs) == 3
    
    # Verify games per kind
    pair_by_baseline = {p.baseline.name: p for p in pairs}
    # s3 vs s2 (neighbor)
    assert pair_by_baseline["s2.pt"].games == 8
    assert pair_by_baseline["s2.pt"].kind == "neighbor"
    # s2 vs s1, s3 vs s1 (anchor)
    # Note that s1.pt baseline appears twice, so let's verify both
    anchor_pairs = [p for p in pairs if p.kind == "anchor"]
    assert len(anchor_pairs) == 2
    for p in anchor_pairs:
        assert p.games == 16
        assert p.baseline.name == "s1.pt"


def test_blue_orange_game_to_ordo_pgn_blue_wins() -> None:
    # Candidate is BLUE (1), Winner is BLUE (1) -> White (candidate) wins -> "1-0"
    game = {
        "candidate_player": 1,
        "winner": 1,
    }
    pgn = module.blue_orange_game_to_ordo_pgn(game, "cand", "base")
    assert '[White "cand"]' in pgn
    assert '[Black "base"]' in pgn
    assert '[Result "1-0"]' in pgn


def test_blue_orange_game_to_ordo_pgn_orange_wins() -> None:
    # Candidate is ORANGE (2), Winner is ORANGE (2) -> Black (candidate) wins -> "0-1"
    game = {
        "candidate_player": 2,
        "winner": 2,
    }
    pgn = module.blue_orange_game_to_ordo_pgn(game, "cand", "base")
    assert '[White "base"]' in pgn
    assert '[Black "cand"]' in pgn
    assert '[Result "0-1"]' in pgn


def test_blue_orange_game_to_ordo_pgn_draw() -> None:
    # Draw/no-winner -> Result "1/2-1/2"
    game = {
        "candidate_player": 1,
        "winner": 0,
    }
    pgn = module.blue_orange_game_to_ordo_pgn(game, "cand", "base")
    assert '[Result "1/2-1/2"]' in pgn


def test_parse_ordo_output_representative_table() -> None:
    ordo_text = """
#  name             rating    error  points  played   %
1  snapshot-000125    42.3     18.7    52.5      96  54.7%
2  snapshot-000120     0.0     15.2    48.0      96  50.0%
"""
    ratings = module.parse_ordo_output(ordo_text)
    assert len(ratings) == 2
    assert ratings[0].name == "snapshot-000125"
    assert ratings[0].elo == 42.3
    assert ratings[0].error == 18.7
    assert ratings[0].games == 96

    assert ratings[1].name == "snapshot-000120"
    assert ratings[1].elo == 0.0
    assert ratings[1].error == 15.2
    assert ratings[1].games == 96


def test_parse_ordo_output_alternate_format() -> None:
    ordo_text = """
   # name                          rating    DRA   %   games
   1 snapshot-000125                 42.3   18.7  54.7%     96
   2 snapshot-000120                  0.0   15.2  50.0%     96
"""
    ratings = module.parse_ordo_output(ordo_text)
    assert len(ratings) == 2
    assert ratings[0].name == "snapshot-000125"
    assert ratings[0].elo == 42.3
    assert ratings[0].error == 18.7
    assert ratings[0].games == 96
