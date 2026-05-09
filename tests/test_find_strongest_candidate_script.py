from __future__ import annotations

import importlib.util
import json
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


def test_match_promotes_only_when_challenger_wins_on_both_sides(tmp_path: Path) -> None:
    champion = tmp_path / "candidate-000001.pt"
    challenger = tmp_path / "candidate-000002.pt"
    champion.write_text("champion", encoding="utf-8")
    challenger.write_text("challenger", encoding="utf-8")
    report_path = tmp_path / "report.json"
    _write_report(
        report_path,
        candidate_blue_wins=55,
        candidate_blue_games=100,
        candidate_orange_wins=50,
        candidate_orange_games=100,
    )

    result = module._run_or_load_match(
        match_index=1,
        champion=champion,
        challenger=challenger,
        report_path=report_path,
        arena_config=module.ArenaConfig(promotion_threshold=0.5),
        force=False,
    )

    assert result.winner == challenger
    assert result.challenger_blue_win_rate == pytest.approx(0.55)
    assert result.challenger_orange_win_rate == pytest.approx(0.5)


def test_match_rejects_challenger_when_one_side_is_below_threshold(tmp_path: Path) -> None:
    champion = tmp_path / "candidate-000001.pt"
    challenger = tmp_path / "candidate-000002.pt"
    champion.write_text("champion", encoding="utf-8")
    challenger.write_text("challenger", encoding="utf-8")
    report_path = tmp_path / "report.json"
    _write_report(
        report_path,
        candidate_blue_wins=64,
        candidate_blue_games=100,
        candidate_orange_wins=43,
        candidate_orange_games=100,
    )

    result = module._run_or_load_match(
        match_index=1,
        champion=champion,
        challenger=challenger,
        report_path=report_path,
        arena_config=module.ArenaConfig(promotion_threshold=0.5),
        force=False,
    )

    assert result.winner == champion
    assert result.challenger_win_rate == pytest.approx(0.535)
    assert result.challenger_blue_win_rate == pytest.approx(0.64)
    assert result.challenger_orange_win_rate == pytest.approx(0.43)


def _write_report(
    path: Path,
    *,
    candidate_blue_wins: int,
    candidate_blue_games: int,
    candidate_orange_wins: int,
    candidate_orange_games: int,
) -> None:
    candidate_wins = candidate_blue_wins + candidate_orange_wins
    games = candidate_blue_games + candidate_orange_games
    path.write_text(
        json.dumps(
            {
                "summary": {
                    "candidate_win_rate": candidate_wins / games,
                    "best_win_rate": (games - candidate_wins) / games,
                    "candidate_blue_games": candidate_blue_games,
                    "candidate_blue_wins": candidate_blue_wins,
                    "candidate_orange_games": candidate_orange_games,
                    "candidate_orange_wins": candidate_orange_wins,
                    "games": games,
                }
            }
        ),
        encoding="utf-8",
    )
