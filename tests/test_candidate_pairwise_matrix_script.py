from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_candidate_pairwise_matrix.py"
SPEC = importlib.util.spec_from_file_location("run_candidate_pairwise_matrix", SCRIPT_PATH)
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


def test_select_candidate_checkpoints_can_limit_to_latest_candidates(tmp_path: Path) -> None:
    candidate_dir = tmp_path / "candidates"
    _touch_candidates(candidate_dir, [1, 2, 3, 4, 5])

    selected = module.select_candidate_checkpoints(
        candidate_dir,
        glob="candidate-*.pt",
        interval=2,
        start_iteration=2,
        end_iteration=5,
        include_latest=True,
    )[-2:]

    assert [path.name for path in selected] == [
        "candidate-000004.pt",
        "candidate-000005.pt",
    ]


def test_auto_games_for_pair_count_uses_total_matrix_budget() -> None:
    assert module.auto_games_for_pair_count(
        total_games=120,
        pair_count=6,
        paired_seeds=False,
    ) == 20
    assert module.auto_games_for_pair_count(
        total_games=121,
        pair_count=6,
        paired_seeds=False,
    ) == 21


def test_auto_games_for_pair_count_keeps_paired_seed_games_even() -> None:
    assert module.auto_games_for_pair_count(
        total_games=121,
        pair_count=6,
        paired_seeds=True,
    ) == 22
    assert module.auto_games_for_pair_count(
        total_games=1,
        pair_count=6,
        paired_seeds=True,
    ) == 2


def test_apply_auto_games_total_overrides_config_games() -> None:
    config = module.ArenaConfig(games=60, paired_seeds=True)

    adjusted = module.apply_auto_games_total(
        config,
        auto_games_total=100,
        pair_count=6,
    )

    assert adjusted.games == 18
    assert adjusted.paired_seeds is True
    assert config.games == 60


def test_auto_games_total_is_mutually_exclusive_with_games() -> None:
    parser = module.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--games", "20", "--auto-games-total", "100"])


def test_match_from_summary_derives_baseline_side_results(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate-000002.pt"
    baseline = tmp_path / "candidate-000001.pt"
    report = tmp_path / "report.json"

    match = module._match_from_summary(
        match_index=1,
        candidate=candidate,
        baseline=baseline,
        report_path=report,
        summary={
            "candidate_win_rate": 0.6,
            "best_win_rate": 0.4,
            "candidate_blue_games": 10,
            "candidate_blue_wins": 7,
            "candidate_orange_games": 10,
            "candidate_orange_wins": 5,
            "games": 20,
        },
    )

    assert match.candidate_blue_win_rate == pytest.approx(0.7)
    assert match.candidate_orange_win_rate == pytest.approx(0.5)
    assert match.baseline_blue_games == 10
    assert match.baseline_blue_wins == 5
    assert match.baseline_orange_games == 10
    assert match.baseline_orange_wins == 3
    assert match.baseline_blue_win_rate == pytest.approx(0.5)
    assert match.baseline_orange_win_rate == pytest.approx(0.3)


def test_summarize_pairwise_ranks_average_and_worst_win_rate(tmp_path: Path) -> None:
    a = tmp_path / "candidate-000001.pt"
    b = tmp_path / "candidate-000002.pt"
    c = tmp_path / "candidate-000003.pt"
    candidates = [a, b, c]
    matches = [
        module.PairwiseMatchResult(
            match_index=1,
            candidate=b,
            baseline=a,
            report_path=tmp_path / "b-vs-a.json",
            candidate_win_rate=0.6,
            baseline_win_rate=0.4,
            candidate_blue_games=10,
            candidate_blue_wins=6,
            candidate_orange_games=10,
            candidate_orange_wins=6,
            baseline_blue_games=10,
            baseline_blue_wins=4,
            baseline_orange_games=10,
            baseline_orange_wins=4,
            candidate_blue_win_rate=0.6,
            candidate_orange_win_rate=0.6,
            baseline_blue_win_rate=0.4,
            baseline_orange_win_rate=0.4,
            games=20,
        ),
        module.PairwiseMatchResult(
            match_index=2,
            candidate=c,
            baseline=a,
            report_path=tmp_path / "c-vs-a.json",
            candidate_win_rate=0.7,
            baseline_win_rate=0.3,
            candidate_blue_games=10,
            candidate_blue_wins=7,
            candidate_orange_games=10,
            candidate_orange_wins=7,
            baseline_blue_games=10,
            baseline_blue_wins=3,
            baseline_orange_games=10,
            baseline_orange_wins=3,
            candidate_blue_win_rate=0.7,
            candidate_orange_win_rate=0.7,
            baseline_blue_win_rate=0.3,
            baseline_orange_win_rate=0.3,
            games=20,
        ),
        module.PairwiseMatchResult(
            match_index=3,
            candidate=c,
            baseline=b,
            report_path=tmp_path / "c-vs-b.json",
            candidate_win_rate=0.45,
            baseline_win_rate=0.55,
            candidate_blue_games=10,
            candidate_blue_wins=4,
            candidate_orange_games=10,
            candidate_orange_wins=5,
            baseline_blue_games=10,
            baseline_blue_wins=5,
            baseline_orange_games=10,
            baseline_orange_wins=6,
            candidate_blue_win_rate=0.4,
            candidate_orange_win_rate=0.5,
            baseline_blue_win_rate=0.5,
            baseline_orange_win_rate=0.6,
            games=20,
        ),
    ]

    summary = module.summarize_pairwise(candidates, matches)

    assert summary["matrix"]["candidate-000003"]["candidate-000001"] == pytest.approx(0.7)
    assert summary["matrix"]["candidate-000002"]["candidate-000003"] == pytest.approx(0.55)
    assert summary["ranking"][0]["candidate"] == str(b)
    assert summary["ranking"][0]["average_win_rate"] == pytest.approx(0.575)
    assert summary["ranking"][0]["worst_win_rate"] == pytest.approx(0.55)
    assert summary["ranking"][0]["blue_win_rate"] == pytest.approx(11 / 20)
    assert summary["winner"] == str(b)


def test_load_arena_summary_requires_summary_object(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"not_summary": {}}), encoding="utf-8")

    with pytest.raises(ValueError, match="arena summary"):
        module._load_arena_summary(report)
