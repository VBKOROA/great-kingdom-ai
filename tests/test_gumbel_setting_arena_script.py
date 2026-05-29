from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_gumbel_setting_arena.py"
SPEC = importlib.util.spec_from_file_location("run_gumbel_setting_arena", SCRIPT_PATH)
assert SPEC is not None
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_parse_settings_defaults_to_c1_vs_c01_with_policy_point_one() -> None:
    settings = module.parse_settings(None)

    assert [(row.name, row.gumbel_c_scale, row.policy_target_c_scale) for row in settings] == [
        ("c1_p01", 1.0, 0.1),
        ("c01_p01", 0.1, 0.1),
    ]


def test_parse_setting_rejects_non_positive_scale() -> None:
    with pytest.raises(ValueError, match="GUMBEL_C_SCALE"):
        module.parse_setting(["bad", "0", "0.1"])

    with pytest.raises(ValueError, match="POLICY_TARGET_C_SCALE"):
        module.parse_setting(["bad", "0.1", "-1"])


def test_setting_config_overrides_only_setting_scales() -> None:
    base = module.ArenaConfig(
        games=12,
        batch_size=99,
        gumbel_c_scale=1.0,
        policy_target_c_scale=0.25,
        gumbel_simulations=64,
    )
    setting = module.GumbelSetting("c01", 0.1, 0.1)

    adjusted = module.setting_config(base, setting)

    assert adjusted.gumbel_c_scale == pytest.approx(0.1)
    assert adjusted.policy_target_c_scale == pytest.approx(0.1)
    assert adjusted.games == 12
    assert adjusted.batch_size == 99
    assert adjusted.gumbel_simulations == 64
    assert base.gumbel_c_scale == pytest.approx(1.0)


def test_match_from_summary_derives_baseline_side_results(tmp_path: Path) -> None:
    candidate = module.GumbelSetting("c01", 0.1, 0.1)
    baseline = module.GumbelSetting("c1", 1.0, 0.1)

    match = module._match_from_summary(
        match_index=1,
        candidate_setting=candidate,
        baseline_setting=baseline,
        report_path=tmp_path / "report.json",
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


def test_summarize_setting_matches_ranks_average_and_worst_win_rate(tmp_path: Path) -> None:
    a = module.GumbelSetting("a", 0.1, 0.1)
    b = module.GumbelSetting("b", 0.5, 0.1)
    c = module.GumbelSetting("c", 1.0, 0.1)
    matches = [
        module.SettingMatchResult(
            match_index=1,
            candidate_setting=b,
            baseline_setting=a,
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
        module.SettingMatchResult(
            match_index=2,
            candidate_setting=c,
            baseline_setting=a,
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
        module.SettingMatchResult(
            match_index=3,
            candidate_setting=c,
            baseline_setting=b,
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

    summary = module.summarize_setting_matches([a, b, c], matches)

    assert summary["matrix"]["c"]["a"] == pytest.approx(0.7)
    assert summary["matrix"]["b"]["c"] == pytest.approx(0.55)
    assert summary["ranking"][0]["setting"] == "b"
    assert summary["ranking"][0]["average_win_rate"] == pytest.approx(0.575)
    assert summary["ranking"][0]["worst_win_rate"] == pytest.approx(0.55)
    assert summary["ranking"][0]["blue_win_rate"] == pytest.approx(11 / 20)
    assert summary["winner"] == "b"
