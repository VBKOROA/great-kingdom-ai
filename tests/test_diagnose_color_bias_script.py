from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_color_bias.py"
SPEC = importlib.util.spec_from_file_location("diagnose_color_bias", SCRIPT_PATH)
assert SPEC is not None
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_analyze_game_logs_reports_overall_and_recent_bias(tmp_path: Path) -> None:
    path = tmp_path / "game_logs.jsonl"
    rows = [
        {"seed": 3, "winner": 1, "moves": [{"turn": 0}], "end_reason": 1, "territory_scores": [5, 2]},
        {"seed": 1, "winner": 2, "moves": [{"turn": 0}, {"turn": 1}], "end_reason": 2, "territory_scores": [1, 6]},
        {"seed": 2, "winner": 1, "moves": [], "end_reason": 3, "territory_scores": [4, 3]},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    records = module.load_game_log_records([path])
    payload = module.analyze_records(
        records,
        sources=[str(path)],
        recent_windows=[2],
        include_model_breakdown=True,
    )

    assert payload["overall"]["games"] == 3
    assert payload["overall"]["blue_wins"] == 2
    assert payload["overall"]["orange_wins"] == 1
    assert payload["overall"]["blue_win_rate"] == pytest.approx(2 / 3)
    assert payload["overall"]["average_game_length"] == pytest.approx(1.0)
    assert payload["recent_windows"][0]["last_games"] == 2
    assert payload["recent_windows"][0]["blue_wins"] == 2


def test_load_replay_records_can_group_by_model_version(tmp_path: Path) -> None:
    path = tmp_path / "trajectory-replay.npz"
    np.savez(
        path,
        episode_winners=np.asarray([1, 2], dtype=np.int64),
        episode_offsets=np.asarray([0, 2, 5], dtype=np.int64),
        territory_scores=np.asarray([[3, 1], [2, 4]], dtype=np.int64),
        model_versions=np.asarray([7, 7, 8, 8, 8], dtype=np.int64),
        created_iterations=np.asarray([70, 70, 80, 80, 80], dtype=np.int64),
    )

    records = module.load_replay_records(path)
    payload = module.analyze_records(
        records,
        sources=[str(path)],
        recent_windows=[1],
        include_model_breakdown=True,
    )

    assert payload["overall"]["games"] == 2
    assert payload["overall"]["blue_wins"] == 1
    assert payload["overall"]["orange_wins"] == 1
    assert payload["by_model_version"] == [
        {
            "key": 7,
            **module.summarize_records([records[0]]),
        },
        {
            "key": 8,
            **module.summarize_records([records[1]]),
        },
    ]
