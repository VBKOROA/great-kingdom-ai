"""Compare Gumbel search settings with the same checkpoint on both sides."""

from __future__ import annotations

import argparse
import gc
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NoReturn

from great_kingdom_ai.arena_runner import ArenaRuntime, run_arena_batched
from great_kingdom_ai.arena_types import (
    ArenaConfig,
    ArenaReport,
    validate_arena_config,
)
from great_kingdom_ai.evaluate import load_arena_config, load_model_from_checkpoint
from great_kingdom_ai.evaluator import (
    evaluate_feature_arrays_logits_values,
    evaluate_feature_batch,
    evaluate_feature_batch_logits_values,
)
from great_kingdom_ai.game_core import (
    create_core_game_state,
    create_core_gumbel_search,
    import_core,
)

DEFAULT_ARENA_CONFIG = Path("configs/runpod/arena.json")
DEFAULT_SETTINGS = (
    ("c1_p01", "1.0", "0.1"),
    ("c01_p01", "0.1", "0.1"),
)


@dataclass(frozen=True)
class GumbelSetting:
    name: str
    gumbel_c_scale: float
    policy_target_c_scale: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SettingMatchResult:
    match_index: int
    candidate_setting: GumbelSetting
    baseline_setting: GumbelSetting
    report_path: Path
    candidate_win_rate: float
    baseline_win_rate: float
    candidate_blue_games: int
    candidate_blue_wins: int
    candidate_orange_games: int
    candidate_orange_wins: int
    baseline_blue_games: int
    baseline_blue_wins: int
    baseline_orange_games: int
    baseline_orange_wins: int
    candidate_blue_win_rate: float
    candidate_orange_win_rate: float
    baseline_blue_win_rate: float
    baseline_orange_win_rate: float
    games: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "match_index": self.match_index,
            "candidate_setting": self.candidate_setting.to_dict(),
            "baseline_setting": self.baseline_setting.to_dict(),
            "report_path": str(self.report_path),
            "candidate_win_rate": self.candidate_win_rate,
            "baseline_win_rate": self.baseline_win_rate,
            "candidate_blue_games": self.candidate_blue_games,
            "candidate_blue_wins": self.candidate_blue_wins,
            "candidate_orange_games": self.candidate_orange_games,
            "candidate_orange_wins": self.candidate_orange_wins,
            "baseline_blue_games": self.baseline_blue_games,
            "baseline_blue_wins": self.baseline_blue_wins,
            "baseline_orange_games": self.baseline_orange_games,
            "baseline_orange_wins": self.baseline_orange_wins,
            "candidate_blue_win_rate": self.candidate_blue_win_rate,
            "candidate_orange_win_rate": self.candidate_orange_win_rate,
            "baseline_blue_win_rate": self.baseline_blue_win_rate,
            "baseline_orange_win_rate": self.baseline_orange_win_rate,
            "games": self.games,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a same-checkpoint arena matrix where each side uses a different "
            "Gumbel c_scale / policy_target_c_scale setting."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--arena-config", type=Path, default=DEFAULT_ARENA_CONFIG)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="report output folder; defaults to CHECKPOINT_PARENT/gumbel-setting-arena",
    )
    parser.add_argument(
        "--setting",
        nargs=3,
        action="append",
        metavar=("NAME", "GUMBEL_C_SCALE", "POLICY_TARGET_C_SCALE"),
        default=None,
        help=(
            "setting to compare; repeat at least twice. "
            "Default: c1_p01 1.0 0.1 and c01_p01 0.1 0.1"
        ),
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--games", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument("--gumbel-seed", type=int, default=None)
    parser.add_argument("--gumbel-simulations", type=int, default=None)
    parser.add_argument("--gumbel-max-considered-actions", type=int, default=None)
    parser.add_argument("--gumbel-c-visit", type=float, default=None)
    parser.add_argument("--gumbel-scale", type=float, default=None)
    parser.add_argument("--opening-gumbel-turns", type=int, default=None)
    parser.add_argument("--opening-gumbel-scale", type=float, default=None)
    parser.add_argument("--policy-target-c-visit", type=float, default=None)
    parser.add_argument("--policy-target-temperature", type=float, default=None)
    parser.add_argument("--leaf-batch-size", type=int, default=None)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument(
        "--paired-seeds",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use each arena seed for a candidate-setting Blue/Orange pair.",
    )
    parser.add_argument("--promotion-threshold", type=float, default=None)
    parser.add_argument(
        "--require-side-win-rates-for-promotion",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--prefer-ema",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="load EMA weights from the checkpoint when available",
    )
    parser.add_argument("--force", action="store_true", help="rerun existing match reports")
    parser.add_argument("--progress", action="store_true", help="print one JSON line per game")
    return parser


def main() -> NoReturn:
    args = build_parser().parse_args()
    config = _load_effective_arena_config(args)
    settings = parse_settings(args.setting)
    if len(settings) < 2:
        raise SystemExit("at least two settings are required")
    _validate_unique_setting_names(settings)

    output_dir = args.output_dir or args.checkpoint.parent / "gumbel-setting-arena"
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model_from_checkpoint(
        args.checkpoint,
        device=config.device,
        prefer_ema=args.prefer_ema,
    )
    print(
        json.dumps(
            {
                "event": "gumbel_setting_arena_start",
                "checkpoint": str(args.checkpoint),
                "arena_config": asdict(config),
                "settings": [setting.to_dict() for setting in settings],
                "output_dir": str(output_dir),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    matches: list[SettingMatchResult] = []
    match_index = 0
    for baseline_index, baseline_setting in enumerate(settings):
        for candidate_setting in settings[baseline_index + 1 :]:
            match_index += 1
            report_path = output_dir / _match_report_name(
                match_index,
                candidate_setting,
                baseline_setting,
            )
            result = _run_or_load_setting_match(
                match_index=match_index,
                model=model,
                candidate_setting=candidate_setting,
                baseline_setting=baseline_setting,
                base_config=config,
                report_path=report_path,
                force=args.force,
                progress=args.progress,
            )
            matches.append(result)
            print(
                json.dumps(
                    {"event": "gumbel_setting_arena_match", **result.to_dict()},
                    sort_keys=True,
                ),
                flush=True,
            )

    payload = {
        "event": "gumbel_setting_arena_summary",
        "checkpoint": str(args.checkpoint),
        "arena_config": asdict(config),
        "settings": [setting.to_dict() for setting in settings],
        "matches": [match.to_dict() for match in matches],
        **summarize_setting_matches(settings, matches),
    }
    _write_json(output_dir / "summary.json", payload)
    print(json.dumps(payload, sort_keys=True), flush=True)
    _release_cuda_cache()
    raise SystemExit(0)


def parse_settings(rows: list[list[str]] | None) -> list[GumbelSetting]:
    raw_rows = rows if rows is not None else [list(row) for row in DEFAULT_SETTINGS]
    return [parse_setting(row) for row in raw_rows]


def parse_setting(row: list[str]) -> GumbelSetting:
    if len(row) != 3:
        raise ValueError("setting rows must be NAME GUMBEL_C_SCALE POLICY_TARGET_C_SCALE")
    name, gumbel_c_scale_text, policy_target_c_scale_text = row
    if not name:
        raise ValueError("setting name must not be empty")
    setting = GumbelSetting(
        name=name,
        gumbel_c_scale=float(gumbel_c_scale_text),
        policy_target_c_scale=float(policy_target_c_scale_text),
    )
    if not math.isfinite(setting.gumbel_c_scale) or setting.gumbel_c_scale <= 0.0:
        raise ValueError("GUMBEL_C_SCALE must be positive")
    if (
        not math.isfinite(setting.policy_target_c_scale)
        or setting.policy_target_c_scale <= 0.0
    ):
        raise ValueError("POLICY_TARGET_C_SCALE must be positive")
    return setting


def summarize_setting_matches(
    settings: list[GumbelSetting],
    matches: list[SettingMatchResult],
) -> dict[str, Any]:
    matrix: dict[str, dict[str, float | None]] = {
        setting.name: {other.name: None for other in settings} for setting in settings
    }
    side_totals: dict[str, dict[str, int]] = {
        setting.name: {"blue_wins": 0, "blue_games": 0, "orange_wins": 0, "orange_games": 0}
        for setting in settings
    }
    for setting in settings:
        matrix[setting.name][setting.name] = 0.5

    for match in matches:
        candidate_name = match.candidate_setting.name
        baseline_name = match.baseline_setting.name
        matrix[candidate_name][baseline_name] = match.candidate_win_rate
        matrix[baseline_name][candidate_name] = match.baseline_win_rate
        _add_side_result(
            side_totals[candidate_name],
            blue_wins=match.candidate_blue_wins,
            blue_games=match.candidate_blue_games,
            orange_wins=match.candidate_orange_wins,
            orange_games=match.candidate_orange_games,
        )
        _add_side_result(
            side_totals[baseline_name],
            blue_wins=match.baseline_blue_wins,
            blue_games=match.baseline_blue_games,
            orange_wins=match.baseline_orange_wins,
            orange_games=match.baseline_orange_games,
        )

    ranking = [
        _setting_summary(setting, matrix=matrix, side_totals=side_totals)
        for setting in settings
    ]
    ranking.sort(
        key=lambda row: (
            float(row["average_win_rate"]),
            float(row["worst_win_rate"]),
            float(row["best_win_rate"]),
            str(row["setting"]),
        ),
        reverse=True,
    )
    return {
        "matrix": matrix,
        "ranking": ranking,
        "winner": ranking[0]["setting"] if ranking else None,
    }


def setting_config(base_config: ArenaConfig, setting: GumbelSetting) -> ArenaConfig:
    config = ArenaConfig(
        **{
            **asdict(base_config),
            "gumbel_c_scale": setting.gumbel_c_scale,
            "policy_target_c_scale": setting.policy_target_c_scale,
        }
    )
    validate_arena_config(config)
    return config


def run_setting_match(
    *,
    model: Any,
    candidate_setting: GumbelSetting,
    baseline_setting: GumbelSetting,
    base_config: ArenaConfig,
    progress: bool,
) -> ArenaReport:
    validate_arena_config(base_config)
    runtime = ArenaRuntime(
        create_game_state=create_core_game_state,
        create_search_engine=lambda config, offset: create_core_gumbel_search(
            config,
            seed_offset=offset,
        ),
        create_batch=lambda config, count, seed_start, index_start: create_setting_arena_batch(
            config,
            game_count=count,
            seed_start=seed_start,
            game_index_start=index_start,
            candidate_setting=candidate_setting,
            baseline_setting=baseline_setting,
        ),
        evaluate_feature_batch=evaluate_feature_batch,
        evaluate_feature_batch_logits_values=evaluate_feature_batch_logits_values,
        evaluate_feature_arrays_logits_values=evaluate_feature_arrays_logits_values,
    )
    return run_arena_batched(
        candidate_model=model,
        best_model=model,
        config=base_config,
        onnx_evaluators=None,
        progress_callback=(
            _progress_callback(candidate_setting, baseline_setting) if progress else None
        ),
        runtime=runtime,
    )


def create_setting_arena_batch(
    config: ArenaConfig,
    *,
    game_count: int,
    seed_start: int,
    game_index_start: int,
    candidate_setting: GumbelSetting,
    baseline_setting: GumbelSetting,
) -> Any:
    core = import_core("same-checkpoint Gumbel setting arena")
    if not hasattr(core, "GumbelArenaBatch"):
        raise RuntimeError(
            "great_kingdom_core.GumbelArenaBatch is not available. "
            "Rebuild the Rust extension before batched setting arena evaluation."
        )
    return core.GumbelArenaBatch(
        game_count=game_count,
        seed_start=seed_start,
        game_index_start=game_index_start,
        simulations=config.gumbel_simulations,
        max_considered_actions=config.gumbel_max_considered_actions,
        c_visit=config.gumbel_c_visit,
        c_scale=config.gumbel_c_scale,
        seed=config.gumbel_seed,
        gumbel_scale=config.gumbel_scale,
        policy_target_temperature=config.policy_target_temperature,
        policy_target_c_visit=config.policy_target_c_visit,
        policy_target_c_scale=config.policy_target_c_scale,
        paired_seeds=config.paired_seeds,
        candidate_c_scale=candidate_setting.gumbel_c_scale,
        candidate_policy_target_c_scale=candidate_setting.policy_target_c_scale,
        best_c_scale=baseline_setting.gumbel_c_scale,
        best_policy_target_c_scale=baseline_setting.policy_target_c_scale,
    )


def _progress_callback(
    candidate_setting: GumbelSetting,
    baseline_setting: GumbelSetting,
) -> Any:
    def callback(completed: int, total: int, game: Any) -> None:
        print(
            json.dumps(
                {
                    "event": "gumbel_setting_arena_game",
                    "game": completed,
                    "games": total,
                    "candidate_setting": candidate_setting.name,
                    "baseline_setting": baseline_setting.name,
                    "seed": game.seed,
                    "candidate_player": game.candidate_player,
                    "winner": game.winner,
                    "turns": len(game.moves),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    return callback


def _run_or_load_setting_match(
    *,
    match_index: int,
    model: Any,
    candidate_setting: GumbelSetting,
    baseline_setting: GumbelSetting,
    base_config: ArenaConfig,
    report_path: Path,
    force: bool,
    progress: bool,
) -> SettingMatchResult:
    if report_path.exists() and not force:
        summary = _load_arena_summary(report_path)
    else:
        report = run_setting_match(
            model=model,
            candidate_setting=candidate_setting,
            baseline_setting=baseline_setting,
            base_config=base_config,
            progress=progress,
        )
        summary = report.summary.to_dict()
        _write_json(
            report_path,
            {
                "base_config": asdict(base_config),
                "candidate_setting": candidate_setting.to_dict(),
                "baseline_setting": baseline_setting.to_dict(),
                "games": [game.to_dict() for game in report.games],
                "summary": summary,
            },
        )

    return _match_from_summary(
        match_index=match_index,
        candidate_setting=candidate_setting,
        baseline_setting=baseline_setting,
        report_path=report_path,
        summary=summary,
    )


def _match_from_summary(
    *,
    match_index: int,
    candidate_setting: GumbelSetting,
    baseline_setting: GumbelSetting,
    report_path: Path,
    summary: dict[str, Any],
) -> SettingMatchResult:
    candidate_blue_games = int(summary["candidate_blue_games"])
    candidate_orange_games = int(summary["candidate_orange_games"])
    candidate_blue_wins = int(summary["candidate_blue_wins"])
    candidate_orange_wins = int(summary["candidate_orange_wins"])
    baseline_blue_games = candidate_orange_games
    baseline_orange_games = candidate_blue_games
    baseline_blue_wins = candidate_orange_games - candidate_orange_wins
    baseline_orange_wins = candidate_blue_games - candidate_blue_wins
    return SettingMatchResult(
        match_index=match_index,
        candidate_setting=candidate_setting,
        baseline_setting=baseline_setting,
        report_path=report_path,
        candidate_win_rate=float(summary["candidate_win_rate"]),
        baseline_win_rate=float(summary["best_win_rate"]),
        candidate_blue_games=candidate_blue_games,
        candidate_blue_wins=candidate_blue_wins,
        candidate_orange_games=candidate_orange_games,
        candidate_orange_wins=candidate_orange_wins,
        baseline_blue_games=baseline_blue_games,
        baseline_blue_wins=baseline_blue_wins,
        baseline_orange_games=baseline_orange_games,
        baseline_orange_wins=baseline_orange_wins,
        candidate_blue_win_rate=_side_win_rate(
            wins=candidate_blue_wins,
            games=candidate_blue_games,
        ),
        candidate_orange_win_rate=_side_win_rate(
            wins=candidate_orange_wins,
            games=candidate_orange_games,
        ),
        baseline_blue_win_rate=_side_win_rate(
            wins=baseline_blue_wins,
            games=baseline_blue_games,
        ),
        baseline_orange_win_rate=_side_win_rate(
            wins=baseline_orange_wins,
            games=baseline_orange_games,
        ),
        games=int(summary["games"]),
    )


def _load_effective_arena_config(args: argparse.Namespace) -> ArenaConfig:
    config = load_arena_config(
        args.arena_config,
        randomize_missing_seed_start=args.seed_start is None,
        randomize_missing_gumbel_seed=args.gumbel_seed is None,
    )
    data = asdict(config)
    overrides = {
        "device": args.device,
        "games": args.games,
        "batch_size": args.batch_size,
        "seed_start": args.seed_start,
        "gumbel_seed": args.gumbel_seed,
        "gumbel_simulations": args.gumbel_simulations,
        "gumbel_max_considered_actions": args.gumbel_max_considered_actions,
        "gumbel_c_visit": args.gumbel_c_visit,
        "gumbel_scale": args.gumbel_scale,
        "opening_gumbel_turns": args.opening_gumbel_turns,
        "opening_gumbel_scale": args.opening_gumbel_scale,
        "policy_target_c_visit": args.policy_target_c_visit,
        "policy_target_temperature": args.policy_target_temperature,
        "leaf_batch_size": args.leaf_batch_size,
        "max_turns": args.max_turns,
        "paired_seeds": args.paired_seeds,
        "promotion_threshold": args.promotion_threshold,
        "require_side_win_rates_for_promotion": args.require_side_win_rates_for_promotion,
    }
    data.update({key: value for key, value in overrides.items() if value is not None})
    return ArenaConfig(**data)


def _setting_summary(
    setting: GumbelSetting,
    *,
    matrix: dict[str, dict[str, float | None]],
    side_totals: dict[str, dict[str, int]],
) -> dict[str, Any]:
    win_rates = [
        float(value)
        for opponent_name, value in matrix[setting.name].items()
        if opponent_name != setting.name and value is not None
    ]
    totals = side_totals[setting.name]
    return {
        "setting": setting.name,
        **setting.to_dict(),
        "matches": len(win_rates),
        "average_win_rate": sum(win_rates) / len(win_rates) if win_rates else 0.0,
        "worst_win_rate": min(win_rates) if win_rates else 0.0,
        "best_win_rate": max(win_rates) if win_rates else 0.0,
        "winning_match_count": sum(1 for rate in win_rates if rate > 0.5),
        "non_losing_match_count": sum(1 for rate in win_rates if rate >= 0.5),
        "blue_win_rate": _side_win_rate(
            wins=totals["blue_wins"],
            games=totals["blue_games"],
        ),
        "orange_win_rate": _side_win_rate(
            wins=totals["orange_wins"],
            games=totals["orange_games"],
        ),
    }


def _add_side_result(
    totals: dict[str, int],
    *,
    blue_wins: int,
    blue_games: int,
    orange_wins: int,
    orange_games: int,
) -> None:
    totals["blue_wins"] += blue_wins
    totals["blue_games"] += blue_games
    totals["orange_wins"] += orange_wins
    totals["orange_games"] += orange_games


def _side_win_rate(*, wins: int, games: int) -> float:
    return wins / games if games > 0 else 0.0


def _validate_unique_setting_names(settings: list[GumbelSetting]) -> None:
    names = [setting.name for setting in settings]
    if len(set(names)) != len(names):
        raise ValueError("setting names must be unique")


def _match_report_name(
    match_index: int,
    candidate_setting: GumbelSetting,
    baseline_setting: GumbelSetting,
) -> str:
    return f"pair-{match_index:03d}-{candidate_setting.name}-vs-{baseline_setting.name}.json"


def _load_arena_summary(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = data.get("summary")
    if not isinstance(summary, dict):
        raise ValueError(f"{path} does not contain an arena summary")
    return summary


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _release_cuda_cache() -> None:
    gc.collect()
    try:
        import torch
    except ModuleNotFoundError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
