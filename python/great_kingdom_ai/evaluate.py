"""Arena evaluation and best-model promotion public API."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, NoReturn, cast

from great_kingdom_ai import game_core
from great_kingdom_ai.arena_runner import ArenaRuntime, summarize_arena
from great_kingdom_ai.arena_runner import evaluate_state_policies as _evaluate_state_policies
from great_kingdom_ai.arena_runner import (
    evaluate_state_policy_logits_batch as _evaluate_state_policy_logits_batch,
)
from great_kingdom_ai.arena_runner import play_arena_game as _play_arena_game
from great_kingdom_ai.arena_runner import run_arena as _run_arena
from great_kingdom_ai.arena_runner import run_arena_batched as _run_arena_batched
from great_kingdom_ai.arena_types import (
    BLUE,
    ORANGE,
    ArenaBatchLike,
    ArenaConfig,
    ArenaGameResult,
    ArenaOnnxEvaluators,
    ArenaReport,
    ArenaSearchLike,
    ArenaSearchResultLike,
    ArenaSummary,
    other_player,
    validate_arena_config,
)
from great_kingdom_ai.evaluator import (
    evaluate_feature_arrays_logits_values,
    evaluate_feature_batch,
    evaluate_feature_batch_logits_values,
)
from great_kingdom_ai.onnx_export import export_checkpoint_to_onnx
from great_kingdom_ai.self_play import SelfPlayState, create_core_game_state


def _runtime() -> ArenaRuntime:
    return ArenaRuntime(
        create_game_state=create_core_game_state,
        create_search_engine=lambda config, offset: create_core_search_engine(
            config,
            seed_offset=offset,
        ),
        create_batch=lambda config, count, seed_start, index_start: create_core_arena_batch(
            config,
            game_count=count,
            seed_start=seed_start,
            game_index_start=index_start,
        ),
        evaluate_feature_batch=evaluate_feature_batch,
        evaluate_feature_batch_logits_values=evaluate_feature_batch_logits_values,
        evaluate_feature_arrays_logits_values=evaluate_feature_arrays_logits_values,
    )


def play_arena_game(
    *,
    seed: int,
    candidate_model: Any,
    best_model: Any,
    candidate_player: int,
    config: ArenaConfig,
    state: SelfPlayState | None = None,
    search_factory: Any | None = None,
) -> ArenaGameResult:
    return _play_arena_game(
        seed=seed,
        candidate_model=candidate_model,
        best_model=best_model,
        candidate_player=candidate_player,
        config=config,
        state=state,
        search_factory=search_factory,
        runtime=_runtime(),
    )


def run_arena(
    *,
    candidate_model: Any,
    best_model: Any,
    config: ArenaConfig | None = None,
    state_factory: Any | None = None,
    search_factory: Any | None = None,
    progress_callback: Any | None = None,
) -> ArenaReport:
    return _run_arena(
        candidate_model=candidate_model,
        best_model=best_model,
        config=config,
        state_factory=state_factory,
        search_factory=search_factory,
        progress_callback=progress_callback,
        batched_runner=run_arena_batched,
        runtime=_runtime(),
    )


def run_arena_batched(
    *,
    candidate_model: Any,
    best_model: Any,
    config: ArenaConfig | None = None,
    onnx_evaluators: ArenaOnnxEvaluators | None = None,
    progress_callback: Any | None = None,
) -> ArenaReport:
    return _run_arena_batched(
        candidate_model=candidate_model,
        best_model=best_model,
        config=config,
        onnx_evaluators=onnx_evaluators,
        progress_callback=progress_callback,
        runtime=_runtime(),
    )


def evaluate_state_policy(
    model: Any,
    state: SelfPlayState,
    *,
    device: Any | str | None = None,
) -> list[float]:
    return evaluate_state_policies(model, [state], device=device)[0]


def evaluate_state_policy_logits(
    model: Any,
    state: SelfPlayState,
    *,
    device: Any | str | None = None,
) -> list[float]:
    return evaluate_state_policy_logits_batch(model, [state], device=device)[0]


def evaluate_state_policies(
    model: Any,
    states: list[SelfPlayState],
    *,
    device: Any | str | None = None,
) -> list[list[float]]:
    return _evaluate_state_policies(model, states, device=device, runtime=_runtime())


def evaluate_state_policy_logits_batch(
    model: Any,
    states: list[SelfPlayState],
    *,
    device: Any | str | None = None,
) -> list[list[float]]:
    return _evaluate_state_policy_logits_batch(
        model,
        states,
        device=device,
        runtime=_runtime(),
    )


def save_arena_report(report: ArenaReport, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as file:
        json.dump(report.to_dict(), file, indent=2, sort_keys=True)
    return destination


def promote_candidate_if_needed(
    *,
    candidate_checkpoint: str | Path,
    best_checkpoint: str | Path,
    report: ArenaReport,
) -> bool:
    if not report.summary.promoted:
        return False
    source = Path(candidate_checkpoint)
    destination = Path(best_checkpoint)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return True


def load_model_from_checkpoint(
    path: str | Path,
    *,
    device: str = "cpu",
    prefer_ema: bool = True,
) -> Any:
    from great_kingdom_ai.training import load_checkpoint

    state = load_checkpoint(path, device=device, prefer_ema=prefer_ema)
    state.model.eval()
    return state.model


def create_onnx_evaluator(
    path: str | Path,
    *,
    device: str = "cpu",
    max_batch_size: int = 8192,
) -> Any:
    core = game_core.import_core("ONNX arena")
    if not hasattr(core, "OnnxEvaluator"):
        raise RuntimeError(
            "great_kingdom_core.OnnxEvaluator is not available. "
            "Rebuild the Rust extension with ONNX support."
        )
    return core.OnnxEvaluator(str(path), device=device, max_batch_size=max_batch_size)


def run_arena_onnx(
    *,
    candidate_onnx_path: str | Path,
    best_onnx_path: str | Path,
    config: ArenaConfig | None = None,
    onnx_max_batch_size: int = 8192,
    progress_callback: Any | None = None,
) -> ArenaReport:
    config = config if config is not None else ArenaConfig()
    validate_arena_config(config)
    if config.batch_size <= 1:
        config = ArenaConfig(**{**asdict(config), "batch_size": max(1, config.games)})
    evaluators = ArenaOnnxEvaluators(
        candidate=create_onnx_evaluator(
            candidate_onnx_path,
            device=config.device,
            max_batch_size=onnx_max_batch_size,
        ),
        best=create_onnx_evaluator(
            best_onnx_path,
            device=config.device,
            max_batch_size=onnx_max_batch_size,
        ),
    )
    return run_arena_batched(
        candidate_model=None,
        best_model=None,
        config=config,
        onnx_evaluators=evaluators,
        progress_callback=progress_callback,
    )


def run_arena_checkpoints_onnx(
    *,
    candidate_checkpoint: str | Path,
    best_checkpoint: str | Path,
    config: ArenaConfig | None = None,
    onnx_max_batch_size: int = 8192,
    onnx_precision: str = "fp16",
    progress_callback: Any | None = None,
) -> ArenaReport:
    config = config if config is not None else ArenaConfig()
    with tempfile.TemporaryDirectory(prefix="gka-arena-onnx-") as temp_dir:
        temp_path = Path(temp_dir)
        candidate_onnx = _arena_onnx_path(
            candidate_checkpoint,
            temp_path / "candidate.onnx",
            device=config.device,
            precision=onnx_precision,
        )
        best_onnx = _arena_onnx_path(
            best_checkpoint,
            temp_path / "best.onnx",
            device=config.device,
            precision=onnx_precision,
        )
        return run_arena_onnx(
            candidate_onnx_path=candidate_onnx,
            best_onnx_path=best_onnx,
            config=config,
            onnx_max_batch_size=onnx_max_batch_size,
            progress_callback=progress_callback,
        )


def create_core_search_engine(
    config: ArenaConfig,
    *,
    seed_offset: int = 0,
) -> ArenaSearchLike:
    return cast(
        ArenaSearchLike,
        game_core.create_core_gumbel_search(config, seed_offset=seed_offset),
    )


def create_core_arena_batch(
    config: ArenaConfig,
    *,
    game_count: int,
    seed_start: int,
    game_index_start: int = 0,
) -> ArenaBatchLike:
    return cast(
        ArenaBatchLike,
        game_core.create_core_arena_batch(
            config,
            game_count=game_count,
            seed_start=seed_start,
            game_index_start=game_index_start,
        ),
    )


def load_arena_config(path: str | Path) -> ArenaConfig:
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError("arena config must be a JSON object")
    return ArenaConfig(**data)


def _arena_onnx_path(
    checkpoint_or_onnx: str | Path,
    output_path: Path,
    *,
    device: str,
    precision: str,
) -> Path:
    source = Path(checkpoint_or_onnx)
    if source.suffix == ".onnx":
        return source
    export_checkpoint_to_onnx(
        source,
        output_path,
        device=device,
        precision=precision,
    )
    return output_path


def _other_player(player: int) -> int:
    return other_player(player)


def _validate_arena_config(config: ArenaConfig) -> None:
    validate_arena_config(config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a candidate model against the best model"
    )
    parser.add_argument("--candidate", type=Path, required=True, help="Candidate checkpoint path")
    parser.add_argument("--best", type=Path, required=True, help="Best checkpoint path")
    parser.add_argument("--report", type=Path, required=True, help="Output arena report JSON")
    parser.add_argument("--config", type=Path, default=None, help="JSON ArenaConfig override")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--games", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--gumbel-simulations", type=int, default=None)
    parser.add_argument(
        "--gumbel-max-considered-actions",
        "--gumbel-max-consider",
        dest="gumbel_max_considered_actions",
        type=int,
        default=None,
    )
    parser.add_argument("--gumbel-c-visit", type=float, default=None)
    parser.add_argument("--gumbel-c-scale", type=float, default=None)
    parser.add_argument("--gumbel-scale", type=float, default=None)
    parser.add_argument("--opening-gumbel-turns", type=int, default=None)
    parser.add_argument("--opening-gumbel-scale", type=float, default=None)
    parser.add_argument("--policy-target-c-visit", type=float, default=None)
    parser.add_argument("--policy-target-c-scale", type=float, default=None)
    parser.add_argument("--policy-target-temperature", type=float, default=None)
    parser.add_argument("--gumbel-seed", type=int, default=None)
    parser.add_argument(
        "--paired-seeds",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use each arena seed for a candidate-Blue/candidate-Orange pair.",
    )
    parser.add_argument("--leaf-batch-size", type=int, default=None)
    parser.add_argument("--promotion-threshold", type=float, default=None)
    parser.add_argument(
        "--backend",
        choices=["onnx", "pytorch"],
        default="onnx",
        help="arena inference backend; ONNX uses the Rust evaluator path",
    )
    parser.add_argument("--onnx-max-batch-size", type=int, default=8192)
    parser.add_argument("--onnx-precision", choices=["fp32", "fp16"], default=None)
    parser.add_argument("--promote", action="store_true")
    return parser


def _config_from_args(args: argparse.Namespace) -> ArenaConfig:
    config = load_arena_config(args.config) if args.config is not None else ArenaConfig()
    overrides = {
        "device": args.device,
        "games": args.games,
        "batch_size": args.batch_size,
        "seed_start": args.seed_start,
        "max_turns": args.max_turns,
        "gumbel_simulations": args.gumbel_simulations,
        "gumbel_max_considered_actions": args.gumbel_max_considered_actions,
        "gumbel_c_visit": args.gumbel_c_visit,
        "gumbel_c_scale": args.gumbel_c_scale,
        "gumbel_scale": args.gumbel_scale,
        "opening_gumbel_turns": args.opening_gumbel_turns,
        "opening_gumbel_scale": args.opening_gumbel_scale,
        "policy_target_c_visit": args.policy_target_c_visit,
        "policy_target_c_scale": args.policy_target_c_scale,
        "policy_target_temperature": args.policy_target_temperature,
        "gumbel_seed": args.gumbel_seed,
        "paired_seeds": args.paired_seeds,
        "leaf_batch_size": args.leaf_batch_size,
        "promotion_threshold": args.promotion_threshold,
    }
    data = asdict(config)
    data.update({key: value for key, value in overrides.items() if value is not None})
    return ArenaConfig(**data)


def main() -> NoReturn:
    args = build_parser().parse_args()
    config = _config_from_args(args)
    if args.backend == "onnx":
        precision = args.onnx_precision or ("fp16" if config.device == "cuda" else "fp32")
        report = run_arena_checkpoints_onnx(
            candidate_checkpoint=args.candidate,
            best_checkpoint=args.best,
            config=config,
            onnx_max_batch_size=args.onnx_max_batch_size,
            onnx_precision=precision,
        )
    else:
        candidate_model = load_model_from_checkpoint(args.candidate, device=config.device)
        best_model = load_model_from_checkpoint(args.best, device=config.device)
        report = run_arena(candidate_model=candidate_model, best_model=best_model, config=config)
    save_arena_report(report, args.report)
    promoted = (
        promote_candidate_if_needed(
            candidate_checkpoint=args.candidate,
            best_checkpoint=args.best,
            report=report,
        )
        if args.promote
        else False
    )
    print(
        json.dumps(
            {
                "report": str(args.report),
                "promoted": promoted,
                "summary": report.summary.to_dict(),
            },
            sort_keys=True,
        )
    )
    raise SystemExit(0)

__all__ = [
    "BLUE",
    "ORANGE",
    "ArenaBatchLike",
    "ArenaConfig",
    "ArenaGameResult",
    "ArenaOnnxEvaluators",
    "ArenaReport",
    "ArenaSearchLike",
    "ArenaSearchResultLike",
    "ArenaSummary",
    "create_core_arena_batch",
    "create_onnx_evaluator",
    "create_core_search_engine",
    "evaluate_state_policy",
    "evaluate_state_policy_logits",
    "evaluate_state_policy_logits_batch",
    "load_arena_config",
    "load_model_from_checkpoint",
    "play_arena_game",
    "promote_candidate_if_needed",
    "run_arena",
    "run_arena_batched",
    "run_arena_checkpoints_onnx",
    "run_arena_onnx",
    "save_arena_report",
    "summarize_arena",
]
