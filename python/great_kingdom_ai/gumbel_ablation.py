"""Quick pure-vs-modified Gumbel training ablation."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, NoReturn

from great_kingdom_ai.evaluate import (
    ArenaConfig,
    load_arena_config,
    load_model_from_checkpoint,
    run_arena,
    save_arena_report,
)
from great_kingdom_ai.pipeline import PipelinePrinter
from great_kingdom_ai.rust_onnx_pipeline import (
    RustOnnxPipelineConfig,
    RustOnnxPipelineSummary,
    load_rust_onnx_pipeline_config,
    run_rust_onnx_pipeline,
)
from great_kingdom_ai.train import (
    TrainingConfig,
    create_train_state,
    load_training_config,
    save_checkpoint,
)

DEFAULT_PURE_PIPELINE_CONFIG = Path("configs/ablation/gumbel-pure-pipeline.json")
DEFAULT_MODIFIED_PIPELINE_CONFIG = Path("configs/ablation/gumbel-aggregate-pipeline.json")
DEFAULT_TRAIN_CONFIG = Path("configs/ablation/gumbel-train-runpod-fast.json")
DEFAULT_ARENA_CONFIG = Path("configs/ablation/gumbel-arena-runpod-fast.json")
DEFAULT_RUN_ROOT = Path("data/ablation/gumbel")


@dataclass(frozen=True)
class GumbelAblationSummary:
    run_dir: Path
    initial_checkpoint: Path
    pure_pipeline: RustOnnxPipelineSummary
    aggregate_pipeline: RustOnnxPipelineSummary
    arena_report: Path
    arena_summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_dir": str(self.run_dir),
            "initial_checkpoint": str(self.initial_checkpoint),
            "pure_pipeline": self.pure_pipeline.to_dict(),
            "aggregate_pipeline": self.aggregate_pipeline.to_dict(),
            "arena_report": str(self.arena_report),
            "arena_summary": self.arena_summary,
        }


def run_gumbel_ablation(
    *,
    pure_pipeline_config: RustOnnxPipelineConfig,
    modified_pipeline_config: RustOnnxPipelineConfig,
    train_config: TrainingConfig,
    arena_config: ArenaConfig,
    run_dir: Path,
    overwrite: bool = False,
    printer: PipelinePrinter | None = None,
) -> GumbelAblationSummary:
    printer = printer if printer is not None else PipelinePrinter()
    _prepare_run_dir(run_dir, overwrite=overwrite)
    initial_checkpoint = run_dir / "checkpoints" / "initial-best.pt"
    _save_initial_checkpoint(initial_checkpoint, train_config)

    pure_config = _variant_pipeline_config(
        pure_pipeline_config,
        run_dir / "pure",
        min_replay_samples=train_config.batch_size,
    )
    modified_config = _variant_pipeline_config(
        modified_pipeline_config,
        run_dir / "aggregate",
        min_replay_samples=train_config.batch_size,
    )

    printer.title("Pure Gumbel")
    _install_initial_checkpoint(initial_checkpoint, pure_config)
    pure_summary = run_rust_onnx_pipeline(
        pipeline_config=pure_config,
        train_config=train_config,
        arena_config=arena_config,
        printer=printer,
    )

    printer.title("Aggregate-only Gumbel")
    _install_initial_checkpoint(initial_checkpoint, modified_config)
    modified_summary = run_rust_onnx_pipeline(
        pipeline_config=modified_config,
        train_config=train_config,
        arena_config=arena_config,
        printer=printer,
    )

    pure_candidate = _last_candidate_checkpoint(pure_summary, "pure")
    modified_candidate = _last_candidate_checkpoint(modified_summary, "modified")
    report_path = run_dir / "reports" / "aggregate-vs-pure-arena.json"

    printer.title("Aggregate-only vs Pure Arena")
    report = run_arena(
        candidate_model=load_model_from_checkpoint(modified_candidate, device=arena_config.device),
        best_model=load_model_from_checkpoint(pure_candidate, device=arena_config.device),
        config=arena_config,
        progress_callback=lambda current, target, game: printer.progress(
            "arena games",
            current,
            target,
            detail=f"winner={game.winner}, aggregate_player={game.candidate_player}",
        ),
    )
    save_arena_report(report, report_path)
    summary = GumbelAblationSummary(
        run_dir=run_dir,
        initial_checkpoint=initial_checkpoint,
        pure_pipeline=pure_summary,
        aggregate_pipeline=modified_summary,
        arena_report=report_path,
        arena_summary=report.summary.to_dict(),
    )
    _write_json(run_dir / "reports" / "summary.json", summary.to_dict())
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a quick pure-vs-aggregate-only Gumbel training ablation and arena."
    )
    parser.add_argument("--pure-pipeline-config", type=Path, default=DEFAULT_PURE_PIPELINE_CONFIG)
    parser.add_argument(
        "--aggregate-pipeline-config",
        "--modified-pipeline-config",
        dest="modified_pipeline_config",
        type=Path,
        default=DEFAULT_MODIFIED_PIPELINE_CONFIG,
        metavar="AGGREGATE_PIPELINE_CONFIG",
    )
    parser.add_argument("--train-config", type=Path, default=DEFAULT_TRAIN_CONFIG)
    parser.add_argument("--arena-config", type=Path, default=DEFAULT_ARENA_CONFIG)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--onnx-device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--train-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--self-play-games", type=int, default=None)
    parser.add_argument("--min-replay-samples", type=int, default=None)
    parser.add_argument("--arena-games", type=int, default=None)
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> NoReturn:
    args = build_parser().parse_args()
    pure = load_rust_onnx_pipeline_config(args.pure_pipeline_config)
    modified = load_rust_onnx_pipeline_config(args.modified_pipeline_config)
    train = load_training_config(args.train_config)
    arena = load_arena_config(args.arena_config)

    pure, modified, train, arena = _apply_overrides(
        pure=pure,
        modified=modified,
        train=train,
        arena=arena,
        device=args.device,
        onnx_device=args.onnx_device,
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        self_play_games=args.self_play_games,
        min_replay_samples=args.min_replay_samples,
        arena_games=args.arena_games,
    )
    run_dir = args.run_dir if args.run_dir is not None else _default_run_dir()
    summary = run_gumbel_ablation(
        pure_pipeline_config=pure,
        modified_pipeline_config=modified,
        train_config=train,
        arena_config=arena,
        run_dir=run_dir,
        overwrite=args.overwrite,
        printer=PipelinePrinter(enabled=not args.json),
    )
    print(json.dumps(summary.to_dict(), indent=None if args.json else 2, sort_keys=True))
    raise SystemExit(0)


def _apply_overrides(
    *,
    pure: RustOnnxPipelineConfig,
    modified: RustOnnxPipelineConfig,
    train: TrainingConfig,
    arena: ArenaConfig,
    device: str | None,
    onnx_device: str | None,
    train_steps: int | None,
    batch_size: int | None,
    self_play_games: int | None,
    min_replay_samples: int | None,
    arena_games: int | None,
) -> tuple[RustOnnxPipelineConfig, RustOnnxPipelineConfig, TrainingConfig, ArenaConfig]:
    if device is not None:
        train = replace(train, device=device)
        arena = replace(arena, device=device)
    if train_steps is not None:
        train = replace(train, steps=train_steps)
    if batch_size is not None:
        train = replace(train, batch_size=batch_size)
    if arena_games is not None:
        arena = replace(arena, games=arena_games)

    pipeline_updates: dict[str, Any] = {}
    if onnx_device is not None:
        pipeline_updates["onnx_device"] = onnx_device
    elif device is not None:
        pipeline_updates["onnx_device"] = device
    if self_play_games is not None:
        pipeline_updates["self_play_games"] = self_play_games
        pipeline_updates["max_self_play_games"] = max(
            self_play_games,
            pure.max_self_play_games or self_play_games,
            modified.max_self_play_games or self_play_games,
        )
    if min_replay_samples is not None:
        pipeline_updates["min_replay_samples"] = min_replay_samples

    if pipeline_updates:
        pure = replace(pure, **pipeline_updates)
        modified = replace(modified, **pipeline_updates)
    return pure, modified, train, arena


def _prepare_run_dir(run_dir: Path, *, overwrite: bool) -> None:
    if run_dir.exists() and any(run_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{run_dir} already exists; pass --overwrite or choose a new dir")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)


def _save_initial_checkpoint(path: Path, config: TrainingConfig) -> Path:
    torch = _import_torch()
    torch.manual_seed(config.seed)
    return save_checkpoint(create_train_state(config), path)


def _install_initial_checkpoint(
    initial_checkpoint: Path,
    config: RustOnnxPipelineConfig,
) -> None:
    destination = config.work_dir / "checkpoints" / "best.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(initial_checkpoint, destination)


def _variant_pipeline_config(
    config: RustOnnxPipelineConfig,
    work_dir: Path,
    *,
    min_replay_samples: int,
) -> RustOnnxPipelineConfig:
    return replace(
        config,
        work_dir=work_dir,
        min_replay_samples=max(config.min_replay_samples, min_replay_samples),
        resume=False,
        skip_arena=True,
        promote=False,
    )


def _last_candidate_checkpoint(summary: RustOnnxPipelineSummary, label: str) -> Path:
    if not summary.iterations:
        raise ValueError(f"{label} pipeline did not run any iterations")
    return summary.iterations[-1].candidate_checkpoint


def _default_run_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return DEFAULT_RUN_ROOT / stamp


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _import_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required for Gumbel ablation") from exc
    return torch


if __name__ == "__main__":
    main()


__all__ = [
    "GumbelAblationSummary",
    "build_parser",
    "run_gumbel_ablation",
]
