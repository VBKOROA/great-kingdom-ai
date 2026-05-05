"""Rust ONNX self-play pipeline orchestration."""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NoReturn

from great_kingdom_ai.evaluate import (
    ArenaConfig,
    load_arena_config,
    load_model_from_checkpoint,
    promote_candidate_if_needed,
    run_arena,
    save_arena_report,
)
from great_kingdom_ai.onnx_export import export_checkpoint_to_onnx
from great_kingdom_ai.pipeline import PipelinePrinter
from great_kingdom_ai.replay_aggregate import (
    aggregate_duplicate_replay,
    load_replay,
    save_replay,
)
from great_kingdom_ai.replay_buffer import ReplayBuffer
from great_kingdom_ai.rust_onnx_replay import (
    RustReplayImportSummary,
    import_legacy_pipeline_data,
    import_rust_self_play_artifacts,
)
from great_kingdom_ai.rust_onnx_self_play import (
    RustOnnxSelfPlayConfig,
    RustSelfPlayRunSummary,
    run_rust_onnx_self_play,
)
from great_kingdom_ai.self_play import SelfPlayConfig
from great_kingdom_ai.train import (
    TrainingConfig,
    create_train_state,
    load_training_config,
    save_checkpoint,
    train_from_replay,
)


@dataclass(frozen=True)
class RustOnnxPipelineConfig:
    work_dir: Path = Path("data/onnx-pipeline")
    iterations: int = 1
    legacy_import_dir: Path | None = None
    import_legacy_on_first_run: bool = False
    replay_capacity: int = 10000
    self_play_games: int = 2
    min_replay_samples: int = 0
    max_self_play_games: int | None = None
    seed_start: int = 0
    onnx_device: str = "cpu"
    onnx_max_batch_size: int = 128
    rust_self_play_batch_size: int = 2
    skip_arena: bool = True
    promote: bool = True
    resume: bool = True
    aggregate_replay: bool = True
    self_play: SelfPlayConfig = SelfPlayConfig()


@dataclass(frozen=True)
class RustOnnxPipelineIterationSummary:
    iteration: int
    onnx_model_path: Path
    self_play: RustSelfPlayRunSummary
    replay_import: RustReplayImportSummary
    train_start_step: int
    train_end_step: int
    candidate_checkpoint: Path
    candidate_win_rate: float | None
    promoted: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "onnx_model_path": str(self.onnx_model_path),
            "self_play": self.self_play.to_dict(),
            "replay_import": self.replay_import.to_dict(),
            "train_start_step": self.train_start_step,
            "train_end_step": self.train_end_step,
            "candidate_checkpoint": str(self.candidate_checkpoint),
            "candidate_win_rate": self.candidate_win_rate,
            "promoted": self.promoted,
        }


@dataclass(frozen=True)
class RustOnnxPipelineSummary:
    iterations: list[RustOnnxPipelineIterationSummary]
    replay_samples: int
    best_checkpoint: Path
    replay_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "iterations": [iteration.to_dict() for iteration in self.iterations],
            "replay_samples": self.replay_samples,
            "best_checkpoint": str(self.best_checkpoint),
            "replay_path": str(self.replay_path),
        }


def run_rust_onnx_pipeline(
    *,
    pipeline_config: RustOnnxPipelineConfig,
    train_config: TrainingConfig,
    arena_config: ArenaConfig,
    printer: PipelinePrinter | None = None,
    rust_self_play_runner: Callable[[RustOnnxSelfPlayConfig], RustSelfPlayRunSummary]
    | None = None,
) -> RustOnnxPipelineSummary:
    _validate_config(pipeline_config)
    printer = printer if printer is not None else PipelinePrinter()
    paths = _paths(pipeline_config)
    _ensure_dirs(pipeline_config)
    runner = rust_self_play_runner if rust_self_play_runner is not None else run_rust_onnx_self_play

    if (
        pipeline_config.import_legacy_on_first_run
        and pipeline_config.legacy_import_dir is not None
        and not paths["legacy_marker"].exists()
    ):
        printer.step("importing legacy pipeline artifacts")
        import_legacy_pipeline_data(
            legacy_work_dir=pipeline_config.legacy_import_dir,
            onnx_work_dir=pipeline_config.work_dir,
            replay_capacity=pipeline_config.replay_capacity,
        )

    if not paths["best_checkpoint"].exists():
        printer.step(f"initializing best checkpoint at {paths['best_checkpoint']}")
        save_checkpoint(create_train_state(train_config), paths["best_checkpoint"])

    completed_iterations = _load_completed_iteration_count(paths["metrics_path"], pipeline_config)
    seed_cursor = _initial_seed_cursor(pipeline_config)
    summaries: list[RustOnnxPipelineIterationSummary] = []
    first_iteration = completed_iterations + 1
    last_iteration = completed_iterations + pipeline_config.iterations

    printer.title("Rust ONNX Pipeline")
    printer.metric("work dir", pipeline_config.work_dir)
    printer.metric("resume", pipeline_config.resume)
    printer.metric("completed iterations", completed_iterations)
    printer.metric("train device", train_config.device)
    printer.metric("onnx device", pipeline_config.onnx_device)
    printer.metric(
        "self-play",
        (
            f"games>={pipeline_config.self_play_games}, "
            f"samples>={pipeline_config.min_replay_samples}, "
            f"batch={pipeline_config.rust_self_play_batch_size}, "
            f"onnx_batch={pipeline_config.onnx_max_batch_size}"
        ),
    )
    printer.metric("training", f"steps={train_config.steps}, batch={train_config.batch_size}")

    for iteration in range(
        first_iteration,
        last_iteration + 1,
    ):
        phase_total = 5 if pipeline_config.skip_arena else 6
        printer.title(f"Rust ONNX Iteration {iteration}/{last_iteration}")
        onnx_path = paths["onnx_checkpoint_dir"] / f"best-{iteration:06d}.onnx"
        printer.step(f"exporting best checkpoint -> {onnx_path}")
        export_checkpoint_to_onnx(paths["best_checkpoint"], onnx_path, device=train_config.device)
        printer.progress("iteration", 1, phase_total, detail="onnx export complete")

        artifact_root = paths["self_play_dir"] / f"iteration-{iteration:06d}"
        self_play_summary, replay_import, seed_cursor = _generate_and_import_self_play(
            pipeline_config=pipeline_config,
            onnx_path=onnx_path,
            artifact_root=artifact_root,
            replay_path=paths["replay_path"],
            game_log_path=paths["game_log_path"],
            seed_cursor=seed_cursor,
            runner=runner,
            printer=printer,
        )
        printer.metric("imported games", replay_import.imported_games)
        printer.metric("imported samples", replay_import.imported_samples)
        printer.progress("iteration", 2, phase_total, detail="self-play/import complete")
        printer.progress("iteration", 3, phase_total, detail="replay import complete")

        replay = _prepare_training_replay(
            paths=paths,
            replay_capacity=pipeline_config.replay_capacity,
            aggregate_replay=pipeline_config.aggregate_replay,
            printer=printer,
        )
        candidate_checkpoint = paths["candidate_dir"] / f"candidate-{iteration:06d}.pt"
        printer.metric("replay samples", len(replay))
        if pipeline_config.aggregate_replay:
            printer.metric("training replay", paths["aggregated_replay_path"])
        printer.step(f"training candidate -> {candidate_checkpoint}")
        train_summary = train_from_replay(
            replay,
            train_config,
            checkpoint_path=candidate_checkpoint,
            resume_path=None,
            bootstrap_weights_path=paths["best_checkpoint"],
            log_every=max(1, train_config.steps // 10),
            progress_callback=lambda current, target, loss: printer.progress(
                "train",
                current,
                target,
                detail=_format_train_loss_detail(loss),
            ),
        )
        shutil.copy2(candidate_checkpoint, paths["candidate_checkpoint"])
        printer.metric("train steps", f"{train_summary.start_step}->{train_summary.end_step}")
        printer.progress("iteration", 4, phase_total, detail="training complete")

        candidate_win_rate: float | None = None
        promoted = False
        if not pipeline_config.skip_arena:
            report_path = paths["arena_dir"] / f"arena-{iteration:06d}.json"
            printer.step(f"arena evaluation -> {report_path}")
            arena_search_config = _arena_config_for_pipeline(
                arena_config,
                iteration=iteration,
            )
            candidate_model = load_model_from_checkpoint(
                candidate_checkpoint,
                device=arena_config.device,
            )
            best_model = load_model_from_checkpoint(
                paths["best_checkpoint"],
                device=arena_config.device,
            )
            report = run_arena(
                candidate_model=candidate_model,
                best_model=best_model,
                config=arena_search_config,
                progress_callback=lambda current, target, game: printer.progress(
                    "arena games",
                    current,
                    target,
                    detail=f"winner={game.winner}, elapsed={printer.elapsed()}",
                ),
            )
            save_arena_report(report, report_path)
            candidate_win_rate = report.summary.candidate_win_rate
            printer.metric("candidate win rate", f"{candidate_win_rate:.3f}")
            promoted = (
                promote_candidate_if_needed(
                    candidate_checkpoint=candidate_checkpoint,
                    best_checkpoint=paths["best_checkpoint"],
                    report=report,
                )
                if pipeline_config.promote
                else False
            )
            printer.metric("promoted", promoted)
            printer.progress("iteration", 5, phase_total, detail="arena complete")
        else:
            printer.progress("iteration", 5, phase_total, detail="arena skipped")

        iteration_summary = RustOnnxPipelineIterationSummary(
            iteration=iteration,
            onnx_model_path=onnx_path,
            self_play=self_play_summary,
            replay_import=replay_import,
            train_start_step=train_summary.start_step,
            train_end_step=train_summary.end_step,
            candidate_checkpoint=candidate_checkpoint,
            candidate_win_rate=candidate_win_rate,
            promoted=promoted,
        )
        summaries.append(iteration_summary)
        _append_metrics(paths["metrics_path"], iteration_summary)
        printer.done(f"iteration {iteration} complete in {printer.elapsed()}")
        printer.progress("iteration", phase_total, phase_total, detail="metrics written")

    return RustOnnxPipelineSummary(
        iterations=summaries,
        replay_samples=len(ReplayBuffer.load(paths["replay_path"])),
        best_checkpoint=paths["best_checkpoint"],
        replay_path=paths["replay_path"],
    )


def load_rust_onnx_pipeline_config(path: str | Path) -> RustOnnxPipelineConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Rust ONNX pipeline config must be a JSON object")
    if "work_dir" in data:
        data["work_dir"] = Path(data["work_dir"])
    if data.get("legacy_import_dir") is not None:
        data["legacy_import_dir"] = Path(data["legacy_import_dir"])
    self_play_data = data.pop("self_play", None)
    if isinstance(self_play_data, dict):
        _require_policy_target_scale(self_play_data, "Rust ONNX self_play config")
        data["self_play"] = SelfPlayConfig(**self_play_data)
    elif self_play_data is None:
        raise ValueError("Rust ONNX pipeline config must set self_play")
    return RustOnnxPipelineConfig(**data)


def _require_policy_target_scale(data: dict[str, Any], label: str) -> None:
    missing = [
        key
        for key in ("policy_target_c_visit", "policy_target_c_scale")
        if key not in data
    ]
    if missing:
        raise ValueError(f"{label} must set {', '.join(missing)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="great-kingdom-rust-onnx-pipeline",
        description="Run the Rust ONNX self-play pipeline.",
    )
    parser.add_argument("--pipeline-config", type=Path, default=None)
    parser.add_argument(
        "--train-config",
        type=Path,
        default=Path("configs/test/m8-train-smoke.json"),
    )
    parser.add_argument(
        "--arena-config",
        type=Path,
        default=Path("configs/test/m9-arena-smoke.json"),
    )
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--onnx-device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--self-play-games", type=int, default=None)
    parser.add_argument("--min-replay-samples", type=int, default=None)
    parser.add_argument("--max-self-play-games", type=int, default=None)
    parser.add_argument("--skip-arena", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> NoReturn:
    args = build_parser().parse_args()
    config = (
        load_rust_onnx_pipeline_config(args.pipeline_config)
        if args.pipeline_config is not None
        else RustOnnxPipelineConfig()
    )
    data = asdict(config)
    for key, value in {
        "work_dir": args.work_dir,
        "iterations": args.iterations,
        "onnx_device": args.onnx_device if args.onnx_device is not None else args.device,
        "self_play_games": args.self_play_games,
        "min_replay_samples": args.min_replay_samples,
        "max_self_play_games": args.max_self_play_games,
        "skip_arena": True if args.skip_arena else None,
    }.items():
        if value is not None:
            data[key] = value
    if isinstance(data.get("self_play"), dict):
        data["self_play"] = SelfPlayConfig(**data["self_play"])
    config = RustOnnxPipelineConfig(**data)
    train = load_training_config(args.train_config)
    arena = load_arena_config(args.arena_config)
    if args.device is not None:
        train = TrainingConfig(**{**asdict(train), "device": args.device})
        arena = ArenaConfig(**{**asdict(arena), "device": args.device})
    summary = run_rust_onnx_pipeline(
        pipeline_config=config,
        train_config=train,
        arena_config=arena,
        printer=PipelinePrinter(enabled=not args.json),
    )
    print(json.dumps(summary.to_dict(), indent=None if args.json else 2, sort_keys=True))
    raise SystemExit(0)


def _paths(config: RustOnnxPipelineConfig) -> dict[str, Path]:
    return {
        "replay_path": config.work_dir / "replay" / "replay.npz",
        "aggregated_replay_path": config.work_dir / "replay" / "replay-aggregated.npz",
        "game_log_path": config.work_dir / "replay" / "game_logs.json",
        "best_checkpoint": config.work_dir / "checkpoints" / "best.pt",
        "candidate_checkpoint": config.work_dir / "checkpoints" / "candidate.pt",
        "candidate_dir": config.work_dir / "checkpoints" / "candidates",
        "onnx_checkpoint_dir": config.work_dir / "checkpoints" / "onnx",
        "self_play_dir": config.work_dir / "self-play",
        "metrics_path": config.work_dir / "reports" / "metrics.jsonl",
        "legacy_marker": config.work_dir / "reports" / "legacy-import.json",
        "arena_dir": config.work_dir / "reports" / "arena",
    }


def _ensure_dirs(config: RustOnnxPipelineConfig) -> None:
    for path in _paths(config).values():
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            path.mkdir(parents=True, exist_ok=True)


def _generate_and_import_self_play(
    *,
    pipeline_config: RustOnnxPipelineConfig,
    onnx_path: Path,
    artifact_root: Path,
    replay_path: Path,
    game_log_path: Path,
    seed_cursor: int,
    runner: Callable[[RustOnnxSelfPlayConfig], RustSelfPlayRunSummary],
    printer: PipelinePrinter,
) -> tuple[RustSelfPlayRunSummary, RustReplayImportSummary, int]:
    total_games = 0
    total_samples = 0
    imported_games = 0
    imported_samples = 0
    replay_samples = 0
    batch_index = 0

    while (
        total_games < pipeline_config.self_play_games
        or total_samples < pipeline_config.min_replay_samples
    ):
        if (
            pipeline_config.max_self_play_games is not None
            and total_games >= pipeline_config.max_self_play_games
        ):
            raise RuntimeError(
                "Rust ONNX self-play reached max_self_play_games="
                f"{pipeline_config.max_self_play_games} with samples="
                f"{total_samples} < {pipeline_config.min_replay_samples}"
            )

        game_count = _next_self_play_game_count(
            pipeline_config,
            total_games=total_games,
            total_samples=total_samples,
        )
        batch_index += 1
        artifact_dir = artifact_root / f"batch-{batch_index:03d}"
        printer.step(f"running Rust ONNX self-play -> {artifact_dir}")
        self_play_summary = runner(
            RustOnnxSelfPlayConfig(
                onnx_model_path=onnx_path,
                output_dir=artifact_dir,
                games=game_count,
                seed_start=seed_cursor,
                onnx_device=pipeline_config.onnx_device,
                onnx_max_batch_size=pipeline_config.onnx_max_batch_size,
                rust_self_play_batch_size=pipeline_config.rust_self_play_batch_size,
                self_play=pipeline_config.self_play,
            )
        )
        seed_cursor += self_play_summary.games
        total_games += self_play_summary.games
        total_samples += self_play_summary.samples
        printer.progress(
            "self-play games",
            total_games,
            pipeline_config.self_play_games,
            detail=f"samples={total_samples}, seed_next={seed_cursor}",
        )

        printer.step("importing Rust self-play artifacts into replay")
        replay_import = import_rust_self_play_artifacts(
            artifact_dir=self_play_summary.artifact_dir,
            replay_path=replay_path,
            replay_capacity=pipeline_config.replay_capacity,
            game_log_path=game_log_path,
        )
        imported_games += replay_import.imported_games
        imported_samples += replay_import.imported_samples
        replay_samples = replay_import.replay_samples

    return (
        RustSelfPlayRunSummary(
            artifact_dir=artifact_root,
            games=total_games,
            samples=total_samples,
            onnx_model_path=onnx_path,
            onnx_device=pipeline_config.onnx_device,
        ),
        RustReplayImportSummary(
            artifact_dir=artifact_root,
            replay_path=replay_path,
            imported_samples=imported_samples,
            replay_samples=replay_samples,
            imported_games=imported_games,
        ),
        seed_cursor,
    )


def _next_self_play_game_count(
    config: RustOnnxPipelineConfig,
    *,
    total_games: int,
    total_samples: int,
) -> int:
    remaining_min_games = max(1, config.self_play_games - total_games)
    if total_samples < config.min_replay_samples and total_games >= config.self_play_games:
        remaining_min_games = max(remaining_min_games, config.self_play_games)
    if config.max_self_play_games is not None:
        remaining_min_games = min(remaining_min_games, config.max_self_play_games - total_games)
    return max(1, remaining_min_games)


def _validate_config(config: RustOnnxPipelineConfig) -> None:
    if config.iterations <= 0:
        raise ValueError("iterations must be positive")
    if config.replay_capacity <= 0:
        raise ValueError("replay_capacity must be positive")
    if config.self_play_games < 0:
        raise ValueError("self_play_games must be non-negative")
    if config.min_replay_samples < 0:
        raise ValueError("min_replay_samples must be non-negative")
    if (
        config.max_self_play_games is not None
        and config.max_self_play_games < config.self_play_games
    ):
        raise ValueError("max_self_play_games must be at least self_play_games")
    if config.onnx_max_batch_size <= 0:
        raise ValueError("onnx_max_batch_size must be positive")
    if config.rust_self_play_batch_size <= 0:
        raise ValueError("rust_self_play_batch_size must be positive")


def _prepare_training_replay(
    *,
    paths: dict[str, Path],
    replay_capacity: int,
    aggregate_replay: bool,
    printer: PipelinePrinter,
) -> ReplayBuffer:
    if not aggregate_replay:
        return ReplayBuffer.load(paths["replay_path"])

    printer.step("aggregating duplicate replay states for training")
    features, policies, values, capacity = load_replay(paths["replay_path"])
    aggregated = aggregate_duplicate_replay(
        features=features,
        policies=policies,
        values=values,
        capacity=capacity if capacity is not None else replay_capacity,
    )
    save_replay(paths["aggregated_replay_path"], aggregated)
    printer.metric("raw replay samples", features.shape[0])
    printer.metric("aggregated samples", aggregated.features.shape[0])
    return ReplayBuffer.load(paths["aggregated_replay_path"])


def _arena_config_for_pipeline(
    arena_config: ArenaConfig,
    *,
    iteration: int,
) -> ArenaConfig:
    if iteration <= 0:
        raise ValueError("iteration must be positive")
    data = asdict(arena_config)
    data["seed_start"] = arena_config.seed_start + (iteration - 1) * arena_config.games
    return ArenaConfig(**data)


def _load_completed_iteration_count(path: Path, config: RustOnnxPipelineConfig) -> int:
    if not config.resume or not path.exists():
        return 0
    completed = 0
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if stripped:
                data = json.loads(stripped)
                completed = max(completed, int(data["iteration"]))
    return completed


def _initial_seed_cursor(config: RustOnnxPipelineConfig) -> int:
    logs_path = config.work_dir / "replay" / "game_logs.json"
    if not config.resume or not logs_path.exists():
        return config.seed_start
    data = json.loads(logs_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("game_logs.json must contain a list")
    seeds = [
        int(item["seed"])
        for item in data
        if isinstance(item, dict) and isinstance(item.get("seed"), int)
    ]
    return max(config.seed_start, max(seeds, default=config.seed_start - 1) + 1)


def _append_metrics(path: Path, summary: RustOnnxPipelineIterationSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(summary.to_dict(), sort_keys=True))
        file.write("\n")


def _format_train_loss_detail(loss: dict[str, float]) -> str:
    detail = f"loss={loss['total']:.4f}"
    if {"policy", "value", "policy_kl"}.issubset(loss):
        detail += (
            f" policy={loss['policy']:.4f}"
            f" value={loss['value']:.4f}"
            f" kl={loss['policy_kl']:.4f}"
        )
    return detail


if __name__ == "__main__":
    main()


__all__ = [
    "RustOnnxPipelineConfig",
    "RustOnnxPipelineSummary",
    "load_rust_onnx_pipeline_config",
    "run_rust_onnx_pipeline",
]
