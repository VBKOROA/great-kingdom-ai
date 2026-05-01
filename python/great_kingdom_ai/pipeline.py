"""End-to-end self-play, training, and arena promotion orchestration."""

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
    ArenaReport,
    evaluate_state_policy,
    load_arena_config,
    load_model_from_checkpoint,
    promote_candidate_if_needed,
    run_arena,
    save_arena_report,
)
from great_kingdom_ai.replay_buffer import ReplayBuffer, ReplaySample
from great_kingdom_ai.self_play import (
    GameLog,
    MctsSelfPlayConfig,
    create_core_mcts_search,
    play_mcts_game,
)
from great_kingdom_ai.train import (
    TrainingConfig,
    create_train_state,
    load_training_config,
    save_checkpoint,
    train_from_replay,
)


@dataclass(frozen=True)
class PipelineConfig:
    work_dir: Path = Path("data/pipeline")
    iterations: int = 1
    self_play_games: int = 2
    min_replay_samples: int = 4
    max_self_play_games: int = 20
    seed_start: int = 0
    replay_capacity: int = 10000
    mcts_simulations: int = 8
    mcts_c_puct: float = 1.5
    self_play_max_turns: int = 200
    temperature_turns: int = 10
    sampling_temperature: float = 1.0
    root_noise: bool = True
    promote: bool = True
    skip_arena: bool = False
    resume: bool = True


@dataclass(frozen=True)
class PipelineArtifacts:
    replay_path: Path
    self_play_log_path: Path
    candidate_checkpoint: Path
    best_checkpoint: Path
    arena_report_path: Path | None
    metrics_path: Path

    def to_dict(self) -> dict[str, str | None]:
        return {
            "replay": str(self.replay_path),
            "self_play_logs": str(self.self_play_log_path),
            "candidate_checkpoint": str(self.candidate_checkpoint),
            "best_checkpoint": str(self.best_checkpoint),
            "arena_report": str(self.arena_report_path) if self.arena_report_path else None,
            "metrics": str(self.metrics_path),
        }


@dataclass(frozen=True)
class PipelineIterationSummary:
    iteration: int
    self_play_games: int
    new_samples: int
    replay_samples: int
    train_start_step: int
    train_end_step: int
    candidate_win_rate: float | None
    promoted: bool
    candidate_checkpoint: Path
    arena_report_path: Path | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "self_play_games": self.self_play_games,
            "new_samples": self.new_samples,
            "replay_samples": self.replay_samples,
            "train_start_step": self.train_start_step,
            "train_end_step": self.train_end_step,
            "candidate_win_rate": self.candidate_win_rate,
            "promoted": self.promoted,
            "candidate_checkpoint": str(self.candidate_checkpoint),
            "arena_report": str(self.arena_report_path) if self.arena_report_path else None,
        }


@dataclass(frozen=True)
class PipelineSummary:
    iterations: list[PipelineIterationSummary]
    self_play_games: int
    replay_samples: int
    train_start_step: int
    train_end_step: int
    candidate_win_rate: float | None
    promoted: bool
    artifacts: PipelineArtifacts

    def to_dict(self) -> dict[str, Any]:
        return {
            "iterations": [iteration.to_dict() for iteration in self.iterations],
            "self_play_games": self.self_play_games,
            "replay_samples": self.replay_samples,
            "train_start_step": self.train_start_step,
            "train_end_step": self.train_end_step,
            "candidate_win_rate": self.candidate_win_rate,
            "promoted": self.promoted,
            "artifacts": self.artifacts.to_dict(),
        }


class PipelinePrinter:
    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled

    def title(self, text: str) -> None:
        if self.enabled:
            print(f"\n== {text} ==")

    def step(self, text: str) -> None:
        if self.enabled:
            print(f"  -> {text}")

    def done(self, text: str) -> None:
        if self.enabled:
            print(f"  ok {text}")

    def metric(self, key: str, value: object) -> None:
        if self.enabled:
            print(f"  {key:<18} {value}")


def run_pipeline(
    *,
    pipeline_config: PipelineConfig,
    train_config: TrainingConfig,
    arena_config: ArenaConfig,
    printer: PipelinePrinter | None = None,
    self_play_runner: Callable[[int, MctsSelfPlayConfig], tuple[GameLog, list[ReplaySample]]]
    | None = None,
) -> PipelineSummary:
    _validate_pipeline_config(pipeline_config)
    printer = printer if printer is not None else PipelinePrinter()
    paths = _pipeline_paths(pipeline_config)
    _ensure_pipeline_dirs(pipeline_config)

    if not paths.best_checkpoint.exists():
        printer.title("Bootstrap")
        printer.step(f"initializing best checkpoint at {paths.best_checkpoint}")
        save_checkpoint(create_train_state(train_config), paths.best_checkpoint)

    replay = _load_or_create_replay(paths.replay_path, pipeline_config)
    saved_log_dicts = _load_saved_log_dicts(paths.self_play_log_path, pipeline_config)
    seed_cursor = pipeline_config.seed_start + len(saved_log_dicts)
    iteration_summaries: list[PipelineIterationSummary] = []

    for iteration in range(1, pipeline_config.iterations + 1):
        printer.title(f"Iteration {iteration}/{pipeline_config.iterations}")

        printer.step("loading best model for self-play")
        self_play_model = (
            None
            if self_play_runner is not None
            else load_model_from_checkpoint(paths.best_checkpoint, device=train_config.device)
        )
        prior_provider: Callable[[Any], list[float]] | None = None
        if self_play_model is not None:

            def prior_provider(state: Any, model: Any = self_play_model) -> list[float]:
                return evaluate_state_policy(
                    model,
                    state,
                    device=train_config.device,
                )

        logs, samples = generate_self_play_samples(
            pipeline_config=pipeline_config,
            seed_start=seed_cursor,
            runner=self_play_runner,
            printer=printer,
            prior_provider=prior_provider,
        )
        seed_cursor += len(logs)
        replay.extend(samples)
        replay.save(paths.replay_path)
        saved_log_dicts.extend(log.to_dict() for log in logs)
        _save_json(paths.self_play_log_path, saved_log_dicts)
        printer.metric("new games", len(logs))
        printer.metric("new samples", len(samples))
        printer.metric("replay samples", len(replay))

        candidate_checkpoint = _iteration_candidate_checkpoint(paths, iteration)
        printer.step(f"training candidate -> {candidate_checkpoint}")
        train_summary = train_from_replay(
            replay,
            train_config,
            checkpoint_path=candidate_checkpoint,
            log_every=max(1, train_config.steps),
        )
        shutil.copy2(candidate_checkpoint, paths.candidate_checkpoint)
        printer.metric("train steps", f"{train_summary.start_step}->{train_summary.end_step}")

        report: ArenaReport | None = None
        promoted = False
        arena_report_path: Path | None = None
        if not pipeline_config.skip_arena:
            arena_report_path = _iteration_arena_report_path(paths, iteration)
            printer.step(f"arena evaluation -> {arena_report_path}")
            candidate_model = load_model_from_checkpoint(
                candidate_checkpoint,
                device=arena_config.device,
            )
            best_model = load_model_from_checkpoint(
                paths.best_checkpoint,
                device=arena_config.device,
            )
            report = run_arena(
                candidate_model=candidate_model,
                best_model=best_model,
                config=arena_config,
            )
            save_arena_report(report, arena_report_path)
            if paths.arena_report_path is not None:
                shutil.copy2(arena_report_path, paths.arena_report_path)
            promoted = (
                promote_candidate_if_needed(
                    candidate_checkpoint=candidate_checkpoint,
                    best_checkpoint=paths.best_checkpoint,
                    report=report,
                )
                if pipeline_config.promote
                else False
            )
            printer.metric("win rate", f"{report.summary.candidate_win_rate:.3f}")
            printer.metric("promoted", promoted)

        iteration_summary = PipelineIterationSummary(
            iteration=iteration,
            self_play_games=len(logs),
            new_samples=len(samples),
            replay_samples=len(replay),
            train_start_step=train_summary.start_step,
            train_end_step=train_summary.end_step,
            candidate_win_rate=(
                report.summary.candidate_win_rate if report is not None else None
            ),
            promoted=promoted,
            candidate_checkpoint=candidate_checkpoint,
            arena_report_path=arena_report_path,
        )
        iteration_summaries.append(iteration_summary)
        _append_metrics(paths.metrics_path, iteration_summary)

    last = iteration_summaries[-1]
    return PipelineSummary(
        iterations=iteration_summaries,
        self_play_games=sum(iteration.self_play_games for iteration in iteration_summaries),
        replay_samples=len(replay),
        train_start_step=last.train_start_step,
        train_end_step=last.train_end_step,
        candidate_win_rate=last.candidate_win_rate,
        promoted=last.promoted,
        artifacts=paths,
    )


def generate_self_play_samples(
    *,
    pipeline_config: PipelineConfig,
    seed_start: int | None = None,
    runner: Callable[[int, MctsSelfPlayConfig], tuple[GameLog, list[ReplaySample]]] | None = None,
    printer: PipelinePrinter | None = None,
    prior_provider: Callable[[Any], list[float]] | None = None,
) -> tuple[list[GameLog], list[ReplaySample]]:
    printer = printer if printer is not None else PipelinePrinter()
    config = MctsSelfPlayConfig(
        max_turns=pipeline_config.self_play_max_turns,
        temperature_turns=pipeline_config.temperature_turns,
        sampling_temperature=pipeline_config.sampling_temperature,
        root_noise=pipeline_config.root_noise,
    )
    def default_runner(
        seed: int,
        game_config: MctsSelfPlayConfig,
    ) -> tuple[GameLog, list[ReplaySample]]:
        search = create_core_mcts_search(
            simulations=pipeline_config.mcts_simulations,
            c_puct=pipeline_config.mcts_c_puct,
        )
        return play_mcts_game(
            seed=seed,
            search=search,
            config=game_config,
            prior_provider=prior_provider,
        )

    run_one = runner if runner is not None else default_runner

    logs: list[GameLog] = []
    samples: list[ReplaySample] = []
    seed = pipeline_config.seed_start if seed_start is None else seed_start
    while (
        len(logs) < pipeline_config.self_play_games
        or len(samples) < pipeline_config.min_replay_samples
    ):
        if len(logs) >= pipeline_config.max_self_play_games:
            raise RuntimeError(
                "self-play did not produce enough replay samples: "
                f"{len(samples)} < {pipeline_config.min_replay_samples}"
            )
        printer.step(f"game seed={seed}")
        log, game_samples = run_one(seed, config)
        logs.append(log)
        samples.extend(game_samples)
        seed += 1
    return logs, samples


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError("pipeline config must be a JSON object")
    if "work_dir" in data:
        data["work_dir"] = Path(data["work_dir"])
    return PipelineConfig(**data)


def _pipeline_paths(config: PipelineConfig) -> PipelineArtifacts:
    return PipelineArtifacts(
        replay_path=config.work_dir / "replay" / "replay.npz",
        self_play_log_path=config.work_dir / "replay" / "game_logs.json",
        candidate_checkpoint=config.work_dir / "checkpoints" / "candidate.pt",
        best_checkpoint=config.work_dir / "checkpoints" / "best.pt",
        arena_report_path=None
        if config.skip_arena
        else config.work_dir / "reports" / "arena-report.json",
        metrics_path=config.work_dir / "reports" / "metrics.jsonl",
    )


def _ensure_pipeline_dirs(config: PipelineConfig) -> None:
    (config.work_dir / "replay").mkdir(parents=True, exist_ok=True)
    (config.work_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (config.work_dir / "checkpoints" / "candidates").mkdir(parents=True, exist_ok=True)
    (config.work_dir / "reports").mkdir(parents=True, exist_ok=True)
    if not config.skip_arena:
        (config.work_dir / "reports" / "arena").mkdir(parents=True, exist_ok=True)


def _validate_pipeline_config(config: PipelineConfig) -> None:
    if config.iterations <= 0:
        raise ValueError("iterations must be positive")
    if config.self_play_games < 0:
        raise ValueError("self_play_games must be non-negative")
    if config.min_replay_samples < 0:
        raise ValueError("min_replay_samples must be non-negative")
    if config.max_self_play_games < config.self_play_games:
        raise ValueError("max_self_play_games must be at least self_play_games")
    if config.replay_capacity <= 0:
        raise ValueError("replay_capacity must be positive")
    if config.mcts_simulations <= 0:
        raise ValueError("mcts_simulations must be positive")


def _load_or_create_replay(path: Path, config: PipelineConfig) -> ReplayBuffer:
    if config.resume and path.exists():
        return ReplayBuffer.load(path)
    return ReplayBuffer(capacity=config.replay_capacity)


def _load_saved_log_dicts(path: Path, config: PipelineConfig) -> list[dict[str, Any]]:
    if not config.resume or not path.exists():
        return []
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError("self-play log file must contain a list")
    return [dict(item) for item in data if isinstance(item, dict)]


def _iteration_candidate_checkpoint(paths: PipelineArtifacts, iteration: int) -> Path:
    return paths.candidate_checkpoint.parent / "candidates" / f"candidate-{iteration:06d}.pt"


def _iteration_arena_report_path(paths: PipelineArtifacts, iteration: int) -> Path:
    return paths.metrics_path.parent / "arena" / f"arena-{iteration:06d}.json"


def _save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, sort_keys=True)


def _append_metrics(path: Path, summary: PipelineIterationSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(summary.to_dict(), sort_keys=True))
        file.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="great-kingdom-pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Run Great Kingdom self-play, training, arena evaluation, and promotion.",
        epilog=(
            "Example:\n"
            "  great-kingdom-pipeline --allow-cpu --pipeline-config configs/pipeline-smoke.json"
        ),
    )
    config_group = parser.add_argument_group("config files")
    config_group.add_argument(
        "--pipeline-config",
        type=Path,
        default=None,
        help="PipelineConfig JSON",
    )
    config_group.add_argument(
        "--train-config",
        type=Path,
        default=Path("configs/m8-train-smoke.json"),
        help="TrainingConfig JSON",
    )
    config_group.add_argument(
        "--arena-config",
        type=Path,
        default=Path("configs/m9-arena-smoke.json"),
        help="ArenaConfig JSON",
    )

    run_group = parser.add_argument_group("run controls")
    run_group.add_argument("--work-dir", type=Path, default=None, help="artifact root directory")
    run_group.add_argument("--iterations", type=int, default=None, help="training generations")
    run_group.add_argument(
        "--fresh",
        action="store_true",
        help="ignore existing replay/logs in work-dir",
    )
    run_group.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help="train/eval device",
    )
    run_group.add_argument(
        "--allow-cpu",
        action="store_true",
        help="fall back to CPU when CUDA is unavailable",
    )
    run_group.add_argument(
        "--no-promote",
        action="store_true",
        help="never replace best checkpoint",
    )
    run_group.add_argument(
        "--skip-arena",
        action="store_true",
        help="stop after candidate training",
    )
    run_group.add_argument("--json", action="store_true", help="print only machine-readable JSON")

    self_play_group = parser.add_argument_group("self-play")
    self_play_group.add_argument(
        "--self-play-games",
        type=int,
        default=None,
        help="minimum games per iteration",
    )
    self_play_group.add_argument(
        "--min-replay-samples",
        type=int,
        default=None,
        help="minimum new replay samples per iteration",
    )
    self_play_group.add_argument(
        "--mcts-simulations",
        type=int,
        default=None,
        help="MCTS simulations per self-play move",
    )

    train_group = parser.add_argument_group("training and arena")
    train_group.add_argument("--train-steps", type=int, default=None, help="steps per iteration")
    train_group.add_argument(
        "--arena-games",
        type=int,
        default=None,
        help="arena games per iteration",
    )
    return parser


def _configs_from_args(
    args: argparse.Namespace,
) -> tuple[PipelineConfig, TrainingConfig, ArenaConfig]:
    pipeline = (
        load_pipeline_config(args.pipeline_config)
        if args.pipeline_config is not None
        else PipelineConfig()
    )
    train = load_training_config(args.train_config)
    arena = load_arena_config(args.arena_config)

    pipeline_data = asdict(pipeline)
    for key, value in {
        "work_dir": args.work_dir,
        "iterations": args.iterations,
        "self_play_games": args.self_play_games,
        "min_replay_samples": args.min_replay_samples,
        "mcts_simulations": args.mcts_simulations,
        "promote": False if args.no_promote else None,
        "skip_arena": True if args.skip_arena else None,
        "resume": False if args.fresh else None,
    }.items():
        if value is not None:
            pipeline_data[key] = value
    pipeline = PipelineConfig(**pipeline_data)

    train_data = asdict(train)
    arena_data = asdict(arena)
    if args.train_steps is not None:
        train_data["steps"] = args.train_steps
    if args.arena_games is not None:
        arena_data["games"] = args.arena_games
    if args.device is not None:
        train_data["device"] = args.device
        arena_data["device"] = args.device

    if args.allow_cpu:
        train_data["device"] = _cpu_if_cuda_unavailable(str(train_data["device"]))
        arena_data["device"] = _cpu_if_cuda_unavailable(str(arena_data["device"]))

    return pipeline, TrainingConfig(**train_data), ArenaConfig(**arena_data)


def _cpu_if_cuda_unavailable(device: str) -> str:
    if device != "cuda":
        return device
    try:
        import torch  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def main() -> NoReturn:
    args = build_parser().parse_args()
    pipeline, train, arena = _configs_from_args(args)
    summary = run_pipeline(
        pipeline_config=pipeline,
        train_config=train,
        arena_config=arena,
        printer=PipelinePrinter(enabled=not args.json),
    )
    print(json.dumps(summary.to_dict(), indent=None if args.json else 2, sort_keys=True))
    raise SystemExit(0)


if __name__ == "__main__":
    main()


__all__ = [
    "PipelineArtifacts",
    "PipelineConfig",
    "PipelinePrinter",
    "PipelineSummary",
    "generate_self_play_samples",
    "load_pipeline_config",
    "run_pipeline",
]
