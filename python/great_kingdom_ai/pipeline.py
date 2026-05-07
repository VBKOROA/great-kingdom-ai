"""End-to-end self-play, training, and arena promotion orchestration."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NoReturn

from great_kingdom_ai.evaluate import (
    ArenaConfig,
    ArenaReport,
    evaluate_state_policy_logits,
    evaluate_state_policy_logits_batch,
    load_arena_config,
    load_model_from_checkpoint,
    promote_candidate_if_needed,
    run_arena,
    save_arena_report,
)
from great_kingdom_ai.evaluator import (
    evaluate_feature_batch_logits_values,
    evaluate_request_bytes_logits_values,
)
from great_kingdom_ai.replay_buffer import ReplayBuffer, ReplaySample
from great_kingdom_ai.self_play import (
    GameLog,
    SelfPlayConfig,
    create_core_search_engine,
    play_self_play_game,
    play_self_play_games_batched,
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
    max_self_play_games: int | None = 20
    self_play_batch_size: int = 1
    seed_start: int = 0
    replay_capacity: int = 10000
    gumbel_simulations: int = 128
    gumbel_max_considered_actions: int = 16
    gumbel_c_visit: float = 50.0
    gumbel_c_scale: float = 1.0
    policy_target_c_visit: float = 5.0
    policy_target_c_scale: float = 0.25
    policy_target_temperature: float = 1.0
    gumbel_seed: int = 0
    leaf_batch_size: int = 8
    self_play_max_turns: int = 200
    temperature_turns: int = 10
    sampling_temperature: float = 1.0
    playout_cap_randomization: bool = False
    playout_cap_full_search_fraction: float = 0.25
    playout_cap_fast_simulations: int = 16
    promote: bool = True
    always_promote: bool = False
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
        self.started_at = time.monotonic()
        self.bar_width = 28

    def title(self, text: str) -> None:
        if self.enabled:
            print(f"\n== {text} ==", flush=True)

    def step(self, text: str) -> None:
        if self.enabled:
            print(f"  -> {text}", flush=True)

    def done(self, text: str) -> None:
        if self.enabled:
            print(f"  ok {text}", flush=True)

    def metric(self, key: str, value: object) -> None:
        if self.enabled:
            print(f"  {key:<18} {value}", flush=True)

    def progress(self, key: str, current: int, target: int, *, detail: str = "") -> None:
        if not self.enabled:
            return
        percent = 100.0 if target <= 0 else min(100.0, current / target * 100.0)
        filled = (
            self.bar_width
            if target <= 0
            else round(self.bar_width * min(current, target) / target)
        )
        bar = "#" * filled + "." * (self.bar_width - filled)
        suffix = f"  {detail}" if detail else ""
        print(
            f"  {key:<18} [{bar}] {current:>6}/{target:<6} {percent:>6.1f}%{suffix}",
            flush=True,
        )

    def elapsed(self) -> str:
        return _format_duration(time.monotonic() - self.started_at)


def run_pipeline(
    *,
    pipeline_config: PipelineConfig,
    train_config: TrainingConfig,
    arena_config: ArenaConfig,
    printer: PipelinePrinter | None = None,
    self_play_runner: Callable[[int, SelfPlayConfig], tuple[GameLog, list[ReplaySample]]]
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
    completed_iterations = _load_completed_iteration_count(paths.metrics_path, pipeline_config)
    first_iteration = completed_iterations + 1
    last_iteration = completed_iterations + pipeline_config.iterations
    seed_cursor = pipeline_config.seed_start + len(saved_log_dicts)
    iteration_summaries: list[PipelineIterationSummary] = []

    printer.title("Pipeline")
    printer.metric("work dir", pipeline_config.work_dir)
    printer.metric("resume", pipeline_config.resume)
    printer.metric("device", train_config.device)
    printer.metric("model", train_config.model_preset)
    printer.metric("replay samples", len(replay))
    printer.metric("saved games", len(saved_log_dicts))
    printer.metric("completed iterations", completed_iterations)
    printer.metric("self-play", _self_play_config_summary(pipeline_config))
    printer.metric("training", f"steps={train_config.steps}, batch={train_config.batch_size}")

    for iteration in range(first_iteration, last_iteration + 1):
        printer.title(f"Iteration {iteration}/{last_iteration}")

        printer.step("loading best model for self-play")
        self_play_model = (
            None
            if self_play_runner is not None
            else load_model_from_checkpoint(paths.best_checkpoint, device=train_config.device)
        )
        prior_provider: Callable[[Any], list[float]] | None = None
        batch_prior_provider: Callable[[Sequence[Any]], list[list[float]]] | None = None
        batch_evaluator_provider: Callable[
            [Sequence[Any]],
            tuple[list[list[float]], list[float]],
        ] | None = None
        feature_batch_prior_provider: Callable[
            [Sequence[Sequence[float]], Sequence[Sequence[bool]]],
            list[list[float]],
        ] | None = None
        request_evaluator_provider: Callable[
            [Any],
            tuple[Any, Any],
        ] | None = None
        if self_play_model is not None:

            def _prior_provider(state: Any, model: Any = self_play_model) -> list[float]:
                return evaluate_state_policy_logits(
                    model,
                    state,
                    device=train_config.device,
                )

            prior_provider = _prior_provider

            def _batch_prior_provider(
                states: Sequence[Any],
                model: Any = self_play_model,
            ) -> list[list[float]]:
                return evaluate_state_policy_logits_batch(
                    model,
                    states,
                    device=train_config.device,
                )

            batch_prior_provider = _batch_prior_provider

            def _feature_batch_prior_provider(
                feature_rows: Sequence[Sequence[float]],
                mask_rows: Sequence[Sequence[bool]],
                model: Any = self_play_model,
            ) -> list[list[float]]:
                evaluation = evaluate_feature_batch_logits_values(
                    model,
                    [list(row) for row in feature_rows],
                    [list(row) for row in mask_rows],
                    device=train_config.device,
                )
                policy_rows = evaluation.policy_logits
                return [[float(value) for value in policy] for policy in policy_rows]

            feature_batch_prior_provider = _feature_batch_prior_provider

            def _batch_evaluator_provider(
                states: Sequence[Any],
                model: Any = self_play_model,
            ) -> tuple[list[list[float]], list[float]]:
                evaluation = evaluate_feature_batch_logits_values(
                    model,
                    [state.feature_planes() for state in states],
                    [state.legal_mask() for state in states],
                    device=train_config.device,
                )
                policy_rows = evaluation.policy_logits
                return (
                    [[float(value) for value in policy] for policy in policy_rows],
                    [float(value) for value in evaluation.value],
                )

            batch_evaluator_provider = _batch_evaluator_provider

            def _request_evaluator_provider(
                request: Any,
                model: Any = self_play_model,
            ) -> tuple[Any, Any]:
                evaluation = evaluate_request_bytes_logits_values(
                    model,
                    request,
                    device=train_config.device,
                )
                policy_rows = evaluation.policy_logits
                return policy_rows, evaluation.value

            request_evaluator_provider = _request_evaluator_provider

        logs, samples = generate_self_play_samples(
            pipeline_config=pipeline_config,
            seed_start=seed_cursor,
            runner=self_play_runner,
            printer=printer,
            prior_provider=prior_provider,
            batch_prior_provider=batch_prior_provider,
            batch_evaluator_provider=batch_evaluator_provider,
            feature_batch_prior_provider=feature_batch_prior_provider,
            request_evaluator_provider=request_evaluator_provider,
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
            resume_path=paths.best_checkpoint,
            log_every=max(1, train_config.steps // 10),
            progress_callback=lambda current, target, loss: printer.progress(
                "train",
                current,
                target,
                detail=f"loss={loss['total']:.4f}",
            ),
        )
        shutil.copy2(candidate_checkpoint, paths.candidate_checkpoint)
        printer.metric("train steps", f"{train_summary.start_step}->{train_summary.end_step}")

        report: ArenaReport | None = None
        promoted = False
        arena_report_path: Path | None = None
        if _should_run_arena(pipeline_config):
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
            arena_search_config = _arena_config_for_pipeline(
                arena_config,
                pipeline_config,
                iteration=iteration,
            )
            report = run_arena(
                candidate_model=candidate_model,
                best_model=best_model,
                config=arena_search_config,
                progress_callback=lambda current, target, game: printer.progress(
                    "arena games",
                    current,
                    target,
                    detail=(
                        f"last_seed={game.seed}, moves={len(game.moves)}, "
                        f"winner={game.winner}, elapsed={printer.elapsed()}"
                    ),
                ),
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
        elif pipeline_config.always_promote:
            promoted = _promote_candidate_unconditionally(
                candidate_checkpoint=candidate_checkpoint,
                best_checkpoint=paths.best_checkpoint,
            )
            printer.metric("promoted", promoted)
            printer.step("arena skipped by always_promote")

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
    runner: Callable[[int, SelfPlayConfig], tuple[GameLog, list[ReplaySample]]] | None = None,
    printer: PipelinePrinter | None = None,
    prior_provider: Callable[[Any], list[float]] | None = None,
    batch_prior_provider: Callable[[Sequence[Any]], list[list[float]]] | None = None,
    batch_evaluator_provider: Callable[
        [Sequence[Any]],
        tuple[list[list[float]], list[float]],
    ]
    | None = None,
    feature_batch_prior_provider: Callable[
        [Sequence[Sequence[float]], Sequence[Sequence[bool]]],
        list[list[float]],
    ]
    | None = None,
    request_evaluator_provider: Callable[
        [Any],
        tuple[Any, Any],
    ]
    | None = None,
) -> tuple[list[GameLog], list[ReplaySample]]:
    printer = printer if printer is not None else PipelinePrinter()
    config = SelfPlayConfig(
        max_turns=pipeline_config.self_play_max_turns,
        gumbel_simulations=pipeline_config.gumbel_simulations,
        gumbel_max_considered_actions=pipeline_config.gumbel_max_considered_actions,
        gumbel_c_visit=pipeline_config.gumbel_c_visit,
        gumbel_c_scale=pipeline_config.gumbel_c_scale,
        policy_target_c_visit=pipeline_config.policy_target_c_visit,
        policy_target_c_scale=pipeline_config.policy_target_c_scale,
        policy_target_temperature=pipeline_config.policy_target_temperature,
        gumbel_seed=pipeline_config.gumbel_seed,
        temperature_turns=pipeline_config.temperature_turns,
        sampling_temperature=pipeline_config.sampling_temperature,
        playout_cap_randomization=pipeline_config.playout_cap_randomization,
        playout_cap_full_search_fraction=pipeline_config.playout_cap_full_search_fraction,
        playout_cap_full_simulations=pipeline_config.gumbel_simulations,
        playout_cap_fast_simulations=pipeline_config.playout_cap_fast_simulations,
        leaf_batch_size=pipeline_config.leaf_batch_size,
    )

    def default_runner(
        seed: int,
        game_config: SelfPlayConfig,
    ) -> tuple[GameLog, list[ReplaySample]]:
        search = create_core_search_engine(game_config, seed_offset=seed)
        return play_self_play_game(
            seed=seed,
            search=search,
            config=game_config,
            prior_provider=prior_provider,
            evaluator_provider=batch_evaluator_provider,
        )

    run_one = runner if runner is not None else default_runner

    logs: list[GameLog] = []
    samples: list[ReplaySample] = []
    seed = pipeline_config.seed_start if seed_start is None else seed_start
    started_at = time.monotonic()
    printer.step(
        "self-play target "
        f"games>={pipeline_config.self_play_games}, "
        f"samples>={pipeline_config.min_replay_samples}, "
        f"batch={pipeline_config.self_play_batch_size}, "
        f"sims={pipeline_config.gumbel_simulations}, "
        f"leaf_batch={pipeline_config.leaf_batch_size}, "
        f"pcr={_pcr_summary(pipeline_config)}"
    )
    while (
        len(logs) < pipeline_config.self_play_games
        or len(samples) < pipeline_config.min_replay_samples
    ):
        if (
            pipeline_config.max_self_play_games is not None
            and len(logs) >= pipeline_config.max_self_play_games
        ):
            raise RuntimeError(
                "self-play did not produce enough replay samples: "
                f"{len(samples)} < {pipeline_config.min_replay_samples}"
            )
        remaining_games = max(1, pipeline_config.self_play_games - len(logs))
        if len(samples) < pipeline_config.min_replay_samples:
            remaining_games = max(remaining_games, pipeline_config.self_play_batch_size)
        remaining_cap = (
            pipeline_config.max_self_play_games - len(logs)
            if pipeline_config.max_self_play_games is not None
            else remaining_games
        )
        batch_size = min(pipeline_config.self_play_batch_size, remaining_games, remaining_cap)
        if runner is None and batch_prior_provider is not None and batch_size > 1:
            seeds = list(range(seed, seed + batch_size))
            printer.step(f"game seeds={seeds[0]}..{seeds[-1]}")
            batch_results = play_self_play_games_batched(
                seeds=seeds,
                search_factory=lambda: create_core_search_engine(config),
                config=config,
                prior_provider=batch_prior_provider,
                evaluator_provider=batch_evaluator_provider,
                feature_batch_prior_provider=feature_batch_prior_provider,
                request_evaluator_provider=request_evaluator_provider,
            )
            for log, game_samples in batch_results:
                logs.append(log)
                samples.extend(game_samples)
            seed += len(batch_results)
            _print_self_play_progress(
                printer,
                pipeline_config=pipeline_config,
                logs=logs,
                samples=samples,
                started_at=started_at,
            )
        else:
            printer.step(f"game seed={seed}")
            log, game_samples = run_one(seed, config)
            logs.append(log)
            samples.extend(game_samples)
            seed += 1
            _print_self_play_progress(
                printer,
                pipeline_config=pipeline_config,
                logs=logs,
                samples=samples,
                started_at=started_at,
            )
    return logs, samples


def _print_self_play_progress(
    printer: PipelinePrinter,
    *,
    pipeline_config: PipelineConfig,
    logs: Sequence[GameLog],
    samples: Sequence[ReplaySample],
    started_at: float,
) -> None:
    games = len(logs)
    sample_count = len(samples)
    avg_samples = sample_count / games if games else 0.0
    elapsed = _format_duration(time.monotonic() - started_at)
    detail = f"games={games}, avg_samples/game={avg_samples:.1f}, elapsed={elapsed}"
    printer.progress(
        "self-play samples",
        sample_count,
        pipeline_config.min_replay_samples,
        detail=detail,
    )
    if games < pipeline_config.self_play_games:
        printer.progress("self-play games", games, pipeline_config.self_play_games)


def _self_play_config_summary(config: PipelineConfig) -> str:
    return (
        f"games>={config.self_play_games}, samples>={config.min_replay_samples}, "
        f"batch={config.self_play_batch_size}, "
        f"sims={config.gumbel_simulations}, "
        f"leaf_batch={config.leaf_batch_size}, pcr={_pcr_summary(config)}"
    )


def _arena_config_for_pipeline(
    arena_config: ArenaConfig,
    pipeline_config: PipelineConfig,
    *,
    iteration: int,
) -> ArenaConfig:
    if iteration <= 0:
        raise ValueError("iteration must be positive")
    data = asdict(arena_config)
    data.update(
        {
            "seed_start": arena_config.seed_start + (iteration - 1) * arena_config.games,
            "gumbel_simulations": pipeline_config.gumbel_simulations,
            "gumbel_max_considered_actions": pipeline_config.gumbel_max_considered_actions,
            "gumbel_c_visit": pipeline_config.gumbel_c_visit,
            "gumbel_c_scale": pipeline_config.gumbel_c_scale,
            "gumbel_seed": pipeline_config.gumbel_seed,
        }
    )
    return ArenaConfig(**data)


def _should_run_arena(config: PipelineConfig) -> bool:
    return not config.skip_arena and not config.always_promote


def _promote_candidate_unconditionally(
    *,
    candidate_checkpoint: str | Path,
    best_checkpoint: str | Path,
) -> bool:
    source = Path(candidate_checkpoint)
    destination = Path(best_checkpoint)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return True


def _pcr_summary(config: PipelineConfig) -> str:
    if not config.playout_cap_randomization:
        return "off"
    return (
        "on("
        f"full={config.playout_cap_full_search_fraction:.2f}, "
        f"fast_sims={config.playout_cap_fast_simulations}"
        ")"
    )


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    minutes, second = divmod(total_seconds, 60)
    hours, minute = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minute:02d}m{second:02d}s"
    if minute:
        return f"{minute}m{second:02d}s"
    return f"{second}s"


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError("pipeline config must be a JSON object")
    if "always-promote" in data:
        data["always_promote"] = data.pop("always-promote")
    _require_policy_target_scale(data, "pipeline config")
    if "work_dir" in data:
        data["work_dir"] = Path(data["work_dir"])
    return PipelineConfig(**data)


def _require_policy_target_scale(data: dict[str, Any], label: str) -> None:
    missing = [
        key
        for key in ("policy_target_c_visit", "policy_target_c_scale")
        if key not in data
    ]
    if missing:
        raise ValueError(f"{label} must set {', '.join(missing)}")


def _pipeline_paths(config: PipelineConfig) -> PipelineArtifacts:
    return PipelineArtifacts(
        replay_path=config.work_dir / "replay" / "replay.npz",
        self_play_log_path=config.work_dir / "replay" / "game_logs.json",
        candidate_checkpoint=config.work_dir / "checkpoints" / "candidate.pt",
        best_checkpoint=config.work_dir / "checkpoints" / "best.pt",
        arena_report_path=None
        if not _should_run_arena(config)
        else config.work_dir / "reports" / "arena-report.json",
        metrics_path=config.work_dir / "reports" / "metrics.jsonl",
    )


def _ensure_pipeline_dirs(config: PipelineConfig) -> None:
    (config.work_dir / "replay").mkdir(parents=True, exist_ok=True)
    (config.work_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (config.work_dir / "checkpoints" / "candidates").mkdir(parents=True, exist_ok=True)
    (config.work_dir / "reports").mkdir(parents=True, exist_ok=True)
    if _should_run_arena(config):
        (config.work_dir / "reports" / "arena").mkdir(parents=True, exist_ok=True)


def _validate_pipeline_config(config: PipelineConfig) -> None:
    if config.iterations <= 0:
        raise ValueError("iterations must be positive")
    if config.self_play_games < 0:
        raise ValueError("self_play_games must be non-negative")
    if config.min_replay_samples < 0:
        raise ValueError("min_replay_samples must be non-negative")
    if (
        config.max_self_play_games is not None
        and config.max_self_play_games < config.self_play_games
    ):
        raise ValueError("max_self_play_games must be at least self_play_games")
    if config.self_play_batch_size <= 0:
        raise ValueError("self_play_batch_size must be positive")
    if config.replay_capacity <= 0:
        raise ValueError("replay_capacity must be positive")
    if config.gumbel_simulations <= 0:
        raise ValueError("gumbel_simulations must be positive")
    if config.gumbel_max_considered_actions <= 0:
        raise ValueError("gumbel_max_considered_actions must be positive")
    if config.gumbel_c_visit <= 0.0:
        raise ValueError("gumbel_c_visit must be positive")
    if config.gumbel_c_scale <= 0.0:
        raise ValueError("gumbel_c_scale must be positive")
    if not math.isfinite(config.policy_target_c_visit) or config.policy_target_c_visit <= 0.0:
        raise ValueError("policy_target_c_visit must be finite and positive")
    if not math.isfinite(config.policy_target_c_scale) or config.policy_target_c_scale <= 0.0:
        raise ValueError("policy_target_c_scale must be finite and positive")
    if (
        not math.isfinite(config.policy_target_temperature)
        or config.policy_target_temperature <= 0.0
    ):
        raise ValueError("policy_target_temperature must be finite and positive")
    if config.leaf_batch_size <= 0:
        raise ValueError("leaf_batch_size must be positive")
    if not 0.0 < config.playout_cap_full_search_fraction <= 1.0:
        raise ValueError("playout_cap_full_search_fraction must be in (0, 1]")
    if config.playout_cap_fast_simulations <= 0:
        raise ValueError("playout_cap_fast_simulations must be positive")
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


def _load_completed_iteration_count(path: Path, config: PipelineConfig) -> int:
    if not config.resume or not path.exists():
        return 0
    completed = 0
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            data = json.loads(stripped)
            if not isinstance(data, dict) or not isinstance(data.get("iteration"), int):
                raise ValueError(
                    f"metrics line {line_number} must contain an integer iteration"
                )
            completed = max(completed, data["iteration"])
    return completed


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
            "  great-kingdom-pipeline --allow-cpu "
            "--pipeline-config configs/test/pipeline-smoke.json"
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
        default=Path("configs/test/m8-train-smoke.json"),
        help="TrainingConfig JSON",
    )
    config_group.add_argument(
        "--arena-config",
        type=Path,
        default=Path("configs/test/m9-arena-smoke.json"),
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
    run_group.add_argument(
        "--always-promote",
        action="store_true",
        help="skip arena and always replace best checkpoint with the candidate",
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
        "--gumbel-simulations",
        type=int,
        default=None,
        help="Gumbel simulations per self-play move",
    )
    self_play_group.add_argument(
        "--gumbel-max-considered-actions",
        type=int,
        default=None,
        help="maximum root actions considered by Gumbel search",
    )
    self_play_group.add_argument(
        "--policy-target-temperature",
        type=float,
        default=None,
        help="temperature applied only to replay policy targets",
    )
    self_play_group.add_argument(
        "--policy-target-c-visit",
        type=float,
        default=None,
        help="completed-Q visit scale applied only to replay policy targets",
    )
    self_play_group.add_argument(
        "--policy-target-c-scale",
        type=float,
        default=None,
        help="completed-Q value scale applied only to replay policy targets",
    )
    self_play_group.add_argument(
        "--self-play-batch-size",
        type=int,
        default=None,
        help="parallel self-play games for batched model inference",
    )
    self_play_group.add_argument(
        "--leaf-batch-size",
        type=int,
        default=None,
        help="neural leaf evaluations to batch per Rust search callback",
    )
    self_play_group.add_argument(
        "--playout-cap-randomization",
        action="store_true",
        help="use full search only on sampled self-play turns",
    )
    self_play_group.add_argument(
        "--playout-cap-full-search-fraction",
        type=float,
        default=None,
        help="fraction of self-play turns kept as full-search replay samples",
    )
    self_play_group.add_argument(
        "--playout-cap-fast-simulations",
        type=int,
        default=None,
        help="Gumbel simulations for fast self-play turns",
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
        "gumbel_simulations": args.gumbel_simulations,
        "gumbel_max_considered_actions": args.gumbel_max_considered_actions,
        "policy_target_c_visit": args.policy_target_c_visit,
        "policy_target_c_scale": args.policy_target_c_scale,
        "policy_target_temperature": args.policy_target_temperature,
        "self_play_batch_size": args.self_play_batch_size,
        "leaf_batch_size": args.leaf_batch_size,
        "playout_cap_randomization": True if args.playout_cap_randomization else None,
        "playout_cap_full_search_fraction": args.playout_cap_full_search_fraction,
        "playout_cap_fast_simulations": args.playout_cap_fast_simulations,
        "promote": False if args.no_promote else None,
        "always_promote": True if args.always_promote else None,
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
        import torch
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
