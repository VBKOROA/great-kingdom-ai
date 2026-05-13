"""Trajectory replay based v2 training pipeline."""

from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NoReturn, cast

from great_kingdom_ai.evaluate import (
    ArenaConfig,
    load_arena_config,
    promote_candidate_if_needed,
    run_arena_checkpoints_onnx,
    save_arena_report,
)
from great_kingdom_ai.onnx_export import export_checkpoint_to_onnx
from great_kingdom_ai.pipeline_printer import PipelinePrinter
from great_kingdom_ai.reanalyze import (
    ReanalyzeConfig,
    ReanalyzeSummary,
    ReanalyzeTargetSnapshot,
    build_reanalyze_snapshot_from_store,
)
from great_kingdom_ai.runpod_pruning import collect_prune_items, prune_items
from great_kingdom_ai.rust_onnx_self_play import (
    RustOnnxSelfPlayConfig,
    RustSelfPlayRunSummary,
    run_rust_onnx_self_play,
)
from great_kingdom_ai.search_reanalyze import SearchReanalyzeConfig
from great_kingdom_ai.self_play import SelfPlayConfig
from great_kingdom_ai.train import (
    TrainingConfig,
    create_train_state,
    load_training_config,
    save_checkpoint,
    train_from_replay,
)
from great_kingdom_ai.trajectory_replay import (
    TrajectoryEpisode,
    TrajectoryReplayStore,
    TrajectoryTransition,
)


@dataclass(frozen=True)
class TrainV2PipelineConfig:
    work_dir: Path = Path("data/train-v2")
    iterations: int = 1
    replay_capacity: int = 10000
    self_play_games: int = 2
    min_replay_transitions: int = 0
    max_self_play_games: int | None = None
    seed_start: int = 0
    onnx_device: str = "cpu"
    onnx_precision: str = "fp32"
    onnx_max_batch_size: int = 128
    rust_self_play_batch_size: int = 2
    skip_arena: bool = True
    promote: bool = True
    always_promote: bool = False
    resume: bool = True
    train_checkpoint_mode: str = "resume"
    train_reuse_factor: float | None = 1.5
    min_train_steps: int = 1
    max_train_steps: int | None = None
    reanalyze_batch_size: int = 1024
    reanalyze_device: str | None = None
    bootstrap_td_steps: int = 4
    gamma: float = 1.0
    search_reanalyze_fraction: float = 0.0
    search_reanalyze_budget: int | None = None
    search_reanalyze_simulations: int = 32
    search_reanalyze_max_considered_actions: int = 16
    search_reanalyze_leaf_batch_size: int = 8
    search_reanalyze_root_batch_size: int = 128
    search_reanalyze_seed: int = 0
    save_target_snapshots: bool = True
    prune_artifacts: bool = False
    prune_keep_targets: int = 2
    prune_keep_candidates: int = 3
    prune_keep_onnx: int = 1
    self_play: SelfPlayConfig = SelfPlayConfig()


@dataclass(frozen=True)
class TrainV2IterationSummary:
    iteration: int
    onnx_model_path: Path
    self_play_games: int
    new_transitions: int
    replay_transitions: int
    target_snapshot_path: Path
    reanalyze: ReanalyzeSummary
    train_start_step: int
    train_end_step: int
    candidate_checkpoint: Path
    candidate_win_rate: float | None
    promoted: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "onnx_model_path": str(self.onnx_model_path),
            "self_play_games": self.self_play_games,
            "new_transitions": self.new_transitions,
            "replay_transitions": self.replay_transitions,
            "target_snapshot_path": str(self.target_snapshot_path),
            "reanalyze": self.reanalyze.to_dict(),
            "train_start_step": self.train_start_step,
            "train_end_step": self.train_end_step,
            "candidate_checkpoint": str(self.candidate_checkpoint),
            "candidate_win_rate": self.candidate_win_rate,
            "promoted": self.promoted,
        }


@dataclass(frozen=True)
class TrainV2PipelineSummary:
    iterations: list[TrainV2IterationSummary]
    replay_transitions: int
    best_checkpoint: Path
    trajectory_replay_path: Path
    latest_target_snapshot_path: Path | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "iterations": [iteration.to_dict() for iteration in self.iterations],
            "replay_transitions": self.replay_transitions,
            "best_checkpoint": str(self.best_checkpoint),
            "trajectory_replay_path": str(self.trajectory_replay_path),
            "latest_target_snapshot_path": (
                str(self.latest_target_snapshot_path)
                if self.latest_target_snapshot_path is not None
                else None
            ),
        }


def run_train_v2_pipeline(
    *,
    pipeline_config: TrainV2PipelineConfig,
    train_config: TrainingConfig,
    arena_config: ArenaConfig,
    printer: PipelinePrinter | None = None,
    rust_self_play_runner: Callable[[RustOnnxSelfPlayConfig], RustSelfPlayRunSummary]
    | None = None,
) -> TrainV2PipelineSummary:
    _validate_config(pipeline_config)
    printer = printer if printer is not None else PipelinePrinter()
    paths = _paths(pipeline_config)
    _ensure_dirs(paths, pipeline_config)
    runner = rust_self_play_runner if rust_self_play_runner is not None else run_rust_onnx_self_play

    if not paths["best_checkpoint"].exists():
        printer.step(f"initializing best checkpoint at {paths['best_checkpoint']}")
        save_checkpoint(create_train_state(train_config), paths["best_checkpoint"])

    completed_iterations = _load_completed_iteration_count(
        paths["metrics_path"],
        pipeline_config,
    )
    first_iteration = completed_iterations + 1
    last_iteration = completed_iterations + pipeline_config.iterations
    seed_cursor = _initial_seed_cursor(pipeline_config)
    replay = _load_or_create_trajectory_replay(paths["trajectory_replay_path"], pipeline_config)
    summaries: list[TrainV2IterationSummary] = []

    printer.title("Train V2 Pipeline")
    printer.metric("work dir", pipeline_config.work_dir)
    printer.metric("resume", pipeline_config.resume)
    printer.metric("completed iterations", completed_iterations)
    printer.metric("trajectory transitions", len(replay))
    printer.metric("train device", train_config.device)
    printer.metric("onnx precision", pipeline_config.onnx_precision)
    printer.metric("reanalyze device", _reanalyze_device(pipeline_config, train_config))

    for iteration in range(first_iteration, last_iteration + 1):
        base_phase_total = 6 if not _should_run_arena(pipeline_config) else 7
        phase_total = base_phase_total + (1 if pipeline_config.prune_artifacts else 0)
        printer.title(f"Train V2 Iteration {iteration}/{last_iteration}")
        onnx_path = paths["onnx_checkpoint_dir"] / f"best-{iteration:06d}.onnx"
        printer.step(f"exporting best checkpoint -> {onnx_path}")
        export_checkpoint_to_onnx(
            paths["best_checkpoint"],
            onnx_path,
            device=train_config.device,
            precision=pipeline_config.onnx_precision,
        )
        _release_cuda_cache(train_config.device)
        printer.progress("iteration", 1, phase_total, detail="onnx export complete")

        self_play_summary = _generate_trajectory_self_play(
            pipeline_config=pipeline_config,
            onnx_path=onnx_path,
            artifact_root=paths["self_play_dir"] / f"iteration-{iteration:06d}",
            seed_cursor=seed_cursor,
            iteration=iteration,
            runner=runner,
            replay=replay,
            printer=printer,
        )
        seed_cursor += self_play_summary.games
        new_transitions = sum(
            len(episode.transitions) for episode in self_play_summary.trajectory_episodes
        )
        replay.extend_episodes(self_play_summary.trajectory_episodes)
        replay.save(paths["trajectory_replay_path"], compressed=False)
        _append_game_logs(paths["game_log_path"], self_play_summary.game_logs)
        printer.metric("new games", self_play_summary.games)
        printer.metric("new transitions", new_transitions)
        printer.metric("replay transitions", len(replay))
        printer.progress("iteration", 2, phase_total, detail="trajectory replay written")

        target_snapshot_path = paths["target_dir"] / f"targets-{iteration:06d}.npz"
        reanalyze_checkpoint = _training_source_checkpoint(pipeline_config, paths)
        printer.step(f"reanalyzing targets -> {target_snapshot_path}")
        reanalyze_config = ReanalyzeConfig(
            batch_size=pipeline_config.reanalyze_batch_size,
            device=_reanalyze_device(pipeline_config, train_config),
            onnx_model_path=str(onnx_path),
            onnx_device=pipeline_config.onnx_device,
            onnx_max_batch_size=pipeline_config.onnx_max_batch_size,
            bootstrap_td_steps=pipeline_config.bootstrap_td_steps,
            gamma=pipeline_config.gamma,
            model_version=iteration,
            compressed=False,
            search=SearchReanalyzeConfig(
                fraction=pipeline_config.search_reanalyze_fraction,
                budget=pipeline_config.search_reanalyze_budget,
                simulations=pipeline_config.search_reanalyze_simulations,
                max_considered_actions=(
                    pipeline_config.search_reanalyze_max_considered_actions
                ),
                policy_target_c_visit=pipeline_config.self_play.policy_target_c_visit,
                policy_target_c_scale=pipeline_config.self_play.policy_target_c_scale,
                leaf_batch_size=pipeline_config.search_reanalyze_leaf_batch_size,
                root_batch_size=pipeline_config.search_reanalyze_root_batch_size,
                seed=pipeline_config.search_reanalyze_seed,
            ),
        )
        target_replay = build_reanalyze_snapshot_from_store(
            replay,
            checkpoint_path=reanalyze_checkpoint,
            config=reanalyze_config,
            progress_callback=lambda stage, current, target, detail: printer.progress(
                f"reanalyze {stage}",
                current,
                target,
                detail=detail,
            ),
        )
        reanalyze_summary = _reanalyze_summary_for_snapshot(
            target_replay,
            replay_path=paths["trajectory_replay_path"],
            checkpoint_path=reanalyze_checkpoint,
            output_path=target_snapshot_path,
        )
        if pipeline_config.save_target_snapshots:
            printer.progress("reanalyze save", 0, 1, detail=f"output={target_snapshot_path}")
            target_replay.save(target_snapshot_path, compressed=reanalyze_config.compressed)
            printer.progress("reanalyze save", 1, 1, detail=f"rows={len(target_replay)}")
            shutil.copy2(target_snapshot_path, paths["latest_target_snapshot_path"])
        else:
            printer.progress("reanalyze save", 1, 1, detail="skipped")
        _release_cuda_cache(_reanalyze_device(pipeline_config, train_config))
        printer.progress("iteration", 3, phase_total, detail="reanalyze complete")

        effective_train_config = _train_config_for_iteration(
            train_config,
            new_transitions=new_transitions,
            pipeline_config=pipeline_config,
        )
        candidate_checkpoint = paths["candidate_dir"] / f"candidate-{iteration:06d}.pt"
        printer.metric(
            "train steps",
            (
                f"{effective_train_config.steps} "
                f"(reuse={_effective_reuse_factor(effective_train_config, new_transitions):.2f})"
            ),
        )
        printer.step(f"training candidate -> {candidate_checkpoint}")
        train_summary = train_from_replay(
            target_replay,
            effective_train_config,
            checkpoint_path=candidate_checkpoint,
            **_train_checkpoint_kwargs(
                pipeline_config,
                _training_source_checkpoint(pipeline_config, paths),
            ),
            log_every=max(1, effective_train_config.steps // 10),
            progress_callback=lambda current, target, loss: printer.progress(
                "train",
                current,
                target,
                detail=_format_train_loss_detail(loss),
            ),
        )
        shutil.copy2(candidate_checkpoint, paths["candidate_checkpoint"])
        shutil.copy2(candidate_checkpoint, paths["training_checkpoint"])
        _release_cuda_cache(effective_train_config.device)
        printer.metric("train steps", f"{train_summary.start_step}->{train_summary.end_step}")
        printer.progress("iteration", 4, phase_total, detail="training complete")

        candidate_win_rate: float | None = None
        promoted = False
        if _should_run_arena(pipeline_config):
            report_path = paths["arena_dir"] / f"arena-{iteration:06d}.json"
            printer.step(f"arena evaluation -> {report_path}")
            report = run_arena_checkpoints_onnx(
                candidate_checkpoint=candidate_checkpoint,
                best_checkpoint=paths["best_checkpoint"],
                config=_arena_config_for_iteration(arena_config, iteration=iteration),
                onnx_max_batch_size=pipeline_config.onnx_max_batch_size,
                onnx_precision=pipeline_config.onnx_precision,
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
        elif pipeline_config.always_promote:
            promoted = _promote_candidate_unconditionally(
                candidate_checkpoint=candidate_checkpoint,
                best_checkpoint=paths["best_checkpoint"],
            )
            printer.metric("promoted", promoted)
            printer.progress("iteration", 5, phase_total, detail="always promoted")
        else:
            printer.metric("promoted", promoted)
            printer.progress("iteration", 5, phase_total, detail="arena skipped")

        iteration_summary = TrainV2IterationSummary(
            iteration=iteration,
            onnx_model_path=onnx_path,
            self_play_games=self_play_summary.games,
            new_transitions=new_transitions,
            replay_transitions=len(replay),
            target_snapshot_path=target_snapshot_path,
            reanalyze=reanalyze_summary,
            train_start_step=int(train_summary.start_step),
            train_end_step=int(train_summary.end_step),
            candidate_checkpoint=candidate_checkpoint,
            candidate_win_rate=candidate_win_rate,
            promoted=promoted,
        )
        summaries.append(iteration_summary)
        _append_metrics(paths["metrics_path"], iteration_summary)
        printer.progress("iteration", base_phase_total, phase_total, detail="metrics written")
        if pipeline_config.prune_artifacts:
            _prune_pipeline_artifacts(
                pipeline_config=pipeline_config,
                printer=printer,
            )
            printer.progress("iteration", phase_total, phase_total, detail="pruning complete")

    return TrainV2PipelineSummary(
        iterations=summaries,
        replay_transitions=len(replay),
        best_checkpoint=paths["best_checkpoint"],
        trajectory_replay_path=paths["trajectory_replay_path"],
        latest_target_snapshot_path=(
            paths["latest_target_snapshot_path"]
            if pipeline_config.save_target_snapshots
            and paths["latest_target_snapshot_path"].exists()
            else None
        ),
    )


def load_train_v2_pipeline_config(path: str | Path) -> TrainV2PipelineConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("train v2 pipeline config must be a JSON object")
    path_keys = ("work_dir",)
    for key in path_keys:
        if key in data:
            data[key] = Path(data[key])
    self_play_data = data.pop("self_play", None)
    if isinstance(self_play_data, dict):
        data["self_play"] = SelfPlayConfig(**self_play_data)
    elif self_play_data is None:
        raise ValueError("train v2 pipeline config must set self_play")
    return TrainV2PipelineConfig(**data)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="great-kingdom-train-v2",
        description="Run trajectory replay, reanalyze, and learner training.",
    )
    parser.add_argument("--pipeline-config", type=Path, default=None)
    parser.add_argument("--train-config", type=Path, default=Path("configs/runpod/train.json"))
    parser.add_argument("--arena-config", type=Path, default=Path("configs/runpod/arena.json"))
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--onnx-device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--onnx-precision", choices=["fp32", "fp16"], default=None)
    parser.add_argument("--reanalyze-device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--self-play-games", type=int, default=None)
    parser.add_argument("--min-replay-transitions", type=int, default=None)
    parser.add_argument("--max-self-play-games", type=int, default=None)
    parser.add_argument("--bootstrap-td-steps", type=int, default=None)
    parser.add_argument("--train-reuse-factor", type=float, default=None)
    parser.add_argument("--min-train-steps", type=int, default=None)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--search-reanalyze-fraction", type=float, default=None)
    parser.add_argument("--search-reanalyze-budget", type=int, default=None)
    parser.add_argument(
        "--save-target-snapshots",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--prune-artifacts", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--prune-keep-targets", type=int, default=None)
    parser.add_argument("--prune-keep-candidates", type=int, default=None)
    parser.add_argument("--prune-keep-onnx", type=int, default=None)
    parser.add_argument("--skip-arena", action="store_true")
    parser.add_argument("--always-promote", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> NoReturn:
    args = build_parser().parse_args()
    config = (
        load_train_v2_pipeline_config(args.pipeline_config)
        if args.pipeline_config is not None
        else TrainV2PipelineConfig()
    )
    data = asdict(config)
    for key, value in {
        "work_dir": args.work_dir,
        "iterations": args.iterations,
        "onnx_device": args.onnx_device if args.onnx_device is not None else args.device,
        "onnx_precision": args.onnx_precision,
        "reanalyze_device": (
            args.reanalyze_device if args.reanalyze_device is not None else args.device
        ),
        "self_play_games": args.self_play_games,
        "min_replay_transitions": args.min_replay_transitions,
        "max_self_play_games": args.max_self_play_games,
        "bootstrap_td_steps": args.bootstrap_td_steps,
        "train_reuse_factor": args.train_reuse_factor,
        "min_train_steps": args.min_train_steps,
        "max_train_steps": args.max_train_steps,
        "search_reanalyze_fraction": args.search_reanalyze_fraction,
        "search_reanalyze_budget": args.search_reanalyze_budget,
        "save_target_snapshots": args.save_target_snapshots,
        "prune_artifacts": args.prune_artifacts,
        "prune_keep_targets": args.prune_keep_targets,
        "prune_keep_candidates": args.prune_keep_candidates,
        "prune_keep_onnx": args.prune_keep_onnx,
        "skip_arena": True if args.skip_arena else None,
        "always_promote": True if args.always_promote else None,
    }.items():
        if value is not None:
            data[key] = value
    if isinstance(data.get("self_play"), dict):
        data["self_play"] = SelfPlayConfig(**data["self_play"])
    config = TrainV2PipelineConfig(**data)
    train = load_training_config(args.train_config)
    arena = load_arena_config(args.arena_config)
    if args.device is not None:
        train = TrainingConfig(**{**asdict(train), "device": args.device})
        arena = ArenaConfig(**{**asdict(arena), "device": args.device})
    summary = run_train_v2_pipeline(
        pipeline_config=config,
        train_config=train,
        arena_config=arena,
        printer=PipelinePrinter(enabled=not args.json),
    )
    print(json.dumps(summary.to_dict(), indent=None if args.json else 2, sort_keys=True))
    raise SystemExit(0)


def _generate_trajectory_self_play(
    *,
    pipeline_config: TrainV2PipelineConfig,
    onnx_path: Path,
    artifact_root: Path,
    seed_cursor: int,
    iteration: int,
    runner: Callable[[RustOnnxSelfPlayConfig], RustSelfPlayRunSummary],
    replay: TrajectoryReplayStore,
    printer: PipelinePrinter,
) -> RustSelfPlayRunSummary:
    del replay
    total_games = 0
    total_transitions = 0
    all_logs: list[Any] = []
    all_episodes: list[TrajectoryEpisode] = []
    batch_index = 0
    while (
        total_games < pipeline_config.self_play_games
        or total_transitions < pipeline_config.min_replay_transitions
    ):
        if (
            pipeline_config.max_self_play_games is not None
            and total_games >= pipeline_config.max_self_play_games
        ):
            raise RuntimeError(
                "self-play reached max_self_play_games="
                f"{pipeline_config.max_self_play_games} with transitions="
                f"{total_transitions} < {pipeline_config.min_replay_transitions}"
            )
        batch_index += 1
        games = _next_self_play_game_count(
            pipeline_config,
            total_games=total_games,
            total_transitions=total_transitions,
        )
        artifact_dir = artifact_root / f"batch-{batch_index:03d}"
        summary = runner(
            RustOnnxSelfPlayConfig(
                onnx_model_path=onnx_path,
                output_dir=artifact_dir,
                games=games,
                seed_start=seed_cursor,
                onnx_device=pipeline_config.onnx_device,
                onnx_max_batch_size=pipeline_config.onnx_max_batch_size,
                rust_self_play_batch_size=pipeline_config.rust_self_play_batch_size,
                self_play=pipeline_config.self_play,
            )
        )
        if not summary.trajectory_episodes:
            raise RuntimeError("v2 pipeline requires Rust self-play trajectory_episodes")
        tagged = _tag_episodes(
            summary.trajectory_episodes,
            model_version=iteration,
            created_iteration=iteration,
            search_config_hash=_search_config_hash(pipeline_config.self_play),
        )
        all_logs.extend(summary.game_logs)
        all_episodes.extend(tagged)
        seed_cursor += summary.games
        total_games += summary.games
        total_transitions += sum(len(episode.transitions) for episode in tagged)
        printer.progress(
            "self-play games",
            total_games,
            pipeline_config.self_play_games,
            detail=f"transitions={total_transitions}, seed_next={seed_cursor}",
        )
    return RustSelfPlayRunSummary(
        artifact_dir=artifact_root,
        games=total_games,
        samples=total_transitions,
        onnx_model_path=onnx_path,
        onnx_device=pipeline_config.onnx_device,
        game_logs=tuple(all_logs),
        trajectory_episodes=tuple(all_episodes),
    )


def _tag_episodes(
    episodes: tuple[TrajectoryEpisode, ...],
    *,
    model_version: int,
    created_iteration: int,
    search_config_hash: str,
) -> tuple[TrajectoryEpisode, ...]:
    tagged: list[TrajectoryEpisode] = []
    for episode in episodes:
        transitions = tuple(
            TrajectoryTransition(
                episode_id=transition.episode_id,
                timestep=transition.timestep,
                player=transition.player,
                features=transition.features,
                legal_mask=transition.legal_mask,
                action=transition.action,
                policy_target=transition.policy_target,
                root_policy_logits=transition.root_policy_logits,
                root_value=transition.root_value,
                next_features=transition.next_features,
                winner=transition.winner,
                terminal=transition.terminal,
                model_version=model_version,
                search_config_hash=search_config_hash,
                created_iteration=created_iteration,
                sample_weight=transition.sample_weight,
            )
            for transition in episode.transitions
        )
        tagged.append(
            TrajectoryEpisode(
                episode_id=episode.episode_id,
                seed=episode.seed,
                transitions=transitions,
                winner=episode.winner,
                end_reason=episode.end_reason,
                territory_scores=episode.territory_scores,
            )
        )
    return tuple(tagged)


def _next_self_play_game_count(
    config: TrainV2PipelineConfig,
    *,
    total_games: int,
    total_transitions: int,
) -> int:
    remaining = max(1, config.self_play_games - total_games)
    if total_transitions < config.min_replay_transitions and total_games >= config.self_play_games:
        remaining = max(remaining, config.rust_self_play_batch_size)
    if config.max_self_play_games is not None:
        remaining = min(remaining, config.max_self_play_games - total_games)
    return max(1, min(config.rust_self_play_batch_size, remaining))


def _paths(config: TrainV2PipelineConfig) -> dict[str, Path]:
    return {
        "trajectory_replay_path": config.work_dir / "replay" / "trajectory-replay.npz",
        "game_log_path": config.work_dir / "replay" / "game_logs.jsonl",
        "target_dir": config.work_dir / "targets",
        "latest_target_snapshot_path": config.work_dir / "targets" / "latest.npz",
        "best_checkpoint": config.work_dir / "checkpoints" / "best.pt",
        "training_checkpoint": config.work_dir / "checkpoints" / "training-latest.pt",
        "candidate_checkpoint": config.work_dir / "checkpoints" / "candidate.pt",
        "candidate_dir": config.work_dir / "checkpoints" / "candidates",
        "onnx_checkpoint_dir": config.work_dir / "checkpoints" / "onnx",
        "self_play_dir": config.work_dir / "self-play",
        "metrics_path": config.work_dir / "reports" / "metrics.jsonl",
        "arena_dir": config.work_dir / "reports" / "arena",
    }


def _ensure_dirs(paths: dict[str, Path], config: TrainV2PipelineConfig) -> None:
    for name, path in paths.items():
        if name == "arena_dir" and not _should_run_arena(config):
            continue
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            path.mkdir(parents=True, exist_ok=True)


def _load_or_create_trajectory_replay(
    path: Path,
    config: TrainV2PipelineConfig,
) -> TrajectoryReplayStore:
    if config.resume and path.exists():
        return TrajectoryReplayStore.load(path)
    return TrajectoryReplayStore.empty(config.replay_capacity)


def _load_target_snapshot(path: Path) -> Any:
    from great_kingdom_ai.reanalyze import ReanalyzeTargetSnapshot

    return ReanalyzeTargetSnapshot.load(path)


def _reanalyze_summary_for_snapshot(
    snapshot: ReanalyzeTargetSnapshot,
    *,
    replay_path: Path,
    checkpoint_path: Path,
    output_path: Path,
) -> ReanalyzeSummary:
    return ReanalyzeSummary(
        replay_path=replay_path,
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        transitions=len(snapshot),
        model_version=snapshot.model_version,
        bootstrap_td_steps=snapshot.bootstrap_td_steps,
        gamma=snapshot.gamma,
        search_reanalyzed=(
            0
            if snapshot.search_reanalyzed is None
            else int(snapshot.search_reanalyzed.sum())
        ),
    )


def _training_source_checkpoint(config: TrainV2PipelineConfig, paths: dict[str, Path]) -> Path:
    if paths["training_checkpoint"].exists():
        return paths["training_checkpoint"]
    return paths["best_checkpoint"]


def _reanalyze_device(config: TrainV2PipelineConfig, train_config: TrainingConfig) -> str:
    return config.reanalyze_device or train_config.device


def _append_game_logs(path: Path, logs: tuple[Any, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        for log in logs:
            file.write(json.dumps(log.to_dict(), sort_keys=True))
            file.write("\n")


def _append_metrics(path: Path, summary: TrainV2IterationSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(summary.to_dict(), sort_keys=True))
        file.write("\n")


def _load_completed_iteration_count(path: Path, config: TrainV2PipelineConfig) -> int:
    if not config.resume or not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as file:
        return sum(1 for line in file if line.strip())


def _initial_seed_cursor(config: TrainV2PipelineConfig) -> int:
    if not config.resume:
        return config.seed_start
    path = config.work_dir / "replay" / "game_logs.jsonl"
    if not path.exists():
        return config.seed_start
    max_seed = config.seed_start - 1
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            data = json.loads(line)
            if isinstance(data, dict) and "seed" in data:
                max_seed = max(max_seed, int(data["seed"]))
    return max_seed + 1


def _validate_config(config: TrainV2PipelineConfig) -> None:
    if config.iterations <= 0:
        raise ValueError("iterations must be positive")
    if config.replay_capacity <= 0:
        raise ValueError("replay_capacity must be positive")
    if config.self_play_games < 0:
        raise ValueError("self_play_games must be non-negative")
    if config.min_replay_transitions < 0:
        raise ValueError("min_replay_transitions must be non-negative")
    if (
        config.max_self_play_games is not None
        and config.max_self_play_games < config.self_play_games
    ):
        raise ValueError("max_self_play_games must be at least self_play_games")
    if config.onnx_max_batch_size <= 0:
        raise ValueError("onnx_max_batch_size must be positive")
    if config.onnx_precision not in {"fp32", "fp16"}:
        raise ValueError("onnx_precision must be one of: fp32, fp16")
    if config.rust_self_play_batch_size <= 0:
        raise ValueError("rust_self_play_batch_size must be positive")
    if config.reanalyze_batch_size <= 0:
        raise ValueError("reanalyze_batch_size must be positive")
    if config.train_checkpoint_mode not in {"resume", "bootstrap"}:
        raise ValueError("train_checkpoint_mode must be one of: resume, bootstrap")
    if config.train_reuse_factor is not None:
        if not math.isfinite(config.train_reuse_factor) or config.train_reuse_factor <= 0.0:
            raise ValueError("train_reuse_factor must be finite and positive")
    if config.min_train_steps <= 0:
        raise ValueError("min_train_steps must be positive")
    if config.max_train_steps is not None and config.max_train_steps < config.min_train_steps:
        raise ValueError("max_train_steps must be at least min_train_steps")
    for label, value in (
        ("prune_keep_targets", config.prune_keep_targets),
        ("prune_keep_candidates", config.prune_keep_candidates),
        ("prune_keep_onnx", config.prune_keep_onnx),
    ):
        if value < 0:
            raise ValueError(f"{label} must be non-negative")


def _search_config_hash(config: SelfPlayConfig) -> str:
    return json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))


def _train_checkpoint_kwargs(
    config: TrainV2PipelineConfig,
    checkpoint: Path,
) -> dict[str, Path | None]:
    if config.train_checkpoint_mode == "resume":
        return {"resume_path": checkpoint, "bootstrap_weights_path": None}
    if config.train_checkpoint_mode == "bootstrap":
        return {"resume_path": None, "bootstrap_weights_path": checkpoint}
    raise ValueError("train_checkpoint_mode must be one of: resume, bootstrap")


def _should_run_arena(config: TrainV2PipelineConfig) -> bool:
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


def _arena_config_for_iteration(
    arena_config: ArenaConfig,
    *,
    iteration: int,
) -> ArenaConfig:
    if iteration <= 0:
        raise ValueError("iteration must be positive")
    data = asdict(arena_config)
    data["seed_start"] = arena_config.seed_start + (iteration - 1) * arena_config.games
    return ArenaConfig(**data)


def _format_train_loss_detail(loss: dict[str, float]) -> str:
    detail = f"loss={loss['total']:.4f}"
    if {"policy", "value", "policy_kl"}.issubset(loss):
        detail += (
            f" policy={loss['policy']:.4f}"
            f" value={loss['value']:.4f}"
            f" kl={loss['policy_kl']:.4f}"
        )
    return detail


def _release_cuda_cache(device: str | None) -> None:
    if device is None or not str(device).startswith("cuda"):
        return
    gc.collect()
    try:
        import torch
    except ModuleNotFoundError:
        return
    if not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()
    try:
        cast(Callable[[], None], torch.cuda.ipc_collect)()
    except RuntimeError:
        pass


def _train_config_for_iteration(
    train_config: TrainingConfig,
    *,
    new_transitions: int,
    pipeline_config: TrainV2PipelineConfig,
) -> TrainingConfig:
    if pipeline_config.train_reuse_factor is None:
        return train_config
    if new_transitions <= 0:
        steps = pipeline_config.min_train_steps
    else:
        steps = math.ceil(
            new_transitions * pipeline_config.train_reuse_factor / train_config.batch_size
        )
        steps = max(pipeline_config.min_train_steps, steps)
    if pipeline_config.max_train_steps is not None:
        steps = min(steps, pipeline_config.max_train_steps)
    return TrainingConfig(**{**asdict(train_config), "steps": steps})


def _effective_reuse_factor(train_config: TrainingConfig, new_transitions: int) -> float:
    if new_transitions <= 0:
        return 0.0
    return train_config.steps * train_config.batch_size / new_transitions


def _prune_pipeline_artifacts(
    *,
    pipeline_config: TrainV2PipelineConfig,
    printer: PipelinePrinter,
) -> None:
    items = collect_prune_items(
        pipeline_config.work_dir,
        keep_targets=pipeline_config.prune_keep_targets,
        keep_candidates=pipeline_config.prune_keep_candidates,
        keep_onnx=pipeline_config.prune_keep_onnx,
    )
    total_bytes = sum(item.size_bytes for item in items)
    printer.step(
        "pruning regenerable artifacts "
        f"(items={len(items)}, bytes={total_bytes}, elapsed={printer.elapsed()})"
    )
    prune_items(items, delete=True)


__all__ = [
    "TrainV2IterationSummary",
    "TrainV2PipelineConfig",
    "TrainV2PipelineSummary",
    "build_parser",
    "load_train_v2_pipeline_config",
    "run_train_v2_pipeline",
]
