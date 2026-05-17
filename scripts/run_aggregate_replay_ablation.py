"""Run raw-vs-aggregate replay training ablations from an existing Runpod work dir."""

from __future__ import annotations

import argparse
import gc
import json
import shutil
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any, NoReturn

from great_kingdom_ai.evaluate import (
    ArenaConfig,
    load_arena_config,
    load_model_from_checkpoint,
    run_arena,
    save_arena_report,
)
from great_kingdom_ai.replay import TrajectoryReplayDataset, TrajectoryReplayStore
from great_kingdom_ai.replay_aggregate import (
    aggregate_duplicate_replay,
)
from great_kingdom_ai.replay_aggregate import (
    load_replay as load_replay_arrays,
)
from great_kingdom_ai.replay_aggregate import (
    save_replay as save_aggregated_replay,
)
from great_kingdom_ai.training import TrainingConfig, load_training_config, train_from_replay

DEFAULT_WORK_DIR = Path("data/runpod/onnx-aggregate-diagnostics")
DEFAULT_TRAIN_CONFIG = Path("configs/runpod/train.json")
DEFAULT_ARENA_CONFIG = Path("configs/runpod/arena.json")
DEFAULT_STEPS = (50, 100, 200)
DEFAULT_VARIANTS = ("raw", "aggregate")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train raw and count-aware aggregate replay candidates from an existing "
            "Runpod pipeline work dir, then evaluate each candidate against best.pt."
        )
    )
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--train-config", type=Path, default=DEFAULT_TRAIN_CONFIG)
    parser.add_argument("--arena-config", type=Path, default=DEFAULT_ARENA_CONFIG)
    parser.add_argument("--steps", type=int, nargs="+", default=list(DEFAULT_STEPS))
    parser.add_argument(
        "--variants",
        choices=DEFAULT_VARIANTS,
        nargs="+",
        default=list(DEFAULT_VARIANTS),
    )
    parser.add_argument(
        "--aggregate-weight-mode",
        choices=["none", "count", "sqrt_count", "log_count", "capped_count"],
        default="sqrt_count",
    )
    parser.add_argument("--aggregate-weight-cap", type=float, default=16.0)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--arena-games", type=int, default=None)
    parser.add_argument("--arena-seed-start", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="rerun completed candidates/reports")
    parser.add_argument("--skip-arena", action="store_true")
    return parser


def main() -> NoReturn:
    args = build_parser().parse_args()
    work_dir = args.work_dir
    best_checkpoint = work_dir / "checkpoints" / "best.pt"
    raw_replay_path = work_dir / "replay" / "trajectory-replay.npz"
    aggregate_replay_path = work_dir / "replay" / "replay-aggregated.npz"
    output_dir = work_dir / "reports" / "aggregate-ablation"
    candidate_dir = work_dir / "checkpoints" / "aggregate-ablation"
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(parents=True, exist_ok=True)

    _require_file(best_checkpoint, "best checkpoint")
    _require_file(raw_replay_path, "raw replay")
    if "aggregate" in args.variants:
        _prepare_aggregate_replay(
            raw_replay_path=raw_replay_path,
            aggregate_replay_path=aggregate_replay_path,
            weight_mode=args.aggregate_weight_mode,
            weight_cap=args.aggregate_weight_cap,
            force=args.force,
        )

    train_config = _load_effective_train_config(args.train_config, device=args.device)
    arena_config = _load_effective_arena_config(
        args.arena_config,
        device=args.device,
        games=args.arena_games,
        seed_start=args.arena_seed_start,
    )
    results_path = output_dir / "results.jsonl"
    results: list[dict[str, Any]] = []

    best_model = None
    if not args.skip_arena:
        best_model = load_model_from_checkpoint(best_checkpoint, device=arena_config.device)

    for variant in args.variants:
        replay_path = raw_replay_path if variant == "raw" else aggregate_replay_path
        replay = TrajectoryReplayDataset(TrajectoryReplayStore.load(replay_path))
        for steps in args.steps:
            run_id = f"{variant}-steps{steps:04d}"
            candidate_path = candidate_dir / f"{run_id}.pt"
            train_summary_path = output_dir / f"{run_id}-train.json"
            arena_report_path = output_dir / f"{run_id}-arena.json"
            config = TrainingConfig(**{**asdict(train_config), "steps": steps})

            print(
                json.dumps(
                    {
                        "event": "ablation_train_start",
                        "variant": variant,
                        "steps": steps,
                        "replay": str(replay_path),
                        "replay_samples": len(replay),
                        "candidate": str(candidate_path),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if args.force or not candidate_path.exists():
                summary = train_from_replay(
                    replay,
                    config,
                    checkpoint_path=candidate_path,
                    resume_path=None,
                    bootstrap_weights_path=best_checkpoint,
                    log_every=_log_every(args.log_every, steps),
                    progress_callback=partial(
                        _print_train_progress,
                        variant=variant,
                        steps=steps,
                    ),
                )
                _write_json(
                    train_summary_path,
                    {
                        "variant": variant,
                        "steps": steps,
                        "replay": str(replay_path),
                        "replay_samples": len(replay),
                        "candidate": str(candidate_path),
                        "summary": {
                            "start_step": summary.start_step,
                            "end_step": summary.end_step,
                            "losses": summary.losses,
                        },
                    },
                )
            else:
                print(
                    json.dumps(
                        {
                            "event": "ablation_train_skip_existing",
                            "variant": variant,
                            "steps": steps,
                            "candidate": str(candidate_path),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

            arena_summary: dict[str, Any] | None = None
            if not args.skip_arena:
                if args.force or not arena_report_path.exists():
                    assert best_model is not None
                    candidate_model = load_model_from_checkpoint(
                        candidate_path,
                        device=arena_config.device,
                    )
                    report = run_arena(
                        candidate_model=candidate_model,
                        best_model=best_model,
                        config=arena_config,
                        progress_callback=partial(
                            _print_arena_progress,
                            variant=variant,
                            steps=steps,
                        ),
                    )
                    save_arena_report(report, arena_report_path)
                    arena_summary = report.summary.to_dict()
                    del candidate_model
                    _release_cuda_cache()
                else:
                    arena_summary = _load_arena_summary(arena_report_path)

            result = {
                "event": "aggregate_replay_ablation_result",
                "variant": variant,
                "steps": steps,
                "replay": str(replay_path),
                "replay_samples": len(replay),
                "candidate": str(candidate_path),
                "train_summary": str(train_summary_path),
                "arena_report": None if args.skip_arena else str(arena_report_path),
                "arena_summary": arena_summary,
            }
            results.append(result)
            _append_jsonl(results_path, result)
            print(json.dumps(result, sort_keys=True), flush=True)

    _write_json(output_dir / "summary.json", {"results": results})
    raise SystemExit(0)


def _prepare_aggregate_replay(
    *,
    raw_replay_path: Path,
    aggregate_replay_path: Path,
    weight_mode: str,
    weight_cap: float | None,
    force: bool,
) -> None:
    if aggregate_replay_path.exists() and not force:
        return
    features, policies, values, root_policy_logits, capacity = load_replay_arrays(raw_replay_path)
    aggregated = aggregate_duplicate_replay(
        features=features,
        policies=policies,
        values=values,
        root_policy_logits=root_policy_logits,
        capacity=capacity,
        sample_weight_mode=weight_mode,
        sample_weight_cap=weight_cap,
    )
    if aggregate_replay_path.exists():
        backup = aggregate_replay_path.with_suffix(".npz.bak")
        shutil.copy2(aggregate_replay_path, backup)
    save_aggregated_replay(aggregate_replay_path, aggregated)


def _load_effective_train_config(path: Path, *, device: str | None) -> TrainingConfig:
    config = load_training_config(path)
    if device is None:
        return config
    return TrainingConfig(**{**asdict(config), "device": device})


def _load_effective_arena_config(
    path: Path,
    *,
    device: str | None,
    games: int | None,
    seed_start: int | None,
) -> ArenaConfig:
    config = load_arena_config(path)
    overrides = {
        "device": device,
        "games": games,
        "seed_start": seed_start,
    }
    data = asdict(config)
    data.update({key: value for key, value in overrides.items() if value is not None})
    return ArenaConfig(**data)


def _log_every(log_every: int | None, steps: int) -> int:
    if log_every is not None:
        if log_every <= 0:
            raise ValueError("log_every must be positive")
        return log_every
    return max(1, steps // 5)


def _print_train_progress(
    current: int,
    target: int,
    loss: dict[str, float],
    *,
    variant: str,
    steps: int,
) -> None:
    print(
        json.dumps(
            {
                "event": "ablation_train_progress",
                "variant": variant,
                "steps": steps,
                "current": current,
                "target": target,
                "loss": loss,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _print_arena_progress(
    current: int,
    target: int,
    game: Any,
    *,
    variant: str,
    steps: int,
) -> None:
    winner = int(game.winner)
    candidate_player = int(game.candidate_player)
    print(
        json.dumps(
            {
                "event": "ablation_arena_progress",
                "variant": variant,
                "steps": steps,
                "current": current,
                "target": target,
                "winner": winner,
                "candidate_player": candidate_player,
                "candidate_win": winner == candidate_player,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _load_arena_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict) or not isinstance(data.get("summary"), dict):
        raise ValueError(f"{path} does not contain an arena summary")
    return dict(data["summary"])


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, sort_keys=True))
        file.write("\n")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)


def _release_cuda_cache() -> None:
    gc.collect()
    try:
        import torch
    except ModuleNotFoundError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise SystemExit(f"missing {label}: {path}")


if __name__ == "__main__":
    main()
