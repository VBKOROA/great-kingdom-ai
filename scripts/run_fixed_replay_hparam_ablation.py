"""Train/evaluate candidate hparams against one fixed best checkpoint and replay file."""

from __future__ import annotations

import argparse
import gc
import json
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
from great_kingdom_ai.training import TrainingConfig, load_training_config, train_from_replay

DEFAULT_WORK_DIR = Path("data/runpod/pure-gumbel-medium-plus")
DEFAULT_TRAIN_CONFIG = Path("configs/runpod/train.json")
DEFAULT_ARENA_CONFIG = Path("configs/runpod/arena.json")
DEFAULT_SPECS = ("0.0002:150", "0.0002:300", "0.0001:150", "0.0001:300")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run fixed-replay hparam ablations. This does not run self-play and does not "
            "promote; every candidate is trained from the same best checkpoint against the "
            "same replay snapshot."
        )
    )
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--best", type=Path, default=None)
    parser.add_argument("--replay", type=Path, default=None)
    parser.add_argument("--train-config", type=Path, default=DEFAULT_TRAIN_CONFIG)
    parser.add_argument("--arena-config", type=Path, default=DEFAULT_ARENA_CONFIG)
    parser.add_argument(
        "--spec",
        nargs="+",
        default=list(DEFAULT_SPECS),
        metavar="LR:STEPS",
        help="Training variants, for example: 0.0002:150 0.0001:300",
    )
    parser.add_argument(
        "--bootstrap-mode",
        choices=["weights", "resume"],
        default="weights",
        help=(
            "weights loads only best model weights with fresh optimizer/scheduler; "
            "resume reproduces the pipeline's checkpoint resume behavior"
        ),
    )
    parser.add_argument("--train-seed", type=int, nargs="+", default=None)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--arena-games", type=int, default=None)
    parser.add_argument("--arena-seed-start", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--candidate-dir", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-arena", action="store_true")
    return parser


def main() -> NoReturn:
    args = build_parser().parse_args()
    work_dir = args.work_dir
    best_checkpoint = args.best or work_dir / "checkpoints" / "best.pt"
    replay_path = args.replay or work_dir / "replay" / "trajectory-replay.npz"
    output_dir = args.output_dir or work_dir / "reports" / "fixed-replay-hparam-ablation"
    candidate_dir = (
        args.candidate_dir or work_dir / "checkpoints" / "fixed-replay-hparam-ablation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(parents=True, exist_ok=True)

    _require_file(best_checkpoint, "best checkpoint")
    _require_file(replay_path, "replay")

    base_train_config = _load_effective_train_config(args.train_config, device=args.device)
    arena_config = _load_effective_arena_config(
        args.arena_config,
        device=args.device,
        games=args.arena_games,
        seed_start=args.arena_seed_start,
    )
    specs = [_parse_spec(spec) for spec in args.spec]
    train_seeds = args.train_seed or [base_train_config.seed]
    replay = TrajectoryReplayDataset(TrajectoryReplayStore.load(replay_path))
    results_path = output_dir / "results.jsonl"
    results: list[dict[str, Any]] = []

    print(
        json.dumps(
            {
                "event": "fixed_replay_ablation_start",
                "best": str(best_checkpoint),
                "replay": str(replay_path),
                "replay_samples": len(replay),
                "bootstrap_mode": args.bootstrap_mode,
                "specs": [{"learning_rate": lr, "steps": steps} for lr, steps in specs],
                "train_seeds": train_seeds,
                "skip_arena": args.skip_arena,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    best_model = None
    if not args.skip_arena:
        best_model = load_model_from_checkpoint(best_checkpoint, device=arena_config.device)

    for learning_rate, steps in specs:
        for train_seed in train_seeds:
            run_id = _run_id(
                learning_rate=learning_rate,
                steps=steps,
                train_seed=train_seed,
                bootstrap_mode=args.bootstrap_mode,
            )
            candidate_path = candidate_dir / f"{run_id}.pt"
            train_summary_path = output_dir / f"{run_id}-train.json"
            arena_report_path = output_dir / f"{run_id}-arena.json"
            train_config = TrainingConfig(
                **{
                    **asdict(base_train_config),
                    "learning_rate": learning_rate,
                    "steps": steps,
                    "seed": train_seed,
                }
            )

            if args.force or not candidate_path.exists():
                print(
                    json.dumps(
                        {
                            "event": "fixed_replay_train_start",
                            "run_id": run_id,
                            "learning_rate": learning_rate,
                            "steps": steps,
                            "train_seed": train_seed,
                            "candidate": str(candidate_path),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                summary = train_from_replay(
                    replay,
                    train_config,
                    checkpoint_path=candidate_path,
                    resume_path=(
                        best_checkpoint if args.bootstrap_mode == "resume" else None
                    ),
                    bootstrap_weights_path=(
                        best_checkpoint if args.bootstrap_mode == "weights" else None
                    ),
                    log_every=_log_every(args.log_every, steps),
                    progress_callback=partial(_print_train_progress, run_id=run_id),
                )
                _write_json(
                    train_summary_path,
                    {
                        "run_id": run_id,
                        "best": str(best_checkpoint),
                        "replay": str(replay_path),
                        "replay_samples": len(replay),
                        "bootstrap_mode": args.bootstrap_mode,
                        "config": asdict(train_config),
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
                            "event": "fixed_replay_train_skip_existing",
                            "run_id": run_id,
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
                        progress_callback=partial(_print_arena_progress, run_id=run_id),
                    )
                    save_arena_report(report, arena_report_path)
                    arena_summary = report.summary.to_dict()
                    del candidate_model
                    _release_cuda_cache()
                else:
                    arena_summary = _load_arena_summary(arena_report_path)

            result = {
                "event": "fixed_replay_ablation_result",
                "run_id": run_id,
                "learning_rate": learning_rate,
                "steps": steps,
                "train_seed": train_seed,
                "bootstrap_mode": args.bootstrap_mode,
                "best": str(best_checkpoint),
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


def _parse_spec(raw: str) -> tuple[float, int]:
    try:
        lr_raw, steps_raw = raw.split(":", maxsplit=1)
        learning_rate = float(lr_raw)
        steps = int(steps_raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"spec must be LR:STEPS, got {raw!r}"
        ) from exc
    if learning_rate <= 0.0:
        raise argparse.ArgumentTypeError("learning rate must be positive")
    if steps <= 0:
        raise argparse.ArgumentTypeError("steps must be positive")
    return learning_rate, steps


def _run_id(
    *,
    learning_rate: float,
    steps: int,
    train_seed: int,
    bootstrap_mode: str,
) -> str:
    lr_label = f"{learning_rate:.0e}".replace("+", "")
    return f"{bootstrap_mode}-lr{lr_label}-s{steps:04d}-seed{train_seed}"


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
    data = asdict(config)
    data.update(
        {
            key: value
            for key, value in {
                "device": device,
                "games": games,
                "seed_start": seed_start,
            }.items()
            if value is not None
        }
    )
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
    run_id: str,
) -> None:
    print(
        json.dumps(
            {
                "event": "fixed_replay_train_progress",
                "run_id": run_id,
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
    run_id: str,
) -> None:
    winner = int(game.winner)
    candidate_player = int(game.candidate_player)
    print(
        json.dumps(
            {
                "event": "fixed_replay_arena_progress",
                "run_id": run_id,
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
