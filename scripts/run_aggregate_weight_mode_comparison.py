"""Compare aggregate replay count weighting modes from one existing work dir."""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any, NoReturn, cast

import numpy as np
from great_kingdom_ai.evaluate import (
    ArenaConfig,
    load_arena_config,
    load_model_from_checkpoint,
    run_arena,
    save_arena_report,
)
from great_kingdom_ai.online_aggregate_replay import OnlineAggregateReplayBuffer
from great_kingdom_ai.replay_aggregate import _sample_weights_from_counts
from great_kingdom_ai.training import TrainingConfig, load_training_config, train_from_replay

DEFAULT_WORK_DIR = Path("data/runpod/pure-gumbel-medium-plus")
DEFAULT_TRAIN_CONFIG = Path("configs/runpod/train.json")
DEFAULT_ARENA_CONFIG = Path("configs/runpod/arena.json")
DEFAULT_WEIGHT_MODES = ("sqrt_count", "count")
SUPPORTED_WEIGHT_MODES = ("none", "sqrt_count", "count")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train aggregate-replay candidates with different count weighting modes "
            "from the current work dir, then run fast head-to-head arenas. This uses an existing "
            "replay-aggregated.npz and rewrites only sample_weights from saved counts."
        )
    )
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--best", type=Path, default=None)
    parser.add_argument("--aggregate-replay", type=Path, default=None)
    parser.add_argument("--train-config", type=Path, default=DEFAULT_TRAIN_CONFIG)
    parser.add_argument("--arena-config", type=Path, default=DEFAULT_ARENA_CONFIG)
    parser.add_argument(
        "--weight-mode",
        choices=SUPPORTED_WEIGHT_MODES,
        nargs="+",
        default=list(DEFAULT_WEIGHT_MODES),
        help="modes to train; default trains sqrt_count and count",
    )
    parser.add_argument(
        "--baseline-weight-mode",
        choices=SUPPORTED_WEIGHT_MODES,
        default="sqrt_count",
        help="trained mode used as the arena baseline",
    )
    parser.add_argument(
        "--sample-weight-cap",
        type=float,
        default=None,
        help="optional final cap applied to generated sample weights",
    )
    parser.add_argument("--train-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--train-seed", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--arena-games", type=int, default=64)
    parser.add_argument("--arena-seed-start", type=int, default=None)
    parser.add_argument("--arena-gumbel-simulations", type=int, default=64)
    parser.add_argument("--arena-gumbel-max-considered-actions", type=int, default=16)
    parser.add_argument("--skip-arena", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> NoReturn:
    args = build_parser().parse_args()
    work_dir = args.work_dir
    best_checkpoint = args.best or work_dir / "checkpoints" / "best.pt"
    aggregate_replay_path = args.aggregate_replay or work_dir / "replay" / "replay-aggregated.npz"
    output_dir = work_dir / "reports" / "aggregate-weight-mode-comparison"
    candidate_dir = work_dir / "checkpoints" / "aggregate-weight-mode-comparison"
    variant_replay_dir = work_dir / "replay" / "aggregate-weight-mode-comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    variant_replay_dir.mkdir(parents=True, exist_ok=True)

    _require_file(best_checkpoint, "best checkpoint")
    _require_file(aggregate_replay_path, "aggregate replay")

    base_train_config = _load_effective_train_config(
        args.train_config,
        device=args.device,
        steps=args.train_steps,
        learning_rate=args.learning_rate,
        train_seed=args.train_seed,
        batch_size=args.batch_size,
    )
    arena_config = _load_effective_arena_config(
        args.arena_config,
        device=args.device,
        games=args.arena_games,
        seed_start=args.arena_seed_start,
        gumbel_simulations=args.arena_gumbel_simulations,
        gumbel_max_considered_actions=args.arena_gumbel_max_considered_actions,
    )

    print(
        json.dumps(
            {
                "event": "aggregate_weight_mode_comparison_start",
                "best": str(best_checkpoint),
                "source_aggregate_replay": str(aggregate_replay_path),
                "weight_modes": args.weight_mode,
                "baseline_weight_mode": args.baseline_weight_mode,
                "sample_weight_cap": args.sample_weight_cap,
                "train_config": asdict(base_train_config),
                "arena_config": None if args.skip_arena else asdict(arena_config),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    results: list[dict[str, Any]] = []
    candidates: dict[str, Path] = {}
    for weight_mode in args.weight_mode:
        run_id = _run_id(
            weight_mode=weight_mode,
            config=base_train_config,
            sample_weight_cap=args.sample_weight_cap,
        )
        replay_path = variant_replay_dir / f"{run_id}-replay.npz"
        candidate_path = candidate_dir / f"{run_id}.pt"
        train_summary_path = output_dir / f"{run_id}-train.json"

        replay_stats = write_reweighted_aggregate_replay(
            source_path=aggregate_replay_path,
            output_path=replay_path,
            weight_mode=weight_mode,
            weight_cap=args.sample_weight_cap,
            force=args.force,
        )
        replay = _load_training_replay(
            replay_path,
            weight_mode=weight_mode,
            weight_cap=args.sample_weight_cap,
        )
        print(
            json.dumps(
                {
                    "event": "aggregate_weight_mode_train_start",
                    "run_id": run_id,
                    "weight_mode": weight_mode,
                    "replay": str(replay_path),
                    "replay_samples": len(replay),
                    "replay_stats": replay_stats,
                    "candidate": str(candidate_path),
                },
                sort_keys=True,
            ),
            flush=True,
        )

        if args.force or not candidate_path.exists():
            summary = train_from_replay(
                replay,
                base_train_config,
                checkpoint_path=candidate_path,
                resume_path=None,
                bootstrap_weights_path=best_checkpoint,
                log_every=_log_every(args.log_every, base_train_config.steps),
                progress_callback=partial(_print_train_progress, run_id=run_id),
            )
            train_payload = {
                "run_id": run_id,
                "weight_mode": weight_mode,
                "best": str(best_checkpoint),
                "replay": str(replay_path),
                "replay_samples": len(replay),
                "replay_stats": replay_stats,
                "config": asdict(base_train_config),
                "summary": {
                    "start_step": summary.start_step,
                    "end_step": summary.end_step,
                    "losses": summary.losses,
                    "tail_loss": _tail_loss_summary(summary.losses),
                },
            }
            _write_json(train_summary_path, train_payload)
        else:
            print(
                json.dumps(
                    {
                        "event": "aggregate_weight_mode_train_skip_existing",
                        "run_id": run_id,
                        "candidate": str(candidate_path),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        candidates[weight_mode] = candidate_path
        result = {
            "event": "aggregate_weight_mode_train_result",
            "run_id": run_id,
            "weight_mode": weight_mode,
            "replay": str(replay_path),
            "replay_samples": len(replay),
            "candidate": str(candidate_path),
            "train_summary": str(train_summary_path),
        }
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)

    arena_results: list[dict[str, Any]] = []
    if not args.skip_arena:
        baseline_mode = args.baseline_weight_mode
        if baseline_mode not in candidates:
            raise ValueError(f"baseline mode must be included in --weight-mode: {baseline_mode}")
        for candidate_mode, candidate_path in candidates.items():
            if candidate_mode == baseline_mode:
                continue
            report_path = output_dir / _arena_report_name(candidate_mode, baseline_mode)
            arena_summary = _run_or_load_head_to_head_arena(
                candidate_checkpoint=candidate_path,
                baseline_checkpoint=candidates[baseline_mode],
                candidate_mode=candidate_mode,
                baseline_mode=baseline_mode,
                report_path=report_path,
                arena_config=arena_config,
                force=args.force,
            )
            arena_result = {
                "event": "aggregate_weight_mode_arena_result",
                "candidate": candidate_mode,
                "baseline": baseline_mode,
                "arena_report": str(report_path),
                "arena_summary": arena_summary,
            }
            arena_results.append(arena_result)
            print(json.dumps(arena_result, sort_keys=True), flush=True)

    _write_json(
        output_dir / "summary.json",
        {
            "results": results,
            "arena_results": arena_results,
        },
    )
    raise SystemExit(0)


def write_reweighted_aggregate_replay(
    *,
    source_path: Path,
    output_path: Path,
    weight_mode: str,
    weight_cap: float | None,
    force: bool,
) -> dict[str, Any]:
    if output_path.exists() and not force:
        return _replay_weight_stats(output_path)

    with np.load(source_path) as data:
        if "counts" not in data:
            raise ValueError(f"{source_path} does not contain aggregate counts")
        counts = np.asarray(data["counts"], dtype=np.int64)
        sample_weights = _sample_weights_from_counts(
            counts,
            mode=weight_mode,
            cap=weight_cap,
        )
        payload: dict[str, np.ndarray] = {
            "capacity": np.asarray(data["capacity"], dtype=np.int64),
            "features": np.asarray(data["features"], dtype=np.float32),
            "policies": np.asarray(data["policies"], dtype=np.float32),
            "values": np.asarray(data["values"], dtype=np.float32),
            "counts": counts,
            "sample_weights": sample_weights,
        }
        if "root_policy_logits" in data:
            payload["root_policy_logits"] = np.asarray(
                data["root_policy_logits"],
                dtype=np.float32,
            )
        if "root_policy_counts" in data:
            payload["root_policy_counts"] = np.asarray(
                data["root_policy_counts"],
                dtype=np.int64,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **cast(dict[str, Any], payload))
    return _weight_stats(counts=counts, sample_weights=sample_weights)


def _load_training_replay(
    path: Path,
    *,
    weight_mode: str,
    weight_cap: float | None,
) -> OnlineAggregateReplayBuffer:
    return OnlineAggregateReplayBuffer.load(
        path,
        sample_weight_mode=weight_mode,
        sample_weight_cap=weight_cap,
    )


def _run_or_load_head_to_head_arena(
    *,
    candidate_checkpoint: Path,
    baseline_checkpoint: Path,
    candidate_mode: str,
    baseline_mode: str,
    report_path: Path,
    arena_config: ArenaConfig,
    force: bool,
) -> dict[str, Any]:
    if report_path.exists() and not force:
        return _load_arena_summary(report_path)

    candidate_model = load_model_from_checkpoint(candidate_checkpoint, device=arena_config.device)
    baseline_model = load_model_from_checkpoint(baseline_checkpoint, device=arena_config.device)
    report = run_arena(
        candidate_model=candidate_model,
        best_model=baseline_model,
        config=arena_config,
        progress_callback=partial(
            _print_arena_progress,
            candidate_mode=candidate_mode,
            baseline_mode=baseline_mode,
        ),
    )
    save_arena_report(report, report_path)
    summary = report.summary.to_dict()
    del candidate_model, baseline_model
    _release_cuda_cache()
    return summary


def _load_effective_train_config(
    path: Path,
    *,
    device: str | None,
    steps: int | None,
    learning_rate: float | None,
    train_seed: int | None,
    batch_size: int | None,
) -> TrainingConfig:
    config = load_training_config(path)
    data = asdict(config)
    data.update(
        {
            key: value
            for key, value in {
                "device": device,
                "steps": steps,
                "learning_rate": learning_rate,
                "seed": train_seed,
                "batch_size": batch_size,
            }.items()
            if value is not None
        }
    )
    return TrainingConfig(**data)


def _load_effective_arena_config(
    path: Path,
    *,
    device: str | None,
    games: int | None,
    seed_start: int | None,
    gumbel_simulations: int | None,
    gumbel_max_considered_actions: int | None,
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
                "gumbel_simulations": gumbel_simulations,
                "gumbel_max_considered_actions": gumbel_max_considered_actions,
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
    return max(1, steps // 20)


def _run_id(
    *,
    weight_mode: str,
    config: TrainingConfig,
    sample_weight_cap: float | None,
) -> str:
    lr_label = f"{config.learning_rate:.0e}".replace("+", "")
    cap_label = "nocap" if sample_weight_cap is None else f"cap{sample_weight_cap:g}"
    return f"{weight_mode}-{cap_label}-lr{lr_label}-s{config.steps:04d}-seed{config.seed}"


def _arena_report_name(candidate_mode: str, baseline_mode: str) -> str:
    return f"{candidate_mode}-vs-{baseline_mode}-arena.json"


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
                "event": "aggregate_weight_mode_train_progress",
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
    candidate_mode: str,
    baseline_mode: str,
) -> None:
    winner = int(game.winner)
    candidate_player = int(game.candidate_player)
    print(
        json.dumps(
            {
                "event": "aggregate_weight_mode_arena_progress",
                "candidate": candidate_mode,
                "baseline": baseline_mode,
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


def _tail_loss_summary(
    losses: list[dict[str, float]],
    *,
    fraction: float = 0.25,
) -> dict[str, float]:
    if not losses:
        return {}
    tail_size = max(1, int(len(losses) * fraction))
    tail = losses[-tail_size:]
    keys = ["total", "policy", "policy_kl", "value", "policy_entropy"]
    summary: dict[str, float] = {}
    for key in keys:
        values = [float(row[key]) for row in tail if key in row]
        if values:
            summary[key] = sum(values) / len(values)
    return summary


def _replay_weight_stats(path: Path) -> dict[str, Any]:
    with np.load(path) as data:
        counts = np.asarray(data["counts"], dtype=np.int64)
        sample_weights = np.asarray(data["sample_weights"], dtype=np.float32)
    return _weight_stats(counts=counts, sample_weights=sample_weights)


def _weight_stats(*, counts: np.ndarray, sample_weights: np.ndarray) -> dict[str, Any]:
    return {
        "samples": int(counts.shape[0]),
        "raw_rows_represented": int(counts.sum(initial=0)),
        "max_count": int(counts.max(initial=0)),
        "max_weight": float(sample_weights.max(initial=1.0)),
        "mean_weight": float(sample_weights.mean()) if sample_weights.size else 0.0,
    }


def _load_arena_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict) or not isinstance(data.get("summary"), dict):
        raise ValueError(f"{path} does not contain an arena summary")
    return dict(data["summary"])


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")


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


if __name__ == "__main__":
    main()
