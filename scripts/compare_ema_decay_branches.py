"""Train short EMA-decay branches from one checkpoint and compare their drift."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, NoReturn

from great_kingdom_ai.train import TrainingConfig, load_training_config, train_from_replay
from great_kingdom_ai.trajectory_dataset import TrajectoryReplayDataset
from great_kingdom_ai.trajectory_replay import TrajectoryReplayStore

DEFAULT_REPLAY = Path("data/runpod/train-v2-gumbel-512k/replay/trajectory-replay.npz")
DEFAULT_SOURCE = Path("data/runpod/train-v2-gumbel-512k/checkpoints/training-latest.pt")
DEFAULT_TRAIN_CONFIG = Path("configs/runpod/train.json")
DEFAULT_OUTPUT_DIR = Path("data/runpod/train-v2-gumbel-512k/reports/ema-branch-compare")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Start multiple short learner branches from the same raw checkpoint state, "
            "using different EMA decays, then run update-pressure diagnostics for each branch."
        )
    )
    parser.add_argument("--replay", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--source-checkpoint", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--train-config", type=Path, default=DEFAULT_TRAIN_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ema-decays", type=float, nargs="+", default=[0.99, 0.995])
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--eval-device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--recent-sample-fraction", type=float, default=None)
    parser.add_argument("--recent-sample-window", type=int, default=None)
    parser.add_argument(
        "--ema-init",
        choices=["raw", "existing"],
        default="raw",
        help=(
            "raw initializes branch EMA from source raw weights; existing starts from the "
            "source checkpoint's existing EMA state"
        ),
    )
    parser.add_argument("--diagnose-rows-per-split", type=int, default=8192)
    parser.add_argument("--diagnose-batch-size", type=int, default=1024)
    parser.add_argument("--keep-branch-sources", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    return parser


def main() -> NoReturn:
    args = build_parser().parse_args()
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    if not args.ema_decays:
        raise ValueError("at least one ema decay is required")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = args.output_dir / "branch-sources"
    checkpoint_dir = args.output_dir / "checkpoints"
    diagnostics_dir = args.output_dir / "diagnostics"
    source_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    replay = TrajectoryReplayStore.load(args.replay)
    dataset = TrajectoryReplayDataset(replay)
    base_train_config = load_training_config(args.train_config)
    branch_summaries: list[dict[str, Any]] = []

    for decay in args.ema_decays:
        decay_id = _safe_decay_id(decay)
        branch_source = source_dir / f"source-ema-{decay_id}.pt"
        branch_checkpoint = checkpoint_dir / f"branch-ema-{decay_id}.pt"
        _write_branch_source_checkpoint(
            source=args.source_checkpoint,
            destination=branch_source,
            ema_decay=decay,
            ema_init=args.ema_init,
        )
        train_config = _branch_train_config(
            base_train_config,
            steps=args.steps,
            ema_decay=decay,
            device=args.device,
            seed=args.seed,
            learning_rate=args.learning_rate,
            recent_sample_fraction=args.recent_sample_fraction,
            recent_sample_window=args.recent_sample_window,
        )
        train_summary = train_from_replay(
            dataset,
            train_config,
            checkpoint_path=branch_checkpoint,
            resume_path=branch_source,
            log_every=max(1, args.steps // 10),
        )
        raw_diag = _run_update_pressure_diagnostic(
            replay=args.replay,
            before=args.source_checkpoint,
            after=branch_checkpoint,
            before_weights="raw",
            after_weights="raw",
            train_config=args.train_config,
            rows_per_split=args.diagnose_rows_per_split,
            batch_size=args.diagnose_batch_size,
            device=args.eval_device or train_config.device,
            output_path=diagnostics_dir / f"source-raw-to-branch-raw-ema-{decay_id}.json",
        )
        ema_diag = _run_update_pressure_diagnostic(
            replay=args.replay,
            before=args.source_checkpoint,
            after=branch_checkpoint,
            before_weights="raw",
            after_weights="ema",
            train_config=args.train_config,
            rows_per_split=args.diagnose_rows_per_split,
            batch_size=args.diagnose_batch_size,
            device=args.eval_device or train_config.device,
            output_path=diagnostics_dir / f"source-raw-to-branch-ema-ema-{decay_id}.json",
        )
        branch_raw_vs_ema_diag = _run_update_pressure_diagnostic(
            replay=args.replay,
            before=branch_checkpoint,
            after=branch_checkpoint,
            before_weights="raw",
            after_weights="ema",
            train_config=args.train_config,
            rows_per_split=args.diagnose_rows_per_split,
            batch_size=args.diagnose_batch_size,
            device=args.eval_device or train_config.device,
            output_path=diagnostics_dir / f"branch-raw-to-branch-ema-ema-{decay_id}.json",
        )
        branch_summaries.append(
            {
                "ema_decay": decay,
                "ema_init": args.ema_init,
                "branch_source": str(branch_source),
                "branch_checkpoint": str(branch_checkpoint),
                "train": {
                    "start_step": int(train_summary.start_step),
                    "end_step": int(train_summary.end_step),
                    "losses": train_summary.losses,
                },
                "diagnostics": {
                    "source_raw_to_branch_raw": raw_diag,
                    "source_raw_to_branch_ema": ema_diag,
                    "branch_raw_to_branch_ema": branch_raw_vs_ema_diag,
                },
            }
        )
        if not args.keep_branch_sources:
            branch_source.unlink(missing_ok=True)

    payload = {
        "source_checkpoint": str(args.source_checkpoint),
        "replay": str(args.replay),
        "train_config": str(args.train_config),
        "output_dir": str(args.output_dir),
        "branches": branch_summaries,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(payload, indent=2 if args.pretty else None, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2 if args.pretty else None, sort_keys=True))
    raise SystemExit(0)


def _branch_train_config(
    base: TrainingConfig,
    *,
    steps: int,
    ema_decay: float,
    device: str | None,
    seed: int | None,
    learning_rate: float | None,
    recent_sample_fraction: float | None,
    recent_sample_window: int | None,
) -> TrainingConfig:
    data = asdict(base)
    data["steps"] = steps
    data["ema_decay"] = ema_decay
    if device is not None:
        data["device"] = device
    if seed is not None:
        data["seed"] = seed
    if learning_rate is not None:
        data["learning_rate"] = learning_rate
    if recent_sample_fraction is not None:
        data["recent_sample_fraction"] = recent_sample_fraction
    if recent_sample_window is not None:
        data["recent_sample_window"] = recent_sample_window
    return TrainingConfig(**data)


def _write_branch_source_checkpoint(
    *,
    source: Path,
    destination: Path,
    ema_decay: float,
    ema_init: str,
) -> None:
    torch = _import_torch()
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    checkpoint = dict(checkpoint)
    if ema_init == "raw":
        checkpoint["ema_model_state"] = _clone_state_dict(checkpoint["model_state"])
    elif ema_init == "existing":
        if checkpoint.get("ema_model_state") is None:
            checkpoint["ema_model_state"] = _clone_state_dict(checkpoint["model_state"])
    else:
        raise ValueError("ema_init must be one of: raw, existing")
    checkpoint["ema_decay"] = float(ema_decay)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, destination)


def _clone_state_dict(state: dict[str, Any]) -> dict[str, Any]:
    cloned = {}
    for key, value in state.items():
        clone = getattr(value, "clone", None)
        cloned[key] = clone() if clone is not None else copy.deepcopy(value)
    return cloned


def _run_update_pressure_diagnostic(
    *,
    replay: Path,
    before: Path,
    after: Path,
    before_weights: str,
    after_weights: str,
    train_config: Path,
    rows_per_split: int,
    batch_size: int,
    device: str,
    output_path: Path,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).with_name("diagnose_async_update_pressure.py")),
        "--replay",
        str(replay),
        "--before",
        str(before),
        "--after",
        str(after),
        "--before-weights",
        before_weights,
        "--after-weights",
        after_weights,
        "--train-config",
        str(train_config),
        "--rows-per-split",
        str(rows_per_split),
        "--batch-size",
        str(batch_size),
        "--device",
        device,
    ]
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result.stdout, encoding="utf-8")
    payload = json.loads(result.stdout)
    return {
        "path": str(output_path),
        "summary": _extract_probe_metric_summary(payload),
    }


def _extract_probe_metric_summary(payload: dict[str, Any]) -> dict[str, dict[str, float | int]]:
    summary = {}
    for probe in payload.get("probes", []):
        name = str(probe["name"])
        summary[name] = {
            "rows": int(probe["rows"]),
            "target_kl_delta_mean": float(probe["target_kl_delta_mean"]),
            "value_mse_delta": float(probe["value_mse_delta"]),
            "before_to_after_kl_mean": float(probe["before_to_after_kl"]["mean"]),
            "top1_flip_rate": float(probe["top1_flip_rate"]),
            "value_prediction_delta_abs_mean": float(
                probe["value_prediction_delta_abs"]["mean"]
            ),
        }
    return summary


def _safe_decay_id(value: float) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def _import_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("torch is required for EMA branch comparison") from exc
    return torch


if __name__ == "__main__":
    main()
