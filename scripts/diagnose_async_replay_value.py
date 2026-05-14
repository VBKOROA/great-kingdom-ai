"""Diagnose async trajectory replay value targets and checkpoint predictions."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NoReturn

import numpy as np

from great_kingdom_ai.self_play_data import value_target_for_player
from great_kingdom_ai.train import load_checkpoint
from great_kingdom_ai.trajectory_replay import TrajectoryReplayStore


@dataclass(frozen=True)
class ValuePredictionSummary:
    checkpoint: str
    prefer_ema: bool
    rows: int
    mse: float
    mae: float
    corr: float | None
    pred_mean: float
    pred_std: float
    pred_min: float
    pred_max: float
    target_mean: float
    target_std: float


def main() -> NoReturn:
    args = _parser().parse_args()
    replay = TrajectoryReplayStore.load(args.replay)
    targets = _terminal_value_targets(replay)
    summary: dict[str, Any] = {
        "replay": str(args.replay),
        "rows": len(replay),
        "episodes": replay.episode_count,
        "capacity": replay.capacity,
        "avg_rows_per_episode": len(replay) / max(1, replay.episode_count),
        "episode_length": _describe_episode_lengths(replay),
        "winners": _counts(replay.episode_winners),
        "players": _counts(replay.players),
        "value_targets": _describe_array(targets),
        "value_target_counts": _counts(targets),
        "mse_zero": float(np.mean(targets**2)) if targets.size else math.nan,
        "policy_entropy": _describe_array(_policy_entropy(replay.policy_targets)),
        "policy_max_prob": _describe_array(replay.policy_targets.max(axis=1)),
        "sample_weights": _describe_array(replay.sample_weights),
        "timesteps": _describe_array(replay.timesteps),
        "model_versions": _counts(replay.model_versions),
        "created_iterations": _counts(replay.created_iterations),
        "terminal_flags": _counts(replay.terminals),
    }
    if args.checkpoint is not None:
        sample_indexes = _sample_indexes(len(replay), args.eval_rows, random.Random(args.seed))
        prediction_summaries = []
        for prefer_ema in ([False, True] if args.compare_ema else [args.prefer_ema]):
            prediction_summaries.append(
                asdict(
                    _evaluate_checkpoint_values(
                        checkpoint=args.checkpoint,
                        replay=replay,
                        targets=targets,
                        indexes=sample_indexes,
                        batch_size=args.batch_size,
                        device=args.device,
                        prefer_ema=prefer_ema,
                    )
                )
            )
        summary["checkpoint_value_predictions"] = prediction_summaries
    print(json.dumps(summary, indent=2, sort_keys=True))
    raise SystemExit(0)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose async trajectory replay value targets and checkpoint predictions.",
    )
    parser.add_argument(
        "--replay",
        type=Path,
        default=Path("data/runpod/train-v2-gumbel-512k/replay/trajectory-replay.npz"),
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--eval-rows", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prefer-ema", action="store_true")
    parser.add_argument("--compare-ema", action="store_true")
    return parser


def _terminal_value_targets(replay: TrajectoryReplayStore) -> np.ndarray:
    episode_indexes = np.searchsorted(
        replay.episode_offsets,
        np.arange(len(replay), dtype=np.int64),
        side="right",
    ) - 1
    values = np.empty((len(replay),), dtype=np.float32)
    for row in range(len(replay)):
        values[row] = np.float32(
            value_target_for_player(
                player=int(replay.players[row]),
                winner=int(replay.episode_winners[int(episode_indexes[row])]),
            )
        )
    return values


def _evaluate_checkpoint_values(
    *,
    checkpoint: Path,
    replay: TrajectoryReplayStore,
    targets: np.ndarray,
    indexes: np.ndarray,
    batch_size: int,
    device: str,
    prefer_ema: bool,
) -> ValuePredictionSummary:
    import torch

    state = load_checkpoint(checkpoint, device=device, prefer_ema=prefer_ema)
    state.model.eval()
    predictions: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, indexes.shape[0], batch_size):
            batch_indexes = indexes[start : start + batch_size]
            features = torch.from_numpy(
                np.ascontiguousarray(replay.features[batch_indexes], dtype=np.float32)
            ).to(device=device)
            _, value = state.model(features)
            predictions.append(value.detach().cpu().numpy().astype(np.float32, copy=False))
    pred = np.concatenate(predictions) if predictions else np.empty((0,), dtype=np.float32)
    target = targets[indexes]
    error = pred - target
    corr = _correlation(pred, target)
    return ValuePredictionSummary(
        checkpoint=str(checkpoint),
        prefer_ema=prefer_ema,
        rows=int(indexes.shape[0]),
        mse=float(np.mean(error**2)) if error.size else math.nan,
        mae=float(np.mean(np.abs(error))) if error.size else math.nan,
        corr=corr,
        pred_mean=float(pred.mean()) if pred.size else math.nan,
        pred_std=float(pred.std()) if pred.size else math.nan,
        pred_min=float(pred.min()) if pred.size else math.nan,
        pred_max=float(pred.max()) if pred.size else math.nan,
        target_mean=float(target.mean()) if target.size else math.nan,
        target_std=float(target.std()) if target.size else math.nan,
    )


def _sample_indexes(size: int, count: int, rng: random.Random) -> np.ndarray:
    if size <= 0:
        return np.empty((0,), dtype=np.int64)
    resolved = min(size, max(1, count))
    return np.asarray(rng.sample(range(size), resolved), dtype=np.int64)


def _describe_episode_lengths(replay: TrajectoryReplayStore) -> dict[str, float]:
    lengths = np.diff(replay.episode_offsets).astype(np.float32, copy=False)
    return _describe_array(lengths)


def _policy_entropy(policies: np.ndarray) -> np.ndarray:
    policy = np.asarray(policies, dtype=np.float32)
    positive = policy > 0.0
    terms = np.zeros_like(policy, dtype=np.float32)
    terms[positive] = policy[positive] * np.log(np.clip(policy[positive], 1e-45, 1.0))
    return -terms.sum(axis=1)


def _describe_array(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values)
    if array.size == 0:
        return {
            "count": 0,
            "mean": math.nan,
            "std": math.nan,
            "min": math.nan,
            "p05": math.nan,
            "p50": math.nan,
            "p95": math.nan,
            "max": math.nan,
        }
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "p05": float(np.percentile(array, 5)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def _counts(values: np.ndarray, *, limit: int = 40) -> dict[str, int]:
    keys, counts = np.unique(values, return_counts=True)
    items = sorted(
        ((str(_json_scalar(key)), int(count)) for key, count in zip(keys, counts, strict=True)),
        key=lambda item: item[1],
        reverse=True,
    )
    if len(items) <= limit:
        return dict(items)
    trimmed = dict(items[:limit])
    trimmed["<other>"] = sum(count for _, count in items[limit:])
    return trimmed


def _json_scalar(value: Any) -> int | float | str:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2 or right.size < 2:
        return None
    if float(left.std()) == 0.0 or float(right.std()) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


if __name__ == "__main__":
    main()
