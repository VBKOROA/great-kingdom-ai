"""Diagnose async actor-learner update pressure between two checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NoReturn

import numpy as np
from great_kingdom_ai.async_v2 import load_learner_v2_config
from great_kingdom_ai.evaluator import evaluate_feature_arrays_logits_values
from great_kingdom_ai.replay import TrajectoryReplayStore
from great_kingdom_ai.self_play_data import value_target_for_player
from great_kingdom_ai.training import load_checkpoint, load_training_config

DEFAULT_REPLAY = Path("data/runpod/train-v3/replay/trajectory-replay.npz")
DEFAULT_TRAIN_CONFIG = Path("configs/runpod/train.json")
DEFAULT_LEARNER_CONFIG = Path("configs/runpod/learner-v2.json")


@dataclass(frozen=True)
class SamplingPressure:
    replay_rows: int
    batch_size: int
    train_steps: int
    train_draws: int
    train_reuse_factor: float
    imported_transitions: int | None
    budget_samples: float | None
    recent_fraction_config: float
    recent_window_config: int
    recent_rows: int
    old_rows: int
    natural_recent_fraction: float
    effective_recent_fraction: float
    recent_draws: float
    old_draws: float
    expected_draws_per_recent_row: float | None
    expected_draws_per_old_row: float | None
    recent_row_overweight: float | None
    recent_fraction_overweight: float | None


@dataclass(frozen=True)
class ArraySummary:
    count: int
    mean: float
    std: float
    min: float
    p05: float
    p50: float
    p95: float
    max: float


@dataclass(frozen=True)
class ProbeSummary:
    name: str
    rows: int
    index_start: int | None
    index_end: int | None
    target_policy_entropy: ArraySummary
    target_policy_max: ArraySummary
    value_target: ArraySummary
    before_target_kl: ArraySummary
    after_target_kl: ArraySummary
    target_kl_delta_mean: float
    before_value_mse: float
    after_value_mse: float
    value_mse_delta: float
    before_value_mae: float
    after_value_mae: float
    value_mae_delta: float
    before_value_corr: float | None
    after_value_corr: float | None
    before_to_after_kl: ArraySummary
    after_to_before_kl: ArraySummary
    top1_flip_rate: float
    value_prediction_delta_abs: ArraySummary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure sampler pressure and checkpoint drift on async trajectory replay. "
            "This is read-only and does not train or modify checkpoints."
        )
    )
    parser.add_argument("--replay", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--before", type=Path, required=True, help="earlier checkpoint")
    parser.add_argument("--after", type=Path, required=True, help="later checkpoint")
    parser.add_argument("--train-config", type=Path, default=DEFAULT_TRAIN_CONFIG)
    parser.add_argument("--learner-config", type=Path, default=DEFAULT_LEARNER_CONFIG)
    parser.add_argument("--rows-per-split", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=1024, help="evaluation batch size")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--before-weights",
        choices=["ema", "raw"],
        default="ema",
        help="weights to load from --before checkpoint",
    )
    parser.add_argument(
        "--after-weights",
        choices=["ema", "raw"],
        default="ema",
        help="weights to load from --after checkpoint",
    )
    parser.add_argument(
        "--imported-transitions",
        type=int,
        default=None,
        help=(
            "optional transitions imported in the learner cycle; when set, train_steps is "
            "computed from imported_transitions * train_reuse_factor"
        ),
    )
    parser.add_argument("--pretty", action="store_true")
    return parser


def main() -> NoReturn:
    args = build_parser().parse_args()
    train_config = load_training_config(args.train_config)
    learner_config = load_learner_v2_config(args.learner_config)
    replay = TrajectoryReplayStore.load(args.replay)
    device = args.device or train_config.device

    pressure = estimate_sampling_pressure(
        replay_rows=len(replay),
        batch_size=train_config.batch_size,
        max_train_steps=train_config.steps,
        recent_fraction=train_config.recent_sample_fraction,
        recent_window=train_config.recent_sample_window,
        train_reuse_factor=learner_config.train_reuse_factor,
        min_replay_transitions=learner_config.min_replay_transitions,
        imported_transitions=args.imported_transitions,
    )
    probes = make_probe_indexes(
        replay_rows=len(replay),
        rows_per_split=args.rows_per_split,
        recent_window=train_config.recent_sample_window,
        seed=args.seed,
    )
    targets = terminal_value_targets(replay)
    before_model = load_checkpoint(
        args.before,
        device=device,
        prefer_ema=args.before_weights == "ema",
    ).model
    after_model = load_checkpoint(
        args.after,
        device=device,
        prefer_ema=args.after_weights == "ema",
    ).model
    before_model.eval()
    after_model.eval()

    probe_summaries = [
        asdict(
            summarize_probe(
                name=name,
                indexes=indexes,
                replay=replay,
                targets=targets,
                before_model=before_model,
                after_model=after_model,
                device=device,
                batch_size=args.batch_size,
            )
        )
        for name, indexes in probes.items()
        if indexes.size > 0
    ]
    payload: dict[str, Any] = {
        "replay": str(args.replay),
        "before": str(args.before),
        "after": str(args.after),
        "before_weights": args.before_weights,
        "after_weights": args.after_weights,
        "device": device,
        "sampling_pressure": asdict(pressure),
        "replay_metadata": replay_metadata_summary(replay),
        "probes": probe_summaries,
    }
    print(json.dumps(payload, indent=2 if args.pretty else None, sort_keys=True))
    raise SystemExit(0)


def estimate_sampling_pressure(
    *,
    replay_rows: int,
    batch_size: int,
    max_train_steps: int,
    recent_fraction: float,
    recent_window: int,
    train_reuse_factor: float,
    min_replay_transitions: int,
    imported_transitions: int | None,
) -> SamplingPressure:
    if replay_rows < 0:
        raise ValueError("replay_rows must be non-negative")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_train_steps < 0:
        raise ValueError("max_train_steps must be non-negative")

    budget_samples = (
        None if imported_transitions is None else imported_transitions * train_reuse_factor
    )
    if replay_rows < min_replay_transitions:
        train_steps = 0
    elif budget_samples is None:
        train_steps = max_train_steps
    else:
        train_steps = min(max_train_steps, math.floor(budget_samples / batch_size))
    train_draws = train_steps * batch_size
    recent_rows = min(max(0, recent_window), replay_rows)
    old_rows = max(0, replay_rows - recent_rows)
    natural_recent_fraction = recent_rows / replay_rows if replay_rows > 0 else 0.0
    if recent_fraction <= 0.0:
        recent_draws = train_draws * natural_recent_fraction
        old_draws = train_draws - recent_draws
    else:
        recent_take_per_batch, old_take_per_batch = _per_batch_recent_old_takes(
            batch_size=batch_size,
            recent_fraction=recent_fraction,
            recent_rows=recent_rows,
            old_rows=old_rows,
        )
        recent_draws = float(train_steps * recent_take_per_batch)
        old_draws = float(train_steps * old_take_per_batch)
    expected_recent = recent_draws / recent_rows if recent_rows > 0 else None
    expected_old = old_draws / old_rows if old_rows > 0 else None
    effective_recent_fraction = recent_draws / train_draws if train_draws > 0 else 0.0
    recent_row_overweight = (
        None
        if expected_recent is None or expected_old is None or expected_old == 0.0
        else expected_recent / expected_old
    )
    recent_fraction_overweight = (
        None
        if natural_recent_fraction <= 0.0
        else effective_recent_fraction / natural_recent_fraction
    )
    return SamplingPressure(
        replay_rows=replay_rows,
        batch_size=batch_size,
        train_steps=train_steps,
        train_draws=train_draws,
        train_reuse_factor=train_reuse_factor,
        imported_transitions=imported_transitions,
        budget_samples=budget_samples,
        recent_fraction_config=recent_fraction,
        recent_window_config=recent_window,
        recent_rows=recent_rows,
        old_rows=old_rows,
        natural_recent_fraction=natural_recent_fraction,
        effective_recent_fraction=effective_recent_fraction,
        recent_draws=recent_draws,
        old_draws=old_draws,
        expected_draws_per_recent_row=expected_recent,
        expected_draws_per_old_row=expected_old,
        recent_row_overweight=recent_row_overweight,
        recent_fraction_overweight=recent_fraction_overweight,
    )


def _per_batch_recent_old_takes(
    *,
    batch_size: int,
    recent_fraction: float,
    recent_rows: int,
    old_rows: int,
) -> tuple[int, int]:
    if recent_fraction <= 0.0:
        return (0, min(batch_size, old_rows + recent_rows))
    target_recent = round(batch_size * recent_fraction)
    recent_take = min(target_recent, recent_rows, batch_size)
    old_take = min(batch_size - recent_take, old_rows)
    recent_take = min(batch_size - old_take, recent_rows)
    old_take = batch_size - recent_take
    if old_take > old_rows:
        old_take = old_rows
        recent_take = batch_size - old_take
    if recent_take > recent_rows:
        recent_take = recent_rows
        old_take = min(batch_size - recent_take, old_rows)
    return (recent_take, old_take)


def make_probe_indexes(
    *,
    replay_rows: int,
    rows_per_split: int,
    recent_window: int,
    seed: int,
) -> dict[str, np.ndarray]:
    if rows_per_split <= 0:
        raise ValueError("rows_per_split must be positive")
    recent_count = min(max(0, recent_window), replay_rows)
    old_count = max(0, replay_rows - recent_count)
    return {
        "all": _sample_indexes(0, replay_rows, rows_per_split, random.Random(seed)),
        "old": _sample_indexes(0, old_count, rows_per_split, random.Random(seed + 1)),
        "recent": _sample_indexes(
            old_count,
            replay_rows,
            rows_per_split,
            random.Random(seed + 2),
        ),
    }


def summarize_probe(
    *,
    name: str,
    indexes: np.ndarray,
    replay: TrajectoryReplayStore,
    targets: np.ndarray,
    before_model: Any,
    after_model: Any,
    device: str,
    batch_size: int,
) -> ProbeSummary:
    features = np.ascontiguousarray(replay.features[indexes], dtype=np.float32)
    legal_masks = np.ascontiguousarray(replay.legal_masks[indexes], dtype=np.bool_)
    target_policies = np.ascontiguousarray(replay.policy_targets[indexes], dtype=np.float32)
    value_targets = np.ascontiguousarray(targets[indexes], dtype=np.float32)
    before_logits, before_values = evaluate_model(
        before_model,
        features=features,
        legal_masks=legal_masks,
        device=device,
        batch_size=batch_size,
    )
    after_logits, after_values = evaluate_model(
        after_model,
        features=features,
        legal_masks=legal_masks,
        device=device,
        batch_size=batch_size,
    )
    before_log_probs = masked_log_softmax(before_logits, legal_masks)
    after_log_probs = masked_log_softmax(after_logits, legal_masks)
    before_probs = np.exp(before_log_probs).astype(np.float32)
    after_probs = np.exp(after_log_probs).astype(np.float32)
    before_top1 = np.argmax(before_probs, axis=1)
    after_top1 = np.argmax(after_probs, axis=1)
    before_target_kl = categorical_kl_from_log_probs(target_policies, before_log_probs)
    after_target_kl = categorical_kl_from_log_probs(target_policies, after_log_probs)
    before_to_after = categorical_kl_from_log_probs(before_probs, after_log_probs)
    after_to_before = categorical_kl_from_log_probs(after_probs, before_log_probs)
    before_value_error = before_values - value_targets
    after_value_error = after_values - value_targets
    return ProbeSummary(
        name=name,
        rows=int(indexes.size),
        index_start=int(indexes.min()) if indexes.size else None,
        index_end=int(indexes.max()) if indexes.size else None,
        target_policy_entropy=describe_array(policy_entropy(target_policies)),
        target_policy_max=describe_array(target_policies.max(axis=1)),
        value_target=describe_array(value_targets),
        before_target_kl=describe_array(before_target_kl),
        after_target_kl=describe_array(after_target_kl),
        target_kl_delta_mean=float(after_target_kl.mean() - before_target_kl.mean()),
        before_value_mse=float(np.mean(before_value_error**2)),
        after_value_mse=float(np.mean(after_value_error**2)),
        value_mse_delta=float(np.mean(after_value_error**2) - np.mean(before_value_error**2)),
        before_value_mae=float(np.mean(np.abs(before_value_error))),
        after_value_mae=float(np.mean(np.abs(after_value_error))),
        value_mae_delta=float(
            np.mean(np.abs(after_value_error)) - np.mean(np.abs(before_value_error))
        ),
        before_value_corr=correlation(before_values, value_targets),
        after_value_corr=correlation(after_values, value_targets),
        before_to_after_kl=describe_array(before_to_after),
        after_to_before_kl=describe_array(after_to_before),
        top1_flip_rate=float(np.mean(before_top1 != after_top1)),
        value_prediction_delta_abs=describe_array(np.abs(after_values - before_values)),
    )


def evaluate_model(
    model: Any,
    *,
    features: np.ndarray,
    legal_masks: np.ndarray,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    logits: list[np.ndarray] = []
    values: list[np.ndarray] = []
    for start in range(0, features.shape[0], batch_size):
        end = start + batch_size
        evaluation = evaluate_feature_arrays_logits_values(
            model,
            features[start:end],
            legal_masks[start:end],
            device=device,
        )
        logits.append(evaluation.policy_logits)
        values.append(evaluation.value)
    return (
        np.concatenate(logits, axis=0) if logits else np.empty((0, 0), dtype=np.float32),
        np.concatenate(values, axis=0) if values else np.empty((0,), dtype=np.float32),
    )


def terminal_value_targets(replay: TrajectoryReplayStore) -> np.ndarray:
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


def masked_log_softmax(logits: np.ndarray, legal_masks: np.ndarray) -> np.ndarray:
    masked = np.where(legal_masks, logits, -np.inf).astype(np.float32, copy=False)
    row_max = np.max(masked, axis=1, keepdims=True)
    if not np.isfinite(row_max).all():
        raise ValueError("each row must have at least one legal action")
    shifted = masked - row_max
    exp = np.where(legal_masks, np.exp(shifted), 0.0)
    logsumexp = np.log(exp.sum(axis=1, keepdims=True)) + row_max
    return np.where(legal_masks, masked - logsumexp, -np.inf).astype(np.float32, copy=False)


def categorical_kl_from_log_probs(policy: np.ndarray, log_probs: np.ndarray) -> np.ndarray:
    clipped_policy = np.clip(policy, 0.0, 1.0).astype(np.float32, copy=False)
    positive = clipped_policy > 0.0
    target_log = np.zeros_like(clipped_policy, dtype=np.float32)
    target_log[positive] = np.log(np.clip(clipped_policy[positive], 1e-45, 1.0))
    terms = np.zeros_like(clipped_policy, dtype=np.float32)
    terms[positive] = clipped_policy[positive] * (
        target_log[positive] - log_probs[positive]
    )
    return np.maximum(terms.sum(axis=1), 0.0).astype(np.float32, copy=False)


def policy_entropy(policy: np.ndarray) -> np.ndarray:
    positive = policy > 0.0
    terms = np.zeros_like(policy, dtype=np.float32)
    terms[positive] = policy[positive] * np.log(np.clip(policy[positive], 1e-45, 1.0))
    return (-terms.sum(axis=1)).astype(np.float32, copy=False)


def replay_metadata_summary(replay: TrajectoryReplayStore) -> dict[str, Any]:
    lengths = np.diff(replay.episode_offsets).astype(np.float32, copy=False)
    return {
        "rows": len(replay),
        "episodes": replay.episode_count,
        "capacity": replay.capacity,
        "episode_lengths": asdict(describe_array(lengths)),
        "model_versions": counts(replay.model_versions),
        "created_iterations": counts(replay.created_iterations),
        "episode_winners": counts(replay.episode_winners),
        "players": counts(replay.players),
        "sample_weights": asdict(describe_array(replay.sample_weights)),
    }


def describe_array(values: np.ndarray) -> ArraySummary:
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        return ArraySummary(
            count=0,
            mean=math.nan,
            std=math.nan,
            min=math.nan,
            p05=math.nan,
            p50=math.nan,
            p95=math.nan,
            max=math.nan,
        )
    return ArraySummary(
        count=int(array.size),
        mean=float(array.mean()),
        std=float(array.std()),
        min=float(array.min()),
        p05=float(np.percentile(array, 5)),
        p50=float(np.percentile(array, 50)),
        p95=float(np.percentile(array, 95)),
        max=float(array.max()),
    )


def counts(values: np.ndarray, *, limit: int = 40) -> dict[str, int]:
    keys, raw_counts = np.unique(values, return_counts=True)
    items = sorted(
        (
            (str(key.item() if isinstance(key, np.generic) else key), int(count))
            for key, count in zip(keys, raw_counts, strict=True)
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    if len(items) <= limit:
        return dict(items)
    trimmed = dict(items[:limit])
    trimmed["<other>"] = sum(count for _key, count in items[limit:])
    return trimmed


def correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2 or right.size < 2:
        return None
    if float(left.std()) == 0.0 or float(right.std()) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _sample_indexes(
    start: int,
    end: int,
    count: int,
    rng: random.Random,
) -> np.ndarray:
    size = max(0, end - start)
    if size == 0:
        return np.empty((0,), dtype=np.int64)
    resolved = min(size, count)
    indexes = rng.sample(range(start, end), resolved)
    indexes.sort()
    return np.asarray(indexes, dtype=np.int64)


if __name__ == "__main__":
    main()
