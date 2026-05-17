"""Replay dataset diagnostics for policy-value training."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn, cast

import numpy as np
import numpy.typing as npt

from great_kingdom_ai.features import (
    ACTION_SPACE,
    BOARD_CELLS,
    BOARD_SIZE,
    LEGAL_PLACE_FEATURE_CHANNEL,
    PASS_ACTION,
)
from great_kingdom_ai.replay.schema import FEATURE_SHAPE

FloatArray = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]

OWN_REMAINING_CASTLES_FEATURE_CHANNEL = 7
OPPONENT_REMAINING_CASTLES_FEATURE_CHANNEL = 8
CASTLES_PER_PLAYER = 40
MOVE_COUNT_BUCKETS = (0, 1, 2, 4, 8, 16, 24, 32, 48, 64, 80)


@dataclass
class ConflictGroup:
    count: int
    values: set[float]
    policy_argmaxes: set[int]


def summarize_replay_arrays(
    *,
    features: FloatArray,
    policies: FloatArray,
    values: FloatArray,
    root_policy_logits: FloatArray | None = None,
    counts: npt.NDArray[np.integer[Any]] | None = None,
    sample_weights: FloatArray | None = None,
    capacity: int | None = None,
    conflict_samples: int = 20_000,
    seed: int = 0,
    top_k: int = 10,
) -> dict[str, Any]:
    """Return JSON-serializable diagnostics for replay arrays."""
    _validate_shapes(features, policies, values)
    if root_policy_logits is not None:
        _validate_root_policy_logits(root_policy_logits, policies.shape)
    if counts is not None and counts.shape != values.shape:
        raise ValueError(f"expected counts shape {(features.shape[0],)}, got {counts.shape}")
    if sample_weights is not None and sample_weights.shape != values.shape:
        raise ValueError(
            f"expected sample_weights shape {(features.shape[0],)}, got {sample_weights.shape}"
        )
    if conflict_samples < 0:
        raise ValueError("conflict_samples must be non-negative")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    legal_mask = _legal_mask_from_features(features)
    policy_entropy = _policy_entropy(policies)
    policy_support = np.count_nonzero(policies > 1e-6, axis=1)
    policy_argmax = np.argmax(policies, axis=1)
    policy_max = np.max(policies, axis=1)
    policy_sum = np.sum(policies, axis=1)
    illegal_target_mass = np.where(~legal_mask, policies, 0.0).sum(axis=1)
    legal_action_counts = legal_mask.sum(axis=1)

    summary: dict[str, Any] = {
        "samples": int(features.shape[0]),
        "capacity": capacity,
        "features_shape": list(features.shape),
        "policies_shape": list(policies.shape),
        "values_shape": list(values.shape),
        "policy": {
            "entropy": _describe(policy_entropy),
            "support": _describe(policy_support.astype(np.float32)),
            "max_probability": _describe(policy_max),
            "sum": _describe(policy_sum),
            "argmax_top": _top_actions(policy_argmax, top_k=top_k),
        },
        "value": {
            **_describe(values),
            "negative_fraction": float(np.mean(values < -0.5)),
            "drawish_fraction": float(np.mean(np.abs(values) <= 0.5)),
            "positive_fraction": float(np.mean(values > 0.5)),
        },
        "legal": {
            "legal_actions": _describe(legal_action_counts.astype(np.float32)),
            "illegal_target_mass": _describe(illegal_target_mass.astype(np.float32)),
            "rows_with_illegal_target_mass": int(np.count_nonzero(illegal_target_mass > 1e-5)),
        },
    }
    if conflict_samples > 0:
        summary["conflicts"] = _sample_conflicts(
            features=features,
            policies=policies,
            values=values,
            sample_count=min(conflict_samples, features.shape[0]),
            seed=seed,
            top_k=top_k,
        )
    if root_policy_logits is not None:
        summary["target_vs_prior"] = _target_vs_prior_diagnostics(
            policies=policies,
            root_policy_logits=root_policy_logits,
            legal_mask=legal_mask,
            top_k=top_k,
        )
    if counts is not None:
        summary["aggregate_counts"] = {
            **_describe(counts.astype(np.float32)),
            "duplicate_groups": int(np.count_nonzero(counts > 1)),
            "represented_raw_rows": int(np.sum(counts)),
        }
    if sample_weights is not None:
        summary["sample_weight"] = {
            **_describe(sample_weights),
            "effective_weighted_rows": float(np.sum(sample_weights)),
        }
    summary["move_count"] = _move_count_diagnostics(
        features=features,
        counts=counts,
        sample_weights=sample_weights,
        top_k=top_k,
    )
    return summary


def load_replay_arrays(
    path: str | Path,
) -> tuple[
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray | None,
    npt.NDArray[np.integer[Any]] | None,
    FloatArray | None,
    int | None,
]:
    with np.load(Path(path)) as data:
        features = np.asarray(data["features"], dtype=np.float32)
        policies = np.asarray(data["policies"], dtype=np.float32)
        values = np.asarray(data["values"], dtype=np.float32)
        root_policy_logits = (
            np.asarray(data["root_policy_logits"], dtype=np.float32)
            if "root_policy_logits" in data
            else None
        )
        counts = np.asarray(data["counts"], dtype=np.int64) if "counts" in data else None
        sample_weights = (
            np.asarray(data["sample_weights"], dtype=np.float32)
            if "sample_weights" in data
            else None
        )
        capacity = int(data["capacity"]) if "capacity" in data else None
    return features, policies, values, root_policy_logits, counts, sample_weights, capacity


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize replay target distribution")
    parser.add_argument("--replay", type=Path, required=True, help="Path to replay .npz file")
    parser.add_argument(
        "--conflict-samples",
        type=int,
        default=20_000,
        help="number of sampled rows used for exact duplicate feature conflict checks",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--pretty", action="store_true", help="print indented JSON")
    return parser


def _validate_shapes(features: FloatArray, policies: FloatArray, values: FloatArray) -> None:
    if features.ndim != 4 or tuple(features.shape[1:]) != FEATURE_SHAPE:
        raise ValueError(f"expected features shape [N, {FEATURE_SHAPE}], got {features.shape}")
    if policies.shape != (features.shape[0], ACTION_SPACE):
        raise ValueError(
            f"expected policies shape {(features.shape[0], ACTION_SPACE)}, got {policies.shape}"
        )
    if values.shape != (features.shape[0],):
        raise ValueError(f"expected values shape {(features.shape[0],)}, got {values.shape}")


def _validate_root_policy_logits(
    root_policy_logits: FloatArray,
    expected_shape: tuple[int, int],
) -> None:
    if root_policy_logits.shape != expected_shape:
        raise ValueError(
            f"expected root_policy_logits shape {expected_shape}, "
            f"got {root_policy_logits.shape}"
        )


def _legal_mask_from_features(features: FloatArray) -> BoolArray:
    legal_place = features[:, LEGAL_PLACE_FEATURE_CHANNEL].reshape(-1, BOARD_CELLS) > 0.5
    legal_mask = np.zeros((features.shape[0], ACTION_SPACE), dtype=np.bool_)
    legal_mask[:, :BOARD_CELLS] = legal_place
    legal_mask[:, PASS_ACTION] = True
    return legal_mask


def _policy_entropy(policies: FloatArray) -> FloatArray:
    positive = policies > 0.0
    log_policy = np.zeros_like(policies, dtype=np.float32)
    log_policy[positive] = np.log(policies[positive])
    entropy = -np.sum(np.where(positive, policies * log_policy, 0.0), axis=1)
    return cast(FloatArray, entropy)


def _target_vs_prior_diagnostics(
    *,
    policies: FloatArray,
    root_policy_logits: FloatArray,
    legal_mask: BoolArray,
    top_k: int,
) -> dict[str, Any]:
    available = np.isfinite(root_policy_logits).all(axis=1)
    missing_rows = int(root_policy_logits.shape[0] - np.count_nonzero(available))
    if not np.any(available):
        return {
            "available_rows": 0,
            "missing_rows": missing_rows,
        }

    target = policies[available]
    prior = _masked_softmax(root_policy_logits[available], legal_mask[available])
    target_argmax = np.argmax(target, axis=1)
    prior_argmax = np.argmax(prior, axis=1)
    target_max = np.max(target, axis=1)
    prior_max = np.max(prior, axis=1)
    top1_delta = target_max - prior_max
    target_prior_kl = _categorical_kl(target, prior)
    prior_target_kl = _categorical_kl(prior, target)
    argmax_mismatch = target_argmax != prior_argmax

    return {
        "available_rows": int(np.count_nonzero(available)),
        "missing_rows": missing_rows,
        "kl_target_prior": _describe(target_prior_kl),
        "kl_prior_target": _describe(prior_target_kl),
        "argmax_mismatch_ratio": float(np.mean(argmax_mismatch)),
        "argmax_mismatch_count": int(np.count_nonzero(argmax_mismatch)),
        "top1_probability_delta": _describe(top1_delta.astype(np.float32)),
        "target_argmax_top": _top_actions(target_argmax, top_k=top_k),
        "prior_argmax_top": _top_actions(prior_argmax, top_k=top_k),
    }


def _masked_softmax(logits: FloatArray, legal_mask: BoolArray) -> FloatArray:
    masked_logits = np.where(legal_mask, logits, -np.inf)
    row_max = np.max(masked_logits, axis=1, keepdims=True)
    shifted = np.where(legal_mask, masked_logits - row_max, -np.inf)
    exp_values = np.where(legal_mask, np.exp(shifted), 0.0)
    sums = exp_values.sum(axis=1, keepdims=True)
    if np.any(sums <= 0.0):
        raise ValueError("legal mask must leave at least one action per row")
    return cast(FloatArray, (exp_values / sums).astype(np.float32))


def _categorical_kl(left: FloatArray, right: FloatArray) -> FloatArray:
    epsilon = np.float32(1e-8)
    left_clipped = np.clip(left, epsilon, 1.0)
    right_clipped = np.clip(right, epsilon, 1.0)
    kl = np.sum(left_clipped * (np.log(left_clipped) - np.log(right_clipped)), axis=1)
    return cast(FloatArray, kl.astype(np.float32))


def _move_count_diagnostics(
    *,
    features: FloatArray,
    counts: npt.NDArray[np.integer[Any]] | None,
    sample_weights: FloatArray | None,
    top_k: int,
) -> dict[str, Any]:
    move_counts = _move_counts_from_features(features)
    raw_counts = (
        counts.astype(np.float64)
        if counts is not None
        else np.ones(features.shape[0], dtype=np.float64)
    )
    weighted_counts = (
        sample_weights.astype(np.float64)
        if sample_weights is not None
        else np.ones(features.shape[0], dtype=np.float64)
    )
    return {
        "unique_rows": _describe(move_counts.astype(np.float32)),
        "raw_rows": _weighted_move_count_summary(move_counts, raw_counts),
        "weighted_rows": _weighted_move_count_summary(move_counts, weighted_counts),
        "top_move_counts_by_raw_rows": _top_move_counts(move_counts, raw_counts, top_k=top_k),
        "buckets": _move_count_buckets(move_counts, raw_counts, weighted_counts),
    }


def _move_counts_from_features(features: FloatArray) -> npt.NDArray[np.int64]:
    own_remaining = features[:, OWN_REMAINING_CASTLES_FEATURE_CHANNEL].mean(axis=(1, 2))
    opponent_remaining = features[:, OPPONENT_REMAINING_CASTLES_FEATURE_CHANNEL].mean(axis=(1, 2))
    used = (2 * CASTLES_PER_PLAYER) - (
        (own_remaining + opponent_remaining) * CASTLES_PER_PLAYER
    )
    move_counts = np.rint(used).clip(0, 2 * CASTLES_PER_PLAYER).astype(np.int64)
    return cast(npt.NDArray[np.int64], move_counts)


def _weighted_move_count_summary(
    move_counts: npt.NDArray[np.integer[Any]],
    weights: npt.NDArray[np.floating[Any]],
) -> dict[str, float]:
    total_weight = float(np.sum(weights))
    if total_weight <= 0.0:
        raise ValueError("move-count weights must have positive mass")
    order = np.argsort(move_counts)
    sorted_moves = move_counts[order].astype(np.float64)
    sorted_weights = weights[order].astype(np.float64)
    cumulative = np.cumsum(sorted_weights)

    def percentile(fraction: float) -> float:
        index = int(np.searchsorted(cumulative, total_weight * fraction, side="left"))
        return float(sorted_moves[min(index, sorted_moves.size - 1)])

    return {
        "min": float(np.min(move_counts)),
        "p10": percentile(0.10),
        "mean": float(np.sum(move_counts.astype(np.float64) * weights) / total_weight),
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "max": float(np.max(move_counts)),
        "total_weight": total_weight,
    }


def _top_move_counts(
    move_counts: npt.NDArray[np.integer[Any]],
    weights: npt.NDArray[np.floating[Any]],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    totals: Counter[int] = Counter()
    for move_count, weight in zip(move_counts.tolist(), weights.tolist(), strict=True):
        totals[int(move_count)] += float(weight)
    total_weight = max(1e-12, float(sum(totals.values())))
    return [
        {
            "move_count": move_count,
            "raw_rows": weight,
            "fraction": weight / total_weight,
        }
        for move_count, weight in totals.most_common(top_k)
    ]


def _move_count_buckets(
    move_counts: npt.NDArray[np.integer[Any]],
    raw_counts: npt.NDArray[np.floating[Any]],
    weighted_counts: npt.NDArray[np.floating[Any]],
) -> list[dict[str, Any]]:
    buckets = []
    total_unique = max(1, move_counts.size)
    total_raw = max(1e-12, float(np.sum(raw_counts)))
    total_weighted = max(1e-12, float(np.sum(weighted_counts)))
    for start, end in zip(MOVE_COUNT_BUCKETS[:-1], MOVE_COUNT_BUCKETS[1:], strict=True):
        if end == MOVE_COUNT_BUCKETS[-1]:
            mask = (move_counts >= start) & (move_counts <= end)
            label = f"{start}-{end}"
        else:
            mask = (move_counts >= start) & (move_counts < end)
            label = f"{start}-{end - 1}"
        unique_rows = int(np.count_nonzero(mask))
        raw_rows = float(np.sum(raw_counts[mask]))
        weighted_rows = float(np.sum(weighted_counts[mask]))
        buckets.append(
            {
                "moves": label,
                "unique_rows": unique_rows,
                "unique_fraction": unique_rows / total_unique,
                "raw_rows": raw_rows,
                "raw_fraction": raw_rows / total_raw,
                "weighted_rows": weighted_rows,
                "weighted_fraction": weighted_rows / total_weighted,
            }
        )
    return buckets


def _describe(values: npt.NDArray[np.floating[Any]]) -> dict[str, float]:
    if values.size == 0:
        raise ValueError("cannot describe an empty array")
    return {
        "min": float(np.min(values)),
        "p10": float(np.percentile(values, 10)),
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
        "std": float(np.std(values)),
    }


def _top_actions(actions: npt.NDArray[np.integer[Any]], *, top_k: int) -> list[dict[str, Any]]:
    counts = Counter(int(action) for action in actions.tolist())
    total = max(1, len(actions))
    return [
        {
            "action": action,
            "label": _action_label(action),
            "count": count,
            "fraction": count / total,
        }
        for action, count in counts.most_common(top_k)
    ]


def _sample_conflicts(
    *,
    features: FloatArray,
    policies: FloatArray,
    values: FloatArray,
    sample_count: int,
    seed: int,
    top_k: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    indexes = rng.choice(features.shape[0], size=sample_count, replace=False)
    groups: dict[str, ConflictGroup] = {}
    for index in indexes.tolist():
        digest = hashlib.blake2b(features[index].tobytes(), digest_size=16).hexdigest()
        group = groups.get(digest)
        if group is None:
            group = ConflictGroup(count=0, values=set(), policy_argmaxes=set())
            groups[digest] = group
        group.count += 1
        group.values.add(round(float(values[index]), 6))
        group.policy_argmaxes.add(int(np.argmax(policies[index])))

    duplicate_groups = [group for group in groups.values() if group.count > 1]
    value_conflicts = [group for group in duplicate_groups if len(group.values) > 1]
    policy_conflicts = [group for group in duplicate_groups if len(group.policy_argmaxes) > 1]
    return {
        "sampled_rows": sample_count,
        "unique_feature_hashes": len(groups),
        "duplicate_groups": len(duplicate_groups),
        "duplicate_rows": sum(group.count for group in duplicate_groups),
        "value_conflict_groups": len(value_conflicts),
        "policy_argmax_conflict_groups": len(policy_conflicts),
        "largest_duplicate_groups": [
            {
                "count": group.count,
                "values": sorted(group.values),
                "policy_argmaxes": [
                    {"action": action, "label": _action_label(action)}
                    for action in sorted(group.policy_argmaxes)
                ],
            }
            for group in sorted(duplicate_groups, key=lambda item: item.count, reverse=True)[:top_k]
        ],
    }


def _action_label(action: int) -> str:
    if action == PASS_ACTION:
        return "PASS"
    row = action // BOARD_SIZE
    col = action % BOARD_SIZE
    return f"{chr(ord('A') + col)}{row + 1}"


def main() -> NoReturn:
    args = build_parser().parse_args()
    (
        features,
        policies,
        values,
        root_policy_logits,
        counts,
        sample_weights,
        capacity,
    ) = load_replay_arrays(args.replay)
    summary = summarize_replay_arrays(
        features=features,
        policies=policies,
        values=values,
        root_policy_logits=root_policy_logits,
        counts=counts,
        sample_weights=sample_weights,
        capacity=capacity,
        conflict_samples=args.conflict_samples,
        seed=args.seed,
        top_k=args.top_k,
    )
    payload = {"event": "replay_diagnostics", "replay": str(args.replay), **summary}
    print(json.dumps(payload, indent=2 if args.pretty else None, sort_keys=True))
    raise SystemExit(0)


if __name__ == "__main__":
    main()


__all__ = [
    "build_parser",
    "load_replay_arrays",
    "summarize_replay_arrays",
]
