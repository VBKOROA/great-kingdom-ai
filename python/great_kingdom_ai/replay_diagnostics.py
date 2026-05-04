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
from great_kingdom_ai.replay_buffer import FEATURE_SHAPE

FloatArray = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]


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
    capacity: int | None = None,
    conflict_samples: int = 20_000,
    seed: int = 0,
    top_k: int = 10,
) -> dict[str, Any]:
    """Return JSON-serializable diagnostics for replay arrays."""
    _validate_shapes(features, policies, values)
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
    return summary


def load_replay_arrays(path: str | Path) -> tuple[FloatArray, FloatArray, FloatArray, int | None]:
    with np.load(Path(path)) as data:
        features = np.asarray(data["features"], dtype=np.float32)
        policies = np.asarray(data["policies"], dtype=np.float32)
        values = np.asarray(data["values"], dtype=np.float32)
        capacity = int(data["capacity"]) if "capacity" in data else None
    return features, policies, values, capacity


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
    features, policies, values, capacity = load_replay_arrays(args.replay)
    summary = summarize_replay_arrays(
        features=features,
        policies=policies,
        values=values,
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
