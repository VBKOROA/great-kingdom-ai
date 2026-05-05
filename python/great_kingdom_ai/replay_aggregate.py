"""Aggregate duplicate replay states by averaging their policy and value targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn, cast

import numpy as np
import numpy.typing as npt

from great_kingdom_ai.features import ACTION_SPACE
from great_kingdom_ai.replay_buffer import FEATURE_SHAPE

FloatArray = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class AggregatedReplay:
    features: FloatArray
    policies: FloatArray
    values: FloatArray
    counts: IntArray
    sample_weights: FloatArray
    capacity: int
    root_policy_logits: FloatArray | None = None


def aggregate_duplicate_replay(
    *,
    features: FloatArray,
    policies: FloatArray,
    values: FloatArray,
    root_policy_logits: FloatArray | None = None,
    capacity: int | None = None,
    sample_weight_mode: str = "sqrt_count",
    sample_weight_cap: float | None = 16.0,
) -> AggregatedReplay:
    """Merge exact duplicate feature rows and average their training targets."""
    _validate_arrays(features, policies, values, root_policy_logits)
    groups: dict[str, int] = {}
    first_indexes: list[int] = []
    policy_sums: list[FloatArray] = []
    value_sums: list[float] = []
    root_policy_sums: list[FloatArray] | None = [] if root_policy_logits is not None else None
    root_policy_counts: list[int] | None = [] if root_policy_logits is not None else None
    counts: list[int] = []

    for index in range(features.shape[0]):
        digest = hashlib.blake2b(features[index].tobytes(), digest_size=16).hexdigest()
        root_logits = (
            root_policy_logits[index]
            if root_policy_logits is not None
            and np.isfinite(root_policy_logits[index]).all()
            else None
        )
        group_index = groups.get(digest)
        if group_index is None:
            groups[digest] = len(first_indexes)
            first_indexes.append(index)
            policy_sums.append(policies[index].astype(np.float32, copy=True))
            value_sums.append(float(values[index]))
            if root_policy_sums is not None and root_policy_counts is not None:
                if root_logits is None:
                    root_policy_sums.append(np.zeros(ACTION_SPACE, dtype=np.float32))
                    root_policy_counts.append(0)
                else:
                    root_policy_sums.append(root_logits.astype(np.float32, copy=True))
                    root_policy_counts.append(1)
            counts.append(1)
        else:
            policy_sums[group_index] += policies[index]
            value_sums[group_index] += float(values[index])
            if (
                root_logits is not None
                and root_policy_sums is not None
                and root_policy_counts is not None
            ):
                root_policy_sums[group_index] += root_logits
                root_policy_counts[group_index] += 1
            counts[group_index] += 1

    count_array = np.asarray(counts, dtype=np.int64)
    aggregated_policies = np.stack(policy_sums, axis=0).astype(np.float32)
    aggregated_policies /= count_array[:, None].astype(np.float32)
    aggregated_policies = _renormalized_policies(aggregated_policies)
    aggregated_values = (
        np.asarray(value_sums, dtype=np.float32) / count_array.astype(np.float32)
    ).astype(np.float32)
    aggregated_root_logits = _averaged_root_policy_logits(
        root_policy_sums,
        root_policy_counts,
        len(first_indexes),
    )
    sample_weights = _sample_weights_from_counts(
        count_array,
        mode=sample_weight_mode,
        cap=sample_weight_cap,
    )
    output_capacity = capacity if capacity is not None else max(1, len(first_indexes))
    output_capacity = max(output_capacity, len(first_indexes))
    return AggregatedReplay(
        features=features[np.asarray(first_indexes, dtype=np.int64)].astype(np.float32, copy=True),
        policies=aggregated_policies,
        values=aggregated_values,
        counts=count_array,
        sample_weights=sample_weights,
        capacity=output_capacity,
        root_policy_logits=aggregated_root_logits,
    )


def load_replay(
    path: str | Path,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray | None, int | None]:
    with np.load(Path(path)) as data:
        features = np.asarray(data["features"], dtype=np.float32)
        policies = np.asarray(data["policies"], dtype=np.float32)
        values = np.asarray(data["values"], dtype=np.float32)
        root_policy_logits = (
            np.asarray(data["root_policy_logits"], dtype=np.float32)
            if "root_policy_logits" in data
            else None
        )
        capacity = int(data["capacity"]) if "capacity" in data else None
    return features, policies, values, root_policy_logits, capacity


def save_replay(path: str | Path, replay: AggregatedReplay) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "capacity": np.asarray(replay.capacity, dtype=np.int64),
        "features": replay.features,
        "policies": replay.policies,
        "values": replay.values,
        "counts": replay.counts,
        "sample_weights": replay.sample_weights,
    }
    if replay.root_policy_logits is not None:
        payload["root_policy_logits"] = replay.root_policy_logits
    np.savez_compressed(destination, **cast(dict[str, Any], payload))
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Average duplicate replay states into one row")
    parser.add_argument("--input", type=Path, required=True, help="Input replay .npz path")
    parser.add_argument("--output", type=Path, required=True, help="Output replay .npz path")
    parser.add_argument(
        "--capacity",
        type=int,
        default=None,
        help="optional output replay capacity; defaults to at least the unique row count",
    )
    parser.add_argument("--pretty", action="store_true", help="print indented JSON")
    parser.add_argument(
        "--sample-weight-mode",
        choices=["none", "count", "sqrt_count", "log_count", "capped_count"],
        default="sqrt_count",
        help="how aggregated row weights are derived from duplicate counts",
    )
    parser.add_argument(
        "--sample-weight-cap",
        type=float,
        default=16.0,
        help="optional cap applied by capped_count mode and as a final weight limit",
    )
    return parser


def _validate_arrays(
    features: FloatArray,
    policies: FloatArray,
    values: FloatArray,
    root_policy_logits: FloatArray | None,
) -> None:
    if features.ndim != 4 or tuple(features.shape[1:]) != FEATURE_SHAPE:
        raise ValueError(f"expected features shape [N, {FEATURE_SHAPE}], got {features.shape}")
    if policies.shape != (features.shape[0], ACTION_SPACE):
        raise ValueError(
            f"expected policies shape {(features.shape[0], ACTION_SPACE)}, got {policies.shape}"
        )
    if values.shape != (features.shape[0],):
        raise ValueError(f"expected values shape {(features.shape[0],)}, got {values.shape}")
    if root_policy_logits is not None and root_policy_logits.shape != policies.shape:
        raise ValueError(
            "expected root_policy_logits shape "
            f"{policies.shape}, got {root_policy_logits.shape}"
        )


def _renormalized_policies(policies: FloatArray) -> FloatArray:
    sums = policies.sum(axis=1, keepdims=True)
    if np.any(sums <= 0.0):
        raise ValueError("policy rows must have positive mass")
    return cast(FloatArray, (policies / sums).astype(np.float32))


def _averaged_root_policy_logits(
    root_policy_sums: list[FloatArray] | None,
    root_policy_counts: list[int] | None,
    rows: int,
) -> FloatArray | None:
    if root_policy_sums is None or root_policy_counts is None:
        return None
    averaged = np.full((rows, ACTION_SPACE), np.nan, dtype=np.float32)
    for index, count in enumerate(root_policy_counts):
        if count <= 0:
            continue
        averaged[index] = root_policy_sums[index] / np.float32(count)
    return averaged


def _sample_weights_from_counts(
    counts: IntArray,
    *,
    mode: str,
    cap: float | None,
) -> FloatArray:
    if counts.size == 0:
        return np.empty((0,), dtype=np.float32)
    if mode == "none":
        weights = np.ones(counts.shape, dtype=np.float32)
    elif mode == "count":
        weights = counts.astype(np.float32)
    elif mode == "sqrt_count":
        weights = np.sqrt(counts.astype(np.float32))
    elif mode == "log_count":
        weights = np.log1p(counts.astype(np.float32))
    elif mode == "capped_count":
        if cap is None:
            raise ValueError("sample_weight_cap is required for capped_count mode")
        weights = np.minimum(counts.astype(np.float32), np.float32(cap))
    else:
        raise ValueError(f"unknown sample_weight_mode: {mode}")
    if cap is not None:
        if not math.isfinite(cap) or cap <= 0.0:
            raise ValueError("sample_weight_cap must be finite and positive")
        weights = np.minimum(weights, np.float32(cap))
    if np.any(weights <= 0.0) or not np.isfinite(weights).all():
        raise ValueError("sample weights must be finite and positive")
    return cast(FloatArray, weights.astype(np.float32))


def _describe(values: npt.NDArray[np.floating[Any]]) -> dict[str, float]:
    if values.size == 0:
        return {
            "min": 0.0,
            "mean": 0.0,
            "max": 0.0,
        }
    return {
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
    }


def _summary(
    *,
    input_rows: int,
    output_path: Path,
    replay: AggregatedReplay,
) -> dict[str, Any]:
    duplicate_groups = int(np.count_nonzero(replay.counts > 1))
    duplicate_rows = int(replay.counts[replay.counts > 1].sum())
    return {
        "event": "replay_aggregate_summary",
        "output": str(output_path),
        "input_rows": input_rows,
        "output_rows": int(replay.features.shape[0]),
        "removed_rows": int(input_rows - replay.features.shape[0]),
        "duplicate_groups": duplicate_groups,
        "duplicate_rows": duplicate_rows,
        "max_group_size": int(replay.counts.max(initial=0)),
        "capacity": replay.capacity,
        "sample_weight": _describe(replay.sample_weights),
        "value": {
            "min": float(replay.values.min(initial=0.0)),
            "mean": float(replay.values.mean()) if replay.values.size else 0.0,
            "max": float(replay.values.max(initial=0.0)),
        },
    }


def main() -> NoReturn:
    args = build_parser().parse_args()
    features, policies, values, root_policy_logits, loaded_capacity = load_replay(args.input)
    replay = aggregate_duplicate_replay(
        features=features,
        policies=policies,
        values=values,
        root_policy_logits=root_policy_logits,
        capacity=args.capacity if args.capacity is not None else loaded_capacity,
        sample_weight_mode=args.sample_weight_mode,
        sample_weight_cap=args.sample_weight_cap,
    )
    output_path = save_replay(args.output, replay)
    print(
        json.dumps(
            _summary(input_rows=features.shape[0], output_path=output_path, replay=replay),
            indent=2 if args.pretty else None,
            sort_keys=True,
        )
    )
    raise SystemExit(0)


if __name__ == "__main__":
    main()


__all__ = [
    "AggregatedReplay",
    "aggregate_duplicate_replay",
    "build_parser",
    "load_replay",
    "save_replay",
]
