"""Aggregate duplicate replay states by averaging their policy and value targets."""

from __future__ import annotations

import argparse
import hashlib
import json
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
    capacity: int


def aggregate_duplicate_replay(
    *,
    features: FloatArray,
    policies: FloatArray,
    values: FloatArray,
    capacity: int | None = None,
) -> AggregatedReplay:
    """Merge exact duplicate feature rows and average their training targets."""
    _validate_arrays(features, policies, values)
    groups: dict[str, int] = {}
    first_indexes: list[int] = []
    policy_sums: list[FloatArray] = []
    value_sums: list[float] = []
    counts: list[int] = []

    for index in range(features.shape[0]):
        digest = hashlib.blake2b(features[index].tobytes(), digest_size=16).hexdigest()
        group_index = groups.get(digest)
        if group_index is None:
            groups[digest] = len(first_indexes)
            first_indexes.append(index)
            policy_sums.append(policies[index].astype(np.float32, copy=True))
            value_sums.append(float(values[index]))
            counts.append(1)
        else:
            policy_sums[group_index] += policies[index]
            value_sums[group_index] += float(values[index])
            counts[group_index] += 1

    count_array = np.asarray(counts, dtype=np.int64)
    aggregated_policies = np.stack(policy_sums, axis=0).astype(np.float32)
    aggregated_policies /= count_array[:, None].astype(np.float32)
    aggregated_policies = _renormalized_policies(aggregated_policies)
    aggregated_values = (
        np.asarray(value_sums, dtype=np.float32) / count_array.astype(np.float32)
    ).astype(np.float32)
    output_capacity = capacity if capacity is not None else max(1, len(first_indexes))
    output_capacity = max(output_capacity, len(first_indexes))
    return AggregatedReplay(
        features=features[np.asarray(first_indexes, dtype=np.int64)].astype(np.float32, copy=True),
        policies=aggregated_policies,
        values=aggregated_values,
        counts=count_array,
        capacity=output_capacity,
    )


def load_replay(path: str | Path) -> tuple[FloatArray, FloatArray, FloatArray, int | None]:
    with np.load(Path(path)) as data:
        features = np.asarray(data["features"], dtype=np.float32)
        policies = np.asarray(data["policies"], dtype=np.float32)
        values = np.asarray(data["values"], dtype=np.float32)
        capacity = int(data["capacity"]) if "capacity" in data else None
    return features, policies, values, capacity


def save_replay(path: str | Path, replay: AggregatedReplay) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        capacity=np.asarray(replay.capacity, dtype=np.int64),
        features=replay.features,
        policies=replay.policies,
        values=replay.values,
    )
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
    return parser


def _validate_arrays(features: FloatArray, policies: FloatArray, values: FloatArray) -> None:
    if features.ndim != 4 or tuple(features.shape[1:]) != FEATURE_SHAPE:
        raise ValueError(f"expected features shape [N, {FEATURE_SHAPE}], got {features.shape}")
    if policies.shape != (features.shape[0], ACTION_SPACE):
        raise ValueError(
            f"expected policies shape {(features.shape[0], ACTION_SPACE)}, got {policies.shape}"
        )
    if values.shape != (features.shape[0],):
        raise ValueError(f"expected values shape {(features.shape[0],)}, got {values.shape}")


def _renormalized_policies(policies: FloatArray) -> FloatArray:
    sums = policies.sum(axis=1, keepdims=True)
    if np.any(sums <= 0.0):
        raise ValueError("policy rows must have positive mass")
    return cast(FloatArray, (policies / sums).astype(np.float32))


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
        "value": {
            "min": float(replay.values.min(initial=0.0)),
            "mean": float(replay.values.mean()) if replay.values.size else 0.0,
            "max": float(replay.values.max(initial=0.0)),
        },
    }


def main() -> NoReturn:
    args = build_parser().parse_args()
    features, policies, values, loaded_capacity = load_replay(args.input)
    replay = aggregate_duplicate_replay(
        features=features,
        policies=policies,
        values=values,
        capacity=args.capacity if args.capacity is not None else loaded_capacity,
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
