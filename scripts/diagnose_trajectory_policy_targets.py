"""Diagnose trajectory replay policy targets and stored root priors."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NoReturn

import numpy as np
from great_kingdom_ai.replay import TrajectoryReplayStore
from great_kingdom_ai.self_play_data import value_target_for_player
from great_kingdom_ai.training import load_training_config

DEFAULT_REPLAY = Path("data/runpod/train-v3/replay/trajectory-replay.npz")
DEFAULT_TRAIN_CONFIG = Path("configs/runpod/train.json")


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize trajectory replay policy target sharpness and root prior drift."
    )
    parser.add_argument("--replay", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--train-config", type=Path, default=None)
    parser.add_argument("--bootstrap-td-steps", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument(
        "--value-bootstrap-source",
        choices=["terminal", "mcts_root"],
        default=None,
    )
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--top-k", type=int, default=10)
    return parser


def main() -> NoReturn:
    args = build_parser().parse_args()
    replay = TrajectoryReplayStore.load(args.replay)
    value_config = resolve_value_target_config(
        train_config_path=args.train_config,
        bootstrap_td_steps=args.bootstrap_td_steps,
        gamma=args.gamma,
        value_bootstrap_source=args.value_bootstrap_source,
    )
    payload = summarize_trajectory_policy_targets(
        replay,
        top_k=args.top_k,
        value_config=value_config,
    )
    print(json.dumps(payload, indent=2 if args.pretty else None, sort_keys=True))
    raise SystemExit(0)


def summarize_trajectory_policy_targets(
    replay: TrajectoryReplayStore,
    *,
    top_k: int = 10,
    value_config: dict[str, float | int | str] | None = None,
) -> dict[str, Any]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    target = np.asarray(replay.policy_targets, dtype=np.float32)
    legal_masks = np.asarray(replay.legal_masks, dtype=np.bool_)
    target_argmax = np.argmax(target, axis=1)
    target_summary: dict[str, Any] = {
        "entropy": asdict(describe_array(policy_entropy(target))),
        "max_probability": asdict(describe_array(target.max(axis=1))),
        "support": asdict(describe_array(np.count_nonzero(target > 1e-6, axis=1))),
        "argmax_top": top_counts(target_argmax, top_k=top_k),
        "illegal_mass": asdict(describe_array(np.where(~legal_masks, target, 0.0).sum(axis=1))),
    }
    payload: dict[str, Any] = {
        "rows": len(replay),
        "episodes": replay.episode_count,
        "capacity": replay.capacity,
        "policy_target": target_summary,
        "value_target": summarize_value_targets(replay, value_config=value_config),
        "model_versions": counts(replay.model_versions),
        "created_iterations": counts(replay.created_iterations),
    }
    if replay.root_policy_logits is None:
        payload["root_prior"] = {
            "available_rows": 0,
            "missing_reason": "trajectory replay does not include root_policy_logits",
        }
        return payload

    root_logits = np.asarray(replay.root_policy_logits, dtype=np.float32)
    available = np.isfinite(root_logits).all(axis=1)
    payload["root_prior"] = summarize_root_prior(
        target=target,
        root_logits=root_logits,
        legal_masks=legal_masks,
        available=available,
        top_k=top_k,
    )
    return payload


def resolve_value_target_config(
    *,
    train_config_path: Path | None,
    bootstrap_td_steps: int | None,
    gamma: float | None,
    value_bootstrap_source: str | None,
) -> dict[str, float | int | str]:
    if train_config_path is not None:
        config = load_training_config(train_config_path)
        resolved: dict[str, float | int | str] = {
            "bootstrap_td_steps": config.bootstrap_td_steps,
            "gamma": config.gamma,
            "value_bootstrap_source": config.value_bootstrap_source,
        }
    elif DEFAULT_TRAIN_CONFIG.is_file():
        config = load_training_config(DEFAULT_TRAIN_CONFIG)
        resolved = {
            "bootstrap_td_steps": config.bootstrap_td_steps,
            "gamma": config.gamma,
            "value_bootstrap_source": config.value_bootstrap_source,
        }
    else:
        resolved = {
            "bootstrap_td_steps": 0,
            "gamma": 1.0,
            "value_bootstrap_source": "terminal",
        }

    if bootstrap_td_steps is not None:
        resolved["bootstrap_td_steps"] = bootstrap_td_steps
    if gamma is not None:
        resolved["gamma"] = gamma
    if value_bootstrap_source is not None:
        resolved["value_bootstrap_source"] = value_bootstrap_source
    return resolved


def summarize_value_targets(
    replay: TrajectoryReplayStore,
    *,
    value_config: dict[str, float | int | str] | None,
) -> dict[str, Any]:
    resolved = value_config or {
        "bootstrap_td_steps": 0,
        "gamma": 1.0,
        "value_bootstrap_source": "terminal",
    }
    terminal = terminal_value_targets(replay)
    training = training_value_targets(
        replay,
        bootstrap_td_steps=int(resolved["bootstrap_td_steps"]),
        gamma=float(resolved["gamma"]),
        value_bootstrap_source=str(resolved["value_bootstrap_source"]),
    )
    deltas = turn_root_value_deltas(replay)
    finite_turn_roots = replay.turn_root_values[np.isfinite(replay.turn_root_values)]
    return {
        "config": dict(resolved),
        "training": value_target_distribution(training),
        "terminal": value_target_distribution(terminal),
        "training_minus_terminal": asdict(describe_array(training - terminal)),
        "turn_root_values": value_target_distribution(
            finite_turn_roots.astype(np.float32, copy=False)
        ),
        "turn_root_value_abs_delta": asdict(describe_array(np.abs(deltas))),
        "turn_root_value_delta": asdict(describe_array(deltas)),
    }


def turn_root_value_deltas(replay: TrajectoryReplayStore) -> np.ndarray:
    rows: list[np.ndarray] = []
    root_values = replay.turn_root_values.astype(np.float32, copy=False)
    for episode_index in range(replay.episode_count):
        start = int(replay.turn_offsets[episode_index])
        end = int(replay.turn_offsets[episode_index + 1])
        if end - start > 1:
            rows.append(np.diff(root_values[start:end]))
    if not rows:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate(rows).astype(np.float32, copy=False)


def value_target_distribution(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float32)
    abs_values = np.abs(values)
    return {
        "value": asdict(describe_array(values)),
        "abs_value": asdict(describe_array(abs_values)),
        "negative_fraction": float(np.mean(values < -0.5)) if values.size else math.nan,
        "drawish_fraction": float(np.mean(abs_values <= 0.5)) if values.size else math.nan,
        "positive_fraction": float(np.mean(values > 0.5)) if values.size else math.nan,
        "near_terminal_fraction_abs_ge_0_9": (
            float(np.mean(abs_values >= 0.9)) if values.size else math.nan
        ),
        "near_zero_fraction_abs_le_0_1": (
            float(np.mean(abs_values <= 0.1)) if values.size else math.nan
        ),
    }


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


def training_value_targets(
    replay: TrajectoryReplayStore,
    *,
    bootstrap_td_steps: int,
    gamma: float,
    value_bootstrap_source: str,
) -> np.ndarray:
    if value_bootstrap_source not in {"terminal", "mcts_root"}:
        raise ValueError("value_bootstrap_source must be one of: terminal, mcts_root")
    if bootstrap_td_steps < 0:
        raise ValueError("bootstrap_td_steps must be non-negative")
    if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be finite and in [0, 1]")
    if value_bootstrap_source == "terminal" or bootstrap_td_steps == 0:
        return terminal_value_targets(replay)
    return mcts_root_bootstrap_targets(replay, td_steps=bootstrap_td_steps, gamma=gamma)


def mcts_root_bootstrap_targets(
    replay: TrajectoryReplayStore,
    *,
    td_steps: int,
    gamma: float,
) -> np.ndarray:
    values = np.empty((len(replay),), dtype=np.float32)
    episode_indexes = np.searchsorted(
        replay.episode_offsets,
        np.arange(len(replay), dtype=np.int64),
        side="right",
    ) - 1
    for row in range(len(replay)):
        episode_index = int(episode_indexes[row])
        turn_start = int(replay.turn_offsets[episode_index])
        turn_end = int(replay.turn_offsets[episode_index + 1])
        local_turn = int(replay.timesteps[row])
        target_local_turn = local_turn + td_steps
        if target_local_turn >= turn_end - turn_start:
            values[row] = np.float32(
                value_target_for_player(
                    player=int(replay.players[row]),
                    winner=int(replay.episode_winners[episode_index]),
                )
            )
            continue
        target_row = turn_start + target_local_turn
        root_value = float(replay.turn_root_values[target_row])
        if not math.isfinite(root_value):
            raise ValueError("mcts_root bootstrap requires finite turn root values")
        if int(replay.turn_players[target_row]) != int(replay.players[row]):
            root_value = -root_value
        values[row] = np.float32((gamma**td_steps) * root_value)
    return values


def summarize_root_prior(
    *,
    target: np.ndarray,
    root_logits: np.ndarray,
    legal_masks: np.ndarray,
    available: np.ndarray,
    top_k: int,
) -> dict[str, Any]:
    missing_rows = int(root_logits.shape[0] - np.count_nonzero(available))
    if not np.any(available):
        return {"available_rows": 0, "missing_rows": missing_rows}
    prior = masked_softmax(root_logits[available], legal_masks[available])
    target_available = target[available]
    target_argmax = np.argmax(target_available, axis=1)
    prior_argmax = np.argmax(prior, axis=1)
    argmax_mismatch = target_argmax != prior_argmax
    target_max = target_available.max(axis=1)
    prior_max = prior.max(axis=1)
    return {
        "available_rows": int(np.count_nonzero(available)),
        "missing_rows": missing_rows,
        "entropy": asdict(describe_array(policy_entropy(prior))),
        "max_probability": asdict(describe_array(prior_max)),
        "argmax_top": top_counts(prior_argmax, top_k=top_k),
        "argmax_mismatch_ratio": float(np.mean(argmax_mismatch)),
        "argmax_mismatch_count": int(np.count_nonzero(argmax_mismatch)),
        "kl_target_prior": asdict(describe_array(categorical_kl(target_available, prior))),
        "kl_prior_target": asdict(describe_array(categorical_kl(prior, target_available))),
        "top1_probability_delta_target_minus_prior": asdict(
            describe_array(target_max - prior_max)
        ),
    }


def masked_softmax(logits: np.ndarray, legal_masks: np.ndarray) -> np.ndarray:
    masked = np.where(legal_masks, logits, -np.inf).astype(np.float32, copy=False)
    row_max = np.max(masked, axis=1, keepdims=True)
    if not np.isfinite(row_max).all():
        raise ValueError("each row must have at least one legal root prior logit")
    exp = np.where(legal_masks, np.exp(masked - row_max), 0.0)
    return (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32, copy=False)


def categorical_kl(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_clipped = np.clip(left, 1e-45, 1.0).astype(np.float32, copy=False)
    right_clipped = np.clip(right, 1e-45, 1.0).astype(np.float32, copy=False)
    return np.maximum(
        np.sum(left_clipped * (np.log(left_clipped) - np.log(right_clipped)), axis=1),
        0.0,
    ).astype(np.float32, copy=False)


def policy_entropy(policy: np.ndarray) -> np.ndarray:
    positive = policy > 0.0
    terms = np.zeros_like(policy, dtype=np.float32)
    terms[positive] = policy[positive] * np.log(np.clip(policy[positive], 1e-45, 1.0))
    return (-terms.sum(axis=1)).astype(np.float32, copy=False)


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


def top_counts(values: np.ndarray, *, top_k: int) -> dict[str, int]:
    counter = Counter(int(value) for value in values.tolist())
    return {str(key): count for key, count in counter.most_common(top_k)}


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


if __name__ == "__main__":
    main()
