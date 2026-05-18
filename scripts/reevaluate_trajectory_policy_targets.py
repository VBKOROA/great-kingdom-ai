"""Re-evaluate sampled trajectory replay policy targets with a checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

import numpy as np
from great_kingdom_ai.replay import TrajectoryReplayStore

if TYPE_CHECKING:
    from great_kingdom_ai.model import PolicyValueNetwork

DEFAULT_REPLAY = Path("data/runpod/train-v3/replay/trajectory-replay.npz")
DEFAULT_CHECKPOINT = Path("data/runpod/train-v3/checkpoints/training-latest.pt")


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
class LoadedModel:
    model: PolicyValueNetwork
    weights_used: str
    ema_available: bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sample trajectory replay rows, forward them through a checkpoint, "
            "and compare checkpoint policy priors against stored MCTS targets."
        )
    )
    parser.add_argument("--replay", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--eval-rows", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--prefer-ema", action="store_true")
    parser.add_argument(
        "--compare-ema",
        action="store_true",
        help="evaluate both raw and EMA weights when EMA weights are present",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--pretty", action="store_true")
    return parser


def main() -> NoReturn:
    args = build_parser().parse_args()
    payload = reevaluate_checkpoint_policy_targets(
        replay_path=args.replay,
        checkpoint_path=args.checkpoint,
        eval_rows=args.eval_rows,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        prefer_ema=args.prefer_ema,
        compare_ema=args.compare_ema,
        top_k=args.top_k,
    )
    print(json.dumps(payload, indent=2 if args.pretty else None, sort_keys=True))
    raise SystemExit(0)


def reevaluate_checkpoint_policy_targets(
    *,
    replay_path: Path,
    checkpoint_path: Path,
    eval_rows: int,
    batch_size: int,
    seed: int,
    device: str,
    prefer_ema: bool,
    compare_ema: bool,
    top_k: int,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    replay = TrajectoryReplayStore.load(replay_path)
    indexes = sample_indexes(len(replay), eval_rows, random.Random(seed))
    target = np.asarray(replay.policy_targets[indexes], dtype=np.float32)
    legal_masks = np.asarray(replay.legal_masks[indexes], dtype=np.bool_)

    resolved_device = resolve_device(device)
    target_summary = summarize_policy_distribution(
        policy=target,
        legal_masks=legal_masks,
        top_k=top_k,
    )
    payload: dict[str, Any] = {
        "replay": str(replay_path),
        "checkpoint": str(checkpoint_path),
        "rows": len(replay),
        "episodes": replay.episode_count,
        "capacity": replay.capacity,
        "sample": {
            "requested_rows": int(eval_rows),
            "rows": int(indexes.shape[0]),
            "seed": int(seed),
            "min_index": int(indexes.min()) if indexes.size else None,
            "max_index": int(indexes.max()) if indexes.size else None,
        },
        "device": resolved_device,
        "policy_target": target_summary,
    }
    stored_root_summary = summarize_stored_root_prior_sample(
        replay=replay,
        indexes=indexes,
        target=target,
        legal_masks=legal_masks,
        top_k=top_k,
    )
    if stored_root_summary is not None:
        payload["stored_root_prior"] = stored_root_summary

    modes = [False, True] if compare_ema else [prefer_ema]
    payload["checkpoint_policy"] = [
        evaluate_checkpoint_policy(
            checkpoint_path=checkpoint_path,
            replay=replay,
            indexes=indexes,
            target=target,
            legal_masks=legal_masks,
            batch_size=batch_size,
            device=resolved_device,
            prefer_ema=mode,
            top_k=top_k,
        )
        for mode in modes
    ]
    return payload


def evaluate_checkpoint_policy(
    *,
    checkpoint_path: Path,
    replay: TrajectoryReplayStore,
    indexes: np.ndarray,
    target: np.ndarray,
    legal_masks: np.ndarray,
    batch_size: int,
    device: str,
    prefer_ema: bool,
    top_k: int,
) -> dict[str, Any]:
    import torch

    started = time.perf_counter()
    loaded = load_model_for_inference(checkpoint_path, device=device, prefer_ema=prefer_ema)
    model = loaded.model
    model.eval()
    policies: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, indexes.shape[0], batch_size):
            batch_indexes = indexes[start : start + batch_size]
            features = torch.from_numpy(
                np.ascontiguousarray(replay.features[batch_indexes], dtype=np.float32)
            ).to(device=device)
            legal = torch.from_numpy(
                np.ascontiguousarray(replay.legal_masks[batch_indexes], dtype=np.bool_)
            ).to(device=device)
            logits, _value = model(features)
            masked_logits = logits.masked_fill(~legal, torch.finfo(logits.dtype).min)
            policy = torch.softmax(masked_logits, dim=1)
            policies.append(policy.detach().cpu().numpy().astype(np.float32, copy=False))
    checkpoint_policy = (
        np.concatenate(policies, axis=0)
        if policies
        else np.empty_like(target, dtype=np.float32)
    )
    summary = compare_policy_to_target(
        target=target,
        prior=checkpoint_policy,
        legal_masks=legal_masks,
        top_k=top_k,
    )
    summary.update(
        {
            "checkpoint": str(checkpoint_path),
            "prefer_ema": bool(prefer_ema),
            "ema_available": loaded.ema_available,
            "weights_used": loaded.weights_used,
            "rows": int(indexes.shape[0]),
            "batch_size": int(batch_size),
            "device": device,
            "elapsed_seconds": float(time.perf_counter() - started),
        }
    )
    return summary


def load_model_for_inference(
    checkpoint_path: Path,
    *,
    device: str,
    prefer_ema: bool,
) -> LoadedModel:
    import torch
    from great_kingdom_ai.model import ModelConfig, PolicyValueNetwork

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = PolicyValueNetwork(ModelConfig(**checkpoint["model_config"])).to(device=device)
    model_state = checkpoint["model_state"]
    ema_state = checkpoint.get("ema_model_state")
    ema_available = ema_state is not None
    weights_used = "raw"
    if prefer_ema and ema_state is not None:
        model_state = ema_state
        weights_used = "ema"
    model.load_state_dict(model_state)
    return LoadedModel(model=model, weights_used=weights_used, ema_available=ema_available)


def summarize_stored_root_prior_sample(
    *,
    replay: TrajectoryReplayStore,
    indexes: np.ndarray,
    target: np.ndarray,
    legal_masks: np.ndarray,
    top_k: int,
) -> dict[str, Any] | None:
    if replay.root_policy_logits is None:
        return None
    root_logits = np.asarray(replay.root_policy_logits[indexes], dtype=np.float32)
    if replay.root_policy_logits_present is None:
        available = np.isfinite(root_logits).all(axis=1)
    else:
        available = np.asarray(replay.root_policy_logits_present[indexes], dtype=np.bool_)
        available &= np.isfinite(root_logits).all(axis=1)
    if not np.any(available):
        return {
            "available_rows": 0,
            "missing_rows": int(indexes.shape[0]),
        }
    prior = masked_softmax(root_logits[available], legal_masks[available])
    summary = compare_policy_to_target(
        target=target[available],
        prior=prior,
        legal_masks=legal_masks[available],
        top_k=top_k,
    )
    summary["available_rows"] = int(np.count_nonzero(available))
    summary["missing_rows"] = int(indexes.shape[0] - np.count_nonzero(available))
    return summary


def compare_policy_to_target(
    *,
    target: np.ndarray,
    prior: np.ndarray,
    legal_masks: np.ndarray,
    top_k: int,
) -> dict[str, Any]:
    target_argmax = np.argmax(target, axis=1)
    prior_argmax = np.argmax(prior, axis=1)
    argmax_mismatch = target_argmax != prior_argmax
    return {
        "entropy": asdict(describe_array(policy_entropy(prior))),
        "max_probability": asdict(describe_array(prior.max(axis=1))),
        "argmax_top": top_counts(prior_argmax, top_k=top_k),
        "argmax_mismatch_ratio": float(np.mean(argmax_mismatch)) if target.size else math.nan,
        "argmax_mismatch_count": int(np.count_nonzero(argmax_mismatch)),
        "kl_target_prior": asdict(describe_array(categorical_kl(target, prior))),
        "kl_prior_target": asdict(describe_array(categorical_kl(prior, target))),
        "policy_cross_entropy": asdict(describe_array(categorical_cross_entropy(target, prior))),
        "top1_probability_delta_target_minus_prior": asdict(
            describe_array(target.max(axis=1) - prior.max(axis=1))
        ),
        "illegal_mass": asdict(describe_array(np.where(~legal_masks, prior, 0.0).sum(axis=1))),
    }


def summarize_policy_distribution(
    *,
    policy: np.ndarray,
    legal_masks: np.ndarray,
    top_k: int,
) -> dict[str, Any]:
    return {
        "entropy": asdict(describe_array(policy_entropy(policy))),
        "max_probability": asdict(describe_array(policy.max(axis=1))),
        "support": asdict(describe_array(np.count_nonzero(policy > 1e-6, axis=1))),
        "argmax_top": top_counts(np.argmax(policy, axis=1), top_k=top_k),
        "illegal_mass": asdict(describe_array(np.where(~legal_masks, policy, 0.0).sum(axis=1))),
    }


def masked_softmax(logits: np.ndarray, legal_masks: np.ndarray) -> np.ndarray:
    masked = np.where(legal_masks, logits, -np.inf).astype(np.float32, copy=False)
    row_max = np.max(masked, axis=1, keepdims=True)
    if not np.isfinite(row_max).all():
        raise ValueError("each row must have at least one legal logit")
    exp = np.where(legal_masks, np.exp(masked - row_max), 0.0)
    return (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32, copy=False)


def categorical_kl(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_clipped = np.clip(left, 1e-45, 1.0).astype(np.float32, copy=False)
    right_clipped = np.clip(right, 1e-45, 1.0).astype(np.float32, copy=False)
    return np.maximum(
        np.sum(left_clipped * (np.log(left_clipped) - np.log(right_clipped)), axis=1),
        0.0,
    ).astype(np.float32, copy=False)


def categorical_cross_entropy(target: np.ndarray, prior: np.ndarray) -> np.ndarray:
    target_clipped = np.clip(target, 0.0, 1.0).astype(np.float32, copy=False)
    prior_clipped = np.clip(prior, 1e-45, 1.0).astype(np.float32, copy=False)
    return (-np.sum(target_clipped * np.log(prior_clipped), axis=1)).astype(
        np.float32,
        copy=False,
    )


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


def sample_indexes(size: int, count: int, rng: random.Random) -> np.ndarray:
    if size <= 0:
        return np.empty((0,), dtype=np.int64)
    resolved = min(size, max(1, count))
    return np.asarray(rng.sample(range(size), resolved), dtype=np.int64)


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def top_counts(values: np.ndarray, *, top_k: int) -> dict[str, int]:
    counter = Counter(int(value) for value in values.tolist())
    return {str(key): count for key, count in counter.most_common(top_k)}


if __name__ == "__main__":
    main()
