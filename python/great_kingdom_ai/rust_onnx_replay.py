"""Replay import/export helpers for the Rust ONNX self-play pipeline."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

import numpy as np
from safetensors.numpy import load_file, save_file

from great_kingdom_ai.features import ACTION_SPACE, BOARD_SIZE, FEATURE_CHANNELS
from great_kingdom_ai.online_aggregate_replay import OnlineAggregateReplayBuffer
from great_kingdom_ai.replay_buffer import ReplayBuffer, ReplaySample
from great_kingdom_ai.self_play import GameLog

FEATURE_SHAPE = (FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE)


@dataclass(frozen=True)
class RustReplayImportSummary:
    artifact_dir: Path
    replay_path: Path
    imported_samples: int
    replay_samples: int
    imported_games: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_dir": str(self.artifact_dir),
            "replay_path": str(self.replay_path),
            "imported_samples": self.imported_samples,
            "replay_samples": self.replay_samples,
            "imported_games": self.imported_games,
        }


@dataclass(frozen=True)
class LegacyImportSummary:
    legacy_work_dir: Path
    onnx_work_dir: Path
    imported: bool
    replay_samples: int
    game_logs: int
    next_seed_start: int
    best_checkpoint: Path | None
    candidate_checkpoint: Path | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "legacy_work_dir": str(self.legacy_work_dir),
            "onnx_work_dir": str(self.onnx_work_dir),
            "imported": self.imported,
            "replay_samples": self.replay_samples,
            "game_logs": self.game_logs,
            "next_seed_start": self.next_seed_start,
            "best_checkpoint": str(self.best_checkpoint) if self.best_checkpoint else None,
            "candidate_checkpoint": (
                str(self.candidate_checkpoint) if self.candidate_checkpoint else None
            ),
        }


def write_rust_self_play_artifacts(
    *,
    output_dir: str | Path,
    samples: list[ReplaySample],
    logs: list[GameLog],
    manifest: dict[str, Any],
) -> Path:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    tensors = _samples_to_tensors(samples)
    save_file(tensors, destination / "samples.safetensors")
    with (destination / "games.jsonl").open("w", encoding="utf-8") as file:
        for log in logs:
            file.write(json.dumps(log.to_dict(), sort_keys=True))
            file.write("\n")
    _write_json(destination / "manifest.json", manifest)
    return destination


def import_rust_self_play_artifacts(
    *,
    artifact_dir: str | Path,
    replay_path: str | Path,
    replay_capacity: int,
    game_log_path: str | Path | None = None,
    aggregate_replay_path: str | Path | None = None,
    aggregate_replay_weight_mode: str = "sqrt_count",
    aggregate_replay_weight_cap: float | None = 16.0,
    materialize_raw_replay: bool = True,
) -> RustReplayImportSummary:
    if not materialize_raw_replay and aggregate_replay_path is None:
        raise ValueError("materialize_raw_replay=False requires aggregate_replay_path")

    source = Path(artifact_dir)
    tensors = load_file(source / "samples.safetensors")
    samples = _samples_from_tensors(tensors)
    replay_file = Path(replay_path)
    replay_samples = 0
    if materialize_raw_replay:
        replay = (
            ReplayBuffer.load(replay_file)
            if replay_file.exists()
            else ReplayBuffer(replay_capacity)
        )
        replay.extend(samples)
        replay.save(replay_file, compressed=False)
        replay_samples = len(replay)
    if aggregate_replay_path is not None:
        aggregate_samples = _extend_online_aggregate_replay(
            aggregate_replay_path=Path(aggregate_replay_path),
            raw_replay_path=replay_file,
            replay_capacity=replay_capacity,
            samples=samples,
            sample_weight_mode=aggregate_replay_weight_mode,
            sample_weight_cap=aggregate_replay_weight_cap,
            raw_replay_includes_samples=materialize_raw_replay,
        )
        if not materialize_raw_replay:
            replay_samples = aggregate_samples

    game_dicts = _read_jsonl_dicts(source / "games.jsonl")
    if game_log_path is not None:
        log_path = Path(game_log_path)
        existing = _read_json_list(log_path) if log_path.exists() else []
        existing.extend(game_dicts)
        _write_json(log_path, existing)

    return RustReplayImportSummary(
        artifact_dir=source,
        replay_path=replay_file,
        imported_samples=len(samples),
        replay_samples=replay_samples,
        imported_games=len(game_dicts),
    )


def import_legacy_pipeline_data(
    *,
    legacy_work_dir: str | Path,
    onnx_work_dir: str | Path,
    replay_capacity: int,
    force: bool = False,
) -> LegacyImportSummary:
    legacy = Path(legacy_work_dir)
    destination = Path(onnx_work_dir)
    marker = destination / "reports" / "legacy-import.json"
    if marker.exists() and not force:
        data = json.loads(marker.read_text(encoding="utf-8"))
        return _legacy_summary_from_dict(data, imported=False)

    (destination / "replay").mkdir(parents=True, exist_ok=True)
    (destination / "checkpoints").mkdir(parents=True, exist_ok=True)
    (destination / "reports").mkdir(parents=True, exist_ok=True)

    replay_samples = 0
    legacy_replay = legacy / "replay" / "replay.npz"
    if legacy_replay.exists():
        replay = _load_replay_with_capacity(legacy_replay, replay_capacity)
        replay_samples = len(replay)
        replay.save(destination / "replay" / "replay.npz")

    game_logs: list[dict[str, Any]] = []
    legacy_logs = legacy / "replay" / "game_logs.json"
    if legacy_logs.exists():
        game_logs = _read_json_list(legacy_logs)
        _write_json(destination / "replay" / "game_logs.json", game_logs)

    best_checkpoint = _copy_if_exists(
        legacy / "checkpoints" / "best.pt",
        destination / "checkpoints" / "best.pt",
    )
    candidate_checkpoint = _copy_if_exists(
        legacy / "checkpoints" / "candidate.pt",
        destination / "checkpoints" / "candidate.pt",
    )
    summary = LegacyImportSummary(
        legacy_work_dir=legacy,
        onnx_work_dir=destination,
        imported=True,
        replay_samples=replay_samples,
        game_logs=len(game_logs),
        next_seed_start=_next_seed_start(game_logs),
        best_checkpoint=best_checkpoint,
        candidate_checkpoint=candidate_checkpoint,
    )
    _write_json(marker, summary.to_dict())
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="great-kingdom-import-legacy-pipeline",
        description="Import legacy Python pipeline artifacts into a Rust ONNX pipeline work dir.",
    )
    parser.add_argument("--legacy-work-dir", type=Path, required=True)
    parser.add_argument("--onnx-work-dir", type=Path, required=True)
    parser.add_argument("--replay-capacity", type=int, default=500_000)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> NoReturn:
    args = build_parser().parse_args()
    summary = import_legacy_pipeline_data(
        legacy_work_dir=args.legacy_work_dir,
        onnx_work_dir=args.onnx_work_dir,
        replay_capacity=args.replay_capacity,
        force=args.force,
    )
    print(json.dumps(summary.to_dict(), sort_keys=True))
    raise SystemExit(0)


def _samples_to_tensors(samples: list[ReplaySample]) -> dict[str, np.ndarray]:
    if samples:
        features = np.stack([sample.features for sample in samples]).astype(np.float32)
        policy = np.stack([sample.policy for sample in samples]).astype(np.float32)
        value = np.asarray([sample.value for sample in samples], dtype=np.float32)
    else:
        features = np.empty((0, *FEATURE_SHAPE), dtype=np.float32)
        policy = np.empty((0, ACTION_SPACE), dtype=np.float32)
        value = np.empty((0,), dtype=np.float32)
    tensors = {"features": features, "policy": policy, "value": value}
    root_policy_logits = _root_policy_logits_array(samples)
    if root_policy_logits is not None:
        tensors["root_policy_logits"] = root_policy_logits
    return tensors


def _samples_from_tensors(tensors: dict[str, np.ndarray]) -> list[ReplaySample]:
    features = np.asarray(tensors["features"], dtype=np.float32)
    policy = np.asarray(tensors["policy"], dtype=np.float32)
    value = np.asarray(tensors["value"], dtype=np.float32)
    root_policy_logits = (
        np.asarray(tensors["root_policy_logits"], dtype=np.float32)
        if "root_policy_logits" in tensors
        else None
    )
    if features.shape[1:] != FEATURE_SHAPE:
        raise ValueError(f"expected features shape [N, {FEATURE_SHAPE}], got {features.shape}")
    if policy.shape != (features.shape[0], ACTION_SPACE):
        expected_policy_shape = (features.shape[0], ACTION_SPACE)
        raise ValueError(f"expected policy shape {expected_policy_shape}, got {policy.shape}")
    if value.shape != (features.shape[0],):
        raise ValueError(f"expected value shape {(features.shape[0],)}, got {value.shape}")
    if root_policy_logits is not None and root_policy_logits.shape != policy.shape:
        raise ValueError(
            "expected root_policy_logits shape to match policy shape, "
            f"got {root_policy_logits.shape}"
        )
    return [
        ReplaySample(
            features=features[index],
            policy=policy[index],
            value=float(value[index]),
            root_policy_logits=(
                root_policy_logits[index]
                if root_policy_logits is not None
                and np.isfinite(root_policy_logits[index]).all()
                else None
            ),
        )
        for index in range(features.shape[0])
    ]


def _load_replay_with_capacity(path: Path, capacity: int) -> ReplayBuffer:
    source = ReplayBuffer.load(path)
    with np.load(path) as data:
        features = np.asarray(data["features"], dtype=np.float32)
        policies = np.asarray(data["policies"], dtype=np.float32)
        values = np.asarray(data["values"], dtype=np.float32)
    replay = ReplayBuffer(capacity)
    start = max(0, features.shape[0] - capacity)
    for index in range(start, features.shape[0]):
        replay.push(
            ReplaySample(
                features=features[index],
                policy=policies[index],
                value=float(values[index]),
            )
        )
    if len(replay) != min(len(source), capacity):
        raise ValueError("legacy replay import produced an inconsistent sample count")
    return replay


def _extend_online_aggregate_replay(
    *,
    aggregate_replay_path: Path,
    raw_replay_path: Path,
    replay_capacity: int,
    samples: list[ReplaySample],
    sample_weight_mode: str,
    sample_weight_cap: float | None,
    raw_replay_includes_samples: bool,
) -> int:
    if aggregate_replay_path.exists():
        replay = OnlineAggregateReplayBuffer.load(
            aggregate_replay_path,
            capacity=replay_capacity,
            sample_weight_mode=sample_weight_mode,
            sample_weight_cap=sample_weight_cap,
        )
        replay.extend(samples)
    else:
        replay = OnlineAggregateReplayBuffer(
            replay_capacity,
            sample_weight_mode=sample_weight_mode,
            sample_weight_cap=sample_weight_cap,
        )
        if raw_replay_path.exists():
            _extend_online_aggregate_from_file(
                replay,
                raw_replay_path,
            )
        if not raw_replay_includes_samples:
            replay.extend(samples)
    replay.save(aggregate_replay_path, compressed=False)
    return len(replay)


def _extend_online_aggregate_from_file(
    replay: OnlineAggregateReplayBuffer,
    raw_replay_path: Path,
) -> None:
    with np.load(raw_replay_path) as data:
        features = np.asarray(data["features"], dtype=np.float32)
        policies = np.asarray(data["policies"], dtype=np.float32)
        values = np.asarray(data["values"], dtype=np.float32)
        root_policy_logits = (
            np.asarray(data["root_policy_logits"], dtype=np.float32)
            if "root_policy_logits" in data
            else None
        )
        sample_weights = (
            np.asarray(data["sample_weights"], dtype=np.float32)
            if "sample_weights" in data
            else np.ones(values.shape, dtype=np.float32)
        )
    for index in range(features.shape[0]):
        replay.push(
            ReplaySample(
                features=features[index],
                policy=policies[index],
                value=float(values[index]),
                root_policy_logits=(
                    root_policy_logits[index]
                    if root_policy_logits is not None
                    and np.isfinite(root_policy_logits[index]).all()
                    else None
                ),
                sample_weight=float(sample_weights[index]),
            )
        )


def _read_json_list(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list")
    return [dict(item) for item in data if isinstance(item, dict)]


def _read_jsonl_dicts(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if stripped:
                data = json.loads(stripped)
                if isinstance(data, dict):
                    rows.append(dict(data))
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _root_policy_logits_array(samples: list[ReplaySample]) -> np.ndarray | None:
    if not any(sample.root_policy_logits is not None for sample in samples):
        return None
    rows = np.full((len(samples), ACTION_SPACE), np.nan, dtype=np.float32)
    for index, sample in enumerate(samples):
        if sample.root_policy_logits is None:
            continue
        rows[index] = np.asarray(sample.root_policy_logits, dtype=np.float32)
    return rows


def _copy_if_exists(source: Path, destination: Path) -> Path | None:
    if not source.exists():
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def _next_seed_start(logs: list[dict[str, Any]]) -> int:
    seeds = [int(log["seed"]) for log in logs if isinstance(log.get("seed"), int)]
    return max(seeds, default=-1) + 1


def _legacy_summary_from_dict(data: dict[str, Any], *, imported: bool) -> LegacyImportSummary:
    return LegacyImportSummary(
        legacy_work_dir=Path(str(data["legacy_work_dir"])),
        onnx_work_dir=Path(str(data["onnx_work_dir"])),
        imported=imported,
        replay_samples=int(data["replay_samples"]),
        game_logs=int(data["game_logs"]),
        next_seed_start=int(data["next_seed_start"]),
        best_checkpoint=(
            Path(str(data["best_checkpoint"])) if data.get("best_checkpoint") else None
        ),
        candidate_checkpoint=(
            Path(str(data["candidate_checkpoint"]))
            if data.get("candidate_checkpoint")
            else None
        ),
    )


__all__ = [
    "LegacyImportSummary",
    "RustReplayImportSummary",
    "import_legacy_pipeline_data",
    "import_rust_self_play_artifacts",
    "write_rust_self_play_artifacts",
]
