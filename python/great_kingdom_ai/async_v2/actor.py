"""Async v2 actor cycle runner."""

from __future__ import annotations

import fcntl
import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from great_kingdom_ai.async_v2.config import ActorV2Config, ActorV2Summary, _load_json_object
from great_kingdom_ai.async_v2.metadata import (
    V2ShardRecord,
    _append_event,
    _utc_now,
    load_v2_shard_records,
)
from great_kingdom_ai.async_v2.paths import _paths
from great_kingdom_ai.pipeline_printer import PipelinePrinter
from great_kingdom_ai.replay import TrajectoryEpisode, TrajectoryReplayStore
from great_kingdom_ai.rust_onnx_self_play import (
    RustOnnxSelfPlayConfig,
    RustSelfPlayRunSummary,
    run_rust_onnx_self_play,
)
from great_kingdom_ai.self_play import GameLog


def run_actor_v2_once(
    config: ActorV2Config,
    *,
    runner: Callable[[RustOnnxSelfPlayConfig], RustSelfPlayRunSummary] | None = None,
    printer: PipelinePrinter | None = None,
) -> ActorV2Summary:
    _validate_actor_config(config)
    printer = printer if printer is not None else PipelinePrinter()
    run_self_play = runner if runner is not None else run_rust_onnx_self_play
    shard_id = config.shard_id or _default_shard_id(
        model_version=config.model_version,
        seed_start=config.seed_start,
        games=config.games,
    )
    paths = _paths(config.work_dir)
    shard_dir = paths["shard_root"] / shard_id
    replay_path = shard_dir / "trajectory-replay.npz"
    log_path = shard_dir / "game_logs.json"
    if replay_path.exists() or log_path.exists():
        raise FileExistsError(f"shard already exists: {shard_dir}")

    printer.title("Actor V2")
    printer.metric("work dir", config.work_dir)
    printer.metric("onnx model", config.onnx_model_path)
    printer.metric("model version", config.model_version)
    if config.model_iteration is not None:
        printer.metric("model iteration", config.model_iteration)
    printer.metric("games", config.games)
    printer.metric("seed start", config.seed_start)
    printer.metric("onnx device", config.onnx_device)
    printer.metric("onnx max batch", config.onnx_max_batch_size)
    printer.metric("self-play batch", config.rust_self_play_batch_size)
    printer.metric("gumbel sims", config.self_play.gumbel_simulations)
    printer.metric("leaf batch", config.self_play.leaf_batch_size)
    printer.step(f"generating trajectory shard {shard_id}")
    summary = run_self_play(
        RustOnnxSelfPlayConfig(
            onnx_model_path=config.onnx_model_path,
            output_dir=shard_dir,
            games=config.games,
            seed_start=config.seed_start,
            model_version=_transition_model_version(config),
            created_iteration=_transition_created_iteration(config),
            onnx_device=config.onnx_device,
            onnx_max_batch_size=config.onnx_max_batch_size,
            rust_self_play_batch_size=config.rust_self_play_batch_size,
            self_play=config.self_play,
        )
    )
    if not summary.trajectory_episodes:
        raise RuntimeError("actor runner did not return trajectory episodes")
    transitions = sum(len(episode.transitions) for episode in summary.trajectory_episodes)
    if len(summary.game_logs) != summary.games:
        raise RuntimeError("actor runner returned inconsistent game log count")

    _save_trajectory_shard(
        shard_dir,
        episodes=_with_transition_model_metadata(summary.trajectory_episodes, config),
        logs=summary.game_logs,
    )
    record = V2ShardRecord(
        shard_id=shard_id,
        status="completed",
        shard_dir=shard_dir,
        replay_path=replay_path,
        log_path=log_path,
        model_version=config.model_version,
        model_path=config.onnx_model_path,
        seed_start=config.seed_start,
        games=summary.games,
        transitions=transitions,
        created_at=_utc_now(),
    )
    _append_event(paths["metadata_path"], {"event": "shard_completed", **record.to_dict()})
    average_length = transitions / max(1, record.games)
    if summary.samples != transitions:
        printer.metric("full-search samples", summary.samples)
    printer.metric("new games", record.games)
    printer.metric("new transitions", transitions)
    printer.metric("avg game length", f"{average_length:.1f}")
    printer.done(f"wrote shard {shard_id} in {printer.elapsed()}")
    return ActorV2Summary(shard=record)

def _save_trajectory_shard(
    shard_dir: Path,
    *,
    episodes: tuple[TrajectoryEpisode, ...],
    logs: tuple[GameLog, ...],
) -> None:
    shard_dir.mkdir(parents=True, exist_ok=True)
    transitions = sum(len(episode.transitions) for episode in episodes)
    replay = TrajectoryReplayStore.from_episodes(max(1, transitions), episodes)
    _drop_async_unused_replay_arrays(replay)
    replay.save(
        shard_dir / "trajectory-replay.npz",
        compressed=False,
    )
    with (shard_dir / "game_logs.json").open("w", encoding="utf-8") as file:
        json.dump([log.to_dict() for log in logs], file, indent=2, sort_keys=True)

def _drop_async_unused_replay_arrays(replay: TrajectoryReplayStore) -> None:
    replay.next_features = None
    replay.next_features_present = None

def _with_transition_model_metadata(
    episodes: tuple[TrajectoryEpisode, ...],
    config: ActorV2Config,
) -> tuple[TrajectoryEpisode, ...]:
    model_version = _transition_model_version(config)
    created_iteration = _transition_created_iteration(config)
    return tuple(
        replace(
            episode,
            transitions=tuple(
                replace(
                    transition,
                    model_version=model_version,
                    created_iteration=created_iteration,
                )
                for transition in episode.transitions
            ),
        )
        for episode in episodes
    )

def _transition_model_version(config: ActorV2Config) -> int:
    if config.model_iteration is not None:
        return int(config.model_iteration)
    return 0

def _transition_created_iteration(config: ActorV2Config) -> int:
    return _transition_model_version(config)

def _next_actor_seed_start(config: ActorV2Config) -> int:
    records = load_v2_shard_records(_paths(config.work_dir)["metadata_path"])
    next_seed = config.seed_start
    for record in records:
        if record.model_version != config.model_version:
            continue
        next_seed = max(next_seed, record.seed_start + record.games)
    return next_seed

def _reserve_actor_seed_start(config: ActorV2Config) -> int:
    paths = _paths(config.work_dir)
    paths["shard_root"].mkdir(parents=True, exist_ok=True)
    lock_path = paths["actor_seed_lock_path"]
    state_path = paths["actor_seed_state_path"]
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            state = _load_actor_seed_state(state_path)
            key = config.model_version
            metadata_next = _next_actor_seed_start(config)
            reserved_start = max(
                config.seed_start,
                int(state.get(key, config.seed_start)),
                metadata_next,
            )
            state[key] = reserved_start + config.games
            _save_actor_seed_state(state_path, state)
            return reserved_start
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

def _load_actor_seed_state(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    data = _load_json_object(path, "actor seed state")
    return {str(key): int(value) for key, value in data.items()}

def _save_actor_seed_state(path: Path, state: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
    temporary_path.replace(path)

def _default_shard_id(*, model_version: str, seed_start: int, games: int) -> str:
    return f"{_safe_id(model_version)}-seed-{seed_start:08d}-games-{games:04d}"

def _safe_id(value: str) -> str:
    return "".join(char if char.isalnum() or char in "_.-" else "-" for char in value).strip("-")

def _validate_actor_config(config: ActorV2Config) -> None:
    if config.games <= 0:
        raise ValueError("games must be positive")
    if config.seed_start < 0:
        raise ValueError("seed_start must be non-negative")
    if config.model_iteration is not None and config.model_iteration < 0:
        raise ValueError("model_iteration must be non-negative")
    if config.onnx_max_batch_size <= 0:
        raise ValueError("onnx_max_batch_size must be positive")
    if config.rust_self_play_batch_size <= 0:
        raise ValueError("rust_self_play_batch_size must be positive")
