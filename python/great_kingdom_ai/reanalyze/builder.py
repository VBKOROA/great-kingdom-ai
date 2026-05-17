"""Build and save reanalyze target snapshots."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from great_kingdom_ai.reanalyze.config import ReanalyzeConfig, ReanalyzeProgressCallback
from great_kingdom_ai.reanalyze.snapshot import ReanalyzeTargetSnapshot, _validated_snapshot
from great_kingdom_ai.reanalyze.summary import ReanalyzeSummary
from great_kingdom_ai.reanalyze_evaluator import (
    create_onnx_evaluator,
    evaluate_policy_logits_values,
    evaluate_policy_logits_values_with_onnx,
)
from great_kingdom_ai.reanalyze_targets import (
    bootstrap_targets_from_store,
    mcts_root_bootstrap_values_from_store,
)
from great_kingdom_ai.replay import TrajectoryReplayStore
from great_kingdom_ai.search_reanalyze import refresh_policies_with_search


def reanalyze_replay(
    *,
    replay_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    config: ReanalyzeConfig | None = None,
    progress_callback: ReanalyzeProgressCallback | None = None,
) -> ReanalyzeSummary:
    resolved_config = config if config is not None else ReanalyzeConfig()
    _report_progress(progress_callback, "load", 0, 1, f"replay={replay_path}")
    replay = TrajectoryReplayStore.load(replay_path)
    _report_progress(progress_callback, "load", 1, 1, f"transitions={len(replay)}")
    snapshot = build_reanalyze_snapshot_from_store(
        replay,
        checkpoint_path=checkpoint_path,
        config=resolved_config,
        progress_callback=progress_callback,
    )
    _report_progress(progress_callback, "save", 0, 1, f"output={output_path}")
    snapshot.save(output_path, compressed=resolved_config.compressed)
    _report_progress(progress_callback, "save", 1, 1, f"rows={len(snapshot)}")
    return ReanalyzeSummary(
        replay_path=Path(replay_path),
        checkpoint_path=Path(checkpoint_path),
        output_path=Path(output_path),
        transitions=len(snapshot),
        model_version=snapshot.model_version,
        bootstrap_td_steps=snapshot.bootstrap_td_steps,
        gamma=snapshot.gamma,
        search_reanalyzed=(
            0
            if snapshot.search_reanalyzed is None
            else int(np.count_nonzero(snapshot.search_reanalyzed))
        ),
    )


def reanalyze_replay_store(
    *,
    replay: TrajectoryReplayStore,
    checkpoint_path: str | Path,
    output_path: str | Path,
    replay_path: str | Path,
    config: ReanalyzeConfig | None = None,
    progress_callback: ReanalyzeProgressCallback | None = None,
) -> ReanalyzeSummary:
    resolved_config = config if config is not None else ReanalyzeConfig()
    snapshot = build_reanalyze_snapshot_from_store(
        replay,
        checkpoint_path=checkpoint_path,
        config=resolved_config,
        progress_callback=progress_callback,
    )
    _report_progress(progress_callback, "save", 0, 1, f"output={output_path}")
    snapshot.save(output_path, compressed=resolved_config.compressed)
    _report_progress(progress_callback, "save", 1, 1, f"rows={len(snapshot)}")
    return ReanalyzeSummary(
        replay_path=Path(replay_path),
        checkpoint_path=Path(checkpoint_path),
        output_path=Path(output_path),
        transitions=len(snapshot),
        model_version=snapshot.model_version,
        bootstrap_td_steps=snapshot.bootstrap_td_steps,
        gamma=snapshot.gamma,
        search_reanalyzed=(
            0
            if snapshot.search_reanalyzed is None
            else int(np.count_nonzero(snapshot.search_reanalyzed))
        ),
    )


def build_reanalyze_snapshot_from_store(
    replay: TrajectoryReplayStore,
    *,
    checkpoint_path: str | Path,
    config: ReanalyzeConfig,
    progress_callback: ReanalyzeProgressCallback | None = None,
) -> ReanalyzeTargetSnapshot:
    from great_kingdom_ai.training import load_checkpoint

    if len(replay) == 0:
        raise ValueError("trajectory replay must contain at least one transition")
    _report_progress(progress_callback, "checkpoint", 0, 1, f"loading {checkpoint_path}")
    checkpoint_device = "cpu" if config.onnx_model_path is not None else config.device
    state = load_checkpoint(checkpoint_path, device=checkpoint_device, prefer_ema=True)
    state.model.eval()
    eval_backend = "onnx" if config.onnx_model_path is not None else "pytorch"
    _report_progress(
        progress_callback,
        "checkpoint",
        1,
        1,
        f"device={config.device}, eval_backend={eval_backend}",
    )
    model_version = state.step if config.model_version is None else config.model_version

    features = np.ascontiguousarray(replay.features, dtype=np.float32)
    legal_masks = np.ascontiguousarray(replay.legal_masks, dtype=np.bool_)
    policies = np.ascontiguousarray(replay.policy_targets, dtype=np.float32)
    _report_progress(progress_callback, "arrays", 1, 1, f"rows={features.shape[0]}")
    onnx_evaluator = (
        None
        if config.onnx_model_path is None
        else create_onnx_evaluator(
            config.onnx_model_path,
            device=config.onnx_device or config.device,
            max_batch_size=config.onnx_max_batch_size,
        )
    )
    if onnx_evaluator is None:
        policy_logits, refreshed_values = evaluate_policy_logits_values(
            state.model,
            features,
            legal_masks,
            batch_size=config.batch_size,
            device=config.device,
            progress_callback=progress_callback,
        )
    else:
        policy_logits, refreshed_values = evaluate_policy_logits_values_with_onnx(
            onnx_evaluator,
            features,
            batch_size=config.batch_size,
            device=config.onnx_device or config.device,
            progress_callback=progress_callback,
        )
    _report_progress(
        progress_callback,
        "bootstrap",
        0,
        1,
        f"td_steps={config.bootstrap_td_steps}, gamma={config.gamma:g}",
    )
    bootstrap_values = (
        refreshed_values
        if config.value_bootstrap_source == "value_head"
        else mcts_root_bootstrap_values_from_store(
            replay,
            policy_logits=policy_logits,
            refreshed_values=refreshed_values,
            model=state.model,
            device=config.device,
            onnx_evaluator=onnx_evaluator,
            config=config.search,
        )
    )
    values = bootstrap_targets_from_store(
        replay,
        bootstrap_values,
        td_steps=config.bootstrap_td_steps,
        gamma=config.gamma,
        model_version=model_version,
        dynamic_horizon_enabled=config.dynamic_horizon_enabled,
        dynamic_horizon_tau=config.dynamic_horizon_tau,
        dynamic_horizon_total_steps=config.dynamic_horizon_total_steps,
    )
    _report_progress(progress_callback, "bootstrap", 1, 1, f"rows={values.shape[0]}")
    source_model_versions = np.asarray(replay.model_versions, dtype=np.int64)
    target_ages = np.maximum(model_version - source_model_versions, 0).astype(np.int64)
    search_reanalyzed: np.ndarray | None = None
    if config.search.enabled:
        episodes = replay.episodes
        search_result = refresh_policies_with_search(
            episodes=episodes,
            policies=policies,
            values=values,
            policy_logits=policy_logits,
            refreshed_values=refreshed_values,
            target_ages=target_ages,
            model=state.model,
            device=config.device,
            onnx_evaluator=onnx_evaluator,
            config=config.search,
            progress_callback=progress_callback,
        )
        policies = search_result.policies
        search_reanalyzed = search_result.search_reanalyzed
    else:
        _report_progress(progress_callback, "search", 0, 0, "disabled")

    return _validated_snapshot(
        ReanalyzeTargetSnapshot(
            features=features,
            policies=policies,
            values=values,
            refreshed_values=refreshed_values,
            sample_weights=np.asarray(replay.sample_weights, dtype=np.float32),
            episode_ids=_transition_episode_values(replay, replay.episode_ids),
            timesteps=np.asarray(replay.timesteps, dtype=np.int64),
            players=np.asarray(replay.players, dtype=np.int64),
            source_model_versions=source_model_versions,
            created_iterations=np.asarray(replay.created_iterations, dtype=np.int64),
            target_ages=target_ages,
            model_version=model_version,
            bootstrap_td_steps=config.bootstrap_td_steps,
            gamma=config.gamma,
            checkpoint_path=str(checkpoint_path),
            policy_logits=policy_logits,
            search_reanalyzed=search_reanalyzed,
        )
    )


def build_reanalyze_snapshot(
    replay: TrajectoryReplayStore,
    *,
    checkpoint_path: str | Path,
    config: ReanalyzeConfig,
    progress_callback: ReanalyzeProgressCallback | None = None,
) -> ReanalyzeTargetSnapshot:
    return build_reanalyze_snapshot_from_store(
        replay,
        checkpoint_path=checkpoint_path,
        config=config,
        progress_callback=progress_callback,
    )


def _transition_episode_values(
    replay: TrajectoryReplayStore,
    episode_values: np.ndarray,
) -> np.ndarray:
    counts = np.diff(replay.episode_offsets)
    return np.repeat(np.asarray(episode_values, dtype=np.int64), counts).astype(np.int64)


def _report_progress(
    progress_callback: ReanalyzeProgressCallback | None,
    stage: str,
    current: int,
    total: int,
    detail: str,
) -> None:
    if progress_callback is not None:
        progress_callback(stage, current, total, detail)


__all__ = [
    "build_reanalyze_snapshot",
    "build_reanalyze_snapshot_from_store",
    "reanalyze_replay",
    "reanalyze_replay_store",
]
