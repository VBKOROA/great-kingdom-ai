"""Sampled-batch reanalyze dataset for trajectory replay training."""

from __future__ import annotations

import math
import random
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from great_kingdom_ai.priority_sampling import (
    PrioritySamplingConfig,
    legal_masks_from_features,
    priority_scores,
    sample_priority_indexes,
)
from great_kingdom_ai.reanalyze import (
    ReanalyzeConfig,
    _create_onnx_evaluator,
    _effective_bootstrap_td_steps,
    _evaluate_policy_logits_values,
    _evaluate_policy_logits_values_with_onnx,
    _sample_indexes,
)
from great_kingdom_ai.replay_buffer import ReplaySample
from great_kingdom_ai.search_reanalyze import refresh_sampled_policies_with_search
from great_kingdom_ai.self_play_data import value_target_for_player
from great_kingdom_ai.trajectory_replay import TrajectoryReplayStore


@dataclass(frozen=True)
class OnSampleReanalyzeBatch:
    indexes: np.ndarray
    features: np.ndarray
    policies: np.ndarray
    values: np.ndarray
    sample_weights: np.ndarray
    legal_masks: np.ndarray
    search_reanalyzed: np.ndarray


@dataclass(frozen=True)
class OnSampleReanalyzeStats:
    sampled_batches: int
    sampled_rows: int
    policy_reanalyzed: int
    search_reanalyzed: int
    sampled_rows_per_batch: float
    value_eval_seconds: float
    search_seconds: float
    policy_reanalyze_ratio_applied: float
    stale_policy_fallbacks: int
    mcts_root_cache_hits: int
    mcts_root_cache_misses: int
    bootstrap_horizon_counts: dict[int, int]
    bootstrap_source_counts: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "sampled_batches": self.sampled_batches,
            "sampled_rows": self.sampled_rows,
            "policy_reanalyzed": self.policy_reanalyzed,
            "search_reanalyzed": self.search_reanalyzed,
            "sampled_rows_per_batch": self.sampled_rows_per_batch,
            "value_eval_seconds": self.value_eval_seconds,
            "search_seconds": self.search_seconds,
            "policy_reanalyze_ratio_applied": self.policy_reanalyze_ratio_applied,
            "stale_policy_fallbacks": self.stale_policy_fallbacks,
            "mcts_root_cache_hits": self.mcts_root_cache_hits,
            "mcts_root_cache_misses": self.mcts_root_cache_misses,
            "bootstrap_horizon_counts": {
                str(horizon): count
                for horizon, count in sorted(self.bootstrap_horizon_counts.items())
            },
            "bootstrap_source_counts": dict(sorted(self.bootstrap_source_counts.items())),
        }


class OnSampleReanalyzeDataset:
    """Refresh replay targets only for the learner batch being sampled."""

    def __init__(
        self,
        replay: TrajectoryReplayStore,
        *,
        checkpoint_path: str | Path,
        config: ReanalyzeConfig,
    ) -> None:
        from great_kingdom_ai.train import load_checkpoint

        if len(replay) == 0:
            raise ValueError("trajectory replay must contain at least one transition")
        checkpoint_device = "cpu" if config.onnx_model_path is not None else config.device
        state = load_checkpoint(checkpoint_path, device=checkpoint_device, prefer_ema=True)
        state.model.eval()

        self._replay = replay
        self._config = config
        self._checkpoint_path = Path(checkpoint_path)
        self._model = state.model
        self._model_version = state.step if config.model_version is None else config.model_version
        self._priorities = _initial_priorities(replay.sample_weights)
        self._sampled_batches = 0
        self._sampled_rows = 0
        self._policy_reanalyzed = 0
        self._stale_policy_fallbacks = 0
        self._value_eval_seconds = 0.0
        self._search_seconds = 0.0
        self._mcts_root_cache: dict[int, np.float32] = {}
        self._mcts_root_cache_hits = 0
        self._mcts_root_cache_misses = 0
        self._bootstrap_horizon_counts: Counter[int] = Counter()
        self._bootstrap_source_counts: Counter[str] = Counter()
        self._onnx_evaluator_local = threading.local()

    @property
    def checkpoint_path(self) -> Path:
        return self._checkpoint_path

    @property
    def model_version(self) -> int:
        return self._model_version

    @property
    def bootstrap_td_steps(self) -> int:
        return self._config.bootstrap_td_steps

    @property
    def gamma(self) -> float:
        return self._config.gamma

    @property
    def value_bootstrap_source(self) -> str:
        return self._config.value_bootstrap_source

    @property
    def dynamic_horizon_enabled(self) -> bool:
        return self._config.dynamic_horizon_enabled

    @property
    def priorities(self) -> np.ndarray:
        return self._priorities.copy()

    def __len__(self) -> int:
        return len(self._replay)

    def target_stats(self) -> OnSampleReanalyzeStats:
        sampled_rows_per_batch = (
            0.0 if self._sampled_batches == 0 else self._sampled_rows / self._sampled_batches
        )
        policy_reanalyze_ratio_applied = (
            0.0 if self._sampled_rows == 0 else self._policy_reanalyzed / self._sampled_rows
        )
        return OnSampleReanalyzeStats(
            sampled_batches=self._sampled_batches,
            sampled_rows=self._sampled_rows,
            policy_reanalyzed=self._policy_reanalyzed,
            search_reanalyzed=self._policy_reanalyzed,
            sampled_rows_per_batch=sampled_rows_per_batch,
            value_eval_seconds=self._value_eval_seconds,
            search_seconds=self._search_seconds,
            policy_reanalyze_ratio_applied=policy_reanalyze_ratio_applied,
            stale_policy_fallbacks=self._stale_policy_fallbacks,
            mcts_root_cache_hits=self._mcts_root_cache_hits,
            mcts_root_cache_misses=self._mcts_root_cache_misses,
            bootstrap_horizon_counts=dict(self._bootstrap_horizon_counts),
            bootstrap_source_counts=dict(self._bootstrap_source_counts),
        )

    def sample(self, batch_size: int, rng: random.Random) -> list[ReplaySample]:
        batch = self.sample_arrays(batch_size, rng)
        return [
            ReplaySample(
                features=batch.features[row],
                policy=batch.policies[row],
                value=float(batch.values[row]),
                sample_weight=float(batch.sample_weights[row]),
            )
            for row in range(batch.features.shape[0])
        ]

    def sample_arrays(
        self,
        batch_size: int,
        rng: random.Random,
        *,
        recent_fraction: float = 0.0,
        recent_window: int = 0,
        priority_config: PrioritySamplingConfig | None = None,
    ) -> OnSampleReanalyzeBatch:
        indexes, importance_weights = self._sample_indexes(
            batch_size,
            rng,
            recent_fraction=recent_fraction,
            recent_window=recent_window,
            priority_config=priority_config,
        )
        features = np.ascontiguousarray(self._replay.features[indexes], dtype=np.float32)
        legal_masks = self._legal_masks_for_rows(indexes, features)
        policies = np.ascontiguousarray(
            self._replay.policy_targets[indexes].astype(np.float32, copy=True),
            dtype=np.float32,
        )
        sampled_evaluation = (
            self._evaluate_logits_values(features, legal_masks)
            if _priority_update_needs_model_eval(priority_config)
            else None
        )
        policies, search_reanalyzed = self._refresh_sampled_policies(
            indexes,
            features,
            legal_masks,
            policies,
            rng,
            sampled_evaluation=sampled_evaluation,
        )
        sample_weights = np.ascontiguousarray(
            self._replay.sample_weights[indexes].astype(np.float32, copy=True)
            * importance_weights,
            dtype=np.float32,
        )
        values = self._sampled_bootstrap_targets(indexes)
        self._update_sampled_priorities(
            indexes=indexes,
            features=features,
            legal_masks=legal_masks,
            policies=policies,
            values=values,
            search_reanalyzed=search_reanalyzed,
            priority_config=priority_config,
            model_evaluation=sampled_evaluation,
        )
        self._sampled_batches += 1
        self._sampled_rows += len(indexes)
        self._policy_reanalyzed += int(np.count_nonzero(search_reanalyzed))
        return OnSampleReanalyzeBatch(
            indexes=np.asarray(indexes, dtype=np.int64),
            features=features,
            policies=policies,
            values=values,
            sample_weights=sample_weights,
            legal_masks=legal_masks,
            search_reanalyzed=search_reanalyzed,
        )

    def _sample_indexes(
        self,
        batch_size: int,
        rng: random.Random,
        *,
        recent_fraction: float,
        recent_window: int,
        priority_config: PrioritySamplingConfig | None,
    ) -> tuple[list[int], np.ndarray]:
        if priority_config is not None and priority_config.enabled:
            sampled = sample_priority_indexes(
                priorities=self._priorities.astype(np.float32, copy=False)
                ** np.float32(priority_config.alpha),
                batch_size=batch_size,
                rng=rng,
                beta=priority_config.beta,
                recent_fraction=recent_fraction,
                recent_window=recent_window,
            )
            return sampled.indexes, sampled.importance_weights
        indexes = _sample_indexes(
            len(self._replay),
            batch_size,
            rng,
            recent_fraction=recent_fraction,
            recent_window=recent_window,
        )
        return indexes, np.ones((batch_size,), dtype=np.float32)

    def _update_sampled_priorities(
        self,
        *,
        indexes: list[int],
        features: np.ndarray,
        legal_masks: np.ndarray,
        policies: np.ndarray,
        values: np.ndarray,
        search_reanalyzed: np.ndarray,
        priority_config: PrioritySamplingConfig | None,
        model_evaluation: tuple[np.ndarray, np.ndarray] | None,
    ) -> None:
        if not _priority_update_has_signal(priority_config):
            return
        if model_evaluation is None and _priority_update_needs_model_eval(priority_config):
            model_evaluation = self._evaluate_logits_values(features, legal_masks)
        policy_logits: np.ndarray | None = None
        value_predictions: np.ndarray | None = None
        if model_evaluation is not None:
            policy_logits, value_predictions = model_evaluation
        target_ages = np.maximum(
            self._model_version - self._replay.model_versions[indexes],
            0,
        ).astype(np.int64)
        self._priorities[indexes] = priority_scores(
            values=values,
            value_predictions=value_predictions,
            policies=policies,
            policy_logits=policy_logits,
            legal_masks=legal_masks,
            target_ages=target_ages,
            search_reanalyzed=search_reanalyzed,
            config=priority_config,
        )

    def _sampled_bootstrap_targets(self, indexes: list[int]) -> np.ndarray:
        targets = np.empty((len(indexes),), dtype=np.float32)
        bootstrap_rows: list[int] = []
        bootstrap_offsets: dict[int, int] = {}
        pending: list[tuple[int, int, int, int]] = []
        horizon_counts: Counter[int] = Counter()
        source_counts: Counter[str] = Counter()
        gamma = self._config.gamma

        for batch_row, replay_row in enumerate(indexes):
            episode_index = int(
                np.searchsorted(self._replay.episode_offsets, replay_row, side="right") - 1
            )
            terminal_row = int(self._replay.episode_offsets[episode_index + 1]) - 1
            effective_td_steps = _effective_bootstrap_td_steps(
                td_steps=self._config.bootstrap_td_steps,
                model_version=self._model_version,
                created_iteration=int(self._replay.created_iterations[replay_row]),
                dynamic_horizon_enabled=self._config.dynamic_horizon_enabled,
                dynamic_horizon_tau=self._config.dynamic_horizon_tau,
                dynamic_horizon_total_steps=self._config.dynamic_horizon_total_steps,
            )
            target_row = replay_row + effective_td_steps
            if (
                effective_td_steps == 0
                or bool(self._replay.terminals[replay_row])
                or target_row >= terminal_row
            ):
                targets[batch_row] = value_target_for_player(
                    player=int(self._replay.players[replay_row]),
                    winner=int(self._replay.episode_winners[episode_index]),
                )
                horizon_counts[0] += 1
                source_counts["terminal"] += 1
                continue
            offset = bootstrap_offsets.get(target_row)
            if offset is None:
                offset = len(bootstrap_rows)
                bootstrap_offsets[target_row] = offset
                bootstrap_rows.append(target_row)
            pending.append((batch_row, replay_row, target_row, effective_td_steps))
            horizon_counts[effective_td_steps] += 1
            source_counts[self._config.value_bootstrap_source] += 1

        if not pending:
            self._record_bootstrap_stats(horizon_counts, source_counts)
            return np.ascontiguousarray(targets, dtype=np.float32)

        bootstrap_features = np.ascontiguousarray(
            self._replay.features[bootstrap_rows],
            dtype=np.float32,
        )
        bootstrap_values = self._bootstrap_values_for_rows(
            bootstrap_rows,
            bootstrap_features,
        )
        for batch_row, replay_row, target_row, effective_td_steps in pending:
            bootstrap = float(bootstrap_values[bootstrap_offsets[target_row]])
            if int(self._replay.players[target_row]) != int(self._replay.players[replay_row]):
                bootstrap = -bootstrap
            targets[batch_row] = np.float32((gamma**effective_td_steps) * bootstrap)
        self._record_bootstrap_stats(horizon_counts, source_counts)
        return np.ascontiguousarray(targets, dtype=np.float32)

    def _record_bootstrap_stats(
        self,
        horizon_counts: Counter[int],
        source_counts: Counter[str],
    ) -> None:
        self._bootstrap_horizon_counts.update(horizon_counts)
        self._bootstrap_source_counts.update(source_counts)

    def _bootstrap_values_for_rows(
        self,
        bootstrap_rows: list[int],
        bootstrap_features: np.ndarray,
    ) -> np.ndarray:
        if self._config.value_bootstrap_source == "mcts_root":
            return self._mcts_root_values_for_rows(bootstrap_rows, bootstrap_features)

        bootstrap_legal_masks = np.ascontiguousarray(
            self._legal_masks_for_rows(bootstrap_rows, bootstrap_features),
            dtype=np.bool_,
        )
        policy_logits, value_head_values = self._evaluate_logits_values(
            bootstrap_features,
            bootstrap_legal_masks,
        )
        return value_head_values

    def _mcts_root_values_for_rows(
        self,
        bootstrap_rows: list[int],
        bootstrap_features: np.ndarray,
    ) -> np.ndarray:
        values = np.empty((len(bootstrap_rows),), dtype=np.float32)
        missing_positions: list[int] = []
        missing_rows: list[int] = []
        for position, replay_row in enumerate(bootstrap_rows):
            cached = self._mcts_root_cache.get(replay_row)
            if cached is None:
                missing_positions.append(position)
                missing_rows.append(replay_row)
            else:
                values[position] = cached
        self._mcts_root_cache_hits += len(bootstrap_rows) - len(missing_rows)
        self._mcts_root_cache_misses += len(missing_rows)
        if not missing_rows:
            return np.ascontiguousarray(values, dtype=np.float32)

        missing_features = np.ascontiguousarray(
            bootstrap_features[missing_positions],
            dtype=np.float32,
        )
        missing_legal_masks = np.ascontiguousarray(
            self._legal_masks_for_rows(missing_rows, missing_features),
            dtype=np.bool_,
        )
        policy_logits, value_head_values = self._evaluate_logits_values(
            missing_features,
            missing_legal_masks,
        )
        search_result = self._refresh_with_search(
            transitions=self._replay.transition_refs(missing_rows),
            policies=np.ascontiguousarray(
                self._replay.policy_targets[missing_rows],
                dtype=np.float32,
            ),
            policy_logits=policy_logits,
            refreshed_values=value_head_values,
            model=self._model,
            device=self._config.device,
            onnx_evaluator=self._onnx_evaluator_for_current_thread(),
            config=self._config.search,
        )
        if search_result.root_values is None or not np.isfinite(search_result.root_values).all():
            raise RuntimeError("MCTS root bootstrap requires root values from search results")
        root_values = np.ascontiguousarray(search_result.root_values, dtype=np.float32)
        for position, replay_row, root_value in zip(
            missing_positions,
            missing_rows,
            root_values,
            strict=True,
        ):
            value = np.float32(root_value)
            self._mcts_root_cache[replay_row] = value
            values[position] = value
        return np.ascontiguousarray(values, dtype=np.float32)

    def _refresh_sampled_policies(
        self,
        indexes: list[int],
        features: np.ndarray,
        legal_masks: np.ndarray,
        policies: np.ndarray,
        rng: random.Random,
        sampled_evaluation: tuple[np.ndarray, np.ndarray] | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        row_count = len(indexes)
        ratio = self._config.policy_reanalyze_ratio
        search_reanalyzed = np.zeros((row_count,), dtype=np.bool_)
        if row_count == 0 or ratio <= 0.0:
            return policies, search_reanalyzed

        reanalyze_count = min(row_count, math.ceil(row_count * ratio))
        selected_positions = sorted(rng.sample(range(row_count), reanalyze_count))
        selected_replay_rows = [indexes[position] for position in selected_positions]
        if sampled_evaluation is None:
            selected_features = np.ascontiguousarray(features[selected_positions], dtype=np.float32)
            selected_legal_masks = np.ascontiguousarray(
                legal_masks[selected_positions],
                dtype=np.bool_,
            )
            policy_logits, refreshed_values = self._evaluate_logits_values(
                selected_features,
                selected_legal_masks,
            )
        else:
            batch_policy_logits, batch_values = sampled_evaluation
            policy_logits = np.ascontiguousarray(
                batch_policy_logits[selected_positions],
                dtype=np.float32,
            )
            refreshed_values = np.ascontiguousarray(
                batch_values[selected_positions],
                dtype=np.float32,
            )
        search_result = self._refresh_with_search(
            transitions=self._replay.transition_refs(selected_replay_rows),
            policies=np.ascontiguousarray(policies[selected_positions], dtype=np.float32),
            policy_logits=policy_logits,
            refreshed_values=refreshed_values,
            model=self._model,
            device=self._config.device,
            onnx_evaluator=self._onnx_evaluator_for_current_thread(),
            config=self._config.search,
        )
        refreshed = policies.copy()
        refreshed[selected_positions] = search_result.policies
        search_reanalyzed[selected_positions] = search_result.search_reanalyzed
        self._stale_policy_fallbacks += int(
            len(selected_positions) - np.count_nonzero(search_result.search_reanalyzed)
        )
        return np.ascontiguousarray(refreshed, dtype=np.float32), search_reanalyzed

    def _refresh_with_search(self, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            return refresh_sampled_policies_with_search(**kwargs)
        finally:
            self._search_seconds += time.perf_counter() - start

    def _legal_masks_for_rows(self, indexes: list[int], features: np.ndarray) -> np.ndarray:
        replay_legal_masks = np.ascontiguousarray(self._replay.legal_masks[indexes], dtype=np.bool_)
        feature_legal_masks = legal_masks_from_features(features)
        if not np.array_equal(replay_legal_masks, feature_legal_masks):
            raise ValueError("replay legal masks do not match feature-derived legal masks")
        return replay_legal_masks

    def _onnx_evaluator_for_current_thread(self) -> Any | None:
        if self._config.onnx_model_path is None:
            return None
        evaluator = getattr(self._onnx_evaluator_local, "evaluator", None)
        if evaluator is None:
            evaluator = _create_onnx_evaluator(
                self._config.onnx_model_path,
                device=self._config.onnx_device or self._config.device,
                max_batch_size=self._config.onnx_max_batch_size,
            )
            self._onnx_evaluator_local.evaluator = evaluator
        return evaluator

    def _evaluate_logits_values(
        self,
        features: np.ndarray,
        legal_masks: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        start = time.perf_counter()
        try:
            onnx_evaluator = self._onnx_evaluator_for_current_thread()
            if onnx_evaluator is None:
                return _evaluate_policy_logits_values(
                    self._model,
                    features,
                    legal_masks,
                    batch_size=self._config.batch_size,
                    device=self._config.device,
                )
            return _evaluate_policy_logits_values_with_onnx(
                cast(Any, onnx_evaluator),
                features,
                batch_size=self._config.batch_size,
                device=self._config.onnx_device or self._config.device,
            )
        finally:
            self._value_eval_seconds += time.perf_counter() - start


def _initial_priorities(sample_weights: np.ndarray) -> np.ndarray:
    priorities = np.asarray(sample_weights, dtype=np.float32).copy()
    if priorities.shape != sample_weights.shape:
        raise ValueError("priority shape must match replay sample weights")
    if not np.isfinite(priorities).all() or np.any(priorities <= 0.0):
        raise ValueError("on-sample priorities must be finite and positive")
    return np.ascontiguousarray(priorities, dtype=np.float32)


def _priority_update_has_signal(config: PrioritySamplingConfig | None) -> bool:
    return (
        config is not None
        and config.enabled
        and (
            config.value_error_weight > 0.0
            or config.policy_kl_weight > 0.0
            or config.target_age_weight > 0.0
            or config.search_reanalyzed_boost > 1.0
        )
    )


def _priority_update_needs_model_eval(config: PrioritySamplingConfig | None) -> bool:
    return (
        config is not None
        and config.enabled
        and (config.value_error_weight > 0.0 or config.policy_kl_weight > 0.0)
    )


__all__ = ["OnSampleReanalyzeBatch", "OnSampleReanalyzeDataset", "OnSampleReanalyzeStats"]
