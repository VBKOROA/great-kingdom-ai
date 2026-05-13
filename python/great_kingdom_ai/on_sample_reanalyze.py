"""Sampled-batch reanalyze dataset for trajectory replay training."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from great_kingdom_ai.priority_sampling import (
    PrioritySamplingConfig,
    legal_masks_from_features,
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
        self._onnx_evaluator = (
            None
            if config.onnx_model_path is None
            else _create_onnx_evaluator(
                config.onnx_model_path,
                device=config.onnx_device or config.device,
                max_batch_size=config.onnx_max_batch_size,
            )
        )

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

    def __len__(self) -> int:
        return len(self._replay)

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
        policies = np.ascontiguousarray(
            self._replay.policy_targets[indexes].astype(np.float32, copy=True),
            dtype=np.float32,
        )
        policies, search_reanalyzed = self._refresh_sampled_policies(
            indexes,
            features,
            policies,
            rng,
        )
        sample_weights = np.ascontiguousarray(
            self._replay.sample_weights[indexes].astype(np.float32, copy=True)
            * importance_weights,
            dtype=np.float32,
        )
        return OnSampleReanalyzeBatch(
            indexes=np.asarray(indexes, dtype=np.int64),
            features=features,
            policies=policies,
            values=self._sampled_bootstrap_targets(indexes),
            sample_weights=sample_weights,
            legal_masks=legal_masks_from_features(features),
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
                priorities=self._replay.sample_weights.astype(np.float32, copy=False)
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

    def _sampled_bootstrap_targets(self, indexes: list[int]) -> np.ndarray:
        targets = np.empty((len(indexes),), dtype=np.float32)
        bootstrap_rows: list[int] = []
        bootstrap_offsets: dict[int, int] = {}
        pending: list[tuple[int, int, int, int]] = []
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
                continue
            offset = bootstrap_offsets.get(target_row)
            if offset is None:
                offset = len(bootstrap_rows)
                bootstrap_offsets[target_row] = offset
                bootstrap_rows.append(target_row)
            pending.append((batch_row, replay_row, target_row, effective_td_steps))

        if not pending:
            return np.ascontiguousarray(targets, dtype=np.float32)

        bootstrap_features = np.ascontiguousarray(
            self._replay.features[bootstrap_rows],
            dtype=np.float32,
        )
        _, bootstrap_values = self._evaluate_logits_values(
            bootstrap_features,
            np.ascontiguousarray(self._replay.legal_masks[bootstrap_rows], dtype=np.bool_),
        )
        for batch_row, replay_row, target_row, effective_td_steps in pending:
            bootstrap = float(bootstrap_values[bootstrap_offsets[target_row]])
            if int(self._replay.players[target_row]) != int(self._replay.players[replay_row]):
                bootstrap = -bootstrap
            targets[batch_row] = np.float32((gamma**effective_td_steps) * bootstrap)
        return np.ascontiguousarray(targets, dtype=np.float32)

    def _refresh_sampled_policies(
        self,
        indexes: list[int],
        features: np.ndarray,
        policies: np.ndarray,
        rng: random.Random,
    ) -> tuple[np.ndarray, np.ndarray]:
        row_count = len(indexes)
        ratio = self._config.policy_reanalyze_ratio
        search_reanalyzed = np.zeros((row_count,), dtype=np.bool_)
        if row_count == 0 or ratio <= 0.0:
            return policies, search_reanalyzed

        reanalyze_count = min(row_count, math.ceil(row_count * ratio))
        selected_positions = sorted(rng.sample(range(row_count), reanalyze_count))
        selected_replay_rows = [indexes[position] for position in selected_positions]
        selected_features = np.ascontiguousarray(features[selected_positions], dtype=np.float32)
        selected_legal_masks = np.ascontiguousarray(
            self._replay.legal_masks[selected_replay_rows],
            dtype=np.bool_,
        )
        policy_logits, refreshed_values = self._evaluate_logits_values(
            selected_features,
            selected_legal_masks,
        )
        search_result = refresh_sampled_policies_with_search(
            transitions=self._replay.transition_refs(selected_replay_rows),
            policies=np.ascontiguousarray(policies[selected_positions], dtype=np.float32),
            policy_logits=policy_logits,
            refreshed_values=refreshed_values,
            model=self._model,
            device=self._config.device,
            onnx_evaluator=self._onnx_evaluator,
            config=self._config.search,
        )
        refreshed = policies.copy()
        refreshed[selected_positions] = search_result.policies
        search_reanalyzed[selected_positions] = search_result.search_reanalyzed
        return np.ascontiguousarray(refreshed, dtype=np.float32), search_reanalyzed

    def _evaluate_logits_values(
        self,
        features: np.ndarray,
        legal_masks: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._onnx_evaluator is None:
            return _evaluate_policy_logits_values(
                self._model,
                features,
                legal_masks,
                batch_size=self._config.batch_size,
                device=self._config.device,
            )
        return _evaluate_policy_logits_values_with_onnx(
            cast(Any, self._onnx_evaluator),
            features,
            batch_size=self._config.batch_size,
            device=self._config.onnx_device or self._config.device,
        )


__all__ = ["OnSampleReanalyzeBatch", "OnSampleReanalyzeDataset"]
