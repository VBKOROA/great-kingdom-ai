"""Replay sampling and tensor batch conversion for training."""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np

from great_kingdom_ai.augmentation import (
    augment_samples_randomly,
    augment_training_arrays_randomly,
)
from great_kingdom_ai.features import BOARD_CELLS, LEGAL_PLACE_FEATURE_CHANNEL, PASS_ACTION
from great_kingdom_ai.learner_prefetch import PrefetchIterator
from great_kingdom_ai.priority_sampling import PrioritySamplingConfig
from great_kingdom_ai.replay.sample import ReplaySample
from great_kingdom_ai.training.config import TrainingConfig
from great_kingdom_ai.training.torch_utils import _import_torch

if TYPE_CHECKING:
    import torch
    from torch import Tensor

@dataclass(frozen=True)
class TrainingBatch:
    features: torch.Tensor
    policy: torch.Tensor
    value: torch.Tensor
    legal_mask: torch.Tensor
    sample_weight: torch.Tensor
    replay_indexes: np.ndarray | None = None


@dataclass(frozen=True)
class TrainingArrays:
    features: np.ndarray
    policies: np.ndarray
    values: np.ndarray
    sample_weights: np.ndarray
    legal_masks: np.ndarray | None = None
    indexes: np.ndarray | None = None



class ReplayDataset(Protocol):
    def __len__(self) -> int: ...

    def sample(self, batch_size: int, rng: random.Random) -> list[ReplaySample]: ...


def samples_to_batch(
    samples: Sequence[ReplaySample],
    *,
    device: torch.device | str | None = None,
    pin_memory: bool = False,
) -> TrainingBatch:
    """Convert replay samples into tensors shaped for the policy-value network."""
    torch = _import_torch()
    if not samples:
        raise ValueError("training batch must contain at least one sample")

    features = np.stack([sample.features for sample in samples], axis=0).astype(np.float32)
    policies = np.stack([sample.policy for sample in samples], axis=0).astype(np.float32)
    values = np.asarray([sample.value for sample in samples], dtype=np.float32)
    sample_weights = np.asarray([sample.sample_weight for sample in samples], dtype=np.float32)
    legal_masks = _legal_masks_from_features(features)

    return TrainingBatch(
        features=_tensor_from_numpy(torch, features, pin_memory=pin_memory).to(device=device),
        policy=_tensor_from_numpy(torch, policies, pin_memory=pin_memory).to(device=device),
        value=_tensor_from_numpy(torch, values, pin_memory=pin_memory).to(device=device),
        legal_mask=_tensor_from_numpy(torch, legal_masks, pin_memory=pin_memory).to(
            device=device
        ),
        sample_weight=_tensor_from_numpy(torch, sample_weights, pin_memory=pin_memory).to(
            device=device
        ),
        replay_indexes=None,
    )


def arrays_to_batch(
    arrays: TrainingArrays,
    *,
    device: torch.device | str | None = None,
    pin_memory: bool = False,
) -> TrainingBatch:
    torch = _import_torch()
    features = np.ascontiguousarray(arrays.features, dtype=np.float32)
    policies = np.ascontiguousarray(arrays.policies, dtype=np.float32)
    values = np.ascontiguousarray(arrays.values, dtype=np.float32)
    sample_weights = np.ascontiguousarray(arrays.sample_weights, dtype=np.float32)
    if features.shape[0] == 0:
        raise ValueError("training batch must contain at least one sample")
    legal_masks = (
        _legal_masks_from_features(features)
        if arrays.legal_masks is None
        else np.ascontiguousarray(arrays.legal_masks, dtype=np.bool_)
    )
    if legal_masks.shape != policies.shape:
        raise ValueError("legal_masks shape must match policies shape")
    return TrainingBatch(
        features=_tensor_from_numpy(torch, features, pin_memory=pin_memory).to(device=device),
        policy=_tensor_from_numpy(torch, policies, pin_memory=pin_memory).to(device=device),
        value=_tensor_from_numpy(torch, values, pin_memory=pin_memory).to(device=device),
        legal_mask=_tensor_from_numpy(torch, legal_masks, pin_memory=pin_memory).to(
            device=device
        ),
        sample_weight=_tensor_from_numpy(torch, sample_weights, pin_memory=pin_memory).to(
            device=device
        ),
        replay_indexes=(
            None
            if arrays.indexes is None
            else np.ascontiguousarray(arrays.indexes, dtype=np.int64)
        ),
    )

def _iter_training_batches(
    replay: ReplayDataset,
    config: TrainingConfig,
    rng: random.Random,
    *,
    torch: Any,
) -> Iterator[TrainingBatch]:
    if not _use_cuda_prefetch(torch, config):
        for _ in range(config.steps):
            yield _sample_training_batch(
                replay,
                config,
                rng,
                device=config.device,
                pin_memory=False,
            )
        return

    cpu_batches = PrefetchIterator(
        producer=lambda: _sample_training_batch(
            replay,
            config,
            rng,
            device=None,
            pin_memory=True,
        ),
        count=config.steps,
        max_prefetch=config.prefetch_batches,
    )
    for batch in cpu_batches:
        yield _batch_to_device(batch, config.device, non_blocking=True)


def _sample_training_replay(
    replay: ReplayDataset,
    config: TrainingConfig,
    rng: random.Random,
) -> list[ReplaySample]:
    if config.priority_enabled:
        sampler = getattr(replay, "sample_priority_biased", None)
        if sampler is None:
            raise ValueError("replay dataset does not support priority-aware sampling")
        return cast(
            list[ReplaySample],
            sampler(
                config.batch_size,
                rng,
                priority_config=_priority_sampling_config(config),
                recent_fraction=config.recent_sample_fraction,
                recent_window=config.recent_sample_window,
            ),
        )
    if config.recent_sample_fraction <= 0.0:
        return replay.sample(config.batch_size, rng)
    if config.recent_sample_fraction > 1.0:
        raise ValueError("recent_sample_fraction must be in [0, 1]")
    if config.recent_sample_window <= 0:
        raise ValueError("recent_sample_window must be positive when recency sampling is enabled")
    sampler = getattr(replay, "sample_recency_biased", None)
    if sampler is None:
        raise ValueError("replay dataset does not support recency-biased sampling")
    return cast(
        list[ReplaySample],
        sampler(
            config.batch_size,
            rng,
            recent_fraction=config.recent_sample_fraction,
            recent_window=config.recent_sample_window,
        ),
    )


def _sample_training_batch(
    replay: ReplayDataset,
    config: TrainingConfig,
    rng: random.Random,
    *,
    device: torch.device | str | None,
    pin_memory: bool,
) -> TrainingBatch:
    array_sampler = getattr(replay, "sample_arrays", None)
    if array_sampler is not None:
        if config.priority_enabled:
            raw_arrays = array_sampler(
                config.batch_size,
                rng,
                recent_fraction=config.recent_sample_fraction,
                recent_window=config.recent_sample_window,
                priority_config=_priority_sampling_config(config),
            )
        else:
            raw_arrays = array_sampler(
                config.batch_size,
                rng,
                recent_fraction=config.recent_sample_fraction,
                recent_window=config.recent_sample_window,
            )
        arrays = TrainingArrays(
            features=np.asarray(raw_arrays.features, dtype=np.float32),
            policies=np.asarray(raw_arrays.policies, dtype=np.float32),
            values=np.asarray(raw_arrays.values, dtype=np.float32),
            sample_weights=np.asarray(raw_arrays.sample_weights, dtype=np.float32),
            legal_masks=(
                None
                if getattr(raw_arrays, "legal_masks", None) is None
                else np.asarray(raw_arrays.legal_masks, dtype=np.bool_)
            ),
            indexes=(
                None
                if getattr(raw_arrays, "indexes", None) is None
                else np.asarray(raw_arrays.indexes, dtype=np.int64)
            ),
        )
        if config.symmetry_augmentation:
            features, policies, legal_masks = augment_training_arrays_randomly(
                arrays.features,
                arrays.policies,
                arrays.legal_masks,
                rng,
            )
            arrays = TrainingArrays(
                features=features,
                policies=policies,
                values=arrays.values,
                sample_weights=arrays.sample_weights,
                legal_masks=legal_masks,
                indexes=arrays.indexes,
            )
        return arrays_to_batch(arrays, device=device, pin_memory=pin_memory)

    samples = _sample_training_replay(replay, config, rng)
    if config.symmetry_augmentation:
        samples = augment_samples_randomly(samples, rng)
    return samples_to_batch(samples, device=device, pin_memory=pin_memory)


def _priority_sampling_config(config: TrainingConfig) -> PrioritySamplingConfig:
    return PrioritySamplingConfig(
        enabled=config.priority_enabled,
        alpha=config.priority_alpha,
        beta=config.priority_beta,
        value_error_weight=config.priority_value_error_weight,
        policy_kl_weight=config.priority_policy_kl_weight,
        target_age_weight=config.priority_target_age_weight,
        search_reanalyzed_boost=config.priority_search_reanalyzed_boost,
        max_priority=config.priority_max_priority,
    )


def _batch_to_device(
    batch: TrainingBatch,
    device: torch.device | str | None,
    *,
    non_blocking: bool = False,
) -> TrainingBatch:
    if device is None:
        return batch
    return TrainingBatch(
        features=batch.features.to(device=device, non_blocking=non_blocking),
        policy=batch.policy.to(device=device, non_blocking=non_blocking),
        value=batch.value.to(device=device, non_blocking=non_blocking),
        legal_mask=batch.legal_mask.to(device=device, non_blocking=non_blocking),
        sample_weight=batch.sample_weight.to(device=device, non_blocking=non_blocking),
        replay_indexes=batch.replay_indexes,
    )


def _tensor_from_numpy(
    torch: Any,
    array: np.ndarray,
    *,
    pin_memory: bool,
) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(array))
    if pin_memory:
        return cast("Tensor", tensor.pin_memory())
    return cast("Tensor", tensor)


def _use_cuda_prefetch(torch: Any, config: TrainingConfig) -> bool:
    return bool(
        config.prefetch_batches > 0
        and config.steps > 0
        and str(config.device).startswith("cuda")
        and torch.cuda.is_available()
    )


def _legal_masks_from_features(features: np.ndarray) -> np.ndarray:
    legal_place = features[:, LEGAL_PLACE_FEATURE_CHANNEL].reshape(-1, BOARD_CELLS) > 0.5
    legal_mask = np.zeros((features.shape[0], BOARD_CELLS + 1), dtype=np.bool_)
    legal_mask[:, :BOARD_CELLS] = legal_place
    legal_mask[:, PASS_ACTION] = True
    return legal_mask


__all__ = [
    "ReplayDataset",
    "TrainingArrays",
    "TrainingBatch",
    "arrays_to_batch",
    "samples_to_batch",
]
