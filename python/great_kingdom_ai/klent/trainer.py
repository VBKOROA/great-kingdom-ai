"""Synchronous KLENT collect/train iteration loop (paper Algorithm 1).

Each iteration freezes theta_k, collects a fresh whole-game buffer with
zero-search self-play, fits only on that buffer, and publishes theta_{k+1}
with an iteration checkpoint. Resume restarts the unfinished iteration from
the last published checkpoint instead of mixing partial state.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from great_kingdom_ai.augmentation import augment_training_arrays_randomly
from great_kingdom_ai.klent._torch import _import_torch
from great_kingdom_ai.klent.checkpoint import (
    KlentTrainState,
    load_klent_checkpoint,
    save_klent_checkpoint,
    warm_start_klent_model,
)
from great_kingdom_ai.klent.dataset import KlentReplayDataset
from great_kingdom_ai.klent.loss import KlentLossBreakdown, compute_klent_losses
from great_kingdom_ai.klent.publish import publish_klent_onnx_artifacts
from great_kingdom_ai.klent.self_play import KlentSelfPlayConfig, play_klent_game
from great_kingdom_ai.klent.shards import (
    KlentShardMetadata,
    save_klent_shard,
    shard_metadata_path,
)
from great_kingdom_ai.klent.types import KlentConfig, KlentPolicyValueModel
from great_kingdom_ai.replay.persistence import copy_file_atomic
from great_kingdom_ai.replay.trajectory import TrajectoryReplayStore
from great_kingdom_ai.training.batch import TrainingArrays, TrainingBatch, arrays_to_batch

if TYPE_CHECKING:
    from torch import nn
    from torch.optim import Optimizer

    from great_kingdom_ai.replay import TrajectoryEpisode


@dataclass(frozen=True)
class KlentTrainConfig:
    work_dir: Path = Path("data/klent")
    klent: KlentConfig = KlentConfig()
    model_preset: str = "strong_attn_klent"
    device: str = "cpu"
    seed: int = 0
    min_transitions: int = 4096
    max_games_per_iteration: int = 64
    fit_epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    gradient_clip_norm: float | None = None
    max_turns: int = 200
    symmetry_augmentation: bool = True
    amp: bool = False
    keep_shards: bool = True
    export_onnx: bool = True
    onnx_device: str = "cpu"
    onnx_precision: str = "fp32"
    check_onnx_parity: bool = True
    use_rust_actor: bool = False
    actor_onnx_path: Path | None = None
    rust_self_play_batch_size: int = 64
    warm_start_checkpoint: Path | None = None

    def __post_init__(self) -> None:
        if self.min_transitions <= 0:
            raise ValueError("min_transitions must be positive")
        if self.max_games_per_iteration <= 0:
            raise ValueError("max_games_per_iteration must be positive")
        if self.fit_epochs <= 0:
            raise ValueError("fit_epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if not np.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("weight_decay must be finite and non-negative")
        if self.max_turns <= 0:
            raise ValueError("max_turns must be positive")
        if self.gradient_clip_norm is not None and (
            not np.isfinite(self.gradient_clip_norm) or self.gradient_clip_norm <= 0.0
        ):
            raise ValueError("gradient_clip_norm must be finite and positive")
        if self.onnx_device not in {"cpu", "cuda"}:
            raise ValueError("onnx_device must be one of: cpu, cuda")
        if self.onnx_precision not in {"fp32", "fp16"}:
            raise ValueError("onnx_precision must be one of: fp32, fp16")
        if self.rust_self_play_batch_size <= 0:
            raise ValueError("rust_self_play_batch_size must be positive")
        if self.export_onnx and self.onnx_precision == "fp16" and self.check_onnx_parity:
            raise ValueError(
                "CPU parity checks only support fp32 ONNX; "
                "disable check_onnx_parity for fp16 publication"
            )


@dataclass(frozen=True)
class KlentIterationSummary:
    iteration: int
    games: int
    transitions: int
    shard_path: Path
    checkpoint_path: Path
    epoch_losses: list[float]
    onnx_version_dir: Path | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "iteration": self.iteration,
            "games": self.games,
            "transitions": self.transitions,
            "shard_path": str(self.shard_path),
            "checkpoint_path": str(self.checkpoint_path),
            "epoch_losses": self.epoch_losses,
            "onnx_version_dir": (
                None if self.onnx_version_dir is None else str(self.onnx_version_dir)
            ),
        }


def run_klent_training(
    config: KlentTrainConfig,
    *,
    iterations: int,
    resume: bool = True,
) -> list[KlentIterationSummary]:
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    torch = _import_torch()
    config.work_dir.mkdir(parents=True, exist_ok=True)
    _iteration_shard_path(config.work_dir, 0).parent.mkdir(parents=True, exist_ok=True)
    _iteration_checkpoint_path(config.work_dir, 0).parent.mkdir(parents=True, exist_ok=True)

    state = _initial_state(config, resume=resume)
    summaries: list[KlentIterationSummary] = []
    while state.iteration < iterations:
        state, summary = run_klent_iteration(state, config)
        summaries.append(summary)
    del torch
    return summaries


def run_klent_iteration(
    state: KlentTrainState,
    config: KlentTrainConfig,
) -> tuple[KlentTrainState, KlentIterationSummary]:
    torch = _import_torch()
    iteration = state.iteration
    model = state.model
    model.eval()
    if config.use_rust_actor:
        episodes = _collect_with_rust_actor(state, config, iteration)
    else:
        episodes = _collect_with_python_actor(model, config, iteration)
    transition_count = sum(len(episode.transitions) for episode in episodes)
    if not episodes:
        raise RuntimeError("KLENT collection produced no episodes")

    store = TrajectoryReplayStore.from_episodes(max(transition_count, 1), episodes)
    metadata = KlentShardMetadata.from_config(
        config.klent,
        iteration=iteration,
        model_version=iteration,
        transitions=len(store),
        games=len(episodes),
    )
    shard_path = _iteration_shard_path(config.work_dir, iteration)
    save_klent_shard(store, shard_path, metadata)

    dataset = KlentReplayDataset(store, config=config.klent, metadata=metadata)
    model.train()
    epoch_losses, total_steps = fit_klent_model(
        model,
        dataset,
        config,
        state.optimizer,
        start_steps=state.total_steps,
        iteration=iteration,
    )
    model.eval()

    next_iteration = iteration + 1
    next_state = KlentTrainState(
        model=model,
        optimizer=state.optimizer,
        iteration=next_iteration,
        total_steps=total_steps,
        klent_config=config.klent,
        model_preset=config.model_preset,
        last_shard=str(shard_path),
    )
    checkpoint_path = _iteration_checkpoint_path(config.work_dir, next_iteration)
    save_klent_checkpoint(next_state, checkpoint_path)
    copy_file_atomic(checkpoint_path, _latest_checkpoint_path(config.work_dir))
    onnx_version_dir: Path | None = None
    if config.export_onnx:
        manifest = publish_klent_onnx_artifacts(
            checkpoint_path,
            config.work_dir,
            model_version=next_iteration,
            iteration=iteration,
            klent_config=config.klent,
            model_preset=config.model_preset,
            device=config.onnx_device,
            precision=config.onnx_precision,
            check_parity=config.check_onnx_parity,
            overwrite=True,
        )
        onnx_version_dir = _onnx_version_dir(config.work_dir, manifest.model_version)
    if not config.keep_shards:
        metadata_path = shard_metadata_path(shard_path)
        if metadata_path.exists():
            metadata_path.unlink()
        if shard_path.exists():
            shard_path.unlink()

    del torch
    return next_state, KlentIterationSummary(
        iteration=iteration,
        games=len(episodes),
        transitions=len(store),
        shard_path=shard_path,
        checkpoint_path=checkpoint_path,
        epoch_losses=epoch_losses,
        onnx_version_dir=onnx_version_dir,
    )


def fit_klent_model(
    model: nn.Module,
    dataset: KlentReplayDataset,
    config: KlentTrainConfig,
    optimizer: Optimizer,
    *,
    start_steps: int,
    iteration: int,
) -> tuple[list[float], int]:
    """Fit for ``fit_epochs`` shuffled passes over the frozen iteration buffer."""
    torch = _import_torch()
    permutation_rng = np.random.default_rng(config.seed + iteration)
    augment_rng = random.Random(config.seed + iteration)
    steps = start_steps
    epoch_losses: list[float] = []
    for _epoch in range(config.fit_epochs):
        order = permutation_rng.permutation(len(dataset))
        batch_total = 0.0
        batch_count = 0
        for start in range(0, len(order), config.batch_size):
            indexes = np.asarray(order[start : start + config.batch_size], dtype=np.int64)
            batch = _klent_batch_from_indexes(dataset, indexes, config, augment_rng)
            optimizer.zero_grad(set_to_none=True)
            losses = compute_klent_losses(model, batch, config.klent)
            losses.total.backward()  # type: ignore[no-untyped-call]
            if config.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            steps += 1
            batch_total += float(losses.total.detach().cpu())
            batch_count += 1
        epoch_losses.append(batch_total / max(batch_count, 1))
    return epoch_losses, steps


def compute_iteration_loss(
    model: nn.Module,
    dataset: KlentReplayDataset,
    config: KlentTrainConfig,
) -> KlentLossBreakdown:
    """Compute the loss over the whole buffer in one batch for diagnostics."""
    indexes = np.arange(len(dataset), dtype=np.int64)
    batch = _klent_batch_from_indexes(dataset, indexes, config, random.Random(0))
    return compute_klent_losses(model, batch, config.klent)


def _collect_with_python_actor(
    model: KlentPolicyValueModel,
    config: KlentTrainConfig,
    iteration: int,
) -> list[TrajectoryEpisode]:
    episodes: list[TrajectoryEpisode] = []
    transition_count = 0
    seed_base = config.seed + iteration * 1_000_003
    while (
        transition_count < config.min_transitions
        and len(episodes) < config.max_games_per_iteration
    ):
        _log, episode = play_klent_game(
            model,
            seed=seed_base + len(episodes),
            config=config.klent,
            self_play=KlentSelfPlayConfig(
                max_turns=config.max_turns,
                episode_id=len(episodes),
                model_version=iteration,
                created_iteration=iteration,
            ),
        )
        episodes.append(episode)
        transition_count += len(episode.transitions)
    return episodes


def _collect_with_rust_actor(
    state: KlentTrainState,
    config: KlentTrainConfig,
    iteration: int,
) -> list[TrajectoryEpisode]:
    from great_kingdom_ai.klent.export import export_klent_checkpoint_to_onnx
    from great_kingdom_ai.klent.rust_actor import (
        RustKlentActorConfig,
        play_rust_klent_zero_search,
    )

    actor_path: Path | None = config.actor_onnx_path
    if actor_path is None:
        source_checkpoint = (
            config.work_dir / "checkpoints" / f"actor-source-{iteration:04d}.pt"
        )
        save_klent_checkpoint(state, source_checkpoint)
        actor_path = config.work_dir / "onnx" / f"actor-source-{iteration:04d}.onnx"
        export_klent_checkpoint_to_onnx(
            source_checkpoint,
            actor_path,
            kind="actor",
            device=config.onnx_device,
            precision=config.onnx_precision,
        )
    elif not actor_path.exists():
        raise FileNotFoundError(f"actor ONNX model is missing: {actor_path}")

    episodes: list[TrajectoryEpisode] = []
    transition_count = 0
    seed_base = config.seed + iteration * 1_000_003
    game_offset = 0
    while (
        transition_count < config.min_transitions
        and game_offset < config.max_games_per_iteration
    ):
        batch_size = min(
            config.rust_self_play_batch_size,
            config.max_games_per_iteration - game_offset,
        )
        summary = play_rust_klent_zero_search(
            RustKlentActorConfig(
                actor_onnx_path=actor_path,
                output_dir=config.work_dir,
                games=batch_size,
                seed_start=seed_base + game_offset,
                alpha=config.klent.alpha,
                beta=config.klent.beta,
                lambda_param=config.klent.lambda_param,
                gamma=config.klent.gamma,
                max_turns=config.max_turns,
                onnx_device=config.onnx_device,
                onnx_max_batch_size=batch_size,
                rust_self_play_batch_size=batch_size,
                model_version=iteration,
                created_iteration=iteration,
                episode_id_offset=game_offset,
            )
        )
        episodes.extend(summary.trajectory_episodes)
        transition_count += summary.transitions
        game_offset += batch_size
    return episodes


def _klent_batch_from_indexes(
    dataset: KlentReplayDataset,
    indexes: np.ndarray,
    config: KlentTrainConfig,
    augment_rng: random.Random,
) -> TrainingBatch:
    batch = dataset.arrays_for_indexes(indexes)
    arrays = TrainingArrays(
        features=batch.features,
        policies=batch.policies,
        values=batch.values,
        sample_weights=batch.sample_weights,
        legal_masks=batch.legal_masks,
        indexes=batch.indexes,
        actions=batch.actions,
    )
    if config.symmetry_augmentation:
        features, policies, legal_masks, _terminal, actions = augment_training_arrays_randomly(
            arrays.features,
            arrays.policies,
            arrays.legal_masks,
            augment_rng,
            actions=arrays.actions,
        )
        arrays = replace(
            arrays,
            features=features,
            policies=policies,
            legal_masks=legal_masks,
            actions=actions,
        )
    return arrays_to_batch(
        arrays,
        device=config.device,
        pin_memory=str(config.device).startswith("cuda"),
    )


def _initial_state(config: KlentTrainConfig, *, resume: bool) -> KlentTrainState:
    torch = _import_torch()
    from great_kingdom_ai.model import create_model
    from great_kingdom_ai.training.checkpoint import create_optimizer
    from great_kingdom_ai.training.config import TrainingConfig

    latest = _latest_checkpoint_path(config.work_dir)
    if resume and latest.exists():
        state = load_klent_checkpoint(
            latest,
            device=config.device,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            optimizer=config.optimizer,
        )
        if state.klent_config != config.klent:
            raise ValueError("resume KLENT config does not match the checkpoint")
        return state

    if config.warm_start_checkpoint is not None:
        model = warm_start_klent_model(
            config.warm_start_checkpoint,
            model_preset=config.model_preset,
            device=config.device,
        )
        if not model.has_action_value_head:
            raise ValueError("KLENT warm start requires action_value_head=True")
    else:
        model = create_model(config.model_preset)
        if not model.has_action_value_head:
            raise ValueError(
                f"KLENT model preset {config.model_preset!r} must enable action_value_head"
            )
        model = model.to(config.device)

    optimizer = create_optimizer(
        torch,
        model,
        TrainingConfig(
            optimizer=config.optimizer,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            device=config.device,
        ),
    )
    return KlentTrainState(
        model=model,
        optimizer=optimizer,
        iteration=0,
        total_steps=0,
        klent_config=config.klent,
        model_preset=config.model_preset,
    )


def _iteration_shard_path(work_dir: Path, iteration: int) -> Path:
    return work_dir / "iterations" / f"iteration-{iteration:04d}.npz"


def _iteration_checkpoint_path(work_dir: Path, iteration: int) -> Path:
    return work_dir / "checkpoints" / f"iteration-{iteration:04d}.pt"


def _latest_checkpoint_path(work_dir: Path) -> Path:
    return work_dir / "checkpoints" / "latest.pt"


def _onnx_version_dir(work_dir: Path, model_version: int) -> Path:
    return work_dir / "onnx" / f"version-{model_version:05d}"


__all__ = [
    "KlentIterationSummary",
    "KlentTrainConfig",
    "compute_iteration_loss",
    "fit_klent_model",
    "run_klent_iteration",
    "run_klent_training",
]