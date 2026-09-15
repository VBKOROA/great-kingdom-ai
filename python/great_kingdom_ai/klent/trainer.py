"""Synchronous KLENT collect/train iteration loop (paper Algorithm 1).

Each iteration freezes theta_k, collects a fresh whole-game buffer with
zero-search self-play, fits only on that buffer, writes the iteration
checkpoint, publishes actor/eval ONNX, and only then advances the latest
pointer. Resume recovers iterations whose training finished but whose ONNX
publication did not, instead of retraining or mixing partial state.
"""

from __future__ import annotations

import json
import random
import uuid
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from great_kingdom_ai.augmentation import augment_training_arrays_randomly
from great_kingdom_ai.klent._torch import _import_torch
from great_kingdom_ai.klent.checkpoint import (
    KlentTrainState,
    load_klent_checkpoint,
    read_klent_checkpoint_run_id,
    save_klent_checkpoint,
    warm_start_klent_model,
)
from great_kingdom_ai.klent.dataset import KlentReplayDataset
from great_kingdom_ai.klent.loss import KlentLossBreakdown, compute_klent_losses
from great_kingdom_ai.klent.publish import (
    load_klent_onnx_pointer,
    publish_klent_onnx_artifacts,
)
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
from great_kingdom_ai.training.torch_utils import _autocast_context, _cuda_amp_enabled

if TYPE_CHECKING:
    from torch import nn
    from torch.optim import Optimizer

    from great_kingdom_ai.replay import TrajectoryEpisode

_ACTOR_OVERRIDE_TOLERANCE = 1e-2


@dataclass(frozen=True)
class KlentTrainConfig:
    work_dir: Path = Path("data/klent")
    klent: KlentConfig = KlentConfig()
    model_preset: str = "strong_attn_klent"
    device: str = "cpu"
    seed: int = 0
    min_transitions: int = 4096
    max_games_per_iteration: int = 256
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
        scaler=state.scaler,
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
        scaler=state.scaler,
        run_id=state.run_id,
    )
    checkpoint_path = _iteration_checkpoint_path(config.work_dir, next_iteration)
    save_klent_checkpoint(next_state, checkpoint_path)
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
    _publish_latest_checkpoint(config.work_dir, next_iteration)
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
    scaler: Any | None = None,
) -> tuple[list[float], int]:
    """Fit for ``fit_epochs`` shuffled passes over the frozen iteration buffer."""
    torch = _import_torch()
    amp_enabled = _cuda_amp_enabled(torch, config.device, enabled=config.amp)
    scaler = scaler if amp_enabled else None
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
            with _autocast_context(torch, enabled=amp_enabled):
                losses = compute_klent_losses(model, batch, config.klent)
            if scaler is not None:
                scaler.scale(losses.total).backward()
                if config.gradient_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.gradient_clip_norm
                    )
                scaler.step(optimizer)
                scaler.update()
            else:
                losses.total.backward()  # type: ignore[no-untyped-call]
                if config.gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.gradient_clip_norm
                    )
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
    while transition_count < config.min_transitions:
        if len(episodes) >= config.max_games_per_iteration:
            raise RuntimeError(
                f"collected {transition_count} transitions in {len(episodes)} games "
                f"before reaching min_transitions={config.min_transitions}; "
                "increase max_games_per_iteration or lower min_transitions"
            )
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

    actor_path = _actor_onnx_for_iteration(
        state,
        config,
        iteration,
        export_klent_checkpoint_to_onnx,
    )

    episodes: list[TrajectoryEpisode] = []
    transition_count = 0
    seed_base = config.seed + iteration * 1_000_003
    game_offset = 0
    while transition_count < config.min_transitions:
        if game_offset >= config.max_games_per_iteration:
            raise RuntimeError(
                f"collected {transition_count} transitions in {game_offset} games "
                f"before reaching min_transitions={config.min_transitions}; "
                "increase max_games_per_iteration or lower min_transitions"
            )
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


def _actor_onnx_for_iteration(
    state: KlentTrainState,
    config: KlentTrainConfig,
    iteration: int,
    export_checkpoint_to_onnx: Callable[..., Any],
) -> Path:
    """Return the actor model that matches the frozen iteration model.

    ``actor_onnx_path`` is only an initial (iteration 0) override, and it is
    used only after its outputs are verified against the learner checkpoint so
    the shard records and the collected behavior cannot diverge. Later
    iterations always export an actor from their own frozen checkpoint.
    """
    if config.actor_onnx_path is not None and iteration == 0:
        provided = config.actor_onnx_path
        if not provided.exists():
            raise FileNotFoundError(f"actor ONNX model is missing: {provided}")
        source_checkpoint = _actor_source_checkpoint_path(config.work_dir, iteration)
        save_klent_checkpoint(state, source_checkpoint)
        _verify_actor_override(source_checkpoint, provided)
        return provided

    source_checkpoint = _actor_source_checkpoint_path(config.work_dir, iteration)
    save_klent_checkpoint(state, source_checkpoint)
    actor_path = config.work_dir / "onnx" / f"actor-source-{iteration:04d}.onnx"
    export_checkpoint_to_onnx(
        source_checkpoint,
        actor_path,
        kind="actor",
        device=config.onnx_device,
        precision=config.onnx_precision,
    )
    return actor_path


def _verify_actor_override(checkpoint_path: Path, actor_onnx_path: Path) -> None:
    from great_kingdom_ai.klent.export import compare_klent_checkpoint_to_onnx

    try:
        summary = compare_klent_checkpoint_to_onnx(
            checkpoint_path,
            actor_onnx_path,
            kind="actor",
            tolerance=_ACTOR_OVERRIDE_TOLERANCE,
        )
    except ValueError as error:
        raise RuntimeError(
            f"actor_onnx_path {actor_onnx_path} cannot be compared with the "
            f"iteration 0 learner checkpoint: {error}"
        ) from error
    if summary.passed:
        return
    raise RuntimeError(
        f"actor_onnx_path {actor_onnx_path} does not match the iteration 0 "
        f"learner checkpoint (policy diff {summary.max_policy_abs_diff:.3g}, "
        f"value diff {summary.max_value_abs_diff:.3g}); start the learner from "
        "the matching warm_start_checkpoint or drop actor_onnx_path"
    )


def _actor_source_checkpoint_path(work_dir: Path, iteration: int) -> Path:
    return work_dir / "checkpoints" / f"actor-source-{iteration:04d}.pt"


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

    run_id = _resolve_run_id(config, resume=resume)
    if resume:
        recovered = _resume_from_iteration_checkpoint(config, run_id)
        if recovered is not None:
            return recovered
        latest = _latest_checkpoint_path(config.work_dir)
        if latest.exists() and _readable_checkpoint_run_id(latest) == run_id:
            state = _load_state(latest, config)
            _ensure_published(config, state, source_path=latest)
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
    scaler: Any | None = None
    if _cuda_amp_enabled(torch, config.device, enabled=config.amp):
        scaler = torch.amp.GradScaler("cuda", enabled=True)
    return KlentTrainState(
        model=model,
        optimizer=optimizer,
        iteration=0,
        total_steps=0,
        klent_config=config.klent,
        model_preset=config.model_preset,
        scaler=scaler,
        run_id=run_id,
    )


def _load_state(path: Path, config: KlentTrainConfig) -> KlentTrainState:
    state = load_klent_checkpoint(
        path,
        device=config.device,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        optimizer=config.optimizer,
        amp=config.amp,
    )
    if state.klent_config != config.klent:
        raise ValueError("resume KLENT config does not match the checkpoint")
    return state


def _resume_from_iteration_checkpoint(
    config: KlentTrainConfig,
    run_id: str,
) -> KlentTrainState | None:
    """Resume from the newest usable iteration checkpoint of the active run.

    Iteration checkpoints are written before ONNX publication, so a crash
    between training and publication is recovered by publishing from the
    existing checkpoint instead of retraining the iteration. Checkpoints from
    other runs and interrupted (unreadable) files are skipped.
    """
    for iteration, path in reversed(_iteration_checkpoints(config.work_dir)):
        if _readable_checkpoint_run_id(path) != run_id:
            continue
        state = _load_state(path, config)
        if state.iteration != iteration:
            raise ValueError(
                f"iteration checkpoint {path} stores iteration {state.iteration} "
                f"but is named for {iteration}"
            )
        _ensure_published(config, state)
        _publish_latest_checkpoint(config.work_dir, state.iteration)
        return state
    return None


def _ensure_published(
    config: KlentTrainConfig,
    state: KlentTrainState,
    *,
    source_path: Path | None = None,
) -> None:
    if not config.export_onnx:
        return
    try:
        pointer = load_klent_onnx_pointer(config.work_dir)
    except ValueError:
        pointer = None
    if pointer is not None and pointer.model_version >= state.iteration:
        return
    checkpoint_path = (
        source_path
        if source_path is not None
        else _iteration_checkpoint_path(config.work_dir, state.iteration)
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"KLENT checkpoint for publication is missing: {checkpoint_path}")
    publish_klent_onnx_artifacts(
        checkpoint_path,
        config.work_dir,
        model_version=state.iteration,
        iteration=state.iteration - 1,
        klent_config=config.klent,
        model_preset=config.model_preset,
        device=config.onnx_device,
        precision=config.onnx_precision,
        check_parity=config.check_onnx_parity,
        overwrite=True,
    )


def _publish_latest_checkpoint(work_dir: Path, iteration: int) -> None:
    checkpoint_path = _iteration_checkpoint_path(work_dir, iteration)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"iteration checkpoint is missing: {checkpoint_path}")
    copy_file_atomic(checkpoint_path, _latest_checkpoint_path(work_dir))


def _resolve_run_id(config: KlentTrainConfig, *, resume: bool) -> str:
    """Return the active run id, rotating it for fresh runs.

    Iteration checkpoints embed the run id so a ``--no-resume`` restart cannot
    later resume into the previous run's checkpoints.
    """
    marker = _run_marker_path(config.work_dir)
    if resume and marker.exists():
        return _read_run_id(marker)
    if resume:
        run_id = _newest_known_run_id(config.work_dir)
    else:
        run_id = uuid.uuid4().hex
    _write_run_id(marker, run_id)
    return run_id


def _newest_known_run_id(work_dir: Path) -> str:
    """Adopt the run id of an existing directory that predates run markers."""
    candidates = [path for _iteration, path in reversed(_iteration_checkpoints(work_dir))]
    candidates.append(_latest_checkpoint_path(work_dir))
    for path in candidates:
        if not path.exists():
            continue
        run_id = _readable_checkpoint_run_id(path)
        if run_id is not None:
            return run_id
    return ""


def _readable_checkpoint_run_id(path: Path) -> str | None:
    try:
        return read_klent_checkpoint_run_id(path)
    except Exception:
        return None


def _run_marker_path(work_dir: Path) -> Path:
    return work_dir / "checkpoints" / "run.json"


def _read_run_id(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "run_id" not in data:
        raise ValueError(f"KLENT run marker is malformed: {path}")
    return str(data["run_id"])


def _write_run_id(path: Path, run_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps({"run_id": run_id}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _iteration_checkpoints(work_dir: Path) -> list[tuple[int, Path]]:
    directory = work_dir / "checkpoints"
    if not directory.exists():
        return []
    checkpoints: list[tuple[int, Path]] = []
    for path in directory.glob("iteration-*.pt"):
        suffix = path.stem.removeprefix("iteration-")
        if not suffix.isdigit():
            continue
        checkpoints.append((int(suffix), path))
    checkpoints.sort()
    return checkpoints


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