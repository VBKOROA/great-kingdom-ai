"""Training package public API."""

from great_kingdom_ai.training.batch import (
    ReplayDataset,
    TrainingArrays,
    TrainingBatch,
    arrays_to_batch,
    samples_to_batch,
)
from great_kingdom_ai.training.checkpoint import (
    TrainState,
    create_lr_scheduler,
    create_train_state,
    load_checkpoint,
    load_checkpoint_weights,
    save_checkpoint,
    summarize_checkpoint_optimizer_state,
    summarize_optimizer_state_dict,
)
from great_kingdom_ai.training.cli import (
    build_parser,
    main,
    print_training_startup_config,
)
from great_kingdom_ai.training.config import TrainingConfig, load_training_config
from great_kingdom_ai.training.loop import (
    LossBreakdown,
    TrainSummary,
    compute_losses,
    train_from_replay,
    train_step,
)

__all__ = [
    "LossBreakdown",
    "ReplayDataset",
    "TrainState",
    "TrainSummary",
    "TrainingArrays",
    "TrainingBatch",
    "TrainingConfig",
    "arrays_to_batch",
    "build_parser",
    "compute_losses",
    "create_lr_scheduler",
    "create_train_state",
    "load_checkpoint",
    "load_checkpoint_weights",
    "load_training_config",
    "main",
    "print_training_startup_config",
    "samples_to_batch",
    "save_checkpoint",
    "summarize_checkpoint_optimizer_state",
    "summarize_optimizer_state_dict",
    "train_from_replay",
    "train_step",
]
