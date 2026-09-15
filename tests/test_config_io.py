from __future__ import annotations

import json
from pathlib import Path

from great_kingdom_ai.async_v2 import load_actor_v2_config, load_learner_v2_config
from great_kingdom_ai.evaluate import load_arena_config
from great_kingdom_ai.training import load_training_config


def test_runpod_yaml_configs_load_with_existing_values() -> None:
    actor = load_actor_v2_config("configs/runpod/actor-v2.yaml")
    learner = load_learner_v2_config("configs/runpod/learner-v2.yaml")
    training = load_training_config("configs/runpod/train.yaml")

    assert actor.work_dir == Path("data/runpod/train-strong-attn")
    assert actor.ema_opponent_fraction == 0.25
    assert actor.self_play.gumbel_simulations == 64
    assert actor.self_play.playout_cap_fast_simulations == 32

    assert learner.work_dir == actor.work_dir
    assert learner.source_checkpoint is None
    assert learner.replay_capacity == 1_048_576
    assert learner.train_reuse_factor == 8.0

    assert training.model_preset == "strong_attn"
    assert training.batch_size == 512
    assert training.priority_enabled is False
    assert training.ema_decay == 0.999


def test_json_loading_remains_available_for_development_tests(tmp_path: Path) -> None:
    config_path = tmp_path / "train.json"
    config_path.write_text(json.dumps({"batch_size": 7}), encoding="utf-8")

    assert load_training_config(config_path).batch_size == 7


def test_remaining_yaml_configs_load_with_existing_values() -> None:
    arena = load_arena_config(
        "configs/runpod/arena.yaml",
        randomize_missing_seed_start=False,
        randomize_missing_gumbel_seed=False,
    )
    fast_matrix = load_arena_config(
        "configs/runpod/fast-matrix.yaml",
        randomize_missing_seed_start=False,
        randomize_missing_gumbel_seed=False,
    )
    full_matrix = load_arena_config(
        "configs/runpod/full-matrix.yaml",
        randomize_missing_seed_start=False,
        randomize_missing_gumbel_seed=False,
    )
    smoke = load_training_config("configs/m8-train-smoke.yaml")

    assert (arena.games, arena.batch_size, arena.promotion_threshold) == (200, 200, 0.5)
    assert (fast_matrix.games, fast_matrix.batch_size) == (2, 2)
    assert (full_matrix.games, full_matrix.batch_size) == (60, 60)
    assert (smoke.batch_size, smoke.steps, smoke.device) == (16, 4, "cpu")
