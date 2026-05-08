from __future__ import annotations

from great_kingdom_ai.evaluate import load_arena_config
from great_kingdom_ai.rust_onnx_pipeline import load_rust_onnx_pipeline_config
from great_kingdom_ai.train import load_training_config


def test_runpod_low_time_high_quality_configs_are_loadable() -> None:
    pipeline = load_rust_onnx_pipeline_config("configs/runpod/pipeline.json")
    train = load_training_config("configs/runpod/train.json")
    arena = load_arena_config("configs/runpod/arena.json")

    assert pipeline.onnx_device == "cuda"
    assert pipeline.aggregate_replay is True
    assert pipeline.aggregate_replay_weight_mode == "log_count"
    assert pipeline.aggregate_replay_weight_cap is None
    assert pipeline.skip_arena is True
    assert pipeline.always_promote is True
    assert pipeline.train_checkpoint_mode == "resume"
    assert pipeline.self_play_games == 1500
    assert pipeline.min_replay_samples == 30000
    assert pipeline.self_play.max_turns == 88
    assert pipeline.self_play.gumbel_simulations == 48
    assert pipeline.self_play.playout_cap_full_simulations == 48
    assert pipeline.self_play.leaf_batch_size == 128
    assert pipeline.self_play.policy_target_c_visit == pipeline.self_play.gumbel_c_visit
    assert pipeline.self_play.policy_target_c_scale == pipeline.self_play.gumbel_c_scale
    assert pipeline.self_play.policy_target_temperature == 1.0

    assert train.device == "cuda"
    assert train.model_preset == "medium_plus"
    assert train.steps == 768
    assert train.learning_rate == 1e-4
    assert train.lr_schedule == "constant_with_warmup"
    assert train.lr_warmup_steps == 32
    assert train.amp is True
    assert train.value_loss_weight == 0.5
    assert train.recent_sample_fraction == 0.5
    assert train.batch_size <= pipeline.min_replay_samples
    assert train.batch_size * train.steps >= pipeline.min_replay_samples * 12

    assert arena.device == "cuda"
    assert arena.max_turns == pipeline.self_play.max_turns
    assert arena.policy_target_c_visit == arena.gumbel_c_visit
    assert arena.policy_target_c_scale == arena.gumbel_c_scale
    assert arena.policy_target_temperature == 1.0
