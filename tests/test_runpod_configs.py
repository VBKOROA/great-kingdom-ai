from __future__ import annotations

from pathlib import Path

from great_kingdom_ai.evaluate import load_arena_config
from great_kingdom_ai.rust_onnx_pipeline import load_rust_onnx_pipeline_config
from great_kingdom_ai.train import load_training_config


def test_runpod_pure_gumbel_configs_are_loadable_and_pure() -> None:
    pipeline = load_rust_onnx_pipeline_config(
        Path("configs/runpod/pure-gumbel-pipeline.json")
    )
    train = load_training_config(Path("configs/runpod/pure-gumbel-train.json"))
    arena = load_arena_config(Path("configs/runpod/pure-gumbel-arena.json"))

    assert pipeline.onnx_device == "cuda"
    assert pipeline.aggregate_replay is True
    assert pipeline.aggregate_replay_weight_mode == "sqrt_count"
    assert pipeline.aggregate_replay_weight_cap == 8.0
    assert pipeline.self_play.policy_target_c_visit == pipeline.self_play.gumbel_c_visit
    assert pipeline.self_play.policy_target_c_scale == pipeline.self_play.gumbel_c_scale
    assert pipeline.self_play.policy_target_temperature == 1.0

    assert train.device == "cuda"
    assert train.model_preset == "medium"
    assert train.batch_size <= pipeline.min_replay_samples

    assert arena.device == "cuda"
    assert arena.policy_target_c_visit == arena.gumbel_c_visit
    assert arena.policy_target_c_scale == arena.gumbel_c_scale
    assert arena.policy_target_temperature == 1.0
