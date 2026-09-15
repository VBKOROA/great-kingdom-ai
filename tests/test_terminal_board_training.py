from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
from great_kingdom_ai.features import ACTION_SPACE, BOARD_SIZE, FEATURE_CHANNELS
from great_kingdom_ai.model import MODEL_PRESETS, ModelConfig, create_model
from great_kingdom_ai.onnx_export import export_checkpoint_to_onnx
from great_kingdom_ai.training.batch import TrainingBatch
from great_kingdom_ai.training.checkpoint import (
    create_train_state,
    load_checkpoint,
    save_checkpoint,
    warm_start_terminal_board_head,
)
from great_kingdom_ai.training.config import TrainingConfig
from great_kingdom_ai.training.loop import _terminal_board_aux_loss, compute_losses


def make_batch(
    batch_size: int = 3,
    *,
    valid: list[bool] | None = None,
    sample_weights: list[float] | None = None,
    targets: torch.Tensor | None = None,
) -> TrainingBatch:
    torch.manual_seed(0)
    valid_tensor = torch.as_tensor(
        valid if valid is not None else [True] * batch_size,
        dtype=torch.bool,
    )
    weights = torch.as_tensor(
        sample_weights if sample_weights is not None else [1.0] * batch_size,
        dtype=torch.float32,
    )
    if targets is None:
        targets = torch.zeros(batch_size, BOARD_SIZE, BOARD_SIZE, dtype=torch.long)
    return TrainingBatch(
        features=torch.randn(batch_size, FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE),
        policy=torch.softmax(torch.randn(batch_size, ACTION_SPACE), dim=1),
        value=torch.zeros(batch_size),
        legal_mask=torch.ones(batch_size, ACTION_SPACE, dtype=torch.bool),
        sample_weight=weights,
        terminal_board_target=targets,
        terminal_board_valid=valid_tensor,
    )


def test_replay_dataset_targets_reach_training_batch() -> None:
    import random

    from great_kingdom_ai.features import PASS_ACTION
    from great_kingdom_ai.replay import (
        TrajectoryEpisode,
        TrajectoryReplayDataset,
        TrajectoryReplayStore,
        TrajectoryTransition,
        legal_mask_from_features,
    )
    from great_kingdom_ai.training.batch import TrainingArrays, arrays_to_batch

    features = np.zeros((FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    features[4, :, :] = 1.0
    policy = np.zeros(ACTION_SPACE, dtype=np.float32)
    policy[PASS_ACTION] = 1.0
    transition = TrajectoryTransition(
        episode_id=0,
        timestep=0,
        player=1,
        features=features,
        legal_mask=legal_mask_from_features(features),
        action=PASS_ACTION,
        policy_target=policy,
        winner=1,
        terminal=True,
    )
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.uint8)
    board[0, 0] = 1
    board[1, 1] = 2
    episode = TrajectoryEpisode(
        episode_id=0,
        seed=0,
        transitions=(transition,),
        winner=1,
        end_reason=3,
        territory_scores=(1, 0),
        terminal_board=board,
    )
    store = TrajectoryReplayStore.from_episodes(4, (episode,))
    arrays = TrajectoryReplayDataset(store).sample_arrays(1, random.Random(0))
    training_arrays = TrainingArrays(
        features=arrays.features,
        policies=arrays.policies,
        values=arrays.values,
        sample_weights=arrays.sample_weights,
        legal_masks=arrays.legal_masks,
        indexes=arrays.indexes,
        terminal_board_targets=arrays.terminal_board_targets,
        terminal_board_valid=arrays.terminal_board_valid,
    )
    batch = arrays_to_batch(training_arrays)
    assert batch.terminal_board_target is not None
    assert batch.terminal_board_valid is not None
    assert bool(batch.terminal_board_valid.all())

    model = create_model("small", terminal_board_head=True)
    losses = compute_losses(
        model,
        batch,
        policy_loss_weight=0.0,
        value_loss_weight=0.0,
        terminal_board_loss_weight=0.1,
    )
    assert losses.terminal_board_valid_count == 1


def test_terminal_board_preset_enables_head() -> None:
    model = create_model("strong_attn_terminal_board")
    assert model.has_terminal_board_head
    policy, value = model(torch.zeros(1, FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE))
    assert policy.shape == (1, ACTION_SPACE)
    assert value.shape == (1,)
    assert MODEL_PRESETS["strong_attn_terminal_board"].terminal_board_head is True
    assert MODEL_PRESETS["strong_attn_terminal_board"].channels == MODEL_PRESETS["strong_attn"].channels
    assert MODEL_PRESETS["strong_attn"].terminal_board_head is False


def test_forward_with_aux_returns_board_logits() -> None:
    model = create_model("small", terminal_board_head=True)
    policy, value, aux = model.forward_with_aux(
        torch.zeros(2, FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    )
    assert policy.shape == (2, ACTION_SPACE)
    assert value.shape == (2,)
    assert aux.shape == (2, 4, BOARD_SIZE, BOARD_SIZE)


def test_aux_loss_matches_hand_computed_weighted_mean() -> None:
    torch_module = torch
    aux_logits = torch.zeros(2, 4, BOARD_SIZE, BOARD_SIZE)
    targets = torch.zeros(2, BOARD_SIZE, BOARD_SIZE, dtype=torch.long)
    log4 = math.log(4.0)
    batch = make_batch(
        2,
        valid=[True, True],
        sample_weights=[1.0, 3.0],
        targets=targets,
    )
    loss, stats = _terminal_board_aux_loss(
        torch_module,
        aux_logits=aux_logits,
        batch=batch,
        enabled=True,
    )
    assert loss is not None
    assert float(loss) == pytest.approx(log4)
    assert stats["terminal_board_valid_count"] == 2

    # A one-hot logit for sample 0 makes its cell loss exactly zero.
    aux_logits = torch.full((2, 4, BOARD_SIZE, BOARD_SIZE), -20.0)
    aux_logits[0, 0] = 20.0
    loss, _stats = _terminal_board_aux_loss(
        torch_module,
        aux_logits=aux_logits,
        batch=batch,
        enabled=True,
    )
    expected = (0.0 * 1.0 + log4 * 3.0) / 4.0
    assert loss is not None
    assert float(loss) == pytest.approx(expected)


def test_aux_loss_with_no_valid_targets_is_zero() -> None:
    batch = make_batch(3, valid=[False, False, False])
    loss, stats = _terminal_board_aux_loss(
        torch,
        aux_logits=torch.zeros(3, 4, BOARD_SIZE, BOARD_SIZE),
        batch=batch,
        enabled=True,
    )
    assert loss is not None
    assert float(loss) == 0.0
    assert stats["terminal_board_valid_count"] == 0


def test_lambda_zero_keeps_existing_loss_and_skips_head() -> None:
    model = create_model("small")
    batch = make_batch(2, targets=torch.ones(2, BOARD_SIZE, BOARD_SIZE, dtype=torch.long))
    losses = compute_losses(
        model,
        batch,
        policy_loss_weight=0.0,
        value_loss_weight=0.0,
        terminal_board_loss_weight=0.0,
    )
    assert "terminal_board_loss" not in losses.to_float_dict()


def test_aux_loss_requires_model_head() -> None:
    model = create_model("small")
    batch = make_batch(2)
    with pytest.raises(ValueError, match="terminal board head"):
        compute_losses(model, batch, terminal_board_loss_weight=0.1)


def test_negative_or_nonfinite_weight_is_rejected() -> None:
    model = create_model("small", terminal_board_head=True)
    batch = make_batch(2)
    with pytest.raises(ValueError, match="non-negative"):
        compute_losses(model, batch, terminal_board_loss_weight=-1.0)
    with pytest.raises(ValueError, match="finite"):
        compute_losses(model, batch, terminal_board_loss_weight=float("nan"))


def test_aux_loss_backward_reaches_backbone_and_head() -> None:
    model = create_model("small", terminal_board_head=True)
    batch = make_batch(2)
    model.zero_grad()
    losses = compute_losses(
        model,
        batch,
        policy_loss_weight=0.0,
        value_loss_weight=0.0,
        terminal_board_loss_weight=1.0,
    )
    losses.total.backward()
    assert model.terminal_board_head is not None
    head_grad = model.terminal_board_head[0].weight.grad
    assert head_grad is not None and float(head_grad.abs().sum()) > 0.0
    backbone_grad = sum(
        float(parameter.grad.abs().sum())
        for parameter in model.backbone.parameters()
        if parameter.grad is not None
    )
    assert backbone_grad > 0.0


def test_aux_loss_decreases_on_fixed_batch() -> None:
    torch.manual_seed(0)
    model = create_model("small", terminal_board_head=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    targets = torch.zeros(4, BOARD_SIZE, BOARD_SIZE, dtype=torch.long)
    targets[:, 0, 0] = 1
    batch = make_batch(4, targets=targets)
    first = None
    last = None
    for _ in range(40):
        optimizer.zero_grad()
        losses = compute_losses(
            model,
            batch,
            policy_loss_weight=0.0,
            value_loss_weight=0.0,
            terminal_board_loss_weight=1.0,
        )
        losses.total.backward()
        optimizer.step()
        if first is None:
            first = float(losses.terminal_board_loss)
        last = float(losses.terminal_board_loss)
    assert first is not None and last is not None
    assert last < first


def test_checkpoint_old_load_disable_head_and_warm_start(tmp_path: Path) -> None:
    config = TrainingConfig(model_preset="strong_attn", device="cpu", ema_decay=0.9)
    state = create_train_state(config)
    base = tmp_path / "base.pt"
    save_checkpoint(state, base)

    loaded = load_checkpoint(base, device="cpu", ema_decay=0.9)
    assert not loaded.model.has_terminal_board_head

    warm = warm_start_terminal_board_head(base, config)
    assert warm.model.has_terminal_board_head
    assert warm.step == 0
    assert warm.ema_model is not None and warm.ema_model.has_terminal_board_head
    source_state = state.model.state_dict()
    warm_state = warm.model.state_dict()
    for key, value in source_state.items():
        assert torch.equal(value, warm_state[key]), key


def test_aux_checkpoint_resume_and_ema_round_trip(tmp_path: Path) -> None:
    config = TrainingConfig(model_preset="strong_attn", device="cpu", ema_decay=0.9)
    state = create_train_state(config)
    base = tmp_path / "base.pt"
    save_checkpoint(state, base)
    warm = warm_start_terminal_board_head(base, config)
    path = tmp_path / "aux.pt"
    save_checkpoint(warm, path)

    resumed = load_checkpoint(path, device="cpu", ema_decay=0.9)
    assert resumed.model.has_terminal_board_head
    assert resumed.ema_model is not None and resumed.ema_model.has_terminal_board_head


def test_warm_start_rejects_shape_mismatch(tmp_path: Path) -> None:
    config = TrainingConfig(model_preset="small", device="cpu")
    state = create_train_state(config)
    base = tmp_path / "base.pt"
    save_checkpoint(state, base)

    checkpoint = torch.load(base, map_location="cpu", weights_only=False)
    del checkpoint["model_state"]["stem.0.weight"]
    broken = tmp_path / "broken.pt"
    torch.save(checkpoint, broken)
    with pytest.raises(ValueError, match="warm-start weight mismatch"):
        warm_start_terminal_board_head(broken, config)


def test_aux_model_onnx_export_excludes_head(tmp_path: Path) -> None:
    onnx = pytest.importorskip("onnx")
    config = TrainingConfig(model_preset="strong_attn", device="cpu")
    state = create_train_state(config)
    base = tmp_path / "base.pt"
    save_checkpoint(state, base)
    warm = warm_start_terminal_board_head(base, config)
    checkpoint = tmp_path / "aux.pt"
    save_checkpoint(warm, checkpoint)

    output = tmp_path / "aux.onnx"
    export_checkpoint_to_onnx(checkpoint, output, prefer_ema=False)
    model = onnx.load(str(output))
    assert [item.name for item in model.graph.output] == ["policy_logits", "value"]
    for node in model.graph.node:
        assert "terminal_board" not in node.name.lower()
        assert "aux" not in node.name.lower()
    for initializer in model.graph.initializer:
        assert "terminal_board" not in initializer.name.lower()


def test_model_config_defaults_disable_head() -> None:
    assert ModelConfig().terminal_board_head is False
