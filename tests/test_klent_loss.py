from __future__ import annotations

import importlib.util

import numpy as np
import pytest

_torch_spec = importlib.util.find_spec("torch")
pytestmark = pytest.mark.skipif(
    _torch_spec is None,
    reason="torch is not installed",
)
torch = importlib.import_module("torch") if _torch_spec is not None else None

from great_kingdom_ai.features import BOARD_SIZE, FEATURE_CHANNELS  # noqa: E402
from great_kingdom_ai.klent.loss import compute_klent_losses  # noqa: E402
from great_kingdom_ai.klent.types import KlentConfig  # noqa: E402
from great_kingdom_ai.training.batch import TrainingBatch  # noqa: E402


class _ConstantQModel(torch.nn.Module):
    def __init__(self, logits: torch.Tensor, q_values: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("_logits", logits.detach().clone().reshape(-1))
        self.register_buffer("_q_values", q_values.detach().clone().reshape(-1))

    def forward_q(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = features.shape[0]
        return (
            self._logits.unsqueeze(0).expand(batch_size, -1).clone(),
            self._q_values.unsqueeze(0).expand(batch_size, -1).clone(),
        )


class _TinyQModel(torch.nn.Module):
    def __init__(self, action_space: int) -> None:
        super().__init__()
        self.action_space = action_space
        self.linear = torch.nn.Linear(
            FEATURE_CHANNELS * BOARD_SIZE * BOARD_SIZE,
            action_space * 2,
        )

    def forward_q(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flat = features.flatten(start_dim=1)
        output = self.linear(flat)
        return output[:, : self.action_space], output[:, self.action_space :]


def _make_batch(
    action_space: int = 4,
    *,
    batch_size: int = 2,
    values: list[float] | None = None,
    actions: list[int] | None = None,
) -> TrainingBatch:
    features = torch.zeros(
        (batch_size, FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE),
        dtype=torch.float32,
    )
    policies = torch.full((batch_size, action_space), 1.0 / action_space, dtype=torch.float32)
    legal_mask = torch.ones((batch_size, action_space), dtype=torch.bool)
    return TrainingBatch(
        features=features,
        policy=policies,
        value=torch.tensor(
            values if values is not None else [0.5, -0.5],
            dtype=torch.float32,
        ),
        legal_mask=legal_mask,
        sample_weight=torch.ones((batch_size,), dtype=torch.float32),
        action=torch.tensor(
            actions if actions is not None else [0, 2],
            dtype=torch.long,
        ),
    )


def test_klent_losses_match_manual_reference() -> None:
    logits = torch.tensor([[1.0, 0.0, -1.0, 0.5]], dtype=torch.float32)
    q_values = torch.tensor([[0.2, -0.3, 0.4, 0.1]], dtype=torch.float32)
    model = _ConstantQModel(logits, q_values)
    batch = _make_batch(4, batch_size=2, values=[0.5, -0.5], actions=[2, 1])

    losses = compute_klent_losses(model, batch, KlentConfig())

    expected_log_policy = torch.log_softmax(logits, dim=1)
    expected_policy = -(batch.policy * expected_log_policy).sum(dim=1).mean()
    picked_q = q_values[0, torch.tensor([2, 1])]
    expected_q = (picked_q - torch.tensor([0.5, -0.5])).pow(2).mean()

    assert float(losses.policy) == pytest.approx(float(expected_policy), abs=1e-6)
    assert float(losses.q_value) == pytest.approx(float(expected_q), abs=1e-6)
    assert float(losses.total) == pytest.approx(
        float(expected_policy) + float(expected_q),
        abs=1e-6,
    )


def test_klent_loss_ignores_illegal_logits_and_q_values() -> None:
    base_logits = torch.tensor([[1.0, 0.0, -1.0, 0.5]], dtype=torch.float32)
    base_q = torch.tensor([[0.2, -0.3, 0.4, 0.1]], dtype=torch.float32)
    extreme_logits = torch.tensor([[1.0, 1e30, -1.0, 0.5]], dtype=torch.float32)
    extreme_q = torch.tensor([[0.2, 1e30, 0.4, 0.1]], dtype=torch.float32)
    batch = _make_batch(4, values=[0.5, -0.5], actions=[0, 2])
    batch = TrainingBatch(
        features=batch.features,
        policy=torch.tensor(
            [[0.4, 0.0, 0.4, 0.2], [0.4, 0.0, 0.4, 0.2]],
            dtype=torch.float32,
        ),
        value=batch.value,
        legal_mask=torch.tensor(
            [[True, False, True, True], [True, False, True, True]],
            dtype=torch.bool,
        ),
        sample_weight=batch.sample_weight,
        action=batch.action,
    )

    base = compute_klent_losses(_ConstantQModel(base_logits, base_q), batch, KlentConfig())
    extreme = compute_klent_losses(
        _ConstantQModel(extreme_logits, extreme_q),
        batch,
        KlentConfig(),
    )

    assert float(extreme.policy) == pytest.approx(float(base.policy), abs=1e-6)
    assert float(extreme.q_value) == pytest.approx(float(base.q_value), abs=1e-6)
    assert bool(torch.isfinite(extreme.total))


def test_klent_loss_requires_action_field() -> None:
    logits = torch.zeros((1, 3), dtype=torch.float32)
    q_values = torch.zeros((1, 3), dtype=torch.float32)
    batch = _make_batch(3, batch_size=1, values=[0.0], actions=[0])
    batch = TrainingBatch(
        features=batch.features,
        policy=batch.policy,
        value=batch.value,
        legal_mask=batch.legal_mask,
        sample_weight=batch.sample_weight,
        action=None,
    )

    with pytest.raises(ValueError, match="batch.action"):
        compute_klent_losses(_ConstantQModel(logits, q_values), batch, KlentConfig())


def test_klent_loss_rejects_illegal_taken_action() -> None:
    logits = torch.zeros((1, 3), dtype=torch.float32)
    q_values = torch.zeros((1, 3), dtype=torch.float32)
    batch = _make_batch(3, batch_size=1, values=[0.0], actions=[1])
    batch = TrainingBatch(
        features=batch.features,
        policy=torch.tensor([[0.5, 0.0, 0.5]], dtype=torch.float32),
        value=batch.value,
        legal_mask=torch.tensor([[True, False, True]], dtype=torch.bool),
        sample_weight=batch.sample_weight,
        action=batch.action,
    )

    with pytest.raises(ValueError, match="taken action must be legal"):
        compute_klent_losses(_ConstantQModel(logits, q_values), batch, KlentConfig())


def test_klent_loss_rejects_out_of_range_action() -> None:
    logits = torch.zeros((1, 3), dtype=torch.float32)
    q_values = torch.zeros((1, 3), dtype=torch.float32)
    batch = _make_batch(3, batch_size=1, values=[0.0], actions=[5])

    with pytest.raises(ValueError, match="action must be in"):
        compute_klent_losses(_ConstantQModel(logits, q_values), batch, KlentConfig())


@pytest.mark.parametrize("value", [1.5, -2.0, float("nan")])
def test_klent_loss_rejects_out_of_range_lambda_targets(value: float) -> None:
    logits = torch.zeros((1, 3), dtype=torch.float32)
    q_values = torch.zeros((1, 3), dtype=torch.float32)
    batch = _make_batch(3, batch_size=1, values=[value], actions=[0])

    with pytest.raises(ValueError, match="lambda return target"):
        compute_klent_losses(_ConstantQModel(logits, q_values), batch, KlentConfig())


def test_klent_loss_rejects_illegal_policy_target_mass() -> None:
    logits = torch.zeros((1, 3), dtype=torch.float32)
    q_values = torch.zeros((1, 3), dtype=torch.float32)
    batch = _make_batch(3, batch_size=1, values=[0.0], actions=[0])
    policy = torch.tensor([[0.5, 0.5, 0.0]], dtype=torch.float32)
    batch = TrainingBatch(
        features=batch.features,
        policy=policy,
        value=batch.value,
        legal_mask=torch.tensor([[True, False, True]], dtype=torch.bool),
        sample_weight=batch.sample_weight,
        action=batch.action,
    )

    with pytest.raises(ValueError, match="illegal actions"):
        compute_klent_losses(_ConstantQModel(logits, q_values), batch, KlentConfig())


def test_klent_loss_decreases_with_optimizer_steps() -> None:
    model = _TinyQModel(action_space=4)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    batch = _make_batch(
        4,
        batch_size=8,
        values=[0.8, -0.8, 0.4, -0.4, 0.2, -0.2, 0.0, 0.6],
        actions=[0, 1, 2, 3, 1, 0, 2, 3],
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    first = float(compute_klent_losses(model, batch, KlentConfig()).total.detach())
    for _ in range(20):
        optimizer.zero_grad(set_to_none=True)
        losses = compute_klent_losses(model, batch, KlentConfig())
        losses.total.backward()
        optimizer.step()
    last = float(compute_klent_losses(model, batch, KlentConfig()).total.detach())

    assert last < first
    assert np.isfinite(first) and np.isfinite(last)


def test_klent_soft_target_ce_stays_above_target_entropy() -> None:
    logits = torch.tensor([[1.0, 0.0, -1.0, 0.5]], dtype=torch.float32)
    q_values = torch.tensor([[0.2, -0.3, 0.4, 0.1]], dtype=torch.float32)
    model = _ConstantQModel(logits, q_values)
    batch = _make_batch(4, batch_size=2, values=[0.5, -0.5], actions=[0, 2])
    policy = torch.tensor(
        [[0.4, 0.3, 0.2, 0.1], [0.25, 0.25, 0.25, 0.25]],
        dtype=torch.float32,
    )
    batch = TrainingBatch(
        features=batch.features,
        policy=policy,
        value=batch.value,
        legal_mask=batch.legal_mask,
        sample_weight=batch.sample_weight,
        action=batch.action,
    )

    losses = compute_klent_losses(model, batch, KlentConfig())

    entropy = -(policy * torch.log(policy.clamp_min(1e-45))).sum(dim=1).mean()
    assert float(losses.policy) >= float(entropy) - 1e-6


def test_klent_loss_backpropagates_without_target_gradients() -> None:
    model = _TinyQModel(action_space=3)
    batch = _make_batch(3, batch_size=2, values=[0.5, -0.5], actions=[0, 1])

    losses = compute_klent_losses(model, batch, KlentConfig())
    losses.total.backward()

    gradients = [parameter.grad for parameter in model.parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
