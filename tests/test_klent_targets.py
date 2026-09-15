from __future__ import annotations

import importlib.util
import math

import numpy as np
import pytest

_torch_spec = importlib.util.find_spec("torch")
pytestmark = pytest.mark.skipif(
    _torch_spec is None,
    reason="torch is not installed",
)
torch = importlib.import_module("torch") if _torch_spec is not None else None

from great_kingdom_ai.features import (  # noqa: E402
    ACTION_SPACE,
    BOARD_CELLS,
    BOARD_SIZE,
    FEATURE_CHANNELS,
    PASS_ACTION,
)
from great_kingdom_ai.klent.targets import (  # noqa: E402
    compute_analytical_policy,
    compute_lambda_returns,
    legal_mask_from_features,
    masked_state_value,
)
from great_kingdom_ai.klent.types import KlentConfig, validate_alpha_beta  # noqa: E402


def _closed_form(
    logits: torch.Tensor,
    q_values: torch.Tensor,
    legal: torch.Tensor,
    alpha: float,
    beta: float,
) -> torch.Tensor:
    result = torch.zeros_like(logits, dtype=torch.float32)
    for row in range(logits.shape[0]):
        legal_indexes = torch.nonzero(legal[row], as_tuple=False).squeeze(1)
        legal_logits = logits[row, legal_indexes]
        log_policy = torch.log_softmax(legal_logits, dim=0)
        scores = (q_values[row, legal_indexes] + beta * log_policy) / (alpha + beta)
        result[row, legal_indexes] = torch.softmax(scores, dim=0)
    return result


def test_klent_config_defaults_match_plan() -> None:
    config = KlentConfig()

    assert config.alpha == 0.03
    assert config.beta == 0.1
    assert config.lambda_param == pytest.approx(0.8825)
    assert config.gamma == 1.0


@pytest.mark.parametrize(
    ("alpha", "beta"),
    [(-0.1, 0.1), (0.1, -0.1), (0.0, 0.0), (float("nan"), 0.1), (0.1, float("inf"))],
)
def test_klent_config_rejects_invalid_coefficients(alpha: float, beta: float) -> None:
    with pytest.raises(ValueError):
        KlentConfig(alpha=alpha, beta=beta)
    with pytest.raises(ValueError):
        validate_alpha_beta(alpha, beta)


@pytest.mark.parametrize("lambda_param", [-0.1, 1.1, float("nan")])
def test_klent_config_rejects_invalid_lambda(lambda_param: float) -> None:
    with pytest.raises(ValueError):
        KlentConfig(lambda_param=lambda_param)


def test_analytical_policy_matches_closed_form_reference() -> None:
    logits = torch.tensor([[1.0, -2.0, 0.5, 3.0]], dtype=torch.float32)
    q_values = torch.tensor([[0.2, 0.9, -0.4, 0.1]], dtype=torch.float32)
    legal = torch.tensor([[True, False, True, True]])

    target = compute_analytical_policy(logits, q_values, legal, alpha=0.03, beta=0.1)
    expected = _closed_form(logits, q_values, legal, alpha=0.03, beta=0.1)

    assert torch.allclose(target, expected, atol=1e-6)
    assert float(target[0, 1]) == 0.0
    assert float(target.sum()) == pytest.approx(1.0, abs=1e-6)


def test_analytical_policy_beta_zero_reduces_to_q_softmax() -> None:
    logits = torch.tensor([[1.0, -2.0, 0.5]], dtype=torch.float32)
    q_values = torch.tensor([[0.2, 0.9, -0.4]], dtype=torch.float32)
    legal = torch.tensor([[True, True, True]])

    target = compute_analytical_policy(logits, q_values, legal, alpha=0.5, beta=0.0)
    expected = torch.softmax(q_values / 0.5, dim=1)

    assert torch.allclose(target, expected, atol=1e-6)


def test_analytical_policy_alpha_zero_keeps_prior_term() -> None:
    logits = torch.tensor([[1.0, 0.0, -1.0]], dtype=torch.float32)
    q_values = torch.tensor([[0.2, 0.1, 0.0]], dtype=torch.float32)
    legal = torch.tensor([[True, True, True]])

    target = compute_analytical_policy(logits, q_values, legal, alpha=0.0, beta=1.0)
    expected = _closed_form(logits, q_values, legal, alpha=0.0, beta=1.0)

    assert torch.allclose(target, expected, atol=1e-6)


def test_analytical_policy_ignores_illegal_logits_and_q_values() -> None:
    base_logits = torch.tensor([[1.0, 0.0, 0.5]], dtype=torch.float32)
    base_q = torch.tensor([[0.2, 0.1, 0.3]], dtype=torch.float32)
    extreme_logits = torch.tensor([[1.0, 1e30, 0.5]], dtype=torch.float32)
    extreme_q = torch.tensor([[0.2, 1e30, 0.3]], dtype=torch.float32)
    legal = torch.tensor([[True, False, True]])

    base = compute_analytical_policy(base_logits, base_q, legal, alpha=0.03, beta=0.1)
    extreme = compute_analytical_policy(extreme_logits, extreme_q, legal, alpha=0.03, beta=0.1)

    assert torch.allclose(base, extreme, atol=1e-6)
    assert bool(torch.isfinite(extreme).all())


def test_analytical_policy_rejects_rows_without_legal_actions() -> None:
    logits = torch.zeros((1, 3), dtype=torch.float32)
    q_values = torch.zeros((1, 3), dtype=torch.float32)
    legal = torch.zeros((1, 3), dtype=torch.bool)

    with pytest.raises(ValueError, match="at least one legal action"):
        compute_analytical_policy(logits, q_values, legal, alpha=0.03, beta=0.1)


def test_analytical_policy_result_is_detached() -> None:
    logits = torch.randn(2, 4, requires_grad=True)
    q_values = torch.randn(2, 4, requires_grad=True)
    legal = torch.ones((2, 4), dtype=torch.bool)

    target = compute_analytical_policy(logits, q_values, legal, alpha=0.03, beta=0.1)

    assert target.requires_grad is False
    assert target.grad_fn is None


def test_lambda_returns_single_move_is_terminal_reward() -> None:
    returns = compute_lambda_returns(
        rewards=[1.0],
        state_values=[0.4],
        players=[2],
        lambda_param=0.5,
    )

    assert returns.shape == (1,)
    assert float(returns[0]) == 1.0


def test_lambda_returns_backward_recursion_flips_sign_between_players() -> None:
    returns = compute_lambda_returns(
        rewards=[0.0, 1.0],
        state_values=[0.5, 0.25],
        players=[1, 2],
        lambda_param=0.5,
    )

    expected_second = 1.0
    expected_first = 0.0 + 1.0 * (-1.0) * ((1.0 - 0.5) * 0.25 + 0.5 * 1.0)
    assert float(returns[1]) == pytest.approx(expected_second)
    assert float(returns[0]) == pytest.approx(expected_first)


def test_lambda_one_is_monte_carlo_return() -> None:
    returns = compute_lambda_returns(
        rewards=[0.0, 0.0, -1.0],
        state_values=[0.5, -0.5, 0.9],
        players=[1, 1, 1],
        lambda_param=1.0,
    )

    assert float(returns[2]) == -1.0
    assert float(returns[1]) == pytest.approx(-1.0)
    assert float(returns[0]) == pytest.approx(-1.0)


def test_lambda_zero_is_one_step_td_return() -> None:
    returns = compute_lambda_returns(
        rewards=[0.0, 0.0, 1.0],
        state_values=[0.5, 0.2, 0.9],
        players=[1, 1, 2],
        lambda_param=0.0,
    )

    assert float(returns[2]) == 1.0
    assert float(returns[1]) == pytest.approx(-0.9)
    assert float(returns[0]) == pytest.approx(0.2)


@pytest.mark.parametrize("gamma", [0.0, 0.5, 1.0])
def test_lambda_returns_respect_gamma_discount(gamma: float) -> None:
    returns = compute_lambda_returns(
        rewards=[0.0, 1.0],
        state_values=[0.5, 0.8],
        players=[1, 1],
        lambda_param=0.0,
        gamma=gamma,
    )

    assert float(returns[0]) == pytest.approx(gamma * 0.8)


def test_lambda_returns_are_bounded_for_random_inputs() -> None:
    generator = np.random.default_rng(0)
    for _ in range(20):
        length = int(generator.integers(1, 8))
        rewards = np.zeros(length, dtype=np.float32)
        rewards[-1] = 1.0 if generator.random() < 0.5 else -1.0
        values = generator.uniform(-1.0, 1.0, size=length).astype(np.float32)
        players = generator.integers(1, 3, size=length).astype(np.int64)
        returns = compute_lambda_returns(
            rewards,
            values,
            players,
            lambda_param=0.8825,
            gamma=1.0,
        )
        assert bool(torch.isfinite(returns).all())
        assert float(returns.min()) >= -1.0 - 1e-6
        assert float(returns.max()) <= 1.0 + 1e-6


def test_lambda_returns_reject_unfinished_game() -> None:
    with pytest.raises(ValueError, match="final reward"):
        compute_lambda_returns(
            rewards=[0.0, 0.0],
            state_values=[0.0, 0.0],
            players=[1, 2],
            lambda_param=0.5,
        )


def test_lambda_returns_reject_non_terminal_rewards() -> None:
    with pytest.raises(ValueError, match="non-terminal rewards"):
        compute_lambda_returns(
            rewards=[0.5, 1.0],
            state_values=[0.0, 0.0],
            players=[1, 2],
            lambda_param=0.5,
        )


@pytest.mark.parametrize("players", [[0, 1], [1, 3]])
def test_lambda_returns_reject_invalid_players(players: list[int]) -> None:
    with pytest.raises(ValueError, match="players"):
        compute_lambda_returns(
            rewards=[0.0, 1.0],
            state_values=[0.0, 0.0],
            players=players,
            lambda_param=0.5,
        )


def test_lambda_returns_reject_out_of_range_state_values() -> None:
    with pytest.raises(ValueError, match="state_values"):
        compute_lambda_returns(
            rewards=[0.0, 1.0],
            state_values=[0.0, 1.5],
            players=[1, 1],
            lambda_param=0.5,
        )


def _features_with_legal_places(legal_places: list[bool]) -> torch.Tensor:
    features = torch.zeros((1, FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=torch.float32)
    plane = torch.as_tensor(legal_places, dtype=torch.float32).reshape(BOARD_SIZE, BOARD_SIZE)
    features[0, 4] = plane
    return features


def test_legal_mask_from_features_marks_places_and_pass() -> None:
    legal_places = [False] * BOARD_CELLS
    legal_places[0] = True
    legal_places[BOARD_CELLS - 1] = True
    features = _features_with_legal_places(legal_places)

    mask = legal_mask_from_features(features)

    assert mask.shape == (1, ACTION_SPACE)
    assert bool(mask[0, 0])
    assert bool(mask[0, BOARD_CELLS - 1])
    assert not bool(mask[0, 1])
    assert bool(mask[0, PASS_ACTION])


def test_masked_state_value_uses_legal_masked_policy() -> None:
    logits = torch.tensor([[1.0, 5.0, -1.0]], dtype=torch.float32)
    q_values = torch.tensor([[0.5, 0.9, -0.7]], dtype=torch.float32)
    legal = torch.tensor([[True, False, True]])

    value = masked_state_value(logits, q_values, legal)
    probabilities = torch.softmax(logits.masked_fill(~legal, torch.finfo(torch.float32).min), dim=1)
    expected = (probabilities * q_values).sum(dim=1)

    assert value.shape == (1,)
    assert float(value[0]) == pytest.approx(float(expected[0]), abs=1e-6)
    assert float(probabilities[0, 1]) == pytest.approx(0.0)


def test_masked_state_value_propagates_gradients() -> None:
    logits = torch.randn(2, 3, requires_grad=True)
    q_values = torch.randn(2, 3, requires_grad=True)
    legal = torch.tensor([[True, False, True], [False, True, True]])

    value = masked_state_value(logits, q_values, legal)
    value.sum().backward()

    assert value.grad_fn is not None
    assert logits.grad is not None
    assert q_values.grad is not None
    assert not bool(torch.isnan(logits.grad).any())


def test_features_helpers_use_expected_channel_and_spatial_shape() -> None:
    features = _features_with_legal_places([True] * BOARD_CELLS)
    assert legal_mask_from_features(features).shape == (1, ACTION_SPACE)

    with pytest.raises(ValueError, match="spatial shape"):
        legal_mask_from_features(torch.zeros((1, FEATURE_CHANNELS, 3, 3)))

    with pytest.raises(ValueError, match="rank 4"):
        legal_mask_from_features(torch.zeros((FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE)))


def test_lambda_return_matches_manual_reference_for_alternating_players() -> None:
    gamma = 1.0
    lambda_param = 0.8825
    rewards = [0.0, 0.0, 1.0]
    values = [0.1, -0.2, 0.3]
    players = [1, 2, 1]

    returns = compute_lambda_returns(
        rewards,
        values,
        players,
        lambda_param=lambda_param,
        gamma=gamma,
    )

    manual = [0.0, 0.0, 0.0]
    manual[2] = rewards[2]
    for index in range(1, -1, -1):
        sign = 1.0 if players[index] == players[index + 1] else -1.0
        continuation = (1.0 - lambda_param) * values[index + 1] + lambda_param * manual[index + 1]
        manual[index] = rewards[index] + gamma * sign * continuation

    for index in range(len(manual)):
        assert float(returns[index]) == pytest.approx(manual[index], abs=1e-6)
    assert math.isfinite(float(returns.sum()))
