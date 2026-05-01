import random

import numpy as np
import pytest

from great_kingdom_ai.features import ACTION_SPACE
from great_kingdom_ai.self_play_data import (
    policy_target_from_visit_counts,
    select_action_from_visit_counts,
    value_target_for_player,
)


def test_policy_target_from_visit_counts_normalizes_to_action_space() -> None:
    visits = [0] * ACTION_SPACE
    visits[3] = 1
    visits[81] = 3

    target = policy_target_from_visit_counts(visits)

    assert target.shape == (ACTION_SPACE,)
    assert target.dtype == np.float32
    assert np.isclose(target.sum(), 1.0)
    assert target[3] == pytest.approx(0.25)
    assert target[81] == pytest.approx(0.75)


def test_policy_target_rejects_empty_visit_distribution() -> None:
    with pytest.raises(ValueError, match="at least one visit"):
        policy_target_from_visit_counts([0] * ACTION_SPACE)


def test_value_target_uses_sample_player_perspective() -> None:
    assert value_target_for_player(player=1, winner=1) == 1.0
    assert value_target_for_player(player=1, winner=2) == -1.0
    assert value_target_for_player(player=2, winner=1) == -1.0


def test_select_action_uses_argmax_when_temperature_is_zero() -> None:
    visits = [0] * ACTION_SPACE
    visits[5] = 10
    visits[6] = 20

    action = select_action_from_visit_counts(visits, random.Random(1), temperature=0.0)

    assert action == 6


def test_select_action_samples_from_positive_visit_counts() -> None:
    visits = [0] * ACTION_SPACE
    visits[2] = 1
    visits[7] = 1
    rng = random.Random(3)

    actions = {
        select_action_from_visit_counts(visits, rng, temperature=1.0)
        for _ in range(20)
    }

    assert actions <= {2, 7}
    assert actions == {2, 7}
