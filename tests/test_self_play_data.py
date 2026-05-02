import numpy as np
import pytest
from great_kingdom_ai.features import ACTION_SPACE
from great_kingdom_ai.self_play_data import (
    policy_target_from_visit_counts,
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
