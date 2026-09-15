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

from great_kingdom_ai.features import ACTION_SPACE  # noqa: E402
from great_kingdom_ai.klent.self_play import (  # noqa: E402
    KlentSelfPlayConfig,
    play_klent_game,
    play_klent_games,
)
from great_kingdom_ai.klent.types import KlentConfig  # noqa: E402
from great_kingdom_ai.replay import TrajectoryReplayStore  # noqa: E402

_rust_core_available = importlib.util.find_spec("great_kingdom_core") is not None
requires_core = pytest.mark.skipif(
    not _rust_core_available,
    reason="great_kingdom_core extension is not installed",
)


class _ScriptedWinState:
    """Three-move scripted game where player 1 wins on the final move."""

    def __init__(self) -> None:
        self.applied: list[int] = []

    def current_player(self) -> int:
        return 1 if len(self.applied) % 2 == 0 else 2

    def legal_actions(self) -> list[int]:
        return [0, 1, 81]

    def apply_action(self, action: int) -> None:
        self.applied.append(action)

    def is_terminal(self) -> bool:
        return len(self.applied) >= 3

    def winner(self) -> int | None:
        return 1 if self.is_terminal() else None

    def end_reason(self) -> int | None:
        return 1 if self.is_terminal() else None

    def territory_scores(self) -> tuple[int, int]:
        return (3, 1)

    def feature_planes(self) -> list[float]:
        from great_kingdom_ai.features import BOARD_SIZE, FEATURE_CHANNELS

        planes = np.zeros((FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        planes[4, 0, 0] = 1.0
        planes[4, 0, 1] = 1.0
        return planes.reshape(-1).tolist()

    def legal_mask(self) -> list[bool]:
        mask = [False] * ACTION_SPACE
        mask[0] = True
        mask[1] = True
        mask[81] = True
        return mask


class _ConstantQModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(()))
        self.forward_calls = 0

    def forward_q(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.forward_calls += 1
        batch = features.shape[0]
        logits = torch.zeros((batch, ACTION_SPACE), device=features.device)
        q_values = torch.zeros((batch, ACTION_SPACE), device=features.device)
        return logits, q_values


@requires_core
def test_play_klent_game_records_targets_actions_and_lambda_returns() -> None:
    model = _ConstantQModel()
    config = KlentConfig()
    log, episode = play_klent_game(
        model,
        seed=0,
        config=config,
        self_play=KlentSelfPlayConfig(episode_id=3, model_version=1, created_iteration=2),
    )

    assert len(log.moves) == len(episode.transitions) > 0
    assert episode.episode_id == 3
    assert episode.winner == log.winner
    for index, transition in enumerate(episode.transitions):
        assert transition.legal_mask is not None
        assert bool(transition.legal_mask[transition.action])
        assert transition.policy_target.sum() == pytest.approx(1.0, abs=1e-6)
        assert np.all(transition.policy_target[~transition.legal_mask] == 0.0)
        assert transition.lambda_return is not None
        assert np.isfinite(transition.lambda_return)
        assert -1.0 <= transition.lambda_return <= 1.0
        assert transition.model_version == 1
        assert transition.created_iteration == 2
        assert transition.terminal is (index == len(episode.transitions) - 1)
    assert model.forward_calls == len(log.moves)


def test_play_klent_game_matches_scripted_lambda_returns_without_core() -> None:
    config = KlentConfig(lambda_param=0.5, gamma=1.0)
    log, episode = play_klent_game(
        _ConstantQModel(),
        seed=0,
        config=config,
        self_play=KlentSelfPlayConfig(max_turns=5),
        state=_ScriptedWinState(),
    )

    assert [move.player for move in log.moves] == [1, 2, 1]
    assert [transition.lambda_return for transition in episode.transitions] == pytest.approx(
        [0.25, -0.5, 1.0]
    )
    assert episode.transitions[-1].terminal is True
    assert episode.transitions[-1].policy_target.sum() == pytest.approx(1.0)


@requires_core
def test_play_klent_game_is_deterministic_for_a_seed() -> None:
    config = KlentConfig()
    first_log, _ = play_klent_game(_ConstantQModel(), seed=11, config=config)
    second_log, _ = play_klent_game(_ConstantQModel(), seed=11, config=config)

    assert [move.action for move in first_log.moves] == [
        move.action for move in second_log.moves
    ]


@requires_core
def test_play_klent_games_builds_strict_replay_store() -> None:
    config = KlentConfig()
    results = play_klent_games(
        _ConstantQModel(),
        seeds=[0, 1],
        config=config,
        model_version=4,
        created_iteration=5,
        max_turns=100,
    )
    episodes = [episode for _log, episode in results]
    store = TrajectoryReplayStore.from_episodes(1024, episodes)

    assert store.lambda_returns is not None
    assert store.lambda_returns_present is not None
    assert bool(store.lambda_returns_present.all())
    assert store.actions.shape == (len(store),)
    assert store.model_versions.tolist() == [4] * len(store)
    assert store.created_iterations.tolist() == [5] * len(store)
    assert set(store.search_config_hash_table.tolist()) == {
        "klent:alpha=0.03,beta=0.1,lambda=0.8825,gamma=1"
    }


@requires_core
def test_play_klent_game_reports_length_limit_as_error() -> None:
    with pytest.raises(RuntimeError, match="max_turns"):
        play_klent_game(
            _ConstantQModel(),
            seed=0,
            config=KlentConfig(),
            self_play=KlentSelfPlayConfig(max_turns=1),
        )