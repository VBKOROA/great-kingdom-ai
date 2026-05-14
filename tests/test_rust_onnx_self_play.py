from __future__ import annotations

from pathlib import Path

import great_kingdom_ai.rust_onnx_self_play as rust_self_play_module
import numpy as np
from great_kingdom_ai.features import ACTION_SPACE, FEATURE_CHANNELS, PASS_ACTION
from great_kingdom_ai.rust_onnx_self_play import RustOnnxSelfPlayConfig
from great_kingdom_ai.self_play import SelfPlayConfig
from great_kingdom_ai.trajectory_replay import TrajectoryReplayStore


class FakeGumbelResult:
    def selected_action(self) -> int:
        return PASS_ACTION

    def policy_target(self) -> list[float]:
        policy = [0.0] * ACTION_SPACE
        policy[PASS_ACTION] = 1.0
        return policy


class FakeEvalRequest:
    def __init__(self, features: list[list[float]]) -> None:
        self._features = features

    def feature_planes(self) -> list[list[float]]:
        return self._features


class FakeOnnxEvaluator:
    def evaluate(self, request: FakeEvalRequest) -> tuple[list[list[float]], list[float]]:
        row_count = len(request.feature_planes())
        logits = [[0.0] * ACTION_SPACE for _ in range(row_count)]
        values = [0.0] * row_count
        return logits, values


class FakeRustSelfPlayBatch:
    def __init__(self, game_count: int, **_kwargs: object) -> None:
        self._game_count = game_count
        self._turn = 0
        self._terminal = [False] * game_count

    def active_game_indexes(self) -> list[int]:
        return [index for index, terminal in enumerate(self._terminal) if not terminal]

    def active_eval_request(self) -> FakeEvalRequest:
        features = []
        for _index in self.active_game_indexes():
            row = [0.0] * (FEATURE_CHANNELS * 9 * 9)
            row[0] = float(self._turn + 1)
            features.append(row)
        return FakeEvalRequest(features)

    def current_players(self) -> list[int]:
        return [1] * self._game_count

    def set_simulations(self, _simulations: list[int | None]) -> None:
        return None

    def set_max_considered_actions(self, _max_considered_actions: list[int | None]) -> None:
        return None

    def search_active_with_onnx_evaluator(
        self,
        _evaluator: FakeOnnxEvaluator,
        *,
        leaf_batch_size: int,
    ) -> list[FakeGumbelResult | None]:
        assert leaf_batch_size == 8
        results: list[FakeGumbelResult | None] = [None] * self._game_count
        for index in self.active_game_indexes():
            results[index] = FakeGumbelResult()
        return results

    def apply_actions(self, actions: list[int | None]) -> list[int | None]:
        assert actions == [PASS_ACTION]
        self._turn += 1
        if self._turn >= 3:
            self._terminal = [True] * self._game_count
        return [1 if terminal else None for terminal in self._terminal]

    def is_terminal(self) -> list[bool]:
        return list(self._terminal)

    def winners(self) -> list[int | None]:
        return [1 if terminal else None for terminal in self._terminal]

    def end_reasons(self) -> list[int | None]:
        return [1 if terminal else None for terminal in self._terminal]

    def territory_scores(self) -> list[tuple[int, int]]:
        return [(0, 0)] * self._game_count


class FakeCore:
    GumbelSelfPlayBatch = FakeRustSelfPlayBatch


def test_rust_onnx_trajectory_keeps_only_full_playout_cap_turns(
    monkeypatch,
    tmp_path: Path,
) -> None:
    full_turns = iter([False, True, False])
    monkeypatch.setattr(
        rust_self_play_module,
        "_use_full_search_turn",
        lambda _rng, _config: next(full_turns),
    )
    config = RustOnnxSelfPlayConfig(
        onnx_model_path=tmp_path / "model.onnx",
        output_dir=tmp_path,
        games=1,
        self_play=SelfPlayConfig(
            max_turns=5,
            playout_cap_randomization=True,
            playout_cap_full_search_fraction=0.25,
            leaf_batch_size=8,
        ),
    )

    logs, samples, episodes = rust_self_play_module._run_one_batch(
        config,
        core=FakeCore(),
        evaluator=FakeOnnxEvaluator(),
        seed_start=0,
        game_count=1,
    )

    assert len(logs) == 1
    assert len(logs[0].moves) == 3
    assert len(samples) == 1
    assert len(episodes) == 1
    assert len(episodes[0].transitions) == 1
    assert episodes[0].transitions[0].timestep == 0
    assert episodes[0].transitions[0].features[0, 0, 0] == np.float32(2.0)

    store = TrajectoryReplayStore.from_episodes(8, episodes)
    assert len(store) == 1
