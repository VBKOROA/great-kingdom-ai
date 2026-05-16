"""Random self-play orchestration backed by the Rust rules engine."""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any, NoReturn, Protocol, cast

import numpy as np

from great_kingdom_ai.features import BOARD_SIZE, FEATURE_CHANNELS
from great_kingdom_ai.replay_buffer import ReplaySample
from great_kingdom_ai.self_play_data import (
    policy_target_from_visit_counts,
    value_target_for_player,
)

PASS_ACTION = 81


class SelfPlayState(Protocol):
    def current_player(self) -> int: ...

    def legal_actions(self) -> list[int]: ...

    def apply_action(self, action_index: int) -> int | None: ...

    def is_terminal(self) -> bool: ...

    def winner(self) -> int | None: ...

    def end_reason(self) -> int | None: ...

    def territory_scores(self) -> tuple[int, int]: ...

    def feature_planes(self) -> list[float]: ...

    def legal_mask(self) -> list[bool]: ...


class SearchResultLike(Protocol):
    def selected_action(self) -> int | None: ...

    def visit_counts(self) -> list[int]: ...


class SearchLike(Protocol):
    def search_with_logits(
        self,
        state: SelfPlayState,
        policy_logits: list[float],
    ) -> SearchResultLike: ...

    def search_with_logits_and_evaluator(
        self,
        state: SelfPlayState,
        policy_logits: list[float],
        evaluator: Callable[[Any], tuple[list[list[float]], list[float]]],
        root_value: float,
        leaf_batch_size: int = 8,
    ) -> SearchResultLike: ...

    def set_simulations(self, simulations: int) -> None: ...

    def set_max_considered_actions(self, max_considered_actions: int) -> None: ...


class SelfPlayBatchLike(Protocol):
    def len(self) -> int: ...

    def active_count(self) -> int: ...

    def active_game_indexes(self) -> list[int]: ...

    def active_eval_request(self) -> Any: ...

    def current_players(self) -> list[int] | bytes: ...

    def is_terminal(self) -> list[bool]: ...

    def winners(self) -> list[int | None]: ...

    def end_reasons(self) -> list[int | None]: ...

    def territory_scores(self) -> list[tuple[int, int]]: ...

    def search_active_with_logits(
        self,
        policy_logits: list[list[float]],
    ) -> list[SearchResultLike | None]: ...

    def search_active_with_logits_and_evaluator(
        self,
        policy_logits: list[list[float]],
        evaluator: Callable[[Any], tuple[list[list[float]], list[float]]],
        root_values: list[float],
        leaf_batch_size: int = 8,
    ) -> list[SearchResultLike | None]: ...

    def apply_actions(self, actions: list[int | None]) -> list[int | None]: ...

    def set_simulations(self, simulations: list[int | None]) -> None: ...

    def set_max_considered_actions(
        self,
        max_considered_actions: list[int | None],
    ) -> None: ...


@dataclass(frozen=True)
class MoveLog:
    turn: int
    player: int
    action: int


@dataclass(frozen=True)
class GameLog:
    seed: int
    moves: list[MoveLog]
    winner: int
    end_reason: int
    territory_scores: tuple[int, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SmokeSummary:
    games: int
    total_moves: int
    max_moves: int
    blue_wins: int
    orange_wins: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class SelfPlayConfig:
    max_turns: int = 200
    gumbel_simulations: int = 128
    gumbel_max_considered_actions: int = 16
    gumbel_c_visit: float = 50.0
    gumbel_c_scale: float = 1.0
    gumbel_scale: float = 1.0
    policy_target_c_visit: float = 5.0
    policy_target_c_scale: float = 0.25
    policy_target_temperature: float = 1.0
    gumbel_seed: int = 0
    temperature_turns: int = 10
    sampling_temperature: float = 1.0
    playout_cap_randomization: bool = False
    playout_cap_full_search_fraction: float = 0.25
    playout_cap_full_simulations: int = 128
    playout_cap_fast_simulations: int = 16
    playout_cap_full_max_considered_actions: int | None = None
    playout_cap_fast_max_considered_actions: int | None = None
    leaf_batch_size: int = 8

    def __post_init__(self) -> None:
        if self.max_turns <= 0:
            raise ValueError("max_turns must be positive")
        if self.gumbel_simulations <= 0:
            raise ValueError("gumbel_simulations must be positive")
        if self.gumbel_max_considered_actions <= 0:
            raise ValueError("gumbel_max_considered_actions must be positive")
        if self.gumbel_c_visit <= 0.0:
            raise ValueError("gumbel_c_visit must be positive")
        if self.gumbel_c_scale <= 0.0:
            raise ValueError("gumbel_c_scale must be positive")
        if not np.isfinite(self.gumbel_scale) or self.gumbel_scale < 0.0:
            raise ValueError("gumbel_scale must be finite and non-negative")
        if not np.isfinite(self.policy_target_c_visit) or self.policy_target_c_visit <= 0.0:
            raise ValueError("policy_target_c_visit must be finite and positive")
        if not np.isfinite(self.policy_target_c_scale) or self.policy_target_c_scale <= 0.0:
            raise ValueError("policy_target_c_scale must be finite and positive")
        if (
            not np.isfinite(self.policy_target_temperature)
            or self.policy_target_temperature <= 0.0
        ):
            raise ValueError("policy_target_temperature must be finite and positive")
        if self.temperature_turns < 0:
            raise ValueError("temperature_turns must be non-negative")
        if self.sampling_temperature < 0.0:
            raise ValueError("sampling_temperature must be non-negative")
        if not 0.0 < self.playout_cap_full_search_fraction <= 1.0:
            raise ValueError("playout_cap_full_search_fraction must be in (0, 1]")
        if self.playout_cap_full_simulations <= 0:
            raise ValueError("playout_cap_full_simulations must be positive")
        if self.playout_cap_fast_simulations <= 0:
            raise ValueError("playout_cap_fast_simulations must be positive")
        if (
            self.playout_cap_full_max_considered_actions is not None
            and self.playout_cap_full_max_considered_actions <= 0
        ):
            raise ValueError("playout_cap_full_max_considered_actions must be positive")
        if (
            self.playout_cap_fast_max_considered_actions is not None
            and self.playout_cap_fast_max_considered_actions <= 0
        ):
            raise ValueError("playout_cap_fast_max_considered_actions must be positive")
        if self.leaf_batch_size <= 0:
            raise ValueError("leaf_batch_size must be positive")


def choose_random_legal_action(
    state: SelfPlayState,
    rng: random.Random,
    *,
    prefer_place: bool = False,
) -> int:
    """Choose one action from the Rust-provided legal action list."""
    legal_actions = state.legal_actions()
    if not legal_actions:
        raise ValueError("state has no legal actions")

    candidates: Sequence[int] = legal_actions
    if prefer_place:
        place_actions = [action for action in legal_actions if action != PASS_ACTION]
        if place_actions:
            candidates = place_actions

    return rng.choice(candidates)


def play_random_game(
    *,
    seed: int,
    state: SelfPlayState | None = None,
    max_turns: int = 200,
    prefer_place: bool = False,
) -> GameLog:
    game_state = state if state is not None else create_core_game_state()
    rng = random.Random(seed)
    moves: list[MoveLog] = []

    for turn in range(max_turns):
        if game_state.is_terminal():
            break

        action = choose_random_legal_action(game_state, rng, prefer_place=prefer_place)
        moves.append(MoveLog(turn=turn, player=game_state.current_player(), action=action))
        game_state.apply_action(action)
    else:
        raise RuntimeError(f"random self-play exceeded max_turns={max_turns}")

    winner = game_state.winner()
    end_reason = game_state.end_reason()
    if winner is None or end_reason is None:
        raise RuntimeError("random self-play stopped before terminal outcome")

    return GameLog(
        seed=seed,
        moves=moves,
        winner=winner,
        end_reason=end_reason,
        territory_scores=game_state.territory_scores(),
    )


def play_random_games(
    *,
    seeds: Sequence[int],
    max_turns: int = 200,
    prefer_place: bool = False,
) -> list[GameLog]:
    return [
        play_random_game(seed=seed, max_turns=max_turns, prefer_place=prefer_place)
        for seed in seeds
    ]


def play_self_play_game(
    *,
    seed: int,
    state: SelfPlayState | None = None,
    search: SearchLike | None = None,
    config: SelfPlayConfig | None = None,
    prior_provider: Callable[[SelfPlayState], Sequence[float]] | None = None,
    evaluator_provider: Callable[
        [Sequence[SelfPlayState]],
        tuple[Sequence[Sequence[float]], Sequence[float]],
    ]
    | None = None,
) -> tuple[GameLog, list[ReplaySample]]:
    """Run one Gumbel self-play game and return replay samples with final value targets."""
    if config is None:
        raise ValueError("config must be provided")
    game_state = state if state is not None else create_core_game_state()
    search_engine = (
        search if search is not None else create_core_search_engine(config, seed_offset=seed)
    )
    rng = random.Random(seed)
    moves: list[MoveLog] = []
    pending_samples: list[tuple[int, np.ndarray, np.ndarray, np.ndarray | None]] = []

    for turn in range(config.max_turns):
        if game_state.is_terminal():
            break

        player = game_state.current_player()
        features = _state_features_for_replay(game_state)
        root_priors = prior_provider(game_state) if prior_provider is not None else None
        use_full_search = _use_full_search_turn(rng, config)
        if config.playout_cap_randomization:
            simulations = (
                config.playout_cap_full_simulations
                if use_full_search
                else config.playout_cap_fast_simulations
            )
            _set_search_simulations(search_engine, simulations)
            _set_search_max_considered_actions(
                search_engine,
                _playout_cap_max_considered_actions(use_full_search, config),
            )
        result, root_policy_logits = _run_self_play_search(
            game_state,
            search_engine,
            rng,
            config,
            root_priors=root_priors,
            evaluator_provider=evaluator_provider,
        )
        policy = _policy_target_from_result(result)
        action = _select_self_play_action(
            result,
            rng,
            config,
            turn=turn,
            legal_actions=game_state.legal_actions(),
        )

        if use_full_search:
            pending_samples.append((player, features, policy, root_policy_logits))
        moves.append(MoveLog(turn=turn, player=player, action=action))
        game_state.apply_action(action)
    else:
        raise RuntimeError(f"self-play exceeded max_turns={config.max_turns}")

    winner = game_state.winner()
    end_reason = game_state.end_reason()
    if winner is None or end_reason is None:
        raise RuntimeError("self-play stopped before terminal outcome")

    samples = [
        ReplaySample(
            features=features,
            policy=policy,
            value=value_target_for_player(player=player, winner=winner),
            root_policy_logits=root_policy_logits,
        )
        for player, features, policy, root_policy_logits in pending_samples
    ]
    return (
        GameLog(
            seed=seed,
            moves=moves,
            winner=winner,
            end_reason=end_reason,
            territory_scores=game_state.territory_scores(),
        ),
        samples,
    )


def play_self_play_games_batched(
    *,
    seeds: Sequence[int],
    search_factory: Callable[[], SearchLike],
    config: SelfPlayConfig,
    prior_provider: Callable[[Sequence[SelfPlayState]], Sequence[Sequence[float]]]
    | None = None,
    evaluator_provider: Callable[
        [Sequence[SelfPlayState]],
        tuple[Sequence[Sequence[float]], Sequence[float]],
    ]
    | None = None,
    feature_batch_prior_provider: Callable[
        [Sequence[Sequence[float]], Sequence[Sequence[bool]]],
        Sequence[Sequence[float]],
    ]
    | None = None,
    request_evaluator_provider: Callable[
        [Any],
        tuple[Any, Any],
    ]
    | None = None,
    state_factory: Callable[[], SelfPlayState] | None = None,
) -> list[tuple[GameLog, list[ReplaySample]]]:
    """Run Gumbel self-play games while batching neural-network root inference."""
    if (
        state_factory is None
        and (prior_provider is not None or feature_batch_prior_provider is not None)
        and _can_create_core_self_play_batch()
    ):
        return _play_self_play_games_core_batched(
            seeds=seeds,
            config=config,
            prior_provider=prior_provider,
            evaluator_provider=evaluator_provider,
            feature_batch_prior_provider=feature_batch_prior_provider,
            request_evaluator_provider=request_evaluator_provider,
        )

    make_state = state_factory if state_factory is not None else create_core_game_state
    games = [
        _BatchedGame(
            seed=seed,
            state=make_state(),
            search=search_factory(),
            rng=random.Random(seed),
        )
        for seed in seeds
    ]
    if not games:
        return []

    for turn in range(config.max_turns):
        active_games = [game for game in games if not game.state.is_terminal()]
        if not active_games:
            break

        batched_priors = _batched_root_priors(active_games, prior_provider)
        for game, root_priors in zip(active_games, batched_priors, strict=True):
            _play_batched_self_play_turn(
                game,
                turn,
                config,
                root_priors=root_priors,
                evaluator_provider=evaluator_provider,
            )
    else:
        unfinished = [game.seed for game in games if not game.state.is_terminal()]
        if unfinished:
            raise RuntimeError(
                f"self-play exceeded max_turns={config.max_turns} "
                f"for seeds={unfinished}"
            )

    return [_finish_batched_game(game) for game in games]


def summarize_logs(logs: Sequence[GameLog]) -> SmokeSummary:
    move_counts = [len(log.moves) for log in logs]
    return SmokeSummary(
        games=len(logs),
        total_moves=sum(move_counts),
        max_moves=max(move_counts, default=0),
        blue_wins=sum(1 for log in logs if log.winner == 1),
        orange_wins=sum(1 for log in logs if log.winner == 2),
    )


def create_core_game_state() -> SelfPlayState:
    try:
        import great_kingdom_core as core  # type: ignore[import-untyped]
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "great_kingdom_core is not installed. Build it with maturin before self-play."
        ) from exc

    return cast(SelfPlayState, core.GameState())


def create_core_search_engine(
    config: SelfPlayConfig,
    *,
    seed_offset: int = 0,
) -> SearchLike:
    try:
        import great_kingdom_core as core
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "great_kingdom_core is not installed. Build it with maturin before self-play."
        ) from exc
    return cast(
        SearchLike,
        core.GumbelSearch(
            simulations=config.gumbel_simulations,
            max_considered_actions=config.gumbel_max_considered_actions,
            c_visit=config.gumbel_c_visit,
            c_scale=config.gumbel_c_scale,
            seed=config.gumbel_seed + seed_offset,
            gumbel_scale=config.gumbel_scale,
            policy_target_temperature=config.policy_target_temperature,
            policy_target_c_visit=config.policy_target_c_visit,
            policy_target_c_scale=config.policy_target_c_scale,
        ),
    )


def create_core_self_play_batch(
    config: SelfPlayConfig,
    *,
    game_count: int,
    seed_offset: int = 0,
) -> SelfPlayBatchLike:
    try:
        import great_kingdom_core as core
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "great_kingdom_core is not installed. Build it with maturin before self-play."
        ) from exc
    return cast(
        SelfPlayBatchLike,
        core.GumbelSelfPlayBatch(
            game_count=game_count,
            simulations=config.gumbel_simulations,
            max_considered_actions=config.gumbel_max_considered_actions,
            c_visit=config.gumbel_c_visit,
            c_scale=config.gumbel_c_scale,
            seed=config.gumbel_seed + seed_offset,
            gumbel_scale=config.gumbel_scale,
            policy_target_temperature=config.policy_target_temperature,
            policy_target_c_visit=config.policy_target_c_visit,
            policy_target_c_scale=config.policy_target_c_scale,
        ),
    )


def _run_self_play_search(
    state: SelfPlayState,
    search: SearchLike,
    rng: random.Random,
    config: SelfPlayConfig,
    *,
    root_priors: Sequence[float] | None = None,
    evaluator_provider: Callable[
        [Sequence[SelfPlayState]],
        tuple[Sequence[Sequence[float]], Sequence[float]],
    ]
    | None = None,
) -> tuple[SearchResultLike, np.ndarray]:
    del rng
    root_logits = list(root_priors) if root_priors is not None else [0.0] * (PASS_ACTION + 1)
    root_value: float | None = None
    if evaluator_provider is not None:
        root_policies, root_values = evaluator_provider([state])
        if len(root_policies) != 1:
            raise ValueError(
                f"expected 1 root policy row from evaluator, got {len(root_policies)}"
            )
        if len(root_values) != 1:
            raise ValueError(f"expected 1 root value from evaluator, got {len(root_values)}")
        if root_priors is None:
            root_logits = [float(value) for value in root_policies[0]]
        root_value = float(root_values[0])
    root_policy_logits = np.asarray(root_logits, dtype=np.float32)
    if evaluator_provider is None:
        return search.search_with_logits(state, root_logits), root_policy_logits
    if root_value is None:
        raise ValueError("Gumbel evaluator search requires an explicit root value")

    def gumbel_leaf_evaluator(request: Any) -> tuple[list[list[float]], list[float]]:
        return _evaluate_core_batch_policy_values(evaluator_provider, request)

    return (
        search.search_with_logits_and_evaluator(
            state,
            root_logits,
            gumbel_leaf_evaluator,
            root_value,
            config.leaf_batch_size,
        ),
        root_policy_logits,
    )


@dataclass
class _BatchedGame:
    seed: int
    state: SelfPlayState
    search: SearchLike
    rng: random.Random
    moves: list[MoveLog] | None = None
    pending_samples: list[tuple[int, np.ndarray, np.ndarray, np.ndarray | None]] | None = None

    def __post_init__(self) -> None:
        if self.moves is None:
            self.moves = []
        if self.pending_samples is None:
            self.pending_samples = []


def _batched_root_priors(
    games: Sequence[_BatchedGame],
    prior_provider: Callable[[Sequence[SelfPlayState]], Sequence[Sequence[float]]] | None,
) -> list[Sequence[float] | None]:
    if prior_provider is None:
        return [None] * len(games)
    priors: list[Sequence[float] | None] = list(prior_provider([game.state for game in games]))
    if len(priors) != len(games):
        raise ValueError(
            f"expected {len(games)} prior rows from batch provider, got {len(priors)}"
        )
    return priors


def _play_self_play_games_core_batched(
    *,
    seeds: Sequence[int],
    config: SelfPlayConfig,
    prior_provider: Callable[[Sequence[SelfPlayState]], Sequence[Sequence[float]]]
    | None = None,
    evaluator_provider: Callable[
        [Sequence[SelfPlayState]],
        tuple[Sequence[Sequence[float]], Sequence[float]],
    ]
    | None = None,
    feature_batch_prior_provider: Callable[
        [Sequence[Sequence[float]], Sequence[Sequence[bool]]],
        Sequence[Sequence[float]],
    ]
    | None = None,
    request_evaluator_provider: Callable[
        [Any],
        tuple[Any, Any],
    ]
    | None = None,
) -> list[tuple[GameLog, list[ReplaySample]]]:
    if not seeds:
        return []

    batch = create_core_self_play_batch(config, game_count=len(seeds))
    rngs = [random.Random(seed) for seed in seeds]
    moves: list[list[MoveLog]] = [[] for _ in seeds]
    pending_samples: list[list[tuple[int, np.ndarray, np.ndarray, np.ndarray | None]]] = [
        [] for _ in seeds
    ]
    territory_scores = [(0, 0) for _ in seeds]

    for turn in range(config.max_turns):
        active_indexes = batch.active_game_indexes()
        if not active_indexes:
            break

        request = batch.active_eval_request()
        players = _as_int_list(batch.current_players())
        feature_rows = request.feature_planes()
        masks = request.legal_masks()
        features_by_game = {
            game_index: _flat_features_for_replay(feature_planes)
            for game_index, feature_planes in zip(
                active_indexes,
                feature_rows,
                strict=True,
            )
        }
        root_values: list[float] | None = None
        if request_evaluator_provider is not None:
            root_policies, root_value_rows = request_evaluator_provider(request)
            priors = [[float(value) for value in row] for row in root_policies]
            root_values = [float(value) for value in root_value_rows]
        elif feature_batch_prior_provider is None:
            if prior_provider is None:
                raise ValueError("prior_provider is required for core batched self-play")
            priors = _evaluate_core_batch_priors(prior_provider, request)
        else:
            priors = [
                [float(value) for value in row]
                for row in feature_batch_prior_provider(feature_rows, masks)
            ]
        if len(priors) != len(active_indexes):
            raise ValueError(
                f"expected {len(active_indexes)} prior rows from batch provider, got {len(priors)}"
            )
        if (
            root_values is None and evaluator_provider is not None
        ):
            _root_policies, root_values = _evaluate_core_batch_policy_values(
                evaluator_provider,
                request,
            )
        if root_values is not None and len(root_values) != len(active_indexes):
            raise ValueError(
                f"expected {len(active_indexes)} root values from batch provider, "
                f"got {len(root_values)}"
            )

        noisy_priors = []
        root_policy_logits_by_game: dict[int, np.ndarray] = {}
        use_full_by_game: dict[int, bool] = {}
        simulation_budgets: list[int | None] = [None] * batch.len()
        for game_index, prior, _mask in zip(active_indexes, priors, masks, strict=True):
            rng = rngs[game_index]
            use_full = _use_full_search_turn(rng, config)
            use_full_by_game[game_index] = use_full
            if config.playout_cap_randomization:
                simulation_budgets[game_index] = (
                    config.playout_cap_full_simulations
                    if use_full
                    else config.playout_cap_fast_simulations
                )
            prior_values = [float(value) for value in prior]
            noisy_priors.append(prior_values)
            root_policy_logits_by_game[game_index] = np.asarray(
                prior_values,
                dtype=np.float32,
            )

        if config.playout_cap_randomization:
            batch.set_simulations(simulation_budgets)
            batch.set_max_considered_actions(
                [
                    (
                        _playout_cap_max_considered_actions(
                            use_full_by_game[game_index],
                            config,
                        )
                        if game_index in use_full_by_game
                        else None
                    )
                    for game_index in range(batch.len())
                ]
            )
        if evaluator_provider is None and request_evaluator_provider is None:
            results = batch.search_active_with_logits(noisy_priors)
        else:
            def evaluator(request: Any) -> tuple[Any, Any]:
                if request_evaluator_provider is not None:
                    return request_evaluator_provider(request)
                if evaluator_provider is None:
                    raise RuntimeError("evaluator_provider is required")
                return _evaluate_core_batch_policy_values(evaluator_provider, request)

            if root_values is None:
                raise ValueError("Gumbel batched evaluator search requires root values")
            results = batch.search_active_with_logits_and_evaluator(
                noisy_priors,
                evaluator,
                root_values=root_values,
                leaf_batch_size=config.leaf_batch_size,
            )
        actions: list[int | None] = [None] * batch.len()
        for game_index in active_indexes:
            result = results[game_index]
            if result is None:
                continue
            policy = _policy_target_from_result(result)
            action = _select_self_play_action(
                result,
                rngs[game_index],
                config,
                turn=turn,
                legal_actions=[
                    action
                    for action, is_legal in enumerate(
                        masks[active_indexes.index(game_index)]
                    )
                    if is_legal
                ],
            )
            if use_full_by_game[game_index]:
                pending_samples[game_index].append(
                    (
                        players[game_index],
                        features_by_game[game_index],
                        policy,
                        root_policy_logits_by_game[game_index],
                    )
                )
            moves[game_index].append(
                MoveLog(turn=turn, player=players[game_index], action=action)
            )
            actions[game_index] = action
        batch.apply_actions(actions)
    else:
        unfinished = [
            seeds[index]
            for index, is_terminal in enumerate(batch.is_terminal())
            if not is_terminal
        ]
        if unfinished:
            raise RuntimeError(
                f"self-play exceeded max_turns={config.max_turns} "
                f"for seeds={unfinished}"
            )

    winners = batch.winners()
    end_reasons = batch.end_reasons()
    territory_scores = batch.territory_scores()
    outputs: list[tuple[GameLog, list[ReplaySample]]] = []
    for game_index, seed in enumerate(seeds):
        winner = winners[game_index]
        end_reason = end_reasons[game_index]
        if winner is None or end_reason is None:
            raise RuntimeError("self-play stopped before terminal outcome")
        samples = [
            ReplaySample(
                features=features,
                policy=policy,
                value=value_target_for_player(player=player, winner=winner),
                root_policy_logits=root_policy_logits,
            )
            for player, features, policy, root_policy_logits in pending_samples[game_index]
        ]
        outputs.append(
            (
                GameLog(
                    seed=seed,
                    moves=moves[game_index],
                    winner=winner,
                    end_reason=end_reason,
                    territory_scores=territory_scores[game_index],
                ),
                samples,
            )
        )
    return outputs


def _evaluate_core_batch_priors(
    prior_provider: Callable[[Sequence[SelfPlayState]], Sequence[Sequence[float]]],
    request: Any,
) -> list[list[float]]:
    states = _request_states(request)
    return [
        [float(value) for value in row]
        for row in prior_provider(cast(Sequence[SelfPlayState], states))
    ]


def _evaluate_core_batch_policy_values(
    evaluator_provider: Callable[
        [Sequence[SelfPlayState]],
        tuple[Sequence[Sequence[float]], Sequence[float]],
    ],
    request: Any,
) -> tuple[list[list[float]], list[float]]:
    states = _request_states(request)
    policies, values = evaluator_provider(cast(Sequence[SelfPlayState], states))
    return (
        [[float(value) for value in row] for row in policies],
        [float(value) for value in values],
    )


def _request_states(request: Any) -> list[Any]:
    class _RequestState:
        def __init__(self, features: list[float], mask: list[bool], player: int | None) -> None:
            self._features = features
            self._mask = mask
            self._player = player

        def feature_planes(self) -> list[float]:
            return self._features

        def legal_mask(self) -> list[bool]:
            return self._mask

        def current_player(self) -> int:
            if self._player is None:
                raise RuntimeError("eval request did not include current player metadata")
            return self._player

    features = request.feature_planes()
    players = (
        [int(player) for player in request.current_players()]
        if hasattr(request, "current_players")
        else [None] * len(features)
    )
    return [
        _RequestState(feature_planes, mask, player)
        for feature_planes, mask, player in zip(
            features,
            request.legal_masks(),
            players,
            strict=True,
        )
    ]


def _flat_features_for_replay(feature_planes: Sequence[float]) -> np.ndarray:
    features = np.asarray(feature_planes, dtype=np.float32)
    expected = FEATURE_CHANNELS * BOARD_SIZE * BOARD_SIZE
    if features.shape != (expected,):
        raise ValueError(f"expected flat feature shape {(expected,)}, got {features.shape}")
    return cast(np.ndarray, features.reshape(FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE))


def _as_int_list(values: Sequence[int] | bytes) -> list[int]:
    return [int(value) for value in values]


def _can_create_core_self_play_batch() -> bool:
    try:
        import great_kingdom_core as core
    except ModuleNotFoundError:
        return False
    return hasattr(core, "GumbelSelfPlayBatch")


def _play_batched_self_play_turn(
    game: _BatchedGame,
    turn: int,
    config: SelfPlayConfig,
    *,
    root_priors: Sequence[float] | None,
    evaluator_provider: Callable[
        [Sequence[SelfPlayState]],
        tuple[Sequence[Sequence[float]], Sequence[float]],
    ]
    | None = None,
) -> None:
    player = game.state.current_player()
    features = _state_features_for_replay(game.state)
    use_full_search = _use_full_search_turn(game.rng, config)
    if config.playout_cap_randomization:
        simulations = (
            config.playout_cap_full_simulations
            if use_full_search
            else config.playout_cap_fast_simulations
        )
        _set_search_simulations(game.search, simulations)
        _set_search_max_considered_actions(
            game.search,
            _playout_cap_max_considered_actions(use_full_search, config),
        )

    result, root_policy_logits = _run_self_play_search(
        game.state,
        game.search,
        game.rng,
        config,
        root_priors=root_priors,
        evaluator_provider=evaluator_provider,
    )
    policy = _policy_target_from_result(result)
    action = _select_self_play_action(
        result,
        game.rng,
        config,
        turn=turn,
        legal_actions=game.state.legal_actions(),
    )

    if use_full_search:
        assert game.pending_samples is not None
        game.pending_samples.append((player, features, policy, root_policy_logits))
    assert game.moves is not None
    game.moves.append(MoveLog(turn=turn, player=player, action=action))
    game.state.apply_action(action)


def _finish_batched_game(game: _BatchedGame) -> tuple[GameLog, list[ReplaySample]]:
    winner = game.state.winner()
    end_reason = game.state.end_reason()
    if winner is None or end_reason is None:
        raise RuntimeError("self-play stopped before terminal outcome")
    assert game.moves is not None
    assert game.pending_samples is not None
    samples = [
        ReplaySample(
            features=features,
            policy=policy,
            value=value_target_for_player(player=player, winner=winner),
            root_policy_logits=root_policy_logits,
        )
        for player, features, policy, root_policy_logits in game.pending_samples
    ]
    return (
        GameLog(
            seed=game.seed,
            moves=game.moves,
            winner=winner,
            end_reason=end_reason,
            territory_scores=game.state.territory_scores(),
        ),
        samples,
    )


def _use_full_search_turn(rng: random.Random, config: SelfPlayConfig) -> bool:
    if not config.playout_cap_randomization:
        return True
    return rng.random() < config.playout_cap_full_search_fraction


def _playout_cap_max_considered_actions(use_full_search: bool, config: SelfPlayConfig) -> int:
    if use_full_search:
        return (
            config.playout_cap_full_max_considered_actions
            if config.playout_cap_full_max_considered_actions is not None
            else config.gumbel_max_considered_actions
        )
    return (
        config.playout_cap_fast_max_considered_actions
        if config.playout_cap_fast_max_considered_actions is not None
        else config.gumbel_max_considered_actions
    )


def _policy_target_from_result(result: SearchResultLike) -> np.ndarray:
    if hasattr(result, "policy_target"):
        policy = np.asarray(cast(Any, result).policy_target(), dtype=np.float32)
        if policy.shape != (PASS_ACTION + 1,):
            raise ValueError(
                f"expected policy target shape {(PASS_ACTION + 1,)}, got {policy.shape}"
            )
        return policy
    return policy_target_from_visit_counts(result.visit_counts())


def _select_self_play_action(
    result: SearchResultLike,
    rng: random.Random,
    config: SelfPlayConfig,
    *,
    turn: int,
    legal_actions: Sequence[int],
) -> int:
    del rng, config, turn
    selected = result.selected_action()
    if selected in set(legal_actions):
        return int(selected)
    visits = result.visit_counts()
    return max(legal_actions, key=lambda action: (visits[action], -action))


def _set_search_simulations(search: SearchLike, simulations: int) -> None:
    search.set_simulations(simulations)


def _set_search_max_considered_actions(
    search: SearchLike,
    max_considered_actions: int,
) -> None:
    search.set_max_considered_actions(max_considered_actions)


def _state_features_for_replay(state: SelfPlayState) -> np.ndarray:
    features = np.asarray(state.feature_planes(), dtype=np.float32)
    expected = FEATURE_CHANNELS * BOARD_SIZE * BOARD_SIZE
    if features.shape != (expected,):
        raise ValueError(f"expected flat feature shape {(expected,)}, got {features.shape}")
    return cast(np.ndarray, features.reshape(FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="great-kingdom-random-self-play",
        description="Run random self-play games against the Rust rules engine.",
    )
    parser.add_argument("--games", type=int, default=10, help="number of games to run")
    parser.add_argument("--seed-start", type=int, default=0, help="first deterministic seed")
    parser.add_argument("--max-turns", type=int, default=200, help="guard per game")
    parser.add_argument(
        "--prefer-place",
        action="store_true",
        help="prefer place actions while any place action is legal",
    )
    parser.add_argument("--json", action="store_true", help="print JSON logs instead of summary")
    return parser


def main() -> NoReturn:
    args = build_parser().parse_args()
    if args.games < 0:
        raise SystemExit("--games must be non-negative")

    seeds = list(range(args.seed_start, args.seed_start + args.games))
    logs = play_random_games(
        seeds=seeds,
        max_turns=args.max_turns,
        prefer_place=args.prefer_place,
    )
    if args.json:
        print(json.dumps([log.to_dict() for log in logs], indent=2, sort_keys=True))
    else:
        print(json.dumps(summarize_logs(logs).to_dict(), sort_keys=True))

    raise SystemExit(0)


if __name__ == "__main__":
    main()
