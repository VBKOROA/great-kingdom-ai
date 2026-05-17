"""Self-play orchestration helpers."""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from great_kingdom_ai.game_core import (
    as_int_list,
    evaluate_core_batch_policy_values,
    evaluate_core_batch_priors,
    flat_features_for_replay,
)
from great_kingdom_ai.replay_buffer import ReplaySample
from great_kingdom_ai.self_play_data import (
    policy_target_from_visit_counts,
    value_target_for_player,
)
from great_kingdom_ai.self_play_types import (
    PASS_ACTION,
    GameLog,
    MoveLog,
    SearchLike,
    SearchResultLike,
    SelfPlayBatchLike,
    SelfPlayConfig,
    SelfPlayState,
)

StateFactory = Callable[[], SelfPlayState]
SearchEngineFactory = Callable[[SelfPlayConfig, int], SearchLike]
CoreBatchFactory = Callable[[SelfPlayConfig, int], SelfPlayBatchLike]


def play_self_play_game(
    *,
    seed: int,
    state: SelfPlayState | None,
    search: SearchLike | None,
    config: SelfPlayConfig | None,
    prior_provider: Callable[[SelfPlayState], Sequence[float]] | None,
    evaluator_provider: Callable[
        [Sequence[SelfPlayState]],
        tuple[Sequence[Sequence[float]], Sequence[float]],
    ]
    | None,
    state_factory: StateFactory,
    search_engine_factory: SearchEngineFactory,
) -> tuple[GameLog, list[ReplaySample]]:
    if config is None:
        raise ValueError("config must be provided")
    game_state = state if state is not None else state_factory()
    search_engine = search if search is not None else search_engine_factory(config, seed)
    rng = random.Random(seed)
    moves: list[MoveLog] = []
    pending_samples: list[tuple[int, np.ndarray, np.ndarray, np.ndarray | None]] = []

    for turn in range(config.max_turns):
        if game_state.is_terminal():
            break

        player = game_state.current_player()
        features = flat_features_for_replay(game_state.feature_planes())
        root_priors = prior_provider(game_state) if prior_provider is not None else None
        use_full_search = use_full_search_turn(rng, config)
        if config.playout_cap_randomization:
            simulations = (
                config.playout_cap_full_simulations
                if use_full_search
                else config.playout_cap_fast_simulations
            )
            search_engine.set_simulations(simulations)
            search_engine.set_max_considered_actions(
                playout_cap_max_considered_actions(use_full_search, config)
            )
        result, root_policy_logits = run_self_play_search(
            game_state,
            search_engine,
            config,
            root_priors=root_priors,
            evaluator_provider=evaluator_provider,
        )
        policy = policy_target_from_result(result)
        action = select_self_play_action(result, legal_actions=game_state.legal_actions())

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
    prior_provider: Callable[[Sequence[SelfPlayState]], Sequence[Sequence[float]]] | None,
    evaluator_provider: Callable[
        [Sequence[SelfPlayState]],
        tuple[Sequence[Sequence[float]], Sequence[float]],
    ]
    | None,
    feature_batch_prior_provider: Callable[
        [Sequence[Sequence[float]], Sequence[Sequence[bool]]],
        Sequence[Sequence[float]],
    ]
    | None,
    request_evaluator_provider: Callable[[Any], tuple[Any, Any]] | None,
    state_factory: Callable[[], SelfPlayState] | None,
    default_state_factory: StateFactory,
    core_batch_factory: CoreBatchFactory,
    core_batch_available: Callable[[], bool],
) -> list[tuple[GameLog, list[ReplaySample]]]:
    if (
        state_factory is None
        and (prior_provider is not None or feature_batch_prior_provider is not None)
        and core_batch_available()
    ):
        return play_self_play_games_core_batched(
            seeds=seeds,
            config=config,
            prior_provider=prior_provider,
            evaluator_provider=evaluator_provider,
            feature_batch_prior_provider=feature_batch_prior_provider,
            request_evaluator_provider=request_evaluator_provider,
            core_batch_factory=core_batch_factory,
        )

    make_state = state_factory if state_factory is not None else default_state_factory
    games = [
        BatchedGame(
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

        batched_priors = batched_root_priors(active_games, prior_provider)
        for game, root_priors in zip(active_games, batched_priors, strict=True):
            play_batched_self_play_turn(
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
                f"self-play exceeded max_turns={config.max_turns} for seeds={unfinished}"
            )

    return [finish_batched_game(game) for game in games]


@dataclass
class BatchedGame:
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


def batched_root_priors(
    games: Sequence[BatchedGame],
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


def play_self_play_games_core_batched(
    *,
    seeds: Sequence[int],
    config: SelfPlayConfig,
    prior_provider: Callable[[Sequence[SelfPlayState]], Sequence[Sequence[float]]] | None,
    evaluator_provider: Callable[
        [Sequence[SelfPlayState]],
        tuple[Sequence[Sequence[float]], Sequence[float]],
    ]
    | None,
    feature_batch_prior_provider: Callable[
        [Sequence[Sequence[float]], Sequence[Sequence[bool]]],
        Sequence[Sequence[float]],
    ]
    | None,
    request_evaluator_provider: Callable[[Any], tuple[Any, Any]] | None,
    core_batch_factory: CoreBatchFactory,
) -> list[tuple[GameLog, list[ReplaySample]]]:
    if not seeds:
        return []

    batch = core_batch_factory(config, len(seeds))
    rngs = [random.Random(seed) for seed in seeds]
    moves: list[list[MoveLog]] = [[] for _ in seeds]
    pending_samples: list[list[tuple[int, np.ndarray, np.ndarray, np.ndarray | None]]] = [
        [] for _ in seeds
    ]

    for turn in range(config.max_turns):
        active_indexes = batch.active_game_indexes()
        if not active_indexes:
            break

        request = batch.active_eval_request()
        players = as_int_list(batch.current_players())
        feature_rows = request.feature_planes()
        masks = request.legal_masks()
        features_by_game = {
            game_index: flat_features_for_replay(feature_planes)
            for game_index, feature_planes in zip(active_indexes, feature_rows, strict=True)
        }
        priors, root_values = evaluate_root_rows(
            request=request,
            active_count=len(active_indexes),
            prior_provider=prior_provider,
            evaluator_provider=evaluator_provider,
            feature_batch_prior_provider=feature_batch_prior_provider,
            request_evaluator_provider=request_evaluator_provider,
            feature_rows=feature_rows,
            masks=masks,
        )
        noisy_priors: list[list[float]] = []
        root_policy_logits_by_game: dict[int, np.ndarray] = {}
        use_full_by_game: dict[int, bool] = {}
        simulation_budgets: list[int | None] = [None] * batch.len()
        for game_index, prior in zip(active_indexes, priors, strict=True):
            use_full = use_full_search_turn(rngs[game_index], config)
            use_full_by_game[game_index] = use_full
            if config.playout_cap_randomization:
                simulation_budgets[game_index] = (
                    config.playout_cap_full_simulations
                    if use_full
                    else config.playout_cap_fast_simulations
                )
            prior_values = [float(value) for value in prior]
            noisy_priors.append(prior_values)
            root_policy_logits_by_game[game_index] = np.asarray(prior_values, dtype=np.float32)

        if config.playout_cap_randomization:
            batch.set_simulations(simulation_budgets)
            batch.set_max_considered_actions(
                [
                    (
                        playout_cap_max_considered_actions(use_full_by_game[game_index], config)
                        if game_index in use_full_by_game
                        else None
                    )
                    for game_index in range(batch.len())
                ]
            )

        results = search_core_batch(
            batch,
            noisy_priors,
            root_values=root_values,
            config=config,
            evaluator_provider=evaluator_provider,
            request_evaluator_provider=request_evaluator_provider,
        )
        actions: list[int | None] = [None] * batch.len()
        for active_offset, game_index in enumerate(active_indexes):
            result = results[game_index]
            if result is None:
                continue
            policy = policy_target_from_result(result)
            action = select_self_play_action(
                result,
                legal_actions=[
                    action for action, is_legal in enumerate(masks[active_offset]) if is_legal
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
                f"self-play exceeded max_turns={config.max_turns} for seeds={unfinished}"
            )

    return finish_core_batch_games(batch, seeds, moves, pending_samples)


def evaluate_root_rows(
    *,
    request: Any,
    active_count: int,
    prior_provider: Callable[[Sequence[SelfPlayState]], Sequence[Sequence[float]]] | None,
    evaluator_provider: Callable[
        [Sequence[SelfPlayState]],
        tuple[Sequence[Sequence[float]], Sequence[float]],
    ]
    | None,
    feature_batch_prior_provider: Callable[
        [Sequence[Sequence[float]], Sequence[Sequence[bool]]],
        Sequence[Sequence[float]],
    ]
    | None,
    request_evaluator_provider: Callable[[Any], tuple[Any, Any]] | None,
    feature_rows: Sequence[Sequence[float]],
    masks: Sequence[Sequence[bool]],
) -> tuple[list[list[float]], list[float] | None]:
    root_values: list[float] | None = None
    if request_evaluator_provider is not None:
        root_policies, root_value_rows = request_evaluator_provider(request)
        priors = [[float(value) for value in row] for row in root_policies]
        root_values = [float(value) for value in root_value_rows]
    elif feature_batch_prior_provider is None:
        if prior_provider is None:
            raise ValueError("prior_provider is required for core batched self-play")
        priors = evaluate_core_batch_priors(prior_provider, request)
    else:
        priors = [
            [float(value) for value in row]
            for row in feature_batch_prior_provider(feature_rows, masks)
        ]
    if len(priors) != active_count:
        raise ValueError(
            f"expected {active_count} prior rows from batch provider, got {len(priors)}"
        )
    if root_values is None and evaluator_provider is not None:
        _root_policies, root_values = evaluate_core_batch_policy_values(
            evaluator_provider,
            request,
        )
    if root_values is not None and len(root_values) != active_count:
        raise ValueError(
            f"expected {active_count} root values from batch provider, got {len(root_values)}"
        )
    return priors, root_values


def search_core_batch(
    batch: SelfPlayBatchLike,
    noisy_priors: list[list[float]],
    *,
    root_values: list[float] | None,
    config: SelfPlayConfig,
    evaluator_provider: Callable[
        [Sequence[SelfPlayState]],
        tuple[Sequence[Sequence[float]], Sequence[float]],
    ]
    | None,
    request_evaluator_provider: Callable[[Any], tuple[Any, Any]] | None,
) -> list[SearchResultLike | None]:
    if evaluator_provider is None and request_evaluator_provider is None:
        return batch.search_active_with_logits(noisy_priors)

    def evaluator(request: Any) -> tuple[Any, Any]:
        if request_evaluator_provider is not None:
            return request_evaluator_provider(request)
        if evaluator_provider is None:
            raise RuntimeError("evaluator_provider is required")
        return evaluate_core_batch_policy_values(evaluator_provider, request)

    if root_values is None:
        raise ValueError("Gumbel batched evaluator search requires root values")
    return batch.search_active_with_logits_and_evaluator(
        noisy_priors,
        evaluator,
        root_values=root_values,
        leaf_batch_size=config.leaf_batch_size,
    )


def run_self_play_search(
    state: SelfPlayState,
    search: SearchLike,
    config: SelfPlayConfig,
    *,
    root_priors: Sequence[float] | None,
    evaluator_provider: Callable[
        [Sequence[SelfPlayState]],
        tuple[Sequence[Sequence[float]], Sequence[float]],
    ]
    | None,
) -> tuple[SearchResultLike, np.ndarray]:
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
        return evaluate_core_batch_policy_values(evaluator_provider, request)

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


def play_batched_self_play_turn(
    game: BatchedGame,
    turn: int,
    config: SelfPlayConfig,
    *,
    root_priors: Sequence[float] | None,
    evaluator_provider: Callable[
        [Sequence[SelfPlayState]],
        tuple[Sequence[Sequence[float]], Sequence[float]],
    ]
    | None,
) -> None:
    player = game.state.current_player()
    features = flat_features_for_replay(game.state.feature_planes())
    use_full_search = use_full_search_turn(game.rng, config)
    if config.playout_cap_randomization:
        simulations = (
            config.playout_cap_full_simulations
            if use_full_search
            else config.playout_cap_fast_simulations
        )
        game.search.set_simulations(simulations)
        game.search.set_max_considered_actions(
            playout_cap_max_considered_actions(use_full_search, config)
        )

    result, root_policy_logits = run_self_play_search(
        game.state,
        game.search,
        config,
        root_priors=root_priors,
        evaluator_provider=evaluator_provider,
    )
    policy = policy_target_from_result(result)
    action = select_self_play_action(result, legal_actions=game.state.legal_actions())

    if use_full_search:
        assert game.pending_samples is not None
        game.pending_samples.append((player, features, policy, root_policy_logits))
    assert game.moves is not None
    game.moves.append(MoveLog(turn=turn, player=player, action=action))
    game.state.apply_action(action)


def finish_batched_game(game: BatchedGame) -> tuple[GameLog, list[ReplaySample]]:
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


def finish_core_batch_games(
    batch: SelfPlayBatchLike,
    seeds: Sequence[int],
    moves: Sequence[list[MoveLog]],
    pending_samples: Sequence[list[tuple[int, np.ndarray, np.ndarray, np.ndarray | None]]],
) -> list[tuple[GameLog, list[ReplaySample]]]:
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


def use_full_search_turn(rng: random.Random, config: SelfPlayConfig) -> bool:
    if not config.playout_cap_randomization:
        return True
    return rng.random() < config.playout_cap_full_search_fraction


def playout_cap_max_considered_actions(use_full_search: bool, config: SelfPlayConfig) -> int:
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


def policy_target_from_result(result: SearchResultLike) -> np.ndarray:
    if hasattr(result, "policy_target"):
        policy = np.asarray(cast(Any, result).policy_target(), dtype=np.float32)
        if policy.shape != (PASS_ACTION + 1,):
            raise ValueError(
                f"expected policy target shape {(PASS_ACTION + 1,)}, got {policy.shape}"
            )
        return policy
    return policy_target_from_visit_counts(result.visit_counts())


def select_self_play_action(
    result: SearchResultLike,
    *,
    legal_actions: Sequence[int],
) -> int:
    selected = result.selected_action()
    if selected in set(legal_actions):
        return int(selected)
    visits = result.visit_counts()
    return max(legal_actions, key=lambda action: (visits[action], -action))
