"""Arena game and batch orchestration."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from great_kingdom_ai.arena_types import (
    BLUE,
    ORANGE,
    ArenaBatchLike,
    ArenaConfig,
    ArenaGameResult,
    ArenaOnnxEvaluators,
    ArenaReport,
    ArenaSearchLike,
    ArenaSearchResultLike,
    ArenaSummary,
    arena_game_seed,
    other_player,
    validate_arena_config,
)
from great_kingdom_ai.game_core import (
    active_legal_masks,
    as_int_list,
    deterministic_action,
    request_current_players,
    request_feature_rows_and_masks,
    request_game_indexes,
)
from great_kingdom_ai.self_play_types import MoveLog, SelfPlayState


@dataclass(frozen=True)
class ArenaRuntime:
    create_game_state: Callable[[], SelfPlayState]
    create_search_engine: Callable[[ArenaConfig, int], ArenaSearchLike]
    create_batch: Callable[[ArenaConfig, int, int, int], ArenaBatchLike]
    evaluate_feature_batch: Callable[..., Any]
    evaluate_feature_batch_logits_values: Callable[..., Any]
    evaluate_feature_arrays_logits_values: Callable[..., Any]


def play_arena_game(
    *,
    seed: int,
    candidate_model: Any,
    best_model: Any,
    candidate_player: int,
    config: ArenaConfig,
    state: SelfPlayState | None,
    search_factory: Callable[[], ArenaSearchLike] | None,
    runtime: ArenaRuntime,
) -> ArenaGameResult:
    if candidate_player not in {BLUE, ORANGE}:
        raise ValueError("candidate_player must be 1 or 2")

    game_state = state if state is not None else runtime.create_game_state()
    searches = (
        {
            BLUE: runtime.create_search_engine(config, seed * 2),
            ORANGE: runtime.create_search_engine(config, seed * 2 + 1),
        }
        if search_factory is None
        else {BLUE: search_factory(), ORANGE: search_factory()}
    )
    best_player = other_player(candidate_player)
    moves: list[MoveLog] = []

    for turn in range(config.max_turns):
        if game_state.is_terminal():
            break

        apply_arena_turn_gumbel_scale(searches.values(), config=config, turn=turn)
        player = game_state.current_player()
        model = candidate_model if player == candidate_player else best_model
        root_evaluation = runtime.evaluate_feature_batch_logits_values(
            model,
            [game_state.feature_planes()],
            [game_state.legal_mask()],
            device=config.device,
        )
        logits = [float(value) for value in root_evaluation.policy_logits[0]]
        root_value = float(root_evaluation.value[0])

        def evaluator(request: Any, m: Any = model) -> tuple[list[list[float]], list[float]]:
            evaluation = runtime.evaluate_feature_batch_logits_values(
                m,
                request.feature_planes(),
                request.legal_masks(),
                device=config.device,
            )
            return (
                [[float(value) for value in policy] for policy in evaluation.policy_logits],
                [float(value) for value in evaluation.value],
            )

        result = searches[player].search_with_logits_and_evaluator(
            game_state,
            logits,
            evaluator,
            root_value,
            config.leaf_batch_size,
        )
        action = deterministic_action(result, logits, game_state.legal_actions())
        moves.append(MoveLog(turn=turn, player=player, action=action))
        game_state.apply_action(action)
    else:
        raise RuntimeError(f"arena game exceeded max_turns={config.max_turns}")

    winner = game_state.winner()
    end_reason = game_state.end_reason()
    if winner is None or end_reason is None:
        raise RuntimeError("arena game stopped before terminal outcome")

    return ArenaGameResult(
        seed=seed,
        candidate_player=candidate_player,
        best_player=best_player,
        winner=winner,
        end_reason=end_reason,
        moves=moves,
        territory_scores=game_state.territory_scores(),
    )


def run_arena(
    *,
    candidate_model: Any,
    best_model: Any,
    config: ArenaConfig | None,
    state_factory: Callable[[], SelfPlayState] | None,
    search_factory: Callable[[], ArenaSearchLike] | None,
    progress_callback: Callable[[int, int, ArenaGameResult], None] | None,
    batched_runner: Callable[..., ArenaReport],
    runtime: ArenaRuntime,
) -> ArenaReport:
    config = config if config is not None else ArenaConfig()
    validate_arena_config(config)
    if config.batch_size > 1:
        if state_factory is not None or search_factory is not None:
            raise ValueError(
                "state_factory and search_factory are only supported for batch_size=1"
            )
        return batched_runner(
            candidate_model=candidate_model,
            best_model=best_model,
            config=config,
            progress_callback=progress_callback,
        )

    make_state = state_factory if state_factory is not None else runtime.create_game_state
    games = []
    for index in range(config.games):
        game = play_arena_game(
            seed=arena_game_seed(config, index),
            candidate_model=candidate_model,
            best_model=best_model,
            candidate_player=BLUE if index % 2 == 0 else ORANGE,
            config=config,
            state=make_state(),
            search_factory=search_factory,
            runtime=runtime,
        )
        games.append(game)
        if progress_callback is not None:
            progress_callback(index + 1, config.games, game)
    return ArenaReport(
        config=config,
        games=games,
        summary=summarize_arena(games, promotion_threshold=config.promotion_threshold),
    )


def run_arena_batched(
    *,
    candidate_model: Any,
    best_model: Any,
    config: ArenaConfig | None,
    onnx_evaluators: ArenaOnnxEvaluators | None,
    progress_callback: Callable[[int, int, ArenaGameResult], None] | None,
    runtime: ArenaRuntime,
) -> ArenaReport:
    config = config if config is not None else ArenaConfig()
    validate_arena_config(config)

    games: list[ArenaGameResult | None] = [None] * config.games
    emitted_games = 0
    for chunk_start in range(0, config.games, config.batch_size):
        chunk_size = min(config.batch_size, config.games - chunk_start)
        batch = runtime.create_batch(
            config,
            chunk_size,
            config.seed_start if config.paired_seeds else config.seed_start + chunk_start,
            chunk_start,
        )
        moves: list[list[MoveLog]] = [[] for _ in range(chunk_size)]
        emitted_in_chunk = 0

        for turn in range(config.max_turns):
            active_indexes = batch.active_game_indexes()
            if not active_indexes:
                break

            apply_arena_batch_turn_gumbel_scale(batch, config=config, turn=turn)
            current_players = as_int_list(batch.current_players())
            candidate_players = as_int_list(batch.candidate_players())
            results, root_logits, masks = search_arena_batch_turn(
                batch=batch,
                candidate_model=candidate_model,
                best_model=best_model,
                config=config,
                onnx_evaluators=onnx_evaluators,
                current_players=current_players,
                candidate_players=candidate_players,
                runtime=runtime,
            )
            actions: list[int | None] = [None] * batch.len()
            for active_offset, game_index in enumerate(active_indexes):
                search_result = results[game_index]
                if search_result is None:
                    continue
                legal_actions = [
                    action for action, is_legal in enumerate(masks[active_offset]) if is_legal
                ]
                action = deterministic_action(
                    search_result,
                    root_logits[active_offset],
                    legal_actions,
                )
                moves[game_index].append(
                    MoveLog(turn=turn, player=current_players[game_index], action=action)
                )
                actions[game_index] = action
            batch.apply_actions(actions)

            chunk_results = finished_arena_batch_results(
                batch=batch,
                config=config,
                chunk_start=chunk_start,
                moves=moves,
            )
            while emitted_in_chunk < chunk_size:
                game_result = chunk_results[emitted_in_chunk]
                if game_result is None:
                    break
                global_index = chunk_start + emitted_in_chunk
                if games[global_index] is None:
                    games[global_index] = game_result
                    emitted_games += 1
                    if progress_callback is not None:
                        progress_callback(emitted_games, config.games, game_result)
                emitted_in_chunk += 1

            if not batch.active_game_indexes():
                break
        else:
            unfinished = [
                config.seed_start + chunk_start + index
                for index, is_terminal in enumerate(batch.is_terminal())
                if not is_terminal
            ]
            if unfinished:
                raise RuntimeError(
                    f"arena batch exceeded max_turns={config.max_turns} for seeds={unfinished}"
                )

    completed_games = [game for game in games if game is not None]
    if len(completed_games) != config.games:
        raise RuntimeError("arena batch stopped before all games reached terminal outcomes")
    return ArenaReport(
        config=config,
        games=completed_games,
        summary=summarize_arena(
            completed_games,
            promotion_threshold=config.promotion_threshold,
        ),
    )


def search_arena_batch_turn(
    *,
    batch: ArenaBatchLike,
    candidate_model: Any,
    best_model: Any,
    config: ArenaConfig,
    onnx_evaluators: ArenaOnnxEvaluators | None,
    current_players: Sequence[int],
    candidate_players: Sequence[int],
    runtime: ArenaRuntime,
) -> tuple[list[ArenaSearchResultLike | None], list[list[float]], list[list[bool]]]:
    active_indexes = batch.active_game_indexes()
    if onnx_evaluators is not None:
        masks = active_legal_masks(batch)
        if not hasattr(batch, "search_active_with_onnx_evaluators"):
            raise RuntimeError(
                "great_kingdom_core.GumbelArenaBatch does not support ONNX arena. "
                "Rebuild the Rust extension."
            )
        results, root_logits = batch.search_active_with_onnx_evaluators(
            onnx_evaluators.candidate,
            onnx_evaluators.best,
            leaf_batch_size=config.leaf_batch_size,
        )
        return results, root_logits, masks

    request = batch.active_eval_request()
    feature_rows, masks, row_count = request_feature_rows_and_masks(request)
    root_logits, root_values = evaluate_arena_rows_by_model(
        candidate_model=candidate_model,
        best_model=best_model,
        feature_rows=feature_rows,
        legal_masks=masks,
        game_indexes=active_indexes,
        current_players=[current_players[index] for index in active_indexes],
        candidate_players=candidate_players,
        device=config.device,
        runtime=runtime,
    )

    def evaluator(
        leaf_request: Any,
        candidate_players: Sequence[int] = candidate_players,
    ) -> tuple[list[list[float]], list[float]]:
        leaf_features, leaf_masks, leaf_row_count = request_feature_rows_and_masks(leaf_request)
        return evaluate_arena_rows_by_model(
            candidate_model=candidate_model,
            best_model=best_model,
            feature_rows=leaf_features,
            legal_masks=leaf_masks,
            game_indexes=request_game_indexes(leaf_request, expected_len=leaf_row_count),
            current_players=request_current_players(leaf_request, expected_len=leaf_row_count),
            candidate_players=candidate_players,
            device=config.device,
            runtime=runtime,
        )

    results = batch.search_active_with_logits_and_evaluator(
        root_logits,
        evaluator,
        root_values=root_values,
        leaf_batch_size=config.leaf_batch_size,
    )
    if row_count != len(active_indexes):
        raise RuntimeError("arena root eval request metadata length did not match active games")
    if isinstance(masks, np.ndarray):
        mask_rows = [[bool(value) for value in row] for row in masks]
    else:
        mask_rows = [[bool(value) for value in row] for row in masks]
    return results, root_logits, mask_rows


def summarize_arena(
    games: Sequence[ArenaGameResult],
    *,
    promotion_threshold: float,
) -> ArenaSummary:
    if not 0.0 <= promotion_threshold <= 1.0:
        raise ValueError("promotion_threshold must be between 0 and 1")

    game_count = len(games)
    candidate_wins = sum(1 for game in games if game.winner == game.candidate_player)
    best_wins = sum(1 for game in games if game.winner == game.best_player)
    candidate_blue_games = sum(1 for game in games if game.candidate_player == BLUE)
    candidate_blue_wins = sum(
        1 for game in games if game.candidate_player == BLUE and game.winner == BLUE
    )
    candidate_orange_games = sum(1 for game in games if game.candidate_player == ORANGE)
    candidate_orange_wins = sum(
        1 for game in games if game.candidate_player == ORANGE and game.winner == ORANGE
    )
    candidate_win_rate = candidate_wins / game_count if game_count else 0.0
    best_win_rate = best_wins / game_count if game_count else 0.0
    average_game_length = (
        sum(len(game.moves) for game in games) / game_count if game_count else 0.0
    )

    return ArenaSummary(
        games=game_count,
        candidate_wins=candidate_wins,
        best_wins=best_wins,
        candidate_win_rate=candidate_win_rate,
        best_win_rate=best_win_rate,
        candidate_blue_games=candidate_blue_games,
        candidate_blue_wins=candidate_blue_wins,
        candidate_orange_games=candidate_orange_games,
        candidate_orange_wins=candidate_orange_wins,
        average_game_length=average_game_length,
        promoted=game_count > 0 and candidate_win_rate >= promotion_threshold,
    )


def evaluate_state_policies(
    model: Any,
    states: Sequence[SelfPlayState],
    *,
    device: Any | str | None,
    runtime: ArenaRuntime,
) -> list[list[float]]:
    evaluation = runtime.evaluate_feature_batch(
        model,
        [state.feature_planes() for state in states],
        [state.legal_mask() for state in states],
        device=device,
    )
    return [[float(value) for value in policy] for policy in evaluation.policy]


def evaluate_state_policy_logits_batch(
    model: Any,
    states: Sequence[SelfPlayState],
    *,
    device: Any | str | None,
    runtime: ArenaRuntime,
) -> list[list[float]]:
    evaluation = runtime.evaluate_feature_batch_logits_values(
        model,
        [state.feature_planes() for state in states],
        [state.legal_mask() for state in states],
        device=device,
    )
    return [[float(value) for value in row] for row in evaluation.policy_logits]


def evaluate_arena_rows_by_model(
    *,
    candidate_model: Any,
    best_model: Any,
    feature_rows: Any,
    legal_masks: Any,
    game_indexes: Sequence[int],
    current_players: Sequence[int],
    candidate_players: Sequence[int],
    device: Any | str | None,
    runtime: ArenaRuntime,
) -> tuple[list[list[float]], list[float]]:
    row_count = len(feature_rows)
    if len(legal_masks) != row_count:
        raise ValueError("feature batch and legal mask batch must have the same length")
    if len(game_indexes) != row_count or len(current_players) != row_count:
        raise RuntimeError("arena eval request metadata length did not match request rows")

    logits_by_row: list[list[float] | None] = [None] * row_count
    values_by_row: list[float | None] = [None] * row_count
    candidate_offsets: list[int] = []
    best_offsets: list[int] = []
    for offset, (game_index, player) in enumerate(zip(game_indexes, current_players, strict=True)):
        if game_index < 0 or game_index >= len(candidate_players):
            raise RuntimeError(f"arena eval request game index out of range: {game_index}")
        if player == candidate_players[game_index]:
            candidate_offsets.append(offset)
        else:
            best_offsets.append(offset)

    def evaluate_offsets(model: Any, offsets: Sequence[int]) -> None:
        if not offsets:
            return
        if isinstance(feature_rows, np.ndarray) and isinstance(legal_masks, np.ndarray):
            evaluation = runtime.evaluate_feature_arrays_logits_values(
                model,
                feature_rows[list(offsets)],
                legal_masks[list(offsets)],
                device=device,
            )
        else:
            evaluation = runtime.evaluate_feature_batch_logits_values(
                model,
                [[float(value) for value in feature_rows[offset]] for offset in offsets],
                [[bool(value) for value in legal_masks[offset]] for offset in offsets],
                device=device,
            )
        if len(evaluation.policy_logits) != len(offsets) or len(evaluation.value) != len(offsets):
            raise ValueError("model evaluation returned a mismatched batch size")
        for offset, policy, value in zip(
            offsets,
            evaluation.policy_logits,
            evaluation.value,
            strict=True,
        ):
            logits_by_row[offset] = [float(item) for item in policy]
            values_by_row[offset] = float(value)

    evaluate_offsets(candidate_model, candidate_offsets)
    evaluate_offsets(best_model, best_offsets)
    return (
        [cast(list[float], row) for row in logits_by_row],
        [cast(float, value) for value in values_by_row],
    )


def finished_arena_batch_results(
    *,
    batch: ArenaBatchLike,
    config: ArenaConfig,
    chunk_start: int,
    moves: Sequence[list[MoveLog]],
) -> list[ArenaGameResult | None]:
    candidate_players = as_int_list(batch.candidate_players())
    seeds = (
        as_int_list(batch.seeds())
        if hasattr(batch, "seeds")
        else [
            arena_game_seed(config, chunk_start + game_index)
            for game_index in range(batch.len())
        ]
    )
    winners = batch.winners()
    end_reasons = batch.end_reasons()
    terminal = batch.is_terminal()
    scores = batch.territory_scores()
    results: list[ArenaGameResult | None] = []
    for game_index, is_terminal in enumerate(terminal):
        if not is_terminal:
            results.append(None)
            continue
        winner = winners[game_index]
        end_reason = end_reasons[game_index]
        if winner is None or end_reason is None:
            raise RuntimeError("arena game stopped before terminal outcome")
        candidate_player = candidate_players[game_index]
        results.append(
            ArenaGameResult(
                seed=seeds[game_index],
                candidate_player=candidate_player,
                best_player=other_player(candidate_player),
                winner=int(winner),
                end_reason=int(end_reason),
                moves=list(moves[game_index]),
                territory_scores=scores[game_index],
            )
        )
    return results


def arena_turn_gumbel_scale(config: ArenaConfig, turn: int) -> float:
    if turn < config.opening_gumbel_turns:
        return config.opening_gumbel_scale
    return config.gumbel_scale


def apply_arena_turn_gumbel_scale(
    searches: Iterable[Any],
    *,
    config: ArenaConfig,
    turn: int,
) -> None:
    if config.opening_gumbel_turns <= 0:
        return
    scale = arena_turn_gumbel_scale(config, turn)
    for search in searches:
        setter = getattr(search, "set_gumbel_scale", None)
        if setter is None:
            raise RuntimeError(
                "arena opening_gumbel_turns requires a GumbelSearch backend with set_gumbel_scale"
            )
        setter(scale)


def apply_arena_batch_turn_gumbel_scale(
    batch: ArenaBatchLike,
    *,
    config: ArenaConfig,
    turn: int,
) -> None:
    if config.opening_gumbel_turns <= 0:
        return
    setter = getattr(batch, "set_gumbel_scale", None)
    if setter is None:
        raise RuntimeError(
            "arena opening_gumbel_turns requires a GumbelArenaBatch backend with set_gumbel_scale"
        )
    setter(arena_turn_gumbel_scale(config, turn))
