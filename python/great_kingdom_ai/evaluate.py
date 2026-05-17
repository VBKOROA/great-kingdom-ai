"""Arena evaluation and best-model promotion helpers."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NoReturn, Protocol, cast

import numpy as np

from great_kingdom_ai.evaluator import (
    evaluate_feature_arrays_logits_values,
    evaluate_feature_batch,
    evaluate_feature_batch_logits_values,
)
from great_kingdom_ai.features import ACTION_SPACE, BOARD_SIZE, FEATURE_CHANNELS
from great_kingdom_ai.onnx_export import export_checkpoint_to_onnx
from great_kingdom_ai.self_play import MoveLog, SelfPlayState, create_core_game_state

BLUE = 1
ORANGE = 2


class ArenaSearchResultLike(Protocol):
    def selected_action(self) -> int | None: ...

    def visit_counts(self) -> list[int]: ...


class ArenaSearchLike(Protocol):
    def search_with_logits_and_evaluator(
        self,
        state: SelfPlayState,
        policy_logits: list[float],
        evaluator: Callable[[Any], tuple[list[list[float]], list[float]]],
        root_value: float,
        leaf_batch_size: int = 8,
    ) -> ArenaSearchResultLike: ...


class ArenaBatchLike(Protocol):
    def len(self) -> int: ...

    def active_game_indexes(self) -> list[int]: ...

    def active_eval_request(self) -> Any: ...

    def active_legal_masks(self) -> list[list[bool]]: ...

    def current_players(self) -> list[int] | bytes: ...

    def candidate_players(self) -> list[int] | bytes: ...

    def seeds(self) -> list[int] | bytes: ...

    def search_active_with_logits_and_evaluator(
        self,
        policy_logits: list[list[float]],
        evaluator: Callable[[Any], tuple[list[list[float]], list[float]]],
        root_values: list[float],
        leaf_batch_size: int = 8,
    ) -> list[ArenaSearchResultLike | None]: ...

    def search_active_with_onnx_evaluators(
        self,
        candidate_evaluator: Any,
        best_evaluator: Any,
        leaf_batch_size: int = 8,
    ) -> tuple[list[ArenaSearchResultLike | None], list[list[float]]]: ...

    def apply_actions(self, actions: list[int | None]) -> list[int | None]: ...

    def is_terminal(self) -> list[bool]: ...

    def winners(self) -> list[int | None]: ...

    def end_reasons(self) -> list[int | None]: ...

    def territory_scores(self) -> list[tuple[int, int]]: ...


@dataclass
class ArenaOnnxEvaluators:
    candidate: Any
    best: Any


@dataclass(frozen=True)
class ArenaConfig:
    games: int = 20
    batch_size: int = 1
    seed_start: int = 0
    max_turns: int = 200
    gumbel_simulations: int = 128
    gumbel_max_considered_actions: int = 16
    gumbel_c_visit: float = 50.0
    gumbel_c_scale: float = 1.0
    gumbel_scale: float = 0.0
    opening_gumbel_turns: int = 0
    opening_gumbel_scale: float = 1.0
    policy_target_c_visit: float = 5.0
    policy_target_c_scale: float = 0.25
    policy_target_temperature: float = 1.0
    gumbel_seed: int = 0
    paired_seeds: bool = False
    leaf_batch_size: int = 8
    device: str = "cpu"
    promotion_threshold: float = 0.55

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")


@dataclass(frozen=True)
class ArenaGameResult:
    seed: int
    candidate_player: int
    best_player: int
    winner: int
    end_reason: int
    moves: list[MoveLog]
    territory_scores: tuple[int, int]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["moves"] = [move.__dict__ for move in self.moves]
        return data


@dataclass(frozen=True)
class ArenaSummary:
    games: int
    candidate_wins: int
    best_wins: int
    candidate_win_rate: float
    best_win_rate: float
    candidate_blue_games: int
    candidate_blue_wins: int
    candidate_orange_games: int
    candidate_orange_wins: int
    average_game_length: float
    promoted: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ArenaReport:
    config: ArenaConfig
    games: list[ArenaGameResult]
    summary: ArenaSummary

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "games": [game.to_dict() for game in self.games],
            "summary": self.summary.to_dict(),
        }


def play_arena_game(
    *,
    seed: int,
    candidate_model: Any,
    best_model: Any,
    candidate_player: int,
    config: ArenaConfig,
    state: SelfPlayState | None = None,
    search_factory: Callable[[], ArenaSearchLike] | None = None,
) -> ArenaGameResult:
    """Play one deterministic candidate-vs-best game.

    Evaluation uses model logits and deterministic action selection.
    """
    if candidate_player not in {BLUE, ORANGE}:
        raise ValueError("candidate_player must be 1 or 2")

    game_state = state if state is not None else create_core_game_state()
    if search_factory is None:
        searches = {
            BLUE: create_core_search_engine(config, seed_offset=seed * 2),
            ORANGE: create_core_search_engine(config, seed_offset=seed * 2 + 1),
        }
    else:
        searches = {BLUE: search_factory(), ORANGE: search_factory()}
    best_player = _other_player(candidate_player)
    moves: list[MoveLog] = []

    for turn in range(config.max_turns):
        if game_state.is_terminal():
            break

        _apply_arena_turn_gumbel_scale(searches.values(), config=config, turn=turn)
        player = game_state.current_player()
        model = candidate_model if player == candidate_player else best_model
        root_evaluation = evaluate_feature_batch_logits_values(
            model,
            [game_state.feature_planes()],
            [game_state.legal_mask()],
            device=config.device,
        )
        root_logits = root_evaluation.policy_logits[0]
        logits = [float(value) for value in root_logits]
        root_value = float(root_evaluation.value[0])

        def evaluator(request: Any, m: Any = model) -> tuple[list[list[float]], list[float]]:
            feature_rows = request.feature_planes()
            mask_rows = request.legal_masks()
            evaluation = evaluate_feature_batch_logits_values(
                m, feature_rows, mask_rows, device=config.device
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
        action = _deterministic_action(result, logits, game_state.legal_actions())

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
    config: ArenaConfig | None = None,
    state_factory: Callable[[], SelfPlayState] | None = None,
    search_factory: Callable[[], ArenaSearchLike] | None = None,
    progress_callback: Callable[[int, int, ArenaGameResult], None] | None = None,
) -> ArenaReport:
    config = config if config is not None else ArenaConfig()
    _validate_arena_config(config)
    if config.batch_size > 1:
        if state_factory is not None or search_factory is not None:
            raise ValueError(
                "state_factory and search_factory are only supported for batch_size=1"
            )
        return run_arena_batched(
            candidate_model=candidate_model,
            best_model=best_model,
            config=config,
            progress_callback=progress_callback,
        )

    make_state = state_factory if state_factory is not None else create_core_game_state
    games = []
    for index in range(config.games):
        game = play_arena_game(
            seed=_arena_game_seed(config, index),
            candidate_model=candidate_model,
            best_model=best_model,
            candidate_player=BLUE if index % 2 == 0 else ORANGE,
            config=config,
            state=make_state(),
            search_factory=search_factory,
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
    config: ArenaConfig | None = None,
    onnx_evaluators: ArenaOnnxEvaluators | None = None,
    progress_callback: Callable[[int, int, ArenaGameResult], None] | None = None,
) -> ArenaReport:
    config = config if config is not None else ArenaConfig()
    _validate_arena_config(config)

    games: list[ArenaGameResult | None] = [None] * config.games
    emitted_games = 0
    for chunk_start in range(0, config.games, config.batch_size):
        chunk_size = min(config.batch_size, config.games - chunk_start)
        batch = create_core_arena_batch(
            config,
            game_count=chunk_size,
            seed_start=(
                config.seed_start
                if config.paired_seeds
                else config.seed_start + chunk_start
            ),
            game_index_start=chunk_start,
        )
        moves: list[list[MoveLog]] = [[] for _ in range(chunk_size)]
        emitted_in_chunk = 0

        for turn in range(config.max_turns):
            active_indexes = batch.active_game_indexes()
            if not active_indexes:
                break

            _apply_arena_batch_turn_gumbel_scale(batch, config=config, turn=turn)
            current_players = _as_int_list(batch.current_players())
            candidate_players = _as_int_list(batch.candidate_players())
            if onnx_evaluators is not None:
                masks = _active_legal_masks(batch)
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
            else:
                request = batch.active_eval_request()
                feature_rows, masks, row_count = _request_feature_rows_and_masks(request)
                root_logits, root_values = _evaluate_arena_rows_by_model(
                    candidate_model=candidate_model,
                    best_model=best_model,
                    feature_rows=feature_rows,
                    legal_masks=masks,
                    game_indexes=active_indexes,
                    current_players=[current_players[index] for index in active_indexes],
                    candidate_players=candidate_players,
                    device=config.device,
                )

                def evaluator(
                    leaf_request: Any,
                    candidate_players: list[int] = candidate_players,
                ) -> tuple[list[list[float]], list[float]]:
                    leaf_features, leaf_masks, leaf_row_count = _request_feature_rows_and_masks(
                        leaf_request
                    )
                    leaf_game_indexes = _request_game_indexes(
                        leaf_request,
                        expected_len=leaf_row_count,
                    )
                    leaf_players = _request_current_players(
                        leaf_request,
                        expected_len=leaf_row_count,
                    )
                    return _evaluate_arena_rows_by_model(
                        candidate_model=candidate_model,
                        best_model=best_model,
                        feature_rows=leaf_features,
                        legal_masks=leaf_masks,
                        game_indexes=leaf_game_indexes,
                        current_players=leaf_players,
                        candidate_players=candidate_players,
                        device=config.device,
                    )

                results = batch.search_active_with_logits_and_evaluator(
                    root_logits,
                    evaluator,
                    root_values=root_values,
                    leaf_batch_size=config.leaf_batch_size,
                )
            actions: list[int | None] = [None] * batch.len()
            for active_offset, game_index in enumerate(active_indexes):
                search_result = results[game_index]
                if search_result is None:
                    continue
                legal_actions = [
                    action for action, is_legal in enumerate(masks[active_offset]) if is_legal
                ]
                action = _deterministic_action(
                    search_result,
                    root_logits[active_offset],
                    legal_actions,
                )
                moves[game_index].append(
                    MoveLog(
                        turn=turn,
                        player=current_players[game_index],
                        action=action,
                    )
                )
                actions[game_index] = action
            batch.apply_actions(actions)

            chunk_results = _finished_arena_batch_results(
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
                    f"arena batch exceeded max_turns={config.max_turns} "
                    f"for seeds={unfinished}"
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


def evaluate_state_policy(
    model: Any,
    state: SelfPlayState,
    *,
    device: Any | str | None = None,
) -> list[float]:
    return evaluate_state_policies(model, [state], device=device)[0]


def evaluate_state_policy_logits(
    model: Any,
    state: SelfPlayState,
    *,
    device: Any | str | None = None,
) -> list[float]:
    return evaluate_state_policy_logits_batch(model, [state], device=device)[0]


def evaluate_state_policies(
    model: Any,
    states: Sequence[SelfPlayState],
    *,
    device: Any | str | None = None,
) -> list[list[float]]:
    evaluation = evaluate_feature_batch(
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
    device: Any | str | None = None,
) -> list[list[float]]:
    evaluation = evaluate_feature_batch_logits_values(
        model,
        [state.feature_planes() for state in states],
        [state.legal_mask() for state in states],
        device=device,
    )
    return [[float(value) for value in row] for row in evaluation.policy_logits]


def save_arena_report(report: ArenaReport, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as file:
        json.dump(report.to_dict(), file, indent=2, sort_keys=True)
    return destination


def promote_candidate_if_needed(
    *,
    candidate_checkpoint: str | Path,
    best_checkpoint: str | Path,
    report: ArenaReport,
) -> bool:
    if not report.summary.promoted:
        return False
    source = Path(candidate_checkpoint)
    destination = Path(best_checkpoint)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return True


def load_model_from_checkpoint(
    path: str | Path,
    *,
    device: str = "cpu",
    prefer_ema: bool = True,
) -> Any:
    from great_kingdom_ai.training import load_checkpoint

    state = load_checkpoint(path, device=device, prefer_ema=prefer_ema)
    state.model.eval()
    return state.model


def create_onnx_evaluator(
    path: str | Path,
    *,
    device: str = "cpu",
    max_batch_size: int = 8192,
) -> Any:
    try:
        import great_kingdom_core as core
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "great_kingdom_core is not installed. Build it with maturin before ONNX arena."
        ) from exc
    if not hasattr(core, "OnnxEvaluator"):
        raise RuntimeError(
            "great_kingdom_core.OnnxEvaluator is not available. "
            "Rebuild the Rust extension with ONNX support."
        )
    return core.OnnxEvaluator(str(path), device=device, max_batch_size=max_batch_size)


def run_arena_onnx(
    *,
    candidate_onnx_path: str | Path,
    best_onnx_path: str | Path,
    config: ArenaConfig | None = None,
    onnx_max_batch_size: int = 8192,
    progress_callback: Callable[[int, int, ArenaGameResult], None] | None = None,
) -> ArenaReport:
    config = config if config is not None else ArenaConfig()
    _validate_arena_config(config)
    if config.batch_size <= 1:
        config = ArenaConfig(**{**asdict(config), "batch_size": max(1, config.games)})
    evaluators = ArenaOnnxEvaluators(
        candidate=create_onnx_evaluator(
            candidate_onnx_path,
            device=config.device,
            max_batch_size=onnx_max_batch_size,
        ),
        best=create_onnx_evaluator(
            best_onnx_path,
            device=config.device,
            max_batch_size=onnx_max_batch_size,
        ),
    )
    return run_arena_batched(
        candidate_model=None,
        best_model=None,
        config=config,
        onnx_evaluators=evaluators,
        progress_callback=progress_callback,
    )


def run_arena_checkpoints_onnx(
    *,
    candidate_checkpoint: str | Path,
    best_checkpoint: str | Path,
    config: ArenaConfig | None = None,
    onnx_max_batch_size: int = 8192,
    onnx_precision: str = "fp16",
    progress_callback: Callable[[int, int, ArenaGameResult], None] | None = None,
) -> ArenaReport:
    config = config if config is not None else ArenaConfig()
    with tempfile.TemporaryDirectory(prefix="gka-arena-onnx-") as temp_dir:
        temp_path = Path(temp_dir)
        candidate_onnx = _arena_onnx_path(
            candidate_checkpoint,
            temp_path / "candidate.onnx",
            device=config.device,
            precision=onnx_precision,
        )
        best_onnx = _arena_onnx_path(
            best_checkpoint,
            temp_path / "best.onnx",
            device=config.device,
            precision=onnx_precision,
        )
        return run_arena_onnx(
            candidate_onnx_path=candidate_onnx,
            best_onnx_path=best_onnx,
            config=config,
            onnx_max_batch_size=onnx_max_batch_size,
            progress_callback=progress_callback,
        )


def create_core_search_engine(
    config: ArenaConfig,
    *,
    seed_offset: int = 0,
) -> ArenaSearchLike:
    try:
        import great_kingdom_core as core  # type: ignore[import-untyped]
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "great_kingdom_core is not installed. Build it with maturin before arena evaluation."
        ) from exc

    return cast(
        ArenaSearchLike,
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


def create_core_arena_batch(
    config: ArenaConfig,
    *,
    game_count: int,
    seed_start: int,
    game_index_start: int = 0,
) -> ArenaBatchLike:
    try:
        import great_kingdom_core as core
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "great_kingdom_core is not installed. Build it with maturin before arena evaluation."
        ) from exc
    if not hasattr(core, "GumbelArenaBatch"):
        raise RuntimeError(
            "great_kingdom_core.GumbelArenaBatch is not available. "
            "Rebuild the Rust extension before batched arena evaluation."
        )

    return cast(
        ArenaBatchLike,
        core.GumbelArenaBatch(
            game_count=game_count,
            seed_start=seed_start,
            game_index_start=game_index_start,
            simulations=config.gumbel_simulations,
            max_considered_actions=config.gumbel_max_considered_actions,
            c_visit=config.gumbel_c_visit,
            c_scale=config.gumbel_c_scale,
            seed=config.gumbel_seed,
            gumbel_scale=config.gumbel_scale,
            policy_target_temperature=config.policy_target_temperature,
            policy_target_c_visit=config.policy_target_c_visit,
            policy_target_c_scale=config.policy_target_c_scale,
            paired_seeds=config.paired_seeds,
        ),
    )


def load_arena_config(path: str | Path) -> ArenaConfig:
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError("arena config must be a JSON object")
    return ArenaConfig(**data)


def _arena_onnx_path(
    checkpoint_or_onnx: str | Path,
    output_path: Path,
    *,
    device: str,
    precision: str,
) -> Path:
    source = Path(checkpoint_or_onnx)
    if source.suffix == ".onnx":
        return source
    export_checkpoint_to_onnx(
        source,
        output_path,
        device=device,
        precision=precision,
    )
    return output_path


def _deterministic_action(
    result: ArenaSearchResultLike,
    priors: Sequence[float],
    legal_actions: Sequence[int],
) -> int:
    selected = result.selected_action()
    legal_set = set(legal_actions)
    if selected in legal_set:
        return int(selected)

    visits = result.visit_counts()
    if len(visits) != len(priors):
        raise ValueError("visit count and prior lengths must match")
    if not legal_actions:
        raise ValueError("state has no legal actions")
    return max(legal_actions, key=lambda action: (visits[action], priors[action], -action))


def _arena_turn_gumbel_scale(config: ArenaConfig, turn: int) -> float:
    if turn < config.opening_gumbel_turns:
        return config.opening_gumbel_scale
    return config.gumbel_scale


def _apply_arena_turn_gumbel_scale(
    searches: Iterable[Any],
    *,
    config: ArenaConfig,
    turn: int,
) -> None:
    if config.opening_gumbel_turns <= 0:
        return
    scale = _arena_turn_gumbel_scale(config, turn)
    for search in searches:
        setter = getattr(search, "set_gumbel_scale", None)
        if setter is None:
            raise RuntimeError(
                "arena opening_gumbel_turns requires a GumbelSearch backend with set_gumbel_scale"
            )
        setter(scale)


def _apply_arena_batch_turn_gumbel_scale(
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
    setter(_arena_turn_gumbel_scale(config, turn))


def _other_player(player: int) -> int:
    return ORANGE if player == BLUE else BLUE


def _arena_game_seed(config: ArenaConfig, game_index: int) -> int:
    seed_offset = game_index // 2 if config.paired_seeds else game_index
    return config.seed_start + seed_offset


def _validate_arena_config(config: ArenaConfig) -> None:
    if config.games < 0:
        raise ValueError("games must be non-negative")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.max_turns <= 0:
        raise ValueError("max_turns must be positive")
    if not 0.0 <= config.promotion_threshold <= 1.0:
        raise ValueError("promotion_threshold must be between 0 and 1")
    if config.gumbel_simulations <= 0:
        raise ValueError("gumbel_simulations must be positive")
    if config.gumbel_max_considered_actions <= 0:
        raise ValueError("gumbel_max_considered_actions must be positive")
    if config.gumbel_c_visit <= 0.0:
        raise ValueError("gumbel_c_visit must be positive")
    if config.gumbel_c_scale <= 0.0:
        raise ValueError("gumbel_c_scale must be positive")
    if not math.isfinite(config.gumbel_scale) or config.gumbel_scale < 0.0:
        raise ValueError("gumbel_scale must be finite and non-negative")
    if config.opening_gumbel_turns < 0:
        raise ValueError("opening_gumbel_turns must be non-negative")
    if (
        not math.isfinite(config.opening_gumbel_scale)
        or config.opening_gumbel_scale < 0.0
    ):
        raise ValueError("opening_gumbel_scale must be finite and non-negative")
    if not math.isfinite(config.policy_target_c_visit) or config.policy_target_c_visit <= 0.0:
        raise ValueError("policy_target_c_visit must be finite and positive")
    if not math.isfinite(config.policy_target_c_scale) or config.policy_target_c_scale <= 0.0:
        raise ValueError("policy_target_c_scale must be finite and positive")
    if (
        not math.isfinite(config.policy_target_temperature)
        or config.policy_target_temperature <= 0.0
    ):
        raise ValueError("policy_target_temperature must be finite and positive")
    if config.leaf_batch_size <= 0:
        raise ValueError("leaf_batch_size must be positive")


def _evaluate_arena_rows_by_model(
    *,
    candidate_model: Any,
    best_model: Any,
    feature_rows: Any,
    legal_masks: Any,
    game_indexes: Sequence[int],
    current_players: Sequence[int],
    candidate_players: Sequence[int],
    device: Any | str | None,
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
            evaluation = evaluate_feature_arrays_logits_values(
                model,
                feature_rows[list(offsets)],
                legal_masks[list(offsets)],
                device=device,
            )
        else:
            evaluation = evaluate_feature_batch_logits_values(
                model,
                [[float(value) for value in feature_rows[offset]] for offset in offsets],
                [[bool(value) for value in legal_masks[offset]] for offset in offsets],
                device=device,
            )
        policy_rows = evaluation.policy_logits
        if len(policy_rows) != len(offsets) or len(evaluation.value) != len(offsets):
            raise ValueError("model evaluation returned a mismatched batch size")
        for offset, policy, value in zip(
            offsets,
            policy_rows,
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


def _request_feature_rows_and_masks(request: Any) -> tuple[Any, Any, int]:
    if (
        hasattr(request, "len")
        and hasattr(request, "feature_plane_bytes")
        and hasattr(request, "legal_mask_bytes")
    ):
        row_count = int(request.len())
        feature_rows = np.frombuffer(request.feature_plane_bytes(), dtype=np.float32).reshape(
            row_count,
            FEATURE_CHANNELS,
            BOARD_SIZE,
            BOARD_SIZE,
        )
        legal_masks = np.frombuffer(request.legal_mask_bytes(), dtype=np.bool_).reshape(
            row_count,
            ACTION_SPACE,
        )
        return feature_rows, legal_masks, row_count

    feature_rows = request.feature_planes()
    legal_masks = request.legal_masks()
    return feature_rows, legal_masks, len(feature_rows)


def _active_legal_masks(batch: ArenaBatchLike) -> list[list[bool]]:
    if hasattr(batch, "active_legal_masks"):
        return [[bool(value) for value in row] for row in batch.active_legal_masks()]
    request = batch.active_eval_request()
    _feature_rows, masks, _row_count = _request_feature_rows_and_masks(request)
    if isinstance(masks, np.ndarray):
        return [[bool(value) for value in row] for row in masks]
    return [[bool(value) for value in row] for row in masks]


def _request_game_indexes(request: Any, *, expected_len: int) -> list[int]:
    if not hasattr(request, "game_indexes"):
        raise RuntimeError("arena eval request did not include game index metadata")
    game_indexes = _as_int_list(request.game_indexes())
    if len(game_indexes) != expected_len:
        raise RuntimeError("arena eval request did not include game index metadata")
    return game_indexes


def _request_current_players(request: Any, *, expected_len: int) -> list[int]:
    if not hasattr(request, "current_players"):
        raise RuntimeError("arena eval request did not include current player metadata")
    players = _as_int_list(request.current_players())
    if len(players) != expected_len:
        raise RuntimeError("arena eval request current player metadata length mismatch")
    return players


def _finished_arena_batch_results(
    *,
    batch: ArenaBatchLike,
    config: ArenaConfig,
    chunk_start: int,
    moves: Sequence[list[MoveLog]],
) -> list[ArenaGameResult | None]:
    candidate_players = _as_int_list(batch.candidate_players())
    seeds = (
        _as_int_list(batch.seeds())
        if hasattr(batch, "seeds")
        else [
            _arena_game_seed(config, chunk_start + game_index)
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
                best_player=_other_player(candidate_player),
                winner=int(winner),
                end_reason=int(end_reason),
                moves=list(moves[game_index]),
                territory_scores=scores[game_index],
            )
        )
    return results


def _as_int_list(values: Sequence[int] | bytes) -> list[int]:
    return [int(value) for value in values]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a candidate model against the best model"
    )
    parser.add_argument("--candidate", type=Path, required=True, help="Candidate checkpoint path")
    parser.add_argument("--best", type=Path, required=True, help="Best checkpoint path")
    parser.add_argument("--report", type=Path, required=True, help="Output arena report JSON")
    parser.add_argument("--config", type=Path, default=None, help="JSON ArenaConfig override")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--games", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--gumbel-simulations", type=int, default=None)
    parser.add_argument(
        "--gumbel-max-considered-actions",
        "--gumbel-max-consider",
        dest="gumbel_max_considered_actions",
        type=int,
        default=None,
    )
    parser.add_argument("--gumbel-c-visit", type=float, default=None)
    parser.add_argument("--gumbel-c-scale", type=float, default=None)
    parser.add_argument("--gumbel-scale", type=float, default=None)
    parser.add_argument("--opening-gumbel-turns", type=int, default=None)
    parser.add_argument("--opening-gumbel-scale", type=float, default=None)
    parser.add_argument("--policy-target-c-visit", type=float, default=None)
    parser.add_argument("--policy-target-c-scale", type=float, default=None)
    parser.add_argument("--policy-target-temperature", type=float, default=None)
    parser.add_argument("--gumbel-seed", type=int, default=None)
    parser.add_argument(
        "--paired-seeds",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use each arena seed for a candidate-Blue/candidate-Orange pair.",
    )
    parser.add_argument("--leaf-batch-size", type=int, default=None)
    parser.add_argument("--promotion-threshold", type=float, default=None)
    parser.add_argument(
        "--backend",
        choices=["onnx", "pytorch"],
        default="onnx",
        help="arena inference backend; ONNX uses the Rust evaluator path",
    )
    parser.add_argument("--onnx-max-batch-size", type=int, default=8192)
    parser.add_argument("--onnx-precision", choices=["fp32", "fp16"], default=None)
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Copy candidate over best if accepted",
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> ArenaConfig:
    config = load_arena_config(args.config) if args.config is not None else ArenaConfig()
    overrides = {
        "device": args.device,
        "games": args.games,
        "batch_size": args.batch_size,
        "seed_start": args.seed_start,
        "max_turns": args.max_turns,
        "gumbel_simulations": args.gumbel_simulations,
        "gumbel_max_considered_actions": args.gumbel_max_considered_actions,
        "gumbel_c_visit": args.gumbel_c_visit,
        "gumbel_c_scale": args.gumbel_c_scale,
        "gumbel_scale": args.gumbel_scale,
        "opening_gumbel_turns": args.opening_gumbel_turns,
        "opening_gumbel_scale": args.opening_gumbel_scale,
        "policy_target_c_visit": args.policy_target_c_visit,
        "policy_target_c_scale": args.policy_target_c_scale,
        "policy_target_temperature": args.policy_target_temperature,
        "gumbel_seed": args.gumbel_seed,
        "paired_seeds": args.paired_seeds,
        "leaf_batch_size": args.leaf_batch_size,
        "promotion_threshold": args.promotion_threshold,
    }
    data = asdict(config)
    data.update({key: value for key, value in overrides.items() if value is not None})
    return ArenaConfig(**data)


def main() -> NoReturn:
    args = build_parser().parse_args()
    config = _config_from_args(args)
    if args.backend == "onnx":
        precision = args.onnx_precision or ("fp16" if config.device == "cuda" else "fp32")
        report = run_arena_checkpoints_onnx(
            candidate_checkpoint=args.candidate,
            best_checkpoint=args.best,
            config=config,
            onnx_max_batch_size=args.onnx_max_batch_size,
            onnx_precision=precision,
        )
    else:
        candidate_model = load_model_from_checkpoint(args.candidate, device=config.device)
        best_model = load_model_from_checkpoint(args.best, device=config.device)
        report = run_arena(
            candidate_model=candidate_model,
            best_model=best_model,
            config=config,
        )
    save_arena_report(report, args.report)
    promoted = (
        promote_candidate_if_needed(
            candidate_checkpoint=args.candidate,
            best_checkpoint=args.best,
            report=report,
        )
        if args.promote
        else False
    )
    print(
        json.dumps(
            {
                "report": str(args.report),
                "promoted": promoted,
                "summary": report.summary.to_dict(),
            },
            sort_keys=True,
        )
    )
    raise SystemExit(0)


if __name__ == "__main__":
    main()


__all__ = [
    "ArenaBatchLike",
    "ArenaConfig",
    "ArenaGameResult",
    "ArenaOnnxEvaluators",
    "ArenaReport",
    "ArenaSummary",
    "create_core_arena_batch",
    "create_onnx_evaluator",
    "create_core_search_engine",
    "evaluate_state_policy",
    "evaluate_state_policy_logits",
    "evaluate_state_policy_logits_batch",
    "load_arena_config",
    "load_model_from_checkpoint",
    "play_arena_game",
    "promote_candidate_if_needed",
    "run_arena",
    "run_arena_batched",
    "run_arena_checkpoints_onnx",
    "run_arena_onnx",
    "save_arena_report",
    "summarize_arena",
]
