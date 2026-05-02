"""Arena evaluation and best-model promotion helpers."""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NoReturn, Protocol, cast

from great_kingdom_ai.evaluator import evaluate_feature_batch
from great_kingdom_ai.self_play import MoveLog, SelfPlayState, create_core_game_state

BLUE = 1
ORANGE = 2


class ArenaSearchResultLike(Protocol):
    def selected_action(self) -> int | None: ...

    def visit_counts(self) -> list[int]: ...


class ArenaSearchLike(Protocol):
    def search_with_priors(
        self,
        state: SelfPlayState,
        priors: list[float],
    ) -> ArenaSearchResultLike: ...

    def search_with_priors_and_evaluator(
        self,
        state: SelfPlayState,
        priors: list[float],
        evaluator: Callable[[Any], tuple[list[list[float]], list[float]]],
        leaf_batch_size: int = 8,
    ) -> ArenaSearchResultLike: ...

    def search_with_logits_and_evaluator(
        self,
        state: SelfPlayState,
        policy_logits: list[float],
        evaluator: Callable[[Any], tuple[list[list[float]], list[float]]],
        leaf_batch_size: int = 8,
    ) -> ArenaSearchResultLike: ...


@dataclass(frozen=True)
class ArenaConfig:
    search_backend: str = "mcts"
    games: int = 20
    seed_start: int = 0
    max_turns: int = 200
    simulations: int = 100
    c_puct: float = 1.5
    gumbel_simulations: int = 128
    gumbel_max_considered_actions: int = 16
    gumbel_c_visit: float = 50.0
    gumbel_c_scale: float = 1.0
    gumbel_seed: int = 0
    leaf_batch_size: int = 8
    device: str = "cpu"
    promotion_threshold: float = 0.55


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

    Evaluation uses model priors and disables self-play-only root noise and temperature.
    """
    if candidate_player not in {BLUE, ORANGE}:
        raise ValueError("candidate_player must be 1 or 2")

    game_state = state if state is not None else create_core_game_state()
    if search_factory is None:
        searches = {
            BLUE: create_core_search_backend(config, seed_offset=seed * 2),
            ORANGE: create_core_search_backend(config, seed_offset=seed * 2 + 1),
        }
    else:
        searches = {BLUE: search_factory(), ORANGE: search_factory()}
    best_player = _other_player(candidate_player)
    moves: list[MoveLog] = []

    for turn in range(config.max_turns):
        if game_state.is_terminal():
            break

        player = game_state.current_player()
        model = candidate_model if player == candidate_player else best_model
        root_evaluation = evaluate_feature_batch(
            model,
            [game_state.feature_planes()],
            [game_state.legal_mask()],
            device=config.device,
        )
        priors = [float(value) for value in root_evaluation.policy[0]]
        root_logits = getattr(root_evaluation, "policy_logits", root_evaluation.policy)[0]
        logits = [float(value) for value in root_logits]

        def evaluator(request: Any, m: Any = model) -> tuple[list[list[float]], list[float]]:
            feature_rows = request.feature_planes()
            mask_rows = request.legal_masks()
            evaluation = evaluate_feature_batch(
                m, feature_rows, mask_rows, device=config.device
            )
            policy_rows = (
                getattr(evaluation, "policy_logits", evaluation.policy)
                if config.search_backend == "gumbel"
                else evaluation.policy
            )
            return (
                [[float(value) for value in policy] for policy in policy_rows],
                [float(value) for value in evaluation.value],
            )

        if config.search_backend == "gumbel":
            result = searches[player].search_with_logits_and_evaluator(
                game_state,
                logits,
                evaluator,
                config.leaf_batch_size,
            )
        else:
            result = searches[player].search_with_priors_and_evaluator(
                game_state,
                priors,
                evaluator,
                config.leaf_batch_size,
            )
        action = _deterministic_action(result, priors, game_state.legal_actions())

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
    if config.games < 0:
        raise ValueError("games must be non-negative")
    if config.max_turns <= 0:
        raise ValueError("max_turns must be positive")
    if not 0.0 <= config.promotion_threshold <= 1.0:
        raise ValueError("promotion_threshold must be between 0 and 1")
    if config.search_backend not in {"mcts", "gumbel"}:
        raise ValueError("search_backend must be 'mcts' or 'gumbel'")

    make_state = state_factory if state_factory is not None else create_core_game_state
    games = []
    for index in range(config.games):
        game = play_arena_game(
            seed=config.seed_start + index,
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
    evaluation = evaluate_feature_batch(
        model,
        [state.feature_planes() for state in states],
        [state.legal_mask() for state in states],
        device=device,
    )
    rows = getattr(evaluation, "policy_logits", evaluation.policy)
    return [[float(value) for value in row] for row in rows]


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


def load_model_from_checkpoint(path: str | Path, *, device: str = "cpu") -> Any:
    from great_kingdom_ai.train import load_checkpoint

    state = load_checkpoint(path, device=device)
    state.model.eval()
    return state.model


def create_core_mcts_search(*, simulations: int, c_puct: float) -> ArenaSearchLike:
    try:
        import great_kingdom_core as core
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "great_kingdom_core is not installed. Build it with maturin before arena evaluation."
        ) from exc

    return cast(ArenaSearchLike, core.MctsSearch(simulations=simulations, c_puct=c_puct))


def create_core_search_backend(
    config: ArenaConfig,
    *,
    seed_offset: int = 0,
) -> ArenaSearchLike:
    try:
        import great_kingdom_core as core
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "great_kingdom_core is not installed. Build it with maturin before arena evaluation."
        ) from exc

    if config.search_backend == "mcts":
        return cast(
            ArenaSearchLike,
            core.MctsSearch(simulations=config.simulations, c_puct=config.c_puct),
        )
    if config.search_backend == "gumbel":
        return cast(
            ArenaSearchLike,
            core.GumbelSearch(
                simulations=config.gumbel_simulations,
                max_considered_actions=config.gumbel_max_considered_actions,
                c_visit=config.gumbel_c_visit,
                c_scale=config.gumbel_c_scale,
                seed=config.gumbel_seed + seed_offset,
            ),
        )
    raise ValueError("search_backend must be 'mcts' or 'gumbel'")


def load_arena_config(path: str | Path) -> ArenaConfig:
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError("arena config must be a JSON object")
    return ArenaConfig(**data)


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


def _other_player(player: int) -> int:
    return ORANGE if player == BLUE else BLUE


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
    parser.add_argument("--simulations", type=int, default=None)
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
        "simulations": args.simulations,
    }
    data = asdict(config)
    data.update({key: value for key, value in overrides.items() if value is not None})
    return ArenaConfig(**data)


def main() -> NoReturn:
    args = build_parser().parse_args()
    config = _config_from_args(args)
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
    "ArenaConfig",
    "ArenaGameResult",
    "ArenaReport",
    "ArenaSummary",
    "create_core_search_backend",
    "evaluate_state_policy",
    "evaluate_state_policy_logits",
    "evaluate_state_policy_logits_batch",
    "load_arena_config",
    "load_model_from_checkpoint",
    "play_arena_game",
    "promote_candidate_if_needed",
    "run_arena",
    "save_arena_report",
    "summarize_arena",
]
