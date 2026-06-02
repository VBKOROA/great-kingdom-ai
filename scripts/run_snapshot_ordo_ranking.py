#!/usr/bin/env python3
"""Rank recent training snapshots with Ordo using sparse time-neighbor pairings."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, NoReturn

from great_kingdom_ai.evaluate import (
    ArenaConfig,
    run_arena_checkpoints_onnx,
)
from run_candidate_pairwise_matrix import (
    DEFAULT_ARENA_CONFIG,
    PairwiseMatchResult,
    _load_arena_summary,
    _load_effective_arena_config,
    _match_from_summary,
    _release_cuda_cache,
)

BLUE = 1
ORANGE = 2


@dataclass(frozen=True)
class SnapshotPair:
    candidate: Path
    baseline: Path
    kind: str  # "neighbor" or "anchor"
    games: int


@dataclass(frozen=True)
class OrdoRating:
    name: str
    elo: float
    error: float | None
    games: int | None


def select_snapshots(
    checkpoint_dir: Path,
    *,
    glob: str = "*.pt",
    recursive: bool = False,
    max_snapshots: int = 25,
    sort_by_mtime: bool = False,
) -> list[Path]:
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {checkpoint_dir}")
    
    paths = checkpoint_dir.rglob(glob) if recursive else checkpoint_dir.glob(glob)
    files = [path for path in paths if path.is_file()]
    
    if sort_by_mtime:
        files.sort(key=lambda p: (p.stat().st_mtime, p.name))
    else:
        files.sort(key=lambda p: (str(p.parent), p.name))
        
    if max_snapshots > 0:
        files = files[-max_snapshots:]
    return files


def generate_sparse_pairs(
    snapshots: list[Path],
    anchors: list[Path],
    *,
    pair_offsets: list[int],
    games: int,
    anchor_games: int,
) -> list[SnapshotPair]:
    pairs: list[SnapshotPair] = []
    
    # Resolved paths to avoid duplicate pair elements
    resolved_anchors = [a.resolve() for a in anchors]
    filtered_snapshots = [s for s in snapshots if s.resolve() not in resolved_anchors]
    
    # 1. Neighbor pairs
    for offset in pair_offsets:
        for i in range(len(filtered_snapshots)):
            if i - offset >= 0:
                pairs.append(SnapshotPair(
                    candidate=filtered_snapshots[i],
                    baseline=filtered_snapshots[i - offset],
                    kind="neighbor",
                    games=games,
                ))
                
    # 2. Anchor pairs
    for snapshot in filtered_snapshots:
        for anchor in anchors:
            pairs.append(SnapshotPair(
                candidate=snapshot,
                baseline=anchor,
                kind="anchor",
                games=anchor_games,
            ))
            
    # Deduplicate based on unique set of paths
    unique_pairs: list[SnapshotPair] = []
    seen = set()
    for pair in pairs:
        c_res = pair.candidate.resolve()
        b_res = pair.baseline.resolve()
        key = frozenset([c_res, b_res])
        if key not in seen:
            seen.add(key)
            unique_pairs.append(pair)
            
    return unique_pairs


def blue_orange_game_to_ordo_pgn(
    game: dict[str, Any],
    candidate_id: str,
    baseline_id: str,
) -> str:
    candidate_player = game.get("candidate_player")
    winner = game.get("winner")
    
    if winner not in (BLUE, ORANGE):
        raise ValueError(
            f"Game has unresolved winner: {winner}. "
            "Great Kingdom must always have a winner."
        )
    
    if candidate_player == BLUE:
        white_player = candidate_id
        black_player = baseline_id
        if winner == BLUE:
            result = "1-0"
        elif winner == ORANGE:
            result = "0-1"
        else:
            result = "1/2-1/2"
    elif candidate_player == ORANGE:
        white_player = baseline_id
        black_player = candidate_id
        if winner == BLUE:
            result = "1-0"
        elif winner == ORANGE:
            result = "0-1"
        else:
            result = "1/2-1/2"
    else:
        raise ValueError(f"Invalid candidate_player: {candidate_player}")
        
    pgn_lines = [
        '[Event "great-kingdom-ai snapshot arena"]',
        f'[White "{white_player}"]',
        f'[Black "{black_player}"]',
        f'[Result "{result}"]',
        "",
        result,
        ""
    ]
    return "\n".join(pgn_lines)


def parse_ordo_output(ordo_text: str) -> list[OrdoRating]:
    ratings: list[OrdoRating] = []
    lines = ordo_text.strip().split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if (
            line.startswith("#")
            or line.startswith("list")
            or line.startswith("Ordo")
            or line.startswith("Ranking")
        ):
            continue
            
        parts = line.split()
        if len(parts) < 3:
            continue
            
        if not parts[0].isdigit():
            continue
            
        name = parts[1]
        try:
            elo = float(parts[2])
        except ValueError:
            continue
            
        error = None
        games = None
        remaining_parts = parts[3:]
        
        floats = []
        for p in remaining_parts:
            clean_p = p.replace("+", "").replace("-", "").replace("/", "").strip()
            if not clean_p or clean_p.endswith("%"):
                continue
            try:
                val = float(clean_p)
                if "." in clean_p:
                    floats.append(val)
            except ValueError:
                pass
                
        if floats:
            error = floats[0]
        elif len(remaining_parts) >= 1:
            try:
                error = float(
                    remaining_parts[0]
                    .replace("+", "")
                    .replace("-", "")
                    .replace("/", "")
                    .strip()
                )
            except ValueError:
                pass
                
        for p in reversed(remaining_parts):
            clean_p = p.strip()
            if clean_p.isdigit():
                games = int(clean_p)
                break
                
        ratings.append(OrdoRating(
            name=name,
            elo=elo,
            error=error,
            games=games,
        ))
    return ratings


def run_or_load_snapshot_pair(
    pair: SnapshotPair,
    *,
    match_index: int,
    report_path: Path,
    arena_config: ArenaConfig,
    backend: Literal["onnx", "pytorch"],
    force: bool,
    onnx_precision: str = "fp16",
    onnx_max_batch_size: int = 8192,
) -> PairwiseMatchResult:
    use_cache = False
    if report_path.exists() and not force:
        try:
            with report_path.open("r", encoding="utf-8") as f:
                cached_data = json.load(f)
            cached_summary = cached_data.get("summary")
            cached_config = cached_data.get("config", {})
            cached_backend = cached_data.get("backend")
            
            is_games_match = cached_config.get("games") == pair.games
            is_backend_match = cached_backend == backend
            is_sims_match = (
                cached_config.get("gumbel_simulations")
                == arena_config.gumbel_simulations
            )
            is_max_actions_match = (
                cached_config.get("gumbel_max_considered_actions")
                == arena_config.gumbel_max_considered_actions
            )
            is_paired_seeds_match = (
                cached_config.get("paired_seeds") == arena_config.paired_seeds
            )
            
            if (
                cached_summary
                and is_games_match
                and is_backend_match
                and is_sims_match
                and is_max_actions_match
                and is_paired_seeds_match
            ):
                use_cache = True
        except Exception:
            use_cache = False
            
    if use_cache:
        summary = _load_arena_summary(report_path)
    else:
        # Override games to match the pair setting
        pair_config = ArenaConfig(**{**asdict(arena_config), "games": pair.games})
        
        if backend == "pytorch":
            from great_kingdom_ai.evaluate import (
                load_model_from_checkpoint,
                run_arena,
            )
            candidate_model = load_model_from_checkpoint(
                pair.candidate,
                device=pair_config.device,
            )
            baseline_model = load_model_from_checkpoint(
                pair.baseline,
                device=pair_config.device,
            )
            report = run_arena(
                candidate_model=candidate_model,
                best_model=baseline_model,
                config=pair_config,
            )
            summary = report.summary.to_dict()
            report_dict = report.to_dict()
            report_dict["backend"] = backend
            
            report_path.write_text(
                json.dumps(report_dict, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            del candidate_model, baseline_model
            _release_cuda_cache()
        elif backend == "onnx":
            report = run_arena_checkpoints_onnx(
                candidate_checkpoint=pair.candidate,
                best_checkpoint=pair.baseline,
                config=pair_config,
                onnx_max_batch_size=onnx_max_batch_size,
                onnx_precision=onnx_precision,
            )
            summary = report.summary.to_dict()
            report_dict = report.to_dict()
            report_dict["backend"] = backend
            
            report_path.write_text(
                json.dumps(report_dict, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            _release_cuda_cache()
        else:
            raise ValueError(f"Unknown backend: {backend}")
            
    return _match_from_summary(
        match_index=match_index,
        candidate=pair.candidate,
        baseline=pair.baseline,
        report_path=report_path,
        summary=summary,
    )


def run_ordo(
    *,
    ordo_bin: str,
    pgn_path: Path,
    output_path: Path,
    anchor_name: str | None = None,
    anchor_elo: float = 0.0,
) -> None:
    # Build ordo command
    cmd = [ordo_bin, "-p", str(pgn_path), "-o", str(output_path)]
    
    # Try running the command
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Ordo binary not found at '{ordo_bin}'. Please install it or verify your PATH.\n"
            f"Error: {e}"
        ) from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Ordo failed with exit code {e.returncode}.\n"
            f"stdout: {e.stdout}\n"
            f"stderr: {e.stderr}"
        ) from e


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rank recent training snapshots with Ordo using sparse neighbor pairings."
    )
    parser.add_argument(
        "checkpoint_dir",
        type=Path,
        help="directory containing snapshot .pt files",
    )
    parser.add_argument(
        "--glob",
        default="*.pt",
        help="snapshot filename pattern",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="recursive scan for snapshots",
    )
    parser.add_argument(
        "--max-snapshots",
        type=int,
        default=25,
        help="keep latest N snapshots; 0 for all",
    )
    parser.add_argument(
        "--anchors",
        type=Path,
        nargs="*",
        default=None,
        help="fixed anchor checkpoints",
    )
    parser.add_argument(
        "--pair-offsets",
        default="1,2,4,8",
        help="comma-separated list of time-neighbor offsets to pair",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=8,
        help="games per normal snapshot pair",
    )
    parser.add_argument(
        "--anchor-games",
        type=int,
        default=16,
        help="games per snapshot-anchor pair",
    )
    parser.add_argument(
        "--arena-config",
        type=Path,
        default=DEFAULT_ARENA_CONFIG,
    )
    parser.add_argument(
        "--backend",
        choices=["onnx", "pytorch"],
        default="onnx",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
    )
    parser.add_argument(
        "--ordo-bin",
        default="ordo",
        help="path to ordo binary",
    )
    parser.add_argument(
        "--ordo-anchor",
        default=None,
        help="post-normalize rating relative to this anchor",
    )
    parser.add_argument(
        "--ordo-anchor-elo",
        type=float,
        default=0.0,
        help="Elo target for anchor normalization",
    )
    parser.add_argument(
        "--force-arena",
        action="store_true",
        help="rerun arena matches even if reports exist",
    )
    parser.add_argument(
        "--force-ordo",
        action="store_true",
        help="rerun Ordo calculation even if no new games",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="report output folder; defaults to CHECKPOINT_DIR/ordo-ranking-reports",
    )
    parser.add_argument(
        "--sort-by-mtime",
        action="store_true",
        help="sort snapshots by mtime instead of filename",
    )
    parser.add_argument(
        "--onnx-precision",
        choices=["fp16", "fp32"],
        default=None,
        help="ONNX precision (fp16 for CUDA, fp32 for CPU by default)",
    )
    parser.add_argument(
        "--onnx-max-batch-size",
        type=int,
        default=8192,
        help="ONNX max batch size",
    )
    
    # Gumbel settings overrides
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument("--gumbel-simulations", type=int, default=None)
    parser.add_argument("--gumbel-max-considered-actions", type=int, default=None)
    
    return parser


def main() -> NoReturn:
    args = build_parser().parse_args()
    
    # 1. Selection
    snapshots = select_snapshots(
        args.checkpoint_dir,
        glob=args.glob,
        recursive=args.recursive,
        max_snapshots=args.max_snapshots,
        sort_by_mtime=args.sort_by_mtime,
    )
    
    anchors = args.anchors or []
    # Deduplicate resolved paths of anchors
    resolved_anchors = [a.resolve() for a in anchors]
    # Deduplicate snapshots that are actually anchors
    filtered_snapshots = [s for s in snapshots if s.resolve() not in resolved_anchors]
    
    if len(filtered_snapshots) + len(anchors) < 2:
        raise SystemExit("At least 2 total checkpoints (snapshots + anchors) are required.")
        
    output_dir = args.output_dir or args.checkpoint_dir / "ordo-ranking-reports"
    pairs_dir = output_dir / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse pair offsets
    try:
        offsets = [int(x.strip()) for x in args.pair_offsets.split(",") if x.strip()]
        if any(x <= 0 for x in offsets):
            raise ValueError("All offsets must be positive integers.")
    except ValueError as e:
        raise SystemExit(
            f"Invalid --pair-offsets format: {args.pair_offsets}. {e}"
        ) from e
        
    # 2. Pairing
    pairs = generate_sparse_pairs(
        filtered_snapshots,
        anchors,
        pair_offsets=offsets,
        games=args.games,
        anchor_games=args.anchor_games,
    )
    
    if not pairs:
        raise SystemExit("No pairs generated. Check your snapshot count and offsets.")
        
    # Load arena config
    arena_config = _load_effective_arena_config(
        args.arena_config,
        device=args.device,
        games=args.games,
        seed_start=args.seed_start,
        gumbel_simulations=args.gumbel_simulations,
        gumbel_max_considered_actions=args.gumbel_max_considered_actions,
    )
    
    effective_device = arena_config.device
    onnx_precision = args.onnx_precision
    if onnx_precision is None:
        onnx_precision = "fp16" if effective_device == "cuda" else "fp32"
        
    print(
        json.dumps(
            {
                "event": "snapshot_ordo_ranking_start",
                "checkpoint_dir": str(args.checkpoint_dir),
                "snapshot_count": len(filtered_snapshots),
                "anchor_count": len(anchors),
                "pair_count": len(pairs),
                "backend": args.backend,
                "output_dir": str(output_dir),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    
    # 3. Arena Matches
    matches: list[PairwiseMatchResult] = []
    for match_index, pair in enumerate(pairs, start=1):
        report_path = (
            pairs_dir
            / f"pair-{match_index:04d}-{pair.candidate.stem}-vs-{pair.baseline.stem}-arena.json"
        )
        
        print(
            json.dumps(
                {
                    "event": "snapshot_ordo_match_start",
                    "match_index": match_index,
                    "candidate": pair.candidate.name,
                    "baseline": pair.baseline.name,
                }
            ),
            flush=True,
        )
        
        result = run_or_load_snapshot_pair(
            pair,
            match_index=match_index,
            report_path=report_path,
            arena_config=arena_config,
            backend=args.backend,
            force=args.force_arena,
            onnx_precision=onnx_precision,
            onnx_max_batch_size=args.onnx_max_batch_size,
        )
        matches.append(result)
        
        print(
            json.dumps(
                {
                    "event": "snapshot_ordo_match_complete",
                    "match_index": match_index,
                    "candidate": pair.candidate.name,
                    "baseline": pair.baseline.name,
                    "games": result.games,
                }
            ),
            flush=True,
        )
        
    # 4. Generate PGN
    pgn_path = output_dir / "games.pgn"
    pgn_content = []
    for match in matches:
        try:
            report_data = json.loads(match.report_path.read_text(encoding="utf-8"))
            games_list = report_data.get("games", [])
            for g in games_list:
                pgn_game = blue_orange_game_to_ordo_pgn(
                    g,
                    candidate_id=match.candidate.stem,
                    baseline_id=match.baseline.stem,
                )
                pgn_content.append(pgn_game)
        except Exception as e:
            print(
                f"Warning: Failed to process report {match.report_path}: {e}",
                flush=True,
            )
            
    pgn_path.write_text("\n".join(pgn_content), encoding="utf-8")
    
    # 5. Run Ordo
    ordo_output_path = output_dir / "ordo.txt"
    ordo_error_message = None
    try:
        run_ordo(
            ordo_bin=args.ordo_bin,
            pgn_path=pgn_path,
            output_path=ordo_output_path,
        )
    except Exception as e:
        ordo_error_message = str(e)
        print(f"Error executing Ordo: {ordo_error_message}", flush=True)
        print("PGN and Arena reports have been saved successfully.", flush=True)
        
    # 6. Parse Ordo Output and post-normalize
    parsed_ratings: list[OrdoRating] = []
    ratings_dict: dict[str, OrdoRating] = {}
    
    if ordo_error_message is None and ordo_output_path.exists():
        try:
            ordo_text = ordo_output_path.read_text(encoding="utf-8")
            parsed_ratings = parse_ordo_output(ordo_text)
            ratings_dict = {r.name: r for r in parsed_ratings}
        except Exception as e:
            ordo_error_message = f"Failed to parse Ordo output: {e}"
            print(ordo_error_message, flush=True)
            
    # Post-normalization
    normalized_ratings: list[dict[str, Any]] = []
    
    if parsed_ratings:
        # Determine anchor rating for offset normalization
        offset = 0.0
        anchor_found = False
        
        target_anchor_name = args.ordo_anchor
        target_anchor_elo = args.ordo_anchor_elo
        
        if target_anchor_name and target_anchor_name in ratings_dict:
            anchor_rating = ratings_dict[target_anchor_name].elo
            offset = target_anchor_elo - anchor_rating
            anchor_found = True
        elif anchors:
            first_anchor_name = anchors[0].resolve().stem
            if first_anchor_name in ratings_dict:
                anchor_rating = ratings_dict[first_anchor_name].elo
                offset = 0.0 - anchor_rating
                anchor_found = True
                
        if not anchor_found and filtered_snapshots:
            oldest_snapshot_name = filtered_snapshots[0].resolve().stem
            if oldest_snapshot_name in ratings_dict:
                anchor_rating = ratings_dict[oldest_snapshot_name].elo
                offset = 0.0 - anchor_rating
                
        # Build normalized ranking
        for idx, rating in enumerate(parsed_ratings, start=1):
            normalized_elo = rating.elo + offset
            
            model_path = None
            is_snapshot = False
            is_anchor = False
            
            for s in filtered_snapshots:
                if s.resolve().stem == rating.name:
                    model_path = str(s)
                    is_snapshot = True
                    break
                    
            if not is_snapshot:
                for a in anchors:
                    if a.resolve().stem == rating.name:
                        model_path = str(a)
                        is_anchor = True
                        break
                        
            normalized_ratings.append(
                {
                    "rank": idx,
                    "model": rating.name,
                    "path": model_path,
                    "elo": round(normalized_elo, 2),
                    "error": rating.error,
                    "games": rating.games,
                    "is_snapshot": is_snapshot,
                    "is_anchor": is_anchor,
                }
            )
            
    latest_summary = None
    if filtered_snapshots and normalized_ratings:
        latest_snapshot = filtered_snapshots[-1]
        latest_name = latest_snapshot.resolve().stem
        
        latest_rating = next(
            (r for r in normalized_ratings if r["model"] == latest_name),
            None,
        )
        if latest_rating:
            previous_rating = None
            if len(filtered_snapshots) >= 2:
                prev_name = filtered_snapshots[-2].resolve().stem
                previous_rating = next(
                    (r for r in normalized_ratings if r["model"] == prev_name),
                    None,
                )
                
            delta_vs_previous = None
            if previous_rating:
                delta_vs_previous = round(
                    latest_rating["elo"] - previous_rating["elo"], 2
                )
                
            delta_vs_best_anchor = None
            anchor_ratings = [r for r in normalized_ratings if r["is_anchor"]]
            if anchor_ratings:
                best_anchor = max(anchor_ratings, key=lambda r: r["elo"])
                delta_vs_best_anchor = round(
                    latest_rating["elo"] - best_anchor["elo"], 2
                )
                
            latest_summary = {
                "model": latest_rating["model"],
                "rank": latest_rating["rank"],
                "elo": latest_rating["elo"],
                "delta_vs_previous_snapshot": delta_vs_previous,
                "delta_vs_best_anchor": delta_vs_best_anchor,
            }
            
    # Write summary.json
    summary_payload: dict[str, Any] = {
        "event": "snapshot_ordo_ranking_summary",
        "checkpoint_dir": str(args.checkpoint_dir),
        "snapshots": [str(s) for s in filtered_snapshots],
        "anchors": [str(a) for a in anchors],
        "pair_offsets": offsets,
        "pair_count": len(pairs),
        "games_total": sum(match.games for match in matches),
        "backend": args.backend,
        "arena_config": asdict(arena_config),
        "pgn_path": str(pgn_path),
        "ordo_output_path": (
            str(ordo_output_path) if ordo_error_message is None else None
        ),
        "ranking": normalized_ratings,
        "latest": latest_summary,
    }
    
    if ordo_error_message is not None:
        summary_payload["error"] = ordo_error_message
        
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    
    print(json.dumps(summary_payload, sort_keys=True), flush=True)
    raise SystemExit(0 if ordo_error_message is None else 1)


if __name__ == "__main__":
    main()
