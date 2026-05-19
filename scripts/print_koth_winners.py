"""Print unique KOTH summary winners in match order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, NoReturn


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print unique winners from a find_strongest_candidate.py summary.json file."
    )
    parser.add_argument("summary", type=Path, help="KOTH summary.json path")
    parser.add_argument(
        "--include-initial-champion",
        action="store_true",
        help="also print the champion_before from the first match before match winners",
    )
    parser.add_argument(
        "--basename",
        action="store_true",
        help="print only checkpoint filenames instead of full paths",
    )
    parser.add_argument(
        "--allow-duplicates",
        action="store_true",
        help="print every match winner, including repeated champions",
    )
    return parser


def main() -> NoReturn:
    args = build_parser().parse_args()
    summary = load_summary(args.summary)
    winners = extract_winners(
        summary,
        include_initial_champion=args.include_initial_champion,
        dedupe=not args.allow_duplicates,
    )
    for winner in winners:
        print(Path(winner).name if args.basename else winner)
    raise SystemExit(0)


def load_summary(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def extract_winners(
    summary: dict[str, Any],
    *,
    include_initial_champion: bool,
    dedupe: bool = True,
) -> list[str]:
    matches = summary.get("matches")
    if not isinstance(matches, list):
        raise ValueError("KOTH summary must contain a matches list")

    sorted_matches = sorted(
        (_validated_match(match, index) for index, match in enumerate(matches)),
        key=lambda row: (row["match_index"], row["source_index"]),
    )
    winners: list[str] = []
    if include_initial_champion and sorted_matches:
        winners.append(str(sorted_matches[0]["champion_before"]))
    winners.extend(str(match["winner"]) for match in sorted_matches)
    return _dedupe_preserving_order(winners) if dedupe else winners


def _dedupe_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _validated_match(match: Any, source_index: int) -> dict[str, Any]:
    if not isinstance(match, dict):
        raise ValueError(f"matches[{source_index}] must be a JSON object")
    if "winner" not in match:
        raise ValueError(f"matches[{source_index}] is missing winner")
    return {
        **match,
        "match_index": _match_index(match, source_index),
        "source_index": source_index,
    }


def _match_index(match: dict[str, Any], source_index: int) -> int:
    value = match.get("match_index", source_index + 1)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"matches[{source_index}] has invalid match_index: {value!r}") from exc


if __name__ == "__main__":
    main()
