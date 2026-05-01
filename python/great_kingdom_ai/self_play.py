"""Random self-play orchestration backed by the Rust rules engine."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Protocol


class SelfPlayState(Protocol):
    def legal_actions(self) -> list[int]: ...


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
        place_actions = [action for action in legal_actions if action != 81]
        if place_actions:
            candidates = place_actions

    return rng.choice(candidates)
