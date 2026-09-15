"""Terminal board target encoding for the auxiliary board-prediction loss.

The Rust engine records the final board with absolute cell codes
(``Cell``: ``0=Empty, 1=Blue, 2=Orange, 3=Neutral``). Replay stores that
absolute array once per episode. Training converts it to the sampled
transition player's perspective:

* ``0`` empty
* ``1`` own castle
* ``2`` opponent castle
* ``3`` neutral castle

Neutral castles and empty cells keep their absolute code, so the perspective
conversion only swaps Blue/Orange for the Orange player.
"""

from __future__ import annotations

import numpy as np

from great_kingdom_ai.features import BOARD_SIZE

TERMINAL_BOARD_ENCODING = "absolute_cells_preserved_v1"
TERMINAL_BOARD_CLASSES = 4

EMPTY_CELL = 0
BLUE_CELL = 1
ORANGE_CELL = 2
NEUTRAL_CELL = 3

OWN_CLASS = 1
OPPONENT_CLASS = 2

TERMINAL_BOARD_SHAPE = (BOARD_SIZE, BOARD_SIZE)


def terminal_board_from_flat(flat: np.ndarray) -> np.ndarray:
    """Reshape a flat engine board (list, bytes, or array) into a 9x9 ``uint8``."""
    if isinstance(flat, bytes | bytearray | memoryview):
        array = np.frombuffer(flat, dtype=np.uint8).reshape(BOARD_SIZE, BOARD_SIZE).copy()
    else:
        array = np.asarray(flat, dtype=np.uint8).reshape(BOARD_SIZE, BOARD_SIZE)
    return validate_absolute_terminal_board(array)


def terminal_board_to_flat_values(board: np.ndarray) -> tuple[int, ...]:
    """Flatten an absolute board into a JSON-friendly tuple of ints."""
    array = validate_absolute_terminal_board(board)
    return tuple(int(value) for value in array.reshape(-1))


def validate_absolute_terminal_board(board: np.ndarray) -> np.ndarray:
    """Validate an absolute engine cell board and return it as ``uint8``."""
    array = np.asarray(board)
    if array.shape != TERMINAL_BOARD_SHAPE:
        raise ValueError(
            f"expected terminal board shape {TERMINAL_BOARD_SHAPE}, got {array.shape}"
        )
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"terminal board must be an integer array, got {array.dtype}")
    if np.any(array < EMPTY_CELL) or np.any(array > NEUTRAL_CELL):
        raise ValueError("terminal board cells must be in [0, 3]")
    return array.astype(np.uint8, copy=False)


def absolute_terminal_boards_to_perspective(
    boards: np.ndarray,
    players: np.ndarray,
) -> np.ndarray:
    """Convert absolute boards into per-player target classes.

    ``boards`` has shape ``[..., BOARD_SIZE, BOARD_SIZE]`` and ``players`` has
    shape ``[...]`` with values in ``{1, 2}``. Class numbers never change under
    board symmetries, so D4 augmentation can transform the result directly.
    """
    board_array = np.asarray(boards, dtype=np.int64)
    player_array = np.asarray(players, dtype=np.int64)
    if board_array.shape[-2:] != TERMINAL_BOARD_SHAPE:
        raise ValueError(
            f"expected board trailing shape {TERMINAL_BOARD_SHAPE}, got {board_array.shape}"
        )
    if player_array.shape != board_array.shape[:-2]:
        raise ValueError("players shape must match the batch of terminal boards")
    if np.any((player_array != 1) & (player_array != 2)):
        raise ValueError("players must be 1 or 2")
    if np.any(board_array < EMPTY_CELL) or np.any(board_array > NEUTRAL_CELL):
        raise ValueError("terminal board cells must be in [0, 3]")

    targets = board_array.copy()
    swap = (player_array == 2)[..., None, None]
    targets = np.where(
        swap & (targets == BLUE_CELL),
        OPPONENT_CLASS,
        np.where(swap & (targets == ORANGE_CELL), OWN_CLASS, targets),
    )
    return targets.astype(np.int64, copy=False)


__all__ = [
    "BLUE_CELL",
    "EMPTY_CELL",
    "NEUTRAL_CELL",
    "OPPONENT_CLASS",
    "ORANGE_CELL",
    "OWN_CLASS",
    "TERMINAL_BOARD_CLASSES",
    "TERMINAL_BOARD_ENCODING",
    "TERMINAL_BOARD_SHAPE",
    "absolute_terminal_boards_to_perspective",
    "terminal_board_from_flat",
    "terminal_board_to_flat_values",
    "validate_absolute_terminal_board",
]
