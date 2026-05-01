use pyo3::{exceptions::PyValueError, prelude::*};

pub const BOARD_SIZE: usize = 9;
pub const BOARD_CELLS: usize = BOARD_SIZE * BOARD_SIZE;
pub const PASS_ACTION: usize = BOARD_CELLS;
pub const ACTION_SPACE: usize = BOARD_CELLS + 1;
pub const CENTER_INDEX: usize = 40;
pub const CASTLES_PER_PLAYER: u8 = 40;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum Cell {
    Empty = 0,
    Blue = 1,
    Orange = 2,
    Neutral = 3,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum Player {
    Blue = 1,
    Orange = 2,
}

impl Player {
    #[must_use]
    pub const fn cell(self) -> Cell {
        match self {
            Self::Blue => Cell::Blue,
            Self::Orange => Cell::Orange,
        }
    }

    #[must_use]
    pub const fn other(self) -> Self {
        match self {
            Self::Blue => Self::Orange,
            Self::Orange => Self::Blue,
        }
    }

    #[must_use]
    pub const fn used_count(self, state: &GameState) -> u8 {
        match self {
            Self::Blue => state.blue_used,
            Self::Orange => state.orange_used,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Action {
    Place { row: usize, col: usize },
    Pass,
}

impl Action {
    #[must_use]
    pub const fn from_index(index: usize) -> Option<Self> {
        if index < BOARD_CELLS {
            Some(Self::Place {
                row: index / BOARD_SIZE,
                col: index % BOARD_SIZE,
            })
        } else if index == PASS_ACTION {
            Some(Self::Pass)
        } else {
            None
        }
    }

    #[must_use]
    pub const fn to_index(self) -> usize {
        match self {
            Self::Place { row, col } => row * BOARD_SIZE + col,
            Self::Pass => PASS_ACTION,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum GameEndReason {
    OpponentCastleDestroyed = 1,
    OwnCastleDestroyed = 2,
    ConsecutivePasses = 3,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum InvalidAction {
    OutOfRange = 1,
    GameAlreadyEnded = 2,
    OccupiedCell = 3,
    NoCastlesRemaining = 4,
    OpponentTerritory = 5,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GameOutcome {
    pub reason: GameEndReason,
    pub winner: Player,
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct GameState {
    pub(crate) board: [Cell; BOARD_CELLS],
    pub(crate) current_player: Player,
    pub(crate) blue_used: u8,
    pub(crate) orange_used: u8,
    pub(crate) previous_pass: bool,
    pub(crate) terminal: bool,
    pub(crate) outcome: Option<GameOutcome>,
}

#[pymethods]
impl GameState {
    #[new]
    #[must_use]
    pub fn new() -> Self {
        let mut board = [Cell::Empty; BOARD_CELLS];
        board[CENTER_INDEX] = Cell::Neutral;

        Self {
            board,
            current_player: Player::Blue,
            blue_used: 0,
            orange_used: 0,
            previous_pass: false,
            terminal: false,
            outcome: None,
        }
    }

    #[must_use]
    pub fn current_player(&self) -> u8 {
        self.current_player as u8
    }

    #[must_use]
    pub fn blue_used(&self) -> u8 {
        self.blue_used
    }

    #[must_use]
    pub fn orange_used(&self) -> u8 {
        self.orange_used
    }

    #[must_use]
    pub fn previous_pass(&self) -> bool {
        self.previous_pass
    }

    #[must_use]
    pub fn board(&self) -> Vec<u8> {
        self.board.iter().map(|cell| *cell as u8).collect()
    }

    #[must_use]
    pub fn cell_at(&self, index: usize) -> Option<u8> {
        self.board.get(index).map(|cell| *cell as u8)
    }

    #[must_use]
    pub fn legal_actions(&self) -> Vec<usize> {
        self.legal_action_indexes()
    }

    pub fn apply_action(&mut self, action_index: usize) -> PyResult<Option<u8>> {
        self.apply(Action::from_index(action_index).ok_or_else(|| {
            PyValueError::new_err(format!("invalid action index: {action_index}"))
        })?)
        .map(|outcome| outcome.map(|outcome| outcome.winner as u8))
        .map_err(|err| PyValueError::new_err(format!("invalid action: {err:?}")))
    }

    #[must_use]
    pub fn is_terminal(&self) -> bool {
        self.terminal
    }

    #[must_use]
    pub fn winner(&self) -> Option<u8> {
        self.outcome.map(|outcome| outcome.winner as u8)
    }

    #[must_use]
    pub fn end_reason(&self) -> Option<u8> {
        self.outcome.map(|outcome| outcome.reason as u8)
    }
}

impl Default for GameState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
pub(crate) fn state_with_board(
    board: [Cell; BOARD_CELLS],
    current_player: Player,
) -> GameState {
    GameState {
        board,
        current_player,
        blue_used: board.iter().filter(|cell| **cell == Cell::Blue).count() as u8,
        orange_used: board.iter().filter(|cell| **cell == Cell::Orange).count() as u8,
        previous_pass: false,
        terminal: false,
        outcome: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn initial_state_has_neutral_center_and_blue_turn() {
        let state = GameState::new();

        assert_eq!(state.board[CENTER_INDEX], Cell::Neutral);
        assert_eq!(state.current_player, Player::Blue);
        assert!(!state.is_terminal());
        assert_eq!(state.winner(), None);
        assert_eq!(state.end_reason(), None);
    }

    #[test]
    fn player_and_action_types_have_stable_mappings() {
        assert_eq!(Player::Blue.other(), Player::Orange);
        assert_eq!(Player::Orange.other(), Player::Blue);
        assert_eq!(Player::Blue.cell(), Cell::Blue);
        assert_eq!(Player::Orange.cell(), Cell::Orange);

        let place = Action::Place { row: 2, col: 3 };
        assert_eq!(place.to_index(), 21);
        assert_eq!(Action::from_index(21), Some(place));
        assert_eq!(Action::from_index(PASS_ACTION), Some(Action::Pass));
        assert_eq!(Action::Pass.to_index(), PASS_ACTION);
        assert_eq!(Action::from_index(ACTION_SPACE), None);
    }
}
