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
    pub const fn used_count(self, state: &GameState) -> u8 {
        match self {
            Self::Blue => state.blue_used,
            Self::Orange => state.orange_used,
        }
    }
}

impl Player {
    #[must_use]
    pub const fn other(self) -> Self {
        match self {
            Self::Blue => Self::Orange,
            Self::Orange => Self::Blue,
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
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GameOutcome {
    pub reason: GameEndReason,
    pub winner: Player,
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct GameState {
    board: [Cell; BOARD_CELLS],
    current_player: Player,
    blue_used: u8,
    orange_used: u8,
    previous_pass: bool,
    terminal: bool,
    outcome: Option<GameOutcome>,
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
        if self.terminal {
            return Vec::new();
        }

        let mut actions: Vec<usize> = if self.current_player.used_count(self) < CASTLES_PER_PLAYER {
            self.board
                .iter()
                .enumerate()
                .filter_map(|(idx, cell)| (*cell == Cell::Empty).then_some(idx))
                .collect()
        } else {
            Vec::new()
        };
        actions.push(PASS_ACTION);
        actions
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

impl GameState {
    pub fn apply(&mut self, action: Action) -> Result<Option<GameOutcome>, InvalidAction> {
        if self.terminal {
            return Err(InvalidAction::GameAlreadyEnded);
        }

        match action {
            Action::Place { row, col } => self.apply_place(row, col),
            Action::Pass => self.apply_pass(),
        }
    }

    fn apply_place(
        &mut self,
        row: usize,
        col: usize,
    ) -> Result<Option<GameOutcome>, InvalidAction> {
        if row >= BOARD_SIZE || col >= BOARD_SIZE {
            return Err(InvalidAction::OutOfRange);
        }
        if self.current_player.used_count(self) >= CASTLES_PER_PLAYER {
            return Err(InvalidAction::NoCastlesRemaining);
        }

        let index = row * BOARD_SIZE + col;
        if self.board[index] != Cell::Empty {
            return Err(InvalidAction::OccupiedCell);
        }

        self.board[index] = self.current_player.cell();
        self.increment_current_player_used();
        self.previous_pass = false;
        self.current_player = self.current_player.other();

        Ok(None)
    }

    fn apply_pass(&mut self) -> Result<Option<GameOutcome>, InvalidAction> {
        self.previous_pass = true;
        self.current_player = self.current_player.other();

        Ok(None)
    }

    fn increment_current_player_used(&mut self) {
        match self.current_player {
            Player::Blue => self.blue_used += 1,
            Player::Orange => self.orange_used += 1,
        }
    }
}

#[pyfunction]
#[must_use]
pub fn action_space() -> usize {
    ACTION_SPACE
}

#[pymodule]
fn great_kingdom_core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<GameState>()?;
    module.add_function(wrap_pyfunction!(action_space, module)?)?;
    module.add("BOARD_SIZE", BOARD_SIZE)?;
    module.add("BOARD_CELLS", BOARD_CELLS)?;
    module.add("PASS_ACTION", PASS_ACTION)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn initial_state_has_neutral_center_and_82_actions() {
        let state = GameState::new();

        assert_eq!(state.board[CENTER_INDEX], Cell::Neutral);
        assert_eq!(state.current_player, Player::Blue);
        assert_eq!(state.legal_actions().len(), ACTION_SPACE - 1);
        assert!(state.legal_actions().contains(&PASS_ACTION));
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

    #[test]
    fn new_game_has_no_terminal_outcome() {
        let state = GameState::new();

        assert!(!state.is_terminal());
        assert_eq!(state.winner(), None);
        assert_eq!(state.end_reason(), None);
    }

    #[test]
    fn place_action_updates_board_usage_and_turn() {
        let mut state = GameState::new();

        assert_eq!(state.apply(Action::Place { row: 0, col: 0 }), Ok(None));

        assert_eq!(state.board[0], Cell::Blue);
        assert_eq!(state.blue_used, 1);
        assert_eq!(state.orange_used, 0);
        assert_eq!(state.current_player, Player::Orange);
        assert!(!state.previous_pass);
        assert_eq!(state.legal_actions().len(), 80);
        assert!(!state.legal_actions().contains(&0));
        assert!(state.legal_actions().contains(&PASS_ACTION));
    }

    #[test]
    fn place_action_rejects_occupied_center_and_out_of_range_cells() {
        let mut state = GameState::new();

        assert_eq!(
            state.apply(Action::Place { row: 4, col: 4 }),
            Err(InvalidAction::OccupiedCell)
        );
        assert_eq!(
            state.apply(Action::Place { row: 9, col: 0 }),
            Err(InvalidAction::OutOfRange)
        );
        assert_eq!(state.current_player, Player::Blue);
        assert_eq!(state.blue_used, 0);
    }

    #[test]
    fn player_with_no_castles_remaining_can_only_pass() {
        let mut state = GameState::new();

        for index in 0..BOARD_CELLS {
            if index == CENTER_INDEX {
                continue;
            }
            assert_eq!(state.apply(Action::from_index(index).unwrap()), Ok(None));
        }

        assert_eq!(state.blue_used, CASTLES_PER_PLAYER);
        assert_eq!(state.orange_used, CASTLES_PER_PLAYER);
        assert_eq!(state.current_player, Player::Blue);
        assert_eq!(state.legal_actions(), vec![PASS_ACTION]);
        assert_eq!(
            state.apply(Action::Place { row: 0, col: 0 }),
            Err(InvalidAction::NoCastlesRemaining)
        );
    }
}
