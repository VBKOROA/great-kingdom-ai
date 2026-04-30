use pyo3::prelude::*;

pub const BOARD_SIZE: usize = 9;
pub const BOARD_CELLS: usize = BOARD_SIZE * BOARD_SIZE;
pub const PASS_ACTION: usize = BOARD_CELLS;
pub const ACTION_SPACE: usize = BOARD_CELLS + 1;
pub const CENTER_INDEX: usize = 40;

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
    pub const fn other(self) -> Self {
        match self {
            Self::Blue => Self::Orange,
            Self::Orange => Self::Blue,
        }
    }
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
    pub fn legal_actions(&self) -> Vec<usize> {
        if self.terminal {
            return Vec::new();
        }

        let mut actions: Vec<usize> = self
            .board
            .iter()
            .enumerate()
            .filter_map(|(idx, cell)| (*cell == Cell::Empty).then_some(idx))
            .collect();
        actions.push(PASS_ACTION);
        actions
    }

    #[must_use]
    pub fn is_terminal(&self) -> bool {
        self.terminal
    }
}

impl Default for GameState {
    fn default() -> Self {
        Self::new()
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
}
