use pyo3::prelude::*;

mod features;
mod game;
mod mcts;
mod rules;
mod territory;

pub use game::{
    ACTION_SPACE, Action, BOARD_CELLS, BOARD_SIZE, CASTLES_PER_PLAYER, CENTER_INDEX, Cell,
    FEATURE_CHANNELS, GameEndReason, GameOutcome, GameState, InvalidAction, PASS_ACTION, Player,
};
pub use mcts::{EvalRequest, MctsConfig, MctsResult, MctsSearch};

#[pyfunction]
#[must_use]
pub fn action_space() -> usize {
    ACTION_SPACE
}

#[pymodule]
fn great_kingdom_core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<GameState>()?;
    module.add_class::<EvalRequest>()?;
    module.add_class::<MctsResult>()?;
    module.add_class::<MctsSearch>()?;
    module.add_function(wrap_pyfunction!(action_space, module)?)?;
    module.add("BOARD_SIZE", BOARD_SIZE)?;
    module.add("BOARD_CELLS", BOARD_CELLS)?;
    module.add("PASS_ACTION", PASS_ACTION)?;
    module.add("FEATURE_CHANNELS", FEATURE_CHANNELS)?;
    Ok(())
}
