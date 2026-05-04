use pyo3::prelude::*;

mod eval_request;
mod features;
mod game;
mod gumbel;
mod onnx;
mod rules;
mod territory;

pub use eval_request::EvalRequest;
pub use game::{
    ACTION_SPACE, Action, BOARD_CELLS, BOARD_SIZE, CASTLES_PER_PLAYER, CENTER_INDEX, Cell,
    FEATURE_CHANNELS, GameEndReason, GameOutcome, GameState, InvalidAction, PASS_ACTION, Player,
};
pub use gumbel::{GumbelArenaBatch, GumbelConfig, GumbelResult, GumbelSearch, GumbelSelfPlayBatch};
pub use onnx::{NetworkOutput, OnnxDevice, OnnxError, OnnxEvaluator, OnnxEvaluatorConfig};

#[pyfunction]
#[must_use]
pub fn action_space() -> usize {
    ACTION_SPACE
}

#[pyfunction]
#[must_use]
pub fn rayon_thread_count() -> usize {
    rayon::current_num_threads()
}

#[pymodule]
fn great_kingdom_core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<GameState>()?;
    module.add_class::<EvalRequest>()?;
    module.add_class::<GumbelConfig>()?;
    module.add_class::<GumbelResult>()?;
    module.add_class::<GumbelSearch>()?;
    module.add_class::<GumbelSelfPlayBatch>()?;
    module.add_class::<GumbelArenaBatch>()?;
    module.add_function(wrap_pyfunction!(action_space, module)?)?;
    module.add_function(wrap_pyfunction!(rayon_thread_count, module)?)?;
    module.add("BOARD_SIZE", BOARD_SIZE)?;
    module.add("BOARD_CELLS", BOARD_CELLS)?;
    module.add("PASS_ACTION", PASS_ACTION)?;
    module.add("FEATURE_CHANNELS", FEATURE_CHANNELS)?;
    Ok(())
}
