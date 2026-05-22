use pyo3::{exceptions::PyRuntimeError, prelude::*};

use super::search::{GumbelEvalBatch, parse_gumbel_eval_response};
use crate::{eval_request::EvalRequest, onnx::OnnxEvaluator};

pub(crate) trait GumbelEvaluator {
    fn check_signals(&self) -> PyResult<()> {
        Ok(())
    }

    fn needs_legal_masks(&self) -> bool {
        true
    }

    fn needs_game_indexes(&self) -> bool {
        false
    }

    fn set_batch_profile_context(&mut self, _wave: u64, _active_games: usize, _leaves: usize) {}

    fn evaluate(&mut self, request: EvalRequest) -> PyResult<GumbelEvalBatch>;
}

pub(crate) struct PythonGumbelEvaluator<'a, 'py> {
    callback: &'a Bound<'py, PyAny>,
}

impl<'a, 'py> PythonGumbelEvaluator<'a, 'py> {
    pub(crate) const fn new(callback: &'a Bound<'py, PyAny>) -> Self {
        Self { callback }
    }
}

impl GumbelEvaluator for PythonGumbelEvaluator<'_, '_> {
    fn check_signals(&self) -> PyResult<()> {
        self.callback.py().check_signals()
    }

    fn evaluate(&mut self, request: EvalRequest) -> PyResult<GumbelEvalBatch> {
        let response = self.callback.call1((request,))?;
        parse_gumbel_eval_response(&response)
    }
}

pub(crate) struct OnnxGumbelEvaluator<'a> {
    evaluator: &'a mut OnnxEvaluator,
}

impl<'a> OnnxGumbelEvaluator<'a> {
    pub(crate) fn new(evaluator: &'a mut OnnxEvaluator) -> Self {
        Self { evaluator }
    }
}

impl GumbelEvaluator for OnnxGumbelEvaluator<'_> {
    fn needs_legal_masks(&self) -> bool {
        false
    }

    fn set_batch_profile_context(&mut self, wave: u64, active_games: usize, leaves: usize) {
        self.evaluator
            .set_gumbel_leaf_profile_context(wave, active_games, leaves);
    }

    fn evaluate(&mut self, request: EvalRequest) -> PyResult<GumbelEvalBatch> {
        let output = self
            .evaluator
            .evaluate_request(&request)
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
        Ok(GumbelEvalBatch::new(output.policy_logits, output.values))
    }
}
