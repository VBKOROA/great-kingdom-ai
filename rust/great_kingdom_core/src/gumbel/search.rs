use pyo3::{exceptions::PyValueError, prelude::*};

use super::{
    config::GumbelConfig,
    policy::{log_priors_from_logits, log_priors_from_priors},
    result::GumbelResult,
    sampling::{RootCandidate, sample_root_candidates, softmax_candidates},
    sequential_halving::RootSequentialHalving,
};
use crate::game::{ACTION_SPACE, GameState};

#[pyclass]
#[derive(Clone, Debug)]
pub struct GumbelSearch {
    config: GumbelConfig,
}

#[pymethods]
impl GumbelSearch {
    #[new]
    #[pyo3(signature = (
        simulations = 128,
        max_considered_actions = 16,
        c_visit = 50.0,
        c_scale = 1.0,
        seed = 0
    ))]
    pub fn py_new(
        simulations: u32,
        max_considered_actions: usize,
        c_visit: f32,
        c_scale: f32,
        seed: u64,
    ) -> PyResult<Self> {
        let config = GumbelConfig::new(simulations, max_considered_actions, c_visit, c_scale, seed);
        config.validate()?;
        Ok(Self::new(config))
    }

    #[must_use]
    pub fn simulations(&self) -> u32 {
        self.config.simulations
    }

    #[must_use]
    pub fn max_considered_actions(&self) -> usize {
        self.config.max_considered_actions
    }

    #[must_use]
    pub fn c_visit(&self) -> f32 {
        self.config.c_visit
    }

    #[must_use]
    pub fn c_scale(&self) -> f32 {
        self.config.c_scale
    }

    #[must_use]
    pub fn seed(&self) -> u64 {
        self.config.seed
    }

    pub fn set_simulations(&mut self, simulations: u32) -> PyResult<()> {
        if simulations == 0 {
            return Err(PyValueError::new_err("simulations must be positive"));
        }
        self.config.simulations = simulations;
        Ok(())
    }

    pub fn set_seed(&mut self, seed: u64) {
        self.config.seed = seed;
    }

    pub fn search_with_logits(
        &mut self,
        state: &GameState,
        policy_logits: Vec<f32>,
    ) -> PyResult<GumbelResult> {
        self.result_from_logits(state, &policy_logits)
    }

    #[pyo3(signature = (state, policy_logits, evaluator, leaf_batch_size = 16))]
    pub fn search_with_logits_and_evaluator(
        &mut self,
        state: &GameState,
        policy_logits: Vec<f32>,
        evaluator: &Bound<'_, PyAny>,
        leaf_batch_size: usize,
    ) -> PyResult<GumbelResult> {
        let _ = evaluator;
        if leaf_batch_size == 0 {
            return Err(PyValueError::new_err("leaf_batch_size must be positive"));
        }
        self.search_with_logits(state, policy_logits)
    }

    pub fn search_with_priors(
        &mut self,
        state: &GameState,
        priors: Vec<f32>,
    ) -> PyResult<GumbelResult> {
        self.result_from_priors(state, &priors)
    }

    #[pyo3(signature = (state, priors, evaluator, leaf_batch_size = 16))]
    pub fn search_with_priors_and_evaluator(
        &mut self,
        state: &GameState,
        priors: Vec<f32>,
        evaluator: &Bound<'_, PyAny>,
        leaf_batch_size: usize,
    ) -> PyResult<GumbelResult> {
        let _ = evaluator;
        if leaf_batch_size == 0 {
            return Err(PyValueError::new_err("leaf_batch_size must be positive"));
        }
        self.search_with_priors(state, priors)
    }
}

impl GumbelSearch {
    #[must_use]
    pub const fn new(config: GumbelConfig) -> Self {
        Self { config }
    }

    #[must_use]
    pub(crate) fn result_from_logits(
        &self,
        state: &GameState,
        logits: &[f32],
    ) -> PyResult<GumbelResult> {
        let legal_actions = state.legal_action_indexes();
        let log_priors = log_priors_from_logits(&legal_actions, logits)?;
        Ok(self.result_from_log_priors(state, &legal_actions, &log_priors))
    }

    pub(crate) fn result_from_priors(
        &self,
        state: &GameState,
        priors: &[f32],
    ) -> PyResult<GumbelResult> {
        let legal_actions = state.legal_action_indexes();
        let log_priors = log_priors_from_priors(&legal_actions, priors)?;
        Ok(self.result_from_log_priors(state, &legal_actions, &log_priors))
    }

    #[must_use]
    pub(crate) fn result_from_log_priors(
        &self,
        state: &GameState,
        legal_actions: &[usize],
        log_priors: &[f32; ACTION_SPACE],
    ) -> GumbelResult {
        if legal_actions.is_empty() || state.is_terminal() {
            return GumbelResult {
                selected_action: None,
                policy_target: [0.0; ACTION_SPACE],
                visit_counts: [0; ACTION_SPACE],
            };
        }

        let candidates = sample_root_candidates(
            legal_actions,
            log_priors,
            self.config.max_considered_actions,
            self.config.simulations,
            self.config.seed,
        );
        self.result_from_candidates(&candidates)
    }

    #[must_use]
    pub(crate) fn result_from_candidates(&self, candidates: &[RootCandidate]) -> GumbelResult {
        if candidates.is_empty() {
            return GumbelResult {
                selected_action: None,
                policy_target: [0.0; ACTION_SPACE],
                visit_counts: [0; ACTION_SPACE],
            };
        }

        let selected_action = candidates
            .iter()
            .max_by(|left, right| {
                left.score
                    .total_cmp(&right.score)
                    .then_with(|| right.action.cmp(&left.action))
            })
            .map(|candidate| candidate.action);

        let mut scheduler = RootSequentialHalving::new(
            candidates
                .iter()
                .map(|candidate| (candidate.action, candidate.score))
                .collect(),
            self.config.simulations,
        );
        for _ in 0..self.config.simulations {
            let Some(action) = scheduler.next_action() else {
                break;
            };
            scheduler.record_visit(action);
            if scheduler.is_finished() {
                break;
            }
        }

        let mut visit_counts = [0; ACTION_SPACE];
        for (action, visits) in scheduler.completed_visits() {
            visit_counts[action] = visits;
        }

        GumbelResult {
            selected_action,
            policy_target: softmax_candidates(candidates),
            visit_counts,
        }
    }

    #[must_use]
    pub fn skeleton_result(&self, state: &GameState, scores: Option<&[f32]>) -> GumbelResult {
        let legal_actions = state.legal_action_indexes();
        if legal_actions.is_empty() || state.is_terminal() {
            return GumbelResult {
                selected_action: None,
                policy_target: [0.0; ACTION_SPACE],
                visit_counts: [0; ACTION_SPACE],
            };
        }

        let selected_action = legal_actions.iter().copied().max_by(|left, right| {
            let left_score = scores.map_or(0.0, |values| values[*left]);
            let right_score = scores.map_or(0.0, |values| values[*right]);
            left_score
                .partial_cmp(&right_score)
                .expect("validated policy scores must be finite")
                .then_with(|| right.cmp(left))
        });

        let mut policy_target = [0.0; ACTION_SPACE];
        let target = 1.0 / legal_actions.len() as f32;
        for action in legal_actions {
            policy_target[action] = target;
        }

        GumbelResult {
            selected_action,
            policy_target,
            visit_counts: [0; ACTION_SPACE],
        }
    }
}
