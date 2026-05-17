use pyo3::{exceptions::PyValueError, prelude::*};

use super::{
    config::GumbelConfig,
    result::GumbelResult,
    search::{GumbelSearch, parse_gumbel_eval_response},
};
use crate::game::GameState;

#[pymethods]
impl GumbelSearch {
    #[new]
    #[pyo3(signature = (
        simulations = 128,
        max_considered_actions = 16,
        c_visit = 50.0,
        c_scale = 1.0,
        seed = 0,
        gumbel_scale = 1.0,
        policy_target_temperature = 1.0,
        policy_target_c_visit = None,
        policy_target_c_scale = None
    ))]
    pub fn py_new(
        simulations: u32,
        max_considered_actions: usize,
        c_visit: f32,
        c_scale: f32,
        seed: u64,
        gumbel_scale: f32,
        policy_target_temperature: f32,
        policy_target_c_visit: Option<f32>,
        policy_target_c_scale: Option<f32>,
    ) -> PyResult<Self> {
        let policy_target_c_visit = policy_target_c_visit
            .ok_or_else(|| PyValueError::new_err("policy_target_c_visit must be set"))?;
        let policy_target_c_scale = policy_target_c_scale
            .ok_or_else(|| PyValueError::new_err("policy_target_c_scale must be set"))?;
        let config = GumbelConfig::new_with_full_config(
            simulations,
            max_considered_actions,
            c_visit,
            c_scale,
            seed,
            gumbel_scale,
            policy_target_temperature,
            policy_target_c_visit,
            policy_target_c_scale,
        );
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
    pub fn gumbel_scale(&self) -> f32 {
        self.config.gumbel_scale
    }

    #[must_use]
    pub fn policy_target_c_visit(&self) -> f32 {
        self.config.policy_target_c_visit
    }

    #[must_use]
    pub fn policy_target_c_scale(&self) -> f32 {
        self.config.policy_target_c_scale
    }

    #[must_use]
    pub fn policy_target_temperature(&self) -> f32 {
        self.config.policy_target_temperature
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

    pub fn set_max_considered_actions(&mut self, max_considered_actions: usize) -> PyResult<()> {
        if max_considered_actions == 0 {
            return Err(PyValueError::new_err(
                "max_considered_actions must be positive",
            ));
        }
        self.config.max_considered_actions = max_considered_actions;
        Ok(())
    }

    pub fn set_seed(&mut self, seed: u64) {
        self.config.seed = seed;
        self.reset_root_search_count();
    }

    pub fn set_gumbel_scale(&mut self, gumbel_scale: f32) -> PyResult<()> {
        if !gumbel_scale.is_finite() || gumbel_scale < 0.0 {
            return Err(PyValueError::new_err(
                "gumbel_scale must be a finite non-negative value",
            ));
        }
        self.config.gumbel_scale = gumbel_scale;
        Ok(())
    }

    pub fn search_with_logits(
        &mut self,
        state: &GameState,
        policy_logits: Vec<f32>,
    ) -> PyResult<GumbelResult> {
        self.result_from_logits(state, &policy_logits)
    }

    #[pyo3(signature = (state, policy_logits, evaluator, root_value, leaf_batch_size = 16))]
    pub fn search_with_logits_and_evaluator(
        &mut self,
        state: &GameState,
        policy_logits: Vec<f32>,
        evaluator: &Bound<'_, PyAny>,
        root_value: f32,
        leaf_batch_size: usize,
    ) -> PyResult<GumbelResult> {
        if leaf_batch_size == 0 {
            return Err(PyValueError::new_err("leaf_batch_size must be positive"));
        }
        self.result_from_logits_with_evaluator(
            state,
            &policy_logits,
            leaf_batch_size,
            root_value,
            |request| {
                let response = evaluator.call1((request,))?;
                parse_gumbel_eval_response(&response)
            },
        )
    }

    pub fn search_with_priors(
        &mut self,
        state: &GameState,
        priors: Vec<f32>,
    ) -> PyResult<GumbelResult> {
        self.result_from_priors(state, &priors)
    }

    #[pyo3(signature = (state, priors, evaluator, root_value, leaf_batch_size = 16))]
    pub fn search_with_priors_and_evaluator(
        &mut self,
        state: &GameState,
        priors: Vec<f32>,
        evaluator: &Bound<'_, PyAny>,
        root_value: f32,
        leaf_batch_size: usize,
    ) -> PyResult<GumbelResult> {
        if leaf_batch_size == 0 {
            return Err(PyValueError::new_err("leaf_batch_size must be positive"));
        }
        self.result_from_priors_with_evaluator(
            state,
            &priors,
            leaf_batch_size,
            root_value,
            |request| {
                let response = evaluator.call1((request,))?;
                parse_gumbel_eval_response(&response)
            },
        )
    }
}
