use pyo3::prelude::*;

use crate::game::ACTION_SPACE;

#[pyclass]
#[derive(Clone, Debug, PartialEq)]
pub struct GumbelResult {
    pub selected_action: Option<usize>,
    pub selected_action_q: Option<f32>,
    pub selected_child_visit_counts: [u32; ACTION_SPACE],
    pub selected_child_completed_q: [f32; ACTION_SPACE],
    pub selected_child_log_priors: [f32; ACTION_SPACE],
    pub policy_target: [f32; ACTION_SPACE],
    pub visit_counts: [u32; ACTION_SPACE],
    pub root_value: f32,
}

#[pymethods]
impl GumbelResult {
    #[must_use]
    pub fn selected_action(&self) -> Option<usize> {
        self.selected_action
    }

    #[must_use]
    pub fn selected_action_q(&self) -> Option<f32> {
        self.selected_action_q
    }

    #[must_use]
    pub fn selected_child_visit_counts(&self) -> Vec<u32> {
        self.selected_child_visit_counts.to_vec()
    }

    #[must_use]
    pub fn selected_child_completed_q(&self) -> Vec<f32> {
        self.selected_child_completed_q.to_vec()
    }

    #[must_use]
    pub fn selected_child_log_priors(&self) -> Vec<f32> {
        self.selected_child_log_priors.to_vec()
    }

    #[must_use]
    pub fn policy_target(&self) -> Vec<f32> {
        self.policy_target.to_vec()
    }

    #[must_use]
    pub fn visit_counts(&self) -> Vec<u32> {
        self.visit_counts.to_vec()
    }

    #[must_use]
    pub fn root_value(&self) -> f32 {
        self.root_value
    }
}
