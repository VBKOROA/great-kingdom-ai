use pyo3::prelude::*;

use crate::game::ACTION_SPACE;

#[pyclass]
#[derive(Clone, Debug, PartialEq)]
pub struct GumbelResult {
    pub selected_action: Option<usize>,
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
