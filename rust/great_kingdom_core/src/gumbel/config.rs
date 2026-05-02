use pyo3::{exceptions::PyValueError, prelude::*};

#[pyclass]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GumbelConfig {
    pub simulations: u32,
    pub max_considered_actions: usize,
    pub c_visit: f32,
    pub c_scale: f32,
    pub seed: u64,
}

impl Default for GumbelConfig {
    fn default() -> Self {
        Self {
            simulations: 128,
            max_considered_actions: 16,
            c_visit: 50.0,
            c_scale: 1.0,
            seed: 0,
        }
    }
}

impl GumbelConfig {
    #[must_use]
    pub const fn new(
        simulations: u32,
        max_considered_actions: usize,
        c_visit: f32,
        c_scale: f32,
        seed: u64,
    ) -> Self {
        Self {
            simulations,
            max_considered_actions,
            c_visit,
            c_scale,
            seed,
        }
    }

    pub fn validate(&self) -> PyResult<()> {
        if self.simulations == 0 {
            return Err(PyValueError::new_err("simulations must be positive"));
        }
        if self.max_considered_actions == 0 {
            return Err(PyValueError::new_err(
                "max_considered_actions must be positive",
            ));
        }
        if !self.c_visit.is_finite() || self.c_visit <= 0.0 {
            return Err(PyValueError::new_err(
                "c_visit must be a finite positive value",
            ));
        }
        if !self.c_scale.is_finite() || self.c_scale <= 0.0 {
            return Err(PyValueError::new_err(
                "c_scale must be a finite positive value",
            ));
        }
        Ok(())
    }
}

#[pymethods]
impl GumbelConfig {
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
        let config = Self::new(simulations, max_considered_actions, c_visit, c_scale, seed);
        config.validate()?;
        Ok(config)
    }

    #[must_use]
    pub fn simulations(&self) -> u32 {
        self.simulations
    }

    #[must_use]
    pub fn max_considered_actions(&self) -> usize {
        self.max_considered_actions
    }

    #[must_use]
    pub fn c_visit(&self) -> f32 {
        self.c_visit
    }

    #[must_use]
    pub fn c_scale(&self) -> f32 {
        self.c_scale
    }

    #[must_use]
    pub fn seed(&self) -> u64 {
        self.seed
    }
}
