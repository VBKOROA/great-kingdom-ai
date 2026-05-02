use pyo3::{prelude::*, types::PyBytes};
use rayon::prelude::*;

use crate::game::{ACTION_SPACE, BOARD_CELLS, FEATURE_CHANNELS, GameState};

#[pyclass]
#[derive(Clone, Debug)]
pub struct EvalRequest {
    states: Vec<GameState>,
    feature_bytes: Option<Vec<u8>>,
    legal_mask_bytes: Option<Vec<u8>>,
}

#[pymethods]
impl EvalRequest {
    #[must_use]
    pub fn len(&self) -> usize {
        self.states.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    #[must_use]
    pub fn feature_planes(&self) -> Vec<Vec<f32>> {
        self.states
            .iter()
            .map(GameState::feature_planes)
            .collect::<Vec<_>>()
    }

    #[must_use]
    pub fn feature_plane_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        if let Some(feature_bytes) = &self.feature_bytes {
            return PyBytes::new(py, feature_bytes);
        }
        let mut features = Vec::with_capacity(self.states.len() * FEATURE_CHANNELS * BOARD_CELLS);
        for state in &self.states {
            features.extend(state.feature_planes());
        }
        PyBytes::new(py, f32_slice_as_bytes(&features))
    }

    #[must_use]
    pub fn legal_masks(&self) -> Vec<Vec<bool>> {
        self.states
            .iter()
            .map(GameState::legal_mask)
            .collect::<Vec<_>>()
    }

    #[must_use]
    pub fn legal_mask_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        if let Some(legal_mask_bytes) = &self.legal_mask_bytes {
            return PyBytes::new(py, legal_mask_bytes);
        }
        let mut masks = Vec::with_capacity(self.states.len() * ACTION_SPACE);
        for state in &self.states {
            masks.extend(
                state
                    .legal_mask()
                    .into_iter()
                    .map(|is_legal| u8::from(is_legal)),
            );
        }
        PyBytes::new(py, &masks)
    }

    #[must_use]
    pub fn current_players(&self) -> Vec<u8> {
        self.states.iter().map(GameState::current_player).collect()
    }
}

impl EvalRequest {
    #[must_use]
    pub(crate) fn new_with_precomputed_bytes(states: Vec<GameState>) -> Self {
        let mut features = Vec::with_capacity(states.len() * FEATURE_CHANNELS * BOARD_CELLS);
        features.resize(states.len() * FEATURE_CHANNELS * BOARD_CELLS, 0.0);
        features
            .par_chunks_mut(FEATURE_CHANNELS * BOARD_CELLS)
            .zip(states.par_iter())
            .for_each(|(chunk, state)| {
                chunk.copy_from_slice(&state.feature_planes());
            });

        let mut masks = Vec::with_capacity(states.len() * ACTION_SPACE);
        masks.resize(states.len() * ACTION_SPACE, 0);
        masks
            .par_chunks_mut(ACTION_SPACE)
            .zip(states.par_iter())
            .for_each(|(chunk, state)| {
                let legal_mask = state.legal_mask();
                for (target, is_legal) in chunk.iter_mut().zip(legal_mask.into_iter()) {
                    *target = u8::from(is_legal);
                }
            });

        Self {
            states,
            feature_bytes: Some(f32_slice_as_bytes(&features).to_vec()),
            legal_mask_bytes: Some(masks),
        }
    }
}

fn f32_slice_as_bytes(values: &[f32]) -> &[u8] {
    let byte_len = core::mem::size_of_val(values);
    let pointer = values.as_ptr().cast::<u8>();
    unsafe { core::slice::from_raw_parts(pointer, byte_len) }
}
