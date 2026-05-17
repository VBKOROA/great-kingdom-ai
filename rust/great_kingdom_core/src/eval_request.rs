use std::borrow::Cow;

use pyo3::{prelude::*, types::PyBytes};
use rayon::prelude::*;

use crate::game::{ACTION_SPACE, BOARD_CELLS, FEATURE_CHANNELS, GameState};

#[pyclass]
#[derive(Clone, Debug)]
pub struct EvalRequest {
    states: Vec<GameState>,
    feature_values: Option<Vec<f32>>,
    feature_bytes: Option<Vec<u8>>,
    legal_mask_bytes: Option<Vec<u8>>,
    game_indexes: Option<Vec<usize>>,
}

#[pymethods]
impl EvalRequest {
    #[staticmethod]
    pub fn from_feature_rows(feature_rows: Vec<Vec<f32>>) -> PyResult<Self> {
        let row_count = feature_rows.len();
        let expected_row = FEATURE_CHANNELS * BOARD_CELLS;
        let mut feature_values = Vec::with_capacity(row_count * expected_row);
        for row in feature_rows {
            if row.len() != expected_row {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "expected feature row length {expected_row}, got {}",
                    row.len()
                )));
            }
            feature_values.extend(row);
        }
        Ok(Self::from_feature_values(row_count, feature_values))
    }

    #[staticmethod]
    pub fn from_feature_plane_bytes(
        row_count: usize,
        feature_bytes: &Bound<'_, PyBytes>,
    ) -> PyResult<Self> {
        let expected = row_count * FEATURE_CHANNELS * BOARD_CELLS * core::mem::size_of::<f32>();
        let bytes = feature_bytes.as_bytes();
        if bytes.len() != expected {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "expected {expected} feature bytes for {row_count} rows, got {}",
                bytes.len()
            )));
        }
        Ok(Self {
            states: vec![GameState::new(); row_count],
            feature_values: None,
            feature_bytes: Some(bytes.to_vec()),
            legal_mask_bytes: None,
            game_indexes: None,
        })
    }

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
        if let Some(feature_values) = &self.feature_values {
            return feature_values
                .chunks_exact(FEATURE_CHANNELS * BOARD_CELLS)
                .map(Vec::from)
                .collect();
        }
        if let Some(feature_bytes) = &self.feature_bytes {
            return feature_bytes
                .chunks_exact(core::mem::size_of::<f32>())
                .map(|bytes| f32::from_ne_bytes(bytes.try_into().expect("f32 chunks are 4 bytes")))
                .collect::<Vec<_>>()
                .chunks_exact(FEATURE_CHANNELS * BOARD_CELLS)
                .map(Vec::from)
                .collect();
        }
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
        if let Some(feature_values) = &self.feature_values {
            return PyBytes::new(py, f32_slice_as_bytes(feature_values));
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
            masks.extend(state.legal_mask().into_iter().map(u8::from));
        }
        PyBytes::new(py, &masks)
    }

    #[must_use]
    pub fn current_players(&self) -> Vec<u8> {
        self.states.iter().map(GameState::current_player).collect()
    }

    #[must_use]
    pub fn game_indexes(&self) -> Vec<usize> {
        self.game_indexes.clone().unwrap_or_default()
    }
}

impl EvalRequest {
    #[must_use]
    pub fn feature_values(&self) -> Vec<f32> {
        self.feature_values_ref().into_owned()
    }

    #[must_use]
    pub(crate) fn feature_values_ref(&self) -> Cow<'_, [f32]> {
        if let Some(feature_values) = &self.feature_values {
            return Cow::Borrowed(feature_values);
        }
        if let Some(feature_bytes) = &self.feature_bytes {
            return Cow::Owned(
                feature_bytes
                    .chunks_exact(core::mem::size_of::<f32>())
                    .map(|bytes| {
                        f32::from_ne_bytes(bytes.try_into().expect("f32 chunks are 4 bytes"))
                    })
                    .collect(),
            );
        }
        Cow::Owned(
            self.states
                .iter()
                .flat_map(GameState::feature_planes)
                .collect(),
        )
    }

    #[must_use]
    pub(crate) fn new_with_precomputed_bytes(states: Vec<GameState>) -> Self {
        Self::new_with_options(states, None, true)
    }

    #[must_use]
    pub(crate) fn new_with_precomputed_features(states: Vec<GameState>) -> Self {
        Self::new_with_options(states, None, false)
    }

    #[must_use]
    pub(crate) fn new_with_game_indexes(states: Vec<GameState>, game_indexes: Vec<usize>) -> Self {
        assert_eq!(
            states.len(),
            game_indexes.len(),
            "EvalRequest game_indexes length must match states length",
        );
        Self::new_with_options(states, Some(game_indexes), true)
    }

    #[must_use]
    fn new_with_options(
        states: Vec<GameState>,
        game_indexes: Option<Vec<usize>>,
        include_legal_masks: bool,
    ) -> Self {
        let mut features = Vec::with_capacity(states.len() * FEATURE_CHANNELS * BOARD_CELLS);
        features.resize(states.len() * FEATURE_CHANNELS * BOARD_CELLS, 0.0);
        features
            .par_chunks_mut(FEATURE_CHANNELS * BOARD_CELLS)
            .zip(states.par_iter())
            .for_each(|(chunk, state)| {
                chunk.copy_from_slice(&state.feature_planes());
            });

        let legal_mask_bytes = include_legal_masks.then(|| {
            let mut masks = vec![0; states.len() * ACTION_SPACE];
            masks
                .par_chunks_mut(ACTION_SPACE)
                .zip(states.par_iter())
                .for_each(|(chunk, state)| {
                    let legal_mask = state.legal_mask();
                    for (target, is_legal) in chunk.iter_mut().zip(legal_mask.into_iter()) {
                        *target = u8::from(is_legal);
                    }
                });
            masks
        });

        Self {
            states,
            feature_values: Some(features),
            feature_bytes: None,
            legal_mask_bytes,
            game_indexes,
        }
    }

    fn from_feature_values(row_count: usize, feature_values: Vec<f32>) -> Self {
        Self {
            states: vec![GameState::new(); row_count],
            feature_values: Some(feature_values),
            feature_bytes: None,
            legal_mask_bytes: None,
            game_indexes: None,
        }
    }
}

fn f32_slice_as_bytes(values: &[f32]) -> &[u8] {
    let byte_len = core::mem::size_of_val(values);
    let pointer = values.as_ptr().cast::<u8>();
    unsafe { core::slice::from_raw_parts(pointer, byte_len) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn requests_without_metadata_return_empty_game_indexes() {
        let request = EvalRequest::new_with_precomputed_bytes(vec![GameState::new()]);

        assert_eq!(request.game_indexes(), Vec::<usize>::new());
    }

    #[test]
    fn requests_can_carry_game_index_metadata() {
        let request = EvalRequest::new_with_game_indexes(
            vec![GameState::new(), GameState::new()],
            vec![3, 8],
        );

        assert_eq!(request.game_indexes(), vec![3, 8]);
    }

    #[test]
    fn feature_only_requests_can_compute_legal_masks_lazily() {
        let request = EvalRequest::new_with_precomputed_features(vec![GameState::new()]);

        assert_eq!(request.legal_masks()[0], GameState::new().legal_mask());
    }
}
