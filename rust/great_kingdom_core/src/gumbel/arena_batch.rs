use pyo3::{exceptions::PyValueError, prelude::*};

use super::{config::GumbelConfig, search::GumbelSearch};
use crate::{
    eval_request::EvalRequest,
    game::{Action, GameState},
};

const BLUE: u8 = 1;
const ORANGE: u8 = 2;

#[pyclass]
#[derive(Clone, Debug)]
pub struct GumbelArenaBatch {
    states: Vec<GameState>,
    #[allow(dead_code)]
    searches: Vec<[GumbelSearch; 2]>,
    candidate_players: Vec<u8>,
    seeds: Vec<u64>,
}

#[pymethods]
impl GumbelArenaBatch {
    #[new]
    #[pyo3(signature = (
        game_count,
        seed_start = 0,
        game_index_start = 0,
        simulations = 128,
        max_considered_actions = 16,
        c_visit = 50.0,
        c_scale = 1.0,
        seed = 2026
    ))]
    pub fn py_new(
        game_count: usize,
        seed_start: u64,
        game_index_start: usize,
        simulations: u32,
        max_considered_actions: usize,
        c_visit: f32,
        c_scale: f32,
        seed: u64,
    ) -> PyResult<Self> {
        if game_count == 0 {
            return Err(PyValueError::new_err("game_count must be positive"));
        }
        let config = GumbelConfig::new(simulations, max_considered_actions, c_visit, c_scale, seed);
        config.validate()?;
        Ok(Self::new(game_count, seed_start, game_index_start, config))
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
    pub fn active_count(&self) -> usize {
        self.active_indexes().len()
    }

    #[must_use]
    pub fn active_game_indexes(&self) -> Vec<usize> {
        self.active_indexes()
    }

    #[must_use]
    pub fn active_eval_request(&self) -> EvalRequest {
        let active_indexes = self.active_indexes();
        EvalRequest::new_with_game_indexes(
            active_indexes
                .iter()
                .map(|index| self.states[*index].clone())
                .collect(),
            active_indexes,
        )
    }

    #[must_use]
    pub fn current_players(&self) -> Vec<u8> {
        self.states.iter().map(GameState::current_player).collect()
    }

    #[must_use]
    pub fn candidate_players(&self) -> Vec<u8> {
        self.candidate_players.clone()
    }

    #[must_use]
    pub fn seeds(&self) -> Vec<u64> {
        self.seeds.clone()
    }

    #[must_use]
    pub fn is_terminal(&self) -> Vec<bool> {
        self.states.iter().map(GameState::is_terminal).collect()
    }

    #[must_use]
    pub fn winners(&self) -> Vec<Option<u8>> {
        self.states.iter().map(GameState::winner).collect()
    }

    #[must_use]
    pub fn end_reasons(&self) -> Vec<Option<u8>> {
        self.states.iter().map(GameState::end_reason).collect()
    }

    #[must_use]
    pub fn territory_scores(&self) -> Vec<(u8, u8)> {
        self.states
            .iter()
            .map(GameState::territory_scores)
            .collect()
    }

    pub fn apply_actions(&mut self, actions: Vec<Option<usize>>) -> PyResult<Vec<Option<u8>>> {
        if actions.len() != self.states.len() {
            return Err(PyValueError::new_err(format!(
                "expected {} action slots, got {}",
                self.states.len(),
                actions.len()
            )));
        }

        let mut outcomes = Vec::with_capacity(actions.len());
        for (game_index, action_index) in actions.into_iter().enumerate() {
            let Some(action_index) = action_index else {
                outcomes.push(None);
                continue;
            };
            let action = Action::from_index(action_index).ok_or_else(|| {
                PyValueError::new_err(format!("invalid action index: {action_index}"))
            })?;
            let outcome = self.states[game_index]
                .apply(action)
                .map_err(|err| PyValueError::new_err(format!("invalid action: {err:?}")))?;
            outcomes.push(outcome.map(|outcome| outcome.winner as u8));
        }
        Ok(outcomes)
    }
}

impl GumbelArenaBatch {
    #[must_use]
    pub fn new(
        game_count: usize,
        seed_start: u64,
        game_index_start: usize,
        config: GumbelConfig,
    ) -> Self {
        let mut searches = Vec::with_capacity(game_count);
        let mut candidate_players = Vec::with_capacity(game_count);
        let mut seeds = Vec::with_capacity(game_count);

        for chunk_index in 0..game_count {
            let game_index = game_index_start + chunk_index;
            let game_seed = seed_start.wrapping_add(chunk_index as u64);
            seeds.push(game_seed);
            candidate_players.push(if game_index % 2 == 0 { BLUE } else { ORANGE });

            let mut blue_config = config;
            blue_config.seed = config.seed.wrapping_add(game_seed.wrapping_mul(2));
            let mut orange_config = config;
            orange_config.seed = config
                .seed
                .wrapping_add(game_seed.wrapping_mul(2).wrapping_add(1));
            searches.push([
                GumbelSearch::new(blue_config),
                GumbelSearch::new(orange_config),
            ]);
        }

        Self {
            states: vec![GameState::new(); game_count],
            searches,
            candidate_players,
            seeds,
        }
    }

    fn active_indexes(&self) -> Vec<usize> {
        self.states
            .iter()
            .enumerate()
            .filter_map(|(index, state)| (!state.is_terminal()).then_some(index))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::GumbelArenaBatch;
    use crate::gumbel::config::GumbelConfig;

    #[test]
    fn new_uses_global_game_index_for_candidate_side_split() {
        let batch = GumbelArenaBatch::new(4, 10, 1, GumbelConfig::new(4, 2, 50.0, 1.0, 7));

        assert_eq!(batch.seeds, vec![10, 11, 12, 13]);
        assert_eq!(batch.candidate_players, vec![2, 1, 2, 1]);
    }

    #[test]
    fn new_offsets_player_search_seeds_from_game_seed() {
        let batch = GumbelArenaBatch::new(2, 10, 0, GumbelConfig::new(4, 2, 50.0, 1.0, 7));

        assert_eq!(batch.searches[0][0].seed(), 27);
        assert_eq!(batch.searches[0][1].seed(), 28);
        assert_eq!(batch.searches[1][0].seed(), 29);
        assert_eq!(batch.searches[1][1].seed(), 30);
    }
}
