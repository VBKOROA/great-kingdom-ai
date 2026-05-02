use pyo3::{
    buffer::PyBuffer,
    exceptions::PyValueError,
    prelude::*,
    types::{PyAny, PyBytes},
};
use rayon::prelude::*;
use std::{env, time::Instant};

use crate::game::{
    ACTION_SPACE, Action, BOARD_CELLS, FEATURE_CHANNELS, GameOutcome, GameState, Player,
};

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
    pub(crate) fn new(states: Vec<GameState>) -> Self {
        Self {
            states,
            feature_bytes: None,
            legal_mask_bytes: None,
        }
    }

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

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MctsConfig {
    pub simulations: u32,
    pub c_puct: f32,
}

impl Default for MctsConfig {
    fn default() -> Self {
        Self {
            simulations: 50,
            c_puct: 1.5,
        }
    }
}

impl MctsConfig {
    #[must_use]
    pub const fn new(simulations: u32, c_puct: f32) -> Self {
        Self {
            simulations,
            c_puct,
        }
    }
}

#[pyclass]
#[derive(Clone, Debug, PartialEq)]
pub struct MctsResult {
    pub selected_action: Option<usize>,
    pub visit_counts: [u32; ACTION_SPACE],
}

#[pymethods]
impl MctsResult {
    #[must_use]
    pub fn selected_action(&self) -> Option<usize> {
        self.selected_action
    }

    #[must_use]
    pub fn visit_counts(&self) -> Vec<u32> {
        self.visit_counts.to_vec()
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct MctsSearch {
    config: MctsConfig,
    nodes: Vec<Node>,
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct MctsSelfPlayBatch {
    states: Vec<GameState>,
    searches: Vec<MctsSearch>,
}

#[pymethods]
impl MctsSearch {
    #[new]
    #[pyo3(signature = (simulations = 50, c_puct = 1.5))]
    pub fn py_new(simulations: u32, c_puct: f32) -> PyResult<Self> {
        if simulations == 0 {
            return Err(PyValueError::new_err("simulations must be positive"));
        }
        if !c_puct.is_finite() || c_puct < 0.0 {
            return Err(PyValueError::new_err(
                "c_puct must be a finite non-negative value",
            ));
        }

        Ok(Self::new(MctsConfig::new(simulations, c_puct)))
    }

    #[must_use]
    pub fn simulations(&self) -> u32 {
        self.config.simulations
    }

    pub fn set_simulations(&mut self, simulations: u32) -> PyResult<()> {
        if simulations == 0 {
            return Err(PyValueError::new_err("simulations must be positive"));
        }
        self.config.simulations = simulations;
        Ok(())
    }

    #[must_use]
    pub fn c_puct(&self) -> f32 {
        self.config.c_puct
    }

    pub fn search(&mut self, state: &GameState) -> MctsResult {
        self.run(state)
    }

    pub fn search_with_priors(
        &mut self,
        state: &GameState,
        priors: Vec<f32>,
    ) -> PyResult<MctsResult> {
        if priors.len() != ACTION_SPACE {
            return Err(PyValueError::new_err(format!(
                "expected {ACTION_SPACE} priors, got {}",
                priors.len()
            )));
        }
        if priors
            .iter()
            .any(|prior| !prior.is_finite() || *prior < 0.0)
        {
            return Err(PyValueError::new_err(
                "priors must be finite non-negative values",
            ));
        }

        let mut prior_array = [0.0; ACTION_SPACE];
        prior_array.copy_from_slice(&priors);
        Ok(self.run_with_root_priors(state, &prior_array))
    }

    pub fn root_eval_request(&self, state: &GameState) -> EvalRequest {
        EvalRequest::new(vec![state.clone()])
    }

    #[pyo3(signature = (state, evaluator, leaf_batch_size = 8))]
    pub fn search_with_evaluator(
        &mut self,
        state: &GameState,
        evaluator: &Bound<'_, PyAny>,
        leaf_batch_size: usize,
    ) -> PyResult<MctsResult> {
        if leaf_batch_size == 0 {
            return Err(PyValueError::new_err("leaf_batch_size must be positive"));
        }

        self.run_with_leaf_evaluator(state, leaf_batch_size, |request| {
            let response = evaluator.call1((request,))?;
            parse_eval_response(&response)
        })
    }

    #[pyo3(signature = (state, priors, evaluator, leaf_batch_size = 8))]
    pub fn search_with_priors_and_evaluator(
        &mut self,
        state: &GameState,
        priors: Vec<f32>,
        evaluator: &Bound<'_, PyAny>,
        leaf_batch_size: usize,
    ) -> PyResult<MctsResult> {
        if leaf_batch_size == 0 {
            return Err(PyValueError::new_err("leaf_batch_size must be positive"));
        }
        let prior_array = parse_policy_row(priors, 0)?;
        self.run_with_root_priors_and_leaf_evaluator(
            state,
            &prior_array,
            leaf_batch_size,
            |request| {
                let response = evaluator.call1((request,))?;
                parse_eval_response(&response)
            },
        )
    }
}

#[pymethods]
impl MctsSelfPlayBatch {
    #[new]
    #[pyo3(signature = (game_count, simulations = 50, c_puct = 1.5))]
    pub fn py_new(game_count: usize, simulations: u32, c_puct: f32) -> PyResult<Self> {
        if game_count == 0 {
            return Err(PyValueError::new_err("game_count must be positive"));
        }
        if simulations == 0 {
            return Err(PyValueError::new_err("simulations must be positive"));
        }
        if !c_puct.is_finite() || c_puct < 0.0 {
            return Err(PyValueError::new_err(
                "c_puct must be a finite non-negative value",
            ));
        }
        Ok(Self::new(game_count, MctsConfig::new(simulations, c_puct)))
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

    #[must_use]
    pub fn current_players(&self) -> Vec<u8> {
        self.states.iter().map(GameState::current_player).collect()
    }

    #[must_use]
    pub fn active_eval_request(&self) -> EvalRequest {
        EvalRequest::new(
            self.active_indexes()
                .into_iter()
                .map(|index| self.states[index].clone())
                .collect(),
        )
    }

    pub fn play_turns_with_priors(
        &mut self,
        priors: Vec<Vec<f32>>,
    ) -> PyResult<Vec<Option<MctsResult>>> {
        let results = self.search_active_with_priors(priors)?;
        let actions = results
            .iter()
            .map(|result| result.as_ref().and_then(MctsResult::selected_action))
            .collect();
        self.apply_actions(actions)?;
        Ok(results)
    }

    pub fn search_active_with_priors(
        &mut self,
        priors: Vec<Vec<f32>>,
    ) -> PyResult<Vec<Option<MctsResult>>> {
        let active_indexes = self.active_indexes();
        if priors.len() != active_indexes.len() {
            return Err(PyValueError::new_err(format!(
                "expected {} prior rows for active games, got {}",
                active_indexes.len(),
                priors.len()
            )));
        }

        let mut results = vec![None; self.states.len()];
        for (game_index, prior_row) in active_indexes.into_iter().zip(priors.into_iter()) {
            let prior_array = parse_policy_row(prior_row, game_index)?;
            let result = self.searches[game_index]
                .run_with_root_priors(&self.states[game_index], &prior_array);
            results[game_index] = Some(result);
        }
        Ok(results)
    }

    #[pyo3(signature = (priors, evaluator, leaf_batch_size = 8))]
    pub fn search_active_with_priors_and_evaluator(
        &mut self,
        priors: Vec<Vec<f32>>,
        evaluator: &Bound<'_, PyAny>,
        leaf_batch_size: usize,
    ) -> PyResult<Vec<Option<MctsResult>>> {
        if leaf_batch_size == 0 {
            return Err(PyValueError::new_err("leaf_batch_size must be positive"));
        }
        let active_indexes = self.active_indexes();
        if priors.len() != active_indexes.len() {
            return Err(PyValueError::new_err(format!(
                "expected {} prior rows for active games, got {}",
                active_indexes.len(),
                priors.len()
            )));
        }

        let profile = MctsProfile::new("search_active_with_priors_and_evaluator");
        let root_start = Instant::now();
        let mut root_indexes = vec![None; self.states.len()];
        let mut completed = vec![0; self.states.len()];
        for (game_index, prior_row) in active_indexes.iter().copied().zip(priors.into_iter()) {
            let prior_array = parse_policy_row(prior_row, game_index)?;
            self.searches[game_index].nodes.clear();
            root_indexes[game_index] = Some(
                self.searches[game_index]
                    .expand_node_with_priors(&self.states[game_index], &prior_array),
            );
        }
        profile.root(active_indexes.len(), root_start.elapsed());

        let mut wave = 0_u64;
        while active_indexes
            .iter()
            .any(|index| completed[*index] < self.searches[*index].config.simulations)
        {
            wave += 1;
            evaluator.py().check_signals()?;
            let select_start = Instant::now();
            let pending_by_game: Vec<Vec<_>> = self
                .states
                .par_iter()
                .zip(self.searches.par_iter_mut())
                .zip(completed.par_iter_mut())
                .zip(root_indexes.par_iter())
                .enumerate()
                .map(|(game_index, (((state, search), comp), root_index))| {
                    let Some(root_index) = *root_index else {
                        return Vec::new();
                    };
                    let batch_target =
                        (search.config.simulations - *comp).min(leaf_batch_size as u32) as usize;
                    let mut local_pending = Vec::with_capacity(batch_target);

                    for _ in 0..batch_target {
                        if *comp + local_pending.len() as u32 >= search.config.simulations {
                            break;
                        }
                        let mut simulation_state = state.clone();
                        match search.select_eval_leaf(root_index, &mut simulation_state) {
                            PendingSimulation::NeedsEvaluation {
                                path,
                                state: leaf_state,
                            } => {
                                reserve_path(&mut search.nodes, &path);
                                local_pending.push(PendingGameLeaf {
                                    game_index,
                                    path,
                                    state: leaf_state,
                                });
                            }
                            PendingSimulation::Terminal {
                                path,
                                last_edge_value,
                            } => {
                                backup_path_from_last_edge(
                                    &mut search.nodes,
                                    &path,
                                    last_edge_value,
                                );
                                *comp += 1;
                            }
                            PendingSimulation::RootTerminal => {
                                *comp += 1;
                            }
                        }
                    }
                    local_pending
                })
                .collect();
            let selected_games = pending_by_game
                .iter()
                .filter(|leaves| !leaves.is_empty())
                .count();
            let select_elapsed = select_start.elapsed();
            let flatten_start = Instant::now();
            let pending_leaves = pending_by_game.into_iter().flatten().collect::<Vec<_>>();
            let flatten_elapsed = flatten_start.elapsed();

            if pending_leaves.is_empty() {
                profile.empty_wave(wave, selected_games, select_elapsed, flatten_elapsed);
                continue;
            }

            let request_start = Instant::now();
            let request_states = pending_leaves
                .iter()
                .map(|leaf| leaf.state.clone())
                .collect::<Vec<_>>();
            let request = EvalRequest::new_with_precomputed_bytes(request_states);
            let request_elapsed = request_start.elapsed();
            let eval_start = Instant::now();
            let response = evaluator.call1((request,))?;
            let eval_elapsed = eval_start.elapsed();
            let parse_start = Instant::now();
            let eval = parse_eval_response(&response)?;
            eval.validate_len(pending_leaves.len())?;
            let parse_elapsed = parse_start.elapsed();

            let backup_start = Instant::now();
            let leaf_count = pending_leaves.len();
            backup_pending_game_evaluations(
                &mut self.searches,
                &mut completed,
                pending_leaves,
                eval,
            );
            let backup_elapsed = backup_start.elapsed();
            profile.wave(MctsWaveProfile {
                wave,
                active_games: active_indexes.len(),
                selected_games,
                leaves: leaf_count,
                select_elapsed,
                flatten_elapsed,
                request_elapsed,
                eval_elapsed,
                parse_elapsed,
                backup_elapsed,
            });
        }

        let mut results = vec![None; self.states.len()];
        for game_index in active_indexes {
            let root_index = root_indexes[game_index]
                .expect("active game root must be initialized before result export");
            let root = &self.searches[game_index].nodes[root_index];
            results[game_index] = Some(MctsResult {
                selected_action: root.most_visited_action(),
                visit_counts: root.visit_counts(),
            });
        }
        Ok(results)
    }

    #[pyo3(signature = (evaluator, leaf_batch_size = 8))]
    pub fn play_turns_with_evaluator(
        &mut self,
        evaluator: &Bound<'_, PyAny>,
        leaf_batch_size: usize,
    ) -> PyResult<Vec<Option<MctsResult>>> {
        let results = self.search_active_with_evaluator(evaluator, leaf_batch_size)?;
        let actions = results
            .iter()
            .map(|result| result.as_ref().and_then(MctsResult::selected_action))
            .collect();
        self.apply_actions(actions)?;
        Ok(results)
    }

    #[pyo3(signature = (evaluator, leaf_batch_size = 8))]
    pub fn search_active_with_evaluator(
        &mut self,
        evaluator: &Bound<'_, PyAny>,
        leaf_batch_size: usize,
    ) -> PyResult<Vec<Option<MctsResult>>> {
        if leaf_batch_size == 0 {
            return Err(PyValueError::new_err("leaf_batch_size must be positive"));
        }

        let active_indexes = self.active_indexes();

        let profile = MctsProfile::new("search_active_with_evaluator");
        let root_eval_start = Instant::now();
        let root_states = active_indexes
            .iter()
            .map(|index| self.states[*index].clone())
            .collect::<Vec<_>>();
        let root_request_elapsed = root_eval_start.elapsed();
        let root_call_start = Instant::now();
        let root_response = evaluator.call1((EvalRequest::new(root_states),))?;
        let root_call_elapsed = root_call_start.elapsed();
        let root_parse_start = Instant::now();
        let root_eval = parse_eval_response(&root_response)?;
        root_eval.validate_len(active_indexes.len())?;
        let root_parse_elapsed = root_parse_start.elapsed();

        let root_expand_start = Instant::now();
        let mut root_indexes = vec![None; self.states.len()];
        let mut completed = vec![0; self.states.len()];
        for (game_index, prior_row) in active_indexes
            .iter()
            .copied()
            .zip(root_eval.policies.into_iter())
        {
            self.searches[game_index].nodes.clear();
            root_indexes[game_index] = Some(
                self.searches[game_index]
                    .expand_node_with_priors(&self.states[game_index], &prior_row),
            );
        }
        profile.root_eval(
            active_indexes.len(),
            root_request_elapsed,
            root_call_elapsed,
            root_parse_elapsed,
            root_expand_start.elapsed(),
        );

        let mut wave = 0_u64;
        while active_indexes
            .iter()
            .any(|index| completed[*index] < self.searches[*index].config.simulations)
        {
            wave += 1;
            evaluator.py().check_signals()?;
            let select_start = Instant::now();
            let pending_by_game: Vec<Vec<_>> = self
                .states
                .par_iter()
                .zip(self.searches.par_iter_mut())
                .zip(completed.par_iter_mut())
                .zip(root_indexes.par_iter())
                .enumerate()
                .map(|(game_index, (((state, search), comp), root_index))| {
                    let Some(root_index) = *root_index else {
                        return Vec::new();
                    };
                    let batch_target =
                        (search.config.simulations - *comp).min(leaf_batch_size as u32) as usize;
                    let mut local_pending = Vec::with_capacity(batch_target);

                    for _ in 0..batch_target {
                        if *comp + local_pending.len() as u32 >= search.config.simulations {
                            break;
                        }
                        let mut simulation_state = state.clone();
                        match search.select_eval_leaf(root_index, &mut simulation_state) {
                            PendingSimulation::NeedsEvaluation {
                                path,
                                state: leaf_state,
                            } => {
                                reserve_path(&mut search.nodes, &path);
                                local_pending.push(PendingGameLeaf {
                                    game_index,
                                    path,
                                    state: leaf_state,
                                });
                            }
                            PendingSimulation::Terminal {
                                path,
                                last_edge_value,
                            } => {
                                backup_path_from_last_edge(
                                    &mut search.nodes,
                                    &path,
                                    last_edge_value,
                                );
                                *comp += 1;
                            }
                            PendingSimulation::RootTerminal => {
                                *comp += 1;
                            }
                        }
                    }
                    local_pending
                })
                .collect();
            let selected_games = pending_by_game
                .iter()
                .filter(|leaves| !leaves.is_empty())
                .count();
            let select_elapsed = select_start.elapsed();
            let flatten_start = Instant::now();
            let pending_leaves = pending_by_game.into_iter().flatten().collect::<Vec<_>>();
            let flatten_elapsed = flatten_start.elapsed();

            if pending_leaves.is_empty() {
                profile.empty_wave(wave, selected_games, select_elapsed, flatten_elapsed);
                continue;
            }

            let request_start = Instant::now();
            let request_states = pending_leaves
                .iter()
                .map(|leaf| leaf.state.clone())
                .collect::<Vec<_>>();
            let request = EvalRequest::new_with_precomputed_bytes(request_states);
            let request_elapsed = request_start.elapsed();
            let eval_start = Instant::now();
            let response = evaluator.call1((request,))?;
            let eval_elapsed = eval_start.elapsed();
            let parse_start = Instant::now();
            let eval = parse_eval_response(&response)?;
            eval.validate_len(pending_leaves.len())?;
            let parse_elapsed = parse_start.elapsed();

            let backup_start = Instant::now();
            let leaf_count = pending_leaves.len();
            backup_pending_game_evaluations(
                &mut self.searches,
                &mut completed,
                pending_leaves,
                eval,
            );
            let backup_elapsed = backup_start.elapsed();
            profile.wave(MctsWaveProfile {
                wave,
                active_games: active_indexes.len(),
                selected_games,
                leaves: leaf_count,
                select_elapsed,
                flatten_elapsed,
                request_elapsed,
                eval_elapsed,
                parse_elapsed,
                backup_elapsed,
            });
        }

        let mut results = vec![None; self.states.len()];
        for game_index in active_indexes {
            let root_index = root_indexes[game_index]
                .expect("active game root must be initialized before result export");
            let root = &self.searches[game_index].nodes[root_index];
            results[game_index] = Some(MctsResult {
                selected_action: root.most_visited_action(),
                visit_counts: root.visit_counts(),
            });
        }
        Ok(results)
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

    pub fn set_simulations(&mut self, simulations: Vec<Option<u32>>) -> PyResult<()> {
        if simulations.len() != self.searches.len() {
            return Err(PyValueError::new_err(format!(
                "expected {} simulation slots, got {}",
                self.searches.len(),
                simulations.len()
            )));
        }
        for (search, simulations) in self.searches.iter_mut().zip(simulations.into_iter()) {
            let Some(simulations) = simulations else {
                continue;
            };
            search.set_simulations(simulations)?;
        }
        Ok(())
    }
}

impl MctsSelfPlayBatch {
    #[must_use]
    pub fn new(game_count: usize, config: MctsConfig) -> Self {
        Self {
            states: vec![GameState::new(); game_count],
            searches: vec![MctsSearch::new(config); game_count],
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

impl MctsSearch {
    #[must_use]
    pub const fn new(config: MctsConfig) -> Self {
        Self {
            config,
            nodes: Vec::new(),
        }
    }

    pub fn run(&mut self, state: &GameState) -> MctsResult {
        self.nodes.clear();
        let root_index = self.expand_node(state);
        self.run_simulations_from_root(state, root_index)
    }

    pub fn run_with_root_priors(
        &mut self,
        state: &GameState,
        priors: &[f32; ACTION_SPACE],
    ) -> MctsResult {
        self.nodes.clear();
        let root_index = self.expand_node_with_priors(state, priors);
        self.run_simulations_from_root(state, root_index)
    }

    pub fn run_with_leaf_evaluator<F>(
        &mut self,
        state: &GameState,
        leaf_batch_size: usize,
        mut evaluator: F,
    ) -> PyResult<MctsResult>
    where
        F: FnMut(EvalRequest) -> PyResult<EvalBatch>,
    {
        self.nodes.clear();
        if state.outcome_value().is_some() {
            return Ok(MctsResult {
                selected_action: None,
                visit_counts: [0; ACTION_SPACE],
            });
        }

        let root_eval = evaluator(EvalRequest::new(vec![state.clone()]))?;
        let root_policy = root_eval.single_policy()?;
        let root_index = self.expand_node_with_priors(state, root_policy);

        let mut completed = 0;
        while completed < self.config.simulations {
            let batch_target =
                (self.config.simulations - completed).min(leaf_batch_size as u32) as usize;
            let mut pending = Vec::with_capacity(batch_target);

            for _ in 0..batch_target {
                let mut simulation_state = state.clone();
                match self.select_eval_leaf(root_index, &mut simulation_state) {
                    PendingSimulation::NeedsEvaluation { path, state } => {
                        reserve_path(&mut self.nodes, &path);
                        pending.push(PendingLeaf { path, state });
                    }
                    PendingSimulation::Terminal {
                        path,
                        last_edge_value,
                    } => {
                        backup_path_from_last_edge(&mut self.nodes, &path, last_edge_value);
                        completed += 1;
                    }
                    PendingSimulation::RootTerminal => {
                        completed += 1;
                    }
                }
            }

            if pending.is_empty() {
                continue;
            }

            let request_states = pending
                .iter()
                .map(|leaf| leaf.state.clone())
                .collect::<Vec<_>>();
            let eval = evaluator(EvalRequest::new_with_precomputed_bytes(request_states))?;
            eval.validate_len(pending.len())?;
            for (leaf, (policy, value)) in pending
                .into_iter()
                .zip(eval.policies.into_iter().zip(eval.values.into_iter()))
            {
                unreserve_path(&mut self.nodes, &leaf.path);
                let child_index = self.expand_node_with_priors(&leaf.state, &policy);
                if let Some((parent_index, edge_index)) = leaf.path.last().copied() {
                    self.nodes[parent_index].edges[edge_index].child = Some(child_index);
                }
                backup_path_from_leaf_value(&mut self.nodes, &leaf.path, value);
                completed += 1;
            }
        }

        let root = &self.nodes[root_index];
        Ok(MctsResult {
            selected_action: root.most_visited_action(),
            visit_counts: root.visit_counts(),
        })
    }

    pub fn run_with_root_priors_and_leaf_evaluator<F>(
        &mut self,
        state: &GameState,
        priors: &[f32; ACTION_SPACE],
        leaf_batch_size: usize,
        evaluator: F,
    ) -> PyResult<MctsResult>
    where
        F: FnMut(EvalRequest) -> PyResult<EvalBatch>,
    {
        self.nodes.clear();
        if state.outcome_value().is_some() {
            return Ok(MctsResult {
                selected_action: None,
                visit_counts: [0; ACTION_SPACE],
            });
        }

        let root_index = self.expand_node_with_priors(state, priors);
        self.run_simulations_with_leaf_evaluator(state, root_index, leaf_batch_size, evaluator)
    }

    fn run_simulations_with_leaf_evaluator<F>(
        &mut self,
        state: &GameState,
        root_index: usize,
        leaf_batch_size: usize,
        mut evaluator: F,
    ) -> PyResult<MctsResult>
    where
        F: FnMut(EvalRequest) -> PyResult<EvalBatch>,
    {
        if leaf_batch_size == 0 {
            return Err(PyValueError::new_err("leaf_batch_size must be positive"));
        }

        let mut completed = 0;
        while completed < self.config.simulations {
            let batch_target =
                (self.config.simulations - completed).min(leaf_batch_size as u32) as usize;
            let mut pending = Vec::with_capacity(batch_target);

            for _ in 0..batch_target {
                let mut simulation_state = state.clone();
                match self.select_eval_leaf(root_index, &mut simulation_state) {
                    PendingSimulation::NeedsEvaluation { path, state } => {
                        reserve_path(&mut self.nodes, &path);
                        pending.push(PendingLeaf { path, state });
                    }
                    PendingSimulation::Terminal {
                        path,
                        last_edge_value,
                    } => {
                        backup_path_from_last_edge(&mut self.nodes, &path, last_edge_value);
                        completed += 1;
                    }
                    PendingSimulation::RootTerminal => {
                        completed += 1;
                    }
                }
            }

            if pending.is_empty() {
                continue;
            }

            let request_states = pending
                .iter()
                .map(|leaf| leaf.state.clone())
                .collect::<Vec<_>>();
            let eval = evaluator(EvalRequest::new_with_precomputed_bytes(request_states))?;
            eval.validate_len(pending.len())?;
            for (leaf, (policy, value)) in pending
                .into_iter()
                .zip(eval.policies.into_iter().zip(eval.values.into_iter()))
            {
                unreserve_path(&mut self.nodes, &leaf.path);
                let child_index = self.expand_node_with_priors(&leaf.state, &policy);
                if let Some((parent_index, edge_index)) = leaf.path.last().copied() {
                    self.nodes[parent_index].edges[edge_index].child = Some(child_index);
                }
                backup_path_from_leaf_value(&mut self.nodes, &leaf.path, value);
                completed += 1;
            }
        }

        let root = &self.nodes[root_index];
        Ok(MctsResult {
            selected_action: root.most_visited_action(),
            visit_counts: root.visit_counts(),
        })
    }

    fn run_simulations_from_root(&mut self, state: &GameState, root_index: usize) -> MctsResult {
        for _ in 0..self.config.simulations {
            let mut simulation_state = state.clone();
            self.simulate(root_index, &mut simulation_state);
        }

        let root = &self.nodes[root_index];
        MctsResult {
            selected_action: root.most_visited_action(),
            visit_counts: root.visit_counts(),
        }
    }

    fn simulate(&mut self, node_index: usize, state: &mut GameState) -> f32 {
        if let Some(outcome) = state.outcome_value() {
            return value_for_player(outcome, self.nodes[node_index].to_play);
        }

        if self.nodes[node_index].edges.is_empty() {
            return 0.0;
        }

        let edge_index = self.select_edge_index(node_index);
        let player = self.nodes[node_index].to_play;
        let action = self.nodes[node_index].edges[edge_index].action;
        let outcome = state
            .apply(action)
            .expect("MCTS selected an action from legal_action_indexes");

        let value = if let Some(outcome) = outcome {
            value_for_player(outcome, player)
        } else {
            let child_index = match self.nodes[node_index].edges[edge_index].child {
                Some(child_index) => child_index,
                None => {
                    let child_index = self.expand_node(state);
                    self.nodes[node_index].edges[edge_index].child = Some(child_index);
                    child_index
                }
            };
            -self.simulate(child_index, state)
        };

        self.nodes[node_index].visit_count += 1;
        self.nodes[node_index].edges[edge_index].update(value);
        value
    }

    fn expand_node(&mut self, state: &GameState) -> usize {
        let node_index = self.nodes.len();
        self.nodes.push(Node::expanded_with_uniform_priors(state));
        node_index
    }

    fn expand_node_with_priors(
        &mut self,
        state: &GameState,
        priors: &[f32; ACTION_SPACE],
    ) -> usize {
        let node_index = self.nodes.len();
        self.nodes.push(Node::expanded_with_priors(state, priors));
        node_index
    }

    fn select_edge_index(&self, node_index: usize) -> usize {
        let node = &self.nodes[node_index];
        node.edges
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| {
                let parent_visits = node.visit_count + node.virtual_visit_count;
                let left_score = puct_score(parent_visits, left, self.config.c_puct);
                let right_score = puct_score(parent_visits, right, self.config.c_puct);
                left_score
                    .partial_cmp(&right_score)
                    .expect("PUCT score must be finite")
            })
            .map(|(index, _)| index)
            .expect("expanded MCTS node must contain at least one edge")
    }

    fn select_eval_leaf(&self, root_index: usize, state: &mut GameState) -> PendingSimulation {
        if state.outcome_value().is_some() {
            return PendingSimulation::RootTerminal;
        }

        let mut node_index = root_index;
        let mut path = Vec::new();
        loop {
            if self.nodes[node_index].edges.is_empty() {
                return PendingSimulation::NeedsEvaluation {
                    path,
                    state: state.clone(),
                };
            }

            let edge_index = self.select_edge_index(node_index);
            let player = self.nodes[node_index].to_play;
            let action = self.nodes[node_index].edges[edge_index].action;
            let outcome = state
                .apply(action)
                .expect("MCTS selected an action from legal_action_indexes");
            path.push((node_index, edge_index));

            if let Some(outcome) = outcome {
                return PendingSimulation::Terminal {
                    path,
                    last_edge_value: value_for_player(outcome, player),
                };
            }

            match self.nodes[node_index].edges[edge_index].child {
                Some(child_index) => node_index = child_index,
                None => {
                    return PendingSimulation::NeedsEvaluation {
                        path,
                        state: state.clone(),
                    };
                }
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct EvalBatch {
    policies: Vec<[f32; ACTION_SPACE]>,
    values: Vec<f32>,
}

impl EvalBatch {
    #[must_use]
    pub fn new(policies: Vec<[f32; ACTION_SPACE]>, values: Vec<f32>) -> Self {
        Self { policies, values }
    }

    fn single_policy(&self) -> PyResult<&[f32; ACTION_SPACE]> {
        self.validate_len(1)?;
        Ok(&self.policies[0])
    }

    fn validate_len(&self, expected: usize) -> PyResult<()> {
        if self.policies.len() != expected || self.values.len() != expected {
            return Err(PyValueError::new_err(format!(
                "expected {expected} policy/value rows, got {}/{}",
                self.policies.len(),
                self.values.len()
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct PendingLeaf {
    path: Vec<(usize, usize)>,
    state: GameState,
}

#[derive(Clone, Debug)]
struct PendingGameLeaf {
    game_index: usize,
    path: Vec<(usize, usize)>,
    state: GameState,
}

struct PendingGameEvaluation {
    path: Vec<(usize, usize)>,
    state: GameState,
    policy: [f32; ACTION_SPACE],
    value: f32,
}

#[derive(Clone, Debug)]
enum PendingSimulation {
    NeedsEvaluation {
        path: Vec<(usize, usize)>,
        state: GameState,
    },
    Terminal {
        path: Vec<(usize, usize)>,
        last_edge_value: f32,
    },
    RootTerminal,
}

#[derive(Clone, Debug)]
pub(crate) struct EdgeStats {
    action: Action,
    prior: f32,
    visit_count: u32,
    value_sum: f32,
    virtual_visit_count: u32,
    virtual_value_sum: f32,
    child: Option<usize>,
}

impl EdgeStats {
    fn new(action: Action, prior: f32) -> Self {
        Self {
            action,
            prior,
            visit_count: 0,
            value_sum: 0.0,
            virtual_visit_count: 0,
            virtual_value_sum: 0.0,
            child: None,
        }
    }

    fn update(&mut self, value: f32) {
        self.visit_count += 1;
        self.value_sum += value;
    }

    fn mean_value(&self) -> f32 {
        let total_visits = self.visit_count + self.virtual_visit_count;
        if total_visits == 0 {
            0.0
        } else {
            (self.value_sum + self.virtual_value_sum) / total_visits as f32
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Node {
    to_play: Player,
    visit_count: u32,
    virtual_visit_count: u32,
    edges: Vec<EdgeStats>,
}

impl Node {
    fn expanded_with_uniform_priors(state: &GameState) -> Self {
        let legal_actions = state.legal_action_indexes();
        let prior = if legal_actions.is_empty() {
            0.0
        } else {
            1.0 / legal_actions.len() as f32
        };
        let edges = legal_actions
            .into_iter()
            .filter_map(Action::from_index)
            .map(|action| EdgeStats::new(action, prior))
            .collect();

        Self {
            to_play: state.current_player_value(),
            visit_count: 0,
            virtual_visit_count: 0,
            edges,
        }
    }

    fn expanded_with_priors(state: &GameState, priors: &[f32; ACTION_SPACE]) -> Self {
        let legal_actions = state.legal_action_indexes();
        let legal_prior_sum = legal_actions
            .iter()
            .map(|action| priors[*action])
            .sum::<f32>();
        let fallback_prior = if legal_actions.is_empty() {
            0.0
        } else {
            1.0 / legal_actions.len() as f32
        };
        let edges = legal_actions
            .into_iter()
            .filter_map(Action::from_index)
            .map(|action| {
                let action_index = action.to_index();
                let prior = if legal_prior_sum > 0.0 {
                    priors[action_index] / legal_prior_sum
                } else {
                    fallback_prior
                };
                EdgeStats::new(action, prior)
            })
            .collect();

        Self {
            to_play: state.current_player_value(),
            visit_count: 0,
            virtual_visit_count: 0,
            edges,
        }
    }

    fn visit_counts(&self) -> [u32; ACTION_SPACE] {
        let mut counts = [0; ACTION_SPACE];
        for edge in &self.edges {
            counts[edge.action.to_index()] = edge.visit_count;
        }
        counts
    }

    fn most_visited_action(&self) -> Option<usize> {
        self.edges
            .iter()
            .max_by(|left, right| {
                left.visit_count
                    .cmp(&right.visit_count)
                    .then_with(|| {
                        left.mean_value()
                            .partial_cmp(&right.mean_value())
                            .expect("edge mean value must be finite")
                    })
                    .then_with(|| {
                        left.prior
                            .partial_cmp(&right.prior)
                            .expect("edge prior must be finite")
                    })
                    .then_with(|| right.action.to_index().cmp(&left.action.to_index()))
            })
            .map(|edge| edge.action.to_index())
    }
}

impl GameState {
    #[must_use]
    pub(crate) const fn current_player_value(&self) -> Player {
        self.current_player
    }

    #[must_use]
    pub(crate) const fn outcome_value(&self) -> Option<GameOutcome> {
        self.outcome
    }
}

fn puct_score(parent_visits: u32, edge: &EdgeStats, c_puct: f32) -> f32 {
    let exploration_visits = parent_visits.max(1) as f32;
    let edge_visits = edge.visit_count + edge.virtual_visit_count;
    edge.mean_value() + c_puct * edge.prior * exploration_visits.sqrt() / (1.0 + edge_visits as f32)
}

fn value_for_player(outcome: GameOutcome, player: Player) -> f32 {
    if outcome.winner == player { 1.0 } else { -1.0 }
}

fn reserve_path(nodes: &mut [Node], path: &[(usize, usize)]) {
    for (node_index, edge_index) in path.iter().copied() {
        nodes[node_index].virtual_visit_count += 1;
        nodes[node_index].edges[edge_index].virtual_visit_count += 1;
        nodes[node_index].edges[edge_index].virtual_value_sum -= 1.0;
    }
}

fn unreserve_path(nodes: &mut [Node], path: &[(usize, usize)]) {
    for (node_index, edge_index) in path.iter().copied() {
        nodes[node_index].virtual_visit_count = nodes[node_index]
            .virtual_visit_count
            .checked_sub(1)
            .expect("virtual node visit count underflow");
        nodes[node_index].edges[edge_index].virtual_visit_count = nodes[node_index].edges
            [edge_index]
            .virtual_visit_count
            .checked_sub(1)
            .expect("virtual edge visit count underflow");
        nodes[node_index].edges[edge_index].virtual_value_sum += 1.0;
    }
}

fn backup_path_from_leaf_value(nodes: &mut [Node], path: &[(usize, usize)], leaf_value: f32) {
    let mut value = leaf_value;
    for (node_index, edge_index) in path.iter().rev().copied() {
        value = -value;
        nodes[node_index].visit_count += 1;
        nodes[node_index].edges[edge_index].update(value);
    }
}

fn backup_path_from_last_edge(nodes: &mut [Node], path: &[(usize, usize)], last_edge_value: f32) {
    let mut value = last_edge_value;
    for (node_index, edge_index) in path.iter().rev().copied() {
        nodes[node_index].visit_count += 1;
        nodes[node_index].edges[edge_index].update(value);
        value = -value;
    }
}

fn backup_pending_game_evaluations(
    searches: &mut [MctsSearch],
    completed: &mut [u32],
    pending_leaves: Vec<PendingGameLeaf>,
    eval: EvalBatch,
) {
    let mut by_game = (0..searches.len()).map(|_| Vec::new()).collect::<Vec<_>>();
    for (leaf, (policy, value)) in pending_leaves
        .into_iter()
        .zip(eval.policies.into_iter().zip(eval.values.into_iter()))
    {
        by_game[leaf.game_index].push(PendingGameEvaluation {
            path: leaf.path,
            state: leaf.state,
            policy,
            value,
        });
    }

    searches
        .par_iter_mut()
        .zip(completed.par_iter_mut())
        .zip(by_game.into_par_iter())
        .for_each(|((search, comp), evaluations)| {
            let completed_count = evaluations.len() as u32;
            for evaluation in evaluations {
                unreserve_path(&mut search.nodes, &evaluation.path);
                let child_index =
                    search.expand_node_with_priors(&evaluation.state, &evaluation.policy);
                if let Some((parent_index, edge_index)) = evaluation.path.last().copied() {
                    search.nodes[parent_index].edges[edge_index].child = Some(child_index);
                }
                backup_path_from_leaf_value(&mut search.nodes, &evaluation.path, evaluation.value);
            }
            *comp += completed_count;
        });
}

#[derive(Clone, Copy)]
struct MctsProfile {
    enabled: bool,
    interval: u64,
    name: &'static str,
}

struct MctsWaveProfile {
    wave: u64,
    active_games: usize,
    selected_games: usize,
    leaves: usize,
    select_elapsed: std::time::Duration,
    flatten_elapsed: std::time::Duration,
    request_elapsed: std::time::Duration,
    eval_elapsed: std::time::Duration,
    parse_elapsed: std::time::Duration,
    backup_elapsed: std::time::Duration,
}

impl MctsProfile {
    fn new(name: &'static str) -> Self {
        Self {
            enabled: env_flag("GKA_MCTS_PROFILE"),
            interval: env::var("GKA_MCTS_PROFILE_INTERVAL")
                .ok()
                .and_then(|value| value.parse::<u64>().ok())
                .filter(|value| *value > 0)
                .unwrap_or(1),
            name,
        }
    }

    fn root(&self, active_games: usize, elapsed: std::time::Duration) {
        if !self.enabled {
            return;
        }
        eprintln!(
            "[gka-mcts-profile] fn={} root active_games={} rayon_threads={} root_expand={:.3}s",
            self.name,
            active_games,
            rayon::current_num_threads(),
            elapsed.as_secs_f64(),
        );
    }

    fn root_eval(
        &self,
        active_games: usize,
        request_elapsed: std::time::Duration,
        eval_elapsed: std::time::Duration,
        parse_elapsed: std::time::Duration,
        expand_elapsed: std::time::Duration,
    ) {
        if !self.enabled {
            return;
        }
        eprintln!(
            "[gka-mcts-profile] fn={} root_eval active_games={} rayon_threads={} request={:.3}s eval_call={:.3}s parse={:.3}s expand={:.3}s",
            self.name,
            active_games,
            rayon::current_num_threads(),
            request_elapsed.as_secs_f64(),
            eval_elapsed.as_secs_f64(),
            parse_elapsed.as_secs_f64(),
            expand_elapsed.as_secs_f64(),
        );
    }

    fn empty_wave(
        &self,
        wave: u64,
        selected_games: usize,
        select_elapsed: std::time::Duration,
        flatten_elapsed: std::time::Duration,
    ) {
        if !self.enabled || wave % self.interval != 0 {
            return;
        }
        eprintln!(
            "[gka-mcts-profile] fn={} wave={} empty selected_games={} select={:.3}s flatten={:.3}s",
            self.name,
            wave,
            selected_games,
            select_elapsed.as_secs_f64(),
            flatten_elapsed.as_secs_f64(),
        );
    }

    fn wave(&self, profile: MctsWaveProfile) {
        if !self.enabled || profile.wave % self.interval != 0 {
            return;
        }
        let total = profile.select_elapsed
            + profile.flatten_elapsed
            + profile.request_elapsed
            + profile.eval_elapsed
            + profile.parse_elapsed
            + profile.backup_elapsed;
        eprintln!(
            "[gka-mcts-profile] fn={} wave={} active_games={} selected_games={} leaves={} select={:.3}s flatten={:.3}s request={:.3}s eval_call={:.3}s parse={:.3}s backup={:.3}s total={:.3}s",
            self.name,
            profile.wave,
            profile.active_games,
            profile.selected_games,
            profile.leaves,
            profile.select_elapsed.as_secs_f64(),
            profile.flatten_elapsed.as_secs_f64(),
            profile.request_elapsed.as_secs_f64(),
            profile.eval_elapsed.as_secs_f64(),
            profile.parse_elapsed.as_secs_f64(),
            profile.backup_elapsed.as_secs_f64(),
            total.as_secs_f64(),
        );
    }
}

fn env_flag(name: &str) -> bool {
    !matches!(
        env::var(name).as_deref(),
        Err(_) | Ok("") | Ok("0") | Ok("false") | Ok("False") | Ok("no") | Ok("No")
    )
}

fn f32_slice_as_bytes(values: &[f32]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

fn parse_eval_response(response: &Bound<'_, PyAny>) -> PyResult<EvalBatch> {
    if let Ok((policy_obj, value_obj)) = response.extract::<(Bound<'_, PyAny>, Bound<'_, PyAny>)>()
    {
        if let Ok(eval) = parse_eval_response_buffers(&policy_obj, &value_obj) {
            return Ok(eval);
        }
    }

    let (policy_rows, values): (Vec<Vec<f32>>, Vec<f32>) = response.extract()?;
    let mut policies = Vec::with_capacity(policy_rows.len());
    for (row_index, row) in policy_rows.into_iter().enumerate() {
        policies.push(parse_policy_row(row, row_index)?);
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err("values must be finite"));
    }
    Ok(EvalBatch::new(policies, values))
}

fn parse_eval_response_buffers(
    policy_obj: &Bound<'_, PyAny>,
    value_obj: &Bound<'_, PyAny>,
) -> PyResult<EvalBatch> {
    let py = policy_obj.py();
    let policy_buffer = PyBuffer::<f32>::get(policy_obj)?;
    let value_buffer = PyBuffer::<f32>::get(value_obj)?;
    if !policy_buffer.is_c_contiguous() || !value_buffer.is_c_contiguous() {
        return Err(PyValueError::new_err(
            "policy/value buffers must be C-contiguous float32 arrays",
        ));
    }
    let policy_count = policy_buffer.item_count();
    if policy_count % ACTION_SPACE != 0 {
        return Err(PyValueError::new_err(format!(
            "policy buffer length must be divisible by {ACTION_SPACE}, got {policy_count}",
        )));
    }
    let batch_size = policy_count / ACTION_SPACE;
    if value_buffer.item_count() != batch_size {
        return Err(PyValueError::new_err(format!(
            "expected {batch_size} values, got {}",
            value_buffer.item_count()
        )));
    }

    let policy_values = policy_buffer.to_vec(py)?;
    let values = value_buffer.to_vec(py)?;
    let mut policies = Vec::with_capacity(batch_size);
    for (row_index, row) in policy_values.chunks_exact(ACTION_SPACE).enumerate() {
        if row.iter().any(|prior| !prior.is_finite() || *prior < 0.0) {
            return Err(PyValueError::new_err(format!(
                "policy row {row_index} contains invalid priors"
            )));
        }
        let mut policy = [0.0; ACTION_SPACE];
        policy.copy_from_slice(row);
        policies.push(policy);
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err("values must be finite"));
    }
    Ok(EvalBatch::new(policies, values))
}

fn parse_policy_row(row: Vec<f32>, row_index: usize) -> PyResult<[f32; ACTION_SPACE]> {
    if row.len() != ACTION_SPACE {
        return Err(PyValueError::new_err(format!(
            "policy row {row_index} must have length {ACTION_SPACE}, got {}",
            row.len()
        )));
    }
    if row.iter().any(|prior| !prior.is_finite() || *prior < 0.0) {
        return Err(PyValueError::new_err(
            "policy priors must be finite non-negative values",
        ));
    }
    let mut policy = [0.0; ACTION_SPACE];
    policy.copy_from_slice(&row);
    Ok(policy)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::{BOARD_CELLS, PASS_ACTION};
    use pretty_assertions::assert_eq;

    #[test]
    fn mcts_config_has_local_smoke_test_defaults() {
        let config = MctsConfig::default();

        assert_eq!(config.simulations, 50);
        assert!((config.c_puct - 1.5).abs() < f32::EPSILON);
        assert_eq!(MctsConfig::new(8, 2.0).simulations, 8);
    }

    #[test]
    fn mcts_search_can_update_simulation_budget() {
        let mut search = MctsSearch::new(MctsConfig::new(8, 1.5));

        search.set_simulations(3).unwrap();
        let result = search.run(&GameState::new());

        assert_eq!(search.simulations(), 3);
        assert_eq!(result.visit_counts.iter().sum::<u32>(), 3);
        assert!(search.set_simulations(0).is_err());
    }

    #[test]
    fn mcts_python_constructor_rejects_zero_simulation_budget() {
        assert!(MctsSearch::py_new(0, 1.5).is_err());
    }

    #[test]
    fn edge_stats_tracks_visits_and_mean_value() {
        let mut edge = EdgeStats::new(Action::Pass, 0.25);

        assert_eq!(edge.visit_count, 0);
        assert_eq!(edge.mean_value(), 0.0);

        edge.update(1.0);
        edge.update(-0.5);

        assert_eq!(edge.visit_count, 2);
        assert!((edge.mean_value() - 0.25).abs() < f32::EPSILON);
    }

    #[test]
    fn node_expands_legal_actions_with_uniform_priors() {
        let state = GameState::new();
        let node = Node::expanded_with_uniform_priors(&state);

        assert_eq!(node.to_play, Player::Blue);
        assert_eq!(node.visit_count, 0);
        assert_eq!(node.edges.len(), BOARD_CELLS);
        assert!(node.edges.iter().any(|edge| edge.action == Action::Pass));
        assert!(
            !node
                .edges
                .iter()
                .any(|edge| edge.action.to_index() == crate::game::CENTER_INDEX)
        );
        assert!(
            node.edges
                .iter()
                .all(|edge| { (edge.prior - (1.0 / BOARD_CELLS as f32)).abs() < f32::EPSILON })
        );

        let counts = node.visit_counts();
        assert_eq!(counts[PASS_ACTION], 0);
    }

    #[test]
    fn node_expands_model_priors_only_for_legal_actions() {
        let state = GameState::new();
        let mut priors = [0.0; ACTION_SPACE];
        priors[crate::game::CENTER_INDEX] = 100.0;
        priors[0] = 2.0;
        priors[PASS_ACTION] = 1.0;

        let node = Node::expanded_with_priors(&state, &priors);

        assert!(
            !node
                .edges
                .iter()
                .any(|edge| edge.action.to_index() == crate::game::CENTER_INDEX)
        );
        let action_zero = node
            .edges
            .iter()
            .find(|edge| edge.action.to_index() == 0)
            .unwrap();
        let pass = node
            .edges
            .iter()
            .find(|edge| edge.action == Action::Pass)
            .unwrap();
        assert!((action_zero.prior - (2.0 / 3.0)).abs() < f32::EPSILON);
        assert!((pass.prior - (1.0 / 3.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn node_expands_zero_model_priors_with_uniform_legal_fallback() {
        let state = GameState::new();
        let priors = [0.0; ACTION_SPACE];

        let node = Node::expanded_with_priors(&state, &priors);

        assert!(
            node.edges
                .iter()
                .all(|edge| { (edge.prior - (1.0 / BOARD_CELLS as f32)).abs() < f32::EPSILON })
        );
    }

    #[test]
    fn most_visited_action_breaks_ties_without_defaulting_to_pass() {
        let node = Node {
            to_play: Player::Blue,
            visit_count: 2,
            virtual_visit_count: 0,
            edges: vec![
                EdgeStats {
                    action: Action::from_index(5).unwrap(),
                    prior: 0.5,
                    visit_count: 1,
                    value_sum: 0.0,
                    virtual_visit_count: 0,
                    virtual_value_sum: 0.0,
                    child: None,
                },
                EdgeStats {
                    action: Action::Pass,
                    prior: 0.5,
                    visit_count: 1,
                    value_sum: 0.0,
                    virtual_visit_count: 0,
                    virtual_value_sum: 0.0,
                    child: None,
                },
            ],
        };

        assert_eq!(node.most_visited_action(), Some(5));
    }

    #[test]
    fn puct_selection_reflects_prior_and_visit_count() {
        let config = MctsConfig::new(1, 2.0);
        let search = MctsSearch::new(config);
        let node = Node {
            to_play: Player::Blue,
            visit_count: 16,
            virtual_visit_count: 0,
            edges: vec![
                EdgeStats {
                    action: Action::from_index(0).unwrap(),
                    prior: 0.9,
                    visit_count: 0,
                    value_sum: 0.0,
                    virtual_visit_count: 0,
                    virtual_value_sum: 0.0,
                    child: None,
                },
                EdgeStats {
                    action: Action::Pass,
                    prior: 0.1,
                    visit_count: 10,
                    value_sum: 8.0,
                    virtual_visit_count: 0,
                    virtual_value_sum: 0.0,
                    child: None,
                },
            ],
        };
        let mut search = search;
        search.nodes.push(node);

        assert_eq!(search.select_edge_index(0), 0);
    }

    #[test]
    fn mcts_search_returns_only_legal_actions_and_visit_distribution() {
        let state = GameState::new();
        let legal_actions = state.legal_action_indexes();
        let mut search = MctsSearch::new(MctsConfig::new(8, 1.5));

        let result = search.run(&state);

        assert!(legal_actions.contains(&result.selected_action.unwrap()));
        assert_eq!(result.visit_counts.iter().sum::<u32>(), 8);
        assert_eq!(result.visit_counts[crate::game::CENTER_INDEX], 0);
    }

    #[test]
    fn mcts_search_with_priors_masks_illegal_root_actions() {
        let state = GameState::new();
        let mut priors = vec![0.0; ACTION_SPACE];
        priors[crate::game::CENTER_INDEX] = 100.0;
        priors[0] = 1.0;
        let mut search = MctsSearch::new(MctsConfig::new(4, 1.5));

        let result = search.search_with_priors(&state, priors).unwrap();

        assert_eq!(result.visit_counts[crate::game::CENTER_INDEX], 0);
        assert!(
            state
                .legal_action_indexes()
                .contains(&result.selected_action.unwrap())
        );
    }

    #[test]
    #[ignore = "debug helper for inspecting root stats in a consecutive-pass loss position"]
    fn debug_consecutive_pass_loss_root_stats() {
        let mut state = GameState::new();
        state.apply(Action::from_index(15).unwrap()).unwrap();
        state.apply(Action::Pass).unwrap();
        assert_eq!(state.current_player_value(), Player::Blue);

        for (label, pass_prior) in [("uniform", 1.0_f32), ("pass10", 10.0), ("pass100", 100.0)] {
            let mut priors = [1.0_f32; ACTION_SPACE];
            priors[crate::game::CENTER_INDEX] = 0.0;
            priors[PASS_ACTION] = pass_prior;
            let mut search = MctsSearch::new(MctsConfig::new(200, 1.5));

            let result = search.run_with_root_priors(&state, &priors);
            let root = &search.nodes[0];
            let mut rows = root
                .edges
                .iter()
                .map(|edge| {
                    (
                        edge.action.to_index(),
                        edge.prior,
                        edge.visit_count,
                        edge.value_sum,
                        edge.mean_value(),
                    )
                })
                .collect::<Vec<_>>();
            rows.sort_by(|left, right| {
                right
                    .2
                    .cmp(&left.2)
                    .then_with(|| {
                        right
                            .4
                            .partial_cmp(&left.4)
                            .expect("mean value must be finite")
                    })
                    .then_with(|| left.0.cmp(&right.0))
            });
            let pass = rows
                .iter()
                .find(|(action, _, _, _, _)| *action == PASS_ACTION)
                .copied()
                .unwrap();
            println!(
                "\n[{label}] selected={:?} pass_prior_input={pass_prior} root_visits={}",
                result.selected_action, root.visit_count
            );
            println!(
                "pass: action={} prior={:.6} visits={} value_sum={:.3} mean={:.3}",
                pass.0, pass.1, pass.2, pass.3, pass.4
            );
            println!("top root edges:");
            for (rank, (action, prior, visits, value_sum, mean)) in rows.iter().take(12).enumerate() {
                println!(
                    "#{:02} action={:02} prior={:.6} visits={:03} value_sum={:.3} mean={:.3}",
                    rank + 1,
                    action,
                    prior,
                    visits,
                    value_sum,
                    mean
                );
            }
        }

        for (label, pass_prior) in [
            ("eval0_uniform", 1.0_f32),
            ("eval0_pass10", 10.0),
            ("eval0_pass100", 100.0),
        ] {
            let mut priors = [1.0_f32; ACTION_SPACE];
            priors[crate::game::CENTER_INDEX] = 0.0;
            priors[PASS_ACTION] = pass_prior;
            let mut search = MctsSearch::new(MctsConfig::new(200, 1.5));
            let result = search
                .run_with_root_priors_and_leaf_evaluator(&state, &priors, 16, |request| {
                    let policy = [1.0_f32; ACTION_SPACE];
                    Ok(EvalBatch::new(
                        vec![policy; request.len()],
                        vec![0.0; request.len()],
                    ))
                })
                .unwrap();
            let root = &search.nodes[0];
            let mut rows = root
                .edges
                .iter()
                .map(|edge| {
                    (
                        edge.action.to_index(),
                        edge.prior,
                        edge.visit_count,
                        edge.value_sum,
                        edge.mean_value(),
                    )
                })
                .collect::<Vec<_>>();
            rows.sort_by(|left, right| {
                right
                    .2
                    .cmp(&left.2)
                    .then_with(|| {
                        right
                            .4
                            .partial_cmp(&left.4)
                            .expect("mean value must be finite")
                    })
                    .then_with(|| left.0.cmp(&right.0))
            });
            let pass = rows
                .iter()
                .find(|(action, _, _, _, _)| *action == PASS_ACTION)
                .copied()
                .unwrap();
            println!(
                "\n[{label}] selected={:?} pass_prior_input={pass_prior} root_visits={}",
                result.selected_action, root.visit_count
            );
            println!(
                "pass: action={} prior={:.6} visits={} value_sum={:.3} mean={:.3}",
                pass.0, pass.1, pass.2, pass.3, pass.4
            );
            println!("top root edges:");
            for (rank, (action, prior, visits, value_sum, mean)) in rows.iter().take(12).enumerate() {
                println!(
                    "#{:02} action={:02} prior={:.6} visits={:03} value_sum={:.3} mean={:.3}",
                    rank + 1,
                    action,
                    prior,
                    visits,
                    value_sum,
                    mean
                );
            }
        }
    }

    #[test]
    fn terminal_win_is_backed_up_to_parent_edge() {
        let mut board = [crate::game::Cell::Empty; crate::game::BOARD_CELLS];
        board[crate::game::CENTER_INDEX] = crate::game::Cell::Neutral;
        board[1] = crate::game::Cell::Orange;
        board[2] = crate::game::Cell::Blue;
        board[10] = crate::game::Cell::Blue;
        let mut state = crate::game::state_with_board(board, Player::Blue);
        let mut search = MctsSearch::new(MctsConfig::new(1, 1.5));
        search.nodes.push(Node {
            to_play: Player::Blue,
            visit_count: 0,
            virtual_visit_count: 0,
            edges: vec![EdgeStats::new(Action::from_index(0).unwrap(), 1.0)],
        });

        let value = search.simulate(0, &mut state);

        assert_eq!(value, 1.0);
        assert_eq!(search.nodes[0].visit_count, 1);
        assert_eq!(search.nodes[0].visit_counts()[0], 1);
    }

    #[test]
    fn root_eval_request_returns_feature_and_mask_batch() {
        let state = GameState::new();
        let search = MctsSearch::new(MctsConfig::new(1, 1.5));
        let request = search.root_eval_request(&state);

        assert_eq!(request.len(), 1);
        assert!(!request.is_empty());
        assert_eq!(
            request.feature_planes()[0].len(),
            crate::game::FEATURE_CHANNELS * crate::game::BOARD_CELLS
        );
        assert_eq!(request.legal_masks()[0].len(), ACTION_SPACE);
        assert_eq!(request.legal_masks()[0][crate::game::PASS_ACTION], true);
    }

    #[test]
    fn mcts_search_with_leaf_evaluator_uses_policy_and_value_batches() {
        let state = GameState::new();
        let mut search = MctsSearch::new(MctsConfig::new(4, 1.5));
        let mut batch_sizes = Vec::new();

        let result = search
            .run_with_leaf_evaluator(&state, 2, |request| {
                batch_sizes.push(request.len());
                let mut policies = Vec::new();
                let mut values = Vec::new();
                for mask in request.legal_masks() {
                    let mut policy = [0.0; ACTION_SPACE];
                    for (index, is_legal) in mask.into_iter().enumerate() {
                        if is_legal {
                            policy[index] = if index == 0 { 10.0 } else { 1.0 };
                        }
                    }
                    policies.push(policy);
                    values.push(0.25);
                }
                Ok(EvalBatch::new(policies, values))
            })
            .unwrap();

        assert_eq!(batch_sizes[0], 1);
        assert!(batch_sizes.iter().skip(1).any(|size| *size > 0));
        assert_eq!(result.visit_counts.iter().sum::<u32>(), 4);
        assert!(
            state
                .legal_action_indexes()
                .contains(&result.selected_action.unwrap())
        );
        assert_eq!(result.visit_counts[crate::game::CENTER_INDEX], 0);
    }

    #[test]
    fn mcts_self_play_batch_exposes_active_eval_request_and_advances_turns() {
        let mut batch = MctsSelfPlayBatch::new(2, MctsConfig::new(2, 1.5));

        let request = batch.active_eval_request();
        assert_eq!(batch.len(), 2);
        assert_eq!(batch.active_count(), 2);
        assert_eq!(request.len(), 2);
        assert_eq!(request.legal_masks().len(), 2);

        let priors = request
            .legal_masks()
            .into_iter()
            .map(|mask| {
                mask.into_iter()
                    .map(|is_legal| if is_legal { 1.0 } else { 0.0 })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let results = batch.play_turns_with_priors(priors).unwrap();

        assert_eq!(results.len(), 2);
        assert!(results.iter().all(Option::is_some));
        assert_eq!(batch.current_players(), vec![Player::Orange as u8; 2]);
        assert_eq!(batch.active_count(), 2);
    }

    #[test]
    fn mcts_self_play_batch_rejects_mismatched_prior_rows() {
        let mut batch = MctsSelfPlayBatch::new(2, MctsConfig::new(2, 1.5));

        let err = batch.play_turns_with_priors(vec![vec![1.0; ACTION_SPACE]]);

        assert!(err.is_err());
    }
}
