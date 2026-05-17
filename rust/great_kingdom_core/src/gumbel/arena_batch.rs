use pyo3::{exceptions::PyRuntimeError, exceptions::PyValueError, prelude::*};
use rayon::prelude::*;

use super::{
    config::GumbelConfig,
    result::GumbelResult,
    root::{RootSearchState, empty_result, start_root_search},
    search::{
        GumbelEvalBatch, GumbelSearch, PendingGumbelSimulation, backup_path,
        parse_gumbel_eval_response, reserve_path, unreserve_path,
    },
};
use crate::{
    eval_request::EvalRequest,
    game::{ACTION_SPACE, Action, GameState},
    onnx::{NetworkOutput, OnnxEvaluator},
};

const BLUE: u8 = 1;
const ORANGE: u8 = 2;
type SearchActiveWithRootLogits = (Vec<Option<GumbelResult>>, Vec<Vec<f32>>);

#[pyclass]
#[derive(Clone, Debug)]
pub struct GumbelArenaBatch {
    states: Vec<GameState>,
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
        seed = 2026,
        gumbel_scale = 1.0,
        policy_target_temperature = 1.0,
        policy_target_c_visit = None,
        policy_target_c_scale = None,
        paired_seeds = false
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn py_new(
        game_count: usize,
        seed_start: u64,
        game_index_start: usize,
        simulations: u32,
        max_considered_actions: usize,
        c_visit: f32,
        c_scale: f32,
        seed: u64,
        gumbel_scale: f32,
        policy_target_temperature: f32,
        policy_target_c_visit: Option<f32>,
        policy_target_c_scale: Option<f32>,
        paired_seeds: bool,
    ) -> PyResult<Self> {
        if game_count == 0 {
            return Err(PyValueError::new_err("game_count must be positive"));
        }
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
        Ok(Self::new(
            game_count,
            seed_start,
            game_index_start,
            config,
            paired_seeds,
        ))
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
    pub fn active_legal_masks(&self) -> Vec<Vec<bool>> {
        self.active_indexes()
            .into_iter()
            .map(|index| self.states[index].legal_mask())
            .collect()
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

    #[pyo3(signature = (policy_logits, evaluator, root_values, leaf_batch_size = 16))]
    pub fn search_active_with_logits_and_evaluator(
        &mut self,
        policy_logits: Vec<Vec<f32>>,
        evaluator: &Bound<'_, PyAny>,
        root_values: Vec<f32>,
        leaf_batch_size: usize,
    ) -> PyResult<Vec<Option<GumbelResult>>> {
        if leaf_batch_size == 0 {
            return Err(PyValueError::new_err("leaf_batch_size must be positive"));
        }
        let mut evaluator = PythonArenaLeafEvaluator::new(evaluator);
        self.search_active_with_logits_evaluator(
            policy_logits,
            &root_values,
            leaf_batch_size,
            &mut evaluator,
        )
    }

    #[pyo3(signature = (candidate_evaluator, best_evaluator, leaf_batch_size = 16))]
    pub fn search_active_with_onnx_evaluators(
        &mut self,
        mut candidate_evaluator: PyRefMut<'_, OnnxEvaluator>,
        mut best_evaluator: PyRefMut<'_, OnnxEvaluator>,
        leaf_batch_size: usize,
    ) -> PyResult<SearchActiveWithRootLogits> {
        if leaf_batch_size == 0 {
            return Err(PyValueError::new_err("leaf_batch_size must be positive"));
        }
        let active_indexes = self.active_indexes();
        let active_states = active_indexes
            .iter()
            .map(|index| self.states[*index].clone())
            .collect::<Vec<_>>();
        let candidate_players = self.candidate_players.clone();
        let mut evaluator = OnnxArenaLeafEvaluator::new(
            &mut candidate_evaluator,
            &mut best_evaluator,
            candidate_players,
        );
        let root_eval = evaluator.evaluate(
            active_states,
            active_indexes.clone(),
            active_indexes.len(),
            0,
        )?;
        root_eval.validate_len(active_indexes.len())?;
        let root_policy_logits = root_eval
            .policies
            .iter()
            .map(|row| Vec::from(*row))
            .collect::<Vec<_>>();
        let root_rows = root_eval
            .policies
            .into_iter()
            .map(Vec::from)
            .collect::<Vec<_>>();
        let results = self.search_active_with_logits_evaluator(
            root_rows,
            &root_eval.values,
            leaf_batch_size,
            &mut evaluator,
        )?;
        Ok((results, root_policy_logits))
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

    pub fn set_gumbel_scale(&mut self, gumbel_scale: f32) -> PyResult<()> {
        if !gumbel_scale.is_finite() || gumbel_scale < 0.0 {
            return Err(PyValueError::new_err(
                "gumbel_scale must be a finite non-negative value",
            ));
        }
        for search_pair in &mut self.searches {
            search_pair[0].config.gumbel_scale = gumbel_scale;
            search_pair[1].config.gumbel_scale = gumbel_scale;
        }
        Ok(())
    }
}

impl GumbelArenaBatch {
    #[must_use]
    pub fn new(
        game_count: usize,
        seed_start: u64,
        game_index_start: usize,
        config: GumbelConfig,
        paired_seeds: bool,
    ) -> Self {
        let mut searches = Vec::with_capacity(game_count);
        let mut candidate_players = Vec::with_capacity(game_count);
        let mut seeds = Vec::with_capacity(game_count);

        for chunk_index in 0..game_count {
            let game_index = game_index_start + chunk_index;
            let seed_offset = if paired_seeds {
                (game_index / 2) as u64
            } else {
                chunk_index as u64
            };
            let game_seed = seed_start.wrapping_add(seed_offset);
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

    fn search_active_with_logits_evaluator(
        &mut self,
        rows: Vec<Vec<f32>>,
        root_values: &[f32],
        leaf_batch_size: usize,
        evaluator: &mut impl ArenaLeafEvaluator,
    ) -> PyResult<Vec<Option<GumbelResult>>> {
        let active_indexes = self.active_indexes();
        if rows.len() != active_indexes.len() {
            return Err(PyValueError::new_err(format!(
                "expected {} policy rows for active games, got {}",
                active_indexes.len(),
                rows.len()
            )));
        }
        if root_values.len() != active_indexes.len() {
            return Err(PyValueError::new_err(format!(
                "expected {} root values for active games, got {}",
                active_indexes.len(),
                root_values.len()
            )));
        }
        if root_values.iter().any(|value| !value.is_finite()) {
            return Err(PyValueError::new_err(
                "root values must be finite for active games",
            ));
        }

        let mut roots: Vec<Option<RootSearchState>> = std::iter::repeat_with(|| None)
            .take(self.states.len())
            .collect();

        for (active_offset, (game_index, row)) in active_indexes
            .iter()
            .copied()
            .zip(rows.into_iter())
            .enumerate()
        {
            if row.len() != ACTION_SPACE {
                return Err(PyValueError::new_err(format!(
                    "expected {ACTION_SPACE} policy values for game {game_index}, got {}",
                    row.len()
                )));
            }
            let search = current_player_search_mut(
                &mut self.searches[game_index],
                &self.states[game_index],
            )?;
            roots[game_index] = start_root_search(
                search,
                &self.states[game_index],
                &row,
                true,
                root_values[active_offset],
            )?;
        }

        let mut wave = 0_u64;
        while active_indexes.iter().any(|index| {
            let Some(search) = current_player_search(&self.searches[*index], &self.states[*index])
            else {
                return false;
            };
            roots[*index]
                .as_ref()
                .is_some_and(|root| root.has_remaining(search))
        }) {
            evaluator.check_signals()?;
            wave = wave.wrapping_add(1);
            let pending_by_game: Vec<Vec<_>> = self
                .states
                .par_iter()
                .zip(self.searches.par_iter_mut())
                .zip(roots.par_iter_mut())
                .enumerate()
                .map(|(game_index, ((state, search_pair), root))| {
                    let Some(root) = root.as_mut() else {
                        return Vec::new();
                    };
                    let Some(search) = current_player_search_mut_or_none(search_pair, state) else {
                        return Vec::new();
                    };
                    let batch_target = (search.config.simulations - root.completed)
                        .min(leaf_batch_size as u32)
                        as usize;
                    let mut local_pending = Vec::with_capacity(batch_target);
                    for _ in 0..batch_target {
                        if root.completed + local_pending.len() as u32 >= search.config.simulations
                        {
                            break;
                        }
                        let Some(root_action) = root.scheduler.next_action() else {
                            break;
                        };
                        let mut simulation_state = state.clone();
                        match search.select_eval_leaf(
                            root.root_index,
                            root_action,
                            &mut simulation_state,
                        ) {
                            PendingGumbelSimulation::NeedsEvaluation {
                                path,
                                state: leaf_state,
                            } => {
                                reserve_path(&mut search.nodes, &path);
                                root.scheduler.reserve_visit(root_action);
                                local_pending.push(PendingArenaLeaf {
                                    game_index,
                                    path,
                                    state: leaf_state,
                                });
                            }
                            PendingGumbelSimulation::Terminal { path, value } => {
                                backup_path(&mut search.nodes, &path, value, false);
                                root.scheduler.reserve_visit(root_action);
                                root.complete_reserved_visits(search);
                                root.completed += 1;
                            }
                            PendingGumbelSimulation::BlockedPending => break,
                        }
                    }
                    local_pending
                })
                .collect();
            let pending = pending_by_game.into_iter().flatten().collect::<Vec<_>>();
            if pending.is_empty() {
                continue;
            }

            let request_states = pending
                .iter()
                .map(|leaf| leaf.state.clone())
                .collect::<Vec<_>>();
            let game_indexes = pending
                .iter()
                .map(|leaf| leaf.game_index)
                .collect::<Vec<_>>();
            let eval =
                evaluator.evaluate(request_states, game_indexes, active_indexes.len(), wave)?;
            eval.validate_len(pending.len())?;

            let mut by_game: Vec<Vec<PendingArenaEvaluation>> =
                (0..self.states.len()).map(|_| Vec::new()).collect();
            for (leaf, (policy_row, value)) in pending
                .into_iter()
                .zip(eval.policies.into_iter().zip(eval.values.into_iter()))
            {
                by_game[leaf.game_index].push(PendingArenaEvaluation {
                    path: leaf.path,
                    state: leaf.state,
                    policy_row,
                    value,
                });
            }
            self.searches
                .par_iter_mut()
                .zip(self.states.par_iter())
                .zip(roots.par_iter_mut())
                .zip(by_game.into_par_iter())
                .try_for_each(|(((search_pair, state), root), evaluations)| {
                    let completed_count = evaluations.len() as u32;
                    let Some(search) = current_player_search_mut_or_none(search_pair, state) else {
                        if completed_count == 0 {
                            return Ok(());
                        }
                        return Err("missing Gumbel search for current player".to_string());
                    };
                    for evaluation in evaluations {
                        unreserve_path(&mut search.nodes, &evaluation.path);
                        let child_index = search
                            .expand_evaluated_node(
                                &evaluation.state,
                                &evaluation.policy_row,
                                evaluation.value,
                                true,
                            )
                            .map_err(|err| err.to_string())?;
                        if let Some((parent_index, edge_index)) = evaluation.path.last().copied() {
                            search.nodes[parent_index].edges[edge_index].child = Some(child_index);
                        }
                        backup_path(&mut search.nodes, &evaluation.path, evaluation.value, true);
                    }
                    if completed_count > 0 {
                        let root = root.as_mut().ok_or_else(|| {
                            "missing Gumbel root index for completed evaluations".to_string()
                        })?;
                        root.complete_reserved_visits(search);
                        root.completed += completed_count;
                    }
                    Ok::<(), String>(())
                })
                .map_err(|err| {
                    PyValueError::new_err(format!("failed to expand Gumbel evaluation: {err}"))
                })?;
        }

        let mut results = vec![None; self.states.len()];
        for game_index in active_indexes {
            let Some(root) = &roots[game_index] else {
                results[game_index] = Some(empty_result());
                continue;
            };
            let search =
                current_player_search(&self.searches[game_index], &self.states[game_index])
                    .ok_or_else(|| PyValueError::new_err("invalid current player"))?;
            results[game_index] = Some(root.finish(search));
        }
        Ok(results)
    }
}

trait ArenaLeafEvaluator {
    fn check_signals(&mut self) -> PyResult<()>;

    fn evaluate(
        &mut self,
        states: Vec<GameState>,
        game_indexes: Vec<usize>,
        active_games: usize,
        wave: u64,
    ) -> PyResult<GumbelEvalBatch>;
}

struct PythonArenaLeafEvaluator<'a, 'py> {
    callback: &'a Bound<'py, PyAny>,
}

impl<'a, 'py> PythonArenaLeafEvaluator<'a, 'py> {
    const fn new(callback: &'a Bound<'py, PyAny>) -> Self {
        Self { callback }
    }
}

impl ArenaLeafEvaluator for PythonArenaLeafEvaluator<'_, '_> {
    fn check_signals(&mut self) -> PyResult<()> {
        self.callback.py().check_signals()
    }

    fn evaluate(
        &mut self,
        states: Vec<GameState>,
        game_indexes: Vec<usize>,
        _active_games: usize,
        _wave: u64,
    ) -> PyResult<GumbelEvalBatch> {
        let request = EvalRequest::new_with_game_indexes(states, game_indexes);
        let response = self.callback.call1((request,))?;
        parse_gumbel_eval_response(&response)
    }
}

struct OnnxArenaLeafEvaluator<'a> {
    candidate: &'a mut OnnxEvaluator,
    best: &'a mut OnnxEvaluator,
    candidate_players: Vec<u8>,
}

impl<'a> OnnxArenaLeafEvaluator<'a> {
    fn new(
        candidate: &'a mut OnnxEvaluator,
        best: &'a mut OnnxEvaluator,
        candidate_players: Vec<u8>,
    ) -> Self {
        Self {
            candidate,
            best,
            candidate_players,
        }
    }
}

impl ArenaLeafEvaluator for OnnxArenaLeafEvaluator<'_> {
    fn check_signals(&mut self) -> PyResult<()> {
        Python::with_gil(|py| py.check_signals())
    }

    fn evaluate(
        &mut self,
        states: Vec<GameState>,
        game_indexes: Vec<usize>,
        active_games: usize,
        wave: u64,
    ) -> PyResult<GumbelEvalBatch> {
        if states.len() != game_indexes.len() {
            return Err(PyValueError::new_err(
                "arena ONNX state and game index counts must match",
            ));
        }

        let mut candidate_rows = Vec::new();
        let mut candidate_states = Vec::new();
        let mut best_rows = Vec::new();
        let mut best_states = Vec::new();
        for (row, (state, game_index)) in
            states.into_iter().zip(game_indexes.into_iter()).enumerate()
        {
            let Some(candidate_player) = self.candidate_players.get(game_index).copied() else {
                return Err(PyValueError::new_err(format!(
                    "arena ONNX game index out of range: {game_index}"
                )));
            };
            match state.current_player() {
                player if player == candidate_player => {
                    candidate_rows.push(row);
                    candidate_states.push(state);
                }
                BLUE | ORANGE => {
                    best_rows.push(row);
                    best_states.push(state);
                }
                player => {
                    return Err(PyValueError::new_err(format!(
                        "invalid arena current player: {player}"
                    )));
                }
            }
        }

        let mut policies = vec![None; candidate_rows.len() + best_rows.len()];
        let mut values = vec![None; policies.len()];
        fill_onnx_rows(
            self.candidate,
            candidate_states,
            candidate_rows,
            &mut policies,
            &mut values,
            active_games,
            wave,
        )?;
        fill_onnx_rows(
            self.best,
            best_states,
            best_rows,
            &mut policies,
            &mut values,
            active_games,
            wave,
        )?;
        Ok(GumbelEvalBatch::new(
            policies
                .into_iter()
                .map(|row| {
                    row.ok_or_else(|| {
                        PyRuntimeError::new_err("arena ONNX evaluation missed a policy row")
                    })
                })
                .collect::<PyResult<Vec<_>>>()?,
            values
                .into_iter()
                .map(|value| {
                    value.ok_or_else(|| {
                        PyRuntimeError::new_err("arena ONNX evaluation missed a value row")
                    })
                })
                .collect::<PyResult<Vec<_>>>()?,
        ))
    }
}

fn fill_onnx_rows(
    evaluator: &mut OnnxEvaluator,
    states: Vec<GameState>,
    rows: Vec<usize>,
    policies: &mut [Option<[f32; ACTION_SPACE]>],
    values: &mut [Option<f32>],
    active_games: usize,
    wave: u64,
) -> PyResult<()> {
    if states.is_empty() {
        return Ok(());
    }
    if wave == 0 {
        evaluator.set_gumbel_root_profile_context(active_games);
    } else {
        evaluator.set_gumbel_leaf_profile_context(wave, active_games, rows.len());
    }
    let output = evaluator
        .evaluate_request(&EvalRequest::new_with_precomputed_features(states))
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
    assign_onnx_output(output, rows, policies, values)
}

fn assign_onnx_output(
    output: NetworkOutput,
    rows: Vec<usize>,
    policies: &mut [Option<[f32; ACTION_SPACE]>],
    values: &mut [Option<f32>],
) -> PyResult<()> {
    if output.policy_logits.len() != rows.len() || output.values.len() != rows.len() {
        return Err(PyRuntimeError::new_err(
            "arena ONNX evaluator returned a mismatched batch size",
        ));
    }
    for (row, (policy, value)) in rows.into_iter().zip(
        output
            .policy_logits
            .into_iter()
            .zip(output.values.into_iter()),
    ) {
        policies[row] = Some(policy);
        values[row] = Some(value);
    }
    Ok(())
}

fn current_player_search<'a>(
    searches: &'a [GumbelSearch; 2],
    state: &GameState,
) -> Option<&'a GumbelSearch> {
    match state.current_player() {
        BLUE => Some(&searches[0]),
        ORANGE => Some(&searches[1]),
        _ => None,
    }
}

fn current_player_search_mut<'a>(
    searches: &'a mut [GumbelSearch; 2],
    state: &GameState,
) -> PyResult<&'a mut GumbelSearch> {
    current_player_search_mut_or_none(searches, state)
        .ok_or_else(|| PyValueError::new_err("invalid current player"))
}

fn current_player_search_mut_or_none<'a>(
    searches: &'a mut [GumbelSearch; 2],
    state: &GameState,
) -> Option<&'a mut GumbelSearch> {
    match state.current_player() {
        BLUE => Some(&mut searches[0]),
        ORANGE => Some(&mut searches[1]),
        _ => None,
    }
}

#[derive(Clone, Debug)]
struct PendingArenaLeaf {
    game_index: usize,
    path: Vec<(usize, usize)>,
    state: GameState,
}

#[derive(Clone, Debug)]
struct PendingArenaEvaluation {
    path: Vec<(usize, usize)>,
    state: GameState,
    policy_row: [f32; ACTION_SPACE],
    value: f32,
}

#[cfg(test)]
mod tests {
    use super::GumbelArenaBatch;
    use crate::gumbel::config::GumbelConfig;

    #[test]
    fn new_uses_global_game_index_for_candidate_side_split() {
        let batch = GumbelArenaBatch::new(4, 10, 1, GumbelConfig::new(4, 2, 50.0, 1.0, 7), false);

        assert_eq!(batch.seeds, vec![10, 11, 12, 13]);
        assert_eq!(batch.candidate_players, vec![2, 1, 2, 1]);
    }

    #[test]
    fn new_can_pair_seeds_by_global_game_index() {
        let batch = GumbelArenaBatch::new(4, 10, 1, GumbelConfig::new(4, 2, 50.0, 1.0, 7), true);

        assert_eq!(batch.seeds, vec![10, 11, 11, 12]);
        assert_eq!(batch.candidate_players, vec![2, 1, 2, 1]);
    }

    #[test]
    fn new_offsets_player_search_seeds_from_game_seed() {
        let batch = GumbelArenaBatch::new(2, 10, 0, GumbelConfig::new(4, 2, 50.0, 1.0, 7), false);

        assert_eq!(batch.searches[0][0].seed(), 27);
        assert_eq!(batch.searches[0][1].seed(), 28);
        assert_eq!(batch.searches[1][0].seed(), 29);
        assert_eq!(batch.searches[1][1].seed(), 30);
    }

    #[test]
    fn active_eval_request_filters_terminal_games() {
        let mut batch =
            GumbelArenaBatch::new(2, 0, 0, GumbelConfig::new(4, 2, 50.0, 1.0, 7), false);

        assert_eq!(
            batch.apply_actions(vec![Some(81), None]).unwrap(),
            vec![None, None]
        );
        assert_eq!(
            batch.apply_actions(vec![Some(81), None]).unwrap(),
            vec![Some(2), None]
        );

        let request = batch.active_eval_request();
        assert_eq!(batch.active_game_indexes(), vec![1]);
        assert_eq!(request.len(), 1);
        assert_eq!(request.game_indexes(), vec![1]);
        assert_eq!(request.current_players(), vec![1]);
    }

    #[test]
    fn apply_actions_updates_terminal_state_and_score_accessors() {
        let mut batch =
            GumbelArenaBatch::new(1, 0, 0, GumbelConfig::new(4, 2, 50.0, 1.0, 7), false);

        assert_eq!(batch.apply_actions(vec![Some(81)]).unwrap(), vec![None]);
        assert_eq!(batch.is_terminal(), vec![false]);
        assert_eq!(batch.apply_actions(vec![Some(81)]).unwrap(), vec![Some(2)]);

        assert_eq!(batch.is_terminal(), vec![true]);
        assert_eq!(batch.winners(), vec![Some(2)]);
        assert_eq!(batch.end_reasons(), vec![Some(3)]);
        assert_eq!(batch.territory_scores(), vec![(0, 0)]);
    }
}
