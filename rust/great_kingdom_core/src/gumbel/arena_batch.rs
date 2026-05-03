use pyo3::{exceptions::PyValueError, prelude::*};
use rayon::prelude::*;

use super::{
    config::GumbelConfig,
    node::GumbelNode,
    policy::{log_priors_from_logits, root_improved_policy_target},
    result::GumbelResult,
    sampling::sample_root_candidates,
    search::{
        GumbelSearch, PendingGumbelSimulation, backup_path, parse_gumbel_eval_response,
        reserve_path, root_ranking_scores, unreserve_path,
    },
    sequential_halving::RootSequentialHalving,
};
use crate::{
    eval_request::EvalRequest,
    game::{ACTION_SPACE, Action, GameState},
};

const BLUE: u8 = 1;
const ORANGE: u8 = 2;

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
        self.search_active_with_logits_evaluator(
            policy_logits,
            evaluator,
            &root_values,
            leaf_batch_size,
        )
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

    fn search_active_with_logits_evaluator(
        &mut self,
        rows: Vec<Vec<f32>>,
        evaluator: &Bound<'_, PyAny>,
        root_values: &[f32],
        leaf_batch_size: usize,
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

        let mut root_indexes = vec![None; self.states.len()];
        let mut root_legal_actions = vec![None; self.states.len()];
        let mut root_log_priors = vec![None; self.states.len()];
        let mut completed = vec![0_u32; self.states.len()];
        let mut schedulers = vec![None; self.states.len()];

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
            let legal_actions = self.states[game_index].legal_action_indexes();
            if legal_actions.is_empty() || self.states[game_index].is_terminal() {
                continue;
            }
            let log_priors = log_priors_from_logits(&legal_actions, &row)?;
            let search = current_player_search_mut(
                &mut self.searches[game_index],
                &self.states[game_index],
            )?;
            let candidates = sample_root_candidates(
                &legal_actions,
                &log_priors,
                search.config.max_considered_actions,
                search.config.simulations,
                search.next_root_seed(),
            );
            if candidates.is_empty() {
                continue;
            }
            search.nodes.clear();
            let root_index = search.nodes.len();
            search.nodes.push(GumbelNode::root_from_candidates(
                &self.states[game_index],
                &candidates,
                root_values[active_offset],
            ));
            root_indexes[game_index] = Some(root_index);
            root_legal_actions[game_index] = Some(legal_actions);
            root_log_priors[game_index] = Some(log_priors);
            schedulers[game_index] = Some(RootSequentialHalving::new(
                candidates
                    .iter()
                    .map(|candidate| (candidate.action, candidate.score))
                    .collect(),
                search.config.simulations,
            ));
        }

        while active_indexes.iter().any(|index| {
            let Some(search) = current_player_search(&self.searches[*index], &self.states[*index])
            else {
                return false;
            };
            completed[*index] < search.config.simulations
        }) {
            evaluator.py().check_signals()?;
            let pending_by_game: Vec<Vec<_>> = self
                .states
                .par_iter()
                .zip(self.searches.par_iter_mut())
                .zip(completed.par_iter_mut())
                .zip(root_indexes.par_iter())
                .zip(schedulers.par_iter_mut())
                .enumerate()
                .map(
                    |(game_index, ((((state, search_pair), comp), root_index), scheduler))| {
                        let Some(root_index) = *root_index else {
                            return Vec::new();
                        };
                        let Some(scheduler) = scheduler.as_mut() else {
                            return Vec::new();
                        };
                        let Some(search) = current_player_search_mut_or_none(search_pair, state)
                        else {
                            return Vec::new();
                        };
                        let batch_target = (search.config.simulations - *comp)
                            .min(leaf_batch_size as u32)
                            as usize;
                        let mut local_pending = Vec::with_capacity(batch_target);
                        for _ in 0..batch_target {
                            if *comp + local_pending.len() as u32 >= search.config.simulations {
                                break;
                            }
                            let Some(root_action) = scheduler.next_action() else {
                                break;
                            };
                            let mut simulation_state = state.clone();
                            match search.select_eval_leaf(
                                root_index,
                                root_action,
                                &mut simulation_state,
                            ) {
                                PendingGumbelSimulation::NeedsEvaluation {
                                    path,
                                    state: leaf_state,
                                } => {
                                    reserve_path(&mut search.nodes, &path);
                                    scheduler.reserve_visit(root_action);
                                    local_pending.push(PendingArenaLeaf {
                                        game_index,
                                        path,
                                        state: leaf_state,
                                    });
                                }
                                PendingGumbelSimulation::Terminal { path, value } => {
                                    backup_path(&mut search.nodes, &path, value, false);
                                    scheduler.reserve_visit(root_action);
                                    scheduler.complete_reserved_visits(&root_ranking_scores(
                                        &search.nodes[root_index],
                                        search.config.c_visit,
                                        search.config.c_scale,
                                    ));
                                    *comp += 1;
                                }
                                PendingGumbelSimulation::BlockedPending => break,
                            }
                        }
                        local_pending
                    },
                )
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
            let request = EvalRequest::new_with_game_indexes(request_states, game_indexes);
            let response = evaluator.call1((request,))?;
            let eval = parse_gumbel_eval_response(&response)?;
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
                .zip(completed.par_iter_mut())
                .zip(root_indexes.par_iter())
                .zip(schedulers.par_iter_mut())
                .zip(by_game.into_par_iter())
                .try_for_each(
                    |(((((search_pair, state), comp), root_index), scheduler), evaluations)| {
                        let completed_count = evaluations.len() as u32;
                        let Some(search) = current_player_search_mut_or_none(search_pair, state)
                        else {
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
                            if let Some((parent_index, edge_index)) =
                                evaluation.path.last().copied()
                            {
                                search.nodes[parent_index].edges[edge_index].child =
                                    Some(child_index);
                            }
                            backup_path(
                                &mut search.nodes,
                                &evaluation.path,
                                evaluation.value,
                                true,
                            );
                        }
                        if completed_count > 0 {
                            let root_index = root_index.ok_or_else(|| {
                                "missing Gumbel root index for completed evaluations".to_string()
                            })?;
                            let scheduler = scheduler.as_mut().ok_or_else(|| {
                                "missing Gumbel scheduler for completed evaluations".to_string()
                            })?;
                            scheduler.complete_reserved_visits(&root_ranking_scores(
                                &search.nodes[root_index],
                                search.config.c_visit,
                                search.config.c_scale,
                            ));
                        }
                        *comp += completed_count;
                        Ok::<(), String>(())
                    },
                )
                .map_err(|err| {
                    PyValueError::new_err(format!("failed to expand Gumbel evaluation: {err}"))
                })?;
        }

        let mut results = vec![None; self.states.len()];
        for game_index in active_indexes {
            let Some(root_index) = root_indexes[game_index] else {
                results[game_index] = Some(GumbelResult {
                    selected_action: None,
                    policy_target: [0.0; ACTION_SPACE],
                    visit_counts: [0; ACTION_SPACE],
                });
                continue;
            };
            let search =
                current_player_search(&self.searches[game_index], &self.states[game_index])
                    .ok_or_else(|| PyValueError::new_err("invalid current player"))?;
            let root = &search.nodes[root_index];
            let legal_actions = root_legal_actions[game_index]
                .as_deref()
                .ok_or_else(|| PyValueError::new_err("missing Gumbel root legal actions"))?;
            let log_priors = root_log_priors[game_index]
                .as_ref()
                .ok_or_else(|| PyValueError::new_err("missing Gumbel root log priors"))?;
            let improved = root_improved_policy_target(
                root,
                legal_actions,
                log_priors,
                search.config.c_visit,
                search.config.c_scale,
            );
            let selected_action = schedulers[game_index]
                .as_ref()
                .and_then(RootSequentialHalving::selected_action);
            results[game_index] = Some(GumbelResult {
                selected_action,
                policy_target: improved.policy_target,
                visit_counts: root.visit_counts(),
            });
        }
        Ok(results)
    }
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
