use pyo3::{exceptions::PyValueError, prelude::*};
use rayon::prelude::*;
use std::{env, time::Instant};

use super::{
    config::GumbelConfig,
    node::GumbelNode,
    policy::{log_priors_from_logits, log_priors_from_priors, root_improved_policy_target},
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

#[pyclass]
#[derive(Clone, Debug)]
pub struct GumbelSelfPlayBatch {
    states: Vec<GameState>,
    searches: Vec<GumbelSearch>,
}

#[pymethods]
impl GumbelSelfPlayBatch {
    #[new]
    #[pyo3(signature = (
        game_count,
        simulations = 128,
        max_considered_actions = 16,
        c_visit = 50.0,
        c_scale = 1.0,
        seed = 2026
    ))]
    pub fn py_new(
        game_count: usize,
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
        Ok(Self::new(game_count, config))
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
        EvalRequest::new_with_precomputed_bytes(
            self.active_indexes()
                .into_iter()
                .map(|index| self.states[index].clone())
                .collect(),
        )
    }

    #[must_use]
    pub fn current_players(&self) -> Vec<u8> {
        self.states.iter().map(GameState::current_player).collect()
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

    pub fn search_active_with_logits(
        &mut self,
        policy_logits: Vec<Vec<f32>>,
    ) -> PyResult<Vec<Option<GumbelResult>>> {
        self.search_active(policy_logits, true)
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
        self.search_active_with_evaluator(
            policy_logits,
            true,
            evaluator,
            leaf_batch_size,
            &root_values,
        )
    }

    pub fn search_active_with_priors(
        &mut self,
        priors: Vec<Vec<f32>>,
    ) -> PyResult<Vec<Option<GumbelResult>>> {
        self.search_active(priors, false)
    }

    #[pyo3(signature = (priors, evaluator, root_values, leaf_batch_size = 16))]
    pub fn search_active_with_priors_and_evaluator(
        &mut self,
        priors: Vec<Vec<f32>>,
        evaluator: &Bound<'_, PyAny>,
        root_values: Vec<f32>,
        leaf_batch_size: usize,
    ) -> PyResult<Vec<Option<GumbelResult>>> {
        if leaf_batch_size == 0 {
            return Err(PyValueError::new_err("leaf_batch_size must be positive"));
        }
        self.search_active_with_evaluator(priors, false, evaluator, leaf_batch_size, &root_values)
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

    pub fn set_seeds(&mut self, seeds: Vec<Option<u64>>) -> PyResult<()> {
        if seeds.len() != self.searches.len() {
            return Err(PyValueError::new_err(format!(
                "expected {} seed slots, got {}",
                self.searches.len(),
                seeds.len()
            )));
        }
        for (search, seed) in self.searches.iter_mut().zip(seeds.into_iter()) {
            if let Some(seed) = seed {
                search.set_seed(seed);
            }
        }
        Ok(())
    }
}

impl GumbelSelfPlayBatch {
    #[must_use]
    pub fn new(game_count: usize, config: GumbelConfig) -> Self {
        let searches = (0..game_count)
            .map(|index| {
                let mut game_config = config;
                game_config.seed = config.seed.wrapping_add(index as u64);
                GumbelSearch::new(game_config)
            })
            .collect();

        Self {
            states: vec![GameState::new(); game_count],
            searches,
        }
    }

    fn active_indexes(&self) -> Vec<usize> {
        self.states
            .iter()
            .enumerate()
            .filter_map(|(index, state)| (!state.is_terminal()).then_some(index))
            .collect()
    }

    fn search_active(
        &mut self,
        rows: Vec<Vec<f32>>,
        logits: bool,
    ) -> PyResult<Vec<Option<GumbelResult>>> {
        let active_indexes = self.active_indexes();
        if rows.len() != active_indexes.len() {
            return Err(PyValueError::new_err(format!(
                "expected {} policy rows for active games, got {}",
                active_indexes.len(),
                rows.len()
            )));
        }

        let mut results = vec![None; self.states.len()];
        for (game_index, row) in active_indexes.into_iter().zip(rows.into_iter()) {
            if row.len() != ACTION_SPACE {
                return Err(PyValueError::new_err(format!(
                    "expected {ACTION_SPACE} policy values for game {game_index}, got {}",
                    row.len()
                )));
            }
            let result = if logits {
                self.searches[game_index].search_with_logits(&self.states[game_index], row)?
            } else {
                self.searches[game_index].search_with_priors(&self.states[game_index], row)?
            };
            results[game_index] = Some(result);
        }
        Ok(results)
    }

    fn search_active_with_evaluator(
        &mut self,
        rows: Vec<Vec<f32>>,
        logits: bool,
        evaluator: &Bound<'_, PyAny>,
        leaf_batch_size: usize,
        root_values: &[f32],
    ) -> PyResult<Vec<Option<GumbelResult>>> {
        let profile = GumbelBatchProfile::new(if logits {
            "search_active_with_logits_and_evaluator"
        } else {
            "search_active_with_priors_and_evaluator"
        });
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

        let root_start = Instant::now();
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
            let log_priors = if logits {
                log_priors_from_logits(&legal_actions, &row)?
            } else {
                log_priors_from_priors(&legal_actions, &row)?
            };
            let search = &mut self.searches[game_index];
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
        profile.root(active_indexes.len(), root_start.elapsed());

        let mut wave = 0_u64;
        while active_indexes
            .iter()
            .any(|index| completed[*index] < self.searches[*index].config.simulations)
        {
            evaluator.py().check_signals()?;
            wave += 1;
            let select_start = Instant::now();
            let pending_by_game: Vec<Vec<_>> = self
                .states
                .par_iter()
                .zip(self.searches.par_iter_mut())
                .zip(completed.par_iter_mut())
                .zip(root_indexes.par_iter())
                .zip(schedulers.par_iter_mut())
                .enumerate()
                .map(
                    |(game_index, ((((state, search), comp), root_index), scheduler))| {
                        let Some(root_index) = *root_index else {
                            return Vec::new();
                        };
                        let Some(scheduler) = scheduler.as_mut() else {
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
                                    local_pending.push(PendingBatchLeaf {
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
            let select_elapsed = select_start.elapsed();
            let flatten_start = Instant::now();
            let pending = pending_by_game.into_iter().flatten().collect::<Vec<_>>();
            let flatten_elapsed = flatten_start.elapsed();
            if pending.is_empty() {
                profile.empty_wave(wave, active_indexes.len(), select_elapsed, flatten_elapsed);
                continue;
            }

            let request_start = Instant::now();
            let request_states = pending
                .iter()
                .map(|leaf| leaf.state.clone())
                .collect::<Vec<_>>();
            let request = EvalRequest::new_with_precomputed_bytes(request_states);
            let request_elapsed = request_start.elapsed();
            let eval_start = Instant::now();
            let response = evaluator.call1((request,))?;
            let eval_elapsed = eval_start.elapsed();
            let parse_start = Instant::now();
            let eval = parse_gumbel_eval_response(&response)?;
            eval.validate_len(pending.len())?;
            let parse_elapsed = parse_start.elapsed();

            let backup_start = Instant::now();
            let leaves = pending.len();
            let mut by_game: Vec<Vec<PendingGameEvaluation>> =
                (0..self.states.len()).map(|_| Vec::new()).collect();
            for (leaf, (policy_row, value)) in pending
                .into_iter()
                .zip(eval.policies.into_iter().zip(eval.values.into_iter()))
            {
                by_game[leaf.game_index].push(PendingGameEvaluation {
                    path: leaf.path,
                    state: leaf.state,
                    policy_row,
                    value,
                });
            }
            self.searches
                .par_iter_mut()
                .zip(completed.par_iter_mut())
                .zip(root_indexes.par_iter())
                .zip(schedulers.par_iter_mut())
                .zip(by_game.into_par_iter())
                .try_for_each(|((((search, comp), root_index), scheduler), evaluations)| {
                    let completed_count = evaluations.len() as u32;
                    for evaluation in evaluations {
                        unreserve_path(&mut search.nodes, &evaluation.path);
                        let child_index = search
                            .expand_evaluated_node(
                                &evaluation.state,
                                &evaluation.policy_row,
                                evaluation.value,
                                logits,
                            )
                            .map_err(|err| err.to_string())?;
                        if let Some((parent_index, edge_index)) = evaluation.path.last().copied() {
                            search.nodes[parent_index].edges[edge_index].child = Some(child_index);
                        }
                        backup_path(&mut search.nodes, &evaluation.path, evaluation.value, true);
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
                })
                .map_err(|err| {
                    PyValueError::new_err(format!("failed to expand Gumbel evaluation: {err}"))
                })?;
            profile.wave(GumbelBatchWaveProfile {
                wave,
                active_games: active_indexes.len(),
                leaves,
                select_elapsed,
                flatten_elapsed,
                request_elapsed,
                eval_elapsed,
                parse_elapsed,
                backup_elapsed: backup_start.elapsed(),
            });
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
            let root = &self.searches[game_index].nodes[root_index];
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
                self.searches[game_index].config.c_visit,
                self.searches[game_index].config.c_scale,
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

#[derive(Clone, Debug)]
struct PendingBatchLeaf {
    game_index: usize,
    path: Vec<(usize, usize)>,
    state: GameState,
}

#[derive(Clone, Debug)]
struct PendingGameEvaluation {
    path: Vec<(usize, usize)>,
    state: GameState,
    policy_row: [f32; ACTION_SPACE],
    value: f32,
}

#[derive(Clone, Copy)]
struct GumbelBatchProfile {
    enabled: bool,
    interval: u64,
    name: &'static str,
}

struct GumbelBatchWaveProfile {
    wave: u64,
    active_games: usize,
    leaves: usize,
    select_elapsed: std::time::Duration,
    flatten_elapsed: std::time::Duration,
    request_elapsed: std::time::Duration,
    eval_elapsed: std::time::Duration,
    parse_elapsed: std::time::Duration,
    backup_elapsed: std::time::Duration,
}

impl GumbelBatchProfile {
    fn new(name: &'static str) -> Self {
        Self {
            enabled: env_flag("GKA_GUMBEL_PROFILE"),
            interval: env::var("GKA_GUMBEL_PROFILE_INTERVAL")
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
            "[gka-gumbel-profile] fn={} root active_games={} rayon_threads={} init={:.3}s",
            self.name,
            active_games,
            rayon::current_num_threads(),
            elapsed.as_secs_f64(),
        );
    }

    fn empty_wave(
        &self,
        wave: u64,
        active_games: usize,
        select_elapsed: std::time::Duration,
        flatten_elapsed: std::time::Duration,
    ) {
        if !self.enabled || wave % self.interval != 0 {
            return;
        }
        eprintln!(
            "[gka-gumbel-profile] fn={} wave={} empty active_games={} select={:.3}s flatten={:.3}s",
            self.name,
            wave,
            active_games,
            select_elapsed.as_secs_f64(),
            flatten_elapsed.as_secs_f64(),
        );
    }

    fn wave(&self, profile: GumbelBatchWaveProfile) {
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
            "[gka-gumbel-profile] fn={} wave={} active_games={} leaves={} select={:.3}s flatten={:.3}s request={:.3}s eval_call={:.3}s parse={:.3}s backup={:.3}s total={:.3}s",
            self.name,
            profile.wave,
            profile.active_games,
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

#[cfg(test)]
mod tests {
    use super::GumbelSelfPlayBatch;
    use crate::gumbel::config::GumbelConfig;

    #[test]
    fn new_offsets_search_seed_per_game() {
        let mut batch = GumbelSelfPlayBatch::new(3, GumbelConfig::new(4, 2, 50.0, 1.0, 7));

        assert_eq!(batch.searches[0].seed(), 7);
        assert_eq!(batch.searches[1].seed(), 8);
        assert_eq!(batch.searches[2].seed(), 9);
        assert_eq!(batch.searches[0].next_root_seed(), 7);
        assert_eq!(batch.searches[1].next_root_seed(), 8);
        assert_eq!(batch.searches[2].next_root_seed(), 9);
    }
}
