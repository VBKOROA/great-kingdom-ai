use pyo3::{buffer::PyBuffer, exceptions::PyValueError, prelude::*};

use super::{
    config::GumbelConfig,
    node::GumbelNode,
    policy::{
        log_priors_from_logits, log_priors_from_priors, root_improved_logits,
        root_improved_policy_target,
    },
    result::GumbelResult,
    sampling::{RootCandidate, sample_root_candidates},
    selection::select_inner_action,
    sequential_halving::RootSequentialHalving,
};
use crate::{
    game::{ACTION_SPACE, GameOutcome, GameState, Player},
    mcts::EvalRequest,
};

#[pyclass]
#[derive(Clone, Debug)]
pub struct GumbelSearch {
    pub(crate) config: GumbelConfig,
    pub(crate) nodes: Vec<GumbelNode>,
    root_search_count: u64,
}

#[pymethods]
impl GumbelSearch {
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
        let config = GumbelConfig::new(simulations, max_considered_actions, c_visit, c_scale, seed);
        config.validate()?;
        Ok(Self::new(config))
    }

    #[must_use]
    pub fn simulations(&self) -> u32 {
        self.config.simulations
    }

    #[must_use]
    pub fn max_considered_actions(&self) -> usize {
        self.config.max_considered_actions
    }

    #[must_use]
    pub fn c_visit(&self) -> f32 {
        self.config.c_visit
    }

    #[must_use]
    pub fn c_scale(&self) -> f32 {
        self.config.c_scale
    }

    #[must_use]
    pub fn seed(&self) -> u64 {
        self.config.seed
    }

    pub fn set_simulations(&mut self, simulations: u32) -> PyResult<()> {
        if simulations == 0 {
            return Err(PyValueError::new_err("simulations must be positive"));
        }
        self.config.simulations = simulations;
        Ok(())
    }

    pub fn set_seed(&mut self, seed: u64) {
        self.config.seed = seed;
        self.root_search_count = 0;
    }

    pub fn search_with_logits(
        &mut self,
        state: &GameState,
        policy_logits: Vec<f32>,
    ) -> PyResult<GumbelResult> {
        self.result_from_logits(state, &policy_logits)
    }

    #[pyo3(signature = (state, policy_logits, evaluator, leaf_batch_size = 16))]
    pub fn search_with_logits_and_evaluator(
        &mut self,
        state: &GameState,
        policy_logits: Vec<f32>,
        evaluator: &Bound<'_, PyAny>,
        leaf_batch_size: usize,
    ) -> PyResult<GumbelResult> {
        if leaf_batch_size == 0 {
            return Err(PyValueError::new_err("leaf_batch_size must be positive"));
        }
        self.result_from_logits_with_evaluator(state, &policy_logits, leaf_batch_size, |request| {
            let response = evaluator.call1((request,))?;
            parse_gumbel_eval_response(&response)
        })
    }

    pub fn search_with_priors(
        &mut self,
        state: &GameState,
        priors: Vec<f32>,
    ) -> PyResult<GumbelResult> {
        self.result_from_priors(state, &priors)
    }

    #[pyo3(signature = (state, priors, evaluator, leaf_batch_size = 16))]
    pub fn search_with_priors_and_evaluator(
        &mut self,
        state: &GameState,
        priors: Vec<f32>,
        evaluator: &Bound<'_, PyAny>,
        leaf_batch_size: usize,
    ) -> PyResult<GumbelResult> {
        if leaf_batch_size == 0 {
            return Err(PyValueError::new_err("leaf_batch_size must be positive"));
        }
        self.result_from_priors_with_evaluator(state, &priors, leaf_batch_size, |request| {
            let response = evaluator.call1((request,))?;
            parse_gumbel_eval_response(&response)
        })
    }
}

impl GumbelSearch {
    #[must_use]
    pub const fn new(config: GumbelConfig) -> Self {
        Self {
            config,
            nodes: Vec::new(),
            root_search_count: 0,
        }
    }

    pub(crate) fn next_root_seed(&mut self) -> u64 {
        let seed = self.config.seed.wrapping_add(self.root_search_count);
        self.root_search_count = self.root_search_count.wrapping_add(1);
        seed
    }

    #[must_use]
    pub(crate) fn result_from_logits(
        &mut self,
        state: &GameState,
        logits: &[f32],
    ) -> PyResult<GumbelResult> {
        let legal_actions = state.legal_action_indexes();
        let log_priors = log_priors_from_logits(&legal_actions, logits)?;
        Ok(self.result_from_log_priors(state, &legal_actions, &log_priors))
    }

    pub(crate) fn result_from_priors(
        &mut self,
        state: &GameState,
        priors: &[f32],
    ) -> PyResult<GumbelResult> {
        let legal_actions = state.legal_action_indexes();
        let log_priors = log_priors_from_priors(&legal_actions, priors)?;
        Ok(self.result_from_log_priors(state, &legal_actions, &log_priors))
    }

    pub(crate) fn result_from_logits_with_evaluator<F>(
        &mut self,
        state: &GameState,
        logits: &[f32],
        leaf_batch_size: usize,
        evaluator: F,
    ) -> PyResult<GumbelResult>
    where
        F: FnMut(EvalRequest) -> PyResult<GumbelEvalBatch>,
    {
        let legal_actions = state.legal_action_indexes();
        let log_priors = log_priors_from_logits(&legal_actions, logits)?;
        self.result_from_log_priors_with_evaluator(
            state,
            &legal_actions,
            &log_priors,
            leaf_batch_size,
            evaluator,
            true,
        )
    }

    pub(crate) fn result_from_priors_with_evaluator<F>(
        &mut self,
        state: &GameState,
        priors: &[f32],
        leaf_batch_size: usize,
        evaluator: F,
    ) -> PyResult<GumbelResult>
    where
        F: FnMut(EvalRequest) -> PyResult<GumbelEvalBatch>,
    {
        let legal_actions = state.legal_action_indexes();
        let log_priors = log_priors_from_priors(&legal_actions, priors)?;
        self.result_from_log_priors_with_evaluator(
            state,
            &legal_actions,
            &log_priors,
            leaf_batch_size,
            evaluator,
            false,
        )
    }

    #[must_use]
    pub(crate) fn result_from_log_priors(
        &mut self,
        state: &GameState,
        legal_actions: &[usize],
        log_priors: &[f32; ACTION_SPACE],
    ) -> GumbelResult {
        if legal_actions.is_empty() || state.is_terminal() {
            return GumbelResult {
                selected_action: None,
                policy_target: [0.0; ACTION_SPACE],
                visit_counts: [0; ACTION_SPACE],
            };
        }

        let candidates = sample_root_candidates(
            legal_actions,
            log_priors,
            self.config.max_considered_actions,
            self.config.simulations,
            self.next_root_seed(),
        );
        self.run_tree_search(state, &candidates)
    }

    fn result_from_log_priors_with_evaluator<F>(
        &mut self,
        state: &GameState,
        legal_actions: &[usize],
        log_priors: &[f32; ACTION_SPACE],
        leaf_batch_size: usize,
        evaluator: F,
        evaluator_returns_logits: bool,
    ) -> PyResult<GumbelResult>
    where
        F: FnMut(EvalRequest) -> PyResult<GumbelEvalBatch>,
    {
        if legal_actions.is_empty() || state.is_terminal() {
            return Ok(GumbelResult {
                selected_action: None,
                policy_target: [0.0; ACTION_SPACE],
                visit_counts: [0; ACTION_SPACE],
            });
        }

        let candidates = sample_root_candidates(
            legal_actions,
            log_priors,
            self.config.max_considered_actions,
            self.config.simulations,
            self.next_root_seed(),
        );
        self.run_tree_search_with_evaluator(
            state,
            &candidates,
            leaf_batch_size,
            evaluator,
            evaluator_returns_logits,
        )
    }

    #[must_use]
    pub fn skeleton_result(&self, state: &GameState, scores: Option<&[f32]>) -> GumbelResult {
        let legal_actions = state.legal_action_indexes();
        if legal_actions.is_empty() || state.is_terminal() {
            return GumbelResult {
                selected_action: None,
                policy_target: [0.0; ACTION_SPACE],
                visit_counts: [0; ACTION_SPACE],
            };
        }

        let selected_action = legal_actions.iter().copied().max_by(|left, right| {
            let left_score = scores.map_or(0.0, |values| values[*left]);
            let right_score = scores.map_or(0.0, |values| values[*right]);
            left_score
                .partial_cmp(&right_score)
                .expect("validated policy scores must be finite")
                .then_with(|| right.cmp(left))
        });

        let mut policy_target = [0.0; ACTION_SPACE];
        let target = 1.0 / legal_actions.len() as f32;
        for action in legal_actions {
            policy_target[action] = target;
        }

        GumbelResult {
            selected_action,
            policy_target,
            visit_counts: [0; ACTION_SPACE],
        }
    }

    fn run_tree_search(&mut self, state: &GameState, candidates: &[RootCandidate]) -> GumbelResult {
        if candidates.is_empty() {
            return GumbelResult {
                selected_action: None,
                policy_target: [0.0; ACTION_SPACE],
                visit_counts: [0; ACTION_SPACE],
            };
        }

        self.nodes.clear();
        let root_index = self.nodes.len();
        self.nodes
            .push(GumbelNode::root_from_candidates(state, candidates));

        let mut scheduler = RootSequentialHalving::new(
            candidates
                .iter()
                .map(|candidate| (candidate.action, candidate.score))
                .collect(),
            self.config.simulations,
        );

        for _ in 0..self.config.simulations {
            let Some(root_action) = scheduler.next_action().or_else(|| {
                scheduler
                    .is_finished()
                    .then(|| self.best_root_action(root_index))
                    .flatten()
            }) else {
                break;
            };
            let mut simulation_state = state.clone();
            let Some(path_value) =
                self.run_one_simulation(root_index, root_action, &mut simulation_state)
            else {
                break;
            };
            backup_path(
                &mut self.nodes,
                &path_value.path,
                path_value.value,
                path_value.is_leaf,
            );
            scheduler.reserve_visit(root_action);
            scheduler.complete_reserved_visits(&root_ranking_scores(
                &self.nodes[root_index],
                self.config.c_visit,
                self.config.c_scale,
            ));
        }

        let improved = root_improved_policy_target(
            &self.nodes[root_index],
            self.config.c_visit,
            self.config.c_scale,
        );

        GumbelResult {
            selected_action: improved.selected_action,
            policy_target: improved.policy_target,
            visit_counts: self.nodes[root_index].visit_counts(),
        }
    }

    fn run_tree_search_with_evaluator<F>(
        &mut self,
        state: &GameState,
        candidates: &[RootCandidate],
        leaf_batch_size: usize,
        mut evaluator: F,
        evaluator_returns_logits: bool,
    ) -> PyResult<GumbelResult>
    where
        F: FnMut(EvalRequest) -> PyResult<GumbelEvalBatch>,
    {
        if candidates.is_empty() {
            return Ok(GumbelResult {
                selected_action: None,
                policy_target: [0.0; ACTION_SPACE],
                visit_counts: [0; ACTION_SPACE],
            });
        }

        self.nodes.clear();
        let root_index = self.nodes.len();
        self.nodes
            .push(GumbelNode::root_from_candidates(state, candidates));

        let mut scheduler = RootSequentialHalving::new(
            candidates
                .iter()
                .map(|candidate| (candidate.action, candidate.score))
                .collect(),
            self.config.simulations,
        );

        let mut completed = 0;
        while completed < self.config.simulations {
            let batch_target =
                (self.config.simulations - completed).min(leaf_batch_size as u32) as usize;
            let mut pending = Vec::with_capacity(batch_target);

            for _ in 0..batch_target {
                if completed + pending.len() as u32 >= self.config.simulations {
                    break;
                }
                let Some(root_action) = scheduler.next_action().or_else(|| {
                    scheduler
                        .is_finished()
                        .then(|| self.best_available_root_action(root_index))
                        .flatten()
                }) else {
                    break;
                };
                let mut simulation_state = state.clone();
                match self.select_eval_leaf(root_index, root_action, &mut simulation_state) {
                    PendingGumbelSimulation::NeedsEvaluation {
                        path,
                        state: leaf_state,
                    } => {
                        reserve_path(&mut self.nodes, &path);
                        scheduler.reserve_visit(root_action);
                        pending.push(PendingGumbelLeaf {
                            path,
                            state: leaf_state,
                        });
                    }
                    PendingGumbelSimulation::Terminal { path, value } => {
                        backup_path(&mut self.nodes, &path, value, false);
                        scheduler.reserve_visit(root_action);
                        scheduler.complete_reserved_visits(&root_ranking_scores(
                            &self.nodes[root_index],
                            self.config.c_visit,
                            self.config.c_scale,
                        ));
                        completed += 1;
                    }
                    PendingGumbelSimulation::BlockedPending => break,
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
            for (leaf, (policy_row, value)) in pending
                .into_iter()
                .zip(eval.policies.into_iter().zip(eval.values.into_iter()))
            {
                unreserve_path(&mut self.nodes, &leaf.path);
                let child_index = self.expand_evaluated_node(
                    &leaf.state,
                    &policy_row,
                    value,
                    evaluator_returns_logits,
                )?;
                if let Some((parent_index, edge_index)) = leaf.path.last().copied() {
                    self.nodes[parent_index].edges[edge_index].child = Some(child_index);
                }
                backup_path(&mut self.nodes, &leaf.path, value, true);
                scheduler.complete_reserved_visits(&root_ranking_scores(
                    &self.nodes[root_index],
                    self.config.c_visit,
                    self.config.c_scale,
                ));
                completed += 1;
            }
        }

        let improved = root_improved_policy_target(
            &self.nodes[root_index],
            self.config.c_visit,
            self.config.c_scale,
        );

        Ok(GumbelResult {
            selected_action: improved.selected_action,
            policy_target: improved.policy_target,
            visit_counts: self.nodes[root_index].visit_counts(),
        })
    }

    fn best_root_action(&self, root_index: usize) -> Option<usize> {
        self.best_root_action_matching(root_index, |_| true)
    }

    pub(crate) fn best_available_root_action(&self, root_index: usize) -> Option<usize> {
        self.best_root_action_matching(root_index, |edge| {
            edge.child.is_some() || !edge.pending_evaluation
        })
    }

    fn best_root_action_matching(
        &self,
        root_index: usize,
        predicate: impl Fn(&super::node::GumbelEdge) -> bool,
    ) -> Option<usize> {
        self.nodes[root_index]
            .edges
            .iter()
            .filter(|edge| predicate(edge))
            .max_by(|left, right| {
                let left_score = left.gumbel.unwrap_or(0.0) + left.log_prior;
                let right_score = right.gumbel.unwrap_or(0.0) + right.log_prior;
                left_score
                    .total_cmp(&right_score)
                    .then_with(|| right.action_index().cmp(&left.action_index()))
            })
            .map(|edge| edge.action_index())
    }

    pub(crate) fn expand_evaluated_node(
        &mut self,
        state: &GameState,
        policy_row: &[f32; ACTION_SPACE],
        value: f32,
        policy_is_logits: bool,
    ) -> PyResult<usize> {
        let legal_actions = state.legal_action_indexes();
        let log_priors = if policy_is_logits {
            log_priors_from_logits(&legal_actions, policy_row)?
        } else {
            log_priors_from_priors(&legal_actions, policy_row)?
        };
        let child_index = self.nodes.len();
        self.nodes
            .push(GumbelNode::from_log_priors(state, &log_priors, value));
        Ok(child_index)
    }

    fn run_one_simulation(
        &mut self,
        root_index: usize,
        root_action: usize,
        state: &mut GameState,
    ) -> Option<PathValue> {
        let root_edge_index = self.nodes[root_index].edge_index_for_action(root_action)?;
        let mut node_index = root_index;
        let mut edge_index = root_edge_index;
        let mut path = Vec::new();

        loop {
            let parent_player = self.nodes[node_index].to_play;
            let action = self.nodes[node_index].edges[edge_index].action;
            let outcome = state
                .apply(action)
                .expect("Gumbel search selected an action from legal_action_indexes");
            path.push((node_index, edge_index));

            if let Some(outcome) = outcome {
                return Some(PathValue {
                    path,
                    value: value_for_player(outcome, parent_player),
                    is_leaf: false,
                });
            }

            if let Some(child_index) = self.nodes[node_index].edges[edge_index].child {
                node_index = child_index;
                let action_index = select_inner_action_index(
                    &self.nodes[node_index],
                    self.config.c_visit,
                    self.config.c_scale,
                )?;
                edge_index = self.nodes[node_index].edge_index_for_action(action_index)?;
                continue;
            }

            let child_index = self.nodes.len();
            self.nodes
                .push(GumbelNode::from_uniform_log_priors(state, 0.0));
            self.nodes[node_index].edges[edge_index].child = Some(child_index);
            return Some(PathValue {
                path,
                value: 0.0,
                is_leaf: true,
            });
        }
    }

    pub(crate) fn select_eval_leaf(
        &self,
        root_index: usize,
        root_action: usize,
        state: &mut GameState,
    ) -> PendingGumbelSimulation {
        let Some(root_edge_index) = self.nodes[root_index].edge_index_for_action(root_action)
        else {
            return PendingGumbelSimulation::BlockedPending;
        };
        let mut node_index = root_index;
        let mut edge_index = root_edge_index;
        let mut path = Vec::new();

        loop {
            let parent_player = self.nodes[node_index].to_play;
            let action = self.nodes[node_index].edges[edge_index].action;
            let outcome = state
                .apply(action)
                .expect("Gumbel search selected an action from legal_action_indexes");
            path.push((node_index, edge_index));

            if let Some(outcome) = outcome {
                return PendingGumbelSimulation::Terminal {
                    path,
                    value: value_for_player(outcome, parent_player),
                };
            }

            if let Some(child_index) = self.nodes[node_index].edges[edge_index].child {
                node_index = child_index;
                let Some(action_index) = select_inner_action_index(
                    &self.nodes[node_index],
                    self.config.c_visit,
                    self.config.c_scale,
                ) else {
                    return PendingGumbelSimulation::BlockedPending;
                };
                let Some(next_edge_index) =
                    self.nodes[node_index].edge_index_for_action(action_index)
                else {
                    return PendingGumbelSimulation::BlockedPending;
                };
                edge_index = next_edge_index;
                continue;
            }

            if self.nodes[node_index].edges[edge_index].pending_evaluation {
                return PendingGumbelSimulation::BlockedPending;
            }

            return PendingGumbelSimulation::NeedsEvaluation {
                path,
                state: state.clone(),
            };
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct PathValue {
    path: Vec<(usize, usize)>,
    value: f32,
    is_leaf: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct GumbelEvalBatch {
    pub(crate) policies: Vec<[f32; ACTION_SPACE]>,
    pub(crate) values: Vec<f32>,
}

impl GumbelEvalBatch {
    #[must_use]
    pub(crate) fn new(policies: Vec<[f32; ACTION_SPACE]>, values: Vec<f32>) -> Self {
        Self { policies, values }
    }

    pub(crate) fn validate_len(&self, expected: usize) -> PyResult<()> {
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
struct PendingGumbelLeaf {
    path: Vec<(usize, usize)>,
    state: GameState,
}

#[derive(Clone, Debug)]
pub(crate) enum PendingGumbelSimulation {
    NeedsEvaluation {
        path: Vec<(usize, usize)>,
        state: GameState,
    },
    Terminal {
        path: Vec<(usize, usize)>,
        value: f32,
    },
    BlockedPending,
}

fn select_inner_action_index(node: &GumbelNode, c_visit: f32, c_scale: f32) -> Option<usize> {
    let edges = node
        .edges
        .iter()
        .map(|edge| edge.inner_stats())
        .collect::<Vec<_>>();
    select_inner_action(&edges, node.node_value, c_visit, c_scale)
}

pub(crate) fn root_ranking_scores(
    root: &GumbelNode,
    c_visit: f32,
    c_scale: f32,
) -> Vec<(usize, f32)> {
    root_improved_logits(root, c_visit, c_scale)
}

pub(crate) fn backup_path(
    nodes: &mut [GumbelNode],
    path: &[(usize, usize)],
    value: f32,
    is_leaf: bool,
) {
    let mut edge_value = value;
    for (node_index, edge_index) in path.iter().rev().copied() {
        if is_leaf {
            edge_value = -edge_value;
        }
        nodes[node_index].visit_count = nodes[node_index].visit_count.saturating_add(1);
        nodes[node_index].edges[edge_index].update(edge_value);
        if !is_leaf {
            edge_value = -edge_value;
        }
    }
}

pub(crate) fn reserve_path(nodes: &mut [GumbelNode], path: &[(usize, usize)]) {
    if let Some((node_index, edge_index)) = path.last().copied() {
        nodes[node_index].edges[edge_index].pending_evaluation = true;
    }
}

pub(crate) fn unreserve_path(nodes: &mut [GumbelNode], path: &[(usize, usize)]) {
    if let Some((node_index, edge_index)) = path.last().copied() {
        nodes[node_index].edges[edge_index].pending_evaluation = false;
    }
}

fn value_for_player(outcome: GameOutcome, player: Player) -> f32 {
    if outcome.winner == player { 1.0 } else { -1.0 }
}

pub(crate) fn parse_gumbel_eval_response(response: &Bound<'_, PyAny>) -> PyResult<GumbelEvalBatch> {
    if let Ok((policy_obj, value_obj)) = response.extract::<(Bound<'_, PyAny>, Bound<'_, PyAny>)>()
    {
        if let Ok(eval) = parse_gumbel_eval_response_buffers(&policy_obj, &value_obj) {
            return Ok(eval);
        }
    }

    let (policy_rows, values): (Vec<Vec<f32>>, Vec<f32>) = response.extract()?;
    let mut policies = Vec::with_capacity(policy_rows.len());
    for (row_index, row) in policy_rows.into_iter().enumerate() {
        policies.push(parse_gumbel_policy_row(row, row_index)?);
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err("values must be finite"));
    }
    Ok(GumbelEvalBatch::new(policies, values))
}

fn parse_gumbel_eval_response_buffers(
    policy_obj: &Bound<'_, PyAny>,
    value_obj: &Bound<'_, PyAny>,
) -> PyResult<GumbelEvalBatch> {
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
        if row.iter().any(|logit| !logit.is_finite()) {
            return Err(PyValueError::new_err(format!(
                "policy row {row_index} contains non-finite logits"
            )));
        }
        let mut policy = [0.0; ACTION_SPACE];
        policy.copy_from_slice(row);
        policies.push(policy);
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err("values must be finite"));
    }
    Ok(GumbelEvalBatch::new(policies, values))
}

fn parse_gumbel_policy_row(row: Vec<f32>, row_index: usize) -> PyResult<[f32; ACTION_SPACE]> {
    if row.len() != ACTION_SPACE {
        return Err(PyValueError::new_err(format!(
            "policy row {row_index} must have length {ACTION_SPACE}, got {}",
            row.len()
        )));
    }
    if row.iter().any(|logit| !logit.is_finite()) {
        return Err(PyValueError::new_err("policy logits must be finite"));
    }
    let mut policy = [0.0; ACTION_SPACE];
    policy.copy_from_slice(&row);
    Ok(policy)
}

#[cfg(test)]
mod tests {
    use super::{GumbelEvalBatch, GumbelSearch};
    use crate::{
        game::{ACTION_SPACE, CENTER_INDEX, Cell, GameState, Player, state_with_board},
        gumbel::config::GumbelConfig,
    };

    fn index(row: usize, col: usize) -> usize {
        row * 9 + col
    }

    #[test]
    fn root_seed_advances_per_search_and_resets_when_seed_is_set() {
        let mut search = GumbelSearch::new(GumbelConfig::new(4, 2, 50.0, 1.0, 7));

        assert_eq!(search.seed(), 7);
        assert_eq!(search.next_root_seed(), 7);
        assert_eq!(search.next_root_seed(), 8);

        search.set_seed(42);

        assert_eq!(search.seed(), 42);
        assert_eq!(search.next_root_seed(), 42);
    }

    #[test]
    fn tree_search_expands_child_nodes_and_records_root_visits() {
        let mut search = GumbelSearch::new(GumbelConfig::new(6, 2, 50.0, 1.0, 7));
        let mut logits = [0.0; ACTION_SPACE];
        logits[0] = 3.0;

        let result = search
            .result_from_logits(&GameState::new(), &logits)
            .unwrap();

        assert!(search.nodes.len() > 1);
        assert_eq!(result.visit_counts.iter().sum::<u32>(), 6);
    }

    #[test]
    fn terminal_root_transition_backs_up_parent_player_win() {
        let mut board = [Cell::Empty; ACTION_SPACE - 1];
        board[CENTER_INDEX] = Cell::Neutral;
        board[index(1, 1)] = Cell::Orange;
        board[index(0, 1)] = Cell::Blue;
        board[index(1, 0)] = Cell::Blue;
        board[index(1, 2)] = Cell::Blue;
        let state = state_with_board(board, Player::Blue);
        let winning_action = index(2, 1);
        let mut logits = [0.0; ACTION_SPACE];
        logits[winning_action] = 10.0;

        let mut search = GumbelSearch::new(GumbelConfig::new(4, 1, 50.0, 1.0, 1));
        let result = search.result_from_logits(&state, &logits).unwrap();

        assert_eq!(result.selected_action, Some(winning_action));
        assert_eq!(result.visit_counts[winning_action], 4);
        assert_eq!(search.nodes[0].edges[0].mean_q(), Some(1.0));
    }

    #[test]
    fn evaluator_path_batches_leaf_requests_and_expands_with_logits() {
        let mut search = GumbelSearch::new(GumbelConfig::new(6, 4, 50.0, 1.0, 7));
        let root_logits = [0.0; ACTION_SPACE];
        let mut max_request_len = 0;

        let result = search
            .result_from_logits_with_evaluator(&GameState::new(), &root_logits, 4, |request| {
                max_request_len = max_request_len.max(request.len());
                let mut rows = Vec::with_capacity(request.len());
                let mut values = Vec::with_capacity(request.len());
                for _ in 0..request.len() {
                    let mut logits = [-3.0; ACTION_SPACE];
                    logits[0] = 4.0;
                    logits[1] = 2.0;
                    rows.push(logits);
                    values.push(0.25);
                }
                Ok(GumbelEvalBatch::new(rows, values))
            })
            .unwrap();

        assert!(max_request_len > 1);
        assert_eq!(result.visit_counts.iter().sum::<u32>(), 6);
        assert!(search.nodes.len() > 1);
        assert!(search.nodes.iter().skip(1).any(|node| {
            let log_priors = node
                .edges
                .iter()
                .map(|edge| edge.log_prior)
                .collect::<Vec<_>>();
            log_priors.windows(2).any(|pair| pair[0] != pair[1])
        }));
    }

    #[test]
    fn pending_guard_prevents_duplicate_leaf_evaluation_in_same_wave() {
        let mut search = GumbelSearch::new(GumbelConfig::new(4, 1, 50.0, 1.0, 7));
        let mut root_logits = [0.0; ACTION_SPACE];
        root_logits[0] = 10.0;
        let mut request_lengths = Vec::new();

        search
            .result_from_logits_with_evaluator(&GameState::new(), &root_logits, 4, |request| {
                request_lengths.push(request.len());
                Ok(GumbelEvalBatch::new(
                    vec![[0.0; ACTION_SPACE]; request.len()],
                    vec![0.0; request.len()],
                ))
            })
            .unwrap();

        assert!(request_lengths.iter().all(|length| *length == 1));
    }
}
