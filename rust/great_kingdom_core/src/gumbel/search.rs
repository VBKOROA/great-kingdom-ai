use pyo3::{exceptions::PyValueError, prelude::*};

use super::{
    config::GumbelConfig,
    node::GumbelNode,
    policy::{log_priors_from_logits, log_priors_from_priors},
    result::GumbelResult,
    sampling::{RootCandidate, sample_root_candidates, softmax_candidates},
    selection::select_inner_action,
    sequential_halving::RootSequentialHalving,
};
use crate::game::{ACTION_SPACE, GameOutcome, GameState, Player};

#[pyclass]
#[derive(Clone, Debug)]
pub struct GumbelSearch {
    config: GumbelConfig,
    nodes: Vec<GumbelNode>,
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
        let _ = evaluator;
        if leaf_batch_size == 0 {
            return Err(PyValueError::new_err("leaf_batch_size must be positive"));
        }
        self.search_with_logits(state, policy_logits)
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
        let _ = evaluator;
        if leaf_batch_size == 0 {
            return Err(PyValueError::new_err("leaf_batch_size must be positive"));
        }
        self.search_with_priors(state, priors)
    }
}

impl GumbelSearch {
    #[must_use]
    pub const fn new(config: GumbelConfig) -> Self {
        Self {
            config,
            nodes: Vec::new(),
        }
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
            self.config.seed,
        );
        self.run_tree_search(state, &candidates)
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

    fn run_tree_search(
        &mut self,
        state: &GameState,
        candidates: &[RootCandidate],
    ) -> GumbelResult {
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
            let Some(root_action) = scheduler
                .next_action()
                .or_else(|| self.best_root_action(root_index))
            else {
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
            scheduler.record_visit(root_action);
        }

        let selected_action = candidates
            .iter()
            .max_by(|left, right| {
                left.score
                    .total_cmp(&right.score)
                    .then_with(|| right.action.cmp(&left.action))
            })
            .map(|candidate| candidate.action);

        GumbelResult {
            selected_action,
            policy_target: softmax_candidates(candidates),
            visit_counts: self.nodes[root_index].visit_counts(),
        }
    }

    fn best_root_action(&self, root_index: usize) -> Option<usize> {
        self.nodes[root_index]
            .edges
            .iter()
            .max_by(|left, right| {
                let left_score = left.gumbel.unwrap_or(0.0) + left.log_prior;
                let right_score = right.gumbel.unwrap_or(0.0) + right.log_prior;
                left_score
                    .total_cmp(&right_score)
                    .then_with(|| right.action_index().cmp(&left.action_index()))
            })
            .map(|edge| edge.action_index())
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
}

#[derive(Clone, Debug, PartialEq)]
struct PathValue {
    path: Vec<(usize, usize)>,
    value: f32,
    is_leaf: bool,
}

fn select_inner_action_index(node: &GumbelNode, c_visit: f32, c_scale: f32) -> Option<usize> {
    let edges = node
        .edges
        .iter()
        .map(|edge| edge.inner_stats())
        .collect::<Vec<_>>();
    select_inner_action(&edges, node.node_value, c_visit, c_scale)
}

fn backup_path(nodes: &mut [GumbelNode], path: &[(usize, usize)], value: f32, is_leaf: bool) {
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

fn value_for_player(outcome: GameOutcome, player: Player) -> f32 {
    if outcome.winner == player { 1.0 } else { -1.0 }
}

#[cfg(test)]
mod tests {
    use super::GumbelSearch;
    use crate::{
        game::{ACTION_SPACE, CENTER_INDEX, Cell, GameState, Player, state_with_board},
        gumbel::config::GumbelConfig,
    };

    fn index(row: usize, col: usize) -> usize {
        row * 9 + col
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
}
