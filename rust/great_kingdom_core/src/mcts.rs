use pyo3::{exceptions::PyValueError, prelude::*};

use crate::game::{ACTION_SPACE, Action, GameOutcome, GameState, Player};

#[pyclass]
#[derive(Clone, Debug)]
pub struct EvalRequest {
    states: Vec<GameState>,
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
    pub fn legal_masks(&self) -> Vec<Vec<bool>> {
        self.states
            .iter()
            .map(GameState::legal_mask)
            .collect::<Vec<_>>()
    }
}

impl EvalRequest {
    #[must_use]
    pub(crate) fn new(states: Vec<GameState>) -> Self {
        Self { states }
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

#[pymethods]
impl MctsSearch {
    #[new]
    #[pyo3(signature = (simulations = 50, c_puct = 1.5))]
    pub fn py_new(simulations: u32, c_puct: f32) -> PyResult<Self> {
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
                let left_score = puct_score(node.visit_count, left, self.config.c_puct);
                let right_score = puct_score(node.visit_count, right, self.config.c_puct);
                left_score
                    .partial_cmp(&right_score)
                    .expect("PUCT score must be finite")
            })
            .map(|(index, _)| index)
            .expect("expanded MCTS node must contain at least one edge")
    }
}

#[derive(Clone, Debug)]
pub(crate) struct EdgeStats {
    action: Action,
    prior: f32,
    visit_count: u32,
    value_sum: f32,
    child: Option<usize>,
}

impl EdgeStats {
    fn new(action: Action, prior: f32) -> Self {
        Self {
            action,
            prior,
            visit_count: 0,
            value_sum: 0.0,
            child: None,
        }
    }

    fn update(&mut self, value: f32) {
        self.visit_count += 1;
        self.value_sum += value;
    }

    fn mean_value(&self) -> f32 {
        if self.visit_count == 0 {
            0.0
        } else {
            self.value_sum / self.visit_count as f32
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Node {
    to_play: Player,
    visit_count: u32,
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
            .max_by_key(|edge| edge.visit_count)
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
    edge.mean_value()
        + c_puct * edge.prior * exploration_visits.sqrt() / (1.0 + edge.visit_count as f32)
}

fn value_for_player(outcome: GameOutcome, player: Player) -> f32 {
    if outcome.winner == player { 1.0 } else { -1.0 }
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
    fn puct_selection_reflects_prior_and_visit_count() {
        let config = MctsConfig::new(1, 2.0);
        let search = MctsSearch::new(config);
        let node = Node {
            to_play: Player::Blue,
            visit_count: 16,
            edges: vec![
                EdgeStats {
                    action: Action::from_index(0).unwrap(),
                    prior: 0.9,
                    visit_count: 0,
                    value_sum: 0.0,
                    child: None,
                },
                EdgeStats {
                    action: Action::Pass,
                    prior: 0.1,
                    visit_count: 10,
                    value_sum: 8.0,
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
}
