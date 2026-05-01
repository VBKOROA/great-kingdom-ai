use pyo3::{exceptions::PyValueError, prelude::*, types::PyAny};

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
            if let Some(action_index) = result.selected_action {
                let action = Action::from_index(action_index).ok_or_else(|| {
                    PyValueError::new_err(format!("invalid action index: {action_index}"))
                })?;
                self.states[game_index]
                    .apply(action)
                    .map_err(|err| PyValueError::new_err(format!("invalid action: {err:?}")))?;
            }
            results[game_index] = Some(result);
        }
        Ok(results)
    }

    #[pyo3(signature = (evaluator, leaf_batch_size = 8))]
    pub fn play_turns_with_evaluator(
        &mut self,
        evaluator: &Bound<'_, PyAny>,
        leaf_batch_size: usize,
    ) -> PyResult<Vec<Option<MctsResult>>> {
        if leaf_batch_size == 0 {
            return Err(PyValueError::new_err("leaf_batch_size must be positive"));
        }

        let active_indexes = self.active_indexes();
        let mut results = vec![None; self.states.len()];
        for game_index in active_indexes {
            let result = self.searches[game_index].run_with_leaf_evaluator(
                &self.states[game_index],
                leaf_batch_size,
                |request| {
                    let response = evaluator.call1((request,))?;
                    parse_eval_response(&response)
                },
            )?;
            if let Some(action_index) = result.selected_action {
                let action = Action::from_index(action_index).ok_or_else(|| {
                    PyValueError::new_err(format!("invalid action index: {action_index}"))
                })?;
                self.states[game_index]
                    .apply(action)
                    .map_err(|err| PyValueError::new_err(format!("invalid action: {err:?}")))?;
            }
            results[game_index] = Some(result);
        }
        Ok(results)
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
            let eval = evaluator(EvalRequest::new(request_states))?;
            eval.validate_len(pending.len())?;
            for (leaf, (policy, value)) in pending
                .into_iter()
                .zip(eval.policies.into_iter().zip(eval.values.into_iter()))
            {
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
                let left_score = puct_score(node.visit_count, left, self.config.c_puct);
                let right_score = puct_score(node.visit_count, right, self.config.c_puct);
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

fn parse_eval_response(response: &Bound<'_, PyAny>) -> PyResult<EvalBatch> {
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
