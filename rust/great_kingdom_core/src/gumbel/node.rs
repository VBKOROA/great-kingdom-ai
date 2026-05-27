//! Tree node and edge storage for the Gumbel search backend.

use crate::game::{ACTION_SPACE, Action, GameState, Player};

use super::{sampling::RootCandidate, selection::InnerEdgeStats};

const MISSING_EDGE_INDEX: u16 = u16::MAX;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct GumbelEdge {
    pub(crate) action: Action,
    pub(crate) log_prior: f32,
    pub(crate) gumbel: Option<f32>,
    pub(crate) visit_count: u32,
    pub(crate) value_sum: f32,
    pub(crate) child: Option<usize>,
    pub(crate) pending_evaluation: bool,
    pub(crate) virtual_visit_count: u32,
    pub(crate) virtual_value_sum: f32,
}

impl GumbelEdge {
    #[must_use]
    pub(crate) const fn new(action: Action, log_prior: f32, gumbel: Option<f32>) -> Self {
        Self {
            action,
            log_prior,
            gumbel,
            visit_count: 0,
            value_sum: 0.0,
            child: None,
            pending_evaluation: false,
            virtual_visit_count: 0,
            virtual_value_sum: 0.0,
        }
    }

    pub(crate) fn update(&mut self, value: f32) {
        self.visit_count = self.visit_count.saturating_add(1);
        self.value_sum += value;
    }

    #[must_use]
    #[cfg(test)]
    pub(crate) fn mean_q(&self) -> Option<f32> {
        (self.visit_count > 0).then_some(self.value_sum / self.visit_count as f32)
    }

    #[must_use]
    pub(crate) fn action_index(&self) -> usize {
        self.action.to_index()
    }

    #[must_use]
    pub(crate) fn inner_stats(&self) -> InnerEdgeStats {
        InnerEdgeStats::new(
            self.action_index(),
            self.log_prior,
            self.visit_count,
            self.value_sum,
        )
    }

    #[must_use]
    pub(crate) fn effective_inner_stats(&self) -> InnerEdgeStats {
        InnerEdgeStats::new(
            self.action_index(),
            self.log_prior,
            self.visit_count + self.virtual_visit_count,
            self.value_sum + self.virtual_value_sum,
        )
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct GumbelNode {
    pub(crate) to_play: Player,
    pub(crate) visit_count: u32,
    pub(crate) node_value: f32,
    pub(crate) edges: Vec<GumbelEdge>,
    edge_index_by_action: [u16; ACTION_SPACE],
}

impl GumbelNode {
    #[must_use]
    pub(crate) fn root_from_candidates(
        state: &GameState,
        candidates: &[RootCandidate],
        node_value: f32,
    ) -> Self {
        let edges = candidates
            .iter()
            .filter_map(|candidate| {
                Action::from_index(candidate.action).map(|action| {
                    GumbelEdge::new(action, candidate.log_prior, Some(candidate.gumbel))
                })
            })
            .collect();

        Self::new(state.current_player_value(), node_value, edges)
    }

    #[must_use]
    #[cfg(test)]
    pub(crate) fn from_log_priors(
        state: &GameState,
        log_priors: &[f32; ACTION_SPACE],
        node_value: f32,
    ) -> Self {
        let legal_actions = state.legal_action_indexes();
        Self::from_log_priors_for_actions(state, &legal_actions, log_priors, node_value)
    }

    #[must_use]
    pub(crate) fn from_log_priors_for_actions(
        state: &GameState,
        legal_actions: &[usize],
        log_priors: &[f32; ACTION_SPACE],
        node_value: f32,
    ) -> Self {
        let edges = legal_actions
            .iter()
            .copied()
            .filter_map(Action::from_index)
            .map(|action| {
                let action_index = action.to_index();
                GumbelEdge::new(action, log_priors[action_index], None)
            })
            .collect();

        Self::new(state.current_player_value(), node_value, edges)
    }

    #[must_use]
    pub(crate) fn from_uniform_log_priors(state: &GameState, node_value: f32) -> Self {
        let legal_actions = state.legal_action_indexes();
        let uniform_log_prior = if legal_actions.is_empty() {
            f32::NEG_INFINITY
        } else {
            (1.0 / legal_actions.len() as f32).ln()
        };
        let edges = legal_actions
            .into_iter()
            .filter_map(Action::from_index)
            .map(|action| GumbelEdge::new(action, uniform_log_prior, None))
            .collect();

        Self::new(state.current_player_value(), node_value, edges)
    }

    #[must_use]
    pub(crate) fn edge_index_for_action(&self, action_index: usize) -> Option<usize> {
        self.edge_index_by_action
            .get(action_index)
            .copied()
            .filter(|index| *index != MISSING_EDGE_INDEX)
            .map(usize::from)
    }

    #[must_use]
    pub(crate) fn visit_counts(&self) -> [u32; ACTION_SPACE] {
        let mut counts = [0; ACTION_SPACE];
        for edge in &self.edges {
            counts[edge.action.to_index()] = edge.visit_count;
        }
        counts
    }

    fn new(to_play: Player, node_value: f32, edges: Vec<GumbelEdge>) -> Self {
        let edge_index_by_action = build_edge_index_by_action(&edges);
        Self {
            to_play,
            visit_count: 0,
            node_value,
            edges,
            edge_index_by_action,
        }
    }
}

fn build_edge_index_by_action(edges: &[GumbelEdge]) -> [u16; ACTION_SPACE] {
    let mut edge_index_by_action = [MISSING_EDGE_INDEX; ACTION_SPACE];
    for (edge_index, edge) in edges.iter().enumerate() {
        edge_index_by_action[edge.action.to_index()] =
            u16::try_from(edge_index).expect("Gumbel edge index must fit in u16");
    }
    edge_index_by_action
}

#[cfg(test)]
mod tests {
    use super::GumbelNode;
    use crate::game::{ACTION_SPACE, GameState};

    #[test]
    fn expanded_internal_node_stores_legal_edges_with_log_priors() {
        let state = GameState::new();
        let log_priors = [0.0; ACTION_SPACE];

        let node = GumbelNode::from_log_priors(&state, &log_priors, 0.25);

        assert_eq!(node.node_value, 0.25);
        assert_eq!(node.edges.len(), state.legal_actions().len());
        assert!(node.edge_index_for_action(40).is_none());
        assert!(node.edges.iter().all(|edge| edge.gumbel.is_none()));
    }

    #[test]
    fn node_uses_action_lookup_table_for_edge_indexes() {
        let state = GameState::new();
        let log_priors = [0.0; ACTION_SPACE];
        let legal_actions = [10, 2, 81];

        let node =
            GumbelNode::from_log_priors_for_actions(&state, &legal_actions, &log_priors, 0.0);

        assert_eq!(node.edge_index_for_action(10), Some(0));
        assert_eq!(node.edge_index_for_action(2), Some(1));
        assert_eq!(node.edge_index_for_action(81), Some(2));
        assert_eq!(node.edge_index_for_action(0), None);
        assert_eq!(node.edge_index_for_action(ACTION_SPACE), None);
    }

    #[test]
    fn uniform_internal_node_uses_current_legal_action_set() {
        let state = GameState::new();

        let node = GumbelNode::from_uniform_log_priors(&state, 0.0);

        assert_eq!(node.edges.len(), state.legal_actions().len());
        assert!(node.edges.iter().all(|edge| edge.log_prior.is_finite()));
    }
}
