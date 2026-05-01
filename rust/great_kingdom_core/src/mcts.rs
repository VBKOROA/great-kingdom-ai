use crate::game::{ACTION_SPACE, Action, GameState, Player};

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

#[derive(Clone, Debug)]
pub(crate) struct EdgeStats {
    action: Action,
    prior: f32,
    visit_count: u32,
    value_sum: f32,
}

impl EdgeStats {
    fn new(action: Action, prior: f32) -> Self {
        Self {
            action,
            prior,
            visit_count: 0,
            value_sum: 0.0,
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

    fn visit_counts(&self) -> [u32; ACTION_SPACE] {
        let mut counts = [0; ACTION_SPACE];
        for edge in &self.edges {
            counts[edge.action.to_index()] = edge.visit_count;
        }
        counts
    }
}

impl GameState {
    #[must_use]
    pub(crate) const fn current_player_value(&self) -> Player {
        self.current_player
    }
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
}
