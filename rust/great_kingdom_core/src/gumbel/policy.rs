//! Improved policy target helpers for the Gumbel search backend.

use pyo3::{PyResult, exceptions::PyValueError};

use crate::game::ACTION_SPACE;

use super::{
    node::GumbelNode,
    selection::{completed_q_values, prior_probabilities, transformed_completed_q},
};

pub(crate) const PRIOR_EPSILON: f32 = 1.0e-8;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RootImprovedPolicy {
    pub(crate) selected_action: Option<usize>,
    pub(crate) policy_target: [f32; ACTION_SPACE],
}

pub(crate) fn log_priors_from_logits(
    legal_actions: &[usize],
    logits: &[f32],
) -> PyResult<[f32; ACTION_SPACE]> {
    validate_policy_len(logits, "policy_logits")?;
    validate_legal_values(legal_actions, logits, "policy_logits")?;

    let mut log_priors = [f32::NEG_INFINITY; ACTION_SPACE];
    if legal_actions.is_empty() {
        return Ok(log_priors);
    }

    let max_logit = legal_actions
        .iter()
        .map(|action| logits[*action])
        .fold(f32::NEG_INFINITY, f32::max);
    let sum_exp = legal_actions
        .iter()
        .map(|action| (logits[*action] - max_logit).exp())
        .sum::<f32>();
    let log_z = max_logit + sum_exp.ln();

    for action in legal_actions {
        log_priors[*action] = logits[*action] - log_z;
    }
    Ok(log_priors)
}

pub(crate) fn log_priors_from_priors(
    legal_actions: &[usize],
    priors: &[f32],
) -> PyResult<[f32; ACTION_SPACE]> {
    validate_policy_len(priors, "prior")?;
    if priors
        .iter()
        .any(|prior| !prior.is_finite() || *prior < 0.0)
    {
        return Err(PyValueError::new_err(
            "prior values must be finite non-negative values",
        ));
    }

    let mut log_priors = [f32::NEG_INFINITY; ACTION_SPACE];
    if legal_actions.is_empty() {
        return Ok(log_priors);
    }

    let legal_sum = legal_actions
        .iter()
        .map(|action| priors[*action])
        .sum::<f32>();
    if legal_sum <= 0.0 {
        let uniform_log = (1.0 / legal_actions.len() as f32).ln();
        for action in legal_actions {
            log_priors[*action] = uniform_log;
        }
        return Ok(log_priors);
    }

    for action in legal_actions {
        let normalized = (priors[*action] / legal_sum).max(PRIOR_EPSILON);
        log_priors[*action] = normalized.ln();
    }
    Ok(log_priors)
}

#[must_use]
#[allow(clippy::too_many_arguments)]
pub(crate) fn root_improved_policy_target(
    root: &GumbelNode,
    legal_actions: &[usize],
    log_priors: &[f32; ACTION_SPACE],
    c_visit: f32,
    c_scale: f32,
    policy_target_c_visit: f32,
    policy_target_c_scale: f32,
    policy_target_temperature: f32,
) -> RootImprovedPolicy {
    let mut policy_target = [0.0; ACTION_SPACE];
    if legal_actions.is_empty() {
        return RootImprovedPolicy {
            selected_action: None,
            policy_target,
        };
    }

    let selected_action = root_selected_action(root, c_visit, c_scale);

    let improved_logits = root_policy_target_logits(
        root,
        legal_actions,
        log_priors,
        policy_target_c_visit,
        policy_target_c_scale,
    );
    let temperature = policy_target_temperature.max(1.0e-6);
    let max_logit = improved_logits
        .iter()
        .map(|(_, logit)| *logit / temperature)
        .fold(f32::NEG_INFINITY, f32::max);
    let sum_exp = improved_logits
        .iter()
        .map(|(_, logit)| (*logit / temperature - max_logit).exp())
        .sum::<f32>();

    if sum_exp.is_finite() && sum_exp > 0.0 {
        for (action, logit) in improved_logits {
            policy_target[action] = (logit / temperature - max_logit).exp() / sum_exp;
        }
    }

    RootImprovedPolicy {
        selected_action,
        policy_target,
    }
}

#[must_use]
pub(crate) fn root_improved_action_logits(
    root: &GumbelNode,
    c_visit: f32,
    c_scale: f32,
) -> Vec<(usize, f32)> {
    let edge_stats = root
        .edges
        .iter()
        .map(|edge| edge.inner_stats())
        .collect::<Vec<_>>();
    let prior_probs = prior_probabilities(&edge_stats);
    let completed_q = completed_q_values(&edge_stats, &prior_probs, root.node_value);
    let q_bonus = transformed_completed_q(&edge_stats, &completed_q, c_visit, c_scale);
    root.edges
        .iter()
        .zip(q_bonus)
        .map(|(edge, bonus)| {
            let action = edge.action_index();
            let logit = edge.gumbel.unwrap_or(0.0) + edge.log_prior + bonus;
            (action, logit)
        })
        .collect()
}

#[must_use]
pub(crate) fn root_selected_action(root: &GumbelNode, c_visit: f32, c_scale: f32) -> Option<usize> {
    let max_visit_count = root.edges.iter().map(|edge| edge.visit_count).max()?;
    root_improved_action_logits(root, c_visit, c_scale)
        .into_iter()
        .filter(|(action, _)| {
            root.edge_index_for_action(*action)
                .is_some_and(|edge_index| root.edges[edge_index].visit_count == max_visit_count)
        })
        .max_by(|(left_action, left_logit), (right_action, right_logit)| {
            left_logit
                .total_cmp(right_logit)
                .then_with(|| right_action.cmp(left_action))
        })
        .map(|(action, _)| action)
}

#[must_use]
pub(crate) fn root_policy_target_logits(
    root: &GumbelNode,
    legal_actions: &[usize],
    log_priors: &[f32; ACTION_SPACE],
    c_visit: f32,
    c_scale: f32,
) -> Vec<(usize, f32)> {
    let edge_stats = legal_actions
        .iter()
        .map(|action| {
            root.edge_index_for_action(*action).map_or_else(
                || super::selection::InnerEdgeStats::new(*action, log_priors[*action], 0, 0.0),
                |edge_index| root.edges[edge_index].inner_stats(),
            )
        })
        .collect::<Vec<_>>();
    let prior_probs = prior_probabilities(&edge_stats);
    let completed_q = completed_q_values(&edge_stats, &prior_probs, root.node_value);
    let q_bonus = transformed_completed_q(&edge_stats, &completed_q, c_visit, c_scale);

    legal_actions
        .iter()
        .zip(q_bonus)
        .map(|(action, bonus)| (*action, log_priors[*action] + bonus))
        .collect()
}

#[must_use]
pub(crate) fn root_search_value(
    root: &GumbelNode,
    legal_actions: &[usize],
    log_priors: &[f32; ACTION_SPACE],
    policy_target: &[f32; ACTION_SPACE],
) -> f32 {
    let edge_stats = legal_actions
        .iter()
        .map(|action| {
            root.edge_index_for_action(*action).map_or_else(
                || super::selection::InnerEdgeStats::new(*action, log_priors[*action], 0, 0.0),
                |edge_index| root.edges[edge_index].inner_stats(),
            )
        })
        .collect::<Vec<_>>();
    let prior_probs = prior_probabilities(&edge_stats);
    let completed_q = completed_q_values(&edge_stats, &prior_probs, root.node_value);
    let value = legal_actions
        .iter()
        .zip(completed_q)
        .map(|(action, q)| policy_target[*action] * q)
        .sum::<f32>();

    value.clamp(-1.0, 1.0)
}

#[must_use]
pub(crate) fn root_completed_q_for_action(
    root: &GumbelNode,
    legal_actions: &[usize],
    log_priors: &[f32; ACTION_SPACE],
    action: Option<usize>,
) -> Option<f32> {
    let action = action?;
    let edge_stats = legal_actions
        .iter()
        .map(|legal_action| {
            root.edge_index_for_action(*legal_action).map_or_else(
                || {
                    super::selection::InnerEdgeStats::new(
                        *legal_action,
                        log_priors[*legal_action],
                        0,
                        0.0,
                    )
                },
                |edge_index| root.edges[edge_index].inner_stats(),
            )
        })
        .collect::<Vec<_>>();
    let prior_probs = prior_probabilities(&edge_stats);
    let completed_q = completed_q_values(&edge_stats, &prior_probs, root.node_value);
    legal_actions
        .iter()
        .zip(completed_q)
        .find_map(|(legal_action, q)| (*legal_action == action).then_some(q.clamp(-1.0, 1.0)))
}

#[must_use]
pub(crate) fn completed_q_by_action(node: &GumbelNode) -> [f32; ACTION_SPACE] {
    let edge_stats = node
        .edges
        .iter()
        .map(|edge| edge.inner_stats())
        .collect::<Vec<_>>();
    let prior_probs = prior_probabilities(&edge_stats);
    let completed_q = completed_q_values(&edge_stats, &prior_probs, node.node_value);
    let mut by_action = [0.0; ACTION_SPACE];
    for (edge, q) in node.edges.iter().zip(completed_q) {
        by_action[edge.action_index()] = q.clamp(-1.0, 1.0);
    }
    by_action
}

#[must_use]
pub(crate) fn log_priors_by_action(node: &GumbelNode) -> [f32; ACTION_SPACE] {
    let mut by_action = [f32::NEG_INFINITY; ACTION_SPACE];
    for edge in &node.edges {
        by_action[edge.action_index()] = edge.log_prior;
    }
    by_action
}

fn validate_policy_len(row: &[f32], name: &str) -> PyResult<()> {
    if row.len() != ACTION_SPACE {
        return Err(PyValueError::new_err(format!(
            "expected {ACTION_SPACE} {name} values, got {}",
            row.len()
        )));
    }
    Ok(())
}

fn validate_legal_values(legal_actions: &[usize], row: &[f32], name: &str) -> PyResult<()> {
    if legal_actions.iter().any(|action| !row[*action].is_finite()) {
        return Err(PyValueError::new_err(format!(
            "{name} values for legal actions must be finite"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        log_priors_from_logits, log_priors_from_priors, root_improved_policy_target,
        root_search_value, root_selected_action,
    };
    use crate::{
        game::{ACTION_SPACE, CENTER_INDEX, GameState},
        gumbel::{
            node::GumbelNode,
            sampling::{RootCandidate, softmax_candidates},
        },
    };
    use pretty_assertions::assert_eq;

    fn assert_close(left: f32, right: f32) {
        assert!((left - right).abs() < 1.0e-6, "{left} != {right}");
    }

    #[test]
    fn logits_are_log_softmaxed_over_legal_actions_only() {
        let legal = [0, 1];
        let mut logits = [0.0; ACTION_SPACE];
        logits[0] = 1.0;
        logits[1] = 3.0;
        logits[CENTER_INDEX] = f32::NAN;

        let log_priors = log_priors_from_logits(&legal, &logits).unwrap();

        let z = 1.0_f32.exp() + 3.0_f32.exp();
        assert_close(log_priors[0], (1.0_f32.exp() / z).ln());
        assert_close(log_priors[1], (3.0_f32.exp() / z).ln());
        assert_eq!(log_priors[CENTER_INDEX], f32::NEG_INFINITY);
    }

    #[test]
    fn logits_reject_non_finite_legal_values() {
        let legal = [0, 1];
        let mut logits = [0.0; ACTION_SPACE];
        logits[1] = f32::INFINITY;

        assert!(log_priors_from_logits(&legal, &logits).is_err());
    }

    #[test]
    fn priors_renormalize_over_legal_actions_only() {
        let legal = [0, 2];
        let mut priors = [0.0; ACTION_SPACE];
        priors[0] = 1.0;
        priors[1] = 100.0;
        priors[2] = 3.0;

        let log_priors = log_priors_from_priors(&legal, &priors).unwrap();

        assert_eq!(log_priors[0], 0.25_f32.ln());
        assert_eq!(log_priors[2], 0.75_f32.ln());
        assert_eq!(log_priors[1], f32::NEG_INFINITY);
    }

    #[test]
    fn all_zero_legal_priors_fall_back_to_uniform() {
        let legal = [0, 2, 4];
        let priors = [0.0; ACTION_SPACE];

        let log_priors = log_priors_from_priors(&legal, &priors).unwrap();

        let expected = (1.0_f32 / 3.0).ln();
        assert_eq!(log_priors[0], expected);
        assert_eq!(log_priors[2], expected);
        assert_eq!(log_priors[4], expected);
    }

    #[test]
    fn root_improved_policy_uses_completed_q_not_visit_count_normalization() {
        let candidates = [
            RootCandidate {
                action: 0,
                log_prior: 0.5_f32.ln(),
                gumbel: 0.0,
                score: 0.5_f32.ln(),
            },
            RootCandidate {
                action: 1,
                log_prior: 0.5_f32.ln(),
                gumbel: 0.0,
                score: 0.5_f32.ln(),
            },
        ];
        let mut root = GumbelNode::root_from_candidates(&GameState::new(), &candidates, 0.0);
        root.edges[0].visit_count = 10;
        root.edges[0].value_sum = -10.0;
        root.edges[1].visit_count = 1;
        root.edges[1].value_sum = 1.0;

        let legal = [0, 1];
        let mut log_priors = [f32::NEG_INFINITY; ACTION_SPACE];
        log_priors[0] = 0.5_f32.ln();
        log_priors[1] = 0.5_f32.ln();

        let improved =
            root_improved_policy_target(&root, &legal, &log_priors, 1.0, 1.0, 1.0, 1.0, 1.0);
        let prior_only = softmax_candidates(&candidates);

        assert_eq!(improved.selected_action, Some(0));
        assert!(improved.policy_target[1] > improved.policy_target[0]);
        assert!(improved.policy_target[1] > prior_only[1]);
        assert_close(improved.policy_target.iter().sum::<f32>(), 1.0);
    }

    #[test]
    fn root_policy_target_excludes_gumbel_noise() {
        let candidates = [
            RootCandidate {
                action: 0,
                log_prior: 0.5_f32.ln(),
                gumbel: 100.0,
                score: 100.0 + 0.5_f32.ln(),
            },
            RootCandidate {
                action: 1,
                log_prior: 0.5_f32.ln(),
                gumbel: -100.0,
                score: -100.0 + 0.5_f32.ln(),
            },
        ];
        let root = GumbelNode::root_from_candidates(&GameState::new(), &candidates, 0.0);
        let legal = [0, 1];
        let mut log_priors = [f32::NEG_INFINITY; ACTION_SPACE];
        log_priors[0] = 0.5_f32.ln();
        log_priors[1] = 0.5_f32.ln();

        let improved =
            root_improved_policy_target(&root, &legal, &log_priors, 1.0, 1.0, 1.0, 1.0, 1.0);

        assert_eq!(improved.selected_action, Some(0));
        assert_close(improved.policy_target[0], 0.5);
        assert_close(improved.policy_target[1], 0.5);
    }

    #[test]
    fn root_selected_action_uses_best_score_among_most_visited_candidates() {
        let candidates = [
            RootCandidate {
                action: 0,
                log_prior: 0.5_f32.ln(),
                gumbel: 100.0,
                score: 100.0 + 0.5_f32.ln(),
            },
            RootCandidate {
                action: 1,
                log_prior: 0.5_f32.ln(),
                gumbel: 0.0,
                score: 0.5_f32.ln(),
            },
        ];
        let mut root = GumbelNode::root_from_candidates(&GameState::new(), &candidates, 0.0);
        root.edges[0].visit_count = 1;
        root.edges[1].visit_count = 2;

        assert_eq!(root_selected_action(&root, 1.0, 1.0), Some(1));
    }

    #[test]
    fn root_search_value_uses_improved_policy_completed_q_expectation() {
        let candidates = [
            RootCandidate {
                action: 0,
                log_prior: 0.5_f32.ln(),
                gumbel: 0.0,
                score: 0.5_f32.ln(),
            },
            RootCandidate {
                action: 1,
                log_prior: 0.5_f32.ln(),
                gumbel: 0.0,
                score: 0.5_f32.ln(),
            },
        ];
        let mut root = GumbelNode::root_from_candidates(&GameState::new(), &candidates, 1.0);
        root.edges[0].visit_count = 3;
        root.edges[0].value_sum = -3.0;
        root.edges[1].visit_count = 1;
        root.edges[1].value_sum = 1.0;
        let legal = [0, 1];
        let mut log_priors = [f32::NEG_INFINITY; ACTION_SPACE];
        log_priors[0] = 0.5_f32.ln();
        log_priors[1] = 0.5_f32.ln();
        let mut policy_target = [0.0; ACTION_SPACE];
        policy_target[0] = 0.25;
        policy_target[1] = 0.75;

        assert_close(
            root_search_value(&root, &legal, &log_priors, &policy_target),
            0.5,
        );
    }

    #[test]
    fn root_search_value_uses_mixed_value_for_unvisited_completed_q() {
        let candidates = [RootCandidate {
            action: 0,
            log_prior: 0.5_f32.ln(),
            gumbel: 0.0,
            score: 0.5_f32.ln(),
        }];
        let root = GumbelNode::root_from_candidates(&GameState::new(), &candidates, 0.25);
        let legal = [0, 1];
        let mut log_priors = [f32::NEG_INFINITY; ACTION_SPACE];
        log_priors[0] = 0.5_f32.ln();
        log_priors[1] = 0.5_f32.ln();
        let mut policy_target = [0.0; ACTION_SPACE];
        policy_target[0] = 0.25;
        policy_target[1] = 0.75;

        assert_close(
            root_search_value(&root, &legal, &log_priors, &policy_target),
            0.25,
        );
    }

    #[test]
    fn root_policy_target_keeps_unconsidered_legal_actions() {
        let candidates = [RootCandidate {
            action: 0,
            log_prior: 0.5_f32.ln(),
            gumbel: 0.0,
            score: 0.5_f32.ln(),
        }];
        let root = GumbelNode::root_from_candidates(&GameState::new(), &candidates, 0.0);
        let legal = [0, 1];
        let mut log_priors = [f32::NEG_INFINITY; ACTION_SPACE];
        log_priors[0] = 0.5_f32.ln();
        log_priors[1] = 0.5_f32.ln();

        let improved =
            root_improved_policy_target(&root, &legal, &log_priors, 1.0, 1.0, 1.0, 1.0, 1.0);

        assert!(improved.policy_target[1] > 0.0);
        assert_close(improved.policy_target.iter().sum::<f32>(), 1.0);
    }

    #[test]
    fn root_policy_target_uses_root_value_for_unconsidered_actions() {
        let candidates = [RootCandidate {
            action: 0,
            log_prior: 0.5_f32.ln(),
            gumbel: 0.0,
            score: 0.5_f32.ln(),
        }];
        let legal = [0, 1];
        let mut log_priors = [f32::NEG_INFINITY; ACTION_SPACE];
        log_priors[0] = 0.5_f32.ln();
        log_priors[1] = 0.5_f32.ln();
        let mut low_root = GumbelNode::root_from_candidates(&GameState::new(), &candidates, -1.0);
        let mut high_root = GumbelNode::root_from_candidates(&GameState::new(), &candidates, 1.0);
        low_root.edges[0].visit_count = 1;
        low_root.edges[0].value_sum = 0.0;
        high_root.edges[0].visit_count = 1;
        high_root.edges[0].value_sum = 0.0;

        let low_improved =
            root_improved_policy_target(&low_root, &legal, &log_priors, 1.0, 1.0, 1.0, 1.0, 1.0);
        let high_improved =
            root_improved_policy_target(&high_root, &legal, &log_priors, 1.0, 1.0, 1.0, 1.0, 1.0);

        assert!(high_improved.policy_target[1] > low_improved.policy_target[1]);
    }

    #[test]
    fn policy_target_temperature_increases_entropy() {
        let candidates = [
            RootCandidate {
                action: 0,
                log_prior: 0.5_f32.ln(),
                gumbel: 0.0,
                score: 0.5_f32.ln(),
            },
            RootCandidate {
                action: 1,
                log_prior: 0.5_f32.ln(),
                gumbel: 0.0,
                score: 0.5_f32.ln(),
            },
        ];
        let mut root = GumbelNode::root_from_candidates(&GameState::new(), &candidates, 0.0);
        root.edges[0].visit_count = 1;
        root.edges[0].value_sum = -1.0;
        root.edges[1].visit_count = 1;
        root.edges[1].value_sum = 1.0;
        let legal = [0, 1];
        let mut log_priors = [f32::NEG_INFINITY; ACTION_SPACE];
        log_priors[0] = 0.5_f32.ln();
        log_priors[1] = 0.5_f32.ln();

        let baseline =
            root_improved_policy_target(&root, &legal, &log_priors, 4.0, 1.0, 4.0, 1.0, 1.0);
        let softened =
            root_improved_policy_target(&root, &legal, &log_priors, 4.0, 1.0, 4.0, 1.0, 2.0);

        assert_eq!(baseline.selected_action, softened.selected_action);
        assert!(entropy(&softened.policy_target) > entropy(&baseline.policy_target));
        assert!(softened.policy_target[1] < baseline.policy_target[1]);
        assert_close(softened.policy_target.iter().sum::<f32>(), 1.0);
    }

    #[test]
    fn policy_target_scale_can_be_separated_from_selection_scale() {
        let candidates = [
            RootCandidate {
                action: 0,
                log_prior: 0.5_f32.ln(),
                gumbel: 0.0,
                score: 0.5_f32.ln(),
            },
            RootCandidate {
                action: 1,
                log_prior: 0.5_f32.ln(),
                gumbel: 0.0,
                score: 0.5_f32.ln(),
            },
        ];
        let mut root = GumbelNode::root_from_candidates(&GameState::new(), &candidates, 0.0);
        root.edges[0].visit_count = 1;
        root.edges[0].value_sum = -1.0;
        root.edges[1].visit_count = 1;
        root.edges[1].value_sum = 1.0;
        let legal = [0, 1];
        let mut log_priors = [f32::NEG_INFINITY; ACTION_SPACE];
        log_priors[0] = 0.5_f32.ln();
        log_priors[1] = 0.5_f32.ln();

        let selection_scale = (50.0, 1.0);
        let sharp_target = root_improved_policy_target(
            &root,
            &legal,
            &log_priors,
            selection_scale.0,
            selection_scale.1,
            50.0,
            1.0,
            1.0,
        );
        let soft_target = root_improved_policy_target(
            &root,
            &legal,
            &log_priors,
            selection_scale.0,
            selection_scale.1,
            5.0,
            0.25,
            1.0,
        );

        assert_eq!(sharp_target.selected_action, soft_target.selected_action);
        assert!(entropy(&soft_target.policy_target) > entropy(&sharp_target.policy_target));
        assert!(soft_target.policy_target[1] < sharp_target.policy_target[1]);
        assert_close(soft_target.policy_target.iter().sum::<f32>(), 1.0);
    }

    fn entropy(policy: &[f32; ACTION_SPACE]) -> f32 {
        policy
            .iter()
            .filter(|probability| **probability > 0.0)
            .map(|probability| -probability * probability.ln())
            .sum()
    }
}
