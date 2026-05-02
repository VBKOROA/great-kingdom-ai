//! Internal deterministic action selection for the Gumbel search backend.
#![allow(dead_code)]

const Q_RANGE_EPSILON: f32 = 1.0e-6;
const PRIOR_PROB_EPSILON: f32 = 1.0e-8;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct InnerEdgeStats {
    pub(crate) action: usize,
    pub(crate) log_prior: f32,
    pub(crate) visit_count: u32,
    pub(crate) value_sum: f32,
}

impl InnerEdgeStats {
    #[must_use]
    pub(crate) const fn new(
        action: usize,
        log_prior: f32,
        visit_count: u32,
        value_sum: f32,
    ) -> Self {
        Self {
            action,
            log_prior,
            visit_count,
            value_sum,
        }
    }

    #[must_use]
    pub(crate) fn mean_q(self) -> Option<f32> {
        (self.visit_count > 0).then_some(self.value_sum / self.visit_count as f32)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct InnerPolicyEntry {
    pub(crate) action: usize,
    pub(crate) probability: f32,
    pub(crate) visit_count: u32,
}

#[must_use]
pub(crate) fn inner_improved_policy(
    edges: &[InnerEdgeStats],
    node_raw_value: f32,
    c_visit: f32,
    c_scale: f32,
) -> Vec<InnerPolicyEntry> {
    if edges.is_empty() {
        return Vec::new();
    }

    let prior_probs = prior_probabilities(edges);
    let completed_q = completed_q_values(edges, &prior_probs, node_raw_value);
    let transformed_q = transformed_completed_q(edges, &completed_q, c_visit, c_scale);
    let logits = edges
        .iter()
        .zip(transformed_q)
        .map(|(edge, q_bonus)| (edge.action, edge.visit_count, edge.log_prior + q_bonus))
        .collect::<Vec<_>>();

    let max_logit = logits
        .iter()
        .map(|(_, _, logit)| *logit)
        .fold(f32::NEG_INFINITY, f32::max);
    let sum_exp = logits
        .iter()
        .map(|(_, _, logit)| (*logit - max_logit).exp())
        .sum::<f32>();

    logits
        .into_iter()
        .map(|(action, visit_count, logit)| InnerPolicyEntry {
            action,
            probability: (logit - max_logit).exp() / sum_exp,
            visit_count,
        })
        .collect()
}

#[must_use]
pub(crate) fn select_inner_action(
    edges: &[InnerEdgeStats],
    node_raw_value: f32,
    c_visit: f32,
    c_scale: f32,
) -> Option<usize> {
    let policy = inner_improved_policy(edges, node_raw_value, c_visit, c_scale);
    select_by_visit_ratio(&policy)
}

#[must_use]
pub(crate) fn select_by_visit_ratio(policy: &[InnerPolicyEntry]) -> Option<usize> {
    let total_visits = policy.iter().map(|entry| entry.visit_count).sum::<u32>() as f32;
    policy
        .iter()
        .max_by(|left, right| {
            let left_score = left.probability - left.visit_count as f32 / (1.0 + total_visits);
            let right_score = right.probability - right.visit_count as f32 / (1.0 + total_visits);
            left_score
                .total_cmp(&right_score)
                .then_with(|| right.action.cmp(&left.action))
        })
        .map(|entry| entry.action)
}

fn prior_probabilities(edges: &[InnerEdgeStats]) -> Vec<f32> {
    let max_log_prior = edges
        .iter()
        .map(|edge| edge.log_prior)
        .fold(f32::NEG_INFINITY, f32::max);
    let exp_values = edges
        .iter()
        .map(|edge| (edge.log_prior - max_log_prior).exp())
        .collect::<Vec<_>>();
    let sum_exp = exp_values.iter().sum::<f32>();
    exp_values
        .into_iter()
        .map(|value| value / sum_exp)
        .collect()
}

fn completed_q_values(
    edges: &[InnerEdgeStats],
    prior_probs: &[f32],
    node_raw_value: f32,
) -> Vec<f32> {
    let total_visits = edges.iter().map(|edge| edge.visit_count).sum::<u32>();
    let visited_prior_sum = edges
        .iter()
        .zip(prior_probs.iter())
        .filter(|(edge, _)| edge.visit_count > 0)
        .map(|(_, prior_prob)| prior_prob.max(PRIOR_PROB_EPSILON))
        .sum::<f32>();

    let mixed_value = if total_visits == 0 || visited_prior_sum <= 0.0 {
        node_raw_value
    } else {
        let weighted_q = edges
            .iter()
            .zip(prior_probs.iter())
            .filter_map(|(edge, prior_prob)| {
                edge.mean_q()
                    .map(|q| (prior_prob.max(PRIOR_PROB_EPSILON) / visited_prior_sum) * q)
            })
            .sum::<f32>();
        (node_raw_value + total_visits as f32 * weighted_q) / (total_visits as f32 + 1.0)
    };

    edges
        .iter()
        .map(|edge| edge.mean_q().unwrap_or(mixed_value))
        .collect()
}

fn transformed_completed_q(
    edges: &[InnerEdgeStats],
    completed_q: &[f32],
    c_visit: f32,
    c_scale: f32,
) -> Vec<f32> {
    let q_min = completed_q.iter().copied().fold(f32::INFINITY, f32::min);
    let q_max = completed_q
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let q_range = (q_max - q_min).max(Q_RANGE_EPSILON);
    let max_visit_count = edges.iter().map(|edge| edge.visit_count).max().unwrap_or(0) as f32;
    let visit_scale = (c_visit + max_visit_count) * c_scale;

    completed_q
        .iter()
        .map(|q| visit_scale * ((*q - q_min) / q_range))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{
        InnerEdgeStats, InnerPolicyEntry, completed_q_values, inner_improved_policy,
        select_by_visit_ratio, select_inner_action, transformed_completed_q,
    };

    fn assert_close(left: f32, right: f32) {
        assert!((left - right).abs() < 1.0e-5, "{left} != {right}");
    }

    #[test]
    fn inner_improved_policy_is_softmax_of_prior_and_completed_q_bonus() {
        let edges = [
            InnerEdgeStats::new(0, 0.5_f32.ln(), 1, -1.0),
            InnerEdgeStats::new(1, 0.5_f32.ln(), 1, 1.0),
        ];

        let policy = inner_improved_policy(&edges, 0.0, 1.0, 1.0);

        assert_eq!(policy[0].action, 0);
        assert_eq!(policy[1].action, 1);
        assert!(policy[1].probability > policy[0].probability);
        assert_close(policy[0].probability + policy[1].probability, 1.0);
    }

    #[test]
    fn deterministic_selection_uses_improved_policy_minus_visit_ratio() {
        let policy = [
            InnerPolicyEntry {
                action: 0,
                probability: 0.60,
                visit_count: 9,
            },
            InnerPolicyEntry {
                action: 1,
                probability: 0.40,
                visit_count: 0,
            },
        ];

        assert_eq!(select_by_visit_ratio(&policy), Some(1));
    }

    #[test]
    fn deterministic_selection_tie_breaks_by_lowest_action_index() {
        let policy = [
            InnerPolicyEntry {
                action: 5,
                probability: 0.5,
                visit_count: 0,
            },
            InnerPolicyEntry {
                action: 2,
                probability: 0.5,
                visit_count: 0,
            },
        ];

        assert_eq!(select_by_visit_ratio(&policy), Some(2));
    }

    #[test]
    fn unvisited_edges_are_completed_with_mixed_value_not_zero() {
        let edges = [
            InnerEdgeStats::new(0, 0.5_f32.ln(), 2, 1.0),
            InnerEdgeStats::new(1, 0.5_f32.ln(), 0, 0.0),
        ];
        let priors = [0.5, 0.5];

        let completed = completed_q_values(&edges, &priors, 0.25);

        assert_close(completed[0], 0.5);
        assert_close(completed[1], (0.25 + 2.0 * 0.5) / 3.0);
    }

    #[test]
    fn completed_q_transform_rescales_to_unit_interval_and_visit_scale() {
        let edges = [
            InnerEdgeStats::new(0, 0.5_f32.ln(), 3, -1.0),
            InnerEdgeStats::new(1, 0.5_f32.ln(), 1, 1.0),
        ];
        let completed_q = [-1.0, 1.0];

        let transformed = transformed_completed_q(&edges, &completed_q, 2.0, 1.5);

        assert_close(transformed[0], 0.0);
        assert_close(transformed[1], (2.0 + 3.0) * 1.5);
    }

    #[test]
    fn all_equal_completed_q_transform_stays_finite() {
        let edges = [
            InnerEdgeStats::new(0, 0.5_f32.ln(), 1, 0.25),
            InnerEdgeStats::new(1, 0.5_f32.ln(), 0, 0.0),
        ];
        let completed_q = [0.25, 0.25];

        let transformed = transformed_completed_q(&edges, &completed_q, 50.0, 1.0);

        assert!(transformed.iter().all(|value| value.is_finite()));
        assert_close(transformed[0], 0.0);
        assert_close(transformed[1], 0.0);
    }

    #[test]
    fn select_inner_action_is_deterministic_without_root_gumbel_or_puct_inputs() {
        let edges = [
            InnerEdgeStats::new(7, 0.5_f32.ln(), 10, 10.0),
            InnerEdgeStats::new(3, 0.5_f32.ln(), 0, 0.0),
        ];

        assert_eq!(select_inner_action(&edges, 0.0, 1.0, 0.01), Some(3));
        assert_eq!(select_inner_action(&edges, 0.0, 1.0, 0.01), Some(3));
    }
}
