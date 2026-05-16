//! Root Gumbel sampling utilities.

use crate::game::ACTION_SPACE;

use super::rng::SplitMix64;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RootCandidate {
    pub(crate) action: usize,
    pub(crate) log_prior: f32,
    pub(crate) gumbel: f32,
    pub(crate) score: f32,
}

pub(crate) fn sample_root_candidates(
    legal_actions: &[usize],
    log_priors: &[f32; ACTION_SPACE],
    max_considered_actions: usize,
    gumbel_scale: f32,
    seed: u64,
) -> Vec<RootCandidate> {
    let candidate_count = legal_actions.len().min(max_considered_actions);
    if candidate_count == 0 {
        return Vec::new();
    }

    let mut rng = SplitMix64::new(seed);
    let mut scored = legal_actions
        .iter()
        .map(|action| {
            let gumbel = gumbel_scale * rng.next_gumbel();
            RootCandidate {
                action: *action,
                log_prior: log_priors[*action],
                gumbel,
                score: log_priors[*action] + gumbel,
            }
        })
        .collect::<Vec<_>>();

    scored.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.action.cmp(&right.action))
    });
    scored.truncate(candidate_count);
    scored
}

#[cfg(test)]
pub(crate) fn softmax_candidates(candidates: &[RootCandidate]) -> [f32; ACTION_SPACE] {
    let mut policy = [0.0; ACTION_SPACE];
    if candidates.is_empty() {
        return policy;
    }

    let max_score = candidates
        .iter()
        .map(|candidate| candidate.score)
        .fold(f32::NEG_INFINITY, f32::max);
    let sum_exp = candidates
        .iter()
        .map(|candidate| (candidate.score - max_score).exp())
        .sum::<f32>();
    for candidate in candidates {
        policy[candidate.action] = (candidate.score - max_score).exp() / sum_exp;
    }
    policy
}

#[cfg(test)]
mod tests {
    use super::sample_root_candidates;
    use crate::game::{ACTION_SPACE, CENTER_INDEX};
    use std::collections::HashSet;

    #[test]
    fn root_sampling_is_deterministic_for_fixed_seed() {
        let legal = [0, 1, 2, 3, 4, 5];
        let log_priors = [0.0; ACTION_SPACE];

        let left = sample_root_candidates(&legal, &log_priors, 4, 1.0, 123);
        let right = sample_root_candidates(&legal, &log_priors, 4, 1.0, 123);

        assert_eq!(left, right);
    }

    #[test]
    fn root_sampling_returns_unique_top_k_candidates() {
        let legal = [0, 1, 2, 3, 4, 5];
        let log_priors = [0.0; ACTION_SPACE];

        let candidates = sample_root_candidates(&legal, &log_priors, 4, 1.0, 123);
        let unique_actions = candidates
            .iter()
            .map(|candidate| candidate.action)
            .collect::<HashSet<_>>();

        assert_eq!(candidates.len(), 4);
        assert_eq!(unique_actions.len(), 4);
    }

    #[test]
    fn root_sampling_clamps_to_legal_and_max_considered_actions() {
        let legal = [0, 1, CENTER_INDEX];
        let log_priors = [0.0; ACTION_SPACE];

        assert_eq!(
            sample_root_candidates(&legal, &log_priors, 16, 1.0, 1).len(),
            3
        );
        assert_eq!(
            sample_root_candidates(&legal, &log_priors, 16, 1.0, 1).len(),
            3
        );
        assert_eq!(
            sample_root_candidates(&[0, 1, 2, 3, 4, 5], &log_priors, 4, 1.0, 1).len(),
            4
        );
    }

    #[test]
    fn zero_gumbel_scale_uses_prior_order_deterministically() {
        let legal = [0, 1, 2];
        let mut log_priors = [f32::NEG_INFINITY; ACTION_SPACE];
        log_priors[0] = 0.2;
        log_priors[1] = 1.0;
        log_priors[2] = 0.5;

        let candidates = sample_root_candidates(&legal, &log_priors, 2, 0.0, 123);

        assert_eq!(
            candidates
                .iter()
                .map(|candidate| candidate.action)
                .collect::<Vec<_>>(),
            vec![1, 2]
        );
        assert!(candidates.iter().all(|candidate| candidate.gumbel == 0.0));
    }
}
