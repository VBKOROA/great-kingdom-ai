//! Root sequential halving scheduler for the Gumbel search backend.

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RootHalvingCandidate {
    pub(crate) action: usize,
    pub(crate) ranking_score: f32,
    pub(crate) completed_visits: u32,
    pub(crate) target_visits: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RootSequentialHalving {
    active: Vec<RootHalvingCandidate>,
    round_index: usize,
    simulations: u32,
    completed_total: u32,
    initial_candidate_count: usize,
}

impl RootSequentialHalving {
    #[must_use]
    pub(crate) fn new(candidates: Vec<(usize, f32)>, simulations: u32) -> Self {
        let initial_candidate_count = candidates.len();
        let mut scheduler = Self {
            active: candidates
                .into_iter()
                .map(|(action, ranking_score)| RootHalvingCandidate {
                    action,
                    ranking_score,
                    completed_visits: 0,
                    target_visits: 0,
                })
                .collect(),
            round_index: 0,
            simulations,
            completed_total: 0,
            initial_candidate_count,
        };
        scheduler.assign_round_targets();
        scheduler
    }

    #[must_use]
    #[cfg(test)]
    pub(crate) fn is_finished(&self) -> bool {
        self.active.len() <= 1
    }

    #[must_use]
    pub(crate) fn is_done(&self) -> bool {
        self.completed_total >= self.simulations || self.active.is_empty()
    }

    #[must_use]
    pub(crate) fn selected_action(&self) -> Option<usize> {
        self.active
            .iter()
            .max_by(|left, right| {
                left.ranking_score
                    .total_cmp(&right.ranking_score)
                    .then_with(|| right.action.cmp(&left.action))
            })
            .map(|candidate| candidate.action)
    }

    #[must_use]
    #[cfg(test)]
    pub(crate) fn round_index(&self) -> usize {
        self.round_index
    }

    #[must_use]
    #[cfg(test)]
    pub(crate) fn active_actions(&self) -> Vec<usize> {
        self.active
            .iter()
            .map(|candidate| candidate.action)
            .collect()
    }

    #[must_use]
    pub(crate) fn round_quota(&self) -> u32 {
        Self::quota(
            self.simulations,
            self.initial_candidate_count,
            self.active.len(),
        )
    }

    #[must_use]
    pub(crate) fn next_action(&self) -> Option<usize> {
        if self.is_done() {
            return None;
        }

        self.active
            .iter()
            .filter(|candidate| candidate.completed_visits < candidate.target_visits)
            .min_by(|left, right| {
                left.completed_visits
                    .cmp(&right.completed_visits)
                    .then_with(|| {
                        right
                            .ranking_score
                            .total_cmp(&left.ranking_score)
                            .then_with(|| left.action.cmp(&right.action))
                    })
            })
            .map(|candidate| candidate.action)
    }

    #[cfg(test)]
    pub(crate) fn record_visit(&mut self, action: usize) {
        self.reserve_visit(action);
        self.advance_if_round_complete();
    }

    pub(crate) fn reserve_visit(&mut self, action: usize) {
        if let Some(candidate) = self
            .active
            .iter_mut()
            .find(|candidate| candidate.action == action)
        {
            candidate.completed_visits = candidate.completed_visits.saturating_add(1);
            self.completed_total = self.completed_total.saturating_add(1);
        }
    }

    pub(crate) fn complete_reserved_visits(&mut self, ranking_scores: &[(usize, f32)]) {
        self.update_ranking_scores(ranking_scores);
        self.advance_if_round_complete();
    }

    fn update_ranking_scores(&mut self, ranking_scores: &[(usize, f32)]) {
        for candidate in &mut self.active {
            if let Some((_, score)) = ranking_scores
                .iter()
                .find(|(action, _)| *action == candidate.action)
            {
                candidate.ranking_score = *score;
            }
        }
    }

    fn advance_if_round_complete(&mut self) {
        if self.active.is_empty() || !self.is_round_complete() {
            return;
        }

        if self.active.len() <= 1 {
            self.assign_round_targets();
            return;
        }

        self.active.sort_by(|left, right| {
            right
                .ranking_score
                .total_cmp(&left.ranking_score)
                .then_with(|| left.action.cmp(&right.action))
        });
        let keep_count = self.active.len().div_ceil(2);
        self.active.truncate(keep_count);
        self.round_index += 1;
        self.assign_round_targets();
    }

    fn is_round_complete(&self) -> bool {
        self.active
            .iter()
            .all(|candidate| candidate.completed_visits >= candidate.target_visits)
    }

    fn assign_round_targets(&mut self) {
        if self.active.is_empty() {
            return;
        }
        let remaining = self.simulations.saturating_sub(self.completed_total);
        if remaining == 0 {
            return;
        }

        if self.active.len() == 1 {
            let candidate = &mut self.active[0];
            candidate.target_visits = candidate.completed_visits.saturating_add(remaining);
            return;
        }

        let quota = self.round_quota();
        let mut increments = vec![quota; self.active.len()];

        let round_count = ceil_log2(self.initial_candidate_count).max(1) as u32;
        let base_budget = quota
            .saturating_mul(round_count)
            .saturating_mul(self.active.len() as u32);
        let remainder = self.simulations.saturating_sub(base_budget) as usize;
        if remainder > 0 {
            let mut order = (0..self.active.len()).collect::<Vec<_>>();
            order.sort_by(|left, right| {
                self.active[*right]
                    .ranking_score
                    .total_cmp(&self.active[*left].ranking_score)
                    .then_with(|| self.active[*left].action.cmp(&self.active[*right].action))
            });
            for index in order.into_iter().take(remainder.min(self.active.len())) {
                increments[index] = increments[index].saturating_add(1);
            }
        }

        let mut total_increment = increments.iter().sum::<u32>();
        if total_increment > remaining {
            let mut worst_first = (0..self.active.len()).collect::<Vec<_>>();
            worst_first.sort_by(|left, right| {
                self.active[*left]
                    .ranking_score
                    .total_cmp(&self.active[*right].ranking_score)
                    .then_with(|| self.active[*right].action.cmp(&self.active[*left].action))
            });
            while total_increment > remaining {
                let mut reduced = false;
                for index in &worst_first {
                    if total_increment <= remaining {
                        break;
                    }
                    if increments[*index] > 0 {
                        increments[*index] -= 1;
                        total_increment -= 1;
                        reduced = true;
                    }
                }
                if !reduced {
                    break;
                }
            }
        }

        for (candidate, increment) in self.active.iter_mut().zip(increments) {
            candidate.target_visits = candidate.completed_visits.saturating_add(increment);
        }
    }

    fn quota(simulations: u32, initial_candidate_count: usize, active_count: usize) -> u32 {
        if initial_candidate_count <= 1 || active_count == 0 {
            return 1;
        }
        let round_count = ceil_log2(initial_candidate_count).max(1) as u32;
        (simulations / (round_count * active_count as u32)).max(1)
    }
}

const fn ceil_log2(value: usize) -> usize {
    if value <= 1 {
        return 0;
    }
    usize::BITS as usize - (value - 1).leading_zeros() as usize
}

#[cfg(test)]
mod tests {
    use super::RootSequentialHalving;
    use pretty_assertions::assert_eq;

    #[test]
    fn single_candidate_scheduler_is_finished() {
        let mut scheduler = RootSequentialHalving::new(vec![(3, 1.0)], 8);

        assert!(scheduler.is_finished());
        assert_eq!(scheduler.active_actions(), vec![3]);
        assert_eq!(scheduler.selected_action(), Some(3));

        for _ in 0..8 {
            assert_eq!(scheduler.next_action(), Some(3));
            scheduler.record_visit(3);
        }
        assert!(scheduler.is_done());
        assert_eq!(scheduler.next_action(), None);
    }

    #[test]
    fn round_quota_uses_log2_round_count_and_active_count() {
        let scheduler =
            RootSequentialHalving::new(vec![(0, 0.0), (1, 1.0), (2, 2.0), (3, 3.0)], 16);

        assert_eq!(scheduler.round_quota(), 2);
    }

    #[test]
    fn completed_round_keeps_top_ceil_half_by_ranking_score() {
        let mut scheduler =
            RootSequentialHalving::new(vec![(0, 0.0), (1, 3.0), (2, 2.0), (3, 1.0), (4, 4.0)], 10);

        while scheduler.round_index() == 0 {
            let action = scheduler.next_action().unwrap();
            scheduler.record_visit(action);
        }

        assert_eq!(scheduler.active_actions(), vec![4, 1, 2]);
    }

    #[test]
    fn completed_round_uses_latest_ranking_scores() {
        let mut scheduler =
            RootSequentialHalving::new(vec![(0, 3.0), (1, 2.0), (2, 1.0), (3, 0.0)], 8);

        while scheduler.round_index() == 0 {
            let action = scheduler.next_action().unwrap();
            scheduler.reserve_visit(action);
            scheduler.complete_reserved_visits(&[(0, 0.0), (1, 1.0), (2, 2.0), (3, 3.0)]);
        }

        assert_eq!(scheduler.active_actions(), vec![3, 2]);
    }

    #[test]
    fn odd_active_count_eliminates_lower_half_deterministically() {
        let mut scheduler = RootSequentialHalving::new(vec![(5, 1.0), (2, 1.0), (8, 0.0)], 6);

        while scheduler.round_index() == 0 {
            let action = scheduler.next_action().unwrap();
            scheduler.record_visit(action);
        }

        assert_eq!(scheduler.active_actions(), vec![2, 5]);
    }

    #[test]
    fn remainder_visit_goes_to_best_ranked_candidates_first() {
        let scheduler = RootSequentialHalving::new(vec![(0, 0.0), (1, 2.0), (2, 1.0)], 7);

        assert_eq!(scheduler.active[1].target_visits, 2);
        assert_eq!(scheduler.active[2].target_visits, 1);
        assert_eq!(scheduler.active[0].target_visits, 1);
    }

    #[test]
    fn final_partial_round_spends_remaining_budget_on_best_ranked_candidates() {
        let mut scheduler =
            RootSequentialHalving::new(vec![(0, 0.0), (1, 1.0), (2, 2.0), (3, 3.0)], 5);

        while !scheduler.is_done() {
            let action = scheduler.next_action().unwrap();
            scheduler.record_visit(action);
        }

        assert_eq!(scheduler.active_actions(), vec![3]);
        assert_eq!(scheduler.selected_action(), Some(3));
        assert_eq!(scheduler.active[0].completed_visits, 2);
    }

    #[test]
    fn selected_action_comes_from_surviving_active_candidates() {
        let mut scheduler =
            RootSequentialHalving::new(vec![(0, 100.0), (1, 2.0), (2, 1.0), (3, 0.0)], 4);

        while scheduler.round_index() == 0 {
            let action = scheduler.next_action().unwrap();
            scheduler.reserve_visit(action);
            scheduler.complete_reserved_visits(&[(0, -100.0), (1, 2.0), (2, 1.0), (3, 0.0)]);
        }

        assert_eq!(scheduler.active_actions(), vec![1, 2]);
        assert_eq!(scheduler.selected_action(), Some(1));
    }
}
