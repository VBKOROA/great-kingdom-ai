//! Root sequential halving scheduler for the Gumbel search backend.

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RootHalvingCandidate {
    pub(crate) action: usize,
    pub(crate) ranking_score: f32,
    pub(crate) completed_visits: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RootSequentialHalving {
    active: Vec<RootHalvingCandidate>,
    simulations: u32,
    completed_total: u32,
    considered_visits: Vec<u32>,
}

impl RootSequentialHalving {
    #[must_use]
    pub(crate) fn new(candidates: Vec<(usize, f32)>, simulations: u32) -> Self {
        let considered_visits = sequence_of_considered_visits(candidates.len(), simulations);
        Self {
            active: candidates
                .into_iter()
                .map(|(action, ranking_score)| RootHalvingCandidate {
                    action,
                    ranking_score,
                    completed_visits: 0,
                })
                .collect(),
            simulations,
            completed_total: 0,
            considered_visits,
        }
    }

    #[must_use]
    #[cfg(test)]
    pub(crate) fn is_finished(&self) -> bool {
        self.is_done()
    }

    #[must_use]
    pub(crate) fn is_done(&self) -> bool {
        self.completed_total >= self.simulations || self.active.is_empty()
    }

    #[must_use]
    #[cfg(test)]
    pub(crate) fn selected_action(&self) -> Option<usize> {
        let max_visit_count = self
            .active
            .iter()
            .map(|candidate| candidate.completed_visits)
            .max()?;
        self.active
            .iter()
            .filter(|candidate| candidate.completed_visits == max_visit_count)
            .max_by(|left, right| {
                left.ranking_score
                    .total_cmp(&right.ranking_score)
                    .then_with(|| right.action.cmp(&left.action))
            })
            .map(|candidate| candidate.action)
    }

    #[must_use]
    #[cfg(test)]
    pub(crate) fn completed_visits_by_action(&self) -> Vec<(usize, u32)> {
        self.active
            .iter()
            .map(|candidate| (candidate.action, candidate.completed_visits))
            .collect()
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
    #[cfg(test)]
    pub(crate) fn round_quota(&self) -> u32 {
        if self.active.is_empty() {
            return 0;
        }
        let round_count = ceil_log2(self.active.len()).max(1) as u32;
        (self.simulations / (round_count * self.active.len() as u32)).max(1)
    }

    #[must_use]
    pub(crate) fn next_action(&self) -> Option<usize> {
        if self.is_done() {
            return None;
        }

        let considered_visit = self
            .considered_visits
            .get(self.completed_total as usize)
            .copied()?;
        self.active
            .iter()
            .filter(|candidate| candidate.completed_visits == considered_visit)
            .max_by(|left, right| {
                left.ranking_score
                    .total_cmp(&right.ranking_score)
                    .then_with(|| right.action.cmp(&left.action))
            })
            .map(|candidate| candidate.action)
    }

    #[cfg(test)]
    pub(crate) fn record_visit(&mut self, action: usize) {
        self.reserve_visit(action);
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
}

const fn ceil_log2(value: usize) -> usize {
    if value <= 1 {
        return 0;
    }
    usize::BITS as usize - (value - 1).leading_zeros() as usize
}

fn sequence_of_considered_visits(max_considered_actions: usize, simulations: u32) -> Vec<u32> {
    if max_considered_actions == 0 || simulations == 0 {
        return Vec::new();
    }
    if max_considered_actions <= 1 {
        return (0..simulations).collect();
    }

    let round_count = ceil_log2(max_considered_actions).max(1) as u32;
    let mut sequence = Vec::with_capacity(simulations as usize);
    let mut visits = vec![0_u32; max_considered_actions];
    let mut considered = max_considered_actions;

    while sequence.len() < simulations as usize {
        let extra_visits = (simulations / (round_count * considered as u32)).max(1);
        for _ in 0..extra_visits {
            for visit in visits.iter().take(considered) {
                if sequence.len() >= simulations as usize {
                    break;
                }
                sequence.push(*visit);
            }
            for visit in visits.iter_mut().take(considered) {
                *visit = visit.saturating_add(1);
            }
        }
        considered = 2.max(considered / 2);
    }

    sequence
}

#[cfg(test)]
mod tests {
    use super::RootSequentialHalving;
    use pretty_assertions::assert_eq;

    #[test]
    fn single_candidate_scheduler_is_finished() {
        let mut scheduler = RootSequentialHalving::new(vec![(3, 1.0)], 8);

        assert!(!scheduler.is_finished());
        assert_eq!(scheduler.active_actions(), vec![3]);

        for _ in 0..8 {
            assert_eq!(scheduler.next_action(), Some(3));
            scheduler.record_visit(3);
        }
        assert!(scheduler.is_done());
        assert_eq!(scheduler.next_action(), None);
    }

    #[test]
    fn round_quota_matches_mctx_extra_visit_formula() {
        let scheduler =
            RootSequentialHalving::new(vec![(0, 0.0), (1, 1.0), (2, 2.0), (3, 3.0)], 16);

        assert_eq!(scheduler.round_quota(), 2);
    }

    #[test]
    fn visits_follow_mctx_considered_visit_sequence_for_odd_candidates() {
        let mut scheduler =
            RootSequentialHalving::new(vec![(0, 0.0), (1, 3.0), (2, 2.0), (3, 1.0), (4, 4.0)], 10);
        let mut actions = Vec::new();

        while !scheduler.is_done() {
            let action = scheduler.next_action().unwrap();
            actions.push(action);
            scheduler.record_visit(action);
        }

        assert_eq!(actions, vec![4, 1, 2, 3, 0, 4, 1, 4, 1, 4]);
        assert_eq!(
            scheduler.completed_visits_by_action(),
            vec![(0, 1), (1, 3), (2, 1), (3, 1), (4, 4)]
        );
    }

    #[test]
    fn considered_visit_selection_uses_latest_ranking_scores() {
        let mut scheduler =
            RootSequentialHalving::new(vec![(0, 3.0), (1, 2.0), (2, 1.0), (3, 0.0)], 8);

        assert_eq!(scheduler.next_action(), Some(0));
        scheduler.reserve_visit(0);
        scheduler.complete_reserved_visits(&[(0, 0.0), (1, 1.0), (2, 2.0), (3, 3.0)]);

        assert_eq!(scheduler.next_action(), Some(3));
    }

    #[test]
    fn equal_scores_tie_break_by_lowest_action_index() {
        let scheduler = RootSequentialHalving::new(vec![(5, 1.0), (2, 1.0), (8, 0.0)], 6);

        assert_eq!(scheduler.next_action(), Some(2));
    }

    #[test]
    fn selected_action_comes_from_most_visited_candidates() {
        let mut scheduler = RootSequentialHalving::new(vec![(0, 100.0), (1, 2.0), (2, 1.0)], 7);

        scheduler.reserve_visit(0);
        scheduler.reserve_visit(1);
        scheduler.reserve_visit(2);
        scheduler.reserve_visit(0);
        scheduler.reserve_visit(1);
        scheduler.reserve_visit(0);
        scheduler.reserve_visit(1);

        assert_eq!(
            scheduler.completed_visits_by_action(),
            vec![(0, 3), (1, 3), (2, 1)]
        );
        assert_eq!(scheduler.selected_action(), Some(0));
    }

    #[test]
    fn selected_action_ignores_higher_score_with_fewer_visits() {
        let mut scheduler = RootSequentialHalving::new(vec![(0, 100.0), (1, 2.0), (2, 1.0)], 7);

        scheduler.reserve_visit(0);
        scheduler.reserve_visit(1);
        scheduler.reserve_visit(2);
        scheduler.reserve_visit(1);
        scheduler.reserve_visit(2);

        assert_eq!(scheduler.selected_action(), Some(1));
    }
}
