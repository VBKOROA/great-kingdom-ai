use pyo3::{exceptions::PyValueError, prelude::*};

use super::{
    node::GumbelNode,
    policy::{
        log_priors_from_logits, log_priors_from_priors, root_completed_q_for_action,
        root_improved_policy_target, root_search_value,
    },
    result::GumbelResult,
    sampling::sample_root_candidates,
    search::{GumbelSearch, root_ranking_scores, selected_child_stats},
    sequential_halving::RootSequentialHalving,
};
use crate::game::{ACTION_SPACE, GameState};

pub(crate) struct RootSearchState {
    pub(crate) root_index: usize,
    pub(crate) legal_actions: Vec<usize>,
    pub(crate) log_priors: [f32; ACTION_SPACE],
    pub(crate) scheduler: RootSequentialHalving,
    pub(crate) completed: u32,
}

impl RootSearchState {
    #[must_use]
    pub(crate) fn has_remaining(&self, search: &GumbelSearch) -> bool {
        self.completed < search.config.simulations
    }

    pub(crate) fn complete_reserved_visits(&mut self, search: &GumbelSearch) {
        self.scheduler
            .complete_reserved_visits(&root_ranking_scores(
                &search.nodes[self.root_index],
                search.config.c_visit,
                search.config.c_scale,
            ));
    }

    #[must_use]
    pub(crate) fn finish(&self, search: &GumbelSearch) -> GumbelResult {
        finish_root_result(
            search,
            self.root_index,
            &self.legal_actions,
            &self.log_priors,
        )
    }
}

pub(crate) fn empty_result() -> GumbelResult {
    GumbelResult {
        selected_action: None,
        selected_action_q: None,
        selected_child_visit_counts: [0; ACTION_SPACE],
        selected_child_completed_q: [0.0; ACTION_SPACE],
        selected_child_log_priors: [f32::NEG_INFINITY; ACTION_SPACE],
        policy_target: [0.0; ACTION_SPACE],
        visit_counts: [0; ACTION_SPACE],
        root_value: 0.0,
    }
}

pub(crate) fn start_root_search(
    search: &mut GumbelSearch,
    state: &GameState,
    row: &[f32],
    row_is_logits: bool,
    root_value: f32,
) -> PyResult<Option<RootSearchState>> {
    if row.len() != ACTION_SPACE {
        return Err(PyValueError::new_err(format!(
            "expected {ACTION_SPACE} policy values, got {}",
            row.len()
        )));
    }

    let legal_actions = state.legal_action_indexes();
    if legal_actions.is_empty() || state.is_terminal() {
        return Ok(None);
    }

    let log_priors = if row_is_logits {
        log_priors_from_logits(&legal_actions, row)?
    } else {
        log_priors_from_priors(&legal_actions, row)?
    };
    let candidates = sample_root_candidates(
        &legal_actions,
        &log_priors,
        search.config.max_considered_actions,
        search.config.gumbel_scale,
        search.next_root_seed(),
    );
    if candidates.is_empty() {
        return Ok(None);
    }

    search.nodes.clear();
    let root_index = search.nodes.len();
    search.nodes.push(GumbelNode::root_from_candidates(
        state,
        &candidates,
        root_value,
    ));
    let scheduler = RootSequentialHalving::new(
        candidates
            .iter()
            .map(|candidate| (candidate.action, candidate.score))
            .collect(),
        search.config.simulations,
    );

    Ok(Some(RootSearchState {
        root_index,
        legal_actions,
        log_priors,
        scheduler,
        completed: 0,
    }))
}

pub(crate) fn finish_root_result(
    search: &GumbelSearch,
    root_index: usize,
    legal_actions: &[usize],
    log_priors: &[f32; ACTION_SPACE],
) -> GumbelResult {
    let root = &search.nodes[root_index];
    let improved = root_improved_policy_target(
        root,
        legal_actions,
        log_priors,
        search.config.c_visit,
        search.config.c_scale,
        search.config.policy_target_c_visit,
        search.config.policy_target_c_scale,
        search.config.policy_target_temperature,
    );
    let (selected_child_visit_counts, selected_child_completed_q, selected_child_log_priors) =
        selected_child_stats(&search.nodes, root_index, improved.selected_action);
    GumbelResult {
        selected_action: improved.selected_action,
        selected_action_q: root_completed_q_for_action(
            root,
            legal_actions,
            log_priors,
            improved.selected_action,
        ),
        selected_child_visit_counts,
        selected_child_completed_q,
        selected_child_log_priors,
        policy_target: improved.policy_target,
        visit_counts: root.visit_counts(),
        root_value: root_search_value(root, legal_actions, log_priors, &improved.policy_target),
    }
}
