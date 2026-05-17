use std::{env, iter::Sum, ops::AddAssign, time::Duration};

use super::search::GumbelSelectTrace;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct SelectWaveDetail {
    pub(crate) simulations: u64,
    pub(crate) pending: u64,
    pub(crate) terminal: u64,
    pub(crate) blocked: u64,
    pub(crate) scheduler_next_elapsed: Duration,
    pub(crate) state_clone_elapsed: Duration,
    pub(crate) search_elapsed: Duration,
    pub(crate) reserve_path_elapsed: Duration,
    pub(crate) scheduler_reserve_elapsed: Duration,
    pub(crate) terminal_backup_elapsed: Duration,
    pub(crate) complete_reserved_elapsed: Duration,
    pub(crate) search_trace: GumbelSelectTrace,
}

impl AddAssign for SelectWaveDetail {
    fn add_assign(&mut self, rhs: Self) {
        self.simulations = self.simulations.saturating_add(rhs.simulations);
        self.pending = self.pending.saturating_add(rhs.pending);
        self.terminal = self.terminal.saturating_add(rhs.terminal);
        self.blocked = self.blocked.saturating_add(rhs.blocked);
        self.scheduler_next_elapsed += rhs.scheduler_next_elapsed;
        self.state_clone_elapsed += rhs.state_clone_elapsed;
        self.search_elapsed += rhs.search_elapsed;
        self.reserve_path_elapsed += rhs.reserve_path_elapsed;
        self.scheduler_reserve_elapsed += rhs.scheduler_reserve_elapsed;
        self.terminal_backup_elapsed += rhs.terminal_backup_elapsed;
        self.complete_reserved_elapsed += rhs.complete_reserved_elapsed;
        self.search_trace += rhs.search_trace;
    }
}

impl Sum for SelectWaveDetail {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut total = Self::default();
        for detail in iter {
            total += detail;
        }
        total
    }
}

#[derive(Clone, Copy)]
pub(crate) struct GumbelBatchProfile {
    enabled: bool,
    select_detail_enabled: bool,
    interval: u64,
    name: &'static str,
}

pub(crate) struct GumbelBatchWaveProfile {
    pub(crate) wave: u64,
    pub(crate) active_games: usize,
    pub(crate) leaves: usize,
    pub(crate) select_elapsed: Duration,
    pub(crate) flatten_elapsed: Duration,
    pub(crate) request_elapsed: Duration,
    pub(crate) eval_elapsed: Duration,
    pub(crate) parse_elapsed: Duration,
    pub(crate) backup_elapsed: Duration,
    pub(crate) select_detail: SelectWaveDetail,
}

impl GumbelBatchProfile {
    pub(crate) fn new(name: &'static str) -> Self {
        Self {
            enabled: env_flag("GKA_GUMBEL_PROFILE"),
            select_detail_enabled: env_flag("GKA_GUMBEL_SELECT_DETAIL"),
            interval: env::var("GKA_GUMBEL_PROFILE_INTERVAL")
                .ok()
                .and_then(|value| value.parse::<u64>().ok())
                .filter(|value| *value > 0)
                .unwrap_or(1),
            name,
        }
    }

    pub(crate) const fn select_detail_enabled(&self) -> bool {
        self.enabled && self.select_detail_enabled
    }

    pub(crate) fn root(&self, active_games: usize, elapsed: Duration) {
        if !self.enabled {
            return;
        }
        eprintln!(
            "[gka-gumbel-profile] fn={} root active_games={} rayon_threads={} init={:.3}s",
            self.name,
            active_games,
            rayon::current_num_threads(),
            elapsed.as_secs_f64(),
        );
    }

    pub(crate) fn empty_wave(
        &self,
        wave: u64,
        active_games: usize,
        select_elapsed: Duration,
        flatten_elapsed: Duration,
        select_detail: SelectWaveDetail,
    ) {
        if !self.enabled || wave % self.interval != 0 {
            return;
        }
        eprintln!(
            "[gka-gumbel-profile] fn={} wave={} empty active_games={} select={:.3}s flatten={:.3}s",
            self.name,
            wave,
            active_games,
            select_elapsed.as_secs_f64(),
            flatten_elapsed.as_secs_f64(),
        );
        self.select_detail(wave, active_games, 0, select_detail);
    }

    pub(crate) fn wave(&self, profile: GumbelBatchWaveProfile) {
        if !self.enabled || profile.wave % self.interval != 0 {
            return;
        }
        let total = profile.select_elapsed
            + profile.flatten_elapsed
            + profile.request_elapsed
            + profile.eval_elapsed
            + profile.parse_elapsed
            + profile.backup_elapsed;
        eprintln!(
            "[gka-gumbel-profile] fn={} wave={} active_games={} leaves={} select={:.3}s flatten={:.3}s request={:.3}s eval_call={:.3}s parse={:.3}s backup={:.3}s total={:.3}s",
            self.name,
            profile.wave,
            profile.active_games,
            profile.leaves,
            profile.select_elapsed.as_secs_f64(),
            profile.flatten_elapsed.as_secs_f64(),
            profile.request_elapsed.as_secs_f64(),
            profile.eval_elapsed.as_secs_f64(),
            profile.parse_elapsed.as_secs_f64(),
            profile.backup_elapsed.as_secs_f64(),
            total.as_secs_f64(),
        );
        self.select_detail(
            profile.wave,
            profile.active_games,
            profile.leaves,
            profile.select_detail,
        );
    }

    fn select_detail(
        &self,
        wave: u64,
        active_games: usize,
        leaves: usize,
        detail: SelectWaveDetail,
    ) {
        if !self.select_detail_enabled() {
            return;
        }
        eprintln!(
            "[gka-gumbel-select] fn={} wave={} active_games={} leaves={} sims={} pending={} terminal={} blocked={} scheduler_next={:.3}s state_clone={:.3}s search={:.3}s root_lookup={:.3}s apply={:.3}s path_push={:.3}s inner_select={:.3}s edge_lookup={:.3}s leaf_clone={:.3}s reserve_path={:.3}s scheduler_reserve={:.3}s terminal_backup={:.3}s complete_reserved={:.3}s search_steps={}",
            self.name,
            wave,
            active_games,
            leaves,
            detail.simulations,
            detail.pending,
            detail.terminal,
            detail.blocked,
            detail.scheduler_next_elapsed.as_secs_f64(),
            detail.state_clone_elapsed.as_secs_f64(),
            detail.search_elapsed.as_secs_f64(),
            detail.search_trace.root_lookup_elapsed.as_secs_f64(),
            detail.search_trace.apply_elapsed.as_secs_f64(),
            detail.search_trace.path_push_elapsed.as_secs_f64(),
            detail.search_trace.inner_select_elapsed.as_secs_f64(),
            detail.search_trace.edge_lookup_elapsed.as_secs_f64(),
            detail.search_trace.leaf_clone_elapsed.as_secs_f64(),
            detail.reserve_path_elapsed.as_secs_f64(),
            detail.scheduler_reserve_elapsed.as_secs_f64(),
            detail.terminal_backup_elapsed.as_secs_f64(),
            detail.complete_reserved_elapsed.as_secs_f64(),
            detail.search_trace.steps,
        );
    }
}

fn env_flag(name: &str) -> bool {
    !matches!(
        env::var(name).as_deref(),
        Err(_) | Ok("") | Ok("0") | Ok("false") | Ok("False") | Ok("no") | Ok("No")
    )
}
