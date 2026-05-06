use std::{
    collections::VecDeque,
    env,
    sync::{
        OnceLock,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use crate::game::{
    Action, BOARD_CELLS, BOARD_SIZE, CASTLES_PER_PLAYER, Cell, GameEndReason, GameOutcome,
    GameState, InvalidAction, PASS_ACTION, Player,
};

impl GameState {
    pub fn apply(&mut self, action: Action) -> Result<Option<GameOutcome>, InvalidAction> {
        let profile = game_apply_profile();
        if profile.enabled {
            let mut trace = GameApplyTrace::default();
            let total_start = Instant::now();
            let result = self.apply_traced(action, Some(&mut trace));
            trace.total_elapsed += total_start.elapsed();
            profile.record(trace, &result);
            return result;
        }

        self.apply_traced(action, None)
    }

    fn apply_traced(
        &mut self,
        action: Action,
        mut trace: Option<&mut GameApplyTrace>,
    ) -> Result<Option<GameOutcome>, InvalidAction> {
        let terminal_start = trace.as_ref().map(|_| Instant::now());
        let terminal = self.terminal;
        if let (Some(trace), Some(start)) = (trace.as_deref_mut(), terminal_start) {
            trace.terminal_check_elapsed += start.elapsed();
        }
        if terminal {
            return Err(InvalidAction::GameAlreadyEnded);
        }

        if let Some(trace) = trace.as_deref_mut() {
            match action {
                Action::Place { .. } => trace.place_calls = trace.place_calls.saturating_add(1),
                Action::Pass => trace.pass_calls = trace.pass_calls.saturating_add(1),
            }
        }
        match action {
            Action::Place { row, col } => self.apply_place(row, col, trace),
            Action::Pass => self.apply_pass(trace),
        }
    }

    #[must_use]
    pub fn legal_action_indexes(&self) -> Vec<usize> {
        if self.terminal {
            return Vec::new();
        }

        let mut actions: Vec<usize> = if self.current_player.used_count(self) < CASTLES_PER_PLAYER {
            let opponent = self.current_player.other();
            let territory_owners = self.territory_owners();
            self.board
                .iter()
                .enumerate()
                .filter_map(|(idx, cell)| {
                    (*cell == Cell::Empty && territory_owners[idx] != Some(opponent)).then_some(idx)
                })
                .collect()
        } else {
            Vec::new()
        };
        actions.push(PASS_ACTION);
        actions
    }

    fn apply_place(
        &mut self,
        row: usize,
        col: usize,
        mut trace: Option<&mut GameApplyTrace>,
    ) -> Result<Option<GameOutcome>, InvalidAction> {
        // 착수 가능 여부는 rule-spec.md 7장/22장의 순서대로 보드 변경 전에 검사한다.
        let validate_start = trace.as_ref().map(|_| Instant::now());
        if row >= BOARD_SIZE || col >= BOARD_SIZE {
            if let (Some(trace), Some(start)) = (trace.as_deref_mut(), validate_start) {
                trace.place_validate_elapsed += start.elapsed();
            }
            return Err(InvalidAction::OutOfRange);
        }
        if self.current_player.used_count(self) >= CASTLES_PER_PLAYER {
            if let (Some(trace), Some(start)) = (trace.as_deref_mut(), validate_start) {
                trace.place_validate_elapsed += start.elapsed();
            }
            return Err(InvalidAction::NoCastlesRemaining);
        }

        let index = row * BOARD_SIZE + col;
        if self.board[index] != Cell::Empty {
            if let (Some(trace), Some(start)) = (trace.as_deref_mut(), validate_start) {
                trace.place_validate_elapsed += start.elapsed();
            }
            return Err(InvalidAction::OccupiedCell);
        }
        if let (Some(trace), Some(start)) = (trace.as_deref_mut(), validate_start) {
            trace.place_validate_elapsed += start.elapsed();
        }

        let player = self.current_player;
        let territory_start = trace.as_ref().map(|_| Instant::now());
        if self.is_territory_of(index, player.other()) {
            if let (Some(trace), Some(start)) = (trace.as_deref_mut(), territory_start) {
                trace.territory_check_elapsed += start.elapsed();
            }
            return Err(InvalidAction::OpponentTerritory);
        }
        if let (Some(trace), Some(start)) = (trace.as_deref_mut(), territory_start) {
            trace.territory_check_elapsed += start.elapsed();
        }

        let mutate_start = trace.as_ref().map(|_| Instant::now());
        self.board[index] = player.cell();
        self.increment_current_player_used();
        self.previous_pass = false;
        if let (Some(trace), Some(start)) = (trace.as_deref_mut(), mutate_start) {
            trace.place_mutate_elapsed += start.elapsed();
        }

        // 파괴 판정 순서가 승패를 결정한다. 상대 그룹 파괴가 자살수 판정보다 우선한다.
        let opponent_destroyed_start = trace.as_ref().map(|_| Instant::now());
        if self.has_destroyed_group(player.other().cell()) {
            if let (Some(trace), Some(start)) = (trace.as_deref_mut(), opponent_destroyed_start) {
                trace.opponent_destroyed_elapsed += start.elapsed();
            }
            let finish_start = trace.as_ref().map(|_| Instant::now());
            let outcome = self.finish(GameEndReason::OpponentCastleDestroyed, player);
            if let (Some(trace), Some(start)) = (trace.as_deref_mut(), finish_start) {
                trace.finish_elapsed += start.elapsed();
            }
            return Ok(Some(outcome));
        }
        if let (Some(trace), Some(start)) = (trace.as_deref_mut(), opponent_destroyed_start) {
            trace.opponent_destroyed_elapsed += start.elapsed();
        }

        let own_destroyed_start = trace.as_ref().map(|_| Instant::now());
        if self.group_at_is_destroyed(index) {
            if let (Some(trace), Some(start)) = (trace.as_deref_mut(), own_destroyed_start) {
                trace.own_destroyed_elapsed += start.elapsed();
            }
            let finish_start = trace.as_ref().map(|_| Instant::now());
            let outcome = self.finish(GameEndReason::OwnCastleDestroyed, player.other());
            if let (Some(trace), Some(start)) = (trace.as_deref_mut(), finish_start) {
                trace.finish_elapsed += start.elapsed();
            }
            return Ok(Some(outcome));
        }
        if let (Some(trace), Some(start)) = (trace.as_deref_mut(), own_destroyed_start) {
            trace.own_destroyed_elapsed += start.elapsed();
        }

        let switch_start = trace.as_ref().map(|_| Instant::now());
        self.current_player = player.other();
        if let (Some(trace), Some(start)) = (trace.as_deref_mut(), switch_start) {
            trace.switch_turn_elapsed += start.elapsed();
        }
        Ok(None)
    }

    fn apply_pass(
        &mut self,
        mut trace: Option<&mut GameApplyTrace>,
    ) -> Result<Option<GameOutcome>, InvalidAction> {
        if self.previous_pass {
            // 두 번째 연속 패스에서만 영토 점수를 계산한다.
            let pass_score_start = trace.as_ref().map(|_| Instant::now());
            let winner = self.score_winner_after_consecutive_passes();
            if let (Some(trace), Some(start)) = (trace.as_deref_mut(), pass_score_start) {
                trace.pass_score_elapsed += start.elapsed();
            }
            let finish_start = trace.as_ref().map(|_| Instant::now());
            self.previous_pass = true;
            let outcome = self.finish(GameEndReason::ConsecutivePasses, winner);
            if let (Some(trace), Some(start)) = (trace.as_deref_mut(), finish_start) {
                trace.finish_elapsed += start.elapsed();
            }
            return Ok(Some(outcome));
        }

        let switch_start = trace.as_ref().map(|_| Instant::now());
        self.previous_pass = true;
        self.current_player = self.current_player.other();
        if let (Some(trace), Some(start)) = (trace.as_deref_mut(), switch_start) {
            trace.switch_turn_elapsed += start.elapsed();
        }
        Ok(None)
    }

    fn increment_current_player_used(&mut self) {
        match self.current_player {
            Player::Blue => self.blue_used += 1,
            Player::Orange => self.orange_used += 1,
        }
    }

    fn finish(&mut self, reason: GameEndReason, winner: Player) -> GameOutcome {
        let outcome = GameOutcome { reason, winner };
        self.terminal = true;
        self.outcome = Some(outcome);
        outcome
    }

    pub(crate) fn has_destroyed_group(&self, cell: Cell) -> bool {
        let mut visited = [false; BOARD_CELLS];

        // 성 파괴는 개별 성이 아니라 같은 색으로 연결된 그룹 단위로 판정한다.
        for index in 0..BOARD_CELLS {
            if visited[index] || self.board[index] != cell {
                continue;
            }

            let group = self.group_from(index, cell, &mut visited);
            if !self.group_has_liberty(&group) {
                return true;
            }
        }

        false
    }

    pub(crate) fn group_at_is_destroyed(&self, start: usize) -> bool {
        let cell = self.board[start];
        if !matches!(cell, Cell::Blue | Cell::Orange) {
            return false;
        }

        let mut visited = [false; BOARD_CELLS];
        let group = self.group_from(start, cell, &mut visited);
        !self.group_has_liberty(&group)
    }

    pub(crate) fn group_from(
        &self,
        start: usize,
        cell: Cell,
        visited: &mut [bool; BOARD_CELLS],
    ) -> Vec<usize> {
        let mut group = Vec::new();
        let mut queue = VecDeque::from([start]);
        visited[start] = true;

        while let Some(index) = queue.pop_front() {
            group.push(index);

            for neighbor in neighbors(index) {
                if visited[neighbor] || self.board[neighbor] != cell {
                    continue;
                }
                visited[neighbor] = true;
                queue.push_back(neighbor);
            }
        }

        group
    }

    fn group_has_liberty(&self, group: &[usize]) -> bool {
        // 자유 공간은 상하좌우 인접 Empty뿐이다. 보드 밖, 대각선, 중립 성은 포함하지 않는다.
        group
            .iter()
            .flat_map(|index| neighbors(*index))
            .any(|neighbor| self.board[neighbor] == Cell::Empty)
    }
}

pub(crate) fn neighbors(index: usize) -> impl Iterator<Item = usize> {
    let row = index / BOARD_SIZE;
    let col = index % BOARD_SIZE;
    let mut neighbors = [None; 4];

    if row > 0 {
        neighbors[0] = Some(index - BOARD_SIZE);
    }
    if row + 1 < BOARD_SIZE {
        neighbors[1] = Some(index + BOARD_SIZE);
    }
    if col > 0 {
        neighbors[2] = Some(index - 1);
    }
    if col + 1 < BOARD_SIZE {
        neighbors[3] = Some(index + 1);
    }

    neighbors.into_iter().flatten()
}

#[derive(Clone, Copy, Debug, Default)]
struct GameApplyTrace {
    total_elapsed: Duration,
    terminal_check_elapsed: Duration,
    place_validate_elapsed: Duration,
    territory_check_elapsed: Duration,
    place_mutate_elapsed: Duration,
    opponent_destroyed_elapsed: Duration,
    own_destroyed_elapsed: Duration,
    pass_score_elapsed: Duration,
    finish_elapsed: Duration,
    switch_turn_elapsed: Duration,
    place_calls: u64,
    pass_calls: u64,
}

struct GameApplyProfile {
    enabled: bool,
    interval: u64,
    window_calls: AtomicU64,
    place_calls: AtomicU64,
    pass_calls: AtomicU64,
    errors: AtomicU64,
    terminal_results: AtomicU64,
    total_ns: AtomicU64,
    terminal_check_ns: AtomicU64,
    place_validate_ns: AtomicU64,
    territory_check_ns: AtomicU64,
    place_mutate_ns: AtomicU64,
    opponent_destroyed_ns: AtomicU64,
    own_destroyed_ns: AtomicU64,
    pass_score_ns: AtomicU64,
    finish_ns: AtomicU64,
    switch_turn_ns: AtomicU64,
}

impl GameApplyProfile {
    fn new() -> Self {
        Self {
            enabled: env_flag("GKA_GAME_APPLY_PROFILE"),
            interval: env::var("GKA_GAME_APPLY_PROFILE_INTERVAL")
                .ok()
                .and_then(|value| value.parse::<u64>().ok())
                .filter(|value| *value > 0)
                .unwrap_or(10_000),
            window_calls: AtomicU64::new(0),
            place_calls: AtomicU64::new(0),
            pass_calls: AtomicU64::new(0),
            errors: AtomicU64::new(0),
            terminal_results: AtomicU64::new(0),
            total_ns: AtomicU64::new(0),
            terminal_check_ns: AtomicU64::new(0),
            place_validate_ns: AtomicU64::new(0),
            territory_check_ns: AtomicU64::new(0),
            place_mutate_ns: AtomicU64::new(0),
            opponent_destroyed_ns: AtomicU64::new(0),
            own_destroyed_ns: AtomicU64::new(0),
            pass_score_ns: AtomicU64::new(0),
            finish_ns: AtomicU64::new(0),
            switch_turn_ns: AtomicU64::new(0),
        }
    }

    fn record(&self, trace: GameApplyTrace, result: &Result<Option<GameOutcome>, InvalidAction>) {
        if !self.enabled {
            return;
        }

        self.place_calls
            .fetch_add(trace.place_calls, Ordering::Relaxed);
        self.pass_calls
            .fetch_add(trace.pass_calls, Ordering::Relaxed);
        if result.is_err() {
            self.errors.fetch_add(1, Ordering::Relaxed);
        }
        if matches!(result, Ok(Some(_))) {
            self.terminal_results.fetch_add(1, Ordering::Relaxed);
        }
        self.total_ns
            .fetch_add(duration_ns(trace.total_elapsed), Ordering::Relaxed);
        self.terminal_check_ns
            .fetch_add(duration_ns(trace.terminal_check_elapsed), Ordering::Relaxed);
        self.place_validate_ns
            .fetch_add(duration_ns(trace.place_validate_elapsed), Ordering::Relaxed);
        self.territory_check_ns.fetch_add(
            duration_ns(trace.territory_check_elapsed),
            Ordering::Relaxed,
        );
        self.place_mutate_ns
            .fetch_add(duration_ns(trace.place_mutate_elapsed), Ordering::Relaxed);
        self.opponent_destroyed_ns.fetch_add(
            duration_ns(trace.opponent_destroyed_elapsed),
            Ordering::Relaxed,
        );
        self.own_destroyed_ns
            .fetch_add(duration_ns(trace.own_destroyed_elapsed), Ordering::Relaxed);
        self.pass_score_ns
            .fetch_add(duration_ns(trace.pass_score_elapsed), Ordering::Relaxed);
        self.finish_ns
            .fetch_add(duration_ns(trace.finish_elapsed), Ordering::Relaxed);
        self.switch_turn_ns
            .fetch_add(duration_ns(trace.switch_turn_elapsed), Ordering::Relaxed);

        let calls = self.window_calls.fetch_add(1, Ordering::Relaxed) + 1;
        if calls >= self.interval
            && self
                .window_calls
                .compare_exchange(calls, 0, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
        {
            self.print_and_reset(calls);
        }
    }

    fn print_and_reset(&self, calls: u64) {
        let place_calls = self.place_calls.swap(0, Ordering::Relaxed);
        let pass_calls = self.pass_calls.swap(0, Ordering::Relaxed);
        let errors = self.errors.swap(0, Ordering::Relaxed);
        let terminal_results = self.terminal_results.swap(0, Ordering::Relaxed);
        let total_ns = self.total_ns.swap(0, Ordering::Relaxed);
        let terminal_check_ns = self.terminal_check_ns.swap(0, Ordering::Relaxed);
        let place_validate_ns = self.place_validate_ns.swap(0, Ordering::Relaxed);
        let territory_check_ns = self.territory_check_ns.swap(0, Ordering::Relaxed);
        let place_mutate_ns = self.place_mutate_ns.swap(0, Ordering::Relaxed);
        let opponent_destroyed_ns = self.opponent_destroyed_ns.swap(0, Ordering::Relaxed);
        let own_destroyed_ns = self.own_destroyed_ns.swap(0, Ordering::Relaxed);
        let pass_score_ns = self.pass_score_ns.swap(0, Ordering::Relaxed);
        let finish_ns = self.finish_ns.swap(0, Ordering::Relaxed);
        let switch_turn_ns = self.switch_turn_ns.swap(0, Ordering::Relaxed);

        eprintln!(
            "[gka-game-apply-profile] calls={} place={} pass={} errors={} terminal={} total={:.3}s terminal_check={:.3}s place_validate={:.3}s territory_check={:.3}s place_mutate={:.3}s opponent_destroyed={:.3}s own_destroyed={:.3}s pass_score={:.3}s finish={:.3}s switch_turn={:.3}s",
            calls,
            place_calls,
            pass_calls,
            errors,
            terminal_results,
            ns_seconds(total_ns),
            ns_seconds(terminal_check_ns),
            ns_seconds(place_validate_ns),
            ns_seconds(territory_check_ns),
            ns_seconds(place_mutate_ns),
            ns_seconds(opponent_destroyed_ns),
            ns_seconds(own_destroyed_ns),
            ns_seconds(pass_score_ns),
            ns_seconds(finish_ns),
            ns_seconds(switch_turn_ns),
        );
    }
}

fn game_apply_profile() -> &'static GameApplyProfile {
    static PROFILE: OnceLock<GameApplyProfile> = OnceLock::new();
    PROFILE.get_or_init(GameApplyProfile::new)
}

fn duration_ns(duration: Duration) -> u64 {
    duration.as_nanos().min(u128::from(u64::MAX)) as u64
}

fn ns_seconds(nanoseconds: u64) -> f64 {
    nanoseconds as f64 / 1_000_000_000.0
}

fn env_flag(name: &str) -> bool {
    !matches!(
        env::var(name).as_deref(),
        Err(_) | Ok("") | Ok("0") | Ok("false") | Ok("False") | Ok("no") | Ok("No")
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::{ACTION_SPACE, CENTER_INDEX, state_with_board};
    use pretty_assertions::assert_eq;

    fn index(row: usize, col: usize) -> usize {
        row * BOARD_SIZE + col
    }

    #[test]
    fn new_state_has_80_place_actions_and_pass() {
        let state = GameState::new();

        assert_eq!(state.legal_action_indexes().len(), ACTION_SPACE - 1);
        assert!(state.legal_action_indexes().contains(&PASS_ACTION));
    }

    #[test]
    fn place_action_updates_board_usage_and_turn() {
        let mut state = GameState::new();

        assert_eq!(state.apply(Action::Place { row: 0, col: 0 }), Ok(None));

        assert_eq!(state.board[0], Cell::Blue);
        assert_eq!(state.blue_used, 1);
        assert_eq!(state.orange_used, 0);
        assert_eq!(state.current_player, Player::Orange);
        assert!(!state.previous_pass);
        assert_eq!(state.legal_action_indexes().len(), 80);
        assert!(!state.legal_action_indexes().contains(&0));
        assert!(state.legal_action_indexes().contains(&PASS_ACTION));
    }

    #[test]
    fn place_action_rejects_occupied_center_and_out_of_range_cells() {
        let mut state = GameState::new();

        assert_eq!(
            state.apply(Action::Place { row: 4, col: 4 }),
            Err(InvalidAction::OccupiedCell)
        );
        assert_eq!(
            state.apply(Action::Place { row: 9, col: 0 }),
            Err(InvalidAction::OutOfRange)
        );
        assert_eq!(state.current_player, Player::Blue);
        assert_eq!(state.blue_used, 0);
    }

    #[test]
    fn player_with_no_castles_remaining_can_only_pass() {
        let mut state = GameState {
            blue_used: CASTLES_PER_PLAYER,
            orange_used: CASTLES_PER_PLAYER,
            ..GameState::new()
        };

        assert_eq!(state.legal_action_indexes(), vec![PASS_ACTION]);
        assert_eq!(
            state.apply(Action::Place { row: 0, col: 0 }),
            Err(InvalidAction::NoCastlesRemaining)
        );
    }

    #[test]
    fn single_pass_changes_turn_and_next_place_resets_pass_state() {
        let mut state = GameState::new();

        assert_eq!(state.apply(Action::Pass), Ok(None));
        assert_eq!(state.current_player, Player::Orange);
        assert!(state.previous_pass);
        assert!(!state.is_terminal());

        assert_eq!(state.apply(Action::Place { row: 0, col: 0 }), Ok(None));
        assert_eq!(state.board[0], Cell::Orange);
        assert_eq!(state.current_player, Player::Blue);
        assert!(!state.previous_pass);
    }

    #[test]
    fn consecutive_passes_end_game_with_score_winner() {
        let mut state = GameState::new();

        assert_eq!(state.apply(Action::Pass), Ok(None));
        let outcome = state.apply(Action::Pass).unwrap().unwrap();

        assert!(state.is_terminal());
        assert!(state.previous_pass);
        assert_eq!(state.current_player, Player::Orange);
        assert_eq!(outcome.reason, GameEndReason::ConsecutivePasses);
        assert_eq!(outcome.winner, Player::Orange);
        assert_eq!(
            state.end_reason(),
            Some(GameEndReason::ConsecutivePasses as u8)
        );
        assert_eq!(state.winner(), Some(Player::Orange as u8));
        assert_eq!(
            state.apply(Action::Pass),
            Err(InvalidAction::GameAlreadyEnded)
        );
    }

    #[test]
    fn connected_group_uses_orthogonal_adjacency_only() {
        let mut board = [Cell::Empty; BOARD_CELLS];
        board[CENTER_INDEX] = Cell::Neutral;
        board[index(0, 0)] = Cell::Blue;
        board[index(0, 1)] = Cell::Blue;
        board[index(1, 1)] = Cell::Blue;
        board[index(2, 2)] = Cell::Blue;
        let state = state_with_board(board, Player::Orange);
        let mut visited = [false; BOARD_CELLS];

        let connected = state.group_from(index(0, 0), Cell::Blue, &mut visited);

        assert_eq!(connected.len(), 3);
        assert!(connected.contains(&index(0, 0)));
        assert!(connected.contains(&index(0, 1)));
        assert!(connected.contains(&index(1, 1)));
        assert!(!connected.contains(&index(2, 2)));
    }

    #[test]
    fn liberties_ignore_board_edge_and_diagonal_empty_cells() {
        let mut board = [Cell::Orange; BOARD_CELLS];
        board[CENTER_INDEX] = Cell::Neutral;
        board[index(0, 0)] = Cell::Blue;
        board[index(1, 1)] = Cell::Empty;
        let state = state_with_board(board, Player::Orange);

        assert!(state.group_at_is_destroyed(index(0, 0)));
    }

    #[test]
    fn destroying_any_opponent_group_is_immediate_win() {
        let mut board = [Cell::Empty; BOARD_CELLS];
        board[CENTER_INDEX] = Cell::Neutral;
        board[index(1, 1)] = Cell::Orange;
        board[index(0, 1)] = Cell::Blue;
        board[index(1, 0)] = Cell::Blue;
        board[index(1, 2)] = Cell::Blue;
        let mut state = state_with_board(board, Player::Blue);

        let outcome = state
            .apply(Action::Place { row: 2, col: 1 })
            .unwrap()
            .unwrap();

        assert!(state.is_terminal());
        assert_eq!(outcome.reason, GameEndReason::OpponentCastleDestroyed);
        assert_eq!(outcome.winner, Player::Blue);
        assert_eq!(state.board[index(2, 1)], Cell::Blue);
    }

    #[test]
    fn suicide_without_opponent_capture_is_immediate_loss() {
        let mut board = [Cell::Empty; BOARD_CELLS];
        board[CENTER_INDEX] = Cell::Neutral;
        board[index(0, 1)] = Cell::Orange;
        board[index(1, 0)] = Cell::Orange;
        board[index(1, 2)] = Cell::Blue;
        board[index(0, 2)] = Cell::Orange;
        board[index(1, 3)] = Cell::Orange;
        board[index(2, 2)] = Cell::Orange;
        board[index(2, 1)] = Cell::Orange;
        let mut state = state_with_board(board, Player::Blue);

        let outcome = state
            .apply(Action::Place { row: 1, col: 1 })
            .unwrap()
            .unwrap();

        assert!(state.is_terminal());
        assert_eq!(outcome.reason, GameEndReason::OwnCastleDestroyed);
        assert_eq!(outcome.winner, Player::Orange);
    }

    #[test]
    fn opponent_capture_takes_priority_over_own_destroyed_group() {
        let mut board = [Cell::Blue; BOARD_CELLS];
        board[CENTER_INDEX] = Cell::Neutral;
        board[index(1, 1)] = Cell::Empty;
        board[index(0, 0)] = Cell::Orange;
        board[index(0, 1)] = Cell::Orange;
        board[index(1, 0)] = Cell::Blue;
        board[index(2, 1)] = Cell::Orange;
        let mut state = state_with_board(board, Player::Orange);

        let outcome = state
            .apply(Action::Place { row: 1, col: 1 })
            .unwrap()
            .unwrap();

        assert!(state.is_terminal());
        assert_eq!(outcome.reason, GameEndReason::OpponentCastleDestroyed);
        assert_eq!(outcome.winner, Player::Orange);
    }

    #[test]
    fn cannot_place_inside_current_opponent_territory() {
        let mut board = [Cell::Empty; BOARD_CELLS];
        board[CENTER_INDEX] = Cell::Neutral;
        board[index(0, 1)] = Cell::Blue;
        board[index(1, 0)] = Cell::Blue;
        board[index(1, 2)] = Cell::Blue;
        board[index(2, 1)] = Cell::Blue;
        let mut state = state_with_board(board, Player::Orange);

        assert!(!state.legal_action_indexes().contains(&index(1, 1)));
        assert_eq!(
            state.apply(Action::Place { row: 1, col: 1 }),
            Err(InvalidAction::OpponentTerritory)
        );
    }
}
