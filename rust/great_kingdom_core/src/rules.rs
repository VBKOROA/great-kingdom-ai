use std::collections::VecDeque;

use crate::game::{
    Action, BOARD_CELLS, BOARD_SIZE, CASTLES_PER_PLAYER, Cell, GameEndReason, GameOutcome,
    GameState, InvalidAction, PASS_ACTION, Player,
};

impl GameState {
    pub fn apply(&mut self, action: Action) -> Result<Option<GameOutcome>, InvalidAction> {
        if self.terminal {
            return Err(InvalidAction::GameAlreadyEnded);
        }

        match action {
            Action::Place { row, col } => self.apply_place(row, col),
            Action::Pass => self.apply_pass(),
        }
    }

    #[must_use]
    pub fn legal_action_indexes(&self) -> Vec<usize> {
        if self.terminal {
            return Vec::new();
        }

        let mut actions: Vec<usize> = if self.current_player.used_count(self) < CASTLES_PER_PLAYER {
            self.board
                .iter()
                .enumerate()
                .filter_map(|(idx, cell)| {
                    (*cell == Cell::Empty
                        && !self.is_territory_of(idx, self.current_player.other()))
                    .then_some(idx)
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
    ) -> Result<Option<GameOutcome>, InvalidAction> {
        if row >= BOARD_SIZE || col >= BOARD_SIZE {
            return Err(InvalidAction::OutOfRange);
        }
        if self.current_player.used_count(self) >= CASTLES_PER_PLAYER {
            return Err(InvalidAction::NoCastlesRemaining);
        }

        let index = row * BOARD_SIZE + col;
        if self.board[index] != Cell::Empty {
            return Err(InvalidAction::OccupiedCell);
        }

        let player = self.current_player;
        if self.is_territory_of(index, player.other()) {
            return Err(InvalidAction::OpponentTerritory);
        }

        self.board[index] = player.cell();
        self.increment_current_player_used();
        self.previous_pass = false;

        if self.has_destroyed_group(player.other().cell()) {
            return Ok(Some(
                self.finish(GameEndReason::OpponentCastleDestroyed, player),
            ));
        }

        if self.group_at_is_destroyed(index) {
            return Ok(Some(
                self.finish(GameEndReason::OwnCastleDestroyed, player.other()),
            ));
        }

        self.current_player = player.other();
        Ok(None)
    }

    fn apply_pass(&mut self) -> Result<Option<GameOutcome>, InvalidAction> {
        if self.previous_pass {
            let winner = self.score_winner_after_consecutive_passes();
            self.previous_pass = true;
            return Ok(Some(self.finish(GameEndReason::ConsecutivePasses, winner)));
        }

        self.previous_pass = true;
        self.current_player = self.current_player.other();
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
