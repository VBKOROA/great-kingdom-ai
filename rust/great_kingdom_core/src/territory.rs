use std::collections::VecDeque;

use crate::{
    game::{BOARD_CELLS, BOARD_SIZE, Cell, GameState, Player},
    rules::neighbors,
};

impl GameState {
    pub(crate) fn score_winner_after_consecutive_passes(&self) -> Player {
        let (blue_score, orange_score) = self.territory_scores();
        if blue_score >= orange_score + 3 {
            Player::Blue
        } else {
            Player::Orange
        }
    }

    pub(crate) fn territory_scores(&self) -> (u8, u8) {
        let mut visited = [false; BOARD_CELLS];
        let mut blue_score = 0;
        let mut orange_score = 0;

        for index in 0..BOARD_CELLS {
            if visited[index] || self.board[index] != Cell::Empty {
                continue;
            }

            let region = self.empty_region_from(index, &mut visited);
            if self.region_is_territory_of(&region, Player::Blue) {
                blue_score += region.len() as u8;
            } else if self.region_is_territory_of(&region, Player::Orange) {
                orange_score += region.len() as u8;
            }
        }

        (blue_score, orange_score)
    }

    pub(crate) fn is_territory_of(&self, _index: usize, _player: Player) -> bool {
        if self.board[_index] != Cell::Empty {
            return false;
        }

        let mut visited = [false; BOARD_CELLS];
        let region = self.empty_region_from(_index, &mut visited);
        self.region_is_territory_of(&region, _player)
    }

    fn empty_region_from(&self, start: usize, visited: &mut [bool; BOARD_CELLS]) -> Vec<usize> {
        let mut region = Vec::new();
        let mut queue = VecDeque::from([start]);
        visited[start] = true;

        while let Some(index) = queue.pop_front() {
            region.push(index);

            for neighbor in neighbors(index) {
                if visited[neighbor] || self.board[neighbor] != Cell::Empty {
                    continue;
                }
                visited[neighbor] = true;
                queue.push_back(neighbor);
            }
        }

        region
    }

    fn region_is_territory_of(&self, region: &[usize], player: Player) -> bool {
        let mut has_player_boundary = false;
        let mut touches_top = false;
        let mut touches_bottom = false;
        let mut touches_left = false;
        let mut touches_right = false;

        for index in region {
            let row = index / BOARD_SIZE;
            let col = index % BOARD_SIZE;
            touches_top |= row == 0;
            touches_bottom |= row + 1 == BOARD_SIZE;
            touches_left |= col == 0;
            touches_right |= col + 1 == BOARD_SIZE;

            for neighbor in neighbors(*index) {
                match self.board[neighbor] {
                    Cell::Empty | Cell::Neutral => {}
                    cell if cell == player.cell() => has_player_boundary = true,
                    Cell::Blue | Cell::Orange => return false,
                }
            }
        }

        has_player_boundary && !(touches_top && touches_bottom && touches_left && touches_right)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::{Action, BOARD_CELLS, CENTER_INDEX, GameEndReason, state_with_board};
    use pretty_assertions::assert_eq;

    fn index(row: usize, col: usize) -> usize {
        row * BOARD_SIZE + col
    }

    #[test]
    fn closed_empty_region_scores_for_enclosing_player() {
        let mut board = [Cell::Orange; BOARD_CELLS];
        board[CENTER_INDEX] = Cell::Neutral;
        board[index(1, 1)] = Cell::Empty;
        board[index(0, 1)] = Cell::Blue;
        board[index(1, 0)] = Cell::Blue;
        board[index(1, 2)] = Cell::Blue;
        board[index(2, 1)] = Cell::Blue;
        let state = state_with_board(board, Player::Blue);

        assert!(state.is_territory_of(index(1, 1), Player::Blue));
        assert_eq!(state.territory_scores(), (1, 0));
    }

    #[test]
    fn neutral_castle_can_be_part_of_territory_boundary() {
        let mut board = [Cell::Orange; BOARD_CELLS];
        board[CENTER_INDEX] = Cell::Neutral;
        board[index(4, 3)] = Cell::Empty;
        board[index(3, 3)] = Cell::Blue;
        board[index(5, 3)] = Cell::Blue;
        board[index(4, 2)] = Cell::Blue;
        let state = state_with_board(board, Player::Blue);

        assert!(state.is_territory_of(index(4, 3), Player::Blue));
        assert_eq!(state.territory_scores(), (1, 0));
    }

    #[test]
    fn board_edge_can_be_part_of_territory_boundary() {
        let mut board = [Cell::Orange; BOARD_CELLS];
        board[CENTER_INDEX] = Cell::Neutral;
        board[index(0, 1)] = Cell::Empty;
        board[index(0, 0)] = Cell::Blue;
        board[index(0, 2)] = Cell::Blue;
        board[index(1, 1)] = Cell::Blue;
        let state = state_with_board(board, Player::Blue);

        assert!(state.is_territory_of(index(0, 1), Player::Blue));
        assert_eq!(state.territory_scores(), (1, 0));
    }

    #[test]
    fn region_adjacent_to_opponent_castle_is_neutral() {
        let mut board = [Cell::Orange; BOARD_CELLS];
        board[CENTER_INDEX] = Cell::Neutral;
        board[index(1, 1)] = Cell::Empty;
        board[index(0, 1)] = Cell::Blue;
        board[index(1, 0)] = Cell::Blue;
        board[index(1, 2)] = Cell::Blue;
        let state = state_with_board(board, Player::Blue);

        assert!(!state.is_territory_of(index(1, 1), Player::Blue));
        assert_eq!(state.territory_scores(), (0, 0));
    }

    #[test]
    fn empty_region_touching_all_four_edges_is_neutral() {
        let mut state = GameState::new();
        state.board[index(0, 0)] = Cell::Blue;

        assert!(!state.is_territory_of(index(0, 1), Player::Blue));
        assert_eq!(state.territory_scores(), (0, 0));
    }

    #[test]
    fn blue_wins_score_if_ahead_by_at_least_three() {
        let mut board = [Cell::Orange; BOARD_CELLS];
        board[CENTER_INDEX] = Cell::Neutral;
        for (row, col) in [(1, 1), (1, 4), (4, 1)] {
            board[index(row, col)] = Cell::Empty;
            board[index(row - 1, col)] = Cell::Blue;
            board[index(row + 1, col)] = Cell::Blue;
            board[index(row, col - 1)] = Cell::Blue;
            board[index(row, col + 1)] = Cell::Blue;
        }
        let mut state = state_with_board(board, Player::Blue);

        assert_eq!(state.territory_scores(), (3, 0));
        assert_eq!(state.apply(Action::Pass), Ok(None));
        let outcome = state.apply(Action::Pass).unwrap().unwrap();

        assert_eq!(outcome.reason, GameEndReason::ConsecutivePasses);
        assert_eq!(outcome.winner, Player::Blue);
    }

    #[test]
    fn orange_wins_score_if_blue_leads_by_two_or_less() {
        let mut board = [Cell::Orange; BOARD_CELLS];
        board[CENTER_INDEX] = Cell::Neutral;
        for (row, col) in [(1, 1), (1, 4)] {
            board[index(row, col)] = Cell::Empty;
            board[index(row - 1, col)] = Cell::Blue;
            board[index(row + 1, col)] = Cell::Blue;
            board[index(row, col - 1)] = Cell::Blue;
            board[index(row, col + 1)] = Cell::Blue;
        }
        let mut state = state_with_board(board, Player::Blue);

        assert_eq!(state.territory_scores(), (2, 0));
        assert_eq!(state.apply(Action::Pass), Ok(None));
        let outcome = state.apply(Action::Pass).unwrap().unwrap();

        assert_eq!(outcome.reason, GameEndReason::ConsecutivePasses);
        assert_eq!(outcome.winner, Player::Orange);
    }
}
