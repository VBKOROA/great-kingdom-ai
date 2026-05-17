use crate::game::{
    ACTION_SPACE, BOARD_CELLS, CASTLES_PER_PLAYER, Cell, FEATURE_CHANNELS, GameState, Player,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FeatureChannel {
    OwnCastle = 0,
    OpponentCastle = 1,
    NeutralCastle = 2,
    Empty = 3,
    LegalPlace = 4,
    OwnTerritory = 5,
    OpponentTerritory = 6,
    OwnRemainingCastles = 7,
    OpponentRemainingCastles = 8,
    CurrentPlayerIsBlue = 9,
    PreviousMoveWasPass = 10,
}

impl GameState {
    #[must_use]
    pub(crate) fn legal_action_mask(&self) -> [bool; ACTION_SPACE] {
        let mut mask = [false; ACTION_SPACE];
        for action in self.legal_action_indexes() {
            mask[action] = true;
        }
        mask
    }

    #[must_use]
    pub(crate) fn feature_planes_array(&self) -> [f32; FEATURE_CHANNELS * BOARD_CELLS] {
        let mut planes = [0.0; FEATURE_CHANNELS * BOARD_CELLS];
        let player = self.current_player_value();
        let opponent = player.other();
        let own_remaining = remaining_castles(player, self);
        let opponent_remaining = remaining_castles(opponent, self);
        let current_player_is_blue = f32::from(player == Player::Blue);
        let previous_move_was_pass = f32::from(self.previous_pass);
        let territory_owners = self.territory_owners();
        let can_place = player.used_count(self) < CASTLES_PER_PLAYER;

        for (index, territory_owner) in territory_owners.iter().enumerate().take(BOARD_CELLS) {
            match self.board[index] {
                cell if cell == player.cell() => {
                    set_channel(&mut planes, FeatureChannel::OwnCastle, index, 1.0);
                }
                cell if cell == opponent.cell() => {
                    set_channel(&mut planes, FeatureChannel::OpponentCastle, index, 1.0);
                }
                Cell::Neutral => {
                    set_channel(&mut planes, FeatureChannel::NeutralCastle, index, 1.0);
                }
                Cell::Empty => {
                    set_channel(&mut planes, FeatureChannel::Empty, index, 1.0);
                    if *territory_owner == Some(player) {
                        set_channel(&mut planes, FeatureChannel::OwnTerritory, index, 1.0);
                    }
                    if *territory_owner == Some(opponent) {
                        set_channel(&mut planes, FeatureChannel::OpponentTerritory, index, 1.0);
                    }
                }
                Cell::Blue | Cell::Orange => {}
            }

            if can_place && self.board[index] == Cell::Empty && *territory_owner != Some(opponent) {
                set_channel(&mut planes, FeatureChannel::LegalPlace, index, 1.0);
            }
            set_channel(
                &mut planes,
                FeatureChannel::OwnRemainingCastles,
                index,
                own_remaining,
            );
            set_channel(
                &mut planes,
                FeatureChannel::OpponentRemainingCastles,
                index,
                opponent_remaining,
            );
            set_channel(
                &mut planes,
                FeatureChannel::CurrentPlayerIsBlue,
                index,
                current_player_is_blue,
            );
            set_channel(
                &mut planes,
                FeatureChannel::PreviousMoveWasPass,
                index,
                previous_move_was_pass,
            );
        }

        planes
    }
}

fn remaining_castles(player: Player, state: &GameState) -> f32 {
    f32::from(CASTLES_PER_PLAYER - player.used_count(state)) / f32::from(CASTLES_PER_PLAYER)
}

fn set_channel(
    planes: &mut [f32; FEATURE_CHANNELS * BOARD_CELLS],
    channel: FeatureChannel,
    index: usize,
    value: f32,
) {
    planes[channel as usize * BOARD_CELLS + index] = value;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::{Action, BOARD_SIZE, CENTER_INDEX, state_with_board};
    use pretty_assertions::assert_eq;

    fn value(planes: &[f32], channel: FeatureChannel, index: usize) -> f32 {
        planes[channel as usize * BOARD_CELLS + index]
    }

    fn index(row: usize, col: usize) -> usize {
        row * BOARD_SIZE + col
    }

    #[test]
    fn legal_action_mask_matches_legal_action_indexes() {
        let state = GameState::new();
        let mask = state.legal_action_mask();
        let legal = state.legal_action_indexes();

        assert_eq!(mask.len(), ACTION_SPACE);
        assert_eq!(
            mask.iter().filter(|is_legal| **is_legal).count(),
            legal.len()
        );
        for action in legal {
            assert!(mask[action]);
        }
        assert!(!mask[CENTER_INDEX]);
        assert!(mask[crate::game::PASS_ACTION]);
    }

    #[test]
    fn feature_planes_have_current_player_perspective_channels() {
        let mut board = [Cell::Empty; BOARD_CELLS];
        board[CENTER_INDEX] = Cell::Neutral;
        board[0] = Cell::Blue;
        board[1] = Cell::Orange;
        let state = state_with_board(board, Player::Orange);
        let planes = state.feature_planes_array();

        assert_eq!(planes.len(), FEATURE_CHANNELS * BOARD_SIZE * BOARD_SIZE);
        assert_eq!(value(&planes, FeatureChannel::OwnCastle, 1), 1.0);
        assert_eq!(value(&planes, FeatureChannel::OpponentCastle, 0), 1.0);
        assert_eq!(
            value(&planes, FeatureChannel::NeutralCastle, CENTER_INDEX),
            1.0
        );
        assert_eq!(value(&planes, FeatureChannel::Empty, 2), 1.0);
        assert_eq!(value(&planes, FeatureChannel::CurrentPlayerIsBlue, 2), 0.0);
        assert_eq!(value(&planes, FeatureChannel::PreviousMoveWasPass, 2), 0.0);

        let blue_state = state_with_board(board, Player::Blue);
        let blue_planes = blue_state.feature_planes_array();
        assert_eq!(
            value(&blue_planes, FeatureChannel::CurrentPlayerIsBlue, 2),
            1.0
        );
    }

    #[test]
    fn feature_planes_track_legal_places_and_pass_state() {
        let mut state = GameState::new();
        state.apply(Action::Pass).unwrap();
        let planes = state.feature_planes_array();

        assert_eq!(value(&planes, FeatureChannel::LegalPlace, 0), 1.0);
        assert_eq!(
            value(&planes, FeatureChannel::LegalPlace, CENTER_INDEX),
            0.0
        );
        assert_eq!(value(&planes, FeatureChannel::PreviousMoveWasPass, 0), 1.0);
        assert_eq!(state.legal_action_mask()[crate::game::PASS_ACTION], true);
    }

    #[test]
    fn feature_legal_places_match_rule_legal_mask() {
        let mut board = [Cell::Empty; BOARD_CELLS];
        board[CENTER_INDEX] = Cell::Neutral;
        board[index(0, 1)] = Cell::Blue;
        board[index(1, 0)] = Cell::Blue;
        board[index(1, 2)] = Cell::Blue;
        board[index(2, 1)] = Cell::Blue;
        let state = state_with_board(board, Player::Orange);
        let planes = state.feature_planes_array();
        let legal_mask = state.legal_action_mask();

        for (action, is_legal) in legal_mask.iter().take(BOARD_CELLS).enumerate() {
            assert_eq!(
                value(&planes, FeatureChannel::LegalPlace, action),
                f32::from(*is_legal),
                "legal-place feature diverged from rule legal mask at action {action}",
            );
        }
    }
}
