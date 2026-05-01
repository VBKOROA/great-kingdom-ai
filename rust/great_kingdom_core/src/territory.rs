use crate::game::{GameState, Player};

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
        (0, 0)
    }

    pub(crate) fn is_territory_of(&self, _index: usize, _player: Player) -> bool {
        false
    }
}
