//! KLENT zero-search actor with batched ONNX inference.
//!
//! Each turn evaluates every active game state in one batched forward pass,
//! computes the closed-form target policy `pi'` and `E_pi'[Q]` in FP32, and
//! samples an action from `pi'`. Lambda returns are left to the Python caller,
//! which owns the shared target implementation.

use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};

use crate::{
    eval_request::EvalRequest,
    game::{ACTION_SPACE, Action, GameState},
    gumbel::rng::SplitMix64,
    onnx::{OnnxError, OnnxEvaluator, OnnxEvaluatorConfig, parse_device},
};

#[derive(Clone, Debug)]
struct KlentTurn {
    player: u8,
    action: usize,
    policy_target: Vec<f32>,
    state_value: f32,
}

#[pyclass(unsendable)]
pub struct KlentZeroSearchBatch {
    evaluator: OnnxEvaluator,
    games: Vec<GameState>,
    rngs: Vec<SplitMix64>,
    records: Vec<Vec<KlentTurn>>,
    seeds: Vec<u64>,
    alpha: f32,
    beta: f32,
    max_turns: usize,
    ran: bool,
}

#[pymethods]
impl KlentZeroSearchBatch {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        path,
        games,
        seed,
        device = "cpu",
        max_batch_size = 256,
        alpha = 0.03,
        beta = 0.1,
        max_turns = 200,
    ))]
    fn py_new(
        path: &str,
        games: usize,
        seed: u64,
        device: &str,
        max_batch_size: usize,
        alpha: f32,
        beta: f32,
        max_turns: usize,
    ) -> PyResult<Self> {
        validate_klent_coefficients(alpha, beta)?;
        if games == 0 {
            return Err(PyValueError::new_err("games must be positive"));
        }
        if max_batch_size == 0 {
            return Err(PyValueError::new_err("max_batch_size must be positive"));
        }
        if max_turns == 0 {
            return Err(PyValueError::new_err("max_turns must be positive"));
        }

        let evaluator = OnnxEvaluator::load(
            path,
            OnnxEvaluatorConfig {
                device: parse_device(device)?,
                max_batch_size,
            },
        )
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
        if !evaluator.supports_q_values_output() {
            return Err(PyValueError::new_err(
                "zero-search actor requires a 3-output ONNX model with a 'q_values' output",
            ));
        }

        let seeds = (0..games)
            .map(|game_index| seed.wrapping_add(game_index as u64))
            .collect::<Vec<_>>();
        Ok(Self {
            evaluator,
            games: vec![GameState::new(); games],
            rngs: seeds.iter().copied().map(SplitMix64::new).collect(),
            records: vec![Vec::new(); games],
            seeds,
            alpha,
            beta,
            max_turns,
            ran: false,
        })
    }

    #[must_use]
    fn game_count(&self) -> usize {
        self.games.len()
    }

    #[must_use]
    fn seeds(&self) -> Vec<u64> {
        self.seeds.clone()
    }

    fn run(&mut self) -> PyResult<()> {
        if self.ran {
            return Err(PyValueError::new_err(
                "KlentZeroSearchBatch has already run",
            ));
        }
        for _turn in 0..self.max_turns {
            let active = (0..self.games.len())
                .filter(|game_index| !self.games[*game_index].is_terminal())
                .collect::<Vec<_>>();
            if active.is_empty() {
                break;
            }

            let request = EvalRequest::new_with_precomputed_features(
                active
                    .iter()
                    .map(|game_index| self.games[*game_index].clone())
                    .collect(),
            );
            let output = self
                .evaluator
                .evaluate_request_with_q(&request)
                .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
            let q_values = output
                .q_values
                .expect("q values are validated at construction");

            for (row, game_index) in active.iter().enumerate() {
                let legal_mask = self.games[*game_index].legal_mask();
                let policy_target = analytical_policy_row(
                    &output.policy_logits[row],
                    &q_values[row],
                    &legal_mask,
                    self.alpha,
                    self.beta,
                )
                .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
                let state_value = masked_value_row(&policy_target, &q_values[row]);
                let action = sample_from_target(&policy_target, &mut self.rngs[*game_index]);
                let player = self.games[*game_index].current_player();
                self.games[*game_index]
                    .apply(Action::from_index(action).expect("sampled action is in range"))
                    .map_err(|err| {
                        PyRuntimeError::new_err(format!("invalid sampled action: {err:?}"))
                    })?;
                self.records[*game_index].push(KlentTurn {
                    player,
                    action,
                    policy_target,
                    state_value,
                });
            }
        }

        if let Some(game_index) = self.games.iter().position(|game| !game.is_terminal()) {
            return Err(PyRuntimeError::new_err(format!(
                "game {game_index} exceeded max_turns={} without a terminal outcome",
                self.max_turns
            )));
        }
        self.ran = true;
        Ok(())
    }

    #[must_use]
    fn winners(&self) -> Vec<Option<u8>> {
        self.games.iter().map(GameState::winner).collect()
    }

    #[must_use]
    fn end_reasons(&self) -> Vec<Option<u8>> {
        self.games.iter().map(GameState::end_reason).collect()
    }

    #[must_use]
    fn territory_scores(&self) -> Vec<(u8, u8)> {
        self.games.iter().map(GameState::territory_scores).collect()
    }

    #[must_use]
    fn move_counts(&self) -> Vec<usize> {
        self.records.iter().map(Vec::len).collect()
    }

    fn turn_players(&self, game_index: usize) -> PyResult<Vec<u8>> {
        Ok(self
            .game_records(game_index)?
            .iter()
            .map(|t| t.player)
            .collect())
    }

    fn turn_actions(&self, game_index: usize) -> PyResult<Vec<usize>> {
        Ok(self
            .game_records(game_index)?
            .iter()
            .map(|t| t.action)
            .collect())
    }

    fn policy_targets(&self, game_index: usize) -> PyResult<Vec<Vec<f32>>> {
        Ok(self
            .game_records(game_index)?
            .iter()
            .map(|t| t.policy_target.clone())
            .collect())
    }

    fn state_values(&self, game_index: usize) -> PyResult<Vec<f32>> {
        Ok(self
            .game_records(game_index)?
            .iter()
            .map(|t| t.state_value)
            .collect())
    }
}

impl KlentZeroSearchBatch {
    fn game_records(&self, game_index: usize) -> PyResult<&[KlentTurn]> {
        self.records
            .get(game_index)
            .map(Vec::as_slice)
            .ok_or_else(|| {
                PyValueError::new_err(format!("game index {game_index} is out of range"))
            })
    }
}

fn validate_klent_coefficients(alpha: f32, beta: f32) -> PyResult<()> {
    if !alpha.is_finite() || alpha < 0.0 {
        return Err(PyValueError::new_err(format!(
            "alpha must be finite and non-negative, got {alpha}"
        )));
    }
    if !beta.is_finite() || beta < 0.0 {
        return Err(PyValueError::new_err(format!(
            "beta must be finite and non-negative, got {beta}"
        )));
    }
    if alpha + beta <= 0.0 {
        return Err(PyValueError::new_err("alpha + beta must be positive"));
    }
    Ok(())
}

pub(crate) fn analytical_policy_row(
    policy_logits: &[f32],
    q_values: &[f32],
    legal_mask: &[bool],
    alpha: f32,
    beta: f32,
) -> Result<Vec<f32>, OnnxError> {
    validate_target_inputs(policy_logits, q_values, legal_mask, alpha, beta)?;

    let log_normalizer = legal_log_normalizer(policy_logits, legal_mask)?;
    let denominator = alpha + beta;
    let mut max_score = f32::NEG_INFINITY;
    for index in 0..ACTION_SPACE {
        if !legal_mask[index] {
            continue;
        }
        let log_pi = policy_logits[index] - log_normalizer;
        let score = (q_values[index] + beta * log_pi) / denominator;
        max_score = max_score.max(score);
    }

    let mut target = vec![0.0; ACTION_SPACE];
    let mut exp_sum = 0.0;
    for index in 0..ACTION_SPACE {
        if !legal_mask[index] {
            continue;
        }
        let log_pi = policy_logits[index] - log_normalizer;
        let score = (q_values[index] + beta * log_pi) / denominator;
        let weight = (score - max_score).exp();
        target[index] = weight;
        exp_sum += weight;
    }
    for index in 0..ACTION_SPACE {
        if legal_mask[index] {
            target[index] /= exp_sum;
        }
    }
    Ok(target)
}

pub(crate) fn masked_state_value_row(
    policy_logits: &[f32],
    q_values: &[f32],
    legal_mask: &[bool],
) -> Result<f32, OnnxError> {
    validate_target_inputs(policy_logits, q_values, legal_mask, 1.0, 0.0)?;
    let probabilities = legal_softmax(policy_logits, legal_mask)?;
    Ok(masked_value_row(&probabilities, q_values))
}

fn masked_value_row(probabilities: &[f32], q_values: &[f32]) -> f32 {
    probabilities
        .iter()
        .zip(q_values.iter())
        .map(|(probability, q_value)| probability * q_value)
        .sum()
}

fn validate_target_inputs(
    policy_logits: &[f32],
    q_values: &[f32],
    legal_mask: &[bool],
    alpha: f32,
    beta: f32,
) -> Result<(), OnnxError> {
    if policy_logits.len() != ACTION_SPACE || q_values.len() != ACTION_SPACE {
        return Err(OnnxError::InvalidRequest(format!(
            "expected {ACTION_SPACE} policy logits and q values, got {} and {}",
            policy_logits.len(),
            q_values.len()
        )));
    }
    if legal_mask.len() != ACTION_SPACE {
        return Err(OnnxError::InvalidRequest(format!(
            "expected {ACTION_SPACE} legal mask entries, got {}",
            legal_mask.len()
        )));
    }
    validate_klent_coefficients(alpha, beta)
        .map_err(|err| OnnxError::InvalidRequest(err.to_string()))?;
    Ok(())
}

fn legal_log_normalizer(policy_logits: &[f32], legal_mask: &[bool]) -> Result<f32, OnnxError> {
    let max_legal = legal_mask
        .iter()
        .enumerate()
        .filter(|(_, is_legal)| **is_legal)
        .map(|(index, _)| policy_logits[index])
        .fold(f32::NEG_INFINITY, f32::max);
    if !max_legal.is_finite() {
        return Err(OnnxError::InvalidRequest(
            "legal mask must contain at least one legal action".to_string(),
        ));
    }
    let exp_sum: f32 = legal_mask
        .iter()
        .enumerate()
        .filter(|(_, is_legal)| **is_legal)
        .map(|(index, _)| (policy_logits[index] - max_legal).exp())
        .sum();
    Ok(max_legal + exp_sum.ln())
}

fn legal_softmax(policy_logits: &[f32], legal_mask: &[bool]) -> Result<Vec<f32>, OnnxError> {
    let log_normalizer = legal_log_normalizer(policy_logits, legal_mask)?;
    let mut probabilities = vec![0.0; ACTION_SPACE];
    for index in 0..ACTION_SPACE {
        if legal_mask[index] {
            probabilities[index] = (policy_logits[index] - log_normalizer).exp();
        }
    }
    Ok(probabilities)
}

fn sample_from_target(target: &[f32], rng: &mut SplitMix64) -> usize {
    let total: f32 = target.iter().sum();
    let mut threshold = rng.next_open01() * total;
    let mut fallback = 0;
    for (index, probability) in target.iter().enumerate() {
        if *probability <= 0.0 {
            continue;
        }
        fallback = index;
        threshold -= *probability;
        if threshold <= 0.0 {
            return index;
        }
    }
    fallback
}

#[pyfunction]
#[pyo3(signature = (policy_logits, q_values, legal_mask, alpha, beta))]
pub fn klent_analytical_policy(
    policy_logits: Vec<f32>,
    q_values: Vec<f32>,
    legal_mask: Vec<bool>,
    alpha: f32,
    beta: f32,
) -> PyResult<Vec<f32>> {
    analytical_policy_row(&policy_logits, &q_values, &legal_mask, alpha, beta)
        .map_err(|err| PyValueError::new_err(err.to_string()))
}

#[pyfunction]
#[pyo3(signature = (policy_logits, q_values, legal_mask))]
pub fn klent_masked_state_value(
    policy_logits: Vec<f32>,
    q_values: Vec<f32>,
    legal_mask: Vec<bool>,
) -> PyResult<f32> {
    masked_state_value_row(&policy_logits, &q_values, &legal_mask)
        .map_err(|err| PyValueError::new_err(err.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    fn mask_of(indexes: &[usize]) -> Vec<bool> {
        let mut mask = vec![false; ACTION_SPACE];
        for index in indexes {
            mask[*index] = true;
        }
        mask
    }

    #[test]
    fn analytical_policy_normalizes_over_legal_actions_only() {
        let logits = vec![0.5; ACTION_SPACE];
        let q_values = vec![0.0; ACTION_SPACE];
        let mask = mask_of(&[0, 5, 81]);

        let target = analytical_policy_row(&logits, &q_values, &mask, 0.03, 0.1)
            .expect("target should compute");

        let total: f32 = target.iter().sum();
        assert!((total - 1.0).abs() < 1e-6);
        assert_eq!(target[0], target[5]);
        assert_eq!(target[5], target[81]);
        assert_eq!(target[1], 0.0);
    }

    #[test]
    fn analytical_policy_prefers_larger_q_values() {
        let logits = vec![0.0; ACTION_SPACE];
        let mut q_values = vec![0.0; ACTION_SPACE];
        q_values[3] = 1.0;
        let mask = mask_of(&[3, 7]);

        let target = analytical_policy_row(&logits, &q_values, &mask, 0.03, 0.1)
            .expect("target should compute");

        assert!(target[3] > target[7]);
        assert_eq!(target[0], 0.0);
    }

    #[test]
    fn analytical_policy_rejects_empty_legal_mask() {
        let error = analytical_policy_row(
            &vec![0.0; ACTION_SPACE],
            &vec![0.0; ACTION_SPACE],
            &[false; ACTION_SPACE],
            0.03,
            0.1,
        )
        .unwrap_err();

        assert_eq!(
            error.to_string(),
            "legal mask must contain at least one legal action"
        );
    }

    #[test]
    fn masked_state_value_uses_legal_softmax() {
        let mut logits = vec![0.0; ACTION_SPACE];
        logits[0] = 2.0;
        let mut q_values = vec![0.0; ACTION_SPACE];
        q_values[0] = 0.5;
        q_values[1] = -1.0;
        let mask = mask_of(&[0, 1]);

        let value =
            masked_state_value_row(&logits, &q_values, &mask).expect("value should compute");

        let probability_0 = (2.0_f32).exp() / (2.0_f32.exp() + 1.0);
        let expected = probability_0 * 0.5 + (1.0 - probability_0) * -1.0;
        assert!((value - expected).abs() < 1e-6);
    }

    #[test]
    fn sampler_stays_on_legal_actions() {
        let mut target = vec![0.0; ACTION_SPACE];
        target[10] = 0.25;
        target[81] = 0.75;
        let mut rng = SplitMix64::new(7);

        for _ in 0..64 {
            let action = sample_from_target(&target, &mut rng);
            assert!(action == 10 || action == 81);
        }
    }

    #[test]
    fn coefficient_validation_rejects_invalid_values() {
        assert!(validate_klent_coefficients(-0.1, 0.1).is_err());
        assert!(validate_klent_coefficients(0.0, 0.0).is_err());
        assert!(validate_klent_coefficients(f32::NAN, 0.1).is_err());
        assert!(validate_klent_coefficients(0.03, 0.1).is_ok());
    }
}
