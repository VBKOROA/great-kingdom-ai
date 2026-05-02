//! Improved policy target helpers for the Gumbel search backend.

use pyo3::{PyResult, exceptions::PyValueError};

use crate::game::ACTION_SPACE;

pub(crate) const PRIOR_EPSILON: f32 = 1.0e-8;

pub(crate) fn log_priors_from_logits(
    legal_actions: &[usize],
    logits: &[f32],
) -> PyResult<[f32; ACTION_SPACE]> {
    validate_policy_len(logits, "policy_logits")?;
    validate_legal_values(legal_actions, logits, "policy_logits")?;

    let mut log_priors = [f32::NEG_INFINITY; ACTION_SPACE];
    if legal_actions.is_empty() {
        return Ok(log_priors);
    }

    let max_logit = legal_actions
        .iter()
        .map(|action| logits[*action])
        .fold(f32::NEG_INFINITY, f32::max);
    let sum_exp = legal_actions
        .iter()
        .map(|action| (logits[*action] - max_logit).exp())
        .sum::<f32>();
    let log_z = max_logit + sum_exp.ln();

    for action in legal_actions {
        log_priors[*action] = logits[*action] - log_z;
    }
    Ok(log_priors)
}

pub(crate) fn log_priors_from_priors(
    legal_actions: &[usize],
    priors: &[f32],
) -> PyResult<[f32; ACTION_SPACE]> {
    validate_policy_len(priors, "prior")?;
    if priors
        .iter()
        .any(|prior| !prior.is_finite() || *prior < 0.0)
    {
        return Err(PyValueError::new_err(
            "prior values must be finite non-negative values",
        ));
    }

    let mut log_priors = [f32::NEG_INFINITY; ACTION_SPACE];
    if legal_actions.is_empty() {
        return Ok(log_priors);
    }

    let legal_sum = legal_actions
        .iter()
        .map(|action| priors[*action])
        .sum::<f32>();
    if legal_sum <= 0.0 {
        let uniform_log = (1.0 / legal_actions.len() as f32).ln();
        for action in legal_actions {
            log_priors[*action] = uniform_log;
        }
        return Ok(log_priors);
    }

    for action in legal_actions {
        let normalized = (priors[*action] / legal_sum).max(PRIOR_EPSILON);
        log_priors[*action] = normalized.ln();
    }
    Ok(log_priors)
}

fn validate_policy_len(row: &[f32], name: &str) -> PyResult<()> {
    if row.len() != ACTION_SPACE {
        return Err(PyValueError::new_err(format!(
            "expected {ACTION_SPACE} {name} values, got {}",
            row.len()
        )));
    }
    Ok(())
}

fn validate_legal_values(legal_actions: &[usize], row: &[f32], name: &str) -> PyResult<()> {
    if legal_actions.iter().any(|action| !row[*action].is_finite()) {
        return Err(PyValueError::new_err(format!(
            "{name} values for legal actions must be finite"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{log_priors_from_logits, log_priors_from_priors};
    use crate::game::{ACTION_SPACE, CENTER_INDEX};
    use pretty_assertions::assert_eq;

    fn assert_close(left: f32, right: f32) {
        assert!((left - right).abs() < 1.0e-6, "{left} != {right}");
    }

    #[test]
    fn logits_are_log_softmaxed_over_legal_actions_only() {
        let legal = [0, 1];
        let mut logits = [0.0; ACTION_SPACE];
        logits[0] = 1.0;
        logits[1] = 3.0;
        logits[CENTER_INDEX] = f32::NAN;

        let log_priors = log_priors_from_logits(&legal, &logits).unwrap();

        let z = 1.0_f32.exp() + 3.0_f32.exp();
        assert_close(log_priors[0], (1.0_f32.exp() / z).ln());
        assert_close(log_priors[1], (3.0_f32.exp() / z).ln());
        assert_eq!(log_priors[CENTER_INDEX], f32::NEG_INFINITY);
    }

    #[test]
    fn logits_reject_non_finite_legal_values() {
        let legal = [0, 1];
        let mut logits = [0.0; ACTION_SPACE];
        logits[1] = f32::INFINITY;

        assert!(log_priors_from_logits(&legal, &logits).is_err());
    }

    #[test]
    fn priors_renormalize_over_legal_actions_only() {
        let legal = [0, 2];
        let mut priors = [0.0; ACTION_SPACE];
        priors[0] = 1.0;
        priors[1] = 100.0;
        priors[2] = 3.0;

        let log_priors = log_priors_from_priors(&legal, &priors).unwrap();

        assert_eq!(log_priors[0], 0.25_f32.ln());
        assert_eq!(log_priors[2], 0.75_f32.ln());
        assert_eq!(log_priors[1], f32::NEG_INFINITY);
    }

    #[test]
    fn all_zero_legal_priors_fall_back_to_uniform() {
        let legal = [0, 2, 4];
        let priors = [0.0; ACTION_SPACE];

        let log_priors = log_priors_from_priors(&legal, &priors).unwrap();

        let expected = (1.0_f32 / 3.0).ln();
        assert_eq!(log_priors[0], expected);
        assert_eq!(log_priors[2], expected);
        assert_eq!(log_priors[4], expected);
    }
}
