use std::{fmt, path::Path};

use ort::{
    execution_providers,
    session::{Session, builder::GraphOptimizationLevel},
    value::Tensor,
};

use crate::{
    eval_request::EvalRequest,
    game::{ACTION_SPACE, BOARD_SIZE, FEATURE_CHANNELS},
};

const FEATURE_INPUT: &str = "features";
const POLICY_OUTPUT: &str = "policy_logits";
const VALUE_OUTPUT: &str = "value";
const FEATURE_VALUES_PER_POSITION: usize = FEATURE_CHANNELS * BOARD_SIZE * BOARD_SIZE;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OnnxDevice {
    Cpu,
    Cuda,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OnnxEvaluatorConfig {
    pub device: OnnxDevice,
    pub max_batch_size: usize,
}

impl Default for OnnxEvaluatorConfig {
    fn default() -> Self {
        Self {
            device: OnnxDevice::Cpu,
            max_batch_size: 256,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct NetworkOutput {
    pub policy_logits: Vec<[f32; ACTION_SPACE]>,
    pub values: Vec<f32>,
}

pub struct OnnxEvaluator {
    session: Session,
    config: OnnxEvaluatorConfig,
}

#[derive(Debug)]
pub enum OnnxError {
    InvalidConfig(String),
    InvalidRequest(String),
    InvalidOutput(String),
    Ort(ort::Error),
}

impl fmt::Display for OnnxError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(message)
            | Self::InvalidRequest(message)
            | Self::InvalidOutput(message) => formatter.write_str(message),
            Self::Ort(error) => write!(formatter, "{error}"),
        }
    }
}

impl std::error::Error for OnnxError {}

impl From<ort::Error> for OnnxError {
    fn from(error: ort::Error) -> Self {
        Self::Ort(error)
    }
}

impl OnnxEvaluator {
    pub fn load(path: impl AsRef<Path>, config: OnnxEvaluatorConfig) -> Result<Self, OnnxError> {
        validate_config(config)?;

        let mut builder = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_memory_pattern(false)?;

        builder = match config.device {
            OnnxDevice::Cpu => builder.with_execution_providers([
                execution_providers::CPUExecutionProvider::default().build(),
            ])?,
            OnnxDevice::Cuda => builder.with_execution_providers(cuda_execution_providers())?,
        };

        let session = builder.commit_from_file(path)?;
        validate_session_contract(&session)?;
        Ok(Self { session, config })
    }

    pub fn evaluate_request(&mut self, request: &EvalRequest) -> Result<NetworkOutput, OnnxError> {
        if request.is_empty() {
            return Ok(NetworkOutput {
                policy_logits: Vec::new(),
                values: Vec::new(),
            });
        }

        let features = request.feature_values();
        let expected = request.len() * FEATURE_VALUES_PER_POSITION;
        if features.len() != expected {
            return Err(OnnxError::InvalidRequest(format!(
                "expected {expected} feature values for {} positions, got {}",
                request.len(),
                features.len()
            )));
        }

        let mut policy_logits = Vec::with_capacity(request.len());
        let mut values = Vec::with_capacity(request.len());
        for chunk in features.chunks(self.config.max_batch_size * FEATURE_VALUES_PER_POSITION) {
            let chunk_batch = chunk.len() / FEATURE_VALUES_PER_POSITION;
            let output = self.evaluate_feature_chunk(chunk, chunk_batch)?;
            policy_logits.extend(output.policy_logits);
            values.extend(output.values);
        }

        Ok(NetworkOutput {
            policy_logits,
            values,
        })
    }

    fn evaluate_feature_chunk(
        &mut self,
        features: &[f32],
        batch_size: usize,
    ) -> Result<NetworkOutput, OnnxError> {
        let input = Tensor::from_array((
            [batch_size, FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE],
            features.to_vec().into_boxed_slice(),
        ))?;

        let outputs = self.session.run(ort::inputs![FEATURE_INPUT => input])?;
        let (_, policy_values) = outputs[POLICY_OUTPUT].try_extract_tensor::<f32>()?;
        let (_, value_values) = outputs[VALUE_OUTPUT].try_extract_tensor::<f32>()?;

        parse_network_output(policy_values, value_values, batch_size)
    }
}

fn validate_config(config: OnnxEvaluatorConfig) -> Result<(), OnnxError> {
    if config.max_batch_size == 0 {
        return Err(OnnxError::InvalidConfig(
            "max_batch_size must be positive".to_string(),
        ));
    }
    if matches!(config.device, OnnxDevice::Cuda) && !cfg!(feature = "onnx-cuda") {
        return Err(OnnxError::InvalidConfig(
            "onnx-cuda feature is required for CUDA inference".to_string(),
        ));
    }
    Ok(())
}

fn validate_session_contract(session: &Session) -> Result<(), OnnxError> {
    if !session
        .inputs
        .iter()
        .any(|input| input.name == FEATURE_INPUT)
    {
        return Err(OnnxError::InvalidOutput(format!(
            "ONNX model must have an input named {FEATURE_INPUT:?}"
        )));
    }

    for name in [POLICY_OUTPUT, VALUE_OUTPUT] {
        if !session.outputs.iter().any(|output| output.name == name) {
            return Err(OnnxError::InvalidOutput(format!(
                "ONNX model must have an output named {name:?}"
            )));
        }
    }

    Ok(())
}

fn parse_network_output(
    policy_values: &[f32],
    value_values: &[f32],
    batch_size: usize,
) -> Result<NetworkOutput, OnnxError> {
    let expected_policy_len = batch_size * ACTION_SPACE;
    if policy_values.len() != expected_policy_len {
        return Err(OnnxError::InvalidOutput(format!(
            "expected {expected_policy_len} policy logits, got {}",
            policy_values.len()
        )));
    }
    if value_values.len() != batch_size {
        return Err(OnnxError::InvalidOutput(format!(
            "expected {batch_size} values, got {}",
            value_values.len()
        )));
    }
    if policy_values.iter().any(|value| !value.is_finite()) {
        return Err(OnnxError::InvalidOutput(
            "policy logits must be finite".to_string(),
        ));
    }
    if value_values.iter().any(|value| !value.is_finite()) {
        return Err(OnnxError::InvalidOutput(
            "values must be finite".to_string(),
        ));
    }

    let policy_logits = policy_values
        .chunks_exact(ACTION_SPACE)
        .map(|row| {
            let mut logits = [0.0; ACTION_SPACE];
            logits.copy_from_slice(row);
            logits
        })
        .collect();

    Ok(NetworkOutput {
        policy_logits,
        values: value_values.to_vec(),
    })
}

#[cfg(feature = "onnx-cuda")]
fn cuda_execution_providers() -> Vec<execution_providers::ExecutionProviderDispatch> {
    vec![
        execution_providers::CUDAExecutionProvider::default().build(),
        execution_providers::CPUExecutionProvider::default().build(),
    ]
}

#[cfg(not(feature = "onnx-cuda"))]
fn cuda_execution_providers() -> Vec<execution_providers::ExecutionProviderDispatch> {
    unreachable!("CUDA config is rejected unless the onnx-cuda feature is enabled")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_rejects_zero_max_batch_size() {
        let error = validate_config(OnnxEvaluatorConfig {
            device: OnnxDevice::Cpu,
            max_batch_size: 0,
        })
        .unwrap_err();

        assert_eq!(error.to_string(), "max_batch_size must be positive");
    }

    #[cfg(not(feature = "onnx-cuda"))]
    #[test]
    fn config_rejects_cuda_without_feature() {
        let error = validate_config(OnnxEvaluatorConfig {
            device: OnnxDevice::Cuda,
            max_batch_size: 1,
        })
        .unwrap_err();

        assert_eq!(
            error.to_string(),
            "onnx-cuda feature is required for CUDA inference"
        );
    }

    #[test]
    fn parse_network_output_accepts_valid_batch() {
        let output = parse_network_output(&vec![0.25; ACTION_SPACE * 2], &[0.5, -0.5], 2)
            .expect("valid output should parse");

        assert_eq!(output.policy_logits.len(), 2);
        assert_eq!(output.values, vec![0.5, -0.5]);
        assert_eq!(output.policy_logits[0][0], 0.25);
    }

    #[test]
    fn parse_network_output_rejects_bad_policy_shape() {
        let error = parse_network_output(&vec![0.0; ACTION_SPACE - 1], &[0.0], 1).unwrap_err();

        assert_eq!(
            error.to_string(),
            format!(
                "expected {ACTION_SPACE} policy logits, got {}",
                ACTION_SPACE - 1
            )
        );
    }

    #[test]
    fn parse_network_output_rejects_non_finite_values() {
        let error = parse_network_output(&vec![0.0; ACTION_SPACE], &[f32::NAN], 1).unwrap_err();

        assert_eq!(error.to_string(), "values must be finite");
    }
}
