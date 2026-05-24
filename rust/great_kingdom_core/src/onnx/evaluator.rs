use std::{env, fmt, path::Path, time::Instant};

use ort::{
    execution_providers,
    session::{Session, builder::GraphOptimizationLevel},
    value::Tensor,
};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};

use crate::{
    eval_request::EvalRequest,
    game::{ACTION_SPACE, BOARD_SIZE, FEATURE_CHANNELS},
};

use super::profile::{
    OnnxEvalProfile, OnnxEvalProfileContext, OnnxEvalProfileLog, OnnxEvalTiming, elapsed_since,
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

#[pyclass(unsendable)]
pub struct OnnxEvaluator {
    session: Session,
    config: OnnxEvaluatorConfig,
    batch_buckets: BatchBucketConfig,
    profile: OnnxEvalProfile,
    profile_context: Option<OnnxEvalProfileContext>,
}

#[derive(Clone, Debug, PartialEq)]
struct BatchBucketConfig {
    enabled: bool,
    buckets: Vec<usize>,
    warmup_batches: Vec<usize>,
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

#[pymethods]
impl OnnxEvaluator {
    #[new]
    #[pyo3(signature = (path, device = "cpu", max_batch_size = 256))]
    fn py_new(path: &str, device: &str, max_batch_size: usize) -> PyResult<Self> {
        Self::load(
            path,
            OnnxEvaluatorConfig {
                device: parse_device(device)?,
                max_batch_size,
            },
        )
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }

    #[must_use]
    fn device(&self) -> &'static str {
        match self.config.device {
            OnnxDevice::Cpu => "cpu",
            OnnxDevice::Cuda => "cuda",
        }
    }

    #[must_use]
    fn max_batch_size(&self) -> usize {
        self.config.max_batch_size
    }

    fn evaluate(&mut self, request: &EvalRequest) -> PyResult<(Vec<Vec<f32>>, Vec<f32>)> {
        let output = self
            .evaluate_request(request)
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
        Ok((
            output.policy_logits.into_iter().map(Vec::from).collect(),
            output.values,
        ))
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
        let batch_buckets = BatchBucketConfig::from_env(config.max_batch_size)?;
        let mut evaluator = Self {
            session,
            config,
            batch_buckets,
            profile: OnnxEvalProfile::new(),
            profile_context: None,
        };
        evaluator.warm_up_bucket_batches()?;
        Ok(evaluator)
    }

    pub fn evaluate_request(&mut self, request: &EvalRequest) -> Result<NetworkOutput, OnnxError> {
        let profile_call = self.profile.next_call();
        let profile_context = self.profile_context.take();
        let total_start = profile_call.map(|_| Instant::now());
        if request.is_empty() {
            self.profile.log(OnnxEvalProfileLog {
                call: profile_call,
                context: profile_context,
                request_len: 0,
                max_batch_size: self.config.max_batch_size,
                chunk_batches: Vec::new(),
                timing: OnnxEvalTiming::default(),
                total_elapsed: elapsed_since(total_start),
                device: self.config.device,
            });
            return Ok(NetworkOutput {
                policy_logits: Vec::new(),
                values: Vec::new(),
            });
        }

        let feature_ref_start = profile_call.map(|_| Instant::now());
        let features = request.feature_values_ref();
        let mut timing = OnnxEvalTiming {
            feature_ref_elapsed: elapsed_since(feature_ref_start),
            ..OnnxEvalTiming::default()
        };
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
        let mut chunk_batches = Vec::new();
        for start_row in (0..request.len()).step_by(self.config.max_batch_size) {
            let end_row = (start_row + self.config.max_batch_size).min(request.len());
            let actual_batch = end_row - start_row;
            let tensor_batch = self.batch_buckets.tensor_batch_size(actual_batch);
            let chunk_start = start_row * FEATURE_VALUES_PER_POSITION;
            let chunk_end = end_row * FEATURE_VALUES_PER_POSITION;
            let chunk = &features[chunk_start..chunk_end];
            if profile_call.is_some() {
                chunk_batches.push(tensor_batch);
            }
            let (output, chunk_timing) = self.evaluate_feature_chunk(
                chunk,
                actual_batch,
                tensor_batch,
                profile_call.is_some(),
            )?;
            timing += chunk_timing;
            policy_logits.extend(output.policy_logits);
            values.extend(output.values);
        }

        self.profile.log(OnnxEvalProfileLog {
            call: profile_call,
            context: profile_context,
            request_len: request.len(),
            max_batch_size: self.config.max_batch_size,
            chunk_batches,
            timing,
            total_elapsed: elapsed_since(total_start),
            device: self.config.device,
        });

        Ok(NetworkOutput {
            policy_logits,
            values,
        })
    }

    pub(crate) fn set_gumbel_root_profile_context(&mut self, active_games: usize) {
        self.profile_context = Some(OnnxEvalProfileContext {
            source: "gumbel_root",
            wave: None,
            active_games: Some(active_games),
            leaves: Some(active_games),
        });
    }

    pub(crate) fn set_gumbel_leaf_profile_context(
        &mut self,
        wave: u64,
        active_games: usize,
        leaves: usize,
    ) {
        self.profile_context = Some(OnnxEvalProfileContext {
            source: "gumbel_leaf",
            wave: Some(wave),
            active_games: Some(active_games),
            leaves: Some(leaves),
        });
    }

    fn evaluate_feature_chunk(
        &mut self,
        features: &[f32],
        actual_batch_size: usize,
        tensor_batch_size: usize,
        profile_enabled: bool,
    ) -> Result<(NetworkOutput, OnnxEvalTiming), OnnxError> {
        let tensor_start = profile_enabled.then(Instant::now);
        let tensor_features =
            tensor_features_with_padding(features, actual_batch_size, tensor_batch_size)?;
        let input = Tensor::from_array((
            [tensor_batch_size, FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE],
            tensor_features,
        ))?;
        let tensor_build_elapsed = elapsed_since(tensor_start);

        let session_start = profile_enabled.then(Instant::now);
        let outputs = self.session.run(ort::inputs![FEATURE_INPUT => input])?;
        let session_run_elapsed = elapsed_since(session_start);

        let output_parse_start = profile_enabled.then(Instant::now);
        let (_, policy_values) = outputs[POLICY_OUTPUT].try_extract_tensor::<f32>()?;
        let (_, value_values) = outputs[VALUE_OUTPUT].try_extract_tensor::<f32>()?;

        let output = parse_network_output(
            policy_values,
            value_values,
            actual_batch_size,
            tensor_batch_size,
        )?;
        Ok((
            output,
            OnnxEvalTiming {
                tensor_build_elapsed,
                session_run_elapsed,
                output_parse_elapsed: elapsed_since(output_parse_start),
                ..OnnxEvalTiming::default()
            },
        ))
    }

    fn warm_up_bucket_batches(&mut self) -> Result<(), OnnxError> {
        let warmup_batches = self.batch_buckets.warmup_batches.clone();
        for batch_size in warmup_batches {
            let features = vec![0.0; batch_size * FEATURE_VALUES_PER_POSITION];
            let _ = self.evaluate_feature_chunk(&features, batch_size, batch_size, false)?;
        }
        Ok(())
    }
}

impl BatchBucketConfig {
    fn from_env(max_batch_size: usize) -> Result<Self, OnnxError> {
        let enabled = env_flag("GKA_ONNX_BATCH_BUCKETING");
        let buckets = if enabled {
            parse_batch_sizes_env("GKA_ONNX_BATCH_BUCKETS")?
                .unwrap_or_else(|| default_batch_buckets(max_batch_size))
        } else {
            Vec::new()
        };
        let buckets = sanitize_batch_sizes(buckets, max_batch_size);
        if enabled && buckets.is_empty() {
            return Err(OnnxError::InvalidConfig(
                "GKA_ONNX_BATCH_BUCKETS did not contain any valid bucket sizes".to_string(),
            ));
        }

        let warmup_batches = if enabled && env_flag_default_true("GKA_ONNX_BUCKET_WARMUP") {
            parse_batch_sizes_env("GKA_ONNX_WARMUP_BUCKETS")?.unwrap_or_else(|| buckets.clone())
        } else {
            Vec::new()
        };
        let warmup_batches = sanitize_batch_sizes(warmup_batches, max_batch_size);

        Ok(Self {
            enabled,
            buckets,
            warmup_batches,
        })
    }

    fn tensor_batch_size(&self, actual_batch_size: usize) -> usize {
        if !self.enabled {
            return actual_batch_size;
        }
        self.buckets
            .iter()
            .copied()
            .find(|bucket| *bucket >= actual_batch_size)
            .unwrap_or(actual_batch_size)
    }
}

fn parse_device(device: &str) -> PyResult<OnnxDevice> {
    match device {
        "cpu" => Ok(OnnxDevice::Cpu),
        "cuda" => Ok(OnnxDevice::Cuda),
        other => Err(PyValueError::new_err(format!(
            "unsupported ONNX device {other:?}; expected 'cpu' or 'cuda'"
        ))),
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

fn tensor_features_with_padding(
    features: &[f32],
    actual_batch_size: usize,
    tensor_batch_size: usize,
) -> Result<Box<[f32]>, OnnxError> {
    let expected_actual = actual_batch_size * FEATURE_VALUES_PER_POSITION;
    if features.len() != expected_actual {
        return Err(OnnxError::InvalidRequest(format!(
            "expected {expected_actual} feature values for chunk, got {}",
            features.len()
        )));
    }
    if tensor_batch_size < actual_batch_size {
        return Err(OnnxError::InvalidRequest(format!(
            "tensor batch size {tensor_batch_size} is smaller than actual batch size {actual_batch_size}"
        )));
    }
    if tensor_batch_size == actual_batch_size {
        return Ok(features.to_vec().into_boxed_slice());
    }

    let mut padded = vec![0.0; tensor_batch_size * FEATURE_VALUES_PER_POSITION];
    padded[..features.len()].copy_from_slice(features);
    Ok(padded.into_boxed_slice())
}

fn parse_network_output(
    policy_values: &[f32],
    value_values: &[f32],
    actual_batch_size: usize,
    tensor_batch_size: usize,
) -> Result<NetworkOutput, OnnxError> {
    if tensor_batch_size < actual_batch_size {
        return Err(OnnxError::InvalidOutput(format!(
            "tensor batch size {tensor_batch_size} is smaller than actual batch size {actual_batch_size}"
        )));
    }
    let expected_policy_len = tensor_batch_size * ACTION_SPACE;
    if policy_values.len() != expected_policy_len {
        return Err(OnnxError::InvalidOutput(format!(
            "expected {expected_policy_len} policy logits, got {}",
            policy_values.len()
        )));
    }
    if value_values.len() != tensor_batch_size {
        return Err(OnnxError::InvalidOutput(format!(
            "expected {tensor_batch_size} values, got {}",
            value_values.len()
        )));
    }
    let actual_policy_values = &policy_values[..actual_batch_size * ACTION_SPACE];
    let actual_value_values = &value_values[..actual_batch_size];
    if actual_policy_values.iter().any(|value| !value.is_finite()) {
        return Err(OnnxError::InvalidOutput(
            "policy logits must be finite".to_string(),
        ));
    }
    if actual_value_values.iter().any(|value| !value.is_finite()) {
        return Err(OnnxError::InvalidOutput(
            "values must be finite".to_string(),
        ));
    }

    let policy_logits = actual_policy_values
        .chunks_exact(ACTION_SPACE)
        .map(|row| {
            let mut logits = [0.0; ACTION_SPACE];
            logits.copy_from_slice(row);
            logits
        })
        .collect();

    Ok(NetworkOutput {
        policy_logits,
        values: actual_value_values.to_vec(),
    })
}

fn default_batch_buckets(max_batch_size: usize) -> Vec<usize> {
    let mut buckets = Vec::new();
    let mut bucket = 32;
    while bucket < max_batch_size {
        buckets.push(bucket);
        bucket *= 2;
    }
    buckets.push(max_batch_size);
    buckets
}

fn parse_batch_sizes_env(name: &str) -> Result<Option<Vec<usize>>, OnnxError> {
    let Ok(value) = env::var(name) else {
        return Ok(None);
    };
    if value.trim().is_empty() {
        return Ok(None);
    }
    let mut sizes = Vec::new();
    for item in value.split(',') {
        let item = item.trim();
        if item.is_empty() {
            continue;
        }
        let size = item.parse::<usize>().map_err(|err| {
            OnnxError::InvalidConfig(format!("invalid {name} batch size {item:?}: {err}"))
        })?;
        sizes.push(size);
    }
    Ok(Some(sizes))
}

fn sanitize_batch_sizes(mut sizes: Vec<usize>, max_batch_size: usize) -> Vec<usize> {
    sizes.retain(|size| *size > 0 && *size <= max_batch_size);
    sizes.sort_unstable();
    sizes.dedup();
    sizes
}

fn env_flag(name: &str) -> bool {
    !matches!(
        env::var(name).as_deref(),
        Err(_) | Ok("") | Ok("0") | Ok("false") | Ok("False") | Ok("no") | Ok("No")
    )
}

fn env_flag_default_true(name: &str) -> bool {
    !matches!(
        env::var(name).as_deref(),
        Ok("0") | Ok("false") | Ok("False") | Ok("no") | Ok("No")
    )
}

#[cfg(feature = "onnx-cuda")]
fn cuda_execution_providers() -> Vec<execution_providers::ExecutionProviderDispatch> {
    vec![
        execution_providers::CUDAExecutionProvider::default()
            .build()
            .error_on_failure(),
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
        let output = parse_network_output(&vec![0.25; ACTION_SPACE * 2], &[0.5, -0.5], 2, 2)
            .expect("valid output should parse");

        assert_eq!(output.policy_logits.len(), 2);
        assert_eq!(output.values, vec![0.5, -0.5]);
        assert_eq!(output.policy_logits[0][0], 0.25);
    }

    #[test]
    fn parse_network_output_ignores_padded_rows() {
        let mut policy = vec![0.25; ACTION_SPACE * 4];
        policy[ACTION_SPACE * 2] = f32::NAN;
        let output = parse_network_output(&policy, &[0.5, -0.5, f32::NAN, f32::NAN], 2, 4)
            .expect("padded rows should be ignored");

        assert_eq!(output.policy_logits.len(), 2);
        assert_eq!(output.values, vec![0.5, -0.5]);
    }

    #[test]
    fn parse_network_output_rejects_bad_policy_shape() {
        let error = parse_network_output(&vec![0.0; ACTION_SPACE - 1], &[0.0], 1, 1).unwrap_err();

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
        let error = parse_network_output(&vec![0.0; ACTION_SPACE], &[f32::NAN], 1, 1).unwrap_err();

        assert_eq!(error.to_string(), "values must be finite");
    }

    #[test]
    fn tensor_features_with_padding_zero_fills_extra_rows() {
        let features = vec![1.0; FEATURE_VALUES_PER_POSITION * 2];

        let padded = tensor_features_with_padding(&features, 2, 4).expect("padding should work");

        assert_eq!(padded.len(), FEATURE_VALUES_PER_POSITION * 4);
        assert!(padded[..features.len()].iter().all(|value| *value == 1.0));
        assert!(padded[features.len()..].iter().all(|value| *value == 0.0));
    }

    #[test]
    fn batch_bucket_config_rounds_up_to_next_bucket() {
        let config = BatchBucketConfig {
            enabled: true,
            buckets: vec![32, 64, 128, 256],
            warmup_batches: Vec::new(),
        };

        assert_eq!(config.tensor_batch_size(48), 64);
        assert_eq!(config.tensor_batch_size(114), 128);
        assert_eq!(config.tensor_batch_size(228), 256);
        assert_eq!(config.tensor_batch_size(300), 300);
    }

    #[test]
    fn batch_bucket_helpers_filter_and_sort_sizes() {
        assert_eq!(default_batch_buckets(256), vec![32, 64, 128, 256]);
        assert_eq!(
            sanitize_batch_sizes(vec![0, 256, 64, 64, 8192, 32], 512),
            vec![32, 64, 256]
        );
    }
}
