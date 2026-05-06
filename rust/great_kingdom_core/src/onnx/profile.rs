use std::{
    env,
    ops::AddAssign,
    time::{Duration, Instant},
};

use super::evaluator::OnnxDevice;

#[derive(Clone, Copy, Debug)]
pub(super) struct OnnxEvalProfile {
    enabled: bool,
    interval: u64,
    calls: u64,
}

impl OnnxEvalProfile {
    pub(super) fn new() -> Self {
        Self {
            enabled: env_flag("GKA_ONNX_EVAL_PROFILE"),
            interval: env::var("GKA_ONNX_EVAL_PROFILE_INTERVAL")
                .ok()
                .and_then(|value| value.parse::<u64>().ok())
                .filter(|value| *value > 0)
                .unwrap_or(1),
            calls: 0,
        }
    }

    pub(super) fn next_call(&mut self) -> Option<u64> {
        self.calls = self.calls.saturating_add(1);
        if self.enabled && self.calls % self.interval == 0 {
            Some(self.calls)
        } else {
            None
        }
    }

    pub(super) fn log(&self, log: OnnxEvalProfileLog) {
        let Some(call) = log.call else {
            return;
        };
        let context = log.context.unwrap_or_default();
        eprintln!(
            "[gka-onnx-eval] call={} source={} wave={} active_games={} leaves={} request_len={} max_batch_size={} chunk_count={} chunk_batches={} device={} feature_ref={:.6}s tensor_build={:.6}s session_run={:.6}s output_parse={:.6}s total_eval={:.6}s",
            call,
            context.source,
            format_optional_u64(context.wave),
            format_optional_usize(context.active_games),
            format_optional_usize(context.leaves),
            log.request_len,
            log.max_batch_size,
            log.chunk_batches.len(),
            format_chunk_batches(&log.chunk_batches),
            device_label(log.device),
            log.timing.feature_ref_elapsed.as_secs_f64(),
            log.timing.tensor_build_elapsed.as_secs_f64(),
            log.timing.session_run_elapsed.as_secs_f64(),
            log.timing.output_parse_elapsed.as_secs_f64(),
            log.total_elapsed.as_secs_f64(),
        );
    }
}

pub(super) struct OnnxEvalProfileLog {
    pub(super) call: Option<u64>,
    pub(super) context: Option<OnnxEvalProfileContext>,
    pub(super) request_len: usize,
    pub(super) max_batch_size: usize,
    pub(super) chunk_batches: Vec<usize>,
    pub(super) timing: OnnxEvalTiming,
    pub(super) total_elapsed: Duration,
    pub(super) device: OnnxDevice,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct OnnxEvalProfileContext {
    pub(super) source: &'static str,
    pub(super) wave: Option<u64>,
    pub(super) active_games: Option<usize>,
    pub(super) leaves: Option<usize>,
}

impl Default for OnnxEvalProfileContext {
    fn default() -> Self {
        Self {
            source: "direct",
            wave: None,
            active_games: None,
            leaves: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct OnnxEvalTiming {
    pub(super) feature_ref_elapsed: Duration,
    pub(super) tensor_build_elapsed: Duration,
    pub(super) session_run_elapsed: Duration,
    pub(super) output_parse_elapsed: Duration,
}

impl AddAssign for OnnxEvalTiming {
    fn add_assign(&mut self, rhs: Self) {
        self.feature_ref_elapsed += rhs.feature_ref_elapsed;
        self.tensor_build_elapsed += rhs.tensor_build_elapsed;
        self.session_run_elapsed += rhs.session_run_elapsed;
        self.output_parse_elapsed += rhs.output_parse_elapsed;
    }
}

pub(super) fn elapsed_since(start: Option<Instant>) -> Duration {
    start.map_or(Duration::ZERO, |start| start.elapsed())
}

fn device_label(device: OnnxDevice) -> &'static str {
    match device {
        OnnxDevice::Cpu => "cpu",
        OnnxDevice::Cuda => "cuda",
    }
}

fn format_optional_u64(value: Option<u64>) -> String {
    value.map_or_else(|| "-".to_string(), |value| value.to_string())
}

fn format_optional_usize(value: Option<usize>) -> String {
    value.map_or_else(|| "-".to_string(), |value| value.to_string())
}

fn format_chunk_batches(chunk_batches: &[usize]) -> String {
    if chunk_batches.is_empty() {
        "[]".to_string()
    } else {
        let values = chunk_batches
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",");
        format!("[{values}]")
    }
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

    #[test]
    fn profile_interval_logs_only_selected_calls() {
        let mut profile = OnnxEvalProfile {
            enabled: true,
            interval: 2,
            calls: 0,
        };

        assert_eq!(profile.next_call(), None);
        assert_eq!(profile.next_call(), Some(2));
        assert_eq!(profile.next_call(), None);
        assert_eq!(profile.next_call(), Some(4));
    }

    #[test]
    fn profile_formats_chunk_batches() {
        assert_eq!(format_chunk_batches(&[]), "[]");
        assert_eq!(format_chunk_batches(&[256, 256, 12]), "[256,256,12]");
    }
}
