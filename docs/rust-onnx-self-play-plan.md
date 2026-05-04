# Rust ONNX Self-Play 구현 계획

## 목표

Self-play 추론을 Python/PyTorch 콜백 경로에서 Rust/ONNX Runtime 경로로 옮겨 Runpod RTX 3090 환경에서 데이터 생성 속도를 높인다.

최종 구조는 다음과 같다.

- Python: PyTorch 학습, checkpoint 저장, ONNX export
- Rust: 규칙 엔진, Gumbel search, ONNX 모델 추론, self-play 데이터 생성
- 기존 Python pipeline: PyTorch self-play 경로를 안정적인 legacy/fallback 경로로 유지
- 신규 Rust ONNX pipeline: Rust가 만든 self-play 데이터를 replay buffer로 적재하고 학습 수행

## 현재 구조

현재 프로젝트는 이미 Rust self-play로 확장하기 좋은 형태다.

- Rust `great_kingdom_core`가 규칙 엔진과 Gumbel search를 담당한다.
- Python `self_play.py`와 `pipeline.py`가 self-play 실행 흐름을 담당한다.
- Rust `EvalRequest`가 feature plane과 legal mask bytes를 만들어 Python evaluator에 넘긴다.
- Python `evaluator.py`가 PyTorch 모델을 호출하고 policy logits/value를 Rust search에 돌려준다.

현재 병목 후보는 다음과 같다.

- root/leaf evaluation마다 Python 콜백 진입
- NumPy 배열 생성 및 PyTorch tensor 변환
- 반복적인 CPU/GPU 전송
- Python orchestration 비용

## 제외 범위

- 기존 PyTorch self-play 경로를 즉시 삭제하지 않는다.
- 로컬 CPU 개발 환경에는 ONNX Runtime CPU provider를 포함한다.
- 로컬 CPU 개발 환경에 ONNX Runtime CUDA 의존성은 강제하지 않는다.
- 첫 구현에서는 별도 async dynamic batching queue를 만들지 않는다.
- 첫 단계에서 TensorRT 최적화까지 포함하지 않는다.
- 첫 단계에서 FP16/mixed precision 최적화까지 포함하지 않는다.
- replay 저장 포맷을 한 번에 크게 바꾸지 않는다.
- 기존 legacy pipeline 데이터를 in-place로 변경하지 않는다.

## 설계 원칙

- 기존 Python/PyTorch evaluator는 fallback 및 정답 비교용으로 유지한다.
- 기존 `pipeline.py`는 큰 backend switch를 넣지 않고 legacy pipeline으로 유지한다.
- Rust ONNX orchestration은 별도 pipeline 모듈과 CLI로 만든다.
- Rust ONNX CPU inference는 기본 개발 경로에 포함한다.
- CUDA provider 지원은 `Cargo feature` 또는 Runpod 전용 빌드 설정으로 분리한다.
- 수치 일치 테스트를 먼저 만들고 성능 최적화는 그 다음에 한다.
- 파일이 비대해지면 ONNX evaluator, export, Rust runner wrapper, pipeline, legacy data migration을 역할별로 분리한다.
- ONNX Runtime CPU provider는 로컬 검증 기준에 포함하고, CUDA provider 검증은 Runpod로 분리한다.

## 첫 구현 범위

첫 구현은 Rust ONNX inference 경로가 정확히 동작하는지 검증하는 데 집중한다.

포함:

- FP32 ONNX export
- opset 17 기준 export
- batch axis dynamic axes 설정
- 공통 `OnnxEvaluator`
- 기존 `GumbelSelfPlayBatch`가 만드는 root/leaf batch를 ONNX evaluator에 연결
- CPU provider 로컬 테스트
- Runpod CUDA provider smoke test
- `safetensors` 기반 replay tensor 저장과 JSON/JSONL metadata 저장
- 별도 Rust ONNX pipeline CLI
- legacy pipeline replay/checkpoint/log import 기능

제외:

- 여러 self-play worker의 요청을 channel로 모으는 별도 dynamic batching queue
- session pool 또는 async inference scheduler
- FP16/mixed precision/TensorRT 최적화
- arena ONNX 전환
- 기존 `pipeline.py`에 대규모 `rust_onnx` backend 분기 추가

별도 dynamic batching queue는 기존 `GumbelSelfPlayBatch` 기반 구현의 batch size, GPU utilization, wall time을 측정한 뒤 2차 최적화로 판단한다.

## 제안 구조

### 1. Python ONNX Export

Python checkpoint를 ONNX 모델로 내보내는 기능을 추가한다.

입력:

- PyTorch checkpoint `.pt`
- model preset/config
- 출력 경로 `.onnx`

ONNX 모델 계약:

- 입력: `features`, shape `[batch, FEATURE_CHANNELS, 9, 9]`, dtype `float32`
- 출력 1: `policy_logits`, shape `[batch, ACTION_SPACE]`, dtype `float32`
- 출력 2: `value`, shape `[batch]`, dtype `float32`
- opset: 17
- dynamic batch axis 지원: 입력과 두 출력의 0번째 axis를 `batch`로 둔다.
- 첫 구현 precision: FP32

export 세부 설정:

```python
model.eval()
with torch.inference_mode():
    torch.onnx.export(
        model,
        dummy_features,
        output_path,
        input_names=["features"],
        output_names=["policy_logits", "value"],
        dynamic_axes={
            "features": {0: "batch"},
            "policy_logits": {0: "batch"},
            "value": {0: "batch"},
        },
        opset_version=17,
    )
```

BatchNorm 계층이 있으므로 export와 parity test는 반드시 eval mode에서 수행한다.

FP16 또는 ONNX Runtime mixed precision은 첫 구현에서 제외한다. CUDA smoke와 throughput 기준선이 확보된 뒤 Tensor Core 활용을 위해 별도 최적화 단계에서 검토한다.

추가 후보 파일:

- `python/great_kingdom_ai/onnx_export.py`
- `tests/test_onnx_export.py`

### 2. Rust ONNX Evaluator

Rust core에 ONNX evaluator를 추가한다.

추가 후보 파일:

- `rust/great_kingdom_core/src/onnx/mod.rs`
- `rust/great_kingdom_core/src/onnx/evaluator.rs`

Cargo 설정:

- `ort` dependency 추가
- CPU provider는 로컬 기본 빌드/테스트에서 사용한다.
- CUDA provider는 별도 feature 또는 Runpod 전용 빌드 설정으로 분리한다.

`OnnxEvaluator`는 CPU/GPU용으로 나누지 않는다. 공통 evaluator 하나를 두고, load 단계에서 ONNX Runtime execution provider만 device 옵션에 따라 다르게 설정한다. feature/mask 처리, batch shape 검증, policy/value 출력 변환은 CPU와 CUDA가 같은 코드를 사용한다.

첫 구현에서는 `OnnxEvaluator`가 기존 Rust batch search가 만든 `EvalRequest`를 즉시 평가한다. 여러 worker의 요청을 별도 channel로 모으는 dynamic batching queue는 만들지 않는다.

`EvalRequest.len()`이 `max_batch_size`를 넘으면 `OnnxEvaluator`가 내부에서 여러 chunk로 나누어 순차 실행하고 결과를 다시 합친다. 이 방식은 GPU 효율을 극대화하는 최종 batching 전략은 아니지만, 첫 구현에서 메모리 사용량을 제한하고 shape 처리를 단순하게 유지한다.

초기 API 형태:

```rust
pub struct OnnxEvaluator {
    // ONNX Runtime session and buffers
}

pub struct OnnxEvaluatorConfig {
    pub device: OnnxDevice,
    pub max_batch_size: usize,
}

pub struct NetworkOutput {
    pub policy_logits: Vec<[f32; ACTION_SPACE]>,
    pub values: Vec<f32>,
}

impl OnnxEvaluator {
    pub fn load(path: impl AsRef<Path>, config: OnnxEvaluatorConfig) -> Result<Self, OnnxError>;
    pub fn evaluate_request(&mut self, request: &EvalRequest) -> Result<NetworkOutput, OnnxError>;
}
```

지원 device:

- `cpu`
- `cuda`

Session 공유:

- 첫 구현에서는 단일 self-play runner가 하나의 `OnnxEvaluator`를 소유한다.
- `ort::Session`은 thread-safe 특성을 활용할 수 있지만, Rust API의 실행 메서드가 mutable access를 요구할 수 있으므로 첫 구현에서 무리하게 `Arc<Session>` 공유 구조를 만들지 않는다.
- 다중 runner가 필요해지면 2차 최적화에서 single inference worker, `Arc<Mutex<Session>>`, session pool 중 측정 결과에 맞는 방식을 선택한다.

### 3. Search Evaluator Trait

Python callback과 ONNX evaluator를 같은 search path에서 사용할 수 있도록 trait로 분리한다.

후보 trait:

```rust
pub trait GumbelEvaluator {
    fn evaluate(&mut self, request: EvalRequest) -> Result<GumbelEvalBatch, EvalError>;
}
```

구현체:

- Python callback adapter
- ONNX Runtime adapter

이렇게 하면 기존 PyO3 메서드는 유지하면서, Rust 내부에서 Python 콜백 없이 self-play를 실행할 수 있다.

현재 `GumbelSelfPlayBatch`의 evaluator search 루프는 PyO3 callback에 직접 묶여 있다. 따라서 첫 구현에서는 단순히 trait만 추가하지 않고, 다음 순서로 분리한다.

1. leaf selection, request 생성, backup, result 생성 로직을 순수 Rust evaluator를 받을 수 있는 내부 함수로 분리한다.
2. 기존 PyO3 메서드는 Python callback adapter를 만들어 내부 함수에 위임한다.
3. ONNX evaluator adapter도 같은 내부 함수에 연결한다.

이렇게 해야 기존 Python 테스트와 API를 유지하면서 Rust ONNX 경로가 Python callback 없이 실행된다.

### 4. Rust Self-Play Runner

Rust에서 self-play game batch를 독립 실행하는 runner를 추가한다.

담당 역할:

- ONNX 모델 로드
- `GumbelSelfPlayBatch` 생성
- Rust 내부에서 root/leaf inference 실행
- 선택된 action 적용
- move log와 replay sample 수집
- 결과를 파일로 저장

첫 구현의 출력 포맷 후보:

- self-play log: JSON
- replay tensor: `safetensors`
- replay metadata: JSON 또는 JSONL

권장 첫 단계:

- Rust는 tensor 데이터와 metadata를 분리해 쓴다.
- Python이 해당 파일을 읽어서 기존 `ReplayBuffer`에 넣는다.
- 병목이 확인된 뒤 필요하면 direct `.npz` 또는 memory-mapped 포맷으로 최적화한다.

초기 저장 스펙:

- `samples.safetensors`
  - `features`: shape `[samples, FEATURE_CHANNELS, 9, 9]`, dtype `float32`
  - `policy`: shape `[samples, ACTION_SPACE]`, dtype `float32`
  - `value`: shape `[samples]`, dtype `float32`
- `games.jsonl`
  - game별 `seed`, `moves`, `winner`, `end_reason`, `territory_scores`
- `manifest.json`
  - `format_version`
  - `board_size`
  - `feature_channels`
  - `action_space`
  - `model_path`
  - `onnx_device`
  - `gumbel_config`
  - `created_at`

### 5. Pipeline 연동

기존 `pipeline.py`에 큰 backend switch를 넣지 않고 Rust ONNX 전용 pipeline을 새로 만든다.

추가 후보 파일:

- `python/great_kingdom_ai/onnx_export.py`
- `python/great_kingdom_ai/rust_onnx_self_play.py`
- `python/great_kingdom_ai/rust_onnx_replay.py`
- `python/great_kingdom_ai/rust_onnx_pipeline.py`
- `tests/test_rust_onnx_pipeline.py`
- `tests/test_rust_onnx_replay.py`

추가 CLI:

- `great-kingdom-export-onnx`
- `great-kingdom-rust-onnx-pipeline`
- `great-kingdom-import-legacy-pipeline`

Rust ONNX pipeline config 예시:

```json
{
  "work_dir": "data/runpod/onnx-pipeline",
  "legacy_import_dir": "data/runpod/pipeline",
  "import_legacy_on_first_run": true,
  "onnx_model_path": "data/runpod/onnx-pipeline/checkpoints/best.onnx",
  "onnx_device": "cuda",
  "onnx_max_batch_size": 1024,
  "rust_self_play_batch_size": 256
}
```

신규 pipeline의 책임:

- legacy pipeline에서 replay/checkpoint/log를 import한다.
- best checkpoint를 ONNX로 export한다.
- Rust ONNX self-play runner를 실행한다.
- Rust self-play artifact를 기존 `ReplayBuffer`에 import한다.
- PyTorch로 candidate를 학습한다.
- arena/promotion은 초기에는 기존 Python/PyTorch 경로를 호출한다.

신규 pipeline iteration 흐름:

1. work dir을 초기화한다.
2. 첫 실행이고 `import_legacy_on_first_run=true`이면 legacy 데이터를 import한다.
3. best checkpoint가 없으면 legacy best checkpoint를 복사하거나 새 checkpoint를 초기화한다.
4. best checkpoint를 FP32 ONNX(opset 17, dynamic batch axis)로 export한다.
5. Rust ONNX self-play를 실행한다.
6. 생성된 replay sample을 추가한다.
7. PyTorch로 candidate를 학습한다.
8. arena/promotion은 초기에는 기존 경로를 유지한다.

Arena는 self-play 경로가 안정화된 뒤 별도 단계에서 ONNX로 옮긴다.

### 6. Legacy Pipeline Data Import

기존 legacy pipeline 데이터는 신규 Rust ONNX pipeline에서 재사용할 수 있게 import 기능을 둔다.

legacy 입력 구조:

- `legacy_work_dir/replay/replay.npz`
- `legacy_work_dir/replay/game_logs.json`
- `legacy_work_dir/checkpoints/best.pt`
- `legacy_work_dir/checkpoints/candidate.pt`
- `legacy_work_dir/reports/metrics.jsonl`

신규 pipeline import 결과:

- `onnx_work_dir/replay/replay.npz`
- `onnx_work_dir/replay/game_logs.json`
- `onnx_work_dir/checkpoints/best.pt`
- `onnx_work_dir/reports/legacy-import.json`

import 정책:

- replay는 기존 `ReplayBuffer.load()`로 읽고 신규 capacity에 맞게 다시 저장한다.
- policy target sum, non-negative, value range 검증은 기존 `ReplayBuffer` 검증을 그대로 사용한다.
- best checkpoint는 ONNX export의 원본이므로 가능한 한 그대로 복사한다.
- game logs는 seed 충돌 방지를 위해 신규 pipeline의 `seed_start`를 `max(existing seeds)+1`로 잡을 수 있게 한다.
- metrics는 신규 pipeline iteration count로 직접 이어 붙이지 않고 `legacy-import.json`에 provenance로 보관한다.
- import는 idempotent하게 만든다. 이미 import marker가 있으면 기본적으로 다시 실행하지 않는다.

추가 명령 예시:

```bash
great-kingdom-import-legacy-pipeline \
  --legacy-work-dir data/runpod/pipeline \
  --onnx-work-dir data/runpod/onnx-pipeline \
  --replay-capacity 500000
```

신규 pipeline은 `--legacy-work-dir` 또는 config의 `legacy_import_dir`가 지정되어 있고 첫 실행이면 같은 import 로직을 호출한다.

## 구현 단계

### ~~1단계: Export와 수치 일치 테스트~~

작업:

- ONNX export 함수와 CLI를 추가한다.
- ONNX 검증용 Python dependency를 개발 의존성에 추가한다.
- opset 17, FP32, dynamic batch axis를 export 기본값으로 둔다.
- random feature batch에서 PyTorch 출력과 ONNX 출력을 비교한다.
- export 명령을 문서화한다.

검증 기준:

- policy logits shape가 `[batch, 82]`이다.
- value shape가 `[batch]`이다.
- FP32 CPU 기준 max absolute diff가 `1e-5` 안에 들어온다.

### ~~2단계: Rust ONNX Runtime 기본 골격~~

작업:

- `ort` dependency를 추가하고 CPU provider 기준으로 로컬 빌드/테스트가 가능하게 한다.
- 공통 `OnnxEvaluator`를 만들고 CPU provider로 먼저 검증한다.
- CUDA provider는 같은 `OnnxEvaluator`의 device 옵션으로 추가한다.
- shape validation과 error handling용 Rust unit test를 추가한다.
- 첫 구현에서는 단일 runner가 `OnnxEvaluator`를 소유하고 직접 호출한다.
- CUDA provider 없이도 로컬 CPU 테스트가 통과하게 유지한다.

검증 기준:

- 로컬에서 ONNX Runtime CPU provider 기반 `cargo test`가 통과한다.
- Runpod에서는 같은 evaluator를 CUDA provider로 실행해 별도 smoke test를 통과시킨다.

### ~~3단계: Python 콜백 없는 Batched Self-Play~~

작업:

- evaluator trait와 adapter를 추가한다.
- 기존 Gumbel batch 코드를 Rust evaluator와 함께 재사용한다.
- 기존 `GumbelSelfPlayBatch`가 모은 active root/leaf `EvalRequest`를 ONNX evaluator에 그대로 연결한다.
- 별도 async dynamic batching queue는 만들지 않는다.
- ONNX self-play smoke run용 Rust/Python 노출 메서드를 작게 추가한다.
- 기존 Python evaluator 메서드는 유지한다.

검증 기준:

- 기존 Python 테스트가 통과한다.
- 신규 smoke test가 최소 1개의 terminal game을 만든다.
- replay sample shape가 올바르고 값이 finite이다.

### ~~4단계: Rust ONNX 전용 Pipeline과 Legacy Import~~

작업:

- `rust_onnx_pipeline.py`와 전용 config를 추가한다.
- `rust_onnx_self_play.py`에 Rust runner 호출 wrapper를 추가한다.
- `rust_onnx_replay.py`에 Rust artifact import와 legacy pipeline import 기능을 추가한다.
- 첫 실행 시 legacy replay/checkpoint/log를 신규 work dir로 가져올 수 있게 한다.
- self-play 전에 best checkpoint를 export한다.
- Rust가 생성한 `safetensors` replay tensor와 JSON/JSONL metadata를 기존 replay buffer에 import한다.
- pipeline dispatch는 fake Rust runner 테스트로도 검증해 GPU 의존성을 피한다.

검증 기준:

- 기존 pipeline smoke test가 `python_torch` backend로 계속 통과한다.
- 신규 Rust ONNX pipeline dispatch test가 GPU 없이 runner 호출과 replay import를 검증한다.
- legacy `replay.npz`와 `game_logs.json` import test가 통과한다.
- legacy `best.pt`를 신규 work dir로 복사하고 ONNX export 입력으로 사용하는 테스트가 통과한다.

### 5단계: Runpod CUDA 검증

작업:

- Runpod에서 ONNX Runtime CUDA provider 설치와 로딩을 확인한다.
- 실제 checkpoint에서 ONNX export를 실행한다.
- Rust ONNX self-play를 CUDA로 실행한다.
- 현재 Python/PyTorch 경로와 처리량을 비교한다.

측정 지표:

- games/sec
- samples/sec
- 평균 ONNX batch size
- ONNX batch size 분포
- GPU utilization
- VRAM usage
- 장시간 실행 후 VRAM/RSS 증가 여부
- pipeline iteration별 self-play wall time

## 테스트 전략

기본 로컬 테스트에는 ONNX Runtime CPU provider 검증을 포함한다. CUDA provider 검증은 Runpod에서 수행한다.

필수 테스트:

- ONNX export shape test
- PyTorch vs ONNX parity test
- Rust ONNX Runtime CPU provider test
- `GumbelSelfPlayBatch` 기반 ONNX evaluator smoke test
- fake Rust runner 기반 신규 Rust ONNX pipeline dispatch test
- Rust 생성 replay data import test
- legacy pipeline replay/checkpoint/log import test
- `safetensors` tensor shape/dtype 검증 test

재현성 기준:

- self-play seed, Gumbel seed, playout cap randomization seed를 config와 manifest에 기록한다.
- 로컬 CPU parity tolerance는 FP32 기준 `1e-5`로 시작한다.
- CUDA provider parity tolerance는 Runpod 검증 후 `1e-4` 수준에서 별도 확정한다.
- CUDA provider는 backend 선택과 연산 순서에 따라 완전 결정론을 보장하지 않을 수 있으므로, smoke test는 허용 오차와 seed 기록을 함께 사용한다.

기존 확인 명령:

```bash
python -m pytest
python -m ruff check .
python -m mypy
cd rust/great_kingdom_core && cargo test
```

Runpod 검증 명령은 ONNX Runtime 설치 방식이 확정된 뒤 추가한다.

## 위험 요소

### ONNX Runtime CUDA 호환성

Runpod 기준 CUDA 버전은 12.4다. ONNX Runtime GPU package/provider가 이 환경에서 정상 동작하는지 확인해야 한다.

대응:

- CPU ONNX 경로를 유지한다.
- Python/PyTorch 경로를 유지한다.
- 첫 성공 실행 후 정확한 Runpod setup을 문서화한다.

### 수치 차이

ONNX 출력은 PyTorch 출력과 약간 다를 수 있다.

대응:

- tolerance 기반 parity test를 둔다.
- policy logits와 value를 따로 비교한다.
- 고정 random seed를 사용한다.
- seed와 주요 실행 config를 `manifest.json`에 저장한다.
- CUDA provider의 완전 결정론은 보장하지 않고 tolerance 기반 검증을 사용한다.

### Rust Core 복잡도 증가

`great_kingdom_core`는 현재 core Rust library이면서 PyO3 extension이다.

대응:

- ONNX 코드를 별도 모듈로 분리한다.
- CUDA provider 관련 설정만 `Cargo feature` 또는 Runpod 전용 빌드 설정으로 분리한다.
- 모델 로딩 코드를 Gumbel search 내부에 직접 섞지 않는다.

### Replay 포맷 변경 비용

Rust에서 바로 `.npz`를 쓰는 방식은 초기에 불필요한 복잡도를 만들 수 있다.

대응:

- `safetensors`와 JSON/JSONL metadata 조합으로 시작한다.
- Python converter를 통해 기존 `ReplayBuffer`에 적재한다.
- 실제 병목을 측정한 뒤 저장 포맷을 최적화한다.

### Legacy Import 혼선

legacy pipeline의 `metrics.jsonl`을 그대로 신규 pipeline의 iteration history로 이어 붙이면 generation 번호, arena seed window, self-play seed cursor가 섞일 수 있다.

대응:

- legacy metrics는 신규 `metrics.jsonl`에 합치지 않고 `legacy-import.json`에 provenance로 저장한다.
- 신규 pipeline iteration은 1부터 시작한다.
- seed cursor는 imported game logs의 최대 seed 다음 값에서 시작할 수 있게 별도 계산한다.
- legacy work dir은 읽기 전용으로 취급하고 신규 work dir에 복사/변환 결과만 쓴다.

### VRAM 메모리 누수 및 단편화

동적 batch 크기와 반복 추론으로 인해 ONNX Runtime 내부 allocation이 늘어나거나 VRAM 단편화가 발생할 수 있다.

대응:

- 첫 구현에서는 `max_batch_size`를 설정하고 초과 요청은 여러 batch로 나누어 실행한다.
- 입력/output buffer 재사용 가능성을 `OnnxEvaluator` 내부 설계에 남긴다.
- Runpod smoke에서 장시간 실행 후 VRAM/RSS 증가 여부를 측정한다.
- 필요하면 후속 최적화에서 I/O binding, pre-allocation, batch size bucket을 검토한다.

## 첫 구현 완료 기준

첫 구현은 다음 항목이 모두 동작하면 완료로 본다.

- `great-kingdom-export-onnx` 명령 추가
- `best.pt`를 FP32 opset 17 `best.onnx`로 export
- 작은 random batch에서 PyTorch checkpoint inference와 ONNX inference를 비교하는 parity test 추가
- Rust `OnnxEvaluator` CPU provider test 통과
- 기존 `GumbelSelfPlayBatch` 기반 ONNX self-play smoke test 통과
- Rust self-play 결과를 `samples.safetensors`, `games.jsonl`, `manifest.json`으로 저장
- Python converter가 저장 결과를 기존 `ReplayBuffer`에 import
- `great-kingdom-rust-onnx-pipeline` 명령 추가
- legacy pipeline의 `replay.npz`, `game_logs.json`, `best.pt`를 신규 ONNX pipeline work dir로 import
- Runpod에서 CUDA provider smoke test와 기본 throughput/VRAM 지표 기록

이 기준선이 확보된 뒤 별도 dynamic batching queue, FP16/mixed precision, TensorRT, arena ONNX 전환을 후속 최적화로 검토한다.
