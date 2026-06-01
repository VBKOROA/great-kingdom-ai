# Great Kingdom AI

Great Kingdom 보드게임을 위한 self-play 학습 실험 저장소입니다.

Rust 규칙 엔진과 Gumbel search로 self-play를 만들고, Python에서 trajectory replay,
policy-value network 학습, ONNX export, arena 평가를 실행합니다. 현재 Runpod 권장 학습
경로는 `great-kingdom-actor-v2`와 `great-kingdom-learner-v2`를 같이 돌리는 light async
구조입니다.

## 구조

```text
.
├── python/great_kingdom_ai/      # Python 학습/평가/파이프라인 코드
├── rust/great_kingdom_core/      # Rust 규칙 엔진, Gumbel search, ONNX evaluator
├── configs/                     # 테스트/Runpod 설정
├── scripts/                     # smoke, Runpod setup, artifact 정리 유틸
├── tests/                       # Python 테스트
└── data/                        # 실행 중 생성되는 replay/checkpoint/metrics 산출물
```

### 모델 구조 및 프리셋

Great Kingdom AI는 전통적인 residual CNN backbone과 기하학적 어텐션 메커니즘을 융합한 하이브리드 네트워크 아키텍처를 지원합니다.

* **CNN Backbone**: 3x3 Conv-BN-ReLU Stem과 다수의 `ResidualBlock`으로 구성된 깨끗한 CNN backbone.
* **Board Self-Attention**: 토큰화된 9x9 보드 공간에 대해 scaled dot-product attention 연산을 수행하는 커스텀 `BoardSelfAttentionBlock`을 Backbone 뒤쪽에 배치 가능.
  - **Full 2D Relative Position Bias**: Manhattan 거리에 의존하지 않고 모든 방향과 상대 오프셋($-8 \sim 8$)을 온전히 보존하도록 학습 가능한 $17 \times 17$ (289가지 관계) 2D 상대 위치 bias 테이블을 적용.
  - **LayerScale**: Self-Attention 및 FFN 잔차 경로(residual branch)에 초기값 `1e-3` 크기의 LayerScale 파라미터를 적용하여 학습 초기 수렴성 극대화.
* **Spatial Value Head**: 글로벌 풀링(GAP) 방식 대신 $1 \times 1$ Conv와 Flatten-Linear 구조를 채택하여 보드 공간 내 밀집 배치에 강건하게 대응.

#### 제공 프리셋 명세

| 프리셋 이름 | residual_blocks | channels | attention_blocks | attention_heads | spatial_value_head | 비고 |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`strong_clean`** | 10 | 128 | 0 | - | True | 기준 ResNet 모델 |
| **`strong_attn`** | 10 | 128 | 2 | 4 | True | 2-Block Self-Attention 추가 실험군 |

## 환경

로컬 개발은 CPU 노트북 기준입니다. 규칙, 데이터 구조, 작은 smoke run, 테스트를 확인하는
용도입니다.

본격 학습은 다음 Runpod 환경을 기준으로 합니다.

- GPU: RTX 3090 24GB
- Template: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
- Python은 venv 사용

## 로컬 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,ai]'
```

Rust 규칙 엔진을 Python 확장으로 빌드합니다.

```bash
cd rust/great_kingdom_core
../../.venv/bin/python -m maturin develop
cd ../..
```

기본 검증:

```bash
python -m pytest
python -m ruff check .
python -m mypy
```

## 권장 학습: async v2

Runpod에서는 actor와 learner를 별도 터미널에서 동시에 실행합니다. actor는 최신 ONNX로
self-play shard를 만들고, learner는 replay를 메모리에 유지하면서 새 shard를 import합니다.
학습량은 shard 개수가 아니라 `imported_transitions * train_reuse_factor`로 적립한 train
budget에 따라 정하고, 학습 후 `training-latest.pt`와 `training-latest.onnx`를 갱신합니다.

```text
training-latest.onnx
  -> actor-v2 Rust ONNX self-play
  -> shards/<shard-id>/
  -> learner-v2 import
  -> replay/trajectory-replay.npz
  -> learner training
  -> checkpoints/training-latest.pt
  -> checkpoints/onnx/training-latest.onnx
```

핵심 산출물:

- `replay/trajectory-replay.npz`: 재개에 필요한 원본 trajectory replay
- `replay/game_logs.jsonl`: import된 게임 로그
- `shards/metadata.jsonl`: shard 완료/import 기록과 actor seed resume 기준
- `checkpoints/training-latest.pt`: 현재 async v2의 주 학습 checkpoint
- `checkpoints/onnx/training-latest.onnx`: actor가 self-play에 쓰는 최신 ONNX
- `checkpoints/candidate.pt`: learner가 방금 만든 candidate 사본
- `checkpoints/best.pt`: 초기 fallback/복구용 anchor

### 실행 순서

#### 0. 최초 학습 초기화 (Initialization)

처음으로 완전히 새로운 자율 학습(Self-play) 루프를 가동할 경우, actor들이 self-play 데이터를 생성하기 전에 **초기화된 무작위 가중치 텐서 및 기본 ONNX 모델 파일**이 작업 디렉토리에 미리 존재해야 합니다. 아래의 초기화 CLI 도구를 사용해 이를 선제 생성합니다.

```bash
great-kingdom-init-async-v2 \
  --train-config configs/runpod/train.json \
  --work-dir data/runpod/train-strong-attn \
  --model-preset strong_attn \
  --overwrite
```

* **수행 결과**:
  - `data/runpod/train-strong-attn/checkpoints/training-latest.pt` 초기화 체크포인트 저장.
  - actor가 로드하여 첫 self-play shard를 제작할 수 있는 `checkpoints/onnx/training-latest.onnx` 및 `training-latest-ema.onnx` 자동 생성.
  - 기존에 누적되어 있던 이전 실험 체크포인트를 완전히 초기화하고 덮어쓰기 위해 `--overwrite` 옵션을 적용합니다.

#### 1. 환경 준비 및 프로세스 기동

먼저 Runpod 가상 환경을 활성화합니다.

```bash
source .venv/bin/activate
```

터미널 1~4: actor

```bash
source .venv/bin/activate
RAYON_NUM_THREADS=4 \
GKA_ONNX_BATCH_BUCKETING=1 \
great-kingdom-actor-v2 \
  --actor-config configs/runpod/actor-v2.json \
  --loop 
```

actor는 재시작 시 `shards/metadata.jsonl`을 보고 같은 `model_version`의 다음 seed부터
이어갑니다. 기존 shard와 충돌하면 최신 코드 반영 후 다시 실행하거나, 임시로
`--seed-start`를 다음 값으로 지정합니다.

터미널 2: learner

```bash
source .venv/bin/activate
great-kingdom-learner-v2 \
  --learner-config configs/runpod/learner-v2.json \
  --train-config configs/runpod/train.json \
  --loop \
  --sleep-seconds 3
```

터미널 3: 시스템 모니터링

```bash
./scripts/monitor.sh 10
```

터미널 4: 10분마다 checkpoint snapshot 저장

```bash
./scripts/save_training_snapshots.sh
```

비교용 baseline을 먼저 남기려면:

```bash
mkdir -p data/runpod/train-strong-attn/checkpoints/snapshots
cp data/runpod/train-strong-attn/checkpoints/training-latest.pt \
  data/runpod/train-strong-attn/checkpoints/snapshots/baseline.pt
```

최근 12개 snapshot만 유지하려면:

```bash
./scripts/save_training_snapshots.sh \
  data/runpod/train-strong-attn/checkpoints/training-latest.pt \
  data/runpod/train-strong-attn/snapshots \
  300 \
  12
```

백그라운드로 돌릴 때:

```bash
nohup great-kingdom-actor-v2 \
  --actor-config configs/runpod/actor-v2.json \
  --loop \
  --sleep-seconds 1 \
  > actor.log 2>&1 &

nohup great-kingdom-learner-v2 \
  --learner-config configs/runpod/learner-v2.json \
  --train-config configs/runpod/train.json \
  --loop \
  --sleep-seconds 5 \
  > learner.log 2>&1 &

nohup ./scripts/save_training_snapshots.sh > snapshots.log 2>&1 &
```

현재 async v2에서는 arena promote를 자동으로 하지 않습니다. 가장 최신 모델은
`checkpoints/training-latest.pt`이고, "검증된 최강"은 snapshot끼리 arena 비교해서 고릅니다.

## Runpod 설치

Runpod에서는 기본 PyTorch/CUDA 설치를 유지하기 위해 setup script를 사용합니다.

```bash
bash scripts/setup_runpod.sh
source .venv/bin/activate
```

스크립트는 venv 생성, 개발 의존성 설치, CUDA 확인, Rust PyO3 확장 빌드, Rust/Python 테스트를
수행합니다.

CUDA smoke:

```bash
python scripts/run_m6_smoke.py --device cuda
python scripts/run_m8_train_smoke.py --device cuda
```

학습 중 CPU/RAM/GPU 상태:

```bash
./scripts/monitor.sh 10
```

## 용량 정리

async v2 learner는 import 완료된 shard 원본 디렉터리를 학습 성공 사이클 마지막에 자동으로
삭제합니다. replay, metadata, active checkpoint는 보존합니다.

자동 삭제 대상:

- `shards/<imported-shard-id>/trajectory-replay.npz`
- `shards/<imported-shard-id>/game_logs.json`
- `shards/<imported-shard-id>/` 디렉터리

자동 보존 대상:

- `replay/trajectory-replay.npz`
- `replay/game_logs.jsonl`
- `shards/metadata.jsonl`
- `checkpoints/best.pt`
- `checkpoints/training-latest.pt`
- `checkpoints/candidate.pt`
- `checkpoints/onnx/training-latest.onnx`
- `checkpoints/snapshots/*.pt`

단일 `train-v2` 파이프라인 산출물이나 오래된 재생성 가능 artifact는 별도 스크립트로 정리할 수
있습니다. 정리 스크립트는 기본 dry-run입니다.

```bash
python scripts/prune_runpod_artifacts.py \
  --work-dir data/runpod/train-strong-attn
```

실제 삭제:

```bash
python scripts/prune_runpod_artifacts.py \
  --work-dir data/runpod/train-strong-attn \
  --delete
```

스크립트 기본 삭제 대상:

- 오래된 `targets/targets-*.npz`
- 오래된 `checkpoints/candidates/candidate-*.pt`
- 오래된 `checkpoints/onnx/best-*.onnx`
- `self-play/iteration-*`

스크립트 기본 보존 대상:

- `replay/trajectory-replay.npz`
- `targets/latest.npz`
- `checkpoints/best.pt`
- `checkpoints/training-latest.pt`
- `checkpoints/candidate.pt`
- `reports/metrics.jsonl`
- `replay/game_logs.jsonl`

캐시까지 지우려면:

```bash
python scripts/prune_runpod_artifacts.py \
  --work-dir data/runpod/train-strong-attn \
  --include-build-cache \
  --delete
```

더 빡세게 줄이려면 최신 artifact 보존 개수를 낮춥니다.

```bash
python scripts/prune_runpod_artifacts.py \
  --work-dir data/runpod/train-strong-attn \
  --keep-targets 1 \
  --keep-candidates 1 \
  --keep-onnx 0 \
  --delete
```

## 평가와 플레이

현재 async v2의 최신 모델과 저장해 둔 snapshot을 arena에서 비교합니다. `.pt`를 직접 넣으면
평가 시작 시 임시 ONNX export가 들어가므로, 반복 비교할 baseline은 ONNX로 한 번 만들어 두는
편이 빠릅니다.

baseline ONNX 준비:

```bash
great-kingdom-export-onnx \
  --checkpoint data/runpod/train-strong-attn/checkpoints/snapshots/baseline.pt \
  --output data/runpod/train-strong-attn/checkpoints/snapshots/baseline.onnx
```

학습 중 부담 적은 quick check:

```bash
great-kingdom-evaluate \
  --candidate data/runpod/train-strong-attn/checkpoints/onnx/training-latest.onnx \
  --best data/runpod/train-strong-attn/checkpoints/snapshots/baseline.onnx \
  --report data/runpod/train-strong-attn/reports/arena-quick-latest-vs-baseline.json \
  --config configs/runpod/arena.json \
  --games 20 \
  --batch-size 20 \
  --gumbel-simulations 16 \
  --gumbel-max-considered-actions 8 \
  --leaf-batch-size 512 \
  --onnx-max-batch-size 4096 \
  --device cuda
```

조금 더 믿을 만한 중간 평가:

```bash
great-kingdom-evaluate \
  --candidate data/runpod/train-strong-attn/checkpoints/onnx/training-latest.onnx \
  --best data/runpod/train-strong-attn/checkpoints/snapshots/baseline.onnx \
  --report data/runpod/train-strong-attn/reports/arena-40-latest-vs-baseline.json \
  --config configs/runpod/arena.json \
  --games 40 \
  --batch-size 40 \
  --gumbel-simulations 32 \
  --gumbel-max-considered-actions 12 \
  --leaf-batch-size 512 \
  --onnx-max-batch-size 4096 \
  --device cuda
```

학습을 멈추고 보는 정식 평가는 기본 arena 설정을 사용합니다.

```bash
great-kingdom-evaluate \
  --candidate data/runpod/train-strong-attn/checkpoints/onnx/training-latest.onnx \
  --best data/runpod/train-strong-attn/checkpoints/snapshots/baseline.onnx \
  --report data/runpod/train-strong-attn/reports/arena-full-latest-vs-baseline.json \
  --config configs/runpod/arena.json \
  --games 80 \
  --batch-size 80 \
  --device cuda
```

snapshot끼리 비교할 때는 `--candidate`와 `--best`에 비교할 `.pt` 또는 `.onnx` 파일을 각각
넣습니다. async v2는 자동 promote를 하지 않으므로, 가장 강한 모델은 이런 arena 비교 결과로
고릅니다.

`training-latest`가 특정 snapshot에게 반복해서 지면
[UPDATE_PRESSURE_TUNING.md](UPDATE_PRESSURE_TUNING.md)에 따라
update pressure를 진단하고 `learning_rate`, `train_reuse_factor`, `ema_decay`, replay sampling
파라미터를 조정합니다.

직접 대국:

```bash
great-kingdom-play \
  --model-checkpoint data/runpod/train-strong-attn/checkpoints/training-latest.pt \
  --human-player blue \
  --device cpu \
  --model-simulations 64
```

두 checkpoint끼리 한 판 대국:

```bash
great-kingdom-play \
  --arena-checkpoints \
    data/runpod/train-strong-attn/checkpoints/snapshots/baseline.pt \
    data/runpod/train-strong-attn/checkpoints/training-latest.pt \
  --device cpu \
  --model-simulations 64
```

CLI 입력:

- `A1`부터 `I9`: 해당 좌표에 현재 플레이어 성 놓기
- `5 5`: 행/열 숫자로 착수
- `p` 또는 `pass`: 패스
- `l` 또는 `legal`: 현재 합법 수 출력
- `b` 또는 `board`: 보드 다시 출력
- `i <0-81>`: raw action index 입력
- `q` 또는 `quit`: 종료

수순 재생:

```bash
great-kingdom-play --replay-actions '20,68,77'
great-kingdom-play --replay-actions '20,68,77' --pause
```

## 기타 명령

async v2 운영 경로에서는 `great-kingdom-learner-v2`를 사용합니다. 저수준 learner CLI는 이미
준비된 trajectory replay를 대상으로 한 수동 학습/진단에만 사용합니다.

```bash
great-kingdom-train \
  --replay data/runpod/train-strong-attn/replay/trajectory-replay.npz \
  --checkpoint data/runpod/train-strong-attn/checkpoints/candidate.pt \
  --resume data/runpod/train-strong-attn/checkpoints/training-latest.pt \
  --config configs/runpod/train.json \
  --device cuda
```

단일 batch overfit으로 learner wiring을 확인:

```bash
great-kingdom-single-batch-overfit \
  --replay data/runpod/train-strong-attn/replay/trajectory-replay.npz \
  --config configs/runpod/train.json \
  --device cuda \
  --steps 1000 \
  --batch-size 512 \
  --log-every 50
```

ONNX export:

```bash
great-kingdom-export-onnx \
  --checkpoint data/runpod/train-strong-attn/checkpoints/training-latest.pt \
  --output data/runpod/train-strong-attn/checkpoints/onnx/training-latest.onnx \
  --check-parity
```

CPU serving용 ONNX export + selective QDQ S8S8 quantization:

```bash
python scripts/export_ema_weights.py \
  --checkpoint data/runpod/train-strong-attn/checkpoints/training-latest.pt \
  --onnx-output data/runpod/train-strong-attn/checkpoints/onnx/training-latest-ema.fp32.onnx \
  --quantized-onnx-output data/runpod/train-strong-attn/checkpoints/onnx/training-latest-ema.selective-qdq-s8s8.onnx \
  --quantization-format qdq-s8s8 \
  --quantization-mode selective-attention \
  --calibration-features data/runpod/train-strong-attn/replay/trajectory-replay.npz \
  --calibration-samples 2048 \
  --calibration-batch-size 32
```

권장 흐름은 checkpoint에서 **FP32 ONNX**를 먼저 만들고, ONNX Runtime `quant_pre_process` 후
실제 replay feature로 static **QDQ S8S8** quantization을 수행하는 방식이다. attention이 들어간
모델은 `--quantization-mode selective-attention`을 우선 사용한다. 이 모드는 `Conv`/`Gemm` 중심으로
양자화하고 attention 이름을 가진 노드와 `MatMul`/`Softmax`/`LayerNormalization`은 FP32로 남긴다.
CPU serving에서는 `fp16` ONNX보다 `fp32.onnx` 원본과 selective `qdq-s8s8.onnx` 후보를 함께 보관한
뒤 latency, policy/value diff, arena 결과를 보고 최종 선택한다. calibration data는 attention
quantization 품질에 직접 영향을 주므로 synthetic calibration보다 실제 trajectory replay를 우선
사용한다.

전역 QDQ S8S8을 비교 후보로 만들 때는 `--quantization-mode full`을 사용한다.
MinMax calibration에서 policy/value diff가 크면 `--calibration-method percentile`도 비교한다.

## 개발 원칙

- 테스트하기 쉬운 구조를 유지합니다.
- 수정하기 쉬운 작은 모듈로 나눕니다.
- 로컬 CPU 환경에서는 빠른 검증을, Runpod GPU 환경에서는 본격 학습을 수행합니다.
- 규칙 엔진 동작이 의심될 때는 수동 CLI와 규칙 문서를 확인합니다.
