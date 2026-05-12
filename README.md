# Great Kingdom AI

Great Kingdom 보드게임을 위한 self-play 학습 실험 저장소입니다.

Rust 규칙 엔진과 Gumbel search로 self-play를 만들고, Python에서 trajectory replay,
reanalyze target snapshot, policy-value network 학습, arena 평가를 실행합니다. 현재 권장
학습 경로는 `great-kingdom-train-v2`입니다.

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

## 권장 학습: v2

v2 파이프라인은 원본 replay와 학습 target을 분리합니다.

```text
best.pt
  -> ONNX export
  -> Rust ONNX self-play
  -> trajectory-replay.npz
  -> reanalyze target snapshot
  -> learner training
  -> candidate/training-latest/best checkpoint
```

핵심 산출물:

- `replay/trajectory-replay.npz`: 재개에 필요한 원본 trajectory replay
- `targets/targets-*.npz`: 특정 checkpoint/config로 만든 학습 target snapshot
- `targets/latest.npz`: 최신 target snapshot 사본
- `checkpoints/best.pt`: 다음 self-play/export 기준 모델
- `checkpoints/training-latest.pt`: learner resume 기준 checkpoint
- `reports/metrics.jsonl`: 완료 iteration 기록
- `replay/game_logs.jsonl`: seed cursor 계산용 로그

Runpod 권장 실행:

```bash
source .venv/bin/activate
great-kingdom-train-v2 \
  --device cuda \
  --pipeline-config configs/runpod/train-v2-pipeline.json \
  --train-config configs/runpod/train.json \
  --arena-config configs/runpod/arena.json
```

현재 Runpod v2 권장 설정은 `configs/runpod/train-v2-pipeline.json`입니다. 3090 24GB에서
장시간 재학습을 염두에 둔 균형형 설정입니다.

- trajectory replay capacity: `500000`
- self-play: `1500` games/iteration
- Gumbel full search: `96` simulations
- playout-cap: full fraction `0.35`, fast simulations `24`
- bootstrap target: `8` steps
- search reanalyze: `2%`, 최대 `2048` states/iteration
- arena는 기본 skip, candidate는 자동 promote

v2는 aggregate replay를 쓰지 않습니다. 중복 state 평균 대신 trajectory replay, reanalyze,
target age, priority sampling, 제한된 search reanalyze로 replay 재사용 효율을 올립니다.

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

## 용량 정리

Runpod 30GB 디스크에서는 오래된 target snapshot과 checkpoint history를 주기적으로 정리하는
편이 좋습니다. 정리 스크립트는 기본 dry-run입니다.

```bash
python scripts/prune_runpod_artifacts.py \
  --work-dir data/runpod/train-v2-recommended-medium-plus
```

실제 삭제:

```bash
python scripts/prune_runpod_artifacts.py \
  --work-dir data/runpod/train-v2-recommended-medium-plus \
  --delete
```

기본 삭제 대상:

- 오래된 `targets/targets-*.npz`
- 오래된 `checkpoints/candidates/candidate-*.pt`
- 오래된 `checkpoints/onnx/best-*.onnx`
- `self-play/iteration-*`

기본 보존 대상:

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
  --work-dir data/runpod/train-v2-recommended-medium-plus \
  --include-build-cache \
  --delete
```

더 빡세게 줄이려면 최신 artifact 보존 개수를 낮춥니다.

```bash
python scripts/prune_runpod_artifacts.py \
  --work-dir data/runpod/train-v2-recommended-medium-plus \
  --keep-targets 1 \
  --keep-candidates 1 \
  --keep-onnx 0 \
  --delete
```

## 평가와 플레이

후보 모델과 best 모델을 arena에서 비교:

```bash
great-kingdom-evaluate \
  --candidate data/runpod/train-v2-recommended-medium-plus/checkpoints/candidate.pt \
  --best data/runpod/train-v2-recommended-medium-plus/checkpoints/best.pt \
  --report data/runpod/train-v2-recommended-medium-plus/reports/arena-manual.json \
  --config configs/runpod/arena.json \
  --device cuda
```

직접 대국:

```bash
great-kingdom-play \
  --model-checkpoint data/runpod/train-v2-recommended-medium-plus/checkpoints/best.pt \
  --human-player blue \
  --device cpu \
  --model-simulations 64
```

두 checkpoint끼리 한 판 대국:

```bash
great-kingdom-play \
  --arena-checkpoints best.pt best-2.pt \
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

Replay 또는 target snapshot이 준비되어 있다면 learner만 직접 실행할 수 있습니다.

```bash
great-kingdom-train \
  --replay data/runpod/train-v2-recommended-medium-plus/targets/latest.npz \
  --checkpoint data/runpod/train-v2-recommended-medium-plus/checkpoints/candidate.pt \
  --resume data/runpod/train-v2-recommended-medium-plus/checkpoints/training-latest.pt \
  --config configs/runpod/train.json \
  --device cuda
```

단일 batch overfit으로 learner wiring을 확인:

```bash
great-kingdom-single-batch-overfit \
  --replay data/runpod/train-v2-recommended-medium-plus/targets/latest.npz \
  --config configs/runpod/train.json \
  --device cuda \
  --steps 1000 \
  --batch-size 512 \
  --log-every 50
```

ONNX export:

```bash
great-kingdom-export-onnx \
  --checkpoint data/runpod/train-v2-recommended-medium-plus/checkpoints/best.pt \
  --output data/runpod/train-v2-recommended-medium-plus/checkpoints/best.onnx \
  --check-parity
```

기존 `great-kingdom-rust-onnx-pipeline`은 v1/비교용 경로입니다. 이 경로는 sample replay와
optional aggregate replay를 사용합니다.

```bash
great-kingdom-rust-onnx-pipeline \
  --device cuda \
  --pipeline-config configs/runpod/pipeline.json \
  --train-config configs/runpod/train.json \
  --arena-config configs/runpod/arena.json
```

## 개발 원칙

- 테스트하기 쉬운 구조를 유지합니다.
- 수정하기 쉬운 작은 모듈로 나눕니다.
- 로컬 CPU 환경에서는 빠른 검증을, Runpod GPU 환경에서는 본격 학습을 수행합니다.
- 규칙 엔진 동작이 의심될 때는 수동 CLI와 규칙 문서를 확인합니다.
