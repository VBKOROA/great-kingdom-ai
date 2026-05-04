# Great Kingdom AI

Great Kingdom 보드게임을 위한 AlphaZero-lite 실험 저장소입니다.

이 프로젝트는 9x9 Great Kingdom 규칙 엔진을 Rust로 구현하고, Python에서 self-play, replay buffer, policy-value network 학습, arena 평가, 모델 승격 파이프라인을 실행할 수 있게 구성되어 있습니다.

## 프로젝트 구성

```text
.
├── python/great_kingdom_ai/      # Python 학습/평가/CLI 코드
├── rust/great_kingdom_core/      # Rust 규칙 엔진과 Gumbel search PyO3 확장
├── configs/                     # 테스트/Runpod용 JSON 설정
├── scripts/                     # smoke test와 Runpod 준비 스크립트
├── tests/                       # Python 테스트
├── docs/                        # 규칙 명세와 수동 테스트 문서
└── data/                        # 실행 중 생성되는 replay/checkpoint/metrics 산출물
```

핵심 문서는 다음부터 보면 좋습니다.

- [docs/rule-spec.md](docs/rule-spec.md): Great Kingdom 재현용 규칙 명세
- [docs/manual-cli-test-cases.md](docs/manual-cli-test-cases.md): 수동 CLI로 규칙을 확인하는 테스트 케이스

## 개발 환경

권장 환경은 Python 3.11 이상, Rust/Cargo, Python 가상환경입니다.

로컬 사무용 노트북처럼 GPU가 없는 환경에서는 CPU로 규칙, 데이터, 테스트, 작은 smoke run을 확인하는 용도로 사용합니다. 본격적인 학습은 Runpod의 RTX 3090 24GB 환경을 기준으로 합니다.

## 로컬 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

PyTorch가 필요한 학습/평가 코드를 실행하려면 AI 의존성도 설치합니다.

```bash
python -m pip install -e '.[dev,ai]'
```

Rust 규칙 엔진을 Python 확장으로 빌드합니다.

```bash
cd rust/great_kingdom_core
../../.venv/bin/python -m maturin develop
cd ../..
```

설치가 끝나면 테스트를 실행합니다.

```bash
python -m pytest
```

## 빠른 실행

수동으로 게임 규칙 엔진을 확인하려면 다음 명령을 사용합니다.

```bash
great-kingdom-play
```

Arena나 self-play 로그의 action index 수순을 CLI에서 재생하려면 다음처럼 실행합니다.

```bash
great-kingdom-play --replay-actions '20,68,77'
great-kingdom-play --replay-actions '20,68,77' --pause
```

CLI에서 사용할 수 있는 입력은 다음과 같습니다.

- `A1`부터 `I9`: 해당 좌표에 현재 플레이어 성 놓기
- `5 5`: 행/열 숫자로 착수
- `p` 또는 `pass`: 패스
- `l` 또는 `legal`: 현재 합법 수 출력
- `b` 또는 `board`: 보드 다시 출력
- `i <0-81>`: raw action index 입력
- `q` 또는 `quit`: 종료

무작위 self-play smoke run은 다음처럼 실행합니다.

```bash
great-kingdom-random-self-play --games 3 --prefer-place
```

CPU에서 전체 파이프라인을 작게 확인하려면 테스트 설정을 사용합니다.

```bash
great-kingdom-pipeline \
  --allow-cpu \
  --pipeline-config configs/test/pipeline-smoke.json \
  --train-config configs/test/m8-train-smoke.json \
  --arena-config configs/test/m9-arena-smoke.json
```

실행 결과는 기본적으로 `data/` 아래에 replay, checkpoint, arena report, metrics 형태로 저장됩니다.

## 학습과 평가 명령

Replay 파일이 준비되어 있다면 직접 학습할 수 있습니다.

```bash
great-kingdom-train \
  --replay data/pipeline/replay.npz \
  --checkpoint data/pipeline/checkpoints/candidate.pt \
  --config configs/test/m8-train-smoke.json \
  --device cpu
```

후보 모델과 best 모델을 arena에서 비교하려면 다음 명령을 사용합니다.

```bash
great-kingdom-evaluate \
  --candidate data/pipeline/checkpoints/candidate.pt \
  --best data/pipeline/checkpoints/best.pt \
  --report data/pipeline/arena-report.json \
  --config configs/test/m9-arena-smoke.json \
  --device cpu
```

후보 모델이 기준 승률을 넘으면 best 모델로 승격시키고 싶을 때는 `--promote`를 추가합니다.

Arena의 `batch_size`는 동시에 진행할 평가 게임 수입니다. `batch_size=1`은 기존 순차 실행이고,
Runpod RTX 3090에서는 우선 `20`을 권장합니다. CUDA OOM이 나면 `8` 또는 `4`로 낮춰서 다시
실행합니다.

## Runpod 학습 환경

본격적인 CUDA 학습은 다음 환경을 기준으로 합니다.

- GPU: RTX 3090 24GB
- Pod template: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
- Jupyter 포함 템플릿

Runpod에서는 기본 PyTorch/CUDA 설치를 최대한 유지하기 위해 `scripts/setup_runpod.sh`를 사용합니다.

```bash
bash scripts/setup_runpod.sh
```

스크립트는 다음 작업을 수행합니다.

- `.venv` 생성
- 개발 의존성 설치
- Runpod의 PyTorch/CUDA 인식 확인
- Rust PyO3 확장 빌드
- Rust 테스트와 Python 테스트 실행

CUDA smoke test는 다음처럼 실행합니다.

```bash
source .venv/bin/activate
python scripts/run_m6_smoke.py --device cuda
python scripts/run_m8_train_smoke.py --device cuda
```

Runpod용 설정 파일은 `configs/runpod/` 아래에 있습니다.

```bash
great-kingdom-pipeline \
  --device cuda \
  --pipeline-config configs/runpod/pipeline-runpod.json \
  --train-config configs/runpod/train-runpod.json \
  --arena-config configs/runpod/arena-runpod.json
```

## 테스트와 품질 확인

일반적인 확인 순서는 다음과 같습니다.

```bash
python -m pytest
python -m ruff check .
python -m mypy
```

Rust 쪽 테스트는 다음처럼 실행합니다.

```bash
cd rust/great_kingdom_core
cargo test
cd ../..
```

## 개발 원칙

이 저장소에서는 다음 원칙을 우선합니다.

- 테스트하기 쉬운 구조를 유지합니다.
- 파일이 너무 커지면 역할에 맞게 나눕니다.
- 로컬 CPU 환경에서는 빠른 검증을, Runpod GPU 환경에서는 본격 학습을 수행합니다.
- 규칙 엔진의 동작이 의심될 때는 [docs/rule-spec.md](docs/rule-spec.md)와 수동 CLI 테스트를 먼저 확인합니다.

## 문제 해결

`great_kingdom_core is not installed`가 나오면 Rust 확장이 아직 빌드되지 않은 상태입니다.

```bash
cd rust/great_kingdom_core
../../.venv/bin/python -m maturin develop
cd ../..
```

`great-kingdom-play: command not found`가 나오면 가상환경을 활성화했는지 확인합니다.

```bash
source .venv/bin/activate
hash -r
```

CUDA를 기대했는데 CPU로 실행된다면 PyTorch가 CUDA를 인식하는지 확인합니다.

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```
