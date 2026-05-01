# Great Kingdom AI

Great Kingdom AI는 9x9 추상 전략 게임 **Great Kingdom**을 재현하고, 자가대국 기반 학습으로 강해지는 AI 플레이어를 만들기 위한 프로젝트입니다.

현재 저장소는 Rust 규칙 엔진, PyO3 바인딩, 수동 규칙 확인 CLI, 랜덤 self-play smoke runner, PyTorch 모델, self-play 데이터 저장, 학습 루프와 checkpoint 저장/재개 도구, candidate-vs-best arena 평가와 best checkpoint 교체 도구를 포함합니다. 핵심 규칙과 학습 구조, Python/Rust 분리 아키텍처는 문서로 정리해 둔 상태입니다.

## 목표

이 프로젝트의 목표는 다음 세 가지입니다.

1. Great Kingdom 규칙을 정확히 재현하는 규칙 엔진을 만든다.
2. 규칙 엔진 위에서 MCTS 기반 자가대국을 수행한다.
3. 자가대국 데이터를 이용해 AlphaZero-lite 형태의 신경망 플레이어를 학습한다.

정통 AlphaZero를 완전히 재현하기보다는, GPU 없는 로컬 개발 환경과 Runpod RTX 4090 24GB 학습 환경에서 현실적으로 실험 가능한 소형 구조를 목표로 합니다.

## 게임 개요

Great Kingdom은 두 플레이어가 번갈아 성을 놓으며 영토를 넓히는 2인 추상 전략 게임입니다.

- 보드는 9x9입니다.
- 중앙에는 중립 성 1개가 놓입니다.
- 각 플레이어는 최대 40개의 성을 놓을 수 있습니다.
- 자신의 차례에는 빈 칸에 성을 놓거나 패스할 수 있습니다.
- 상대 성 그룹을 완전히 포위해 파괴하면 즉시 승리합니다.
- 자신의 성 그룹이 파괴되면 즉시 패배합니다.
- 두 플레이어가 연속으로 패스하면 영토 점수로 승패를 판정합니다.

자세한 규칙은 [docs/rule-spec.md](docs/rule-spec.md)를 기준으로 합니다.

## AI 설계

AI는 AlphaZero-lite 구조를 따릅니다.

- 신경망은 현재 국면을 입력받아 정책과 가치를 예측합니다.
- MCTS는 신경망의 예측을 이용해 더 좋은 착수를 탐색합니다.
- 자가대국으로 학습 데이터를 만들고, 이를 반복 학습합니다.

행동 공간은 총 82개입니다.

- 0~80: 9x9 보드의 각 칸에 성을 놓는 행동
- 81: 패스

신경망은 항상 현재 차례 플레이어의 관점에서 국면을 평가합니다. 즉, 가치값이 양수면 현재 플레이어에게 유리하고, 음수면 상대에게 유리한 상태로 해석합니다.

자세한 학습 구조는 [docs/alphazero-lite.md](docs/alphazero-lite.md)를 참고하세요.

## 아키텍처 방향

프로젝트는 Python과 Rust를 분리해서 구현할 계획입니다.

Python은 AI와 학습 레이어를 담당합니다.

- PyTorch 모델 정의
- 신경망 추론과 학습
- 리플레이 버퍼 관리
- 자가대국 orchestration
- 체크포인트 저장과 평가 리포트 생성

Rust는 성능과 규칙 정확성이 중요한 코어 레이어를 담당합니다.

- 보드와 게임 상태 표현
- 합법 수 생성
- 착수 적용
- 즉시 승패 판정
- 영토 점수 계산
- MCTS 트리와 PUCT 탐색
- Python 바인딩

Python과 Rust 연결은 PyO3와 maturin을 사용하는 방향으로 설계되어 있습니다. 자세한 내용은 [docs/python-rust-architecture.md](docs/python-rust-architecture.md)를 참고하세요.

## 문서

현재 저장소의 주요 문서는 다음과 같습니다.

- [docs/rule-spec.md](docs/rule-spec.md): Great Kingdom 재현용 규칙 명세
- [docs/alphazero-lite.md](docs/alphazero-lite.md): AlphaZero-lite 학습 구조 명세
- [docs/python-rust-architecture.md](docs/python-rust-architecture.md): Python + Rust 구현 구조 명세
- [docs/development-plan.md](docs/development-plan.md): 단계별 개발 계획표

## 현재 상태

현재 단계에서 이 저장소는 다음을 제공합니다.

- 게임 규칙의 기준 문서
- AI 학습 방식의 설계 문서
- Rust 규칙 엔진과 Python 바인딩
- 수동 self-play CLI와 랜덤 self-play smoke runner
- PyTorch policy-value 모델과 학습 루프
- self-play replay artifact, checkpoint 저장과 resume
- candidate 모델과 best 모델의 arena 평가 및 best checkpoint 교체
- 개발 순서와 책임 분리 기준

## 개발 환경

Python 3.11 이상과 Rust 1.85 이상을 기준으로 합니다. 로컬에서는 venv를 사용하고, GPU가 필요한 긴 self-play와 학습은 Runpod에서 실행합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
cd rust/great_kingdom_core
python -m maturin develop
cargo test
cd ../..
python -m pytest
```

`great-kingdom-play`, `great-kingdom-random-self-play` 같은 venv 명령은 `pip install -e '.[dev]'`가 만든 console script입니다. `pyproject.toml`의 `[project.scripts]`가 바뀐 뒤 명령을 찾지 못하면 venv를 활성화한 상태에서 `python -m pip install -e '.[dev]'`를 다시 실행하세요.

Rust 포맷터와 린터는 Ubuntu/Debian 기준으로 다음 패키지가 필요합니다.

```bash
sudo apt install -y rustfmt rust-clippy
```

## 학습 환경

Runpod 학습 환경은 다음 템플릿을 기준으로 합니다.

```text
runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
```

학습 코드는 Python 3.11에서 동작해야 하며, 로컬과 CI의 기본 테스트는 GPU 없이 통과해야 합니다. 현재 Runpod 정식 학습 설정은 RTX 4090 24GB 기준 `large` 모델 preset을 사용하고, 처리량 확인이 필요할 때만 `medium`으로 낮춰 비교합니다.

Runpod에서는 이미지에 포함된 PyTorch/CUDA 조합을 유지하기 위해 venv를 system site packages로 만들고, `ai` extra를 설치하지 않습니다. 새 pod에서는 다음 스크립트로 개발 의존성 설치, Rust extension 빌드, Rust/Python 테스트를 한 번에 실행할 수 있습니다.

```bash
./scripts/setup_runpod.sh
source .venv/bin/activate
python scripts/run_m6_smoke.py
python scripts/run_m8_train_smoke.py
```

기존 replay artifact로 학습을 직접 실행할 때는 다음 console script를 사용할 수 있습니다.

```bash
great-kingdom-train \
  --replay data/runpod/replay/replay.npz \
  --checkpoint data/runpod/checkpoints/latest.pt \
  --config configs/m8-train-smoke.json
```

candidate checkpoint를 현재 best checkpoint와 비교 평가할 때는 다음 console script를 사용할 수 있습니다. 평가는 선후공을 교대하며, self-play용 root noise와 temperature sampling을 쓰지 않습니다.

```bash
great-kingdom-evaluate \
  --candidate data/runpod/checkpoints/candidate.pt \
  --best data/runpod/checkpoints/best.pt \
  --report data/runpod/logs/arena-report.json \
  --config configs/m9-arena-smoke.json \
  --promote
```

`--promote`를 붙이면 candidate 승률이 설정의 `promotion_threshold` 이상일 때 candidate checkpoint를 best checkpoint 위치로 복사합니다.

자가대국, 학습, arena 평가, best 교체를 반복해서 돌릴 때는 pipeline CLI를 사용합니다. pipeline은 현재 best checkpoint를 보장한 뒤, 매 iteration마다 best 모델의 root policy prior로 self-play 데이터를 만들고 replay를 누적합니다. 이어서 candidate를 학습하고 arena 기준을 넘으면 best를 교체합니다. 세대별 candidate와 arena report, 전체 replay, metrics JSONL은 `work_dir` 아래에 남습니다. 로컬 smoke에서는 CPU fallback을 허용하고 작은 설정으로 확인합니다.

```bash
great-kingdom-pipeline \
  --pipeline-config configs/pipeline-smoke.json \
  --train-config configs/m8-train-smoke.json \
  --arena-config configs/m9-arena-smoke.json \
  --iterations 2 \
  --allow-cpu
```

editable install을 다시 하지 않은 상태라 console script가 아직 없으면 같은 동작을 모듈로 실행할 수 있습니다.

```bash
python -m great_kingdom_ai.pipeline \
  --pipeline-config configs/pipeline-smoke.json \
  --iterations 2 \
  --allow-cpu
```

Runpod에서 정식 학습을 시작할 때는 smoke 설정 대신 Runpod용 설정을 사용합니다. `configs/pipeline-train.json`은 `max_self_play_games`를 `null`로 두어, PCR처럼 저장되는 sample 수가 가변적인 설정에서도 매 iteration마다 `min_replay_samples`를 채울 때까지 self-play를 계속 실행합니다.

```bash
source .venv/bin/activate
great-kingdom-pipeline \
  --pipeline-config configs/pipeline-train.json \
  --train-config configs/train-runpod.json \
  --arena-config configs/arena-runpod.json
```

완전히 새 run을 시작하려면 기존 replay를 무시하도록 `--fresh`와 별도 `work_dir`를 함께 지정합니다.

```bash
source .venv/bin/activate
great-kingdom-pipeline \
  --pipeline-config configs/pipeline-train.json \
  --train-config configs/train-runpod.json \
  --arena-config configs/arena-runpod.json \
  --device cuda \
  --playout-cap-randomization \
  --playout-cap-full-search-fraction 0.25 \
  --playout-cap-fast-simulations 16 \
  --fresh \
  --work-dir data/runpod/pipeline-YYYYMMDD
```

중단 후 같은 `work_dir`에서 다시 실행하면 replay와 로그를 이어서 사용합니다. console script가 갱신되지 않은 pod에서는 `python -m great_kingdom_ai.pipeline ...` 형태로 같은 인자를 넘기면 됩니다.

## 수동 규칙 확인 CLI

Rust 규칙 엔진을 빌드한 뒤 사람 둘이 직접 self-play로 규칙을 확인할 수 있습니다.

```bash
great-kingdom-play
```

입력은 `A1`부터 `I9`, `5 5`, `p`/`pass`, `l`/`legal`, `q`/`quit`을 지원합니다.

## 랜덤 self-play smoke test

Rust 규칙 엔진을 빌드한 뒤 여러 seed의 랜덤 게임을 실행할 수 있습니다. 로컬 smoke test에서는 작은 판 수와 고정 seed를 사용합니다.

```bash
great-kingdom-random-self-play --games 10 --seed-start 0 --prefer-place
```

`--json`을 붙이면 seed, 착수 목록, 종료 사유, 최종 영토 점수를 포함한 게임 로그를 출력합니다.

## 예정 디렉터리 구조

초기 구현은 다음 구조를 기준으로 진행할 예정입니다.

```text
great-kingdom-ai/
  docs/
    rule-spec.md
    alphazero-lite.md
    python-rust-architecture.md
  python/
    great_kingdom_ai/
      model.py
      train.py
      self_play.py
      replay_buffer.py
      evaluate.py
  rust/
    great_kingdom_core/
      Cargo.toml
      pyproject.toml
      src/
        lib.rs
        game.rs
        rules.rs
        territory.rs
        mcts.rs
        ffi.rs
```

## 개발 순서

권장 개발 순서는 다음과 같습니다.

1. Rust 규칙 엔진 구현
2. 합법 수, 착수 적용, 파괴 판정, 패스 종료, 영토 점수 테스트 작성
3. PyO3 바인딩으로 Python에서 `GameState` 호출 가능하게 구성
4. Python에서 랜덤 자가대국 검증
5. Rust MCTS를 uniform prior와 value 0으로 먼저 연결
6. PyTorch 모델과 batch leaf evaluation 연결
7. self-play 데이터 저장
8. 정책/value 학습 루프 구현
9. best model 평가와 교체 루프 구현

규칙 엔진의 오류는 이후 학습 데이터 전체를 오염시키므로, 구현 초반에는 Rust 규칙 테스트를 가장 우선합니다.

## 개발 원칙

- 게임 규칙의 단일 기준은 Rust 규칙 엔진으로 둡니다.
- Python에서는 규칙을 중복 구현하지 않습니다.
- MCTS와 신경망에는 사람이 정의한 전술 보정을 넣지 않습니다.
- 즉시 승패는 규칙 엔진의 터미널 결과로 처리합니다.
- 터미널이 아닌 후보 수의 선호도는 policy prior, 방문 횟수, value backup으로 결정합니다.

## 라이선스

라이선스는 아직 정해지지 않았습니다.
