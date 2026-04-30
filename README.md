# Great Kingdom AI

Great Kingdom AI는 9x9 추상 전략 게임 **Great Kingdom**을 재현하고, 자가대국 기반 학습으로 강해지는 AI 플레이어를 만들기 위한 프로젝트입니다.

현재 저장소는 초기 개발 환경과 최소 패키지 스캐폴드를 포함합니다. 핵심 규칙과 학습 구조, Python/Rust 분리 아키텍처는 문서로 정리해 둔 상태입니다.

## 목표

이 프로젝트의 목표는 다음 세 가지입니다.

1. Great Kingdom 규칙을 정확히 재현하는 규칙 엔진을 만든다.
2. 규칙 엔진 위에서 MCTS 기반 자가대국을 수행한다.
3. 자가대국 데이터를 이용해 AlphaZero-lite 형태의 신경망 플레이어를 학습한다.

정통 AlphaZero를 완전히 재현하기보다는, Colab GPU와 일반 개발 환경에서 현실적으로 실험 가능한 소형 구조를 목표로 합니다.

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

아직 완성된 게임 엔진과 학습 코드는 포함되어 있지 않습니다.

현재 단계에서 이 저장소는 다음을 제공합니다.

- 게임 규칙의 기준 문서
- AI 학습 방식의 설계 문서
- 초기 Python 패키지와 Rust/PyO3 crate 구조
- 개발 순서와 책임 분리 기준

## 개발 환경

Python 3.13과 Rust 1.85 이상을 기준으로 합니다.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
cd rust/great_kingdom_core
../../.venv/bin/python -m maturin develop
cargo test
cd ../..
.venv/bin/python -m pytest
```

Rust 포맷터와 린터는 Ubuntu/Debian 기준으로 다음 패키지가 필요합니다.

```bash
sudo apt install -y rustfmt rust-clippy
```

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
