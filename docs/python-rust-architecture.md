# Python + Rust 구조 명세

## 1. 목표

Great Kingdom AI는 Python과 Rust를 분리해서 만든다.

Python은 순수 AI 레이어를 담당한다.

Rust는 규칙 엔진과 MCTS처럼 반복 호출이 많고 성능이 중요한 레이어를 담당한다.

목표는 Python의 PyTorch 생태계를 유지하면서, 병목이 되는 게임 상태 처리와 탐색은 Rust로 옮기는 것이다.

---

# 2. 역할 분리

## Python

Python은 다음 책임을 가진다.

신경망 모델 정의.

PyTorch 추론과 학습.

리플레이 버퍼 관리.

자가대국 작업 orchestration.

체크포인트 저장과 로드.

학습 로그와 평가 리포트 생성.

Rust MCTS가 요청한 leaf state 배치 평가.

Python은 게임 규칙을 중복 구현하지 않는다. 디버깅용 렌더링이나 테스트 helper를 제외하면, 합법 수와 승패 판정의 기준은 Rust 규칙 엔진 하나로 유지한다.

## Rust

Rust는 다음 책임을 가진다.

보드와 게임 상태 표현.

합법 수 생성.

착수 적용.

즉시 승패 판정.

패스 종료와 영토 점수 판정.

MCTS 트리 관리.

PUCT 선택과 가치 백업.

트랜스포지션 캐시.

자가대국 중 여러 게임 상태의 병렬 진행.

Rust는 신경망 학습을 담당하지 않는다. 신경망 추론도 기본적으로 Python에 맡긴다.

---

# 3. 바인딩 방식

Python과 Rust 연결은 PyO3와 maturin을 사용한다.

Rust crate는 Python extension module로 빌드한다.

Python에서는 일반 패키지처럼 import한다.

예상 패키지 이름은 `great_kingdom_core`로 둔다.

예상 호출 형태는 다음과 같다.

```python
import great_kingdom_core as gk

state = gk.GameState.new()
legal = state.legal_actions()
next_state, outcome = state.apply_action(action)
```

MCTS는 단일 착수 선택 API와 배치 자가대국 API를 모두 제공한다.

```python
search = gk.MctsSearch(config)
request = search.start(state)
```

---

# 4. 신경망 평가 경계

MCTS는 leaf node에 도달하면 신경망 평가가 필요하다.

하지만 Rust에서 PyTorch를 직접 호출하지 않는다.

대신 Rust는 평가가 필요한 상태들을 모아 Python에 넘긴다.

Python은 상태 배치를 tensor로 변환하고, 모델로 policy와 value를 계산한 뒤 Rust에 돌려준다.

기본 흐름은 다음과 같다.

1. Rust MCTS가 여러 self-play game을 한 스텝씩 진행한다.
2. 확장해야 할 leaf state들을 모아 `EvalRequest`로 반환한다.
3. Python이 `EvalRequest.states`를 PyTorch tensor로 변환한다.
4. Python 모델이 batch inference를 실행한다.
5. Python이 policy logits와 value를 Rust에 `EvalResult`로 전달한다.
6. Rust가 legal mask를 적용하고 MCTS 백업을 계속한다.

이 구조는 GPU를 Python/PyTorch에 집중시키고, Rust는 CPU 탐색을 빠르게 처리하게 한다.

---

# 5. 데이터 형식

행동 공간은 82개로 고정한다.

0부터 80까지는 9x9 보드 좌표다.

81은 패스다.

Python으로 넘기는 상태 입력은 현재 플레이어 관점으로 정규화된 채널 텐서다.

권장 shape는 다음과 같다.

```text
[batch, channels, 9, 9]
```

Rust는 모델 입력용 feature plane을 생성할 수 있어야 한다.

Python은 그 배열을 tensor로 감싸고 dtype과 device만 맞춘다.

정책 출력은 82개 action logit이다.

가치 출력은 현재 플레이어 기준 `[-1, 1]` 범위의 scalar다.

---

# 6. MCTS 원칙

MCTS는 AlphaZero-lite 명세의 순수 학습 원칙을 따른다.

규칙 외 전술 보정은 넣지 않는다.

즉시 승리 수를 강제 선택하지 않는다.

자살수나 즉시 패배 수에 별도 페널티를 넣지 않는다.

상대의 즉시 승리 수를 미리 찾아 후보 수를 감점하지 않는다.

단, 규칙 엔진이 수를 적용한 결과 터미널 승패를 반환하면 MCTS는 그 값을 그대로 백업한다.

선택식은 PUCT를 사용한다.

```text
score(s, a) = Q(s, a) + c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))
```

policy prior는 합법 수에 대해서만 정규화한다.

불법 수는 방문 후보에서 제외한다.

---

# 7. 병렬화 구조

초기 구현은 단일 프로세스 Python과 Rust 내부 멀티스레드 조합을 권장한다.

Python은 self-play worker 묶음을 관리한다.

Rust는 각 worker의 게임 상태와 MCTS를 진행한다.

평가가 필요한 leaf state는 가능한 한 모아서 Python에 넘긴다.

Python은 한 번의 GPU batch inference로 여러 leaf를 평가한다.

Colab 기준에서는 너무 많은 Python 프로세스를 띄우기보다, Rust 쪽에서 CPU 병렬 탐색을 처리하고 Python은 GPU batch를 크게 만드는 편이 단순하다.

---

# 8. 저장 책임

자가대국 결과 저장은 Python이 담당한다.

Rust는 각 턴마다 다음 데이터를 반환한다.

현재 상태 feature 또는 재생 가능한 compact state.

MCTS 방문 횟수 기반 policy target.

현재 플레이어 정보.

최종 승자.

Python은 이 데이터를 리플레이 버퍼에 넣고, 학습 batch를 만들 때 대칭 증강을 적용한다.

원본 self-play 데이터는 사람이 만든 전술 label을 포함하지 않는다.

---

# 9. 권장 디렉터리 구조

초기 디렉터리 구조는 다음과 같이 둔다.

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

`game.rs`는 상태와 action 타입을 정의한다.

`rules.rs`는 합법 수와 착수 적용을 담당한다.

`territory.rs`는 영토 판정을 담당한다.

`mcts.rs`는 탐색 트리와 PUCT를 담당한다.

`ffi.rs`는 Python에 노출할 PyO3 wrapper만 담당한다.

---

# 10. 개발 순서

1. Rust 규칙 엔진을 먼저 만든다.
2. Rust 단위 테스트로 합법 수, 포획, 자살수, 패스 종료, 영토 점수를 검증한다.
3. PyO3 바인딩으로 Python에서 `GameState`를 호출할 수 있게 한다.
4. Python에서 랜덤 자가대국을 돌려 게임이 정상 종료되는지 확인한다.
5. Rust MCTS를 신경망 없이 uniform prior와 value 0으로 붙인다.
6. Python PyTorch 모델을 붙이고 배치 leaf evaluation을 연결한다.
7. self-play 데이터를 저장한다.
8. 저장 데이터로 정책/value 학습을 돌린다.
9. best model 평가와 교체 루프를 붙인다.

규칙 엔진이 흔들리면 학습 결과 전체가 오염되므로, Rust 규칙 테스트를 가장 먼저 충분히 만든다.

---

# 11. 결론

이 프로젝트는 Python이 AI 실험 속도를 담당하고, Rust가 규칙과 탐색 성능을 담당하는 구조로 간다.

Python은 모델과 학습 루프를 빠르게 바꾸기 위한 레이어다.

Rust는 Great Kingdom의 단일 truth source이며, MCTS를 빠르고 정확하게 실행하는 코어다.
