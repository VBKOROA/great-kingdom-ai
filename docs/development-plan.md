# 개발 계획표

이 문서는 Great Kingdom AI를 구현하기 위한 단계별 개발 계획이다.

기준 문서는 다음 세 문서로 둔다.

- [rule-spec.md](rule-spec.md): 게임 규칙과 승패 판정의 단일 기준
- [python-rust-architecture.md](python-rust-architecture.md): Python과 Rust의 책임 분리
- [alphazero-lite.md](alphazero-lite.md): MCTS, self-play, 학습 루프 설계

개발 환경은 GPU 없는 사무용 노트북이고, 학습 환경은 Runpod RTX 4090 24GB를 기준으로 한다. 따라서 로컬에서는 규칙 정확성, 테스트 용이성, 빠른 smoke test를 우선하고, 긴 self-play와 본 학습은 학습 환경에서 실행하는 구조로 나눈다.

## 1. 개발 원칙

1. 규칙 엔진은 Rust를 단일 기준으로 둔다.
2. Python에서는 합법 수, 승패 판정, 영토 판정을 중복 구현하지 않는다.
3. 모든 핵심 규칙은 작은 단위 테스트로 검증 가능해야 한다.
4. MCTS와 학습 코드는 로컬에서도 최소 smoke test가 가능해야 한다.
5. 성능 최적화는 정확성 테스트가 충분히 쌓인 뒤 진행한다.

## 2. 마일스톤 요약

| 단계 | 목표 | 핵심 산출물 | 완료 기준 |
| --- | --- | --- | --- |
| M0 | 프로젝트 기준 정리 | 문서, 패키지 스캐폴드 | `pytest`, `cargo test`가 최소 통과 |
| M1 | Rust 규칙 엔진 구현 | `GameState`, 보드, 액션, 종료 상태 | 기본 착수와 패스 테스트 통과 |
| M2 | 영토와 파괴 판정 구현 | 영토 계산, 그룹 탐색, 즉시 승패 | 규칙 문서의 예외 케이스 테스트 통과 |
| M3 | Python 바인딩 연결 | PyO3 API, Python import | Python에서 `GameState` 조작 가능 |
| M4 | 랜덤 self-play 검증 | 랜덤 플레이어, 게임 로그 | 수천 판 smoke test에서 panic 없음 |
| M5 | Rust MCTS 1차 구현 | PUCT, uniform prior, value 0 | MCTS가 합법 수만 선택 |
| M6 | PyTorch 모델 연결 | 작은 policy-value network | batch inference smoke test 통과 |
| M7 | self-play 데이터 파이프라인 | replay buffer, target 저장 | 한 판의 학습 샘플 저장과 로드 가능 |
| M8 | 학습 루프 구현 | train script, checkpoint | 작은 batch로 loss가 계산되고 저장됨 |
| M9 | 평가와 모델 교체 | arena 평가, best model 관리 | 후보 모델과 best model 비교 가능 |

## 3. 단계별 계획

### ~~M0. 기준 정리와 개발 환경~~

| 작업 | 위치 | 검증 |
| --- | --- | --- |
| 문서 기준 확인 | `docs/` | README와 docs 링크가 일관됨 |
| Python 패키지 import 테스트 유지 | `tests/` | `pytest` 통과 |
| Rust crate 빌드 확인 | `rust/great_kingdom_core/` | `cargo test` 통과 |

우선순위는 낮지만, 개발이 진행되면 README의 현재 상태와 실제 구현 상태를 함께 갱신한다.

### ~~M1. Rust 규칙 엔진 기본형~~

| 작업 | 위치 | 검증 |
| --- | --- | --- |
| 칸 상태 enum 정의 | `rust/great_kingdom_core/src/` | 초기 중앙 중립 성 테스트 |
| 플레이어, 액션, 종료 사유 타입 정의 | `rust/great_kingdom_core/src/` | 타입별 생성 테스트 |
| `GameState::new()` 구현 | `rust/great_kingdom_core/src/` | 9x9 보드와 선공 차례 확인 |
| 합법 수 생성 1차 구현 | `rust/great_kingdom_core/src/` | 빈 칸 80개와 패스 확인 |
| 착수 적용 기본 구현 | `rust/great_kingdom_core/src/` | 차례 전환, 사용 성 개수 확인 |
| 패스 처리 구현 | `rust/great_kingdom_core/src/` | 연속 패스 종료 확인 |

이 단계에서는 영토와 파괴 판정을 단순화하지 말고, 구현하지 않은 부분은 명시적으로 테스트에서 드러나게 둔다.

### ~~M2. 영토, 포위, 승패 판정~~

| 작업 | 위치 | 검증 |
| --- | --- | --- |
| 연결 성 그룹 탐색 | `rust/great_kingdom_core/src/` | 단일 그룹, 분리 그룹 테스트 |
| 자유 공간 판정 | `rust/great_kingdom_core/src/` | 보드 밖과 대각선이 자유 공간이 아님을 확인 |
| 상대 성 파괴 판정 | `rust/great_kingdom_core/src/` | 상대 그룹 파괴 시 즉시 승리 |
| 자살수 판정 | `rust/great_kingdom_core/src/` | 상대 파괴 없으면 즉시 패배 |
| 판정 순서 구현 | `rust/great_kingdom_core/src/` | 동시 파괴 상황에서 현재 플레이어 승리 |
| 영토 계산 구현 | `rust/great_kingdom_core/src/` | 폐쇄 영역, 중립 성, 보드 가장자리 케이스 |
| 점수 승패 판정 | `rust/great_kingdom_core/src/` | 파랑 3점 이상 우위 조건 확인 |

이 단계의 테스트가 전체 프로젝트의 기반이다. 이후 AI 코드에서 이상한 착수가 나오면 먼저 이 테스트 묶음으로 회귀 여부를 확인한다.

### ~~M3. Python 바인딩~~

| 작업 | 위치 | 검증 |
| --- | --- | --- |
| PyO3 module export | `rust/great_kingdom_core/src/lib.rs` | `import great_kingdom_core` 성공 |
| `GameState` Python class 노출 | `rust/great_kingdom_core/src/` | Python에서 새 게임 생성 |
| `legal_actions()` 노출 | `rust/great_kingdom_core/src/` | Python list 또는 tuple 반환 |
| `apply_action()` 노출 | `rust/great_kingdom_core/src/` | 다음 상태와 outcome 반환 |

Python API는 테스트하기 쉬운 작은 메서드 중심으로 시작한다. self-play orchestration은 이 API가 안정된 뒤 붙인다.

### M4. 랜덤 self-play 검증

| 작업 | 위치 | 검증 |
| --- | --- | --- |
| 랜덤 합법 수 선택기 | `python/great_kingdom_ai/` | 불법 수 선택 없음 |
| 랜덤 self-play loop | `python/great_kingdom_ai/` | 한 판 종료 가능 |
| 게임 로그 포맷 | `python/great_kingdom_ai/` | 재현 가능한 seed 저장 |
| 대량 smoke test | `tests/` | 여러 seed에서 panic과 무한 루프 없음 |

로컬에서도 빠르게 돌 수 있어야 하므로, 이 단계의 테스트는 작은 판 수와 고정 seed를 사용한다.

### M5. Rust MCTS 1차 구현

| 작업 | 위치 | 검증 |
| --- | --- | --- |
| MCTS config 정의 | `rust/great_kingdom_core/src/` | 기본값 테스트 |
| 노드와 edge 통계 구조 | `rust/great_kingdom_core/src/` | 방문 횟수와 가치 업데이트 테스트 |
| PUCT 선택 구현 | `rust/great_kingdom_core/src/` | prior와 visit count 반영 확인 |
| terminal value backup | `rust/great_kingdom_core/src/` | 즉시 승패가 백업됨 |
| uniform prior evaluator | `rust/great_kingdom_core/src/` | 신경망 없이 탐색 가능 |
| Python 호출 API | `rust/great_kingdom_core/src/` | MCTS가 합법 수 하나를 반환 |
| visit count export | `rust/great_kingdom_core/src/` | 82차원 방문 횟수 분포 반환 |

초기 MCTS는 신경망 없이 uniform prior와 value 0으로 시작한다. 이 구조가 안정되면 PyTorch leaf evaluation을 연결한다. 이후 self-play 학습 데이터로 쓰기 위해 선택된 수뿐 아니라 root visit count 분포도 Python으로 넘길 수 있어야 한다.

### M6. PyTorch 모델과 batch 평가

| 작업 | 위치 | 검증 |
| --- | --- | --- |
| feature plane 생성 API 노출 | `rust/great_kingdom_core/src/` | shape와 값 범위 테스트 |
| feature tensor 변환 | `python/great_kingdom_ai/` | `[batch, channels, 9, 9]` 확인 |
| 작은 policy-value network | `python/great_kingdom_ai/model.py` | forward shape 테스트 |
| legal mask API 노출 | `rust/great_kingdom_core/src/` | 82차원 mask와 `legal_actions()` 일치 |
| Rust eval request 구조 | `rust/great_kingdom_core/src/` | leaf state batch 반환 |
| Python batch inference 연결 | `python/great_kingdom_ai/` | policy 82개, value scalar 반환 |
| Rust MCTS prior masking | `rust/great_kingdom_core/src/` | 불법 수 prior가 탐색 후보에서 제외 |

모델은 로컬 smoke test가 가능한 작은 CNN으로 시작한다. 학습 처리량 최적화는 Runpod RTX 4090 24GB 환경에서 학습 루프가 닫힌 뒤 별도 작업으로 둔다.

### M7. self-play 데이터 파이프라인

| 작업 | 위치 | 검증 |
| --- | --- | --- |
| 방문 횟수 기반 policy target 저장 | `python/great_kingdom_ai/` | 합이 1인 82차원 target |
| value target 계산 | `python/great_kingdom_ai/` | 현재 플레이어 관점 부호 확인 |
| replay buffer 구현 | `python/great_kingdom_ai/replay_buffer.py` | push, sample, save, load 테스트 |
| 대칭 증강 구현 | `python/great_kingdom_ai/` | 보드와 policy index가 함께 변환 |
| self-play artifact 저장 | `data/` 또는 설정 경로 | 재시작 후 로드 가능 |

데이터는 사람이 만든 전술 label을 포함하지 않는다. 규칙 결과, MCTS 방문 분포, 최종 승패만 저장한다.

### M8. 학습 루프

| 작업 | 위치 | 검증 |
| --- | --- | --- |
| loss 함수 구현 | `python/great_kingdom_ai/train.py` | policy loss와 value loss 계산 |
| optimizer와 scheduler 설정 | `python/great_kingdom_ai/train.py` | 1 step 업데이트 성공 |
| checkpoint 저장 | `python/great_kingdom_ai/` | 모델과 optimizer 상태 저장 |
| checkpoint 로드 | `python/great_kingdom_ai/` | 로드 후 같은 입력에 같은 출력 |
| 작은 end-to-end 학습 | `tests/` 또는 script | toy batch로 학습 script 완료 |

이 단계의 목표는 강한 모델이 아니라 닫힌 루프다. self-play 데이터 생성, 학습, checkpoint 저장이 한 번에 이어지면 다음 단계로 넘어간다.

### M9. 평가와 모델 교체

| 작업 | 위치 | 검증 |
| --- | --- | --- |
| arena match runner | `python/great_kingdom_ai/evaluate.py` | 두 모델 대국 실행 |
| best model 관리 | `checkpoints/` 또는 설정 경로 | best와 candidate 구분 |
| 승률 리포트 | `python/great_kingdom_ai/` | seed, 판 수, 승률 기록 |
| 모델 교체 조건 | `python/great_kingdom_ai/` | 기준 승률 이상이면 best 갱신 |
| 회귀 평가 | `tests/` 또는 script | 작은 판 수로 평가 루프 smoke test |

로컬에서는 평가 판 수를 작게 유지하고, 긴 학습과 대량 평가는 Runpod 학습 환경으로 넘긴다.

## 4. 권장 작업 순서

1. `cargo test` 기반 Rust 규칙 테스트를 먼저 촘촘히 만든다.
2. Rust `GameState` API를 Python에서 호출할 수 있게 만든다.
3. Python 랜덤 self-play로 규칙 엔진의 안정성을 검증한다.
4. Rust MCTS를 신경망 없이 먼저 동작시킨다.
5. 작은 PyTorch 모델을 붙이고 batch 평가 경계를 확정한다.
6. self-play 데이터 저장과 replay buffer를 만든다.
7. 학습 루프와 checkpoint를 연결한다.
8. 평가 루프와 best model 교체를 붙인다.

## 5. 테스트 계획

| 범위 | 테스트 종류 | 목적 |
| --- | --- | --- |
| Rust 규칙 엔진 | 단위 테스트 | 규칙 문서의 세부 조건 고정 |
| Rust MCTS | 단위 테스트 | 합법 수 선택, PUCT, backup 검증 |
| PyO3 바인딩 | 통합 테스트 | Python에서 Rust API 호출 가능 여부 |
| Python 모델 | 단위 테스트 | tensor shape, dtype, mask 검증 |
| self-play | smoke test | 무한 루프와 crash 방지 |
| 학습 루프 | smoke test | 작은 batch로 forward, backward, save 확인 |
| 평가 루프 | smoke test | 모델 비교가 끝까지 실행되는지 확인 |

테스트는 빠른 기본 세트와 느린 실험 세트를 분리한다. 기본 세트는 노트북에서 자주 실행할 수 있어야 한다.

## 6. 환경별 실행 기준

| 환경 | 용도 | 기준 |
| --- | --- | --- |
| 로컬 개발 환경 | 규칙 엔진, 바인딩, 작은 smoke test | 빠르게 반복 가능한 테스트만 기본값으로 둔다 |
| Runpod 학습 환경 | self-play 대량 생성, 본 학습, 장시간 평가 | RTX 4090 24GB 기준으로 batch와 worker 수를 조정한다 |
| CI 또는 기본 테스트 | 회귀 방지 | GPU 없이도 통과해야 하는 테스트만 포함한다 |

## 7. 완료 정의

프로젝트의 1차 완료 기준은 다음과 같다.

1. Rust 규칙 엔진이 문서화된 핵심 규칙 테스트를 통과한다.
2. Python에서 Rust `GameState`와 MCTS를 호출할 수 있다.
3. 랜덤 self-play와 MCTS self-play가 모두 종료까지 실행된다.
4. self-play 데이터로 작은 PyTorch 모델을 학습할 수 있다.
5. candidate model과 best model의 평가 및 교체 루프가 동작한다.

이 기준을 만족하면 이후 작업은 성능 최적화, 학습 품질 개선, 더 큰 모델 실험, UI 또는 분석 도구 추가로 분리해서 진행한다.
