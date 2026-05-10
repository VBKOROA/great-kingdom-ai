# Training Efficiency Roadmap

이 문서는 Great Kingdom AI의 학습 파이프라인을 현실적인 학습 환경 제약 안에서
샘플 효율 중심으로 재설계하기 위한 방향을 정리한다.

목표는 EfficientZero를 그대로 복제하는 것이 아니다. 이 프로젝트는 게임 규칙과 상태
전이가 Rust rule engine으로 정확하게 구현되어 있으므로, MuZero/EfficientZero의
learned dynamics를 게임 룰 모델로 다시 학습할 필요는 낮다. 대신 EfficientZero에서
효과가 큰 아이디어 중 현재 구조에 맞는 부분만 가져온다.

프로젝트를 뒤엎는 수준의 개선이기에 기존호환성을 과감히 포기한다.

- trajectory 기반 replay
- 최신 네트워크 기반 reanalyze
- terminal-only value target 개선
- target age/priority 기반 replay sampling
- Python learner의 GPU 활용률 개선
- actor/learner/reanalyze 단계 분리

참고 논문: EfficientZero, "Mastering Atari Games with Limited Data"
https://proceedings.neurips.cc/paper/2021/hash/d5eca8dc3820cad9fe56a3bafda65ca1-Abstract.html

## 환경 제약

### 로컬 개발 환경

- GPU 없는 사무용 노트북
- Python은 venv 사용
- 역할: 테스트, 작은 smoke run, 데이터 구조/로직 검증
- 전제: 로컬에서 본격 학습 성능을 검증하려고 하지 않는다.

### 학습 환경

- Runpod RTX 3090 24GB
- AMD EPYC 7C13
- Pod template: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
- 역할: 본격 self-play, replay 누적, CUDA 학습, ablation

3090 24GB는 중간 규모 residual policy-value network 학습에는 충분하지만, 거대한
actor pool이나 대형 learned world model을 함께 돌리기에는 여유가 크지 않다. 따라서
우선순위는 "복잡한 모델 추가"보다 "같은 self-play 데이터를 더 잘 재사용하는 구조"에 둔다.

## 현재 파이프라인의 핵심 한계

현재 구조는 AlphaZero-lite에 가깝다.

1. Rust/ONNX self-play가 게임을 생성한다.
2. 각 착수 state에서 policy target과 terminal value target을 저장한다.
3. Python learner가 replay에서 배치를 뽑아 `policy CE + value MSE`를 학습한다.
4. candidate를 저장하고 best로 promote한다.

이 방식은 단순하고 안정적이지만 샘플 효율 측면에서는 몇 가지 한계가 있다.

### Replay가 transition/trajectory를 잃는다

현재 replay sample은 독립 샘플 중심이다.

- `features`
- `policy`
- `value`
- `root_policy_logits`
- `sample_weight`

이 구조는 `s_t -> a_t -> s_{t+1}` 관계를 직접 표현하지 않는다. 그 결과 다음 기능을
깔끔하게 붙이기 어렵다.

- n-step value target
- reanalyze target refresh
- target age 관리
- prioritized replay
- representation consistency auxiliary loss

### Value target이 terminal result에 강하게 묶여 있다

현재 value target은 대부분 승패 기준 `+1/-1`이다. 중간 state의 value를 terminal까지
기다려서만 학습하면 credit assignment가 느리다. 특히 초반 state는 terminal outcome과
거리도 멀고 노이즈도 크다.

Great Kingdom은 보드게임이고 environment reward가 sparse한 편이므로, EfficientZero의
value prefix를 그대로 가져오는 것보다 다음 접근이 더 현실적이다.

- terminal value는 유지한다.
- 최신 value network로 bootstrap target을 추가한다.
- trajectory 위치와 target age에 따라 bootstrap 길이를 조절한다.

### 오래된 target을 계속 학습한다

self-play 당시의 policy target은 그 시점의 네트워크와 search 설정으로 만들어진다.
학습이 진행되면 오래된 target은 현재 네트워크 관점에서 낮은 품질일 수 있다.

현재는 `root_policy_logits`를 저장하지만 학습 loss에 직접 쓰거나, target을 최신 모델로
갱신하는 reanalyze 단계는 없다.

### Python learner가 매 step 배치를 즉석 구성한다

현재 learner는 매 step마다 replay에서 샘플을 뽑고, NumPy 배열을 stack하고, legal mask를
만든 뒤 GPU로 복사한다. batch size가 커질수록 GPU 연산보다 CPU 준비/복사가 병목이 될 수
있다.

## 가져오지 않을 것

### Learned dynamics를 rule engine 대체용으로 넣지 않는다

MuZero/EfficientZero의 learned dynamics는 환경 transition을 모르는 상황에서 latent
state로 MCTS를 진행하기 위한 구성이다. 이 프로젝트는 이미 Rust rule engine이 정확한
next state를 제공한다.

따라서 learned dynamics로 게임 룰을 다시 학습하면 다음 문제가 생긴다.

- 정확한 rule engine을 근사 모델로 대체한다.
- 모델 오류가 search 품질을 떨어뜨릴 수 있다.
- ONNX/Rust/Python 경계가 복잡해진다.
- 3090 24GB 환경에서 학습/추론 메모리와 구현 부담이 커진다.

단, learned dynamics를 완전히 금지하는 것은 아니다. 나중에 auxiliary representation
learning 용도로 `h(s_t), a_t -> h(s_{t+1})`를 예측하게 하는 것은 가능하다. 하지만 이 경우도
search의 transition source는 Rust rule engine으로 유지한다.

## 목표 아키텍처

기존 파이프라인 호환성은 필수가 아니다. 새 파이프라인은 다음 역할로 나눈다.

```text
Rust self-play actor
  -> trajectory shards
  -> trajectory replay store
  -> reanalyze worker
  -> training replay view
  -> Python learner
  -> checkpoint / ONNX export
```

### 1. Self-play actor

역할:

- Rust rule engine + Gumbel search로 game trajectory 생성
- search policy target 저장
- root value/logits 저장
- 사용한 model version 저장

저장 단위는 개별 sample보다 episode/transition이 좋다.

권장 transition schema:

```text
episode_id: int
timestep: int
player: int
features: float32[C, 9, 9]
legal_mask: bool[82]
action: int
policy_target: float32[82]
root_policy_logits: float32[82] | missing
root_value: float32 | missing
next_features: float32[C, 9, 9] | optional
winner: int | missing
terminal: bool
model_version: int
search_config_hash: str
created_iteration: int
```

`next_features`는 저장 공간을 더 쓰지만, auxiliary loss와 n-step target 구현을 단순하게
한다. 저장 용량이 부담되면 `episode_id + timestep`으로 다음 row를 찾아도 된다.

### 2. Trajectory replay store

초기 구현은 복잡한 DB보다 shard 파일이 낫다.

권장 방식:

- iteration 또는 batch 단위 `.npz` shard 저장
- 별도 index JSONL 또는 SQLite metadata 저장
- train 시에는 여러 shard를 memory-map 또는 lazy load

로컬 테스트 용이성을 위해 첫 버전은 다음을 우선한다.

- 작은 in-memory `TrajectoryReplayBuffer`
- `.npz` 저장/로드
- deterministic sampling 테스트

이후 Runpod용으로 shard 기반 loader를 확장한다.

### 3. Reanalyze worker

역할:

- 오래된 trajectory state를 최신 checkpoint로 다시 평가한다.
- 필요하면 작은 search를 다시 돌려 policy target을 갱신한다.
- value bootstrap target을 계산한다.
- target freshness metadata를 갱신한다.

처음부터 full search reanalyze를 넣을 필요는 없다. 단계적으로 나눈다.

#### Reanalyze v1: network-only refresh

- 최신 model로 `policy_logits`, `value`만 계산
- value bootstrap target 생성
- policy target은 기존 search target 유지

장점:

- 구현이 쉽다.
- Rust search 재실행이 필요 없다.
- value 학습 효율을 먼저 개선할 수 있다.

#### Reanalyze v2: root search refresh

- 선택된 오래된 state에 대해 Rust Gumbel search를 다시 실행
- 최신 model 기반 policy target으로 갱신
- 비용이 크므로 priority가 높은 일부 state에만 적용

### 4. Training replay view

learner가 직접 raw trajectory를 다루지 않고, 학습용 view가 batch를 제공한다.

배치에는 다음 target을 포함한다.

```text
features
policy_target
value_target
legal_mask
sample_weight
target_age
priority
```

향후 auxiliary loss를 넣으면 다음도 추가한다.

```text
action
next_features
next_legal_mask
```

이 계층을 두면 replay 저장 구조가 바뀌어도 training loop 변경을 줄일 수 있다.

### 5. Python learner

학습 loss는 단계적으로 확장한다.

초기:

```text
loss = policy_loss + value_loss
```

1차 개선:

```text
loss = policy_loss + value_loss_bootstrapped
```

2차 개선:

```text
loss = policy_loss
     + value_loss_bootstrapped
     + auxiliary_consistency_loss
```

현재 모델을 바로 갈아엎기보다, 먼저 target/replay 구조를 개선한 뒤 model split을 검토한다.

## 우선순위별 실행 계획

### Phase 0: 측정 기준 정리

구현 전에 학습 효율을 비교할 기준을 고정한다.

필수 metric:

- self-play games
- raw transitions
- unique replay rows
- learner steps
- wall-clock time
- policy KL
- value MSE
- arena win rate 또는 fixed candidate matrix
- replay target age 분포
- GPU utilization
- samples/sec

비교 기준:

- 같은 self-play games에서 더 강한가
- 같은 wall-clock에서 더 강한가
- 같은 learner steps에서 loss/arena가 더 빠르게 개선되는가

### 기존의 best 모델이 있기에 개선된 파이프라인으로 학습시킨 후, 간단히 arena 해보는 것으로 효율을 비교할 수 있다.
### 고로 Phase 0 은 무시한다.

### ~~Phase 1: Trajectory replay 도입~~

목표:

- sample replay와 별도로 trajectory replay를 추가한다.
- 기존 파이프라인이 깨져도 괜찮으므로 새 entrypoint를 만들어도 된다.

구현 후보:

- `python/great_kingdom_ai/trajectory_replay.py`
- `python/great_kingdom_ai/trajectory_targets.py`
- `tests/test_trajectory_replay.py`

완료 조건:

- self-play 결과를 episode/transition 단위로 저장할 수 있다.
- 저장/로드 후 deterministic하게 같은 batch를 샘플링할 수 있다.
- 기존 `ReplaySample`로 변환하는 compatibility view를 제공한다.

### ~~Phase 2: n-step/bootstrap value target~~

목표:

- terminal-only value target에서 벗어난다.
- sparse reward 보드게임에 맞게 `V(s_{t+k})` bootstrap을 사용한다.

Great Kingdom에서는 중간 reward가 없다면 기본 target은 다음 형태가 된다.

```text
z_t = gamma^k * V(s_{t+k})
```

terminal이 k-step 안에 있으면:

```text
z_t = terminal_result
```

권장 시작값:

- `gamma = 1.0`
- `k = 4` 또는 `8`
- target age가 오래될수록 `k`를 짧게 둔다.

주의:

- two-player zero-sum perspective 변환을 명확히 해야 한다.
- `V(s_{t+k})`는 `s_t`의 player perspective로 변환되어야 한다.
- bootstrap target은 stop-gradient target이다.

### ~~Phase 3: Reanalyze v1~~

목표:

- 최신 checkpoint로 replay state value를 다시 평가한다.
- training target file을 별도로 생성한다.

구현 후보:

- `python/great_kingdom_ai/reanalyze.py`
- `great-kingdom-reanalyze` CLI

입력:

- trajectory replay shard
- checkpoint
- target output path
- batch size
- device

출력:

- refreshed value predictions
- model version
- target age
- bootstrap value target

완료 조건:

- 동일 trajectory에 대해 checkpoint별 target snapshot을 만들 수 있다.
- learner는 raw replay가 아니라 target snapshot을 읽어 학습할 수 있다.

### ~~Phase 4: Priority/age-aware sampling~~

목표:

- 오래된 target, value error가 큰 sample, policy KL이 큰 sample을 더 자주 학습한다.

초기 priority:

```text
priority = 1
         + a * abs(value_target - value_pred)
         + b * policy_kl
         + c * target_age_score
```

처음부터 sum-tree까지 만들 필요는 없다. shard 단위에서는 NumPy 확률 샘플링으로 충분하다.
성능 병목이 확인되면 alias table 또는 segment tree를 도입한다.

### Phase 5: Learner throughput 개선

목표:

- 3090에서 GPU가 놀지 않게 한다.

우선순위:

1. replay batch를 NumPy contiguous array로 제공
2. pinned memory 사용
3. async CUDA copy
4. prefetch thread/process
5. legal mask precompute 또는 GPU 생성
6. augmentation을 batch tensor 연산으로 이동
7. `torch.compile` 실험

주의:

- 로컬 CPU 테스트가 깨지면 안 된다.
- DataLoader worker가 PyTorch/CUDA context와 충돌하지 않게 CPU preprocessing과 CUDA copy 경계를 분리한다.

### Phase 6: Actor/Learner 프로세스 분리

목표:

- self-play actor와 Python learner를 코드/프로세스 단위로 분리한다.
- 단일 Runpod 3090 안에서 GPU 경합을 통제할 수 있게 한다.
- 기존 sequential pipeline을 유지하지 않아도 되는 v2 구조로 전환한다.

Runpod 단일 GPU 환경에서도 actor/learner 분리는 충분히 일반적이다. 다만 여기서의 분리는
대규모 분산 시스템이 아니라, 같은 machine 안에서 역할을 나눈 light async 구조를 의미한다.

권장 구조:

```text
actor process
  - best-N.onnx를 읽는다.
  - Rust rule engine + ONNX evaluator로 self-play를 생성한다.
  - trajectory shard를 append-only로 저장한다.

reanalyze process
  - learner checkpoint를 읽는다.
  - trajectory shard에서 target snapshot을 만든다.
  - GPU budget이 부족하면 learner가 쉬는 구간에만 실행한다.

learner process
  - target snapshot 또는 replay view를 읽는다.
  - CUDA 학습을 수행한다.
  - candidate/training-latest checkpoint를 저장한다.

coordinator process
  - model version, shard status, checkpoint promotion을 관리한다.
```

단일 3090에서 동시에 돌릴 때의 원칙:

- learner가 CUDA를 가장 우선 사용한다.
- actor의 ONNX inference batch는 learner와 동시에 돌릴 경우 작게 제한한다.
- self-play actor가 GPU를 많이 쓰면 actor와 learner를 시간 분할한다.
- 처음에는 `actor -> learner -> actor` 순차 실행에 가깝게 두고, shard I/O와 target 생성만 비동기화한다.
- GPU utilization, samples/sec, self-play games/hour를 보고 overlap을 늘린다.

권장 구현 단계:

1. actor와 learner entrypoint를 분리한다.
2. shard 파일을 통해서만 데이터를 교환한다.
3. `metadata.jsonl` 또는 SQLite로 shard 상태를 관리한다.
4. actor는 특정 checkpoint version에 고정된 trajectory만 생성한다.
5. learner는 target snapshot version을 명시적으로 읽는다.
6. 안정화 후 actor prefetch와 reanalyze를 background process로 둔다.

피해야 할 구조:

- 하나의 Python process에서 actor/search/learner 상태를 모두 들고 있는 구조
- learner가 쓰는 checkpoint를 actor가 동시에 덮어쓰는 구조
- replay 원본을 reanalyze가 직접 덮어쓰는 구조
- GPU OOM이 날 때까지 actor batch와 learner batch를 동시에 키우는 구조

### Phase 7: Reanalyze v2

목표:

- 일부 state에 대해 최신 checkpoint + Rust search로 policy target을 다시 만든다.

대상:

- target age가 큰 state
- policy KL이 큰 state
- value error가 큰 state
- 중요한 opening/midgame state

처음부터 모든 replay를 search reanalyze하면 비용이 크다. `top p%` 또는 고정 budget으로 제한한다.

## 보류할 개선

### Full learned dynamics

보류 이유:

- 정확한 Rust rule engine이 이미 있다.
- search transition source를 learned model로 바꾸면 검증 난이도가 급증한다.
- 3090 24GB 단일 GPU 환경에서 actor/search/learner/reanalyze와 함께 운영하기 부담스럽다.

### 대형 모델 우선 확장

모델을 키우는 것은 마지막 수단이다. 샘플 효율이 낮은 상태에서 모델만 키우면 self-play target
품질과 replay 재사용 문제가 그대로 남는다.

### 복잡한 분산 actor system

초기에는 단일 Runpod 안에서 sequential 또는 light async 구조로 충분하다. actor/learner를
완전히 분산하면 운영 복잡도가 먼저 커질 수 있다.

## 권장 v2 파이프라인

기존 파이프라인이 깨져도 된다는 전제라면 다음 entrypoint를 새로 두는 것이 가장 깔끔하다.

```text
great-kingdom-train-v2
```

실행 단계:

1. 현재 best checkpoint를 ONNX로 export한다.
2. Rust self-play actor가 trajectory shard를 생성한다.
3. reanalyze v1이 최신 checkpoint로 value target snapshot을 만든다.
4. learner가 target snapshot에서 학습한다.
5. candidate checkpoint를 저장한다.
6. optional arena 또는 fixed evaluation을 실행한다.
7. best/training checkpoint를 갱신한다.
8. metrics를 JSONL로 남긴다.

중요한 점은 replay 원본과 target snapshot을 분리하는 것이다.

```text
trajectory replay = 관측된 사실
target snapshot = 특정 checkpoint/search 설정으로 계산한 학습 target
```

이 분리를 해두면 reanalyze, target policy 변경, value target 변경, priority 변경을 replay
원본 손상 없이 반복 실험할 수 있다.

## 첫 구현 범위

가장 현실적인 첫 PR 범위는 다음이다.

1. `TrajectorySample`, `TrajectoryEpisode`, `TrajectoryReplayBuffer` 추가
2. Rust self-play output을 trajectory 형태로 변환하는 adapter 추가
3. trajectory replay를 기존 `ReplaySample` batch로 변환하는 view 추가
4. n-step/bootstrap target 계산 유틸 추가
5. 작은 synthetic trajectory 테스트 추가

이 범위에서는 learner/model을 크게 바꾸지 않는다. replay 구조부터 바꾸고, 기존 학습 loss에
넣을 수 있는 형태로 compatibility view를 제공한다.

## 성공 기준

단기 성공 기준:

- 기존 sample replay 없이도 v2 replay에서 학습 batch를 만들 수 있다.
- terminal-only target과 bootstrap target을 같은 trajectory에서 비교할 수 있다.
- 로컬 CPU 테스트가 빠르게 돈다.

중기 성공 기준:

- 같은 self-play game 수에서 v1보다 policy KL/value loss가 빠르게 내려간다.
- 같은 Runpod wall-clock에서 fixed arena 성능이 개선된다.
- replay target age와 reanalyze 효과를 metrics로 확인할 수 있다.

장기 성공 기준:

- self-play game 수를 늘리지 않고도 candidate promotion 속도가 개선된다.
- full-search sample 비율을 줄여도 성능 하락이 작다.
- reanalyze와 priority sampling으로 replay 재사용 효율이 증가한다.
