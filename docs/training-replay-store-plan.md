# Gumbel Policy Target 개선 계획

## 배경

Runpod 학습에서 policy loss가 잘 내려가지 않는 문제가 있었다.

관찰 결과:

- single-batch overfit은 성공했다.
  - `total: 5.85 -> 0.35`
  - `policy_kl: 3.99 -> 0.11`
  - `value: 1.68 -> 0.06`
- 원본 replay 일반 학습은 느렸다.
  - `lr=1e-4`, no augmentation, 1000 step 기준 `policy_kl ~= 2.76`
- replay policy target이 사실상 hard label에 가까웠다.
  - entropy mean: `0.1919`
  - entropy p50: `0.00145`
  - max probability mean: `0.9234`
  - max probability p50: `0.99986`
- replay에는 같은 feature에 서로 다른 target이 붙는 중복 state 충돌도 있었다.
  - 50,000 sampled rows 중 duplicate groups: `2852`
  - 가장 큰 duplicate group: 같은 feature가 `1783`번 등장
  - 같은 feature group 안에서 `value`가 `-1.0`, `1.0` 둘 다 존재
  - 같은 feature group 안에서 policy argmax가 수십 개로 갈림
- 기존 replay를 feature 기준으로 aggregate한 뒤 학습하면 개선됐다.
  - aggregated replay samples: `231271 -> 154609`
  - 5000 step 기준 `policy_kl: 2.96 -> 0.98`
  - `value: 0.57 -> 0.07`
  - 20-game arena에서 candidate win rate `55%`

aggregation은 효과가 있었지만, 정식 해결의 첫 번째 방향으로 삼기보다는 Gumbel policy target이
너무 날카롭게 붕괴되는 문제를 먼저 고친다.

## 현재 코드 판단

현재 구현은 selected action을 one-hot target으로 저장하는 버그는 아니다.

Rust ONNX self-play에서는 replay에 다음 값을 저장한다.

```python
policy = np.asarray(result.policy_target(), dtype=np.float32)
action = int(result.selected_action())
```

`selected_action()`은 게임 진행용이고, replay policy target은 `policy_target()`이다.

Rust 쪽도 completed/improved policy target을 계산한다.

- `root_improved_policy_target()`
- `root_policy_target_logits()`
- `completed_q_values()`
- `transformed_completed_q()`

문제는 completed/improved policy를 만들고 있는데도 그 분포가 너무 sharp하다는 점이다.

가장 의심되는 직접 원인은 `transformed_completed_q()`에서 search용 `c_visit=50.0`,
`c_scale=1.0`이 policy target logits에도 그대로 쓰이는 것이다.

현재 변환은 다음 형태다.

```rust
let visit_scale = (c_visit + max_visit_count) * c_scale;
visit_scale * ((q - q_min) / q_range)
```

Runpod 설정에서는 `c_visit=50.0`이므로 policy target softmax 직전 logit 차이가 매우 커질 수
있다. logit 차이가 수십 단위가 되면 softmax 결과는 사실상 one-hot이 된다.

## 개선 방향 우선순위

우선순위는 다음과 같이 잡는다.

1. **Policy target에 temperature scaling을 적용한다.**
2. **Policy label smoothing을 선택적으로 추가한다.**
3. **초반 수 exploration 강제를 선택적으로 추가한다.**

1번은 구현이 가장 작고, 기존 Gumbel search scale 의미를 거의 건드리지 않으면서 target
sharpness만 직접 낮출 수 있다. 2번과 3번은 1번 적용 후에도 target entropy나 self-play
다양성이 부족할 때 추가한다.

## 1. Policy Target Temperature Scaling

### 목표

Search 내부 선택은 기존 `gumbel_c_visit`, `gumbel_c_scale`을 유지하되, replay에 저장하는
policy target softmax에만 별도 temperature를 적용한다.

즉 다음 두 목적을 분리한다.

- search action selection: 강한 ranking과 적은 simulation 효율을 위해 기존 scale 사용
- training target: 모델이 학습 가능한 soft label을 만들기 위해 target logits를 temperature로 완화

### 설정 후보

`SelfPlayConfig`에 다음 필드를 추가한다.

```python
policy_target_temperature: float = 1.0
```

동작:

- `1.0`이면 기존 동작과 같다.
- `> 1.0`이면 replay target policy가 더 부드러워진다.
- `<= 0.0`은 config validation에서 거부한다.

Runpod 후보값:

```json
{
  "gumbel_c_visit": 50.0,
  "gumbel_c_scale": 1.0,
  "policy_target_temperature": 2.0
}
```

실험 후보:

- `T=1.0`: 현재 baseline
- `T=1.5`: 약한 완화
- `T=2.0`: 첫 추천값
- `T=3.0`: 여전히 너무 sharp할 때

### 구현 위치

Rust:

- `GumbelConfig`에 `policy_target_temperature` 필드 추가
- `root_improved_policy_target()`에서 `root_policy_target_logits()` 결과를 softmax하기 전에
  logits를 `policy_target_temperature`로 나눈다.
- `GumbelSelfPlayBatch`와 `GumbelSearch` PyO3 생성자에 optional parameter 추가

Python:

- `SelfPlayConfig`에 `policy_target_temperature` 필드 추가
- `create_core_search_engine()`
- `create_core_self_play_batch()`
- `run_rust_onnx_self_play()`의 `core.GumbelSelfPlayBatch(...)`
- config JSON 로딩 경로

구현 예시:

```rust
let temperature = policy_target_temperature.max(1.0e-6);
let scaled_logits = improved_logits
    .iter()
    .map(|(action, logit)| (*action, *logit / temperature))
    .collect::<Vec<_>>();
```

### 검증 기준

새 self-play artifact 또는 replay diagnostics 기준:

- `policy.entropy.p50`이 `0.001` 수준에서 유의미하게 상승한다.
- `policy.max_probability.p50`이 `0.999` 수준에서 내려간다.
- `legal.rows_with_illegal_target_mass = 0`은 유지된다.
- 첫 목표는 `entropy.p50 >= 0.3`, `max_probability.p50 <= 0.9` 근처다.

권장 smoke:

```bash
great-kingdom-rust-onnx-pipeline \
  --device cuda \
  --pipeline-config configs/runpod/pipeline-runpod.json \
  --train-config configs/runpod/train-runpod.json \
  --arena-config configs/runpod/arena-runpod.json \
  --iterations 1
```

Runpod 비용을 줄이려면 self-play games를 작게 override해서 target entropy만 먼저 본다.

### Replay 재시작 방침

`policy_target_temperature`는 replay에 저장되는 policy target 생성 방식을 바꾼다. 따라서 기존
`replay.npz`에 이미 저장된 target은 자동으로 바뀌지 않는다.

권장 방침:

- replay/work dir은 새로 시작한다.
  - 예: `data/runpod/onnx-pipeline-t2`
  - 기존 raw replay와 새 temperature target replay를 섞지 않는다.
- model checkpoint는 버리지 않는다.
  - 기존 `best.pt` 또는 가장 유망한 candidate를 새 work dir의 `checkpoints/best.pt`로 복사해
    bootstrap으로 사용한다.
- 새 replay로 1 iteration smoke를 먼저 돌린다.
  - replay diagnostics에서 entropy와 max probability를 확인한다.
  - 신호가 좋을 때만 긴 학습과 arena로 넘어간다.

기존 replay를 후처리로 부드럽게 만드는 근사도 가능하다.

```text
p_new(a) ∝ p_old(a)^(1 / T)
```

하지만 기존 replay에는 completed policy logits가 없고 이미 softmax된 확률만 있으므로, 이 방식은
정확한 target temperature 적용과 같지 않다. 정식 검증은 새 self-play replay로 진행한다.

## 2. Policy Label Smoothing

### 목표

1번 이후에도 policy target이 너무 sharp하면 학습 loss 계산 직전 또는 replay 저장 직전에
legal action uniform을 섞는다.

형태:

```python
policy = (1.0 - eps) * policy + eps * legal_uniform
```

중요한 점:

- smoothing은 legal action에만 분배한다.
- illegal action에는 target mass를 주면 안 된다.
- `eps`는 작게 시작한다.

후보값:

```json
{
  "policy_label_smoothing": 0.03
}
```

### 적용 위치

선호 순서:

1. 학습 batch 생성 시 적용
   - 기존 replay를 다시 만들지 않고 실험 가능
   - config만 바꿔 비교하기 쉽다.
2. replay 저장 시 적용
   - 저장 target 자체가 바뀐다.
   - 여러 실험을 비교하기는 덜 편하다.

초기 구현은 학습 batch 생성 시 적용한다.

### 검증 기준

- `compute_losses()`에서 masked policy loss와 함께 정상 동작한다.
- illegal target mass가 0으로 유지된다.
- `policy_kl` 하락이 더 안정적인지 본다.

이 단계는 선택 사항이다. 1번만으로 target entropy와 학습 곡선이 충분히 개선되면 보류한다.

## 3. 초반 수 Exploration 강제

### 목표

초반 state가 반복되면서 비슷한 라인만 생성되는 문제를 줄인다.

Gumbel search는 이미 Gumbel noise를 사용하지만, 현재 Python config의 `temperature_turns`,
`sampling_temperature`는 Rust ONNX 경로에서 action 선택에 실질적으로 반영되지 않는다.

선택지는 두 가지다.

### A. Root logit temperature

초반 N수 동안 neural net root logits를 search에 넘기기 전에 temperature로 나눈다.

```python
if turn < root_logit_temperature_turns:
    logits = logits / root_logit_temperature
```

후보:

```json
{
  "root_logit_temperature_turns": 20,
  "root_logit_temperature": 1.5
}
```

장점:

- action sampling 로직을 크게 바꾸지 않는다.
- Gumbel 후보 sampling 전에 prior를 평평하게 만들어 초반 다양성을 늘린다.

주의:

- leaf evaluator logits에는 적용하지 않고 root self-play search에만 적용한다.
- 너무 큰 temperature는 self-play 품질을 낮출 수 있다.

### B. Improved policy에서 action sampling

초반 N수 동안 `selected_action()` 대신 `policy_target()`에서 action을 sample한다.

장점:

- AlphaZero의 temperature action sampling에 더 가깝다.

주의:

- 현재 target이 너무 sharp하면 1번 없이 효과가 작다.
- 구현이 action 선택 경로에 직접 영향을 주므로 arena/debug 비교가 더 복잡하다.

초기 구현은 A를 우선한다.

## Aggregated Replay의 위치

현재 추가된 replay aggregation은 버리지 않는다.

역할:

- 기존 replay를 살려 학습하는 fallback
- policy target 개선 전후를 비교하는 diagnostic tool
- 중복 state 충돌이 계속 심할 때 사용할 보조 처방

하지만 정식 우선순위는 aggregation이 아니라 target 생성 개선이다.

권장 순서:

1. policy target temperature 적용
2. 새 self-play replay의 policy entropy 확인
3. raw replay 학습 재시도
4. 필요하면 label smoothing 적용
5. 필요하면 early exploration 적용
6. 그래도 같은 state target 충돌이 크면 incremental aggregated training replay store를 정식화

## 구현 체크리스트

- `SelfPlayConfig`에 `policy_target_temperature` 추가
- Rust `GumbelConfig`에 `policy_target_temperature` 추가
- PyO3 생성자 인자 추가
- `root_improved_policy_target()`의 target softmax에 temperature 적용
- 기존 테스트가 기본값에서 기존 동작을 유지하는지 확인
- `policy_target_temperature > 1.0`에서 target entropy가 증가하는 테스트 추가
- replay diagnostics로 entropy 변화 확인
- Runpod config에 보수적 후보값 적용

## Runpod 재검증

Runpod를 다시 켤 때 최소 확인 순서:

1. 작은 self-play/pipeline 1 iteration 실행
2. replay diagnostics 실행
3. entropy와 max probability 확인
4. 1000~5000 step train smoke
5. 유망하면 arena 20 games
6. 통과하면 arena 50~100 games

기대 신호:

- `policy.entropy.p50`이 기존 `0.00145`보다 크게 상승
- `policy.max_probability.p50`이 기존 `0.99986`보다 하락
- 학습 중 `policy_kl`이 1000 step 기준 기존 `2.7`대보다 낮아짐
- value 학습은 기존처럼 안정적으로 하락

## 결론

가장 직접적인 수정은 **policy target에 temperature scaling을 적용하는 것**이다.

현재 구현은 completed policy target을 저장하고 있지만, search용 scale이 target softmax에도
그대로 들어가면서 target이 사실상 one-hot으로 붕괴할 가능성이 높다. 먼저 target softmax에
temperature를 적용하고, 필요할 때만 label smoothing과 초반 exploration을 추가한다.
