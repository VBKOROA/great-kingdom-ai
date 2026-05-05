# Gumbel Replay 개선 방향

## 목적

`docs/training-replay-current-diagnosis.md`의 현재 결론을 바탕으로, Gumbel MCTS(Gumbel AlphaZero /
MuZero) 특성에 맞춘 다음 개선 순서를 정리한다.

핵심 목표는 다음 네 가지다.

- target scale 변경 실험에서 optimizer/scheduler 상태가 섞이지 않게 한다.
- replay policy target이 search 이전 prior를 그대로 복사하는지 확인한다.
- exact-state 중복과 terminal value variance를 정식 replay store로 줄인다.
- Completed-Q value blending은 바로 도입하지 않고 별도 ablation으로 격리한다.

## 현재 판단

현재 `policy_target_c_visit/c_scale` 분리는 hard policy target 문제를 크게 완화했다.

```text
entropy.p50:        0.05098 -> 3.16285
max_probability.p50: 0.99194 -> 0.25517
policy KL:          2.x     -> 0.55~0.60
```

하지만 이 결과만으로 search improvement가 충분하다고 결론내릴 수는 없다. Gumbel target은
대략 다음 구조다.

```text
policy_target = softmax(log_prior + completed_Q_bonus)
```

따라서 completed-Q bonus scale이 너무 작거나 temperature가 너무 크면 target이 search-improved
policy가 아니라 root prior에 가까워질 수 있다. 반대로 scale이 너무 크면 다시 one-hot에 가까운
hard target으로 붕괴한다.

현재의 중요한 미해결 문제는 다음이다.

- 같은 시작/초반 exact state가 replay에 반복 저장된다.
- 같은 exact state에 서로 다른 terminal outcome이 붙어 value target variance가 크다.
- value loss `0.8~0.9`는 학습 실패라기보다 terminal Monte Carlo target 구조의 자연스러운 바닥일 수
  있다.

## 원칙

### 1. Arena가 최종 판정이다

policy loss는 target entropy가 커지면 자연스럽게 커진다. 현재 구간에서는 `policy loss`보다
`policy KL`, target-vs-prior diagnostics, arena 결과를 함께 본다.

성공으로 보는 조건:

- `policy KL`이 대략 `0.4~0.8` 범위에서 안정적이다.
- arena win rate가 개선되거나 promotion이 발생한다.
- replay target이 root prior와 완전히 같지는 않다는 진단 지표가 나온다.

### 2. Search improvement를 직접 계측한다

`policy_target_c_scale=0.25`, `policy_target_temperature=2.0`이 너무 soft한지 판단하려면 replay
policy target과 root prior의 거리를 별도로 기록해야 한다.

추가할 diagnostics 후보:

```text
KL(policy_target || root_prior)
KL(root_prior || policy_target)
target_argmax != prior_argmax ratio
target_top1_probability - prior_top1_probability
E_target[completed_Q] - E_prior[completed_Q]
```

처음에는 전부 구현하지 않아도 된다. 최소한 `KL(policy_target || root_prior)`와
`target_argmax != prior_argmax ratio`는 넣는 것이 좋다.

### 3. Aggregate는 정식 replay store 방향으로 본다

Gumbel search에서는 같은 exact state가 여러 번 등장하는 것이 단순 낭비가 아니다. 서로 다른 Gumbel
noise와 이후 trajectory를 샘플링한 결과이므로, 같은 state의 policy/value 평균은 variance reduction
역할을 한다.

다만 raw aggregate를 그대로 최종 해법으로 두면 방문 빈도 정보를 버린다. 정식 방향은 다음 형태가
더 맞다.

```text
state_hash -> {
  features,
  policy_sum / policy_mean,
  value_sum / value_mean,
  count,
  generation_or_age
}
```

학습 시에는 중복 count를 완전히 버리지 말고 완만하게 반영한다.

```text
sample_weight = log1p(count)
sample_weight = sqrt(count)
sample_weight = min(count, 16)
```

목표는 시작 state가 401배로 gradient를 지배하지 않게 하면서도, 자주 등장하는 중요한 opening state를
완전히 과소학습하지 않는 것이다.

### 4. Completed-Q value blending은 후순위다

`V_target = sum(policy_target * completed_Q)`를 terminal outcome `z`와 섞는 아이디어는 value variance를
낮출 수 있다.

하지만 지금 바로 기본 경로에 넣기에는 위험하다.

- 현재 replay value는 terminal outcome만 저장한다.
- Rust `GumbelResult`는 policy target과 visit counts만 노출하고, root completed-Q value를 노출하지
  않는다.
- completed-Q 기반 value target은 모델 자신의 평가값을 더 강하게 재사용하므로 bootstrapping bias와
  self-confirmation 위험이 있다.
- value loss 숫자는 낮아질 수 있지만 arena strength 개선을 보장하지 않는다.

따라서 기본 경로에서는 terminal `z`를 유지하고, exact-state aggregate로 `value_mean`을 만드는 쪽을
먼저 적용한다. Completed-Q value blending은 나중에 `z`와 `root_completed_value`를 둘 다 저장한 뒤
ablation으로 비교한다.

## 실행 우선순위

### 0. Weight-only bootstrap

먼저 checkpoint bootstrap을 깨끗하게 만든다.

필요한 동작:

- checkpoint에서 model state만 로드한다.
- optimizer와 scheduler는 `TrainingConfig` 기준으로 새로 만든다.
- training step은 0부터 시작한다.
- ONNX export와 기존 checkpoint format 호환은 유지한다.

이 작업은 모든 후속 실험의 전제다. target distribution을 크게 바꾼 뒤 Adam moment와 scheduler
step을 이어받으면 실험 변인이 섞인다.

### 1. 현재 target scale의 improvement 진단

현재 후보 설정:

```json
{
  "gumbel_c_visit": 50.0,
  "gumbel_c_scale": 1.0,
  "policy_target_c_visit": 5.0,
  "policy_target_c_scale": 0.25,
  "policy_target_temperature": 2.0
}
```

확인할 것:

- replay target entropy와 max probability
- `policy KL`
- target-vs-prior KL
- target/prior argmax mismatch ratio
- arena win rate

판정:

- arena가 좋아지고 target-vs-prior 거리가 유의미하면 유지한다.
- loss만 예쁘고 arena가 정체되며 target-vs-prior 거리가 작으면 self-imitation 가능성이 높다.

### 2. Target sharpen ablation

현재 설정이 너무 soft하면 다음 설정을 먼저 비교한다.

```json
{
  "policy_target_c_visit": 10.0,
  "policy_target_c_scale": 0.5,
  "policy_target_temperature": 2.0
}
```

대안:

```json
{
  "policy_target_c_visit": 5.0,
  "policy_target_c_scale": 0.25,
  "policy_target_temperature": 1.0
}
```

처음에는 `10.0 / 0.5 / T=2.0` 쪽이 더 보수적이다. temperature를 낮추는 방식은 전체 logit 차이를
직접 키우므로 hard target으로 되돌아갈 가능성이 더 크다.

### 3. Count-aware aggregate replay store

arena가 여전히 정체되거나 opening duplicate가 계속 gradient를 지배하면 replay store를 정식화한다.

필요한 구현 방향:

- replay 저장 단위를 raw row list에서 state hash 기반 record로 확장한다.
- 같은 feature는 policy/value 누적 평균을 갱신한다.
- `count`, `generation_or_age`를 저장한다.
- train batch에서는 `sample_weight`를 loss에 반영하거나 sampling probability에 반영한다.

권장 순서:

1. 현재 `replay_aggregate.py`의 offline aggregate를 count 저장까지 확장한다.
2. train loss에 optional sample weight를 추가한다.
3. pipeline replay store를 online aggregate 형태로 바꾼다.

처음부터 online store까지 한 번에 바꾸지 말고, offline artifact로 arena까지 검증한 뒤 pipeline에
넣는다.

### 4. Completed-Q value ablation

위 작업 후에도 value variance가 명확한 병목으로 남을 때만 진행한다.

조건:

- terminal `z` 기반 aggregate replay와 sharpen ablation을 이미 비교했다.
- arena 정체 원인이 value 쪽이라는 근거가 있다.
- Rust result에 root completed value를 노출하는 변경을 감수할 만큼 필요성이 있다.

실험 형태:

```text
value_replay = (1 - alpha) * z + alpha * root_completed_value
```

후보:

```text
alpha = 0.25
alpha = 0.50
```

주의:

- loss 개선만으로 성공 판정하지 않는다.
- arena와 calibration을 함께 본다.
- root completed value와 terminal z를 모두 저장해 나중에 재가공할 수 있게 한다.

## 후순위 또는 비추천

### min replay turn

opening duplicate를 빠르게 줄일 수 있지만 데이터를 버리는 방식이다. opening을 학습하지 않겠다는
부작용이 있으므로 기본 해법으로 두지 않는다. 필요하면 diagnostic ablation으로만 사용한다.

### 초반 exploration 강제

현재는 trajectory 다양성 부족보다 exact-state 중복과 target variance가 더 직접적인 문제다. 초반
exploration을 더 강하게 주면 policy/value target variance가 커질 수 있다.

도입한다면 다음 조건을 지킨다.

- action selection에만 적용한다.
- replay policy target을 더 랜덤하게 만들지 않는다.
- aggregate/weighting 이후에도 trajectory 다양성이 부족할 때 검토한다.

## 최종 권장 순서

```text
1. [x] weight-only bootstrap 구현
2. target-vs-prior diagnostics 추가
3. 현재 5 / 0.25 / T=2 설정으로 arena 판정
4. 약하면 10 / 0.5 / T=2 sharpen ablation
5. count-aware aggregate replay store 검증
6. 마지막으로 Completed-Q value blending ablation
```

가장 중요한 점은 value loss를 낮추는 변경을 성급히 기본 경로에 넣지 않는 것이다. 지금은 terminal
outcome target을 유지한 채, Gumbel target의 search improvement 여부와 replay 중복/분산 구조를 먼저
분리해서 다루는 편이 더 안전하다.
