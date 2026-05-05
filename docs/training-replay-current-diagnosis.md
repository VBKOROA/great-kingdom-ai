# Training Replay 현재 진단과 진행 상황

## 목적

`Policy Target Temperature Scaling` 구현 후 Runpod 실제 pipeline에서 학습 정체가 다시 나타났다.
이 문서는 현재 관찰된 증상, target scale 분리 실험 결과, aggregate/min replay에 대한 위치, 그리고
다음 판단 기준을 정리한다.

## 이전 관찰: temperature만 적용한 replay

새 work dir에서 기존 replay를 섞지 않고, 기존 best model의 weight만 bootstrap해 실험했다.
`policy_target_temperature=2.0`만 적용한 replay의 1 iteration 학습은 다음 정도로 내려갔다.

```text
step 100:  loss=3.9723 policy=3.1025 value=0.8698 kl=2.7980
step 500:  loss=3.4571 policy=2.7382 value=0.7188 kl=2.4296
step 1000: loss=2.9759 policy=2.4557 value=0.5202 kl=2.1301
```

학습 loss는 내려가지만 policy KL은 여전히 높았다.

Replay diagnostics:

```text
samples: 11565
policy.entropy.p50: 0.05098
policy.max_probability.p50: 0.99194
legal.rows_with_illegal_target_mass: 0

duplicate_groups: 241
duplicate_rows: 1749
value_conflict_groups: 167
policy_argmax_conflict_groups: 144
largest duplicate group: 401 rows
```

중복 state 분석:

```text
pieces 0-2: rows=3083, value_mean=-0.002
pieces 2-4: rows=3316, value_mean=0.022
pieces 4-6: rows=2089, value_mean=0.050

largest group:
  rows=401
  pieces=0.0
  value_mean=-0.042
  value_pos=0.479
  policy_argmaxes=10
```

초반/극초반 state가 전체 replay의 큰 비중을 차지하고, 같은 exact feature에 서로 다른 policy/value
target이 많이 붙는다.

## 진행 상황: policy target scale 분리

`policy_target_temperature`만 올리는 대신, action selection용 Gumbel scale과 replay policy target용
scale을 분리했다.

현재 Runpod 실험 설정:

```json
{
  "gumbel_c_visit": 50.0,
  "gumbel_c_scale": 1.0,
  "policy_target_c_visit": 5.0,
  "policy_target_c_scale": 0.25,
  "policy_target_temperature": 2.0
}
```

의도:

- action selection은 기존처럼 강한 ranking을 유지한다.
- replay policy target은 더 완만한 completed-Q bonus를 사용한다.
- 같은 state에서 policy target argmax가 hard하게 갈리는 문제를 줄인다.

새 work dir `data/runpod/onnx-pipeline-target-scale`에서 기존 best model weight만 bootstrap하고, 기존
replay/log는 섞지 않은 replay diagnostics:

```text
samples: 10331
policy.entropy.p50: 3.16285
policy.max_probability.p50: 0.25517
legal.rows_with_illegal_target_mass: 0

duplicate_groups: 245
duplicate_rows: 1715
value_conflict_groups: 177
policy_argmax_conflict_groups: 125
largest duplicate group: 401 rows
```

정리:

- target sharpness는 크게 완화됐다.
- 이전 `entropy.p50=0.05098`, `max_probability.p50=0.99194`와 비교하면 hard target 문제는 사실상
  해소됐다.
- duplicate/value conflict는 여전히 남아 있다.
- 시작 state 401개 문제도 그대로 남아 있다.

학습 로그는 다음 범위에서 움직였다.

```text
step 100:  loss ~= 4.59  policy ~= 3.69  value ~= 0.90  kl ~= 0.60
step 500:  loss ~= 4.48  policy ~= 3.62  value ~= 0.86  kl ~= 0.58
step 1000: loss ~= 4.43  policy ~= 3.60  value ~= 0.84  kl ~= 0.56
```

여기서 `policy loss`가 커 보이는 것은 target entropy 자체가 높아졌기 때문이다. 현재 policy 학습
상태는 `policy loss`보다 `policy KL`로 보는 것이 맞다.

```text
policy entropy mean ~= 3.05
policy loss ~= 3.6
policy KL ~= 0.55~0.60
```

따라서 이전처럼 `KL=2.x`에 박혀 있던 문제는 해결된 것으로 본다.

## Checkpoint bootstrap 주의

기존 `best.pt`를 새 work dir로 그대로 복사하는 것은 model weight bootstrap 목적에서는 맞다.
하지만 현재 checkpoint format에는 optimizer/scheduler state도 들어 있다.

따라서 `best.pt`를 그대로 `resume_path`로 학습하면 다음 상태까지 이어받을 수 있다.

- Adam moments
- scheduler step
- checkpoint step
- 이미 decay된 effective learning rate

target distribution을 크게 바꾼 실험에서는 optimizer/scheduler는 fresh로 시작하는 것이 더 깨끗하다.
다만 `optimizer_state`와 `scheduler_state`를 단순 삭제하면 현재 `load_checkpoint()`와 ONNX export가
실패한다. 현재 코드에서는 다음 중 하나가 필요하다.

1. model weight만 로드하고 optimizer/scheduler를 새로 만드는 정식 코드 경로 추가
2. 임시로 fresh optimizer/scheduler state를 포함한 checkpoint를 다시 저장

현재 문서 기준의 실험 해석에서는 optimizer/scheduler 상태가 섞였을 가능성을 residual risk로 둔다.

## 현재 판단

### 해결된 문제

`policy_target_c_visit/c_scale` 분리로 replay policy target sharpness 문제는 크게 완화됐다.

- `entropy.p50`: `0.05098 -> 3.16285`
- `max_probability.p50`: `0.99194 -> 0.25517`
- `policy KL`: `2.x -> 0.55~0.60`

따라서 현재 1순위 병목은 더 이상 hard policy target이 아니다.

### 남은 문제

1. **Opening replay distribution collapse**
   - 시작/초반 exact state가 과대표집된다.
   - largest duplicate group은 여전히 401 rows다.
   - 시작 state는 exploration을 늘려도 매 게임 동일하게 등장한다.

2. **High variance Monte Carlo value target**
   - 같은 state에 value `-1/+1`이 동시에 붙는다.
   - 시작 state처럼 실제 기대값이 0 근처인 state도 label은 항상 terminal outcome인 `-1` 또는 `+1`이다.
   - 50:50 state에서 모델이 `0`을 예측해도 MSE floor는 대략 `1.0`이다.
   - 따라서 value loss `0.8~0.9`는 학습 실패라기보다 replay target 구조의 결과일 수 있다.

3. **Policy target이 너무 soft할 가능성**
   - `max_probability.p50=0.255`는 hard target 문제를 해결했지만, 너무 완만할 수도 있다.
   - arena strength가 붙지 않으면 target을 조금 sharpen하는 실험이 필요하다.

## Gemini 반론에 대한 정리

### 반론 1: 같은 state가 여러 outcome을 가지는 것은 정상이다

맞다. 같은 state에서 여러 terminal outcome이 나오는 것은 self-play Monte Carlo target의 자연스러운
특성이다. MSE value loss는 같은 입력 `x`에 여러 `y_i`가 있을 때 평균 target으로 가는 것이 기대
gradient와 일치한다.

Policy cross entropy도 여러 target policy `p_i`의 평균을 학습하는 방향과 연결된다.

다만 현재 문제는 outcome 다양성 자체가 아니라, 같은 exact state가 수백 번 반복되고 그 target이
sharp하게 갈리거나, value label이 terminal outcome만으로 고분산이라는 점이다.

### 반론 2: 초반 state가 너무 많이 중복되는 것이 아니냐

맞다. 현재 데이터는 이 반론을 계속 지지한다.

- `pieces 0-2`, `pieces 2-4` 구간이 전체의 큰 비중을 차지한다.
- largest duplicate group은 시작 state다.
- 시작 state group 안에서 value는 거의 50:50이고 policy argmax는 여러 개로 갈린다.

target scale 분리는 policy 쪽 충돌을 완화했지만, replay distribution과 value variance 문제는
그대로 남아 있다.

## Aggregate에 대한 위치

Aggregate는 효과가 있었다.

- 같은 feature를 하나로 묶어 policy/value target을 평균내면 label conflict가 줄어든다.
- smoke에서 학습 곡선이 크게 개선됐다.

하지만 raw aggregate를 최종 해법으로 쓰는 것은 단점이 있다.

- 방문 빈도 정보를 버린다.
- 자주 등장한 state와 한 번 등장한 state가 비슷한 weight를 갖게 된다.
- opening 같은 중요한 state를 과소학습할 수 있다.
- 원인인 target stochasticity와 opening over-representation을 직접 고치지는 않는다.
- loss 개선이 곧 arena strength 개선을 보장하지 않는다.

따라서 aggregate는 좋은 diagnostic intervention이지만, 현재 기본 실험 경로에는 넣지 않는다.
정식화한다면 단순 collapse가 아니라 `state mean + count-aware weighting` 형태가 더 맞다.

## 현재 비추천 또는 후순위

### min replay / 초반 replay 저장 제한

현재 기본 방향으로 두지 않는다.

`min_replay_turn`은 opening duplicate를 직접 줄일 수 있지만, 데이터를 버리는 방식이다. opening을
학습하지 않겠다는 부작용이 있고, 현재 문제의 핵심인 value target variance와 replay weighting을
정식으로 다루지 않는다.

도입한다면 최종 해법이 아니라 diagnostic ablation으로만 본다.

### 초반 수 exploration 강제

지금 증상에는 1순위가 아니다.

이 기능은 trajectory 다양성이 부족할 때 유효하지만, 현재는 시작/초반 state의 target이 이미 여러
argmax로 갈리고 있다. Exploration을 더 강제하면 policy/value target variance가 커질 수 있다.

도입한다면 다음 원칙을 지킨다.

- action selection에만 적용한다.
- replay policy target을 더 랜덤하게 만들지 않는다.
- target 안정화와 replay weighting 이후에도 trajectory 다양성이 부족할 때 검토한다.

## 다음 실험 판정표

### 1. 현재 target scale 설정으로 arena 확인

현재 설정:

```json
{
  "policy_target_c_visit": 5.0,
  "policy_target_c_scale": 0.25,
  "policy_target_temperature": 2.0
}
```

판정:

- arena win rate가 개선되거나 promotion이 일어나면 현재 방향은 성공이다.
- loss가 예쁘게 내려가지 않아도, KL이 `0.4~0.8` 범위에서 안정적이고 arena가 좋아지면 진행한다.
- value loss `0.8~0.9`만으로 실패로 보지 않는다.

### 2. optimizer/scheduler fresh 여부 확인

기존 best checkpoint의 optimizer/scheduler state를 이어받았을 수 있으므로, weight-only bootstrap
경로를 정식으로 만드는 것이 좋다.

필요한 코드 방향:

- checkpoint에서 model state만 로드
- optimizer/scheduler는 `TrainingConfig` 기준으로 새로 생성
- step은 0부터 시작
- ONNX export는 기존 checkpoint format과 호환 유지

### 3. arena가 약하면 target을 조금 sharpen

현재 target이 너무 soft하다고 판단되면 다음 설정을 우선한다.

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

처음에는 `10.0 / 0.5 / T=2.0` 쪽이 더 보수적이다.

### 4. 그래도 막히면 replay store 정식화

다음 단계의 정식 방향은 raw aggregate나 min replay가 아니라 다음에 가깝다.

```text
state_hash -> {
  features,
  policy_mean,
  value_mean,
  count,
  age/generation
}
```

학습 sampling 또는 loss weighting은 예를 들면 다음처럼 둔다.

```text
sample_weight = sqrt(count)
sample_weight = min(count, 16)
sample_weight = log1p(count)
```

목표:

- 같은 state의 stochastic target은 평균 estimator로 안정화한다.
- 방문 빈도 정보는 완전히 버리지 않는다.
- 시작 state가 401배로 gradient를 지배하지 않게 한다.
