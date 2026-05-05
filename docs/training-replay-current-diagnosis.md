# Training Replay 현재 진단과 해결 후보

## 목적

`Policy Target Temperature Scaling` 구현 후 Runpod 실제 pipeline에서 다시 학습 정체가
나타났다. 이 문서는 현재 관찰된 증상, 반론, aggregate를 둘러싼 판단, 그리고 다음 해결 방향을
정리한다.

## 현재 관찰

새 work dir에서 기존 replay를 섞지 않고, 기존 best model의 weight만 bootstrap해 실험했다.
Fresh optimizer/scheduler를 사용한 1 iteration 학습은 다음 정도로 내려갔다.

```text
step 100:  loss=3.9723 policy=3.1025 value=0.8698 kl=2.7980
step 500:  loss=3.4571 policy=2.7382 value=0.7188 kl=2.4296
step 1000: loss=2.9759 policy=2.4557 value=0.5202 kl=2.1301
```

학습은 내려가지만 policy KL은 여전히 높다.

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

## 현재 판단

`policy_target_temperature=2.0`은 target sharpness를 충분히 완화하지 못했다.

- `entropy.p50=0.05098`은 목표였던 `0.3` 근처에 못 미친다.
- `max_probability.p50=0.99194`는 여전히 거의 hard label에 가깝다.
- illegal target mass는 없으므로 mask나 legal action 저장 문제는 아니다.

더 큰 병목은 다음 두 가지가 섞인 것으로 보인다.

1. **Opening replay distribution collapse**
   - 시작/초반 exact state가 과대표집된다.
   - 시작 state는 exploration을 늘려도 매 게임 동일하게 등장한다.

2. **Stochastic target conflict**
   - 같은 state에서 Gumbel noise, action sampling, terminal outcome에 따라 policy/value target이
     크게 흔들린다.
   - value는 평균으로 수렴할 수 있지만, policy target이 sharp하면 서로 다른 argmax가 강하게
     충돌한다.

## Gemini 반론에 대한 정리

### 반론 1: 같은 state가 여러 outcome을 가지는 것은 정상이다

맞다. 같은 state에서 여러 terminal outcome이 나오는 것은 self-play Monte Carlo target의 자연스러운
특성이다. MSE value loss는 같은 입력 `x`에 여러 `y_i`가 있을 때 평균 target으로 가는 것이
기대 gradient와 일치한다.

Policy cross entropy도 여러 target policy `p_i`의 평균을 학습하는 방향과 연결된다.

다만 현재 문제는 outcome 다양성 자체가 아니라, 같은 exact state가 수백 번 반복되고 그 target이
sharp하게 갈리는 것이다. 이 경우 raw SGD가 이론적으로 평균으로 갈 수는 있어도 gradient variance와
초반 state weight가 너무 커진다.

### 반론 2: 초반 state가 너무 많이 중복되는 것이 아니냐

맞다. 현재 데이터는 이 반론을 강하게 지지한다.

- `pieces 0-2`, `pieces 2-4` 구간이 합쳐서 전체의 절반 이상이다.
- largest duplicate group은 `pieces=0.0`인 시작 state다.
- 시작 state group 안에서 value는 거의 50:50이고 policy argmax는 10개로 갈린다.

따라서 문제를 단순히 `temperature`만으로 해결하기 어렵다.

## Aggregate에 대한 위치

Aggregate는 효과가 있었다.

- 같은 feature를 하나로 묶어 policy/value target을 평균내면 label conflict가 줄어든다.
- smoke에서 학습 곡선이 크게 개선됐다.

하지만 aggregate를 최종 해법으로 쓰는 것은 단점이 있다.

- 방문 빈도 정보를 버린다.
- 자주 등장한 state와 한 번 등장한 state가 비슷한 weight를 갖게 된다.
- opening 같은 중요한 state를 과소학습할 수 있다.
- 원인인 target stochasticity와 opening over-representation을 직접 고치지는 않는다.
- loss 개선이 곧 arena strength 개선을 보장하지 않는다.

따라서 aggregate는 좋은 diagnostic intervention이지만, 정식 방향은 더 근본적인 생성/샘플링
수정이 낫다.

## 추천 방향

### 1. `policy_target_temperature=4.0` 실험

현재 진행 중인 실험이다. 목표는 replay diagnostics 기준:

```text
policy.entropy.p50 >= 0.3
policy.max_probability.p50 <= 0.9
legal.rows_with_illegal_target_mass = 0
```

판정:

- 목표에 가까워지면 target sharpness 문제는 완화된 것으로 본다.
- 그래도 max probability p50이 `0.95+`면 temperature만으로 부족하다.

### 2. Policy target 전용 scale 분리

Temperature만 올리는 것은 softmax 후처리에 가깝다. 더 직접적인 해결은 search action selection용
scale과 replay target용 scale을 분리하는 것이다.

예:

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
- 같은 state에서 policy target argmax가 sharp하게 갈리는 현상을 줄인다.

### 3. 초반 replay 저장 제한

Opening duplicate를 줄이는 가장 직접적인 방법이다.

후보:

```json
{
  "min_replay_turn": 2
}
```

또는 더 강한 실험:

```json
{
  "min_replay_turn": 4
}
```

주의:

- 너무 크게 잡으면 opening 학습이 약해질 수 있다.
- 처음에는 `2`부터 추천한다.

이 방법은 aggregate처럼 target을 후처리로 평균내지 않고, replay distribution이 시작 state에
과도하게 지배되는 문제를 직접 낮춘다.

### 4. Target 생성 안정화

Selected action exploration과 replay target 생성을 분리한다.

후보:

- selected action: Gumbel noise와 sampling 유지
- replay target: deterministic 또는 low-noise improved policy 사용
- target seed를 state hash 기반으로 고정해 같은 state에서 target variance를 줄임

이 방향은 구현 난이도는 더 높지만, 같은 state의 target이 매번 흔들리는 문제를 가장 직접적으로
다룬다.

## 현재 비추천 또는 후순위

### 초반 수 exploration 강제

지금 증상에는 1순위가 아니다.

이 기능은 trajectory 다양성이 부족할 때 유효하지만, 현재는 시작/초반 state의 target이 이미 여러
argmax로 갈리고 있다. Exploration을 더 강제하면 policy target variance가 커질 수 있다.

도입한다면 다음 원칙을 지킨다.

- action selection에만 적용한다.
- replay policy target을 더 랜덤하게 만들지 않는다.
- target 안정화와 opening replay 제한 이후에도 trajectory 다양성이 부족할 때 검토한다.

## 다음 실험 판정표

1. `T=4.0` replay diagnostics
   - sharpness가 충분히 내려가면 `min_replay_turn=2`로 진행한다.
   - 여전히 hard하면 `policy_target_c_visit/c_scale` 분리를 우선한다.

2. `min_replay_turn=2`
   - duplicate rows와 largest duplicate group이 얼마나 줄었는지 본다.
   - policy KL과 arena 결과를 같이 본다.

3. `policy_target_c_visit/c_scale` 분리
   - T만 올렸을 때보다 target entropy와 학습 안정성이 나은지 비교한다.

4. 최종 판단
   - aggregate는 baseline/diagnostic으로 남긴다.
   - 정식 pipeline 기본값은 `target scale 분리 + min_replay_turn` 조합을 우선 검토한다.
