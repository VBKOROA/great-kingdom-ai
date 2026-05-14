# Training Collapse 가설 정리

## 목적

원본 replay/data를 지우고 새 학습을 시작하기 전에, 현재까지 관찰한 collapse 현상의 가능한 원인을
분리해서 정리한다.

이 문서는 특정 해법을 확정하기 위한 문서가 아니라, 새 run에서 어떤 로그와 replay metadata를 반드시
남겨야 하는지 정하는 기준이다.

## 현재 큰 그림

가장 가능성이 높은 설명은 다음이다.

```text
초반에는 64 simulation Gumbel search가 모델보다 강해서 loss와 strength가 같이 개선된다.
중반 이후에는 모델이 search teacher를 따라잡고,
sharp policy target, actor/learner lag, tactical cliff가 결합하면서
replay objective를 더 잘 맞출수록 실제 strength가 흔들린다.
```

즉 문제는 "학습이 안 된다"가 아니라, 어느 시점부터는 "잘못되었거나 너무 확신적인 target을 너무 잘
학습한다" 쪽에 가깝다.

## 게임 룰이 주는 위험

`docs/rule-spec.md` 기준으로 Great Kingdom은 Go 계열이지만 일반 Go와 다르다.

- 상대 그룹 capture가 즉시 승리다.
- 상대 territory에는 착수할 수 없다.
- 두 pass 후 territory scoring을 하며, Blue는 3점 이상이어야 이긴다.

이 때문에 value와 policy target이 한 수 차이로 급변하는 tactical cliff가 많을 수 있다. 얕은 search가
전술 반박을 놓치면 틀린 수가 매우 높은 확률의 target으로 저장될 수 있다.

## 관찰된 사실

### Strength regression

나중 checkpoint가 replay objective는 더 잘 맞췄지만 arena strength는 떨어졌다.

```text
110023 vs 030431: 110023 win rate 42%
current latest vs 030431: latest win rate 35%
```

이는 loss 감소가 strength 개선을 보장하지 않는다는 신호다.

### Raw/EMA 비일관성

raw와 EMA의 우열이 checkpoint마다 달랐다.

```text
raw110 vs ema110: raw110 42.5%, ema110 57.5%
raw030 vs ema030: raw030 58.0%, ema030 42.0%
```

EMA는 안정화 요소일 수 있지만, 현재 collapse의 단일 원인으로 보기는 어렵다.

### Replay target sharpness

trajectory replay diagnostics에서 policy target이 매우 sharp했다.

```text
policy_target entropy mean: 0.2054
policy_target max_probability mean: 0.9233
policy_target max_probability median: 0.999656
policy_target support median: 3
```

이 값은 모델 prior가 0.999라는 뜻이 아니다. replay에 저장된 search policy target이 거의 hard label에
가깝다는 뜻이다.

### Root prior 진단 불가

현재 replay에는 `root_policy_logits`가 없었다.

```text
missing_reason: trajectory replay does not include root_policy_logits
```

따라서 search target이 root prior보다 실제로 개선되었는지, 아니면 prior를 거의 복사하고 있는지
분해할 수 없었다.

### 128 simulation 실험

`128-4cycles.pt`는 110023에는 이겼지만 030431에는 졌다.

```text
128-4cycles vs 110023: 57%
128-4cycles vs 030431: 44%
```

search quality를 올리는 방향은 효과가 있을 수 있으나, 이것만으로 최종 해결이라고 보기는 어렵다.

## 주요 가설

### 1. Policy target이 너무 sharp하다

현재 가장 강한 가설이다.

중반 이후 search가 틀린 수를 고르면 그 수가 거의 one-hot target으로 저장된다. learner는 이 target을
잘 맞추며 policy loss를 낮추지만, 실제 게임 strength는 나빠질 수 있다.

확인할 것:

- policy target entropy
- policy target max probability
- target top-1 action frequency
- 같은 exact state에서 policy argmax가 얼마나 충돌하는지

조치 후보:

- `policy_target_temperature`를 `1.25`, `1.5`, `2.0`으로 sweep한다.
- `policy_target_c_visit`, `policy_target_c_scale`을 action selection용 Gumbel scale과 분리한다.

### 2. 64 simulation teacher가 중반 이후 improvement operator로 약하다

64 simulation이 초반 학습에 부족했다는 뜻은 아니다. loss가 `4.x -> 2.x`로 안정적으로 내려갔다면 초반
teacher는 충분히 유효했다.

문제는 모델이 teacher를 따라잡은 뒤다. 이때 search가 모델 prior보다 충분히 좋은 target을 만들지
못하면, 학습은 improvement가 아니라 자기복제와 과확신으로 바뀔 수 있다.

확인할 것:

- `KL(policy_target || root_prior)`
- `target_argmax != prior_argmax` 비율
- target top-1 probability와 prior top-1 probability 차이
- simulation 수별 fixed-position policy/value 변화

조치 후보:

- full sample은 64 baseline과 128을 비교한다.
- 128이 의미 있으면 256도 소규모로 확인한다.
- fast simulation은 현재 학습 샘플에 들어가지 않으므로 이 가설의 핵심 변수가 아니다.

### 3. Replay objective와 실제 strength가 어긋난다

`030431 -> 110023` 진단에서 replay fitting metric은 좋아졌지만 arena는 나빠졌다. 이는 replay target을 더
잘 맞추는 방향이 실제 strength 개선 방향이 아닐 수 있음을 뜻한다.

확인할 것:

- checkpoint별 replay KL/value MSE
- checkpoint별 arena 결과
- checkpoint별 fixed-position eval 결과

조치 후보:

- loss만 보지 않는다.
- learner cycle마다 고정 포지션 probe를 저장한다.
- arena는 promotion gate가 아니라 sanity check로 사용한다.

### 4. Actor/Learner staleness가 방향을 흐린다

actor가 오래된 ONNX로 self-play를 만들고 learner가 최신 모델로 학습하면, replay target과 현재 모델의
시점이 어긋난다.

확인할 것:

- transition의 `model_version`
- transition의 `created_iteration`
- batch sampling 시 model age histogram

조치 후보:

- replay transition에 model version metadata를 반드시 저장한다.
- 오래된 sample 비중을 진단한다.
- `recent_sample_fraction`은 `0.10 ~ 0.20`부터 시작한다.

### 5. Reuse factor가 noisy target을 과학습시킨다

기존 설정의 reuse factor가 높으면 같은 replay target을 여러 번 학습한다. target이 noisy하거나 sharp하면
loss는 잘 내려가지만 policy가 brittle해질 수 있다.

확인할 것:

- replay row당 평균 update 횟수
- effective update pressure
- recent row overweight

조치 후보:

- 새 run은 `train_reuse_factor=4.0`부터 시작한다.
- 안정성이 확인되기 전에는 reuse를 올리지 않는다.

### 6. Learning rate 변경이 resume에서 적용되지 않았다

기존 코드에서는 resume mode가 optimizer와 scheduler state를 복구한다. 따라서 config의 learning rate를
바꿔도 실제 optimizer LR이 바뀌지 않을 수 있다.

확인할 것:

- learner 시작 시 실제 optimizer LR 로그
- checkpoint mode가 `resume`인지 `bootstrap`인지

조치 후보:

- 새 run은 fresh 또는 weight-only bootstrap으로 시작한다.
- optimizer/scheduler는 새로 만든다.
- 시작 LR은 보수적으로 `5e-5`를 사용한다.

### 7. EMA가 시점에 따라 다르게 작동한다

EMA는 일반적으로 평가 안정화에 도움이 될 수 있지만, 현재 결과에서는 checkpoint마다 raw/EMA 우열이
뒤집혔다.

확인할 것:

- raw checkpoint와 EMA checkpoint를 항상 분리 평가한다.
- PyTorch raw, PyTorch EMA, ONNX export 결과를 같은 포지션에서 비교한다.

조치 후보:

- EMA decay는 우선 `0.995`를 유지한다.
- EMA를 원인으로 단정하지 않는다.
- 배포/eval 시 raw와 EMA를 같은 이름으로 섞지 않는다.

### 8. Color bias와 rule asymmetry가 크다

Blue는 3점 이상이어야 승리한다. 또한 착수 제한, territory 판정, capture 즉시승리 때문에 색별 성능이
강하게 갈릴 수 있다.

확인할 것:

- candidate blue win rate
- candidate orange win rate
- paired seed 또는 opening-swap 결과

조치 후보:

- 전체 승률만 보지 않는다.
- color-normalized metric을 별도로 본다.
- fixed opening set을 만든다.

### 9. Value target variance가 크다

terminal outcome `z`만 value target으로 쓰면, 같은 state에도 서로 다른 terminal result가 붙는다. 특히
초반 state는 실제 기대값이 0 근처여도 label은 항상 `-1` 또는 `+1`이다.

확인할 것:

- exact-state duplicate group
- 같은 state의 value target 분산
- game ply별 value loss

조치 후보:

- 같은 exact state의 aggregate replay를 별도 ablation으로 비교한다.
- completed-Q value blending은 바로 기본 경로에 넣지 않고 후순위 실험으로 둔다.

### 10. ONNX/fp16/export 경로 차이

self-play actor는 ONNX를 사용하고 learner는 PyTorch checkpoint를 사용한다. 이 둘의 출력이 다르면 replay
자체가 기대와 다르게 생성된다.

확인할 것:

- 같은 position의 PyTorch raw, PyTorch EMA, ONNX output KL
- legal mask 적용 후 policy 비교
- value output 차이

조치 후보:

- 새 run 초기에 export parity test를 넣는다.
- fp16 ONNX와 fp32 PyTorch 차이를 수치로 저장한다.

### 11. Replay metadata가 부족하다

이번 진단에서 가장 큰 병목은 `root_policy_logits` 부재였다. target이 sharp한 것은 확인했지만, target이
prior보다 개선되었는지 분해할 수 없었다.

새 replay에는 최소한 다음을 저장한다.

```text
root_policy_logits
root_improved_policy
root_value
model_version
created_iteration
num_simulations
max_considered_actions
use_full_sample
game_ply
value_target
legal_actions_count
side_to_move
```

### 12. 실제 룰/관점 버그

항상 열어둬야 하는 가설이다.

후보:

- legal mask 오류
- territory 판정 오류
- pass/end scoring 오류
- color perspective 뒤집힘
- value target sign 반전
- replay에서 player perspective 변환 오류
- ONNX input plane 순서 차이

확인할 것:

- hand-authored tiny positions
- Python/Rust rule parity
- value sign invariant
- legal move count parity
- pass/end scoring regression

## 새 run 전 필수 조치

1. `root_policy_logits`와 model metadata 저장을 복구한다.
2. learner 시작 로그에 실제 optimizer LR을 출력한다.
3. weight-only bootstrap 또는 fresh start로 optimizer/scheduler 상태를 분리한다.
4. raw/EMA/ONNX를 같은 포지션에서 비교하는 parity check를 추가한다.
5. fixed-position eval suite를 만든다.
6. replay diagnostics를 매 cycle 또는 일정 주기로 저장한다.

## 권장 초기 설정

보수적인 시작점:

```text
learning_rate: 5e-5
train_reuse_factor: 4.0
recent_sample_fraction: 0.10 ~ 0.20
ema_decay: 0.995
full_simulations: 64 baseline, 128 comparison
policy_target_temperature: 1.25 first sweep
```

이 설정은 최강 성능을 바로 노리는 값이 아니라, collapse 원인을 분해하기 위한 안정적인 기준선이다.

## 우선순위

가장 먼저 확인할 순서는 다음이다.

```text
1. target sharpness가 완화되었는가?
2. search target이 root prior보다 실제로 개선되었는가?
3. replay fitting 개선이 fixed-position eval 개선으로 이어지는가?
4. arena 결과가 color split 없이도 일관적인가?
5. raw/EMA/ONNX 사이에 의미 있는 출력 차이가 없는가?
```

현재 가장 유력한 원인은 다음 조합이다.

```text
sharp search target
+ 중반 이후 약해지는 64-sim improvement margin
+ replay reuse/recent overweight
+ actor/learner staleness
+ capture instant win rule의 tactical cliff
```

