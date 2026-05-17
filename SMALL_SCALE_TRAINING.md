# Small-Scale Gumbel MuZero Training Structure

이 문서는 Great Kingdom AI를 단일 Runpod RTX 3090급 환경에서 안정적으로 학습시키기 위한
학습 구조를 정리한다. 목표는 대규모 actor pool과 거대한 replay buffer를 흉내 내는 것이
아니라, 제한된 self-play 처리량에서도 meta cycling을 줄이고 replay를 효율적으로 재사용하는
것이다.

## 전제

- 학습 환경은 단일 RTX 3090 24GB와 CPU actor 여러 개를 기준으로 한다.
- self-play actor 수가 많지 않기 때문에 replay buffer 최신화 속도는 제한적이다.
- latest-vs-latest self-play만 사용하면 중반 이후 특정 메타가 순환하는 문제가 생길 수 있다.
- 큰 replay buffer는 다양성을 주지만 오래된 policy 분포가 과하게 섞여 현재 학습을 흐릴 수 있다.
- recent-only replay는 빠르게 적응하지만 과거 메타에 대한 기억을 잃어 cycling을 악화시킬 수 있다.

## 전체 구조

```text
CPU self-play actors
  -> batched GPU/ONNX inference
  -> trajectory shards
  -> learner import
  -> replay: recent + reservoir + hard cases
  -> learner update
  -> latest checkpoint / ONNX export
  -> fp16 weight-only opponent snapshots
```

운영의 핵심은 다음 네 가지다.

- replay는 최신성만 보지 않고 `recent`, `reservoir`, `hard_cases`로 나눈다.
- self-play 상대는 live latest만 쓰지 않고 최근 snapshot과 오래된 anchor를 섞는다.
- MCTS simulation budget은 학습 단계에 따라 점진적으로 올린다.
- reanalyse는 전체 buffer가 아니라 일부 batch와 hard sample 중심으로만 수행한다.

## Opponent Pool

`best` snapshot은 초기 구조에서 제외한다. 별도 arena gate가 필요하고 평가 노이즈와 운영
복잡도가 생기기 때문이다. 대신 recent ring과 old anchor를 사용해 과거 메타를 계속 노출한다.

```yaml
opponents:
  latest:
    ratio: 0.60
    source: live_current_model

  recent_ring:
    ratio: 0.30
    size: 6
    save_every_learner_steps: 10000
    format: fp16_weight_only
    include_optimizer: false

  old_anchor:
    ratio: 0.10
    size: 2
    format: fp16_weight_only
    include_optimizer: false
```

recent ring은 항상 최근 몇 세대의 policy를 보존한다. old anchor는 시간 기준으로 고정한다.
예를 들어 초반 안정화가 끝난 시점과 중반 첫 plateau 지점의 snapshot을 anchor로 남긴다.
이 방식은 arena 없이도 latest policy가 과거 메타를 완전히 잊는 것을 완화한다.

opponent snapshot은 학습 재개용 checkpoint와 분리한다.

```text
training checkpoint:
  model + optimizer + scheduler + training state

opponent snapshot:
  model weights only
  fp16
  no optimizer
  no replay
```

## Replay Structure

replay는 하나의 큰 FIFO로만 운용하지 않는다. batch sampling 단계에서 세 종류의 source를
명시적으로 섞는다.

```yaml
replay_sampling:
  recent:
    ratio: 0.55
    role: current_meta_adaptation

  reservoir:
    ratio: 0.25
    role: historical_memory

  hard_cases:
    ratio: 0.20
    role: anti_cycling_and_error_correction
```

각 영역의 역할은 다르다.

- `recent`: 최신 self-play 분포를 따라가기 위한 working set.
- `reservoir`: 오래된 trajectory를 얇게 유지하는 history anchor.
- `hard_cases`: 진 게임, value error가 큰 상태, reanalyse 후 policy target이 크게 바뀐 상태,
  희귀 opening 또는 희귀 action branch.

권장 시작값:

```yaml
replay_capacity:
  recent_games: 3000-8000
  reservoir_games: 10000-30000
  hard_case_games: 3000-8000
```

transition 단위로 관리하는 경우에는 게임 길이에 맞춰 위 값을 환산한다. 중요한 점은 전체
capacity보다 sampling ratio다. 작은 환경에서는 replay를 무작정 키우는 것보다 어떤 분포를
얼마나 뽑는지가 더 중요하다.

## Hard Case Policy

hard case는 단순히 loss가 큰 샘플만 의미하지 않는다. meta cycling을 줄이기 위해 다음 기준을
조합한다.

```yaml
hard_case_criteria:
  - lost_game
  - high_value_error
  - high_policy_kl_after_reanalyse
  - rare_opening_or_state
  - repeatedly_bad_against_old_anchor
```

가능하면 trajectory 또는 state에 아래 metadata를 남긴다.

```yaml
sample_metadata:
  - model_version
  - created_at_learner_step
  - opponent_version
  - game_result
  - root_policy_or_logits
  - root_value
  - legal_actions
```

과거 policy 자체를 opponent로 많이 저장하지 못하더라도, `root_policy_or_logits`를 저장하면
학습 중 과거 target을 일부 유지할 수 있다.

## Batch-Local PER

학습 시간을 거의 늘리지 않는 priority sampling은 batch-local PER 형태로 구현한다. replay 전체를
현재 model로 다시 평가해서 priority를 갱신하는 방식은 추가 forward 비용이 크므로 기본 구조에서
제외한다. MCTS/reanalyse 기반 priority도 search 비용이 크므로 중반 이후 필요할 때만 검토한다.

PER에서 세 종류의 weight는 분리한다.

```text
sampling_priorities[i]:
  sample i를 얼마나 자주 뽑을지 결정하는 값

sample_loss_weights[i]:
  replay가 원래 가진 loss 가중치
  기본값은 1

importance_weights[i]:
  priority sampling이 만든 bias를 보정하는 IS weight
```

최종 loss row weight는 아래처럼 계산한다.

```text
final_weight_i = sample_loss_weights_i * importance_weights_i
```

`sample_loss_weights`를 priority로도 쓰고 loss weight로도 쓰지 않는다. 두 역할을 섞으면 priority가
높은 샘플이 더 자주 뽑히면서 loss에서도 더 세게 반영되어 이중 가중이 된다.

샘플링 확률은 일반적인 PER 형태를 따른다.

```text
P(i) = sampling_priorities[i]^alpha / sum_j sampling_priorities[j]^alpha

importance_weight_i = (N * P(i))^-beta
importance_weight_i = importance_weight_i / max(importance_weight)
```

batch-local PER의 update는 train step에서 이미 계산한 per-sample loss를 사용한다. 별도 model
forward를 추가하지 않는다.

```text
1. priority 기반으로 replay index batch를 뽑는다.
2. model forward로 policy logits와 value를 계산한다.
3. reduction 전에 per-sample policy KL/value error를 계산한다.
4. weighted loss로 backward/update를 수행한다.
5. 방금 학습한 replay indexes의 sampling priority만 갱신한다.
```

권장 priority score:

```yaml
batch_local_per:
  enabled: true
  alpha: 0.5-0.6
  beta: 0.2-0.4
  epsilon: 0.001
  max_priority: 8-16
  priority_ema: 0.9
  score:
    policy_kl_weight: 1.0
    value_abs_error_weight: 0.5
```

score 계산:

```text
policy_kl_i = cross_entropy(target_policy_i, current_policy_i) - entropy(target_policy_i)
value_abs_error_i = abs(predicted_value_i - target_value_i)

new_priority_i =
  epsilon
  + policy_kl_weight * policy_kl_i
  + value_abs_error_weight * value_abs_error_i

sampling_priorities[i] =
  priority_ema * old_priority_i
  + (1 - priority_ema) * clamp(new_priority_i, epsilon, max_priority)
```

새로 import된 transition은 현재 최대 priority로 초기화한다.

```text
new_sample_priority = current_max_priority
```

이렇게 하면 새 self-play 데이터가 replay에 들어온 뒤 한 번도 학습되지 못하고 묻히는 문제를 줄일 수
있다. 한 번 학습된 뒤에는 실제 per-sample error에 따라 priority가 조정된다.

## Reanalyse

전체 buffer reanalyse는 단일 GPU 환경에서 비싸다. 따라서 batch 일부만 최신 network로 다시
search한다.

```yaml
reanalyze:
  enabled: true
  batch_ratio: 0.25
  prefer:
    - old_samples
    - hard_cases
    - high_priority_samples
  min_sample_age_learner_steps: 5000
```

reanalyse 대상은 최신 sample보다 오래된 sample과 hard case를 우선한다. 목적은 모든 target을
항상 최신화하는 것이 아니라, 오래된 데이터가 현재 policy 기준에서 완전히 잘못된 방향으로
학습을 끌고 가는 것을 막는 것이다.

## MCTS Budget

Gumbel MuZero는 적은 simulation에서도 policy improvement를 얻기 위한 구조이므로, 초반부터 큰
simulation budget을 쓰지 않는다.

```yaml
mcts:
  train_simulations_schedule:
    - learner_steps: 0
      simulations: 16
    - learner_steps: 50000
      simulations: 32
    - learner_steps: 200000
      simulations: 64

  eval_simulations: 64-128
```

학습 초반에는 많은 search보다 다양한 self-play와 빠른 learner update가 더 중요하다. simulation
budget은 policy가 어느 정도 안정된 뒤 올린다.

## Losses

현재 Great Kingdom AI 구조는 Rust 규칙 엔진이 transition을 담당하고, neural network는 search의
leaf/root 평가에 쓰이는 policy-value model에 가깝다. 따라서 EfficientZero의 consistency loss는
현재 구조에 바로 적용하지 않는다.

EfficientZero식 consistency loss는 learned dynamics가 있을 때 의미가 있다. 일반적인 형태는
`representation(observation_t)`로 만든 latent state에 action을 넣어
`dynamics(latent_t, action_t) -> predicted_latent_t+1`을 만들고, 이것이 실제 다음 observation의
representation과 일치하도록 맞추는 self-supervised loss다. 현재 구조처럼 규칙 엔진이 다음
상태를 직접 만들고 network가 dynamics를 학습하지 않는다면 맞출 latent transition이 없다.

현재 구조의 기본 loss는 policy/value 중심으로 둔다.

```yaml
loss:
  policy: 1.0
  value: 1.0
```

`policy_kl_to_previous`는 필수 요소는 아니며, 중반 이후 policy가 특정 메타로 급격히 이동하면서
cycling이 커질 때만 선택적으로 검토한다.

```yaml
optional_loss:
  policy_kl_to_previous: 0.005-0.02
```

아래 조건을 만족하는 learned dynamics 기반 MuZero로 구조를 바꿀 때만 consistency loss를 다시
검토한다.

```yaml
consistency_loss_prerequisites:
  - representation_network
  - learned_dynamics_network
  - projected_latent_prediction
  - target_next_observation_representation
```

## Optimizer

기본 optimizer는 AdamW보다 SGD momentum을 우선 검토한다. AdamW는 초반 수렴이 빠르고 튜닝이
편하지만, 작은 self-play 환경에서는 최근 target과 특정 메타를 민감하게 따라가면서 policy
collapse나 meta cycling을 키울 수 있다. SGD momentum은 느리지만 업데이트 성향이 단순하고,
보드게임 AlphaZero/MuZero 계열 구현에서 널리 쓰이는 안정적인 선택이다.

권장 시작값:

```yaml
optimizer:
  type: sgd
  lr: 0.02-0.10
  momentum: 0.9
  weight_decay: 1e-4
  nesterov: false

scheduler:
  type: cosine_or_step_decay
  warmup_steps: 1000-5000

gradient:
  clip_norm: 5.0
```

batch size에 따라 learning rate를 다르게 시작한다.

```yaml
sgd_lr_by_batch_size:
  batch_128_256: 0.01-0.05
  batch_512_1024: 0.05-0.10
```

weight decay는 모든 parameter에 일괄 적용하지 않는다. 일반적으로 convolution/linear weight에는
decay를 적용하고 bias와 normalization parameter에는 적용하지 않는다.

```yaml
weight_decay_groups:
  decay:
    - convolution_weights
    - linear_weights
  no_decay:
    - bias
    - batchnorm_parameters
    - layernorm_parameters
```

AdamW checkpoint에서 SGD로 전환할 때는 optimizer state를 이어받지 않는다. model weights만
로드하고 SGD optimizer와 scheduler state는 새로 만든다.

```text
AdamW -> SGD migration:
  load model weights
  discard AdamW optimizer state
  create new SGD optimizer
  create new scheduler
  keep replay and training metadata if compatible
```

SGD 전환 후에는 loss 하락 속도보다 안정성 지표를 우선 확인한다.

```text
policy entropy does not collapse too early
latest vs old anchors win rate is not oscillating heavily
value loss decreases slowly but consistently
root action distribution remains diverse enough
```

## Training Ratio

actor throughput이 낮은 환경에서는 같은 replay를 너무 오래 재사용하기 쉽다. learner가 actor를
심하게 앞지르면 현재 self-play 분포와 학습 분포가 어긋난다. 따라서 train-to-self-play ratio를
상한으로 제한한다.

```yaml
training:
  actors: 8-12
  batch_size: 128-1024
  unroll_steps: 5
  amp: true
  train_reuse_factor: capped
```

batch size는 현재 코드 경로와 GPU utilization에 맞춘다. 큰 batch가 항상 좋은 것은 아니며,
actor가 만든 새 trajectory가 충분히 들어오지 않는다면 learner step 수를 늘리는 대신 self-play
cycle을 더 자주 돌리는 편이 낫다.

## Monitoring

loss만 보고 판단하지 않는다. self-play류에서는 loss가 좋아져도 실제 policy가 약해질 수 있다.
다음 지표를 지속적으로 본다.

```text
latest vs recent snapshots win rate
latest vs old anchors win rate
policy entropy
root action distribution collapse
value error by opponent version
replay sample age distribution
hard case sampling rate
```

cycling이 의심될 때의 조정 순서는 다음과 같다.

```yaml
if_meta_cycling:
  recent_ratio: decrease
  reservoir_ratio: increase
  old_anchor_opponent_ratio: increase
  learning_rate: slightly_decrease
  policy_target_temperature: soften
  policy_kl_to_previous: increase_slightly
```

반대로 학습이 너무 느리고 최신 policy에 적응하지 못하면 `recent` 비율을 올리고 old anchor
비율을 낮춘다.

## Recommended Initial Config

최초 실험은 아래 형태로 시작한다.

```yaml
opponents:
  latest_ratio: 0.60
  recent_ring_ratio: 0.30
  old_anchor_ratio: 0.10
  recent_ring_size: 6
  old_anchor_size: 2
  snapshot_every_learner_steps: 10000
  snapshot_dtype: fp16
  snapshot_weight_only: true

replay:
  recent_ratio: 0.55
  reservoir_ratio: 0.25
  hard_case_ratio: 0.20
  recent_games: 3000-8000
  reservoir_games: 10000-30000
  hard_case_games: 3000-8000
  store_policy_logits: true

batch_local_per:
  enabled: true
  alpha: 0.5-0.6
  beta: 0.2-0.4
  epsilon: 0.001
  max_priority: 8-16
  priority_ema: 0.9
  policy_kl_weight: 1.0
  value_abs_error_weight: 0.5

reanalyze:
  enabled: true
  batch_ratio: 0.25
  min_sample_age_learner_steps: 5000

mcts:
  train_simulations:
    early: 16
    mid: 32
    late: 64
  eval_simulations: 64-128

training:
  actors: 8-12
  unroll_steps: 5
  amp: true
  train_reuse_factor: capped

optimizer:
  type: sgd
  lr: 0.02-0.10
  momentum: 0.9
  weight_decay: 1e-4
  warmup_steps: 1000-5000
  clip_norm: 5.0

loss:
  policy_weight: 1.0
  value_weight: 1.0
  policy_kl_to_previous_weight: optional_0.005-0.02
```

## Design Summary

단일 3090 환경에서는 거대한 replay와 많은 actor로 policy 분포를 빠르게 갱신하는 대규모 구조를
그대로 따라가기 어렵다. 대신 이 구조는 작은 최신 working set으로 빠르게 적응하고, reservoir와
old anchor opponent로 과거 메타를 보존하며, hard case와 partial reanalyse로 취약한 상태를
집중적으로 보정한다.

핵심 원칙은 다음과 같다.

- latest-vs-latest만 돌리지 않는다.
- replay를 recent-only로 만들지 않는다.
- priority sampling은 replay-wide refresh가 아니라 batch-local PER로 시작한다.
- best snapshot과 arena gate는 초기 구조에서 제외한다.
- fp16 weight-only snapshot으로 저장공간 부담을 낮춘다.
- 전체 reanalyse보다 partial reanalyse를 우선한다.
- optimizer는 AdamW보다 SGD momentum을 우선 실험한다.
- learned dynamics가 없으므로 EfficientZero식 consistency loss는 현재 구조에 넣지 않는다.
- 학습 안정성 판단은 loss가 아니라 snapshot 상대 승률과 action distribution으로 한다.
