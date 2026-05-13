# EfficientZero/LightZero-style Hyperparameter Guide

이 문서는 `TRAINING_EFFICIENCY_ROADMAP.md`의 v2 학습 파이프라인을 구현한 뒤,
Runpod RTX 3090 24GB 환경에서 어떤 하이퍼파라미터를 어떻게 잡을지 정리한다.

방향성은 EfficientZero/LightZero를 참고하되, Great Kingdom AI에 맞게 조정한다.

- learned dynamics는 게임 룰 대체용으로 쓰지 않는다.
- Rust rule engine을 정확한 dynamics로 유지한다.
- EfficientZero/LightZero의 핵심 중 `trajectory replay`, `td_steps`, `reanalyze`,
  `priority replay`, `collector/learner 분리`, `self-supervised auxiliary loss`만 가져온다.

참고한 외부 기준:

- EfficientZero 논문은 MuZero 기반에서 self-supervised consistency, value prefix,
  off-policy correction을 핵심 개선으로 제안한다.
  https://proceedings.neurips.cc/paper/2021/hash/d5eca8dc3820cad9fe56a3bafda65ca1-Abstract.html
- LightZero 문서는 설정에서 `num_simulations`, `update_per_collect`, `batch_size`,
  `reanalyze_ratio`, `td_steps`, `num_unroll_steps`, `ssl_loss_weight` 등을 주요 튜닝
  파라미터로 둔다.
  https://opendilab.github.io/LightZero/tutorials/config/config.html
- LightZero EfficientZero 기본 정책은 `batch_size=256`, `num_simulations=50`,
  `td_steps=5`, `num_unroll_steps=5`, `value_loss_weight=0.25`, `policy_loss_weight=1`,
  `ssl_loss_weight=2` 같은 값을 기본값으로 둔다. 단, Atari/latent dynamics 기준이므로
  이 프로젝트에 그대로 복사하지 않는다.
  https://github.com/opendilab/LightZero/blob/main/lzero/policy/efficientzero.py

## 현재 설정 기준점

과거 v1 Runpod 설정은 다음 성격이었다. 현재 실행 경로는 v2 설정을 기준으로 한다.

- `iterations`: 50
- `self_play_games`: 2000
- `replay_capacity`: 150000
- `onnx_max_batch_size`: 8192
- `rust_self_play_batch_size`: 2000
- `aggregate_replay`: true
- `aggregate_replay_weight_mode`: none
- `gumbel_simulations`: 64
- playout cap randomization enabled
- full search fraction: 0.25
- full simulations: 64
- fast simulations: 16
- full max considered actions: 16
- fast max considered actions: 4
- `leaf_batch_size`: 256

`configs/runpod/train.json`

- `batch_size`: 512
- `steps`: 192
- `learning_rate`: 0.0001
- `weight_decay`: 0.0001
- `value_loss_weight`: 0.5
- `policy_loss_weight`: 1.0
- `lr_schedule`: constant with warmup
- `lr_warmup_steps`: 16
- `model_preset`: medium_plus
- `amp`: true
- `recent_sample_fraction`: 0.5
- `recent_sample_window`: 30000

이 기준점은 "한 iteration에서 대량 self-play를 만들고, 비교적 적은 learner step을 수행하는"
형태다. v2에서는 replay 재사용과 reanalyze를 강화하므로, 같은 self-play 양에서 learner
update 수와 target refresh 비중을 더 체계적으로 조절해야 한다.

## 파라미터 이름 매핑

| LightZero 계열 이름 | v2 파이프라인 이름 | 의미 |
| --- | --- | --- |
| `collector_env_num` | `actor_processes` 또는 actor batch 병렬도 | 동시에 self-play를 생성하는 단위 |
| `num_simulations` | `gumbel_simulations` / full search sims | MCTS/Gumbel search 계산량 |
| `batch_size` | learner batch size | 한 gradient step의 sample 수 |
| `update_per_collect` | `learner_steps_per_actor_batch` | self-play batch 하나당 learner update 수 |
| `replay_ratio` | `updates_per_new_transition` | 새 transition 대비 재사용 강도 |
| `td_steps` | `bootstrap_td_steps` | value bootstrap에 사용할 미래 step 수 |
| `num_unroll_steps` | `consistency_unroll_steps` | auxiliary consistency를 몇 step 적용할지 |
| `reanalyze_ratio` | `reanalyze_fraction` | 학습 batch 또는 target 중 최신 모델로 갱신할 비율 |
| `reanalyze_batch_size` | `reanalyze_batch_size` | reanalyze inference batch 크기 |
| `ssl_loss_weight` | `consistency_loss_weight` | representation consistency auxiliary loss 가중치 |

## 권장 튜닝 순서

하이퍼파라미터를 한 번에 많이 바꾸면 원인을 알기 어렵다. 다음 순서로 조정한다.

1. `learner_steps_per_actor_batch`
2. `bootstrap_td_steps`
3. `reanalyze_fraction`
4. priority replay
5. search budget
6. consistency loss
7. model size

이 순서는 의도적이다. 샘플 효율을 높이려면 먼저 같은 replay를 더 잘 쓰게 만들고,
그 다음 search 비용과 모델 크기를 조정하는 편이 낫다.

## v2 Baseline Profile

처음 구현 후 안정성을 확인하기 위한 기준값이다.

### Actor/self-play

| 파라미터 | 권장값 |
| --- | --- |
| `actor_processes` | 1 |
| `games_per_actor_batch` | 512-1000 |
| `gumbel_simulations_full` | 64 |
| `gumbel_simulations_fast` | 16 |
| `playout_cap_full_search_fraction` | 0.25 |
| `max_considered_actions_full` | 16 |
| `max_considered_actions_fast` | 4 |
| `leaf_batch_size` | 256 |
| `onnx_max_batch_size` | 4096-8192 |
| `trajectory_shard_games` | 128-256 |

현재 search 설정은 유지해도 된다. v2 초반에는 search 품질보다 target/replay 구조 변경이
주요 변수이므로 search budget을 먼저 흔들지 않는다.

### Learner

| 파라미터 | 권장값 |
| --- | --- |
| `model_preset` | `medium_plus` |
| `batch_size` | 512 |
| `learning_rate` | `1e-4` |
| `weight_decay` | `1e-4` |
| `optimizer` | AdamW |
| `amp` | true |
| `grad_clip_norm` | 5-10 |
| `lr_schedule` | warmup cosine 또는 constant with warmup |
| `warmup_steps` | 32-128 |
| `learner_steps_per_actor_batch` | 256-512 |

현재 `steps=192`는 안정적이지만 replay 재사용 관점에서는 낮은 편이다. v2에서 target
snapshot과 priority sampling이 들어오면 actor batch당 256-512 step부터 시작한다.

### Target/reanalyze

| 파라미터 | 권장값 |
| --- | --- |
| `discount_factor` | 1.0 |
| `bootstrap_td_steps` | 4 |
| `reanalyze_fraction` | 0.25 |
| `reanalyze_batch_size` | 512-2048 |
| `max_target_age_iterations` | 4 |
| `target_snapshot_interval` | every iteration |
| `reanalyze_noise` | false |

Great Kingdom은 finite two-player board game이고 value가 승패 기준이라 `gamma=1.0`부터
시작한다. 시간이 지나 value target이 너무 출렁이면 `bootstrap_td_steps`를 줄이고,
terminal 근처 sample 비중을 늘린다.

### Replay sampling

| 파라미터 | 권장값 |
| --- | --- |
| `replay_capacity_transitions` | 300000-800000 |
| `min_replay_transitions_before_train` | 10000-30000 |
| `recent_sample_fraction` | 0.25-0.5 |
| `recent_sample_window` | 30000-80000 |
| `priority_enabled` | false initially |
| `priority_alpha` | 0.4-0.6 after enabled |
| `priority_beta` | 0.2-0.4 initially |
| `duplicate_aggregation` | optional, off for trajectory replay baseline |

trajectory replay 초기에는 duplicate aggregation을 끄는 쪽이 낫다. 중복 state 평균은
AlphaZero-style sample replay에서는 유효하지만, trajectory 기반 n-step/reanalyze에서는
시간 순서와 target age 정보가 더 중요하다.

## Conservative Profile

목표: OOM과 불안정성을 피하고, v2 구조가 제대로 작동하는지 확인한다.

| 그룹 | 파라미터 | 값 |
| --- | --- | --- |
| actor | `games_per_actor_batch` | 256 |
| actor | `gumbel_simulations_full` | 64 |
| actor | `gumbel_simulations_fast` | 16 |
| actor | `onnx_max_batch_size` | 4096 |
| learner | `batch_size` | 256 |
| learner | `learner_steps_per_actor_batch` | 128 |
| learner | `learning_rate` | `1e-4` |
| target | `bootstrap_td_steps` | 2 |
| target | `reanalyze_fraction` | 0.0-0.25 |
| replay | `recent_sample_fraction` | 0.5 |
| replay | priority | off |
| consistency | `consistency_loss_weight` | 0 |

사용 시점:

- trajectory replay를 처음 붙였을 때
- reanalyze target의 perspective 변환을 검증할 때
- local smoke와 Runpod smoke 사이 차이를 줄이고 싶을 때

## Balanced Profile

목표: v2의 기본 실험값. 가장 먼저 비교 기준으로 삼는다.

| 그룹 | 파라미터 | 값 |
| --- | --- | --- |
| actor | `games_per_actor_batch` | 512-1000 |
| actor | `gumbel_simulations_full` | 64 |
| actor | `gumbel_simulations_fast` | 16 |
| actor | `playout_cap_full_search_fraction` | 0.25 |
| actor | `onnx_max_batch_size` | 8192 |
| learner | `batch_size` | 512 |
| learner | `learner_steps_per_actor_batch` | 256-512 |
| learner | `learning_rate` | `1e-4` |
| target | `bootstrap_td_steps` | 4 |
| target | `reanalyze_fraction` | 0.25 |
| target | `reanalyze_batch_size` | 1024 |
| replay | `recent_sample_fraction` | 0.35 |
| replay | priority | off first, then on |
| consistency | `consistency_loss_weight` | 0 first |

Balanced Profile에서 priority와 consistency는 처음부터 켜지 않는다. 먼저 n-step/bootstrap과
reanalyze만으로 baseline 대비 이득이 있는지 확인한다.

## Aggressive Sample-Reuse Profile

목표: self-play 비용을 줄이고 같은 replay를 더 많이 재사용한다.

| 그룹 | 파라미터 | 값 |
| --- | --- | --- |
| actor | `games_per_actor_batch` | 512 |
| actor | `gumbel_simulations_full` | 64 |
| actor | `gumbel_simulations_fast` | 8-16 |
| actor | `playout_cap_full_search_fraction` | 0.15-0.25 |
| learner | `batch_size` | 512-768 |
| learner | `learner_steps_per_actor_batch` | 768-1500 |
| learner | `learning_rate` | `5e-5`-`1e-4` |
| target | `bootstrap_td_steps` | 2-4 |
| target | `reanalyze_fraction` | 0.5-1.0 |
| replay | priority | on |
| replay | `priority_alpha` | 0.6 |
| replay | `priority_beta` | 0.4 |
| consistency | `consistency_loss_weight` | 0-0.1 |

위험:

- off-policy 성향이 커진다.
- 오래된 policy target에 과적합할 수 있다.
- value bootstrap 오류가 누적될 수 있다.

대응:

- `max_target_age_iterations`를 2-3으로 줄인다.
- `reanalyze_fraction`을 높인다.
- learning rate를 낮춘다.
- arena가 흔들리면 `learner_steps_per_actor_batch`를 줄인다.

## Consistency Auxiliary Profile

목표: EfficientZero의 self-supervised consistency 아이디어를 rule-engine 환경에 맞게 작게
실험한다.

전제:

- `next_features` 또는 trajectory next lookup이 있어야 한다.
- 모델에 representation head 또는 intermediate feature를 꺼낼 수 있는 API가 있어야 한다.
- search transition은 계속 Rust rule engine을 사용한다.

권장값:

| 파라미터 | 값 |
| --- | --- |
| `consistency_unroll_steps` | 1 first, then 2-4 |
| `consistency_loss_weight` | 0.02-0.1 |
| `policy_loss_weight` | 1.0 |
| `value_loss_weight` | 0.25-0.5 |
| `stop_gradient_target` | true |
| `augmentation` | board symmetry only |

LightZero EfficientZero의 기본 `ssl_loss_weight=2`는 latent dynamics Atari 설정 기준이다.
이 프로젝트에서는 auxiliary loss가 main policy/value 학습을 압도하면 안 되므로 훨씬 작게
시작한다.

켜는 순서:

1. `consistency_loss_weight=0`으로 target/replay baseline을 확정한다.
2. `0.02`로 켠다.
3. policy KL과 arena가 악화되지 않으면 `0.05`, `0.1`까지 올린다.
4. value loss가 좋아져도 arena가 나빠지면 끈다.

## Search Budget Tuning

현재 search 설정은 꽤 합리적이다.

- full: 64 simulations, max considered 16
- fast: 16 simulations, max considered 4
- full fraction: 0.25

v2 이후 튜닝 기준:

### self-play가 병목이면

순서대로 낮춘다.

1. `playout_cap_full_search_fraction`: 0.25 -> 0.20 -> 0.15
2. `gumbel_simulations_fast`: 16 -> 8
3. `max_considered_actions_fast`: 4 유지
4. `self_play_games`를 줄이고 learner reuse를 늘린다.

full search simulations는 가능한 유지한다. target anchor 역할을 하기 때문이다.

### target 품질이 낮으면

순서대로 올린다.

1. `playout_cap_full_search_fraction`: 0.25 -> 0.35
2. `gumbel_simulations_full`: 64 -> 96
3. `max_considered_actions_full`: 16 -> 24
4. actor batch를 줄여 OOM과 wall-clock을 관리한다.

### learner가 병목이면

search를 더 줄이지 말고 먼저 learner throughput을 개선한다.

- pinned memory
- batch prefetch
- legal mask precompute
- AMP 유지
- batch size 512 유지, OOM 여유가 있으면 768 시도

## Reanalyze Tuning

reanalyze는 v2의 핵심이다. 단, 비용을 통제해야 한다.

### v1 network-only reanalyze

권장 시작값:

```text
reanalyze_fraction = 0.25
reanalyze_batch_size = 1024
max_target_age_iterations = 4
```

증가 조건:

- replay target age가 빠르게 늙는다.
- value loss는 낮은데 arena가 개선되지 않는다.
- 오래된 sample의 policy KL이 높다.

감소 조건:

- learner보다 reanalyze가 wall-clock 병목이다.
- target이 자주 바뀌어 loss가 심하게 흔들린다.

### v2 search reanalyze

권장 시작값:

```text
search_reanalyze_fraction = 0.05
search_reanalyze_simulations = 32-64
search_reanalyze_priority_only = true
```

search reanalyze는 모든 sample에 적용하지 않는다. priority가 높은 일부 state에만 적용한다.

## TD Steps / Bootstrap Tuning

보드게임 value target에서는 perspective 변환이 가장 중요하다.

권장 시작:

```text
discount_factor = 1.0
bootstrap_td_steps = 4
```

조정 기준:

- value loss가 낮아지지만 arena가 개선되지 않으면 `td_steps`를 2로 줄인다.
- value target이 너무 noisy하면 terminal-near sample 비중을 늘린다.
- 학습이 너무 느리면 `td_steps`를 8로 늘려본다.
- 오래된 trajectory에는 짧은 `td_steps`, 최신 trajectory에는 긴 `td_steps`를 쓴다.

age-aware 예시:

```text
target_age <= 1 iteration: td_steps = 8
target_age <= 3 iterations: td_steps = 4
target_age > 3 iterations: td_steps = 2 or reanalyze required
```

## Learner Step Ratio

현재는 iteration당 self-play 2000 games, learner 192 steps다. v2에서는 transition replay와
target snapshot을 쓰므로 learner step을 늘릴 여지가 있다.

권장 공식:

```text
new_transitions = actor_batch_games * avg_training_samples_per_game
updates_per_new_transition = learner_steps * batch_size / new_transitions
```

초기 목표:

```text
updates_per_new_transition = 0.5-2.0
```

예시:

```text
actor_batch_games = 1000
avg_training_samples_per_game = 25
new_transitions = 25000
batch_size = 512
learner_steps = 256
updates_per_new_transition = 256 * 512 / 25000 = 5.24
```

이 값은 꽤 높은 편이다. target이 신선하지 않으면 off-policy 문제가 생길 수 있다. 따라서
위 예시처럼 transition 수가 적게 저장되는 경우에는 learner step을 줄이거나 reanalyze 비중을
높인다.

권장 범위:

- target refresh 없음: 0.5-1.0
- network-only reanalyze 있음: 1.0-3.0
- priority + reanalyze 안정화 후: 2.0-6.0

## Replay Capacity

현재 `replay_capacity=150000`은 sample replay 기준으로는 적당하지만, trajectory replay와
reanalyze를 쓰면 조금 더 키우는 편이 낫다.

권장:

| 상황 | capacity |
| --- | --- |
| smoke | 20000-50000 |
| baseline v2 | 300000-800000 transitions |
| long run | 1000000-2000000 transitions |

주의:

- capacity를 키우면 오래된 target 비율도 늘어난다.
- capacity 증가는 reanalyze와 age-aware sampling이 있을 때 의미가 크다.
- 메모리에 모두 올리지 말고 shard/lazy loading을 전제로 한다.

## Loss Weights

현재:

```text
policy_loss_weight = 1.0
value_loss_weight = 0.5
```

권장:

| 단계 | policy | value | consistency |
| --- | --- | --- | --- |
| v2 baseline | 1.0 | 0.5 | 0 |
| bootstrap 안정화 후 | 1.0 | 0.25-0.5 | 0 |
| consistency 실험 | 1.0 | 0.25-0.5 | 0.02-0.1 |

LightZero EfficientZero는 value weight 0.25를 기본으로 둔다. 이 프로젝트는 terminal
승패 value의 스케일이 작고 value head가 중요한 보드게임이므로 처음부터 0.25로 낮추지 말고
0.5를 유지한 뒤 비교한다.

## Learning Rate

현재 AdamW `1e-4`는 보수적이고 좋다.

권장:

| 상황 | learning rate |
| --- | --- |
| baseline | `1e-4` |
| learner steps를 크게 늘릴 때 | `5e-5`-`1e-4` |
| batch size 768 이상 | `1e-4` 유지 후 관찰 |
| consistency loss 추가 | `5e-5`-`1e-4` |

SGD `0.2` 같은 LightZero 기본값은 해당 구현의 scalar transform, categorical support,
Atari-style training schedule 기준이다. 현재 PyTorch 모델과 AdamW 설정에는 그대로 맞지 않는다.

## Temperature / Exploration

현재 Gumbel selection과 playout cap randomization이 exploration 역할을 한다.

권장:

- 초반에는 현재 temperature/search 설정 유지
- model이 안정되면 fast search 비중을 늘린다.
- opening 다양성이 줄어들면 root noise 또는 sampling temperature를 올린다.
- policy target entropy가 너무 낮으면 search가 early collapse하는지 확인한다.

추적 metric:

- target policy entropy
- selected action entropy by move number
- opening state duplicate ratio
- target vs prior KL

## Evaluation Cadence

`always_promote=true`는 빠른 실험에는 좋지만, 하이퍼파라미터 비교에는 위험하다.

권장:

| 단계 | 평가 방식 |
| --- | --- |
| 빠른 개발 | always promote, fixed candidate matrix는 주기적으로만 |
| hparam 비교 | fixed seed arena 또는 candidate matrix |
| 장기 run | N iteration마다 arena, 매 iteration lightweight eval |

최소 metric:

- fixed opening set policy/value eval
- best checkpoint 대비 arena
- 이전 N개 checkpoint와 pairwise mini arena

## 추천 실험 순서

### Experiment A: v2 replay only

목표: trajectory replay가 기존 replay와 같은 품질의 batch를 만드는지 확인한다.

변경:

- `bootstrap_td_steps=0`
- `reanalyze_fraction=0`
- priority off
- consistency off

성공 조건:

- 기존 pipeline과 비슷한 loss/arena 추이를 보인다.

### Experiment B: bootstrap value target

변경:

- `bootstrap_td_steps=4`
- network-only target snapshot 사용
- learner steps는 baseline 유지

성공 조건:

- value loss가 더 빨리 안정된다.
- arena가 악화되지 않는다.

### Experiment C: reanalyze fraction

비교:

```text
reanalyze_fraction = 0.0, 0.25, 0.5, 1.0
```

성공 조건:

- 같은 self-play games에서 arena가 개선된다.
- wall-clock 대비 성능도 같이 본다.

### Experiment D: learner reuse

비교:

```text
learner_steps_per_actor_batch = 128, 256, 512, 1024
```

성공 조건:

- 특정 지점 이후 overfit/off-policy 징후가 보이면 그 직전 값을 채택한다.

### Experiment E: priority replay

변경:

- priority alpha 0.4, 0.6
- beta 0.2, 0.4

성공 조건:

- value_priority와 policy_kl high sample을 더 잘 줄인다.
- arena가 흔들리지 않는다.

### Experiment F: consistency auxiliary

변경:

- `consistency_loss_weight=0.02, 0.05, 0.1`
- `consistency_unroll_steps=1, 2`

성공 조건:

- policy/value loss가 악화되지 않는다.
- fixed evaluation과 arena가 개선된다.

## 추천 기본값 요약

v2 첫 장기 run은 다음으로 시작한다.

```json
{
  "actor": {
    "games_per_actor_batch": 1000,
    "trajectory_shard_games": 256,
    "gumbel_simulations_full": 64,
    "gumbel_simulations_fast": 16,
    "playout_cap_full_search_fraction": 0.25,
    "max_considered_actions_full": 16,
    "max_considered_actions_fast": 4,
    "onnx_max_batch_size": 8192,
    "leaf_batch_size": 256
  },
  "learner": {
    "model_preset": "medium_plus",
    "batch_size": 512,
    "learner_steps_per_actor_batch": 256,
    "learning_rate": 0.0001,
    "weight_decay": 0.0001,
    "amp": true,
    "policy_loss_weight": 1.0,
    "value_loss_weight": 0.5,
    "consistency_loss_weight": 0.0,
    "grad_clip_norm": 10.0
  },
  "target": {
    "discount_factor": 1.0,
    "bootstrap_td_steps": 4,
    "reanalyze_fraction": 0.25,
    "reanalyze_batch_size": 1024,
    "max_target_age_iterations": 4,
    "reanalyze_noise": false
  },
  "replay": {
    "capacity_transitions": 500000,
    "min_replay_transitions_before_train": 20000,
    "recent_sample_fraction": 0.35,
    "recent_sample_window": 50000,
    "priority_enabled": false
  }
}
```

이 설정으로 baseline을 만든 뒤, 다음 순서로 하나씩 켠다.

1. `reanalyze_fraction`: 0.25 -> 0.5
2. `learner_steps_per_actor_batch`: 256 -> 512
3. priority replay on
4. `consistency_loss_weight`: 0 -> 0.02
5. fast search budget 감소

## 실패 징후와 조정법

| 징후 | 가능한 원인 | 조정 |
| --- | --- | --- |
| value loss는 낮은데 arena가 나빠짐 | bootstrap target bias, overfit | `td_steps` 감소, reanalyze 증가 |
| policy KL이 계속 높음 | target이 오래됨, search target 불안정 | target age 제한, search reanalyze 일부 적용 |
| GPU util이 낮음 | batch 준비 병목 | prefetch/pinned memory/legal mask precompute |
| actor가 너무 느림 | search budget 과함 | fast sims 감소, full fraction 감소 |
| learner가 너무 느림 | batch/model 과함 | batch 512 유지, model 키우지 않음 |
| opening 다양성 감소 | exploration 부족 | temperature/root noise 조정 |
| 장기 run에서 성능 출렁임 | always promote와 off-policy 누적 | mini arena 도입, EMA/best gating |

## 결론

처음부터 EfficientZero 전체를 구현하려 하지 말고, 하이퍼파라미터도 그에 맞춰 보수적으로 둔다.

가장 먼저 고정할 기준값:

- `batch_size=512`
- `learning_rate=1e-4`
- `gumbel_simulations_full=64`
- `gumbel_simulations_fast=16`
- `bootstrap_td_steps=4`
- `reanalyze_fraction=0.25`
- `learner_steps_per_actor_batch=256`
- priority off
- consistency off

이 baseline이 기존 pipeline보다 안정적으로 같거나 좋아진 뒤에 reanalyze, learner reuse,
priority, consistency 순서로 샘플 효율을 끌어올린다.
