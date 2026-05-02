# Gumbel AlphaZero 논문급 구현 계획

이 문서는 Great Kingdom AI에 **Policy improvement by planning with Gumbel** 스타일의 Gumbel AlphaZero 탐색 backend를 추가하기 위한 개발 계획이다.

목표는 root action에만 Gumbel noise를 붙이는 간이 selector가 아니다. 기존 PUCT `MctsSearch`를 baseline으로 유지하면서, 별도 backend로 **root sequential halving과 내부 node deterministic action selection을 결합한 Gumbel tree search**를 구현한다.

참고 논문:

- DeepMind, ICLR 2022, **Policy improvement by planning with Gumbel**
- OpenReview: <https://openreview.net/forum?id=bERaNdoegnO>

---

## 1. 목표

Gumbel backend의 목표는 다음과 같다.

1. 기존 `MctsSearch`는 변경하지 않고 baseline으로 유지한다.
2. 새 backend는 별도 Rust module과 별도 Python backend 선택 경로로 추가한다.
3. Great Kingdom은 Rust 규칙 엔진의 실제 `GameState` transition을 사용하므로 MuZero의 learned dynamics/reward model은 구현하지 않는다.
4. 단, 탐색 자체는 root-only가 아니라 실제 game tree를 내려가는 Gumbel tree search로 구현한다.
5. root node는 Gumbel sampled considered action set과 sequential halving 상태를 가진다.
6. 내부 node는 Gumbel noise와 sequential halving을 쓰지 않고, improved policy 기반 deterministic action selection을 사용한다.
7. Python/PyTorch는 기존처럼 GPU batch inference를 담당한다.
8. Rust는 CPU-side tree search, root Gumbel sampling, root sequential halving scheduling, 내부 node deterministic selection, terminal backup, completed Q 관리, improved policy target 생성을 담당한다.
9. self-play, arena, pipeline에서 config로 `"mcts"`와 `"gumbel"` backend를 선택할 수 있게 한다.
10. Gumbel backend는 visit count target이 아니라 논문식 improved policy target을 반환한다.

이 작업은 pass collapse를 위한 규칙 패치가 아니다. pass는 계속 합법 수이며, 규칙 외 전술 보정은 넣지 않는다. terminal outcome은 규칙 엔진 결과로만 반영한다.

Great Kingdom의 Rust `GameState` transition은 MuZero의 learned dynamics를 대체하는 완전한 환경 모델이다. 따라서 이 backend는 모델 오차가 있는 MuZero dynamics 구현이 아니라, **AlphaZero 스타일의 완전한 규칙 환경에서 Gumbel 탐색 연산자를 적용한 형태**다.

---

## 2. 설계 원칙

### 기존 MCTS 유지

`rust/great_kingdom_core/src/mcts.rs`의 PUCT MCTS는 baseline으로 유지한다.

Gumbel 구현은 `rust/great_kingdom_core/src/gumbel/` 하위 module로 분리한다. MCTS처럼 파일 하나가 비대해지지 않도록 public glue, config, tree, root scheduler, inner selection, sampling, policy, batch를 나눈다.

권장 layout:

```text
rust/great_kingdom_core/src/gumbel/
  mod.rs
  config.rs
  result.rs
  rng.rs
  sampling.rs
  node.rs
  sequential_halving.rs
  selection.rs
  search.rs
  batch.rs
  policy.rs
  debug.rs
```

### 논문급 Gumbel tree search

구현 대상은 다음이다.

1. root node에서 legal prior logits에 Gumbel noise를 더해 considered action set을 sampling without replacement 한다.
2. root node는 considered action set에 대해 sequential halving round state를 저장한다.
3. 내부 node는 Gumbel noise를 추가하지 않고, completed Q로 계산한 improved policy와 visit ratio를 이용한 deterministic action selection 상태를 가진다.
4. traversal은 root에서는 sequential halving scheduler가 action을 고르고, 내부 node에서는 deterministic selection rule이 action을 고른다.
5. child node가 없고 terminal도 아니면 `EvalRequest`로 leaf evaluation을 요청한다.
6. evaluation response의 policy로 child node를 expand하고 value를 backup한다.
7. backup 이후 root의 sequential halving 상태와 모든 path edge의 completed Q/visit 상태를 갱신한다.
8. root의 final improved policy target은 root node의 Gumbel score, prior logits, completed Q transform으로 만든다.

금지되는 축소 구현:

1. root action만 평가하고 끝내는 root-only planning.
2. root 이후 leaf value만 평균내고 tree node를 만들지 않는 구현.
3. 내부 node는 PUCT로 내려가고 root만 Gumbel로 고르는 혼합 구현.
4. 내부 node에 예산 없는 sequential halving을 억지로 적용하는 구현.
5. pass나 특정 전술을 규칙 외 heuristic으로 보정하는 구현.

### GPU + CPU 역할 분리

Python은 `EvalRequest`를 tensor batch로 변환하고 PyTorch model inference를 수행한다.

Rust는 가능한 많은 leaf evaluation request를 모아 Python callback으로 넘긴다. CPU thread는 state clone/apply, terminal check, root scheduler, inner deterministic selection, backup, batch self-play orchestration에 집중한다.

---

## 3. Rust API 계획

### Module export

`lib.rs`에서 module과 Python class를 export한다.

```rust
mod gumbel;

pub use gumbel::{GumbelConfig, GumbelResult, GumbelSearch, GumbelSelfPlayBatch};
```

Python module에는 다음 class를 추가한다.

```rust
module.add_class::<GumbelResult>()?;
module.add_class::<GumbelSearch>()?;
module.add_class::<GumbelSelfPlayBatch>()?;
```

### GumbelConfig

초기 config:

```rust
pub struct GumbelConfig {
    pub simulations: u32,
    pub max_considered_actions: usize,
    pub c_visit: f32,
    pub c_scale: f32,
    pub seed: u64,
}
```

기본값:

```text
simulations = 128
max_considered_actions = 16
c_visit = 50.0
c_scale = 1.0
seed = 0
```

검증 조건:

1. `simulations > 0`
2. `max_considered_actions > 0`
3. `c_visit`와 `c_scale`은 finite positive
4. root 후보 수는 root legal action 수보다 크면 legal action 수로 clamp

### GumbelResult

```rust
pub struct GumbelResult {
    pub selected_action: Option<usize>,
    pub policy_target: [f32; ACTION_SPACE],
    pub visit_counts: [u32; ACTION_SPACE],
}
```

Python methods:

```python
selected_action() -> int | None
policy_target() -> list[float]
visit_counts() -> list[int]
```

`policy_target`은 root legal action에 대해서만 양수이고 합이 1이어야 한다. 후보 밖 legal action은 0 target일 수 있다. `visit_counts()`는 diagnostics와 backward compatibility 용도이며 학습 target으로 쓰지 않는다.

### GumbelSearch

Python API는 기존 MCTS와 최대한 같은 shape로 둔다.

```python
search = core.GumbelSearch(
    simulations=128,
    max_considered_actions=16,
    c_visit=50.0,
    c_scale=1.0,
    seed=0,
)

result = search.search_with_logits_and_evaluator(
    state,
    policy_logits,
    evaluator,
    leaf_batch_size=16,
)
```

지원 method:

```python
search_with_logits(state, policy_logits) -> GumbelResult
search_with_logits_and_evaluator(state, policy_logits, evaluator, leaf_batch_size=16) -> GumbelResult
search_with_priors(state, priors) -> GumbelResult
search_with_priors_and_evaluator(state, priors, evaluator, leaf_batch_size=16) -> GumbelResult
set_simulations(simulations)
set_seed(seed)
```

`search_with_logits*`가 Gumbel backend의 primary API다. Python model이 만든 raw policy logits를 Rust로 넘기고, Rust가 legal mask와 stable log-softmax를 적용해 `log_prior`를 만든다.

`search_with_priors*`는 기존 MCTS shape와 smoke/debug 호환을 위한 fallback API로 유지한다. 실제 Gumbel self-play/arena/pipeline은 `search_with_logits_and_evaluator`를 사용한다.

### GumbelSelfPlayBatch

batch API는 `MctsSelfPlayBatch`와 호환되는 shape를 유지한다.

```python
batch = core.GumbelSelfPlayBatch(
    game_count=32,
    simulations=128,
    max_considered_actions=16,
    c_visit=50.0,
    c_scale=1.0,
    seed=2026,
)
```

지원 method:

```python
len()
active_count()
active_game_indexes()
active_eval_request()
current_players()
is_terminal()
winners()
end_reasons()
territory_scores()
search_active_with_priors(priors)
search_active_with_priors_and_evaluator(priors, evaluator, leaf_batch_size=16)
search_active_with_logits(policy_logits)
search_active_with_logits_and_evaluator(policy_logits, evaluator, leaf_batch_size=16)
apply_actions(actions)
set_simulations(simulations)
set_seeds(seeds)
```

---

## 4. Gumbel 알고리즘 계획

### 4.1 Node expansion

root node는 다음 상태를 가진다.

```text
to_play
visit_count
legal edges
considered edge indexes
gumbel values per considered action
sequential halving round state
completed visit/value/Q per edge
child node index per edge
node value estimate
```

root expansion 절차:

1. neural policy raw logits는 Python에서 softmax하지 않고 Rust로 들어온다.
2. Rust는 root legal action에 대해서만 stable log-softmax를 적용해 `log_prior`를 만든다.
3. illegal action logit은 무시한다.
4. legal action마다 deterministic RNG로 Gumbel noise를 샘플링한다.
5. `log_prior + gumbel` 상위 `max_considered_actions`를 considered action set으로 저장한다.
6. root sequential halving scheduler를 초기화한다.

root sampling은 fixed seed에서 재현 가능해야 한다. batch self-play에서도 같은 seed schedule이면 같은 root 탐색 결과가 나와야 한다.

내부 node expansion 절차:

1. 내부 node는 evaluator가 반환한 raw logits에서 legal action 기준 `log_prior`를 만든다.
2. 내부 node는 Gumbel noise를 샘플링하지 않는다.
3. 내부 node는 모든 legal edge에 대해 visit/value/Q stats를 가진다.
4. 내부 node selection은 4.2의 deterministic action selection rule을 사용한다.

#### Policy input contract

Gumbel backend의 기본 Rust API인 `search_with_logits*`는 **raw policy logits**를 입력으로 받는다. Gumbel root score와 improved policy target은 log probability 공간에서 계산되므로, Python에서 probability로 변환한 뒤 다시 Rust에서 log를 취하지 않는다.

Rust는 node legal action에 대해서만 stable log-softmax를 수행한다.

```text
max_legal_logit = max(logit(a) for a in legal_actions)
log_z = max_legal_logit + ln(sum(exp(logit(a) - max_legal_logit) for a in legal_actions))
log_prior(a) = logit(a) - log_z
```

규칙:

1. illegal action logit은 무시한다.
2. legal action logit이 하나라도 NaN 또는 infinity면 reject한다.
3. legal action이 없으면 terminal node로 처리하고 policy target은 all-zero 또는 terminal fallback 규칙을 따른다.
4. raw logits는 `search_with_priors*`에 섞어 받지 않는다.
5. `search_with_priors*`는 probability prior fallback API로만 유지한다.
6. probability prior fallback에서는 legal action 기준 재정규화 후 `ln(max(normalized_prior, prior_epsilon))`로 `log_prior`를 만든다.
7. probability prior fallback은 illegal action prior를 무시하고, legal prior 합이 0이면 uniform prior를 사용한다.
8. probability prior fallback은 NaN, infinity, negative prior를 reject한다.

### 4.2 Root node sequential halving & Inner node selection

Sequential halving은 **root node에만** 적용한다. root는 search call마다 고정된 simulation budget을 가지므로 후보 action을 round별로 줄이는 halving schedule을 정의할 수 있다.

Root scheduler는 다음을 저장한다.

```text
active candidates
round index
per-round target visits
completed visits per candidate
candidate ranking score
finished flag
```

root considered action 수는 다음으로 정한다.

```text
K = min(max_considered_actions, root_legal_action_count, simulations)
```

따라서 모든 root candidate는 최소 1회 평가될 수 있다. `K == 0`이면 terminal root로 취급하고 action을 선택하지 않는다. `K == 1`이면 sequential halving round를 만들지 않고 scheduler를 finished로 표시한다. 이 경우 유일한 candidate를 남은 simulation의 root action으로 사용하거나 final selection으로 바로 사용한다.

root에서 simulation을 시작하면:

1. root scheduler가 아직 finished가 아니면, 현재 round에서 추가 평가가 필요한 candidate action을 고른다.
2. action을 apply한다.
3. terminal이면 규칙 outcome value를 즉시 backup한다.
4. child가 있으면 child node로 내려간다.
5. child가 없으면 leaf evaluation request를 만든다.
6. backup 후 root candidate completed Q를 갱신한다.
7. round target을 채운 candidate가 모두 준비되면 ranking score로 하위 candidate를 제거한다.
8. 후보가 1개가 되거나 root simulation budget이 끝나면 scheduler를 finished로 표시한다.

round별 기본 quota는 논문식 sequential halving을 따른다.

```text
if K == 1:
    scheduler.finished = true
else:
    round_count = ceil(log2(K))
    quota(round) = max(1, floor(simulations / (round_count * active_candidate_count)))
```

각 round에서는 active candidate마다 `quota(round)`회까지 평가한다. floor 때문에 남는 simulation은 현재 active candidate의 ranking 상위부터 1개씩 deterministic하게 배분한다. tie-break는 action index 오름차순을 사용한다.

각 round transition에서는 active candidates를 ranking score 내림차순으로 정렬한다. `keep_count = ceil(active_candidate_count / 2)`로 계산하고, 상위 `keep_count`를 유지하며 나머지를 제거한다. tie-break는 ranking score, then action index 오름차순을 사용한다.

root ranking score는 논문식으로 다음 항을 포함한다.

```text
gumbel(action)
+ log_prior(action)
+ transformed_completed_q(action)
```

`c_visit`, `c_scale`은 completed Q transform에 사용한다. 구현 코드에는 논문 수식과 MCTX `qtransform_completed_by_mix_value`와의 파라미터 대응을 주석으로 남긴다.

파라미터 대응:

```text
c_visit = maxvisit_init
c_scale = value_scale
```

completed Q transform은 다음 순서로 계산한다.

1. visited action은 search에서 얻은 mean Q를 사용한다.
2. unvisited action은 mixed value로 complete한다.
3. completed Q vector를 legal action 집합 안에서 `[0, 1]`로 rescale한다.
4. `(c_visit + max_visit_count) * c_scale`을 곱해 policy logit에 더할 Q logit bonus를 만든다.

visited mean Q:

```text
q(a) = value_sum(a) / visit_count(a)   if visit_count(a) > 0
```

mixed value:

```text
visited = { a | visit_count(a) > 0 }
N_sum = sum_a visit_count(a)
prior_prob(a) = softmax(log_prior over legal actions)

if visited is empty:
    mixed_value = node_raw_value
else:
    visited_prior_sum = sum_{a in visited} max(prior_prob(a), tiny)
    weighted_q = sum_{a in visited} (max(prior_prob(a), tiny) / visited_prior_sum) * q(a)
    mixed_value = (node_raw_value + N_sum * weighted_q) / (N_sum + 1)
```

completed Q:

```text
completed_q(a) = q(a)             if visit_count(a) > 0
completed_q(a) = mixed_value      otherwise
```

rescale and transform:

```text
q_min = min_{a in legal_actions} completed_q(a)
q_max = max_{a in legal_actions} completed_q(a)
normalized_completed_q(a) =
    (completed_q(a) - q_min) / max(q_max - q_min, epsilon)

max_visit_count = max_{a in legal_actions} visit_count(a)
visit_scale = c_visit + max_visit_count
transformed_completed_q(a) =
    visit_scale * c_scale * normalized_completed_q(a)
```

논문 본문의 concrete instantiation은 다음 형태다.

```text
sigma(q_hat(a)) = (c_visit + max_b N(b)) * c_scale * q_hat(a)
```

여기서 구현의 `q_hat(a)`는 completed Q를 `[0, 1]`로 rescale한 `normalized_completed_q(a)`로 둔다. 이는 MCTX 기본 `qtransform_completed_by_mix_value(rescale_values=True, use_mixed_value=True)`와 같은 방향이다.

`node_raw_value`는 해당 node의 현재 플레이어 관점 value estimate다. root는 root evaluator value를 저장하고, 내부 node는 해당 node를 expand할 때 evaluator가 반환한 value를 저장한다. terminal edge는 evaluator value를 사용하지 않고 terminal outcome value를 completed Q로 저장한다.

unvisited action을 0으로 처리하지 않는다. value range와 player perspective에 따라 0은 중립값이 아닐 수 있기 때문이다. unvisited action은 mixed value로 complete해서 prior-only collapse와 arbitrary zero bias를 피한다.

내부 node에는 sequential halving을 적용하지 않는다. 내부 node는 root처럼 node-local fixed simulation budget이 없으므로, halving round별 후보 제거를 안정적으로 정의할 수 없다.

대신 내부 node는 **Deterministic Action Selection**을 사용한다. 내부 node에서는 Gumbel noise를 추가하지 않는다. 각 edge의 completed Q-value로 improved policy `π′`를 계산하고, 방문 횟수 비율이 `π′`에 수렴하도록 다음 수식으로 action을 고른다.

```text
Action = argmax_a [ ImprovedPolicy(a) - (VisitCount(a) / (1 + Sum(TotalVisits))) ]
```

여기서 `ImprovedPolicy(a)`는 해당 내부 node의 legal prior와 completed Q transform으로 계산한 정책 개선 분포다. 구현에서는 legal action에 대해서만 값을 계산하고, illegal action은 후보에서 제외한다.

내부 node improved policy는 다음으로 계산한다.

```text
inner_improved_logit(a) = log_prior(a) + transformed_completed_q(a)
ImprovedPolicy = softmax(inner_improved_logits over legal actions)
```

이 selection rule은 PUCT의 exploration bonus로 action을 고르는 방식이 아니라, planning으로 개선된 정책 `π′` 자체를 방문 비율 목표로 삼는다. 따라서 내부 node selection은 기존 PUCT보다 정책 개선 효과를 더 직접적으로 탐색 visit distribution에 반영한다.

### 4.3 Tree traversal

각 simulation은 root에서 시작해 실제 tree를 내려간다.

1. 현재 state가 terminal이면 terminal value를 반환한다.
2. 현재 node가 unexpanded이면 evaluator response로 expand한다.
3. root node에서는 sequential halving scheduler가 다음 action을 선택한다.
4. 내부 node에서는 deterministic selection rule이 다음 action을 선택한다.
5. 선택한 action을 `GameState.apply(action)`으로 적용한다.
6. terminal이면 value를 backup한다.
7. child node가 있으면 child로 이동한다.
8. child node가 없으면 leaf state를 pending eval로 모은다.

root 이후 traversal은 내부 node deterministic selection을 사용한다. 내부 node에서 PUCT로 선택하는 shortcut은 금지한다.

### 4.4 Leaf evaluation and backup

leaf evaluation:

1. terminal outcome은 neural value보다 우선한다.
2. non-terminal leaf는 `EvalRequest`로 Python evaluator에 넘긴다.
3. evaluator response는 `(policy, value)`이다.
4. policy로 leaf node를 expand한다.
5. value는 leaf 현재 플레이어 관점이므로 backup 중 edge/player 관점에 맞게 부호를 번갈아 적용한다.

value/backup convention:

1. node의 `node_value`는 해당 node의 `to_play` player 관점 value다.
2. edge Q는 **edge parent node의 `to_play` player 관점**으로 저장한다.
3. `edge.value_sum / edge.visit_count`는 그 edge를 선택하는 player 관점의 mean Q다.
4. leaf evaluator value는 leaf node의 current player 관점이므로, path를 역순으로 오르며 edge마다 부호를 번갈아 바꾼다.
5. terminal outcome value는 terminal을 만든 edge parent player 관점으로 먼저 계산한 뒤 같은 convention으로 저장한다.

이 convention은 기존 MCTS의 사고 모델과 맞고, 내부 deterministic selection에서 현재 node player 관점의 completed Q를 바로 사용할 수 있게 한다. 기존 MCTS backend 구현은 변경하지 않는다.

backup:

1. path의 edge visit/value sum을 갱신한다.
2. path의 node visit count를 갱신한다.
3. root scheduler의 candidate completed Q/visits를 갱신한다.
4. root scheduler가 round transition 조건을 만족하면 halving을 수행한다.
5. 내부 node는 edge completed Q/visits를 갱신하고 다음 deterministic selection에서 이 값을 사용한다.

### 4.5 Improved policy target

Gumbel backend는 visit count를 policy target으로 쓰지 않는다.

root node의 final improved logits를 기반으로 82-action policy target을 만든다.

1. root considered action들의 improved logits를 계산한다.
2. root improved logit은 `gumbel + log_prior + transformed_completed_q`를 기본으로 한다.
3. 후보 밖 legal action은 0 target을 둔다.
4. 후보 action target은 softmax(improved logits)로 만든다.
5. immediate terminal win 후보가 있으면 그 Q가 자연스럽게 target을 지배해야 한다. 별도 규칙 보정으로 강제 선택하지 않는다.
6. 모든 값은 finite이고 합이 1이어야 한다.

`selected_action()`은 root final improved logits의 argmax를 반환한다. Gumbel backend에서는 arena와 self-play 모두 기본적으로 이 selected action을 실제 착수로 사용한다. `policy_target()`은 replay 학습 target이며, 기본 착수 sampling 분포로 사용하지 않는다.

root improved policy target 수식:

```text
root_improved_logit(a) =
    gumbel(a) + log_prior(a) + transformed_completed_q(a)

policy_target = softmax(root_improved_logits over root considered actions)
```

root target은 visit count normalization으로 만들지 않는다. 단, completed Q transform의 scale 계산에는 논문식으로 visit count statistics를 사용할 수 있다. 반환되는 `visit_counts`는 diagnostics와 compatibility 용도다.

---

## 5. Python 통합 계획

### Search backend config

`PipelineConfig`, `MctsSelfPlayConfig`, `ArenaConfig` 중 search 생성에 필요한 config에 backend 필드를 추가한다.

```python
search_backend: str = "mcts"
```

허용 값:

```text
"mcts"
"gumbel"
```

Gumbel 관련 config:

```python
gumbel_simulations: int = 128
gumbel_max_considered_actions: int = 16
gumbel_c_visit: float = 50.0
gumbel_c_scale: float = 1.0
gumbel_seed: int = 0
```

기존 MCTS config는 그대로 유지한다.

### Policy target extraction

self-play/evaluate protocol은 다음 method를 optional하게 다룬다.

```python
policy_target() -> list[float]
```

target 생성 규칙:

```python
if hasattr(result, "policy_target"):
    policy = np.asarray(result.policy_target(), dtype=np.float32)
else:
    policy = policy_target_from_visit_counts(result.visit_counts())
```

착수 선택:

1. backend가 `"gumbel"`이면 arena와 self-play 모두 `selected_action()`을 실제 착수로 사용한다.
2. backend가 `"gumbel"`이면 `policy_target()`은 replay sample의 policy target으로만 사용한다.
3. backend가 `"mcts"`이면 기존 visit count sampling과 temperature schedule을 유지한다.
4. Gumbel에서 추가 temperature sampling이 필요하면 별도 experiment flag로만 켜고, 기본값은 off로 둔다.

### Backend factory

Python helper를 추가한다.

```python
create_core_search_backend(config, backend, seed_offset=0)
create_core_self_play_batch_backend(config, backend, game_count, seed_offset=0)
```

backend가 `"mcts"`면 기존 `core.MctsSearch`/`core.MctsSelfPlayBatch`를 만든다.

backend가 `"gumbel"`이면 `core.GumbelSearch`/`core.GumbelSelfPlayBatch`를 만든다.

### Arena

arena는 deterministic해야 한다.

Gumbel backend 사용 시 seed를 고정한다. candidate/best가 같은 schedule을 쓰도록 game seed, player color, turn index를 반영해 search seed를 만든다.

평가에서는 root Dirichlet noise와 temperature sampling을 쓰지 않는다.

### Self-play

self-play에서는 Gumbel noise가 exploration 역할을 한다.

기본 정책:

1. backend가 `"mcts"`면 기존 root Dirichlet noise 기본값을 유지한다.
2. backend가 `"gumbel"`이면 root Dirichlet noise는 기본 off로 둔다.
3. Gumbel self-play의 stochasticity는 Gumbel sampling seed schedule로 관리한다.
4. backend가 `"gumbel"`이면 opening temperature sampling은 기본 off로 둔다.
5. backend가 `"gumbel"`이면 실제 착수는 `selected_action()`을 사용하고, replay에는 `policy_target()`을 저장한다.
6. 사용자가 config에서 root noise나 temperature sampling을 명시하면 experiment override로만 허용한다.

---

## 6. Config 계획

merge 시 기본 backend는 `"mcts"`로 둔다. Gumbel은 test/runpod config에서 명시적으로 켠다.

runpod 예시:

```json
{
  "search_backend": "gumbel",
  "gumbel_simulations": 128,
  "gumbel_max_considered_actions": 16,
  "gumbel_c_visit": 50.0,
  "gumbel_c_scale": 1.0,
  "gumbel_seed": 2026,
  "root_noise": false
}
```

test 예시:

```json
{
  "search_backend": "gumbel",
  "gumbel_simulations": 16,
  "gumbel_max_considered_actions": 8,
  "gumbel_c_visit": 50.0,
  "gumbel_c_scale": 1.0,
  "gumbel_seed": 7,
  "root_noise": false
}
```

---

## 7. 테스트 계획

### Rust unit tests

필수 테스트:

1. illegal action은 root considered candidate와 내부 node legal edge에 들어가지 않는다.
2. fixed seed에서 Gumbel candidate sampling이 deterministic하다.
3. candidates는 중복 없이 sampling된다.
4. `max_considered_actions`보다 legal action이 적으면 legal action 수로 clamp된다.
5. `simulations < max_considered_actions`이면 root candidate 수가 simulations로 clamp된다.
6. raw logits 입력은 legal action 기준 stable log-softmax로 `log_prior`를 만든다.
7. illegal action logit은 root candidate와 내부 node prior 계산에서 무시된다.
8. legal action logit의 NaN 또는 infinity는 reject한다.
9. probability prior fallback 입력은 legal action 기준으로 재정규화된다.
10. probability prior fallback의 all-zero legal priors는 uniform fallback을 사용한다.
11. probability prior fallback의 NaN, infinity, negative prior는 reject한다.
12. root node만 sequential halving scheduler를 가진다.
13. `K == 1`이면 root scheduler가 finished로 처리되어 quota division by zero가 발생하지 않는다.
14. root round quota가 `K > 1`에서 `max(1, floor(simulations / (ceil(log2(K)) * active_count)))`로 계산된다.
15. round별 active set이 deterministic ranking으로 `ceil(active_count / 2)`개씩 유지된다.
16. odd active count의 candidate elimination은 상위 `ceil(active_count / 2)` 유지, 나머지 제거로 동작한다.
17. floor remainder simulation은 ranking 상위부터 deterministic하게 배분된다.
18. simulation이 depth 2 이상 tree traversal을 수행하는 smoke test를 둔다.
19. 내부 node selection이 PUCT path나 Gumbel noise가 아니라 deterministic selection 수식을 사용하는지 구조 테스트를 둔다.
20. 내부 node visit ratio가 improved policy 방향으로 수렴하도록 action을 선택한다.
21. unvisited edge completed Q는 0이 아니라 mixed value로 채워진다.
22. mixed value는 raw node value와 visited prior-weighted Q를 `(raw_value + N_sum * weighted_q) / (N_sum + 1)`로 섞는다.
23. completed Q는 legal action 집합 안에서 `[0, 1]`로 rescale된다.
24. all completed Q가 같으면 transform은 finite fallback을 사용한다.
25. transformed completed Q는 `(c_visit + max_visit_count) * c_scale * normalized_completed_q`를 사용한다.
26. terminal win/loss는 neural value보다 우선한다.
27. terminal edge completed Q는 terminal outcome value를 사용한다.
28. backup 부호가 현재 player 관점에서 일관된다.
29. 1-ply terminal win은 parent edge Q `+1`로 저장된다.
30. 1-ply terminal loss는 parent edge Q `-1`로 저장된다.
31. 2-ply evaluator value backup은 edge parent player 관점에 맞게 부호가 뒤집힌다.
32. root sequential halving round transition과 candidate elimination이 deterministic하다.
33. `policy_target`은 82차원이고 합이 1이다.
34. root `policy_target`은 visit count normalization이 아니라 `gumbel + log_prior + transformed_completed_q` softmax를 사용한다.
35. `visit_counts`는 82차원이고 simulation 수와 일관된다.
36. completed Q transform scale 계산은 visit count statistics를 사용할 수 있다.
37. `Blue place -> Orange pass -> Blue to move` 국면에서 pass가 terminal loss로 평가되는지 확인한다.
38. `GumbelSelfPlayBatch`가 여러 active game의 leaf eval을 하나의 evaluator batch로 모은다.

### Python tests

필수 테스트:

1. Gumbel result가 `policy_target()`을 제공하면 self-play replay sample은 이를 그대로 사용한다.
2. MCTS result는 기존 visit count normalization을 계속 사용한다.
3. Gumbel self-play는 opening/hot turn에서도 기본 착수로 `selected_action()`을 사용한다.
4. pipeline config가 `"mcts"`와 `"gumbel"`을 모두 로드한다.
5. Gumbel self-play와 arena path는 model probability가 아니라 raw policy logits를 core `search_with_logits*` API에 전달한다.
6. MCTS backend는 기존 probability prior path를 유지한다.
7. arena backend selection이 올바른 core search class를 만든다.
8. Gumbel backend에서도 report JSON shape는 기존 arena report와 호환된다.
9. Gumbel batched self-play path가 `GumbelSelfPlayBatch`를 사용한다.

### Runpod smoke

Gumbel smoke 기준:

1. 1 iteration pipeline 완료.
2. pass-pass 즉시 종료율 기록.
3. average game length 기록.
4. samples/sec 기록.
5. arena candidate win rate 기록.
6. side split 기록.
7. MCTS와 같은 model/config seed로 A/B 비교 가능.

---

## 8. 지표와 디버깅

pipeline metrics에 다음 값을 추가한다.

1. `search_backend`
2. `pass_rate`
3. `consecutive_pass_end_rate`
4. `average_game_length`
5. `blue_win_rate`
6. `orange_win_rate`
7. `prev_pass_blue_samples`
8. `prev_pass_orange_samples`

Gumbel debug는 production API가 아니라 test-only 또는 env flag로 둔다.

예:

```text
GKA_GUMBEL_DEBUG=1
```

debug 출력은 root와 내부 node별로 다음 값을 포함한다.

1. node id/path/depth
2. action
3. prior
4. sampled gumbel(root only)
5. considered 여부
6. halving round(root only)
7. completed visits
8. completed Q
9. improved policy
10. deterministic selection score(inner only)
11. ranking score(root only)
12. eliminated 여부(root only)
13. final root target

---

## 9. 구현 순서

### ~~Phase 1. 문서와 skeleton~~

1. 이 문서를 root-only가 아닌 논문급 Gumbel tree search 기준으로 수정한다.
2. `rust/great_kingdom_core/src/gumbel/` module skeleton을 추가한다.
3. `GumbelConfig`, `GumbelResult`, `GumbelSearch`, `GumbelSelfPlayBatch`를 정의한다.
4. PyO3 export를 추가한다.
5. 기본 constructor/config validation 테스트를 추가한다.

### ~~Phase 2. Root sampling and scheduler~~

1. legal action stable log-softmax from raw logits.
2. deterministic root Gumbel sampling.
3. root top-k sampling without replacement.
4. root sequential halving scheduler state 구현.
5. candidate list와 root halving transition deterministic test.

### ~~Phase 3. Inner deterministic selection~~

1. 내부 node improved policy 계산 구현.
2. `ImprovedPolicy(a) - VisitCount(a) / (1 + Sum(TotalVisits))` selection 구현.
3. 내부 node에서 Gumbel noise와 PUCT selection을 쓰지 않는 구조 테스트 추가.

### ~~Phase 4. Full tree traversal~~

1. `GumbelNode`/`GumbelEdge` tree storage 구현.
2. root는 sequential halving, child/internal node는 deterministic selection을 사용한다.
3. state clone/apply로 실제 transition을 수행한다.
4. terminal transition evaluation과 backup을 구현한다.
5. depth > 1 traversal 테스트를 추가한다.

### Phase 5. Batched leaf evaluator

1. pending leaf request batching.
2. existing `EvalRequest` 재사용.
3. evaluator response logits/value parsing 공용화.
4. child node expansion.
5. virtual reservation 또는 equivalent pending guard로 같은 wave 안 중복 leaf 평가를 방지한다.
6. evaluator logits/value로 child node expand → uniform fallback 제거

### Phase 6. Improved policy target

1. 논문식 completed Q transform 구현.
2. root improved logits 계산.
3. 82-action policy target 생성.
4. `selected_action()` 결정.
5. finite/sum/legal mask 검증.

### Phase 7. Python integration

1. result target extraction helper 추가.
2. Gumbel self-play single-game path는 raw policy logits provider로 연결.
3. Gumbel arena path는 raw policy logits provider로 연결.
4. pipeline config 연결.
5. backend factory 추가.
6. Gumbel self-play 기본 root noise off 적용.

### Phase 8. Gumbel batched self-play

1. `GumbelSelfPlayBatch` 구현.
2. active games batch evaluator 구현.
3. CPU parallel search와 GPU batch inference가 동시에 동작하도록 request batching 최적화.
4. Runpod smoke config 연결.

---

## 10. 완료 기준

Gumbel backend 1차 완료 기준:

1. `cargo test` 통과.
2. `pytest` 통과.
3. Gumbel backend로 local smoke self-play가 종료된다.
4. Gumbel backend로 arena smoke가 종료된다.
5. Runpod test config에서 1 iteration pipeline이 완료된다.
6. Gumbel report에 pass rate와 consecutive-pass end rate가 기록된다.
7. depth > 1 tree traversal 테스트가 존재한다.
8. root sequential halving 테스트가 존재한다.
9. 내부 node deterministic selection 테스트가 존재한다.
10. 기존 MCTS backend가 regression 없이 계속 동작한다.

---

## 11. 리스크

### 구현 복잡도

논문급 Gumbel은 단순 PUCT 대체나 root selector가 아니다. root sequential halving, 내부 deterministic selection, completed Q transform, target 생성 방식까지 바뀐다. 구현 중 논문 수식과 프로젝트 적용 지점을 코드 주석과 테스트 이름에 명시한다.

### Batch 효율

full tree Gumbel은 leaf request가 depth별로 흩어질 수 있다. Runpod 성능을 위해 `GumbelSelfPlayBatch`에서 active game leaf를 wave 단위로 모아 evaluator call 수를 제한한다.

### Value 외삽 문제

Gumbel은 prior collapse를 줄일 수 있지만, value head가 특정 국면 전체를 잘못 평가하면 완전한 해결책은 아니다. 따라서 replay 분포 지표와 pass-response sample 지표를 같이 본다.

### 기존 MCTS와 비교

Gumbel이 항상 더 강하다고 가정하지 않는다. MCTS backend와 같은 compute budget에서 A/B 비교한다.

---

## 12. 기본 결정 사항

1. 기존 MCTS는 유지한다.
2. Gumbel은 별도 backend로 추가한다.
3. Gumbel은 root-only가 아니라 full tree search로 구현한다.
4. sequential halving은 fixed simulation budget이 있는 root node에만 적용한다.
5. 내부 node는 Gumbel noise 없이 improved policy 기반 deterministic selection을 사용한다.
6. 내부 node를 PUCT로 대체하지 않는다.
7. MuZero learned dynamics는 구현하지 않고 Rust `GameState` transition을 사용한다.
8. 기본 backend는 merge 시점에는 `"mcts"`로 둔다.
9. Runpod 실험 config에서 `"gumbel"`을 켜서 비교한다.
10. Gumbel self-play에서는 root Dirichlet noise를 기본 off로 둔다.
11. Gumbel self-play에서는 실제 착수로 `selected_action()`을 사용한다.
12. Gumbel result의 `policy_target()`을 replay policy target으로 사용한다.
13. `visit_counts()`는 diagnostics와 backward compatibility 용도로 유지한다.
