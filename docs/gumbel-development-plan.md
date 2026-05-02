# Gumbel AlphaZero 구현 계획

이 문서는 Great Kingdom AI에 Gumbel AlphaZero 스타일 탐색 backend를 추가하기 위한 개발 계획이다.

기존 PUCT 기반 `MctsSearch`는 baseline으로 유지한다. 새 구현은 별도 Rust module과 별도 Python backend 선택 경로로 붙인다. 목표는 기존 MCTS를 패치해서 특수 보정하는 것이 아니라, Gumbel AlphaZero의 root policy improvement와 sequential halving을 도입해 제한된 simulation budget에서 더 안정적인 정책 개선을 얻는 것이다.

참고 논문은 DeepMind ICLR 2022 논문 **Policy improvement by planning with Gumbel**이다.

- OpenReview: <https://openreview.net/forum?id=bERaNdoegnO>

---

## 1. 목표

Gumbel backend의 목표는 다음과 같다.

1. 기존 `MctsSearch`를 변경하지 않고 `GumbelSearch`를 추가한다.
2. Rust 규칙 엔진의 실제 `GameState`를 사용하므로 MuZero의 learned dynamics는 구현하지 않는다.
3. Python/PyTorch는 기존처럼 GPU batch inference를 담당한다.
4. Rust는 CPU-side search, Gumbel root action sampling, sequential halving, terminal backup, improved policy target 생성을 담당한다.
5. self-play, arena, pipeline에서 config로 `"mcts"`와 `"gumbel"` backend를 선택할 수 있게 한다.
6. Gumbel backend는 단순 visit count가 아니라 improved policy target을 반환한다.

이 작업은 pass collapse만을 위한 특수 규칙 패치가 아니다. pass는 계속 합법 수이며, 규칙 외 전술 보정은 넣지 않는다. Gumbel backend는 policy prior가 불안정한 초기 학습 구간에서도 root action 비교를 더 안정적으로 만들기 위한 대체 탐색 backend다.

---

## 2. 설계 원칙

### 기존 MCTS 유지

`rust/great_kingdom_core/src/mcts.rs`의 PUCT MCTS는 baseline으로 유지한다.

Gumbel 구현은 새 파일 `rust/great_kingdom_core/src/gumbel.rs`에 둔다. 기존 MCTS와 공통으로 쓸 수 있는 `EvalRequest`, action constants, `GameState` API는 재사용하되, Gumbel search state와 candidate scheduling은 분리한다.

### AlphaZero용 Gumbel 구현

Great Kingdom은 Rust 규칙 엔진이 실제 transition model이다. 따라서 MuZero의 representation, dynamics, reward model은 구현하지 않는다.

구현 대상은 다음이다.

1. root prior logits에 Gumbel noise를 더해 후보 action을 sampling without replacement 한다.
2. 선택된 root 후보들에 대해 sequential halving을 수행한다.
3. terminal outcome은 neural value보다 우선한다.
4. completed Q/value와 prior logits를 이용해 improved policy target을 만든다.

### GPU + CPU 역할 분리

GPU 활용은 기존 구조를 따른다.

Python은 `EvalRequest`를 tensor batch로 변환하고 PyTorch model inference를 수행한다.

Rust는 가능한 많은 leaf evaluation request를 모아 Python callback으로 넘긴다. CPU thread는 root candidate rollout, state clone/apply, terminal check, backup, sequential halving bookkeeping에 집중한다.

---

## 3. Rust API 계획

### 새 module

추가 파일:

```text
rust/great_kingdom_core/src/gumbel.rs
```

`lib.rs`에서 module과 Python class를 export한다.

```rust
mod gumbel;

pub use gumbel::{GumbelConfig, GumbelResult, GumbelSearch};
```

Python module에는 다음 class를 추가한다.

```rust
module.add_class::<GumbelResult>()?;
module.add_class::<GumbelSearch>()?;
```

초기 구현에서는 `GumbelSelfPlayBatch`를 바로 만들지 않는다. 먼저 single-game search API를 완성하고, Python에서 batch orchestration을 붙인 뒤 병목이 확인되면 `GumbelSelfPlayBatch`를 추가한다.

### GumbelConfig

초기 config 필드는 다음으로 고정한다.

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
4. 후보 수는 현재 root legal action 수보다 크면 legal action 수로 clamp

### GumbelResult

Gumbel result는 기존 `MctsResult`와 호환되는 진단 값을 포함하되, 학습 target은 `policy_target`을 우선한다.

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

`policy_target`은 legal action에 대해서만 양수이고 합이 1이어야 한다.

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

result = search.search_with_priors_and_evaluator(
    state,
    priors,
    evaluator,
    leaf_batch_size=16,
)
```

지원 method:

```python
search_with_priors(state, priors) -> GumbelResult
search_with_priors_and_evaluator(state, priors, evaluator, leaf_batch_size=16) -> GumbelResult
set_simulations(simulations)
```

`search_with_priors`는 evaluator 없이 terminal-only / value-zero fallback smoke용으로 사용한다.

---

## 4. Gumbel 알고리즘 계획

### 4.1 Root candidate sampling

root legal actions에 대해서만 후보를 만든다.

1. Python model prior는 legal mask 적용 후 softmax된 확률로 들어온다.
2. Rust는 legal action prior를 다시 정규화한다.
3. prior가 0인 action은 작은 epsilon을 사용해 `log_prior`가 `-inf`로 터지지 않게 한다.
4. 각 legal action에 대해 `gumbel = -ln(-ln(u))`를 샘플링한다.
5. `log_prior + gumbel` 점수로 정렬한다.
6. 상위 `max_considered_actions`를 root candidate로 선택한다.

sampling은 fixed seed에서 deterministic해야 한다.

### 4.2 Sequential halving

선택된 candidate set을 sequential halving으로 줄인다.

초기 후보 수를 `K`라 할 때:

1. round마다 남은 후보에 simulation budget을 균등하게 배분한다.
2. 각 후보를 최소 1회 이상 평가한다.
3. 후보별 completed value estimate를 갱신한다.
4. ranking score로 상위 절반을 남긴다.
5. 후보가 1개가 되거나 simulation budget이 끝나면 종료한다.

ranking score는 다음 요소를 포함한다.

```text
gumbel_score(action)
+ transformed_completed_q(action)
+ prior_logit_adjustment(action)
```

`c_visit`, `c_scale`은 논문식 completed Q 변환에 사용한다. 구현 중 논문 수식을 직접 코드 주석에 명시한다.

### 4.3 Leaf evaluation

각 root candidate rollout은 `GameState.apply(action)`으로 시작한다.

1. 적용 결과가 terminal이면 규칙 outcome을 value로 사용한다.
2. terminal이 아니면 leaf state를 batch에 모아 Python evaluator로 넘긴다.
3. evaluator는 기존 `EvalRequest`를 사용한다.
4. 반환된 value는 leaf 현재 플레이어 관점이므로 root player 관점에 맞게 부호를 조정한다.

초기 버전은 root action 중심 planning으로 제한한다. root action 이후 deep tree traversal까지 완전한 Gumbel tree policy를 구현하는 것은 2차 단계로 둔다. 단, 이 제한은 `docs/gumbel-development-plan.md`와 코드 주석에 명시한다.

### 4.4 Improved policy target

Gumbel backend는 visit count를 그대로 policy target으로 쓰지 않는다.

최종 후보들의 ranking score를 기반으로 legal action 전체 길이의 policy target을 만든다.

1. 후보 action들의 improved logits를 계산한다.
2. 후보 밖 legal action은 0 target을 둔다.
3. 후보 action target은 softmax(improved logits)로 만든다.
4. terminal win이 후보에 있으면 해당 action에 target을 강하게 집중한다.
5. 모든 값은 finite이고 합이 1이어야 한다.

`visit_counts`는 저장하되 학습 target으로는 사용하지 않는다.

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

### Protocol 확장

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

Gumbel backend에서도 `selected_action()`을 우선 사용한다. self-play에서 temperature sampling이 필요한 경우에는 `policy_target()`에서 sampling한다.

### Backend factory

Python helper를 추가한다.

```python
create_core_search_backend(config, backend, seed_offset=0)
```

backend가 `"mcts"`면 기존 `core.MctsSearch`를 만든다.

backend가 `"gumbel"`이면 `core.GumbelSearch`를 만든다.

### Arena

arena는 deterministic해야 한다.

Gumbel backend 사용 시에도 seed를 고정한다. candidate/best가 같은 seed schedule을 쓰도록 game seed와 player 색을 반영해 search seed를 만든다.

평가에서는 root noise와 temperature sampling을 쓰지 않는다.

### Self-play

self-play에서는 root noise와 Gumbel noise의 역할이 겹친다.

초기 정책:

1. backend가 `"mcts"`면 기존 root Dirichlet noise를 유지한다.
2. backend가 `"gumbel"`이면 root Dirichlet noise는 기본 off로 둔다.
3. Gumbel 자체 sampling seed가 exploration 역할을 한다.

이 기본값은 config에서 바꿀 수 있게 하되, Runpod Gumbel config는 root noise off로 시작한다.

---

## 6. Config 계획

runpod config에는 Gumbel 값을 추가하되, 첫 merge 시 기본 backend는 `"mcts"`로 둔다.

예시:

```json
{
  "search_backend": "gumbel",
  "gumbel_simulations": 128,
  "gumbel_max_considered_actions": 16,
  "gumbel_c_visit": 50.0,
  "gumbel_c_scale": 1.0,
  "gumbel_seed": 2026
}
```

test config에는 작은 값을 둔다.

```json
{
  "search_backend": "gumbel",
  "gumbel_simulations": 16,
  "gumbel_max_considered_actions": 8,
  "gumbel_c_visit": 50.0,
  "gumbel_c_scale": 1.0,
  "gumbel_seed": 7
}
```

---

## 7. 테스트 계획

### Rust unit tests

필수 테스트:

1. illegal action은 root candidate에 들어가지 않는다.
2. fixed seed에서 Gumbel candidate sampling이 deterministic하다.
3. root candidates는 중복 없이 sampling된다.
4. `max_considered_actions`보다 legal action이 적으면 legal action 수로 clamp된다.
5. terminal win/loss는 neural value보다 우선한다.
6. `policy_target`은 82차원이고 합이 1이다.
7. `visit_counts`는 82차원이고 simulation 수와 일관된다.
8. `Blue place -> Orange pass -> Blue to move` 국면에서 pass가 terminal loss로 평가되는지 확인한다.

### Python tests

필수 테스트:

1. Gumbel result가 `policy_target()`을 제공하면 self-play replay sample은 이를 그대로 사용한다.
2. MCTS result는 기존 visit count normalization을 계속 사용한다.
3. pipeline config가 `"mcts"`와 `"gumbel"`을 모두 로드한다.
4. arena backend selection이 올바른 core search class를 만든다.
5. Gumbel backend에서도 report JSON shape는 기존 arena report와 호환된다.

### Runpod smoke

Gumbel smoke 기준:

1. 1 iteration pipeline 완료.
2. pass-pass 즉시 종료율 기록.
3. average game length 기록.
4. samples/sec 기록.
5. arena candidate win rate 기록.
6. side split 기록.

MCTS와 같은 model/config seed로 비교할 수 있어야 한다.

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

debug 출력은 root candidate별 다음 값을 포함한다.

1. action
2. root prior
3. sampled gumbel
4. completed visits
5. completed q
6. improved logit
7. final target

---

## 9. 구현 순서

### Phase 1. 문서와 skeleton

1. `docs/gumbel-development-plan.md` 작성.
2. `gumbel.rs` skeleton 추가.
3. `GumbelConfig`, `GumbelResult`, `GumbelSearch` 정의.
4. PyO3 export 추가.
5. 기본 constructor 테스트 추가.

### Phase 2. Root Gumbel sampling

1. legal action prior normalization.
2. Gumbel sampling with fixed seed.
3. top-k sampling without replacement.
4. candidate list deterministic test.

### Phase 3. Sequential halving

1. candidate별 simulation budget scheduling.
2. terminal transition evaluation.
3. batched leaf evaluator callback 연결.
4. completed Q update.
5. candidate halving.

### Phase 4. Improved policy target

1. completed Q transform 구현.
2. candidate improved logits 계산.
3. 82-action policy target 생성.
4. selected action 결정.

### Phase 5. Python integration

1. result target extraction helper 추가.
2. self-play single-game path 연결.
3. arena path 연결.
4. pipeline config 연결.
5. batched self-play path는 우선 MCTS 유지 후, Gumbel batch 필요 시 Phase 6에서 추가.

### Phase 6. Gumbel batched self-play

1. `GumbelSelfPlayBatch` 필요성 확인.
2. 필요하면 MCTS batch 구조를 참고해 active games batch evaluator 구현.
3. CPU parallel search와 GPU batch inference가 동시에 효율적으로 동작하도록 request batching 최적화.

---

## 10. 완료 기준

Gumbel backend 1차 완료 기준:

1. `cargo test` 통과.
2. `pytest` 통과.
3. Gumbel backend로 local smoke self-play가 종료된다.
4. Gumbel backend로 arena smoke가 종료된다.
5. Runpod test config에서 1 iteration pipeline이 완료된다.
6. Gumbel report에 pass rate와 consecutive-pass end rate가 기록된다.
7. 기존 MCTS backend가 regression 없이 계속 동작한다.

---

## 11. 리스크

### 구현 복잡도

Gumbel AlphaZero는 단순 PUCT 대체가 아니라 root policy improvement와 target 생성 방식까지 바꾼다. 구현 중 논문 수식과 프로젝트 단순화 지점을 반드시 문서화해야 한다.

### Batch 효율

초기 single-game Gumbel은 GPU batch 효율이 낮을 수 있다. Runpod 성능을 위해서는 Phase 6의 batched self-play가 필요할 가능성이 높다.

### Value 외삽 문제

Gumbel은 prior collapse를 줄일 수 있지만, value head가 특정 국면 전체를 잘못 평가하면 완전한 해결책은 아니다. 따라서 replay 분포 지표와 pass-response sample 지표를 같이 봐야 한다.

### 기존 MCTS와 비교

Gumbel이 항상 더 강하다고 가정하지 않는다. MCTS backend와 같은 compute budget에서 A/B 비교한다.

---

## 12. 기본 결정 사항

1. 기존 MCTS는 유지한다.
2. Gumbel은 별도 backend로 추가한다.
3. 기본 backend는 merge 시점에는 `"mcts"`로 둔다.
4. Runpod 실험 config에서 `"gumbel"`을 켜서 비교한다.
5. Gumbel self-play에서는 root Dirichlet noise를 기본 off로 둔다.
6. Gumbel result의 `policy_target()`을 replay policy target으로 사용한다.
7. `visit_counts()`는 diagnostics와 backward compatibility 용도로 유지한다.
