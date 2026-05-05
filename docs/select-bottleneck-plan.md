# Select 병목 개선 추천안: Selection hot path allocation 제거

## 목적

Runpod Rust ONNX self-play profile에서 Gumbel search의 최대 병목은 `eval`이 아니라 `select`다.
이 문서는 현재 로그 기준으로 `select` 병목을 줄이는 1순위 추천안과 가성비를 정리한다.

## 관찰 로그 요약

관찰 조건:

- `active_games=256`
- `rayon_threads=27`
- Runpod config 기준 `onnx_device=cuda`
- `onnx_max_batch_size=8192`
- self-play `leaf_batch_size=128`

제공 로그 30 waves 합계:

```text
total   ~= 2.248s
select  ~= 0.916s  (40.7%)
request ~= 0.303s  (13.5%)
eval    ~= 0.506s  (22.5%)
backup  ~= 0.513s  (22.8%)
```

`select`가 단독 1위다. `request + eval`을 합치면 neural evaluation 쪽도 크지만, search wall time을
가장 직접적으로 줄일 1순위는 `select` hot path 개선이다.

## 현재 select 경로

`GumbelSelfPlayBatch::search_active_with_evaluator()`는 wave마다 active game별 pending leaf를 만든다.

핵심 흐름:

```rust
for _ in 0..batch_target {
    let Some(root_action) = scheduler.next_action() else {
        break;
    };
    let mut simulation_state = state.clone();
    match search.select_eval_leaf(root_index, root_action, &mut simulation_state) {
        ...
    }
}
```

각 simulation은 다음 비용을 낸다.

- `RootSequentialHalving::next_action()` 선형 스캔
- root `edge_index_for_action()` 선형 검색
- `GameState::clone()`
- path를 따라 `GameState::apply(action)`
- `path: Vec<(usize, usize)>` allocation/grow
- 내부 node마다 `select_inner_action_index()`
- pending leaf 반환 시 leaf `GameState::clone()`

특히 `select_inner_action_index()`는 내부 node 선택마다 여러 임시 `Vec`를 만든다.

```rust
let edges = node.edges.iter().map(|edge| edge.inner_stats()).collect::<Vec<_>>();
select_inner_action(&edges, node.node_value, c_visit, c_scale)
```

`select_inner_action()` 내부에서는 다시 다음 `Vec`들이 만들어진다.

- `prior_probs`
- `completed_q`
- `transformed_q`
- `logits`
- `InnerPolicyEntry`

즉 현재 `select` 병목은 neural eval이 아니라 **CPU-side search hot path에서 반복 allocation과 선형
스캔이 누적되는 문제**로 보는 것이 맞다.

## 추천 방안 1개

**selection hot path를 allocation-free에 가깝게 바꾼다.**

구체적으로는 `select_inner_action_index()`와 scheduler/action lookup 주변을 먼저 고친다.

첫 구현 범위:

1. `select_inner_action_index()`가 `Vec`를 만들지 않고 `GumbelNode.edges` slice를 직접 순회하게 한다.
2. `inner_improved_policy()`의 중간 `Vec` 생성 없이 한두 번의 pass로 best action을 계산한다.
3. root/inner action lookup을 매번 `edge_index_for_action()` 선형 검색하지 않도록 node별 action index map 또는
   small fixed lookup을 도입한다.
4. `RootSequentialHalving::next_action()` / `reserve_visit()`의 선형 scan 비용을 줄인다.

이 작업은 search semantics를 바꾸지 않는다. 같은 입력에서 같은 tie-break를 유지해야 한다.

## 설계

### 1. `select_inner_action_index()` allocation 제거

현재 구조는 “정책 전체를 Vec로 만든 뒤 best를 고르는 방식”이다.

선택에 필요한 최종 값은 action 하나뿐이다. 따라서 policy vector를 만들 필요가 없다.

현재 수식:

```text
prior_probs = softmax(log_prior)
completed_q = visited q와 prior 기반 mixed value
q_bonus = transformed_completed_q(completed_q)
policy = softmax(log_prior + q_bonus)
score = policy_probability - visit_count / (1 + total_visits)
best = max(score, tie-break)
```

allocation-free 구현 방향:

1. 첫 pass:
   - `max_log_prior`
   - `sum_exp_prior`
   - `total_visits`
   - `visited_prior_sum`
   - `weighted_q`
2. 두 번째 pass:
   - 각 edge의 completed q 계산
   - `q_min`, `q_max`, `max_visit_count` 계산
3. 세 번째 pass:
   - `logit = log_prior + transformed_q`
   - `max_logit`, `sum_exp_logit` 계산
4. 네 번째 pass:
   - `probability`
   - visit ratio penalty
   - best action 선택

pass 수는 늘어도 edge 수는 작다. 기본 `gumbel_max_considered_actions=16`이고 내부 legal action도 최대
82개다. 여기서는 heap allocation 제거가 더 중요하다.

### 2. 기존 public helper 유지

`inner_improved_policy()`는 테스트와 정책 target 계산에서 쓰일 수 있으므로 바로 삭제하지 않는다.

권장 구조:

```rust
pub(crate) fn select_inner_action_from_node(
    node: &GumbelNode,
    c_visit: f32,
    c_scale: f32,
) -> Option<usize> {
    select_inner_action_from_edges(&node.edges, node.node_value, c_visit, c_scale)
}
```

기존 `select_inner_action()`은 테스트 친화 API로 남기고, search hot path만 새 함수로 우회한다.

### 3. Action lookup 경량화

현재는 action index를 edge index로 바꿀 때 매번 `edges.iter().position(...)`을 돈다.

hot path 위치:

- root action 선택 직후
- inner action 선택 직후

개선안:

- `GumbelNode`에 `[Option<usize>; ACTION_SPACE]` 형태의 `edge_index_by_action`을 추가한다.
- node 생성 시 legal edges와 함께 lookup table을 채운다.
- `edge_index_for_action()`은 lookup table을 즉시 조회한다.

장점:

- search semantics 변화 없음
- root/inner lookup 모두 개선
- 테스트 영향이 명확함

단점:

- node memory가 증가한다.
- `ACTION_SPACE=82`, `Option<usize>`는 node당 수백 바이트가 될 수 있다.

메모리가 부담이면 `[u16; ACTION_SPACE]`와 sentinel `u16::MAX`를 쓴다. node 수가 많아질 수 있으므로 이쪽이
더 낫다.

### 4. Scheduler scan 경량화

`RootSequentialHalving::next_action()`은 active 후보 전체를 매번 스캔한다.
`reserve_visit()`도 action을 찾기 위해 다시 스캔한다.

현재 후보 수는 `gumbel_max_considered_actions=16`이라 매우 크지는 않지만, simulation마다 호출되므로
누적된다.

개선안:

- `next_action()`이 action뿐 아니라 active candidate index도 반환한다.
- `reserve_visit_by_index(index)`로 같은 후보를 다시 찾지 않는다.

예시:

```rust
let Some(next) = scheduler.next_candidate() else { break };
let root_action = next.action;
...
scheduler.reserve_candidate(next.index);
```

이 변경은 작지만 hot path에서 중복 스캔을 제거한다.

## 계측

현재 `select_elapsed`는 너무 큰 묶음이다. 최적화 전에 다음 계측을 추가한다.

```text
select_scheduler
select_state_clone
select_tree_walk
select_inner_policy
select_action_lookup
select_terminal_backup
select_pending_push
```

최소 계측만 한다면 다음 3개가 우선이다.

```text
select_scheduler
select_tree_walk
select_inner_policy
```

추천 출력:

```text
[gka-select-profile] wave=8 sims=516 scheduler=0.003s walk=0.010s inner=0.012s terminal=0.001s total=0.029s
```

이 계측이 있어야 `Vec` allocation 제거가 맞는 처방인지, 실제로는 `GameState::apply()`가 지배적인지 판단할
수 있다.

## 가성비 판단

### 수정 effort

중간.

필요 작업:

- allocation-free inner action selection 함수 추가
- 기존 `select_inner_action()`과 결과 parity 테스트
- `GumbelSearch::select_eval_leaf()` hot path에서 새 함수 사용
- action lookup table 또는 compact index map 추가
- scheduler가 candidate index를 반환하도록 보조 API 추가
- Runpod profile 비교

Gumbel 알고리즘 자체를 바꾸지 않고 동일 수식을 더 적은 allocation으로 계산하는 작업이다. 따라서
dynamic batching이나 search algorithm 변경보다 리스크가 낮다.

### 기대 효과

현재 `select`는 전체의 약 `40.7%`다. 이 중 절반만 줄어도 전체 wall time은 약 `20%` 가까이 줄 수 있다.

보수적으로 보면:

```text
select 15% 감소 -> 전체 약 6% 감소
select 30% 감소 -> 전체 약 12% 감소
select 50% 감소 -> 전체 약 20% 감소
```

allocation-free inner selection은 특히 깊은 path를 자주 타는 후반 wave에서 효과가 날 가능성이 있다.
로그에서도 후반 wave는 `leaves=258`인데 `select=0.020~0.035s`로 eval보다 큰 경우가 많다.

### 유지했을 때 병목

현재 구조를 유지하면 ONNX eval을 더 빠르게 만들어도 `select`가 더 지배적이 된다.

예를 들어 eval을 30% 줄여도:

```text
eval 0.506s -> 0.354s
total 2.248s -> 2.096s
select 비중 40.7% -> 43.7%
```

즉 eval 최적화만 하면 병목이 select로 더 선명하게 이동한다. 현재 로그 기준으로는 select를 먼저 잡는 것이
가성비가 더 좋다.

## 다른 대안과 비교

### EvalRequest dedupe

장점:

- 중복 state가 많으면 neural eval row 수를 직접 줄인다.
- correctness 리스크가 낮다.

단점:

- 현재 1위 병목인 `select`는 줄이지 못한다.
- 중복 state가 적은 중후반에는 효과가 작다.

판단:

- 좋은 2순위 후보다.
- select 개선 후에도 `request + eval`이 크면 다시 검토한다.

### leaf_batch_size 조정

장점:

- wave 수와 batch shape를 바꿔 profile을 빠르게 확인할 수 있다.
- 코드 변경 없이 실험 가능하다.

단점:

- 큰 leaf batch는 pending/reservation이 많아지고 search behavior에 영향을 줄 수 있다.
- 근본적으로 simulation당 selection 비용을 줄이지 않는다.

판단:

- 튜닝 실험으로는 좋지만 1순위 구조 개선은 아니다.

### Gumbel search 알고리즘 변경

장점:

- selection 비용 자체를 크게 줄일 수 있다.

단점:

- policy target 품질, exploration, arena strength에 직접 영향이 있다.
- correctness와 학습 결과를 다시 검증해야 한다.

판단:

- 지금 단계의 가성비는 낮다.

## 권장 구현 순서

1. `select` 세부 계측을 추가한다.
2. `select_inner_action_index()` allocation-free 버전을 추가하고 기존 함수와 parity test를 작성한다.
3. search hot path만 새 함수로 교체한다.
4. Runpod profile에서 `select` 감소율을 확인한다.
5. 효과가 확인되면 action lookup table과 scheduler index API를 추가한다.
6. 그래도 `select`가 크면 `GameState::clone/apply` 비용을 별도 최적화 대상으로 분리한다.

## 성공 기준

최소 성공 기준:

- 기존 Gumbel search 테스트 통과
- 새 inner selection 함수가 기존 `select_inner_action()`과 tie-break까지 동일
- Runpod profile에서 `select_elapsed` 감소
- self-play samples/sec 증가

추천 판단 기준:

```text
select_elapsed 20% 이상 감소
전체 search wall time 8% 이상 감소
policy target / selected action parity 테스트 통과
```

이 기준을 만족하지 못하면 `select` 내부 병목이 allocation이 아니라 `GameState::apply()` 또는 scheduler 구조에
있다는 뜻이므로, 다음 단계는 state transition 재사용 또는 path walk 계측 강화로 잡는다.
