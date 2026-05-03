# Batched Arena 구현 계획

## 목표

현재 arena 평가는 `run_arena()`가 게임을 1개씩 순차 실행하고, 각 턴의 root model 평가도 batch size 1로 수행한다. Rust Gumbel search 내부의 leaf 평가는 `leaf_batch_size`로 묶이지만 여러 arena 게임 사이의 root/leaf inference는 batch화되지 않는다.

이 문서의 목표는 arena 평가를 여러 게임 단위로 batch 실행해 Runpod RTX 3090 환경에서 GPU 활용률을 높이고, pipeline의 arena wall time을 줄이는 것이다. 기존 arena report schema, 승격 기준, candidate/best 교대 규칙은 유지한다.

## 현재 병목

- `python/great_kingdom_ai/evaluate.py`의 `run_arena()`는 `for index in range(config.games)`로 게임을 순차 실행한다.
- `play_arena_game()`은 매 턴마다 현재 player의 모델만 `evaluate_feature_batch(..., batch_size=1)`로 평가한다.
- Rust `GumbelSearch.search_with_logits_and_evaluator()`는 단일 게임 안에서 leaf evaluation만 batch화한다.
- self-play에는 `GumbelSelfPlayBatch`가 있지만 arena에는 candidate/best 모델이 player별로 달라지는 요구사항 때문에 전용 batch runner가 없다.

## 설계 요약

`ArenaConfig`에 `batch_size`를 추가하고, `batch_size > 1`이면 batched arena 경로를 사용한다. `batch_size`는 동시에 진행할 arena game 수이며, `config.games`가 더 크면 Python이 여러 chunk/window로 나눠 실행한다. Python은 active game들의 root evaluation을 candidate 모델용 batch와 best 모델용 batch로 나눠 실행하고, Rust는 active game들의 Gumbel leaf search를 한 번에 진행한다.

기본값은 호환성을 위해 `batch_size=1`이다. batch backend가 없는데 `batch_size > 1`이면 조용히 느린 경로로 돌아가지 않고 `RuntimeError`를 발생시킨다. 성능 기능이 켜졌는데 실제로 켜지지 않는 상황을 빨리 드러내기 위해서다.

## Public API 변경

### Python `ArenaConfig`

`python/great_kingdom_ai/evaluate.py`의 `ArenaConfig`에 필드를 추가한다.

```python
batch_size: int = 1
```

validation:

- `batch_size <= 0`이면 `ValueError`
- `batch_size == 1`이면 기존 순차 경로
- `batch_size > 1`이면 batched arena 경로

### Python 함수

기존 public 함수는 유지한다.

```python
def run_arena(...) -> ArenaReport
```

내부 dispatch:

- `config.batch_size == 1`: 기존 순차 구현 사용
- `config.batch_size > 1`: `run_arena_batched(...)` 호출

새 helper:

```python
def run_arena_batched(
    *,
    candidate_model: Any,
    best_model: Any,
    config: ArenaConfig,
    progress_callback: Callable[[int, int, ArenaGameResult], None] | None = None,
) -> ArenaReport
```

`run_arena_batched()`는 전체 arena를 `batch_size` 단위로 chunk 처리한다.

- `chunk_start`는 global game index다.
- `chunk_size = min(config.batch_size, config.games - chunk_start)`
- `create_core_arena_batch(..., game_count=chunk_size, seed_start=config.seed_start + chunk_start, game_index_start=chunk_start)`를 호출한다.
- candidate side split은 chunk-local index가 아니라 global game index 기준으로 유지한다.

```python
def create_core_arena_batch(
    config: ArenaConfig,
    *,
    game_count: int,
    seed_start: int,
    game_index_start: int,
) -> ArenaBatchLike
```

`state_factory`와 `search_factory`는 테스트용 순차 경로에 남긴다. batched 경로 v1은 Rust core backend만 지원한다.

### Rust PyO3 class

`great_kingdom_core.GumbelArenaBatch`를 추가한다.

생성자:

```text
GumbelArenaBatch(
    game_count,
    seed_start = 0,
    game_index_start = 0,
    simulations = 128,
    max_considered_actions = 16,
    c_visit = 50.0,
    c_scale = 1.0,
    seed = 2026
)
```

필수 메서드:

- `len() -> usize`
- `active_game_indexes() -> Vec<usize>`
- `active_eval_request() -> EvalRequest`
- `current_players() -> Vec<u8>`
- `candidate_players() -> Vec<u8>`
- `winners() -> Vec<Option<u8>>`
- `end_reasons() -> Vec<Option<u8>>`
- `territory_scores() -> Vec<(u8, u8)>`
- `search_active_with_logits_and_evaluator(policy_logits, evaluator, root_values, leaf_batch_size) -> Vec<Option<GumbelResult>>`
- `apply_actions(actions: Vec<Option<usize>>) -> Vec<Option<u8>>`

## Rust 구현 계획

새 파일 `rust/great_kingdom_core/src/gumbel/arena_batch.rs`를 만든다. `gumbel/mod.rs`와 `lib.rs`에서 export한다.

`GumbelArenaBatch` 내부 상태:

```rust
states: Vec<GameState>
searches: Vec<[GumbelSearch; 2]>
candidate_players: Vec<u8>
seeds: Vec<u64>
```

candidate player 규칙:

- game index가 짝수면 candidate는 BLUE
- game index가 홀수면 candidate는 ORANGE
- 기존 `run_arena()`와 같은 side split을 유지한다.
- chunk 실행 시에는 `game_index_start + chunk_local_index`를 game index로 사용한다.

seed 규칙:

- game seed = `seed_start + chunk_local_index`
- BLUE search seed offset = `game_seed * 2`
- ORANGE search seed offset = `game_seed * 2 + 1`
- 최종 search seed = `gumbel_seed + offset`

search 선택:

- active game의 `current_player()`가 BLUE면 BLUE search 사용
- ORANGE면 ORANGE search 사용
- candidate/best 여부는 Python 모델 선택에만 영향을 주고, Rust search는 player별 search state만 관리한다.

leaf search:

- `GumbelSelfPlayBatch::search_active_with_evaluator` 구조를 기준으로 구현한다.
- active game별 root node, legal actions, log priors, scheduler, completed count를 준비한다.
- 각 wave에서 active game들의 pending leaf를 모아 하나의 `EvalRequest`로 Python evaluator를 호출한다.
- response를 game별로 다시 나눠 node expansion과 backup을 수행한다.
- 가능한 공통 로직은 helper로 분리해 `batch.rs`와 `arena_batch.rs`의 중복을 줄인다.

주의점:

- `leaf_batch_size`는 game별 wave당 최대 leaf 수다. 전체 Python evaluator 호출 batch 크기는 대략 `active_games * leaf_batch_size`까지 커질 수 있다.
- `policy_logits`와 `root_values`는 active game order와 정확히 같은 길이여야 한다.
- return은 전체 game slot 길이의 `Vec<Option<GumbelResult>>`로 한다. inactive/terminal game은 `None`.

## Python 구현 계획

### Protocol 추가

`evaluate.py`에 `ArenaBatchLike` protocol을 추가한다.

```python
class ArenaBatchLike(Protocol):
    def len(self) -> int: ...
    def active_game_indexes(self) -> list[int]: ...
    def active_eval_request(self) -> Any: ...
    def current_players(self) -> list[int] | bytes: ...
    def candidate_players(self) -> list[int] | bytes: ...
    def search_active_with_logits_and_evaluator(
        self,
        policy_logits: list[list[float]],
        evaluator: Callable[[Any], tuple[list[list[float]], list[float]]],
        root_values: list[float],
        leaf_batch_size: int = 8,
    ) -> list[ArenaSearchResultLike | None]: ...
    def apply_actions(self, actions: list[int | None]) -> list[int | None]: ...
    def winners(self) -> list[int | None]: ...
    def end_reasons(self) -> list[int | None]: ...
    def territory_scores(self) -> list[tuple[int, int]]: ...
```

### Root evaluation

매 turn:

1. `active_indexes = batch.active_game_indexes()`
2. `request = batch.active_eval_request()`
3. 가능하면 `request.feature_plane_bytes()`와 `request.legal_mask_bytes()`를 사용한다.
4. fake backend나 tests에서는 `request.feature_planes()`와 `request.legal_masks()` fallback을 허용한다.
5. `players = batch.current_players()`
6. `candidate_players = batch.candidate_players()`

각 active row에 대해:

- `game_index = active_indexes[active_offset]`
- `player = players[game_index]`
- `candidate_player = candidate_players[game_index]`
- `player == candidate_player`이면 candidate model batch로 보낸다.
- 아니면 best model batch로 보낸다.

candidate/best 평가 결과를 active order로 재조립한다.

```python
root_logits_by_active: list[list[float]]
root_values_by_active: list[float]
```

성능 경로:

- Rust `EvalRequest`는 이미 precomputed byte buffers를 제공한다.
- arena batched evaluator는 candidate/best로 row를 나눌 때 가능한 한 numpy array slicing 기반 helper를 사용한다.
- 단순한 `feature_planes()`/`legal_masks()` 리스트 변환은 테스트 fallback이나 작은 fake backend용으로만 둔다.

### Leaf evaluator callback

Rust search가 leaf evaluator를 호출하면 request rows를 다시 candidate/best 모델별로 나눠야 한다.

단, leaf request의 state들은 각 game에서 파생된 simulation state다. 어떤 game에서 온 leaf인지 알아야 candidate/best 모델을 선택할 수 있다. 따라서 Rust `EvalRequest`에 arena leaf metadata를 포함해야 한다.

v1 결정:

- `EvalRequest`에 optional `game_indexes()` metadata를 추가한다.
- `GumbelArenaBatch`가 leaf request를 만들 때 pending leaf의 `game_index`를 함께 넣는다.
- Python evaluator는 `request.game_indexes()`와 batch의 `candidate_players`/현재 leaf player를 함께 사용한다.
- leaf state의 current player는 기존 `EvalRequest.current_players()`를 사용한다.

모델 선택:

- `leaf_player == candidate_players[game_index]`이면 candidate model
- 아니면 best model

이렇게 해야 search 중 후보/베스트가 번갈아 leaf node를 평가하는 상황에서도 올바른 모델을 사용한다.

### Action 적용과 결과 수집

각 turn에서 `results`를 받은 뒤:

- active game마다 `_deterministic_action(result, logits, legal_actions)`로 action 결정
- `legal_actions`는 `active_eval_request().legal_masks()`의 active offset row에서 복원한다.
- `moves[game_index].append(MoveLog(turn=turn, player=player, action=action))`
- `actions[game_index] = action`
- `batch.apply_actions(actions)`

완료된 게임:

- `batch.winners()`와 `batch.end_reasons()`로 terminal 여부 확인
- 새로 terminal이 된 game은 `ArenaGameResult`로 만들고 progress callback 호출
- callback 순서는 game index 오름차순으로 고정한다.

최종 report:

- `games` list는 game index 오름차순으로 정렬한다.
- `summarize_arena(games, promotion_threshold=config.promotion_threshold)` 사용

## EvalRequest metadata 변경

현재 self-play batch에서 `_request_states()`는 `current_players()`가 있으면 읽고, 없으면 `None`으로 처리한다. Rust `EvalRequest.current_players()`는 이미 존재한다. arena leaf evaluator에는 어떤 arena game에서 온 leaf인지가 추가로 필요하므로 Rust `EvalRequest`에 optional game index metadata만 추가한다.

추가 메서드:

- `game_indexes() -> Vec<usize>`

적용 범위:

- 기존 생성자는 metadata 없이 동작해야 한다.
- `EvalRequest::new_with_precomputed_bytes(...)`는 기존 호출 호환성을 유지한다.
- 새 생성자 또는 builder를 추가한다.

예:

```rust
EvalRequest::new_with_game_indexes(states, game_indexes)
```

실제 구현에서는 `current_players`를 별도로 저장할 필요가 없으면 기존처럼 `states.iter().map(GameState::current_player)`로 계산하고, `game_indexes`만 `Option<Vec<usize>>`로 저장한다.

호환성:

- metadata가 없는 request에서 `game_indexes()` 호출 시 빈 Vec를 반환하거나 PyValueError를 낸다.
- Python arena evaluator는 batched arena leaf request에서 metadata가 없으면 `RuntimeError`를 낸다.
- self-play는 기존 동작을 유지한다.

## Config 변경

`configs/runpod/arena-runpod.json`:

```json
"batch_size": 20
```

test/smoke config는 기본값 `1`을 유지한다. batched smoke가 필요하면 별도 config를 추가한다.

README에는 다음을 추가한다.

- arena `batch_size`는 동시에 진행할 arena games 수다.
- `batch_size=1`은 기존 순차 실행이다.
- Runpod RTX 3090에서는 우선 `20`을 권장하고, OOM이 나면 `8` 또는 `4`로 낮춘다.

## 테스트 계획

### Python tests

`tests/test_evaluate.py`에 추가한다.

- `ArenaConfig(batch_size=0)` validation 실패.
- `batch_size=1`에서 기존 `run_arena()` progress callback 동작 유지.
- core에 `GumbelArenaBatch`가 없고 `batch_size>1`이면 `RuntimeError`.
- fake batch backend로 `run_arena_batched()`가 active root rows를 candidate/best 모델로 올바르게 split하는지 확인.
- candidate가 BLUE인 game과 ORANGE인 game이 섞여도 `candidate_player`, `best_player`, `winner` summary가 정확한지 확인.
- 일부 game이 먼저 terminal이 되어도 완료된 game만 progress callback이 호출되고, 최종 games order는 seed order인지 확인.
- leaf evaluator request에 `game_indexes()` metadata가 없으면 명확한 `RuntimeError`.

### Rust tests

`rust/great_kingdom_core`에 추가한다.

- `GumbelArenaBatch::new()`가 game count, candidate player alternation, seed offset을 올바르게 설정한다.
- `active_eval_request()`가 active game만 반환한다.
- `apply_actions()` 후 terminal 상태, winner, end_reason, territory_scores가 갱신된다.
- `search_active_with_logits_and_evaluator()`가 active game 수와 맞지 않는 rows/root_values에 대해 error를 낸다.
- terminal game이 섞여 있을 때 return vector length는 전체 game count이고 terminal slot은 `None`.
- `EvalRequest` metadata round trip: `current_players()`와 `game_indexes()`가 Python에서 읽힌다.

### Integration smoke

로컬 CPU:

```bash
python -m pytest tests/test_evaluate.py tests/test_pipeline.py
```

Rust:

```bash
cd rust/great_kingdom_core
cargo test
```

Runpod CUDA:

```bash
great-kingdom-evaluate \
  --candidate data/runpod/pipeline/checkpoints/candidate.pt \
  --best data/runpod/pipeline/checkpoints/best.pt \
  --report data/runpod/pipeline/reports/arena-batched-smoke.json \
  --config configs/runpod/arena-runpod.json \
  --games 4 \
  --gumbel-simulations 8 \
  --device cuda
```

프로파일링:

```bash
GKA_EVAL_PROFILE=1 GKA_EVAL_PROFILE_INTERVAL=20 \
GKA_GUMBEL_PROFILE=1 GKA_GUMBEL_PROFILE_INTERVAL=5 \
great-kingdom-pipeline --device cuda ...
```

확인할 지표:

- `avg_batch`가 1보다 충분히 커지는지
- model time 대비 transfer/output overhead가 줄었는지
- arena wall time이 기존 순차 실행보다 감소했는지
- candidate win rate와 side split summary가 정상인지

## 구현 순서

1. [x] `ArenaConfig.batch_size`와 validation을 추가한다.
2. Rust `EvalRequest`에 optional metadata를 추가하고 기존 tests를 통과시킨다.
3. Rust `GumbelArenaBatch` skeleton과 basic state/apply/query 메서드를 구현한다.
4. Rust batched search evaluator 메서드를 구현한다.
5. Python `ArenaBatchLike`, `create_core_arena_batch`, `run_arena_batched`를 추가한다.
6. `run_arena()` dispatch를 연결한다.
7. Python/Rust tests를 추가한다.
8. Runpod config와 README를 갱신한다.
9. CPU smoke와 CUDA smoke로 성능/정확성을 확인한다.

## 명시적 제외 범위

- 조기 종료 arena
- 2단계 arena
- multi-process arena
- arena 결과의 통계적 신뢰구간 계산
- training/self-play batch 구조 변경

이 기능은 arena 실행 구조만 바꾸고, promotion 기준과 report schema는 유지한다.
