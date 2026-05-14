# Async Shard/Index Replay 전환 계획

## 배경

현재 async v2 actor/learner 구조는 actor가 shard를 만들고 learner가 shard를 import한다.
하지만 learner import 단계에서 모든 shard를 하나의 `trajectory-replay.npz`로 합친 뒤 매 cycle마다
전체 replay 파일을 다시 저장한다.

현재 병목은 다음 흐름에서 발생한다.

```text
pending shard load
-> shard episodes 복원
-> 기존 replay + shard replay concatenate
-> capacity 초과분 evict
-> 전체 trajectory-replay.npz 재저장
-> train
```

replay가 `512000` rows에 가까워지면 `features`만 해도 GB 단위가 된다. `next_features`,
`root_policy_logits` 같은 optional 4D/2D 배열까지 포함되면 shard 몇 천 rows를 import하기 위해
매번 수 GB 파일을 다시 쓰게 된다.

단기 완화로 async path에서는 학습에 쓰지 않는 optional 배열을 drop하도록 했다. 중장기적으로는
monolithic replay 파일 자체를 없애고, actor가 만든 shard를 replay storage 단위로 유지하는
shard/index 구조로 전환한다.

## 목표

1. shard import를 O(full replay)에서 O(new shard metadata) 수준으로 낮춘다.
2. learner가 전체 replay 파일을 매번 재저장하지 않게 한다.
3. 기존 `TrainingConfig`, `train_from_replay()`, `sample_arrays()` 인터페이스를 최대한 유지한다.
4. crash recovery와 pruning이 명확한 구조를 만든다.
5. 단일 Runpod 디스크 환경에서 먼저 안정적으로 동작하게 한다.

## 비목표

- 당장 mmap ring buffer를 구현하지 않는다.
- trajectory replay의 모든 legacy 기능을 한 번에 대체하지 않는다.
- snapshot/on-sample reanalyze 전체를 첫 단계에서 shard-index backend로 옮기지 않는다.
- multi-node object storage까지 고려하지 않는다.

## 선택안

### 선택: shard/index 기반 replay

actor가 만든 shard 파일을 그대로 보존하고, learner는 index만 관리한다.

```text
data/runpod/train-v2-gumbel-512k/
  shards/
    metadata.jsonl
    actor-seed-state.json
    training-latest-seed-00100000-games-0248/
      trajectory-replay.npz
      game_logs.json
    training-latest-seed-00100248-games-0248/
      trajectory-replay.npz
      game_logs.json
  replay-index/
    index.jsonl
    state.json
  checkpoints/
    training-latest.pt
    onnx/training-latest.onnx
```

learner는 shard를 import할 때 `trajectory-replay.npz`를 큰 replay 파일에 append하지 않고, index에
등록한다.

```text
pending shard
-> validate shard file
-> append index record
-> mark shard imported
-> evict old indexed shards if capacity exceeded
-> train from indexed replay dataset
```

### 보류: mmap ring buffer

mmap ring buffer는 row-level append와 sampling이 빠를 수 있지만, 이 프로젝트에는 당장 부담이 크다.

- episode boundary와 game log 관리가 까다롭다.
- variable metadata, search hash, optional arrays를 고정 schema로 관리해야 한다.
- partial write 복구와 reader/writer lock 설계가 필요하다.
- 기존 shard 기반 actor output과 잘 맞지 않는다.

현재 병목은 초고속 row append가 아니라 전체 NPZ 재저장이므로 shard/index가 더 작은 변경으로 문제를
해결한다.

## 데이터 모델

### ShardIndexRecord

초기 구현은 `index.jsonl`로 충분하다. 나중에 query가 복잡해지면 SQLite로 교체한다.

```json
{
  "event": "shard_indexed",
  "shard_id": "training-latest-seed-01000000-games-0248",
  "status": "active",
  "replay_path": "data/runpod/.../shards/.../trajectory-replay.npz",
  "log_path": "data/runpod/.../shards/.../game_logs.json",
  "rows": 2574,
  "episodes": 248,
  "seed_start": 1000000,
  "games": 248,
  "model_version": "training-latest",
  "model_path": "data/runpod/.../checkpoints/onnx/training-latest.onnx",
  "created_at": "2026-05-14T03:14:00+00:00",
  "indexed_at": "2026-05-14T03:17:00+00:00"
}
```

Eviction도 event log로 남긴다.

```json
{
  "event": "shard_evicted",
  "shard_id": "training-latest-seed-01000000-games-0248",
  "evicted_at": "2026-05-14T04:20:00+00:00",
  "reason": "capacity"
}
```

`state.json`은 빠른 재시작용 cache다.

```json
{
  "capacity": 512000,
  "active_rows": 510234,
  "active_shards": 198,
  "updated_at": "2026-05-14T04:20:00+00:00"
}
```

진실의 원천은 `index.jsonl`이고, `state.json`은 깨지면 재생성 가능해야 한다.

## Dataset 설계

새 Dataset은 기존 training loop와 호환되도록 다음 인터페이스를 제공한다.

```python
class ShardIndexedTrajectoryDataset:
    def __len__(self) -> int: ...
    def sample(self, batch_size: int, rng: random.Random) -> list[ReplaySample]: ...
    def sample_arrays(
        self,
        batch_size: int,
        rng: random.Random,
        *,
        recent_fraction: float = 0.0,
        recent_window: int = 0,
        priority_config: PrioritySamplingConfig | None = None,
    ) -> TrajectoryArrayBatch: ...
```

### Sampling

active shard들의 row count prefix sum을 만든다.

```text
shard A rows=2500  global rows [0, 2500)
shard B rows=2600  global rows [2500, 5100)
shard C rows=2400  global rows [5100, 7500)
```

batch sampling은 기존 row-level sampling과 같은 의미를 유지한다.

1. global row index를 뽑는다.
2. prefix sum으로 shard와 local row를 찾는다.
3. 같은 shard에 속한 row들을 묶는다.
4. shard replay를 load/cache한다.
5. 필요한 rows만 gather한다.
6. `features`, `policy_targets`, terminal value targets, `legal_masks`, `sample_weights`를 batch로 만든다.

### Recency sampling

현재 `recent_sample_window`는 row 단위 의미다. shard/index에서도 유지한다.

```text
global row range [len(dataset) - recent_window, len(dataset))
```

recent/old split은 기존 `sample_priority_indexes()` 의미와 맞춘다.

### Priority sampling

첫 단계에서는 priority를 끄거나 현재처럼 `sample_weights` 기반만 지원한다.

현재 async 설정은 `priority_enabled=false`라 priority는 migration blocker가 아니다.

나중에 priority를 제대로 지원하려면 다음 중 하나를 선택한다.

- shard별 priority array sidecar 저장
- index-level priority summary + sampled row priority refresh
- SQLite table에 row priority 저장

## Shard cache

매 batch마다 NPZ를 열면 느리다. 작은 LRU cache를 둔다.

```text
cache key: replay_path
cache value: TrajectoryReplayStore
capacity: 8~32 shards 또는 메모리 기준
```

현재 shard는 대략 2500~2600 rows라 cache 몇십 개를 유지해도 full replay 하나보다 작다.

cache eviction은 단순 LRU로 충분하다.

## Learner import 변경

현재 continuous learner:

```python
replay = _load_or_create_replay(...)
for shard in pending:
    shard_replay = TrajectoryReplayStore.load(shard.replay_path)
    replay.extend_episodes(shard_replay.episodes)
    _append_event(... shard_imported ...)
if pending:
    replay.save(...)
dataset = TrajectoryReplayDataset(replay)
```

전환 후:

```python
index = ShardReplayIndex.load_or_create(...)
for shard in pending:
    index.add_shard(shard)
    _append_event(... shard_imported ...)
index.evict_to_capacity(...)
dataset = ShardIndexedTrajectoryDataset(index)
```

중요한 순서:

1. shard file validate
2. index append fsync 또는 atomic write
3. metadata에 `shard_imported` append
4. pruning/eviction

`shard_imported`를 index commit 전에 쓰면 crash 시 shard가 metadata상 imported인데 index에는 없는 상태가 된다.

## Pruning 변경

현재 `prune_keep_imported_shards=0`은 imported shard directory를 바로 지운다. shard/index 구조에서는
imported shard가 replay storage 자체가 되므로 이 정책을 바꿔야 한다.

### 새 pruning 원칙

- `active` indexed shard는 절대 지우지 않는다.
- capacity eviction으로 `evicted` 처리된 shard만 지운다.
- learner가 학습 중 cache로 들고 있는 shard는 cycle 끝까지 지우지 않는다.
- `metadata.jsonl`의 `imported`는 "learner index에 등록됨"을 뜻하고 삭제 가능 여부와 분리한다.

### Config 변경안

기존:

```json
"prune_artifacts": true,
"prune_keep_imported_shards": 0
```

변경:

```json
"prune_artifacts": true,
"prune_evicted_shards": true,
"prune_keep_evicted_shards": 0,
"replay_backend": "shard_index"
```

또는 기존 이름을 유지하되 의미를 바꾸지 않는 편이 안전하다. 추천은 새 이름 도입이다.

### Eviction policy

capacity는 row 수 기준이다.

```text
while active_rows > replay_capacity:
    oldest active shard -> evicted
```

한 shard 단위로 evict하므로 active rows는 capacity보다 shard 크기만큼 약간 작거나 클 수 있다. 단순성을 위해
`active_rows <= capacity`가 될 때까지 오래된 shard를 제거한다.

## Crash recovery

복구 규칙은 event log를 기준으로 단순해야 한다.

1. `metadata.jsonl`에서 completed shard 목록을 읽는다.
2. `index.jsonl`에서 active/evicted 상태를 재구성한다.
3. completed이지만 indexed/imported가 아닌 shard는 pending으로 본다.
4. indexed active인데 파일이 없으면 hard error를 낸다.
5. evicted인데 파일이 없어도 정상이다.
6. `state.json`은 재생성한다.

## Migration 계획

### Phase 0: 현 구조 최적화

완료 또는 진행 중:

- async path에서 `root_policy_logits`, `next_features` drop.
- metadata append atomic write.
- actor seed reservation.

### Phase 1: Index module 추가

추가 파일 후보:

```text
python/great_kingdom_ai/shard_replay_index.py
python/great_kingdom_ai/shard_indexed_dataset.py
```

구현:

- `ShardIndexRecord`
- `ShardReplayIndex`
- event log load/replay
- add shard
- evict to capacity
- active shard prefix sum

테스트:

- index add/load idempotency
- evict oldest shards
- missing active shard error
- state rebuild

### Phase 2: Dataset 추가

구현:

- `ShardIndexedTrajectoryDataset.__len__`
- uniform row sampling
- recency sampling
- shard LRU cache
- terminal value target 계산

테스트:

- 기존 `TrajectoryReplayDataset`과 같은 fixture에서 같은 target 반환
- recency window가 global row 기준으로 동작
- 여러 shard에 걸친 batch gather
- cache hit/miss 기본 동작

### Phase 3: Learner backend flag

`LearnerV2Config`에 backend flag를 추가한다.

```python
replay_backend: str = "monolithic"
```

허용값:

```text
monolithic
shard_index
```

초기에는 default를 `monolithic`으로 두고, Runpod config에서만 `shard_index`를 켠다.

### Phase 4: Pruning 전환

`_prune_learner_artifacts()`를 backend별로 분리한다.

```text
monolithic:
  imported shard directories prune 가능

shard_index:
  active indexed shard 보호
  evicted shard만 prune
```

테스트:

- active shard는 prune 대상이 아님
- evicted shard는 prune 대상
- missing evicted shard는 허용

### Phase 5: Monolithic deprecation

Runpod에서 shard_index가 안정화되면 monolithic replay save를 async v2 default에서 제거한다.
train-v2-pipeline은 별도 경로이므로 즉시 제거하지 않는다.

## Legacy monolithic replay migration

기존 `replay/trajectory-replay.npz`는 버리지 않고 shard/index backend로 옮길 수 있다.
권장 방식은 episode boundary를 유지하면서 여러 shard로 쪼개는 것이다.

### 옵션 A: pseudo-shard 등록

가장 빠른 migration은 기존 replay 파일을 하나의 shard처럼 등록하는 것이다.

```text
replay/trajectory-replay.npz
-> shards/migrated-monolithic-000001/trajectory-replay.npz
-> replay-index/index.jsonl에 active shard로 등록
```

장점:

- 구현이 작다.
- 기존 데이터 손실 위험이 낮다.
- shard/index backend smoke test에 좋다.

단점:

- 큰 파일 하나를 계속 load/cache해야 하므로 import 병목만 줄고 sampling I/O 병목은 남는다.
- capacity eviction이 shard 단위라 큰 migrated shard가 오래 살아 있으면 세밀하게 제거하기 어렵다.
- shard cache 효율이 좋지 않다.

이 방식은 fallback/smoke test용으로만 둔다.

### 옵션 B: episode-boundary split migration

권장 방식은 기존 monolithic replay를 episode 단위로 여러 shard로 나누는 것이다.

```text
replay/trajectory-replay.npz
-> shards/migrated-000001/trajectory-replay.npz
-> shards/migrated-000002/trajectory-replay.npz
-> shards/migrated-000003/trajectory-replay.npz
...
-> replay-index/index.jsonl
```

row 단위로 자르면 안 된다. value target 계산은 episode winner와 episode offset에 의존하므로
episode boundary를 반드시 보존한다.

권장 기본값:

```text
target_rows_per_shard = 4096
compressed = false
drop_optional_arrays = true
```

512k rows replay 기준 약 125개 shard가 만들어진다. actor shard 크기와 비슷해서 cache와 eviction에
유리하다.

### Migration 알고리즘

```text
1. source replay를 TrajectoryReplayStore.load()로 읽는다.
2. source.validate()를 수행한다.
3. source.episodes를 순회한다.
4. 현재 chunk row 수 + 다음 episode row 수가 target_rows_per_shard를 넘으면 chunk를 flush한다.
5. flush 시 TrajectoryReplayStore.from_episodes(chunk_capacity, chunk_episodes)를 만든다.
6. async learner가 쓰지 않는 optional arrays를 drop한다.
   - root_policy_logits
   - root_policy_logits_present
   - next_features
   - next_features_present
7. shard directory에 trajectory-replay.npz를 저장한다.
8. shard row/episode 수를 검증한다.
9. replay-index/index.jsonl에 shard_indexed event를 append한다.
10. 모든 shard 생성 후 총 rows/episodes/value target counts가 source와 같은지 검증한다.
```

flush 기준은 episode를 쪼개지 않는 선에서 `target_rows_per_shard`를 넘길 수 있다.
하나의 episode가 target보다 커도 그 episode만 단일 shard로 저장한다.

### Migration script 형태

스크립트 후보:

```text
scripts/migrate_monolithic_replay_to_shards.py
```

명령 예시:

```bash
.venv/bin/python scripts/migrate_monolithic_replay_to_shards.py \
  --source data/runpod/train-v2-gumbel-512k/replay/trajectory-replay.npz \
  --output-shard-root data/runpod/train-v2-gumbel-512k/shards \
  --index-dir data/runpod/train-v2-gumbel-512k/replay-index \
  --target-rows-per-shard 4096 \
  --drop-optional-arrays
```

dry-run도 지원한다.

```bash
.venv/bin/python scripts/migrate_monolithic_replay_to_shards.py \
  --source data/runpod/train-v2-gumbel-512k/replay/trajectory-replay.npz \
  --output-shard-root data/runpod/train-v2-gumbel-512k/shards \
  --index-dir data/runpod/train-v2-gumbel-512k/replay-index \
  --target-rows-per-shard 4096 \
  --drop-optional-arrays \
  --dry-run
```

### Index event 예시

```json
{
  "event": "shard_indexed",
  "shard_id": "migrated-000001",
  "status": "active",
  "replay_path": "data/runpod/train-v2-gumbel-512k/shards/migrated-000001/trajectory-replay.npz",
  "log_path": null,
  "rows": 4107,
  "episodes": 337,
  "seed_start": null,
  "games": 337,
  "model_version": "migrated",
  "model_path": null,
  "created_at": null,
  "indexed_at": "2026-05-14T04:30:00+00:00",
  "source": "legacy-monolithic",
  "source_replay": "data/runpod/train-v2-gumbel-512k/replay/trajectory-replay.npz"
}
```

### 검증 항목

마이그레이션 후 다음 값이 source와 일치해야 한다.

```text
total rows
total episodes
value target counts
winner counts
player counts
policy entropy summary 허용 오차
sample weight summary 허용 오차
```

또한 shard/index dataset에서 같은 seed로 batch를 여러 번 샘플링해 shape와 target 범위를 검증한다.

```text
features: [batch, 11, 9, 9]
policies: [batch, 82]
values: [-1, 1]
legal_masks: [batch, 82]
sample_weights > 0
```

### Rollback

마이그레이션은 원본 replay를 바로 지우지 않는다.

```text
replay/trajectory-replay.npz
replay-index/
shards/migrated-*/
```

전환 후 문제가 있으면 `LearnerV2Config.replay_backend`을 `monolithic`으로 되돌리고 기존 replay를 다시
사용한다.

마이그레이션 완료 후 충분히 검증되면 원본 replay 삭제는 별도 수동 단계로 둔다.

### Pruning과의 관계

migrated shard도 일반 indexed shard와 동일하게 active/evicted 상태를 따른다.

- active migrated shard는 삭제 금지.
- capacity eviction으로 evicted 된 migrated shard만 prune 가능.
- pseudo-shard 방식으로 등록한 큰 migrated shard는 capacity eviction 때 한 번에 빠질 수 있으므로
  가능하면 split migration을 사용한다.

## Observability

learner 로그에 다음 metric을 추가한다.

```text
indexed active shards
indexed active rows
pending shards
evicted shards
shard cache hit rate
import seconds
index commit seconds
train seconds
prune seconds
```

현재 병목 확인에는 다음 수치가 중요하다.

- shard import 완료까지 걸린 시간
- train 시작 전 대기 시간
- replay/index 디스크 사용량
- pending shard backlog

## 예상 효과

현재:

```text
import 1~N shards
-> full replay concatenate
-> full replay save: O(512k rows)
```

전환 후:

```text
import 1~N shards
-> index append: O(N shards)
-> optional evict old shards
```

학습 batch sampling은 shard load/cache 비용이 추가되지만, shard LRU cache와 row grouping으로 충분히 줄일 수
있다. 전체 replay를 매 cycle 저장하는 비용이 없어지는 효과가 훨씬 크다.

## 리스크

1. batch sampling이 여러 shard를 열면서 작은 random I/O가 늘 수 있다.
   - row를 shard별로 group하고 LRU cache를 둔다.
2. capacity가 shard 단위라 정확히 512000 rows에 맞지 않는다.
   - 약간의 slack을 허용한다.
3. pruning이 active shard를 지우면 replay가 깨진다.
   - active/evicted 상태를 분리하고 테스트한다.
4. 기존 diagnostics가 monolithic replay path를 전제로 한다.
   - shard_index용 diagnostics를 추가한다.
5. reanalyze/snapshot 기능은 처음에는 monolithic backend 전용으로 남을 수 있다.
   - async learner 학습 경로부터 전환한다.

## 구현 우선순위

1. `ShardReplayIndex`
2. `ShardIndexedTrajectoryDataset`
3. learner `replay_backend` flag
4. shard_index pruning
5. shard_index diagnostics
6. monolithic async path 제거 또는 fallback 유지
