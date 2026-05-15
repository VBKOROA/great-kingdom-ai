# Replay Buffer 구조 개선안

작성일: 2026-05-15

## 배경

현재 async v2 구조는 actor가 self-play shard를 만들고 learner가 shard를 import한다.

```text
actor
-> shards/<shard-id>/trajectory-replay.npz
-> learner import
-> replay/trajectory-replay.npz
-> train
```

기존 monolithic replay 방식은 training 단계에서는 빠른 편이다. 학습 루프가 이미 메모리에 올라온
큰 NumPy 배열에서 `features[index_array]`, `policy_targets[index_array]`처럼 batch를 만들기
때문이다.

문제는 shard import 이후 매 cycle마다 `replay/trajectory-replay.npz` 전체를 다시 저장한다는 점이다.
replay가 capacity 근처까지 커지면 몇 천 rows를 추가하기 위해 수 GB 단위 배열을 다시 쓰게 된다.

이 문제를 줄이려고 `docs/async-shard-index-replay-plan.md` 방식의 shard/index replay를 검토했지만,
실제 training 단계에서는 monolith보다 IO 병목이 더 심해질 수 있다. actor shard를 training dataset의
직접 backend로 쓰면 random sampling이 shard 파일 랜덤 접근으로 바뀌기 때문이다.

## 목표

1. actor shard import 후 전체 replay 파일을 매번 재저장하지 않는다.
2. training batch sampling은 monolith replay처럼 큰 연속 배열에서 빠르게 수행한다.
3. imported actor shard는 commit 이후 삭제할 수 있게 한다.
4. 기존 `train_from_replay()`와 `sample_arrays()` 호출 구조를 최대한 유지한다.
5. replay 분석 스크립트와 monitor가 필요한 필드를 보존한다.
6. 사무용 노트북 CPU 환경에서도 단위 테스트와 작은 smoke test가 가능해야 한다.

## 이전 shard/index 계획의 단점

`docs/async-shard-index-replay-plan.md`의 핵심 아이디어는 actor shard를 그대로 보존하고 learner가
index만 관리하는 것이다.

```text
pending shard
-> validate shard file
-> append index record
-> mark shard imported
-> train from indexed replay dataset
```

이 방식은 import 비용은 줄일 수 있지만, training path에는 다음 단점이 있다.

### 1. Batch sampling이 파일 랜덤 접근이 된다

현재 학습은 row 단위 random sampling이다. batch size가 256이면 한 batch 안의 row들이 여러 shard에
흩어진다. shard/index dataset은 다음 작업을 매 batch 반복해야 한다.

```text
global row index sample
-> shard lookup
-> shard별 local row group
-> shard npz load/cache
-> shard 안에서 gather
-> batch concatenate
```

LRU cache를 둬도 replay가 커지고 shard 수가 많아질수록 cache miss가 생긴다. 이 경우 GPU 학습 step이
CPU 파일 IO를 기다리게 된다.

### 2. `.npz` shard는 mmap-friendly storage가 아니다

`.npz`는 zip container다. 작은 파일을 여러 개 여는 비용이 있고, 압축이 꺼져 있어도 random row access에
최적화된 layout이 아니다. shard를 dataset backend로 쓰면 파일 open, zip metadata 처리, 배열 복원 비용이
training loop 안으로 들어온다.

### 3. Import artifact와 training store의 역할이 섞인다

actor shard는 원래 process 간 전달 단위다. learner가 한 번 import한 뒤에는 training에 필요한 durable
store가 따로 있으면 shard는 삭제해도 된다. shard/index 계획은 actor shard를 계속 training store로
재사용하므로, import artifact와 learner replay의 책임이 분리되지 않는다.

### 4. Recency sampling은 되지만 locality가 나쁘다

`recent_sample_window`를 global row range로 유지하는 것은 가능하다. 하지만 recent window 안에도 여러
shard가 섞이면 batch마다 다수 shard를 읽는다. monolith 배열 slicing보다 locality가 좋지 않다.

### 5. Priority replay 확장이 복잡해진다

priority를 row 단위로 갱신하려면 shard별 sidecar, index-level priority table, SQLite 같은 별도 구조가
필요하다. shard 파일을 보존하는 것만으로는 학습 중 priority update가 단순해지지 않는다.

## 제안: Actor shard는 1회용, learner replay는 array ring store

권장 구조는 actor shard와 learner replay store를 분리하는 것이다.

```text
data/runpod/train-v2-gumbel-512k/
  shards/
    metadata.jsonl
    <shard-id>/
      trajectory-replay.npz
      game_logs.json

  replay-store/
    state.json
    commit.jsonl
    features.npy
    policy_targets.npy
    legal_masks.npy
    players.npy
    winners.npy
    episode_ids.npy
    episode_offsets.npy
    timesteps.npy
    actions.npy
    terminals.npy
    model_versions.npy
    created_iterations.npy
    sample_weights.npy
    root_policy_logits.npy
    root_policy_logits_present.npy

  replay/
    game_logs.jsonl
```

흐름은 다음과 같다.

```text
pending actor shard
-> load shard once
-> append rows to replay-store array ring
-> append game logs to replay/game_logs.jsonl
-> fsync or atomic commit state
-> metadata에 shard_imported 기록
-> actor shard directory delete
-> train from replay-store arrays
```

중요한 점은 training이 `shards/`를 보지 않는다는 것이다. training은 항상 `replay-store/*.npy`의 큰 배열
view에서 batch를 만든다.

## Actor shard 삭제 조건

actor shard는 다음 조건이 모두 만족된 뒤 삭제한다.

1. shard의 transition rows가 replay-store에 append되었다.
2. replay-store의 `state.json` 또는 commit log가 원자적으로 갱신되었다.
3. `game_logs.json` 내용이 `replay/game_logs.jsonl`에 append되었다.
4. `shards/metadata.jsonl`에 `shard_imported` 이벤트가 기록되었다.
5. learner가 해당 shard를 다시 import하지 않도록 recovery 기준이 명확하다.

삭제는 학습 성공 이후가 아니라 import commit 이후에 해도 된다. training이 실패해도 replay-store에 이미
commit된 데이터는 다음 learner cycle에서 재사용할 수 있어야 한다.

## 저장 필드

### 학습에 필요한 필드

- `features`
- `policy_targets`
- `legal_masks`
- `players`
- `episode_winners` 또는 row별 value target 계산에 필요한 winner mapping
- `episode_offsets`
- `sample_weights`

### 분석과 monitor를 위해 보존할 필드

- `timesteps`
- `actions`
- `terminals`
- `model_versions`
- `created_iterations`
- `root_policy_logits`
- `root_policy_logits_present`

현재 replay monitor는 `root_policy_logits`가 없으면 critical alert를 낸다. 따라서 `next_features`는
버리더라도 `root_policy_logits`는 보존하는 편이 낫다.

### 버려도 되는 후보

- `next_features`

현재 async import path에서도 `next_features`는 drop하고 있다. training과 현재 분석 스크립트 기준으로는
필수 입력이 아니다. 단, 향후 dynamics/consistency loss 또는 transition model 계열 실험을 하려면 다시
필요할 수 있다.

## Ring buffer 설계

Replay capacity는 transition row 기준으로 유지한다.

```json
{
  "capacity": 512000,
  "size": 510234,
  "write_index": 348120,
  "generation": 42,
  "schema_version": 1,
  "updated_at": "2026-05-15T00:00:00+00:00"
}
```

배열은 capacity 크기로 preallocate한다.

```text
features.npy                 [capacity, 11, 9, 9] float32
policy_targets.npy           [capacity, 82] float32
legal_masks.npy              [capacity, 82] bool
players.npy                  [capacity] int64
sample_weights.npy           [capacity] float32
root_policy_logits.npy       [capacity, 82] float32
root_policy_logits_present.npy [capacity] bool
```

append 시에는 `write_index`부터 row를 쓴다. 끝을 넘어가면 앞쪽으로 wrap한다.

```text
write [write_index, capacity)
write [0, remaining)
update write_index
update size
```

training sampling은 logical row index를 physical row index로 변환한 뒤 큰 배열에서 gather한다.

```text
logical 0 = oldest active row
logical size - 1 = newest active row
physical = (write_index - size + logical) % capacity
```

이렇게 하면 recency sampling도 logical row 기준으로 유지된다.

## Episode boundary 처리

가장 안전한 방식은 episode 단위 append와 episode 단위 eviction이다. 다만 ring buffer에서는 구현이 조금
복잡하다.

첫 단계에서는 다음 중 하나를 선택한다.

### 선택 A: transition ring + row별 value target 저장

import 시점에 terminal value target을 계산해서 `values.npy`에 저장한다. 그러면 training은 episode boundary를
몰라도 된다.

장점:

- sampling이 단순하다.
- ring overwrite가 쉽다.
- training dataset이 `features`, `policies`, `values`, `legal_masks`, `sample_weights`만으로 동작한다.

단점:

- value target 정의를 바꾸면 replay-store를 재생성해야 한다.
- trajectory 기반 reanalyze에서 episode context가 필요하면 별도 metadata가 필요하다.

### 선택 B: episode metadata 유지

`episode_ids`, `episode_offsets`, `episode_winners`를 유지하고 sampling 시 value target을 계산한다.

장점:

- 기존 `TrajectoryReplayDataset` 의미에 가깝다.
- trajectory 분석 정보가 더 풍부하다.

단점:

- ring overwrite 시 오래된 episode metadata 정리가 필요하다.
- 구현과 recovery 테스트가 더 복잡하다.

현재 async learner가 terminal outcome value target을 쓰는 기준이라면 선택 A가 더 단순하다. 분석에는
row별 `values`와 별도 요약 metadata를 제공하면 충분하다.

## 분석 스크립트 호환

현재 분석 도구는 대체로 `replay/trajectory-replay.npz` 또는 `TrajectoryReplayStore.load(path)`를 직접
사용한다.

- `scripts/diagnose_async_replay_value.py`
- `scripts/diagnose_async_update_pressure.py`
- `scripts/diagnose_trajectory_policy_targets.py`
- `great-kingdom-replay-monitor-v2`

array ring store로 바꾸면 다음 중 하나가 필요하다.

### 권장: 공통 ReplayView 인터페이스 추가

```python
class ReplayArrayView:
    capacity: int
    size: int
    features: np.ndarray
    policy_targets: np.ndarray
    legal_masks: np.ndarray
    values: np.ndarray
    sample_weights: np.ndarray
    root_policy_logits: np.ndarray | None
    root_policy_logits_present: np.ndarray | None
```

training dataset과 분석 스크립트가 `TrajectoryReplayStore` 대신 이 view를 받도록 바꾼다.

### 임시: export command 제공

운영 중 replay-store는 `.npy` ring으로 유지하고, 분석이 필요할 때만 snapshot `.npz`를 만든다.

```text
great-kingdom-export-replay-snapshot \
  --store replay-store \
  --output replay/trajectory-replay-snapshot.npz
```

장점은 기존 분석 스크립트 변경이 작다는 것이다. 단점은 snapshot 생성 시 다시 큰 파일 write가 발생한다.
따라서 상시 monitor에는 부적합하고, 수동 분석용으로만 둔다.

## Crash recovery

commit 순서는 다음처럼 둔다.

```text
1. shard load
2. replay-store 배열 write
3. commit.jsonl에 pending commit 기록
4. state.json.tmp write
5. state.json atomic replace
6. metadata.jsonl에 shard_imported 기록
7. game_logs append
8. shard delete
```

더 보수적으로는 game log append를 6번보다 앞에 둔다. 중요한 것은 재시작 시 다음이 가능해야 한다는 점이다.

- `metadata`에는 imported인데 replay-store commit이 없으면 오류로 보고 중단한다.
- replay-store commit은 있는데 metadata imported가 없으면 metadata를 복구하거나 shard를 재import하지 않는다.
- shard directory가 이미 삭제되어도 replay-store commit이 있으면 정상 상태다.

## 단계별 전환안

### Phase 1: monolith 저장 빈도 줄이기

가장 작은 변경이다.

- replay는 메모리에 유지한다.
- shard import event log를 남긴다.
- `trajectory-replay.npz` 전체 save는 매 cycle이 아니라 N개 shard 또는 N분마다 수행한다.
- imported shard는 commit 후 삭제한다.

장점:

- training 속도는 현재 monolith와 같다.
- 구현 위험이 낮다.

단점:

- crash recovery를 위해 journal replay가 필요하다.
- replay가 커질수록 주기적 snapshot 비용은 남는다.

### Phase 2: array ring store 도입

새 `ArrayRingReplayStore`를 추가한다.

- `.npy` 파일 preallocate
- shard import append
- `sample_arrays()` 구현
- recency sampling 유지
- root policy logits 보존
- 분석용 view 또는 snapshot export 추가

이 단계가 장기적으로 권장되는 구조다.

### Phase 3: priority replay 확장

priority를 별도 배열로 저장한다.

```text
priorities.npy [capacity] float32
priority_updated_at.npy [capacity] int64
```

학습 후 TD error나 policy KL 기반 priority update를 row index 기준으로 반영한다.

## 테스트 계획

CPU-only 환경에서 가능한 테스트를 우선 만든다.

1. 작은 capacity ring append/wrap 테스트
2. shard import 후 logical order 유지 테스트
3. recency sampling이 newest logical rows를 보는지 테스트
4. root_policy_logits 보존 테스트
5. imported shard 삭제 후 replay-store만으로 train batch 생성 테스트
6. interrupted commit recovery 테스트
7. snapshot export가 기존 diagnostics 입력 schema와 맞는지 테스트

## 결론

actor shard는 import artifact로 취급하고, commit 이후 삭제하는 것이 맞다.

다만 shard를 training dataset backend로 직접 쓰면 training 단계 random IO가 커진다. 따라서 replay 구조는
다음 원칙으로 바꾸는 것이 좋다.

```text
actor shard = 1회용 전달 파일
learner replay-store = 학습과 분석의 source of truth
training sampler = 큰 연속 배열에서 batch gather
```

단기적으로는 monolith replay의 저장 주기를 줄이고, 장기적으로는 array ring replay-store로 전환한다.
