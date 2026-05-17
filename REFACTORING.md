# Refactoring Plan

이 문서는 Great Kingdom AI 코드베이스를 `async v2 + trajectory replay` 중심으로 단순화하기
위한 리팩토링 계획이다. 기본 방침은 legacy 경로를 호환 유지하지 않고 제거하는 것이다.

## 방향

현재 권장 실행 경로는 Runpod에서 `great-kingdom-actor-v2`와
`great-kingdom-learner-v2`를 함께 돌리는 light async 구조다. 리팩토링의 목표는 이 경로를
명확한 주 경로로 만들고, 이전 실험 경로와 호환 계층을 걷어내는 것이다.

핵심 원칙:

- `async v2`를 제품 코드의 중심으로 둔다.
- `TrajectoryReplayStore`를 학습 replay의 유일한 저장 포맷으로 둔다.
- legacy `ReplayBuffer` 기반 학습/actor/learner 경로는 제거한다.
- migration 코드는 장기 유지하지 않는다. 필요한 산출물은 한 번 변환한 뒤 코드를 삭제한다.
- 큰 파일을 역할별 모듈로 나누되, 외부 CLI 동작은 새 경로 기준으로만 유지한다.
- 삭제 후 깨지는 테스트는 legacy 테스트라면 고치지 말고 제거한다.
- 시간 복잡도를 기존과 동일하게 혹은 더 낮게 유지해야한다.

## 제거 대상

### 1. Actor/Learner v1

제거 대상:

- `python/great_kingdom_ai/actor_learner_processes.py`
- `python/great_kingdom_ai/actor_learner_shards.py`
- `great-kingdom-actor`
- `great-kingdom-learner`
- `tests/test_actor_learner_processes.py`

남길 대상:

- `great-kingdom-actor-v2`
- `great-kingdom-learner-v2`
- `great-kingdom-init-async-v2`

v1 shard metadata와 v2 shard metadata를 통합하려고 시간을 쓰지 않는다. v2 쪽 metadata
구조만 새 모듈로 분리한다.

### 2. Legacy ReplayBuffer 학습 경로

제거 또는 축소 대상:

- `ReplayBuffer`를 직접 저장 포맷으로 쓰는 학습/실험 경로
- `TrajectoryReplayBuffer`의 legacy sample 호환 API
- `trajectory_targets.py`의 legacy replay sample conversion
- `load_training_replay()`의 legacy replay loading
- `tests/test_replay_buffer.py` 중 저장 포맷 보존 테스트
- `tests/test_trajectory_replay.py` 중 legacy compatibility 테스트

목표 상태:

- 학습 루프는 `ReplayDataset` protocol만 의존한다.
- production replay dataset은 `TrajectoryReplayDataset` 하나다.
- 테스트에서 작은 batch를 만들 필요가 있으면 별도 test fixture dataset을 둔다.
- `ReplaySample`은 내부 batch fixture 또는 transition 변환 중간 타입으로만 남기거나 제거한다.

### 3. Migration/one-off 스크립트

제거 대상:

- `checkpoint_migration.py`
- `trajectory_replay_migration.py`
- 이미 끝난 실험용 replay 변환 스크립트
- 해당 migration 테스트

예외:

- 현재 Runpod 산출물을 실제로 한 번 변환해야 한다면 `scripts/one_off/` 아래로 옮긴 뒤
  변환 완료 후 삭제한다.

### 4. 단일 파이프라인 경로

정리 대상:

- `train_v2_pipeline.py`
- `great-kingdom-train-v2`
- 관련 단일 파이프라인 테스트

판단 기준:

- async v2에서 같은 기능을 수행할 수 있으면 삭제한다.
- smoke 용도만 남아 있으면 `scripts/run_m*_smoke.py`에서 async v2 함수 조합으로 대체한다.
- arena 자동 promote 같은 과거 정책은 복구하지 않는다. snapshot 간 비교는 별도 평가 명령으로 둔다.

### 5. Aggregate/ablation 실험 코드

정리 대상:

- `online_aggregate_replay.py`
- `replay_aggregate.py`
- `rust_onnx_replay.py`
- aggregate replay ablation scripts
- fixed replay hparam ablation scripts

남길 수 있는 것:

- 현재 의사결정에 쓰는 진단 스크립트만 남긴다.
- 남기는 스크립트도 `TrajectoryReplayStore`만 입력으로 받게 바꾼다.

## 새 구조

목표 패키지 구조:

```text
python/great_kingdom_ai/
  async_v2/
    __init__.py
    config.py
    paths.py
    metadata.py
    actor.py
    learner.py
    factory.py
    cli.py
  training/
    __init__.py
    config.py
    batch.py
    loop.py
    checkpoint.py
    scheduler.py
    sampling.py
  replay/
    __init__.py
    trajectory.py
    dataset.py
    schema.py
    persistence.py
  reanalyze/
    __init__.py
    config.py
    evaluator.py
    targets.py
    on_sample.py
    search.py
  arena/
    __init__.py
    config.py
    runner.py
    batch.py
    report.py
  self_play/
    __init__.py
    config.py
    runner.py
    batch.py
    logs.py
```

기존 public import를 모두 유지하려고 하지 않는다. CLI entrypoint와 현재 테스트에서 필요한
새 public API만 명시적으로 유지한다.

## 단계별 계획

### Phase 0: 기준선 고정

- 현재 `python -m pytest`, `python -m ruff check .`, `python -m mypy` 상태를 확인한다.
- 리팩토링 중 유지할 핵심 테스트 목록을 정한다.
- legacy 테스트 목록을 별도로 적고 제거 대상으로 표시한다.

핵심 유지 테스트:

- async v2 actor/learner
- trajectory replay store/dataset
- train loop/checkpoint
- ONNX export/evaluator
- Rust Gumbel/self-play integration
- arena evaluation

### Phase 1: async v2 모듈 분리

현재 `actor_learner_v2.py`를 다음 단위로 나눈다.

- config dataclass와 JSON loader
- path 계산
- shard metadata append/load/pending
- actor one-cycle runner
- learner one-cycle runner
- factory init
- CLI parser/main

완료 기준:

- `great-kingdom-actor-v2`, `great-kingdom-learner-v2`, `great-kingdom-init-async-v2`가 새 모듈을 사용한다.
- `actor_learner_v2.py`는 compatibility wrapper로 남기지 않는다. entrypoint를 새 모듈로 직접 바꾼다.
- v1 actor/learner entrypoint는 이 단계 끝에서 삭제한다.

### Phase 2: replay 포맷 단일화

`TrajectoryReplayStore`를 `replay/` 패키지로 옮기고 저장 포맷을 명시한다.

작업:

- schema와 persistence를 분리한다.
- `TrajectoryReplayDataset`을 production 학습 dataset으로 고정한다.
- `TrajectoryReplayBuffer`와 legacy sample view를 제거한다.
- `ReplayBuffer.load()`를 production 코드에서 제거한다.

완료 기준:

- async v2 learner가 `TrajectoryReplayDataset`만 사용한다.
- reanalyze와 diagnostics가 `TrajectoryReplayStore`만 입력으로 받는다.
- legacy replay compatibility 테스트가 삭제된다.

### Phase 3: training 패키지 분리

현재 `train.py`를 training package로 나눈다.

작업:

- config parsing과 CLI를 loop에서 분리한다.
- checkpoint save/load/summary를 독립 모듈로 둔다.
- batch conversion, augmentation, priority sampling, prefetch를 분리한다.
- `load_training_replay()` legacy helper를 제거한다.

완료 기준:

- 학습 루프는 `ReplayDataset` protocol과 `TrainingConfig`만 받는다.
- checkpoint 관련 테스트와 train loop 테스트가 파일 역할별로 나뉜다.
- CLI는 trajectory replay 경로만 받는다.

### Phase 4: reanalyze public API 정리

현재 `on_sample_reanalyze.py`가 `reanalyze.py`의 private 함수에 의존한다. 이 의존을 제거한다.

작업:

- evaluator 생성/호출 API를 public 모듈로 분리한다.
- bootstrap target 계산을 public target service로 분리한다.
- on-sample/offline/search reanalyze가 같은 public API를 사용한다.
- legacy `TrajectoryReplayBuffer` 입력을 제거한다.

완료 기준:

- `from great_kingdom_ai.reanalyze import _...` 형태의 private cross-module import가 없다.
- reanalyze 테스트는 trajectory store 기준으로만 남는다.

### Phase 5: self-play/arena 공통 경계 정리

작업:

- Rust core adapter protocol을 한 곳으로 모은다.
- batch eval request 처리와 action selection을 공통 helper로 둔다.
- self-play runner와 arena runner는 각각 orchestration만 담당한다.

완료 기준:

- `self_play.py`와 `evaluate.py`가 500줄 이하로 줄어든다.
- evaluator 호출 경로가 중복 구현되지 않는다.

### Phase 6: Rust Gumbel search 정리

Python 경계가 안정된 뒤 Rust를 정리한다.

작업:

- PyO3 binding 함수와 순수 search 로직을 분리한다.
- `search.rs`, `batch.rs`, `arena_batch.rs`에서 공통 leaf evaluation flow를 추출한다.
- profiling/debug 구조를 search 로직에서 분리한다.

완료 기준:

- Python API surface는 async v2/self-play/arena에서 실제 쓰는 것만 남는다.
- Rust unit/integration 테스트가 Gumbel search 핵심 동작을 커버한다.

## 삭제 순서

삭제는 한 번에 크게 하지 않고, 각 phase 끝에서 작게 자른다.

1. pyproject entrypoint에서 v1 actor/learner 제거
2. v1 actor/learner tests 제거
3. v1 actor/learner modules 제거
4. legacy replay compatibility tests 제거
5. `TrajectoryReplayBuffer` 제거
6. production code의 `ReplayBuffer` 저장/로드 제거
7. migration modules 제거
8. aggregate/ablation scripts 제거 또는 trajectory-only로 축소
9. 단일 train v2 pipeline 제거

## 테스트 전략

리팩토링 중 매 단계에서 최소 검증:

```bash
python -m pytest tests/test_actor_learner_v2.py
python -m pytest tests/test_train.py tests/test_trajectory_dataset.py
python -m pytest tests/test_trajectory_replay.py tests/test_reanalyze.py
python -m ruff check .
python -m mypy
```

Rust 경로를 건드린 단계:

```bash
python -m pytest tests/test_gumbel.py tests/test_rust_onnx_self_play.py tests/test_evaluate.py
```

Runpod 전 최종 검증:

```bash
python -m pytest
python -m ruff check .
python -m mypy
```

## 하지 않을 일

- 예전 replay 파일을 계속 읽기 위한 compatibility layer 유지
- v1 actor/learner와 v2 actor/learner의 metadata 통합
- 모든 옛 CLI 이름 보존
- 오래된 ablation script의 동작 보존
- docs 중심의 문서 정비

## 최종 상태

리팩토링 완료 후 사용자가 알아야 할 주 경로는 다음 네 가지면 충분해야 한다.

- `great-kingdom-init-async-v2`
- `great-kingdom-actor-v2`
- `great-kingdom-learner-v2`
- `great-kingdom-evaluate`

나머지는 모델 export, replay monitor, 진단 도구처럼 현재 운영에 필요한 보조 명령만 남긴다.
