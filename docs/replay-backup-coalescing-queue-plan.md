# Replay Backup Coalescing Queue Plan

## 배경

현재 learner v2의 local replay 백업은 `_ReplayBackupManager.start_if_idle()`에서 실행 중인 백업이 있으면 새 백업 요청을 시작하지 않고 `replay backup already running`만 출력한다.

이 동작은 데이터가 즉시 디스크 백업으로 반영되지 않는 것처럼 보이고, 특히 `max_cycles` 종료나 프로세스 종료 시점에는 기존 백업의 `future.result()`를 기다리면서 멈춘 것처럼 보일 수 있다.

현재 구조의 실제 의미는 다음과 같다.

- 백업 executor는 `max_workers=1`이라 한 번에 하나만 실행된다.
- 백업 중 새 요청이 오면 스냅샷을 만들지 않고 반환한다.
- staged shard는 메모리에 남아 다음 cycle에서 재시도된다.
- `close()`는 실행 중인 백업 완료를 기다린다.

즉, 완전한 queue가 아니라 "다음 cycle 재시도"에 의존하는 구조다.

## 목표

`_ReplayBackupManager`를 1개짜리 coalescing queue 구조로 바꾼다.

백업 실행 중 새 요청이 들어오면 즉시 버리지 않고 pending slot 하나에 최신 백업 요청을 저장한다. 현재 백업이 끝나면 manager가 pending slot을 즉시 다음 백업으로 시작한다.

## 설계 원칙

- 백업은 계속 한 번에 하나만 실행한다.
- pending backup은 여러 개 쌓지 않고 최신 상태 하나로 병합한다.
- replay 파일 백업은 최신 local replay snapshot 하나만 필요하다.
- metadata/game log append는 shard 단위 이벤트를 중복 없이 포함해야 한다.
- 기존 staged shard 제거 흐름과 충돌하지 않아야 한다.
- 종료 시에는 running backup과 queued backup을 모두 처리하거나, 명시적으로 어떤 요청이 남았는지 알 수 있어야 한다.

## 제안 구조

`_ReplayBackupManager`에 다음 상태를 둔다.

- `_future`: 현재 실행 중인 backup future
- `_request`: 현재 실행 중인 backup request
- `_queued_request`: 실행 대기 중인 최신 backup request

`start_if_idle()`는 `start_or_queue()`로 바꾸는 것이 의미상 명확하다.

동작:

1. `pending`이 비어 있으면 아무 것도 하지 않는다.
2. 현재 실행 중인 백업이 없으면 snapshot을 만들고 즉시 실행한다.
3. 현재 실행 중인 백업이 있으면 snapshot을 만들고 `_queued_request`에 저장한다.
4. 기존 `_queued_request`가 있으면 새 요청으로 교체한다.
5. 교체되는 queued snapshot 파일은 unlink해서 임시 파일 누수를 막는다.
6. `poll()`에서 running backup이 끝난 것을 확인하면 완료 request를 반환하고, queued request가 있으면 즉시 다음 future로 submit한다.

## 병합 기준

coalescing queue는 "가장 최신 replay 파일 상태"를 우선한다.

새 queued request는 기존 queued request를 완전히 대체한다.

그 이유:

- active local replay는 append/import 후 저장된 전체 replay snapshot이다.
- network backup file은 최종 replay 파일 하나만 있으면 된다.
- staged shard metadata/game log는 새 request의 `pending`과 `imported_events`에 현재 staged 전체가 들어가므로 기존 queued request보다 더 완전하다.

주의할 점:

- running request에 포함된 shard는 완료 후 `_remove_backed_up_staged_shards()`가 제거한다.
- queued request가 running request와 일부 shard를 중복 포함할 수 있다.
- 따라서 poll 완료 후 queued request를 시작하기 전에, 가능하면 caller가 staged 상태를 정리한 뒤 다시 queue하도록 할지 결정해야 한다.

권장안은 manager 내부에서는 queued request를 그대로 실행하고, metadata append 중복을 허용하지 않도록 더 높은 레벨에서 staged set을 정리하는 것이다. 그러나 현재 `poll()`이 완료 request를 반환한 뒤 caller가 staged를 정리하는 구조라, manager가 poll 안에서 즉시 queued request를 시작하면 stale queued request가 중복 이벤트를 쓸 수 있다.

따라서 안정적인 구현은 다음 쪽이 낫다.

- manager는 running 완료 시 queued request를 자동 실행하지 않는다.
- `poll()`은 완료 request를 반환하고 `_queued_request` 존재 여부를 유지한다.
- caller가 completed request로 staged를 정리한 뒤 `start_queued_if_idle()`을 호출한다.
- `start_queued_if_idle()`은 queued request의 shard 중 아직 staged에 남은 shard만 필터링해서 실행한다.

하지만 이 방식은 manager와 caller 사이의 계약이 복잡해진다.

더 단순한 권장 구현:

- queue에는 request 전체가 아니라 "queue requested" 플래그만 둔다.
- running 중 새 요청이 오면 `_queued = True`만 표시하고 로그를 `replay backup queued`로 바꾼다.
- running 완료 후 caller가 staged를 정리한다.
- staged가 남아 있으면 기존 코드 경로가 즉시 `start_or_queue()`를 호출해서 최신 local replay로 새 백업을 시작한다.

이 방식은 임시 snapshot을 미리 만들지 않기 때문에 stale snapshot, 중복 metadata append, temp 파일 누수 문제가 없다. 또한 현재 staged 기반 흐름과 가장 잘 맞는다.

## 구현 계획

1. `_ReplayBackupManager` 상태 변경

   - `_queued: bool = False` 추가
   - `start_if_idle()`를 `start_or_queue()`로 변경
   - 실행 중이면 `_queued = True`로 설정하고 `replay backup queued` 출력
   - 실행 중이 아니면 기존처럼 snapshot 생성 후 submit

2. `poll()` 반환값 확장

   - 현재는 완료된 `_ReplayBackupRequest | None`만 반환한다.
   - queued 여부를 caller가 알 수 있게 `completed`와 `queued`를 담은 작은 dataclass를 추가한다.
   - 또는 더 단순히 `has_queued_request()` 메서드를 추가한다.

3. cycle 시작부 흐름 변경

   - `completed_backup = backup_manager.poll(printer)`
   - 완료된 shard를 staged에서 제거한다.
   - staged가 남아 있고 `backup_manager.has_queued_request()`가 true이면 queued flag를 clear하고 최신 replay를 저장한 뒤 `start_or_queue()`를 호출한다.
   - 이때 `start_or_queue()`는 이미 idle 상태이므로 즉시 backup을 시작한다.

4. 기존 호출부 rename

   - `backup_manager.start_if_idle(...)` 호출을 모두 `backup_manager.start_or_queue(...)`로 변경한다.
   - 로그 문구를 다음처럼 정리한다.
     - 즉시 시작: `started replay backup: shards=..., source=..., path=...`
     - 실행 중 queue: `queued replay backup: staged_shards=...`
     - 완료: `replay backup complete: shards=..., path=...`

5. 종료 처리

   - `close()`는 running backup을 기다린다.
   - `_queued`가 true이면 마지막 staged 상태를 반영하지 못했음을 로그로 남길 수 있어야 한다.
   - 더 안전하게 하려면 `close()` 전에 caller가 staged가 남아 있고 queued가 true인 경우, running 완료 후 한 번 더 local replay 저장 및 backup을 수행한다.
   - `max_cycles` 테스트에서는 queued backup까지 처리할지, running만 기다릴지 정책을 명확히 정해야 한다.

권장 종료 정책:

- continuous learner의 정상 종료에서는 queued backup까지 flush한다.
- 강제 종료는 기존과 동일하게 보장하지 않는다.

## 테스트 계획

추가할 테스트:

1. running 중 새 요청이 오면 `replay backup already running`이 아니라 queue 상태가 된다.
2. running backup 완료 후 staged에서 완료 shard를 제거하고, 남은 staged shard만 다음 backup으로 저장한다.
3. queued 상태에서 여러 번 요청해도 backup은 하나만 추가 실행된다.
4. queued backup이 실행될 때 최신 local replay snapshot을 사용한다.
5. `close()` 또는 loop 종료 시 queued backup flush 정책이 지켜진다.

기존 테스트 영향:

- `test_learner_v2_loop_uses_local_replay_and_async_backup`는 기존 즉시 백업 동작을 유지해야 한다.
- async backup 관련 새 테스트는 `copy_file_atomic`을 지연시키는 fake future 또는 blocking hook으로 running 상태를 재현하는 방식이 좋다.

## 리스크

- metadata/game log append 중복이 가장 큰 리스크다.
- queued request를 실제 request 객체로 저장하면 stale staged list 때문에 중복 이벤트가 생길 수 있다.
- 그래서 queued request 객체 대신 boolean queue flag를 쓰는 방식이 더 안전하다.
- local replay snapshot은 queue 시점이 아니라 실제 시작 시점에 만들어야 최신 상태를 담을 수 있다.

## 완료 기준

- 백업 실행 중 새 백업 필요가 생기면 로그가 `queued replay backup`으로 남는다.
- running backup 완료 후 staged에 남은 shard가 있으면 다음 backup이 자동으로 이어진다.
- metadata와 game log에 동일 shard import가 중복 기록되지 않는다.
- loop 종료 시 running backup 때문에 멈춘 것처럼 보이는 상황이 로그상 명확해진다.
- 관련 unit test가 추가되고 통과한다.
