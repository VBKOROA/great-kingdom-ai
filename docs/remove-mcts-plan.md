# MCTS 제거 계획

## 목표

기존 PUCT MCTS 백엔드와 MCTS 전용 설정을 제거하고, self-play / arena / pipeline의 검색 백엔드를 Gumbel 기반으로 단순화한다.

제거 후에도 다음은 유지한다.

- Rust rules engine, feature extraction, game state Python binding
- Gumbel search / Gumbel batched self-play
- Python self-play, pipeline, arena 평가 흐름
- replay sample 정책 타깃 생성 및 학습 데이터 포맷

## 현재 의존성 요약

MCTS 관련 코드는 다음 영역에 걸쳐 있다.

- Rust core
  - `rust/great_kingdom_core/src/mcts.rs`
  - `rust/great_kingdom_core/src/lib.rs`의 `mod mcts`, `Mcts*` export, Python module 등록
  - Gumbel 코드가 `mcts::EvalRequest`를 재사용 중
- Python self-play / pipeline / arena
  - `python/great_kingdom_ai/self_play.py`
  - `python/great_kingdom_ai/pipeline.py`
  - `python/great_kingdom_ai/evaluate.py`
  - `python/great_kingdom_ai/evaluator.py` 문서 문자열
- 설정
  - `configs/test/pipeline-*.json`
  - `configs/runpod/pipeline-runpod.json`
  - `configs/runpod/gumbel/pipeline-runpod.json`
  - `configs/test/arena-test.json`, `configs/runpod/arena-runpod.json` 계열 확인 필요
- 테스트
  - `tests/test_mcts.py`
  - `tests/test_self_play.py`의 MCTS 이름/분기 테스트
  - `tests/test_pipeline.py`, `tests/test_evaluate.py`의 `mcts_*`, `search_backend` 기대값
- 문서
  - `README.md`
  - `rust/great_kingdom_core/pyproject.toml` description

## 원칙

- 한 번에 `mcts.rs`부터 삭제하지 않는다. Gumbel이 쓰는 공용 평가 요청 타입을 먼저 분리한다.
- 외부 설정의 호환성보다 코드 단순화를 우선한다. 단, 제거 중간 커밋은 테스트 가능한 상태를 유지한다.
- 파일이 비대해지면 기능 단위로 분리한다. 특히 `self_play.py`는 MCTS 제거 과정에서 Gumbel 전용 self-play로 이름과 책임을 정리한다.
- 로컬 노트북에는 GPU가 없으므로 검증은 CPU pytest와 Rust unit test 중심으로 한다.

## 단계별 작업

### 1. Rust 공용 평가 요청 타입 분리

목표: `EvalRequest`를 `mcts.rs` 밖으로 이동해서 Gumbel이 MCTS 모듈에 의존하지 않게 한다.

작업:

- `rust/great_kingdom_core/src/eval_request.rs` 추가
- `EvalRequest`와 관련 helper를 `mcts.rs`에서 새 파일로 이동
- `rust/great_kingdom_core/src/lib.rs`에 `mod eval_request;` 추가
- `pub use eval_request::EvalRequest;`로 export 변경
- `gumbel/search.rs`, `gumbel/batch.rs`의 `mcts::EvalRequest` import를 `eval_request::EvalRequest` 또는 crate export로 변경

검증:

- `cargo test` 또는 최소 `cargo test gumbel`
- `pytest tests/test_gumbel.py`

### 2. Python 검색 백엔드 기본값을 Gumbel로 고정

목표: 사용자 설정과 런타임 분기에서 `mcts` 선택지를 제거한다.

작업:

- `MctsSelfPlayConfig` 이름 변경 검토
  - 후보: `SelfPlayConfig`
  - 이 단계에서 이름 변경이 크면 alias를 잠시 두고 다음 단계에서 제거
- `search_backend` 필드를 제거하거나 `"gumbel"`만 허용
- `create_core_mcts_search`, `create_core_mcts_self_play_batch` 제거
- `create_core_search_backend`는 `core.GumbelSearch`만 생성하도록 단순화
- `create_core_self_play_batch_backend`는 `core.GumbelSelfPlayBatch`만 생성하도록 단순화
- `_run_self_play_search`의 MCTS 분기 제거
- `_should_apply_root_noise`에서 MCTS 조건 제거
  - Gumbel에서 root noise를 더 이상 쓰지 않는다면 설정 자체도 제거
  - 유지하려면 Gumbel root prior 처리에 명확히 연결

검증:

- `pytest tests/test_self_play.py`
- `pytest tests/test_pipeline.py tests/test_evaluate.py`

### 3. Pipeline / Arena 설정 스키마 정리

목표: `mcts_*` 설정 키와 CLI 인자를 제거하고 Gumbel 설정만 남긴다.

작업:

- `PipelineConfig`
  - `search_backend` 제거
  - `mcts_simulations` 제거
  - `mcts_c_puct` 제거
  - `playout_cap_full_simulations` 입력은 `gumbel_simulations` 또는 별도 `full_search_simulations`로 정리
- `ArenaConfig`
  - `search_backend` 제거
  - `simulations`, `c_puct` 중 MCTS 의미의 필드 제거 또는 Gumbel 의미로 이름 변경
- CLI
  - `--search-backend` 제거
  - `--mcts-simulations` 제거
  - help text의 MCTS 표현 제거
- JSON config
  - `configs/test/pipeline-test.json`
  - `configs/test/pipeline-smoke.json`
  - `configs/runpod/pipeline-runpod.json`
  - `configs/runpod/gumbel/*` 구조 통합 검토
- 설정 디렉터리 정리
  - Gumbel이 유일한 백엔드가 되면 `configs/*/gumbel/` 하위 디렉터리는 중복이다.
  - 기존 root config를 Gumbel config로 갱신하고, 중복 gumbel 디렉터리를 삭제하는 방향을 권장한다.

검증:

- `pytest tests/test_pipeline.py`
- `pytest tests/test_evaluate.py`
- `python -m great_kingdom_ai.pipeline --config configs/test/pipeline-smoke.json` 형태의 smoke 실행 가능 여부 확인

### 4. Rust MCTS 구현과 Python binding 제거

목표: MCTS Rust 구현과 공개 API를 삭제한다.

작업:

- `rust/great_kingdom_core/src/mcts.rs` 삭제
- `rust/great_kingdom_core/src/lib.rs`
  - `mod mcts;` 제거
  - `MctsConfig`, `MctsResult`, `MctsSearch`, `MctsSelfPlayBatch` export 제거
  - Python module의 `Mcts*` class 등록 제거
- Gumbel이 MCTS 타입 이름을 참조하지 않는지 재확인
- `rust/great_kingdom_core/pyproject.toml` description에서 MCTS 제거

검증:

- `cargo test`
- `maturin develop` 후 Python 테스트

### 5. 테스트 정리

목표: MCTS 전용 테스트를 삭제하고 Gumbel 기준 테스트로 치환한다.

작업:

- `tests/test_mcts.py` 삭제
- `tests/test_self_play.py`
  - `play_mcts_game`, `play_mcts_games_batched` 이름 변경 여부에 맞춰 테스트 갱신
  - MCTS 전용 root noise, c_puct, playout cap 기대값 제거 또는 Gumbel 의미로 재작성
- `tests/test_pipeline.py`
  - `mcts_simulations`, `mcts_c_puct`, `search_backend="mcts"` 기대값 제거
- `tests/test_evaluate.py`
  - arena backend 선택 테스트를 Gumbel 단일 backend 기준으로 정리
- `tests/test_gumbel.py`
  - `EvalRequest` 분리 후 Gumbel leaf evaluator 테스트가 계속 통과하는지 확인

검증:

- `pytest`
- Rust extension이 설치되어 있지 않은 환경에서는 skip이 의도대로 동작하는지 확인

### 6. API 이름 정리

목표: 남아 있는 함수명/문서명에서 `mcts`를 제거한다.

작업:

- Python 함수명 변경
  - `play_mcts_game` -> `play_self_play_game` 또는 `play_search_game`
  - `play_mcts_games_batched` -> `play_self_play_games_batched`
  - `MctsSelfPlayConfig` -> `SelfPlayConfig`
  - `MctsSearchLike`, `MctsResultLike`, `MctsSelfPlayBatchLike` -> backend-neutral 이름
- `self_play_data.py`
  - docstring의 "MCTS root visit counts"를 "search visit counts"로 변경
- `evaluator.py`
  - docstring의 Rust MCTS 표현 제거
- `README.md`
  - MCTS backend 소개 제거
  - configs 경로와 실행 예제 갱신

검증:

- `rg -n "mcts|MCTS|c_puct|PUCT|Monte Carlo|monte carlo" .`
- 남는 경우가 있다면 과거 artifact 설명인지, 제거 대상인지 구분해서 처리

### 7. 최종 검증

로컬 CPU 환경에서 실행한다.

```bash
source .venv/bin/activate
pytest
cd rust/great_kingdom_core
cargo test
```

Rust extension이 필요한 Python 통합 테스트까지 확인하려면 다음을 추가한다.

```bash
source .venv/bin/activate
cd rust/great_kingdom_core
maturin develop
cd ../..
pytest
```

Runpod 환경에서는 설정 파일 정리 후 smoke만 확인한다.

```bash
python -m great_kingdom_ai.pipeline --config configs/test/pipeline-smoke.json
```

## 제거 완료 기준

- `rg -n "mcts|MCTS|c_puct|PUCT" python rust configs tests README.md` 결과가 0건이거나, 의도적으로 남긴 마이그레이션 문서뿐이다.
- `great_kingdom_core` Python module에 `MctsSearch`, `MctsResult`, `MctsSelfPlayBatch`가 노출되지 않는다.
- Gumbel self-play, batched self-play, arena, pipeline 테스트가 통과한다.
- 기본 설정 파일만으로 Gumbel pipeline smoke 실행이 가능하다.

## 리스크와 대응

- `EvalRequest` 이동 중 PyO3 class 등록이 깨질 수 있다.
  - 먼저 타입 분리만 하고 MCTS 구현은 그대로 둔 상태에서 테스트한다.
- `self_play.py`가 이미 크고 역할이 많다.
  - 이름 변경과 backend 제거를 한 커밋에 모두 넣지 말고, 필요하면 `self_play_config.py`, `search_backend.py` 같은 작은 모듈로 분리한다.
- 기존 JSON 설정을 읽는 사용자가 있을 수 있다.
  - 내부 프로젝트라면 즉시 제거한다.
  - 호환이 필요하면 한 릴리스 동안 deprecated key 감지 후 명확한 에러 메시지를 낸다.
- 테스트 fixture가 MCTS 이름에 강하게 묶여 있다.
  - 먼저 public API 이름을 backend-neutral로 바꾸고, 그 다음 MCTS 구현을 삭제하면 실패 위치가 명확해진다.
