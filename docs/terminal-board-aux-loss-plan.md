# 종국 보드 상태 예측 auxiliary loss 구현 계획

## 1. 목적과 범위

다른 구현 에이전트가 이 문서를 기준으로 작업한다. `strong_attn`의 기존 policy/value 학습에 **실제 self-play 종국 보드 배열을 예측하는 보조 loss 하나**를 추가한다. 공유 backbone이 미래의 성 배치와 연결·포위에 유용한 특징을 학습하도록 하는 실험이며, 기력 향상은 arena로 검증한다.

이번 작업에 포함하지 않는 것:

- 포획 시 성 보존/삭제 옵션 및 엔진 규칙 변경
- 빠른 종료 보상, 시간 할인 변경, value 타깃 변경
- 영토 소유권·점수·종료 방식·남은 수 예측 head
- backbone, attention, 기존 policy/value head의 구조 변경

`docs/rule-spec.md`를 준수한다. 로컬은 GPU 없는 노트북과 Python venv, 본 학습은 Runpod RTX 3090 24GB 환경이다. 데이터 변환과 보조 head/loss는 작은 모듈로 분리하고, 이미 큰 `replay/trajectory.py`와 `model.py`에 독립 로직을 몰아넣지 않는다.

## 2. 현재 코드에서 확인한 사실

- `rust/great_kingdom_core/src/rules.rs`: 일반 착수와 `apply_trusted_search_place` 모두 포획/자살 판정 후 `finish`로 종료한다. `finish`는 terminal/outcome만 기록하며 성을 삭제하지 않는다. 따라서 현재 종국 배열은 마지막 착수와 포획된 성을 모두 보존한다. 이전 대화의 “현재 엔진은 성을 삭제한다”는 전제는 적용하지 않는다.
- `rust/great_kingdom_core/src/game.rs`: 단일 `GameState.board()` 접근자가 있다. 배치 self-play는 `gumbel/batch.rs`를 사용하므로 종국 보드를 Python으로 전달할 배치 접근자를 확인하고, 없으면 읽기 전용 접근자를 추가한다.
- `python/great_kingdom_ai/model.py`: `strong_attn`은 128채널, residual 10개, attention 2개이며 `forward`는 `(policy_logits, value)`를 반환한다.
- `training/loop.py`: policy cross-entropy, value MSE, 선택적 L2로 loss를 계산한다.
- `replay/schema.py`: `TrajectoryEpisode`에 승자·종료 사유·영토 점수·전체 수순 필드가 있지만 명시적인 종국 배열은 없다.
- `replay/trajectory.py`는 episode 단위 메타데이터와 transition 배열을 관리한다. 저장 feature가 없는 리플레이도 `replay/dataset.py`가 수순으로 복원한다.
- `self_play_types.py`의 `GameLog`와 Python/Rust ONNX self-play 경로 모두 새 메타데이터 전달을 점검해야 한다.

구현 시작 시 위 사실과 실제 호출 경로를 재확인한다. 파일 이름이나 예시 API는 저장소 관례에 맞게 조정할 수 있지만 아래 데이터 의미와 호환성 계약은 유지한다.

## 3. 학습 타깃 계약

### 3.1 종국의 정의

실제 규칙상 종료가 확정된 직후의 9×9 보드다.

- 상대 포획승: 마지막 착수 후, 포획된 상대 성도 남아 있는 배열
- 자살 패배: 마지막 착수와 자유도가 없는 자기 그룹이 남아 있는 배열
- 연속 패스 종료: 두 번째 패스 후 배열
- 실행 중단이나 `max_turns` 초과는 종국 타깃으로 쓰지 않는다. 현재 예외 처리 동작을 유지한다.

마지막 저장 transition의 입력은 마지막 착수 **전**일 수 있으므로 이를 종국 배열로 간주하지 않는다. full/fast search 샘플링으로 저장 transition이 일부 수순만 포함해도 최종 실제 게임 상태에서 타깃을 얻는다.

### 3.2 저장 및 학습 표현

- 디스크에는 엔진의 절대 색 기준 셀 코드로 episode당 한 번만 저장한다. 예: `terminal_board: uint8[9,9]`. 실제 `Cell` 코드 매핑을 사용하고 중복 정의를 피한다.
- 존재 여부와 의미 버전을 기록한다. 예: `terminal_board_present`, `terminal_board_encoding="absolute_cells_preserved_v1"` 또는 동등한 명시적 스키마 정보.
- 학습 시 각 transition의 `player`를 기준으로 변환한다: `0=빈칸, 1=내 성, 2=상대 성, 3=중립 성`.
- 보조 logits는 `[B,4,9,9]`, 타깃은 정수 `[B,9,9]`, 유효 여부는 `[B]`이다.
- 한 episode의 서로 다른 플레이어 샘플은 내 성/상대 성 클래스가 교환되어야 한다. 종국 상태의 `current_player`로 관점을 정하지 않는다.
- 현재 합법 수 mask를 보조 loss에 적용하지 않는다. 이미 점유된 칸도 타깃에 포함한다.
- 종국 배열은 supervision으로만 사용한다. 모델 입력, 탐색, policy 타깃 생성에 미래 정보가 유입되지 않게 한다.

## 4. 구현 단계

### 단계 A. self-play에서 종국 배열 수집

주요 확인 파일:

- `rust/great_kingdom_core/src/game.rs`, `gumbel/batch.rs`
- `python/great_kingdom_ai/game_core.py`, `self_play_types.py`
- `self_play_runner.py`, `rust_onnx_self_play.py`, `self_play_data.py`, `rust_onnx_replay.py`
- `async_v2/actor.py` 및 shard 생성 호출 경로

단일·배치·Rust ONNX self-play 모두 완료된 게임의 보드를 수집한다. 배치에서는 game index와 episode/seed 대응을 보존하고, 완료 상태의 보드를 리셋 전에 읽는다. 배열은 복사하여 이후 mutable state와 공유하지 않는다. 기존 `GameLog` JSON 로딩은 새 필드 없는 자료도 허용한다. 실제 저장 경로를 확인해 로그→episode 변환과 직접 episode 생성 경로 모두 누락 없이 연결한다.

### 단계 B. episode 저장과 dataset 전달

주요 확인 파일:

- `replay/schema.py`, `replay/trajectory.py`, `replay/persistence.py`
- `replay/dataset.py`, `replay/sample.py`, `training/batch.py`
- `learner_prefetch.py`, `on_sample_reanalyze.py`와 샘플을 다시 만드는 경로

episode 단위 타깃을 NPZ 저장/로드, append/concat, capacity trimming, episode 재구성, shard import에 보존한다. transition마다 디스크에 81칸 배열을 복제하지 않는다. 샘플링 시 episode index로 gather한 뒤 현재 플레이어 관점으로 변환한다.

object 샘플 경로와 array batch 경로, feature 복원 경로, CPU→device/pinned memory/prefetch 경로가 모두 타깃과 유효 mask를 전달해야 한다. validation은 shape, 셀 코드, episode 수 대응, encoding을 검사한다. 알려지지 않은 encoding을 추측해서 읽지 않는다.

기존 자료 호환 정책:

- 종국 필드가 없는 기존 리플레이는 정상 로드하고 보조 타깃만 invalid로 둔다. 기존 policy/value 학습은 유지한다.
- 새 자료와 기존 자료를 섞어도 valid 샘플에만 보조 loss를 적용한다.
- 이번 구현에는 기존 수순을 재생하여 종국을 만드는 자동 backfill을 포함하지 않는다. 마지막 입력이나 영토 점수에서 종국을 추측하지 않는다.
- 보조 loss가 켜졌는데 유효 타깃이 전혀 없으면 coverage 지표와 명확한 로그로 알린다. 정상 학습 중인 것처럼 보조 loss=0만 표시하지 않는다.

### 단계 C. 대칭 augmentation

`augmentation.py`와 `training/batch.py`에서 feature/policy/legal mask에 적용하는 **동일한 D4 변환**을 종국 타깃에도 적용한다. 보조 타깃용 random symmetry를 별도로 추출하면 안 된다. 색 관점 변환과 공간 변환을 별도 함수로 두어 테스트한다. 회전·반사는 클래스 번호를 바꾸지 않는다.

### 단계 D. 선택적 보조 head

예시 구조:

```text
공유 backbone F ─┬─ 기존 policy head
                ├─ 기존 value head
                └─ Conv1×1(128→32) → ReLU → Conv1×1(32→4)
```

처음에는 작은 head 하나로 시작한다. logits에 softmax를 적용하지 않고 CE에 전달한다. 보조 head는 독립 모듈로 둔다.

- `ModelConfig`에 기본값 disabled인 선택 필드와 hidden 채널 수를 추가한다.
- 기존 `strong_attn` 프리셋 의미를 바꾸지 않고 `strong_attn_terminal_board` 같은 실험 프리셋을 추가한다. backbone/head의 기존 weight key를 유지한다.
- 기존 `forward(x) -> (policy_logits, value)` 계약을 유지한다.
- 학습 전용 `forward_with_aux` 또는 동등한 API로 세 출력을 얻되 backbone은 한 번만 계산한다.
- loss를 끈 경우 보조 head 연산도 생략할 수 있게 한다. 설정이 활성인데 모델에 head가 없는 조합은 명확히 거절한다.

### 단계 E. loss, 설정, 관측 지표

```text
cell_loss[i,h,w] = CE(aux_logits[i,:,h,w], target[i,h,w])
board_loss[i] = mean over 81 cells(cell_loss[i])
L_aux = sum(valid[i] × sample_weight[i] × board_loss[i])
        / sum(valid[i] × sample_weight[i])
L_total = 기존 L_total + terminal_board_loss_weight × L_aux
```

- 타깃 유효 샘플이 0개이면 NaN 없이 보조 항을 0으로 처리한다. invalid 타깃 sentinel이 CE에 들어가 오류를 만들지 않도록 먼저 선택하거나 ignore 처리를 한다.
- 81칸을 합산하지 않고 평균하여 보조 loss 스케일을 제어한다. 원래 policy/value 가중치와 replay sampling/priority 정의는 유지한다.
- 설정 기본값 `terminal_board_loss_weight=0.0`; 별도 실험 설정은 `0.1`을 시작값으로 사용한다. 이 값은 검증된 최적값이 아니다.
- 가중치는 finite/non-negative 검증을 한다. gradient는 보조 head와 공유 backbone에 전달한다.
- 첫 버전은 전체 칸 균등 CE로 고정한다. 미래에 바뀐 칸을 oracle 가중치로 강조하는 추가 목적은 넣지 않는다.

필수 지표:

- 가중치 적용 전 `terminal_board_loss`
- 유효 타깃 비율/수
- 전체 칸 accuracy
- **입력에서 빈칸이었던 칸**의 accuracy 및 평가 칸 수

현재와 같은 성을 복사하는 것만으로 전체 정확도가 높아질 수 있으므로 빈칸 지표를 함께 본다. 해당 칸이 없는 경우도 NaN 없이 처리한다. 총 loss 수치는 보조 항 추가 전후 직접 비교하지 않고, 기존 policy KL/value error와 arena 결과를 별도로 비교한다.

### 단계 F. checkpoint, EMA, ONNX 호환성

주요 파일: `training/checkpoint.py`, `training/config.py`, `training/cli.py`, `onnx_export.py`, EMA export 관련 script, async 초기화 경로.

- 새 ModelConfig 필드가 없는 기존 checkpoint는 head disabled로 정상 로드한다.
- 동일 구조 checkpoint resume은 기존 optimizer/scheduler/scaler/EMA 상태를 그대로 복원한다.
- 기존 `strong_attn` weight를 새 보조 모델로 옮기는 명시적 warm-start 경로를 제공한다. 기존 weight는 모두 일치시켜 복원하고 보조 head만 새로 초기화한다. 무조건 `strict=False`로 다른 불일치를 숨기지 않는다.
- 구조가 달라지는 warm-start에서는 optimizer/scheduler/scaler를 새로 초기화하고 step 정책을 문서화한다. EMA도 새 모델에서 초기화한다. 정확한 resume과 구별하고 원본 checkpoint를 덮어쓰지 않는다.
- EMA 업데이트/save/load가 새 보조 파라미터까지 처리되는지 확인한다.
- ONNX는 기존 두 출력만 export하며 보조 head 연산이 그래프에 포함되지 않아야 한다. Rust evaluator 입력/출력 계약은 유지한다.

## 5. 필수 검증

Python은 `.venv/bin/python`을 사용한다. 필요한 Rust/PyO3 빌드는 기존 README/setup 절차를 따른다. 테스트 파일은 기존 관련 테스트를 확장하거나 기능별 작은 파일로 추가한다.

1. **실제 종국 타깃:** 상대 포획, 자살, 연속 패스 fixture에서 마지막 착수와 성 보존을 검증한다. 일반 apply와 trusted search의 종국 결과도 일치해야 한다. 게임 규칙을 변경하지 않는다.
2. **전체 데이터 경로:** Python 단일/배치, Rust ONNX self-play 각각 최종 보드를 episode까지 전달한다. full/fast search 혼합과 마지막 학습 샘플 누락 상황에서도 실제 종국 배열을 얻는다.
3. **저장 호환:** 새 NPZ round-trip, 기존 필드 없는 NPZ, 신구 혼합 append, trimming, feature 없는 replay, shard import에서 episode와 타깃 연결을 검증한다.
4. **관점·대칭:** 같은 절대 배열의 양 플레이어 타깃, 8가지 D4 변환, batch/object 및 prefetch 경로의 정합성을 검증한다.
5. **loss:** 수작업으로 계산 가능한 logits로 칸 평균과 valid/sample weight 정규화를 확인한다. 전부 invalid, lambda=0, 잘못된 설정, AMP 경로를 다룬다. 보조 loss 단독 backward가 backbone과 보조 head에 도달하는지 확인한다.
6. **호환성:** 기존 모델의 두 출력, 기존 checkpoint 로드, 명시적 warm-start, 새 checkpoint resume 및 EMA round-trip을 검증한다. lambda=0의 기존 loss 결과가 유지되어야 한다.
7. **ONNX:** 보조 모델에서도 policy/value PyTorch-ONNX parity를 검증하고 보조 branch가 export되지 않았는지 확인한다. 필요한 ONNX Runtime이 없는 로컬은 미실행 사유를 기록하고 실행 가능한 환경에서 완료한다.
8. **통합 smoke:** 작은 실제 self-play → shard/replay 저장·로드 → batch → 보조 loss 포함 한 학습 step → checkpoint → ONNX 두 출력까지 연결한다. CPU에서는 작은 모델과 적은 탐색량을 사용한다.

추가로 작은 고정 batch에서 보조 loss가 감소하는지 확인한다. 이는 wiring 확인이며 기력 개선의 증거로 해석하지 않는다. 변경에 맞는 기존 Rust/Python 회귀 테스트를 실행하고, GPU 검증은 Runpod에서 수행한다.

## 6. 실험 실행 및 완료 기준

기존 `configs/runpod/train.yaml`과 운영 work directory를 덮어쓰지 않는다. 실험용 설정/별도 work directory 예시를 제공하고, 새 데이터에 타깃이 기록되는지 먼저 확인한다. 초기화 CLI, warm-start, actor-v2/learner-v2 실행, 평가 명령은 구현한 실제 인자로 문서화한다.

비교군은 기존 `strong_attn`, 실험군은 동일 backbone에 보조 loss만 켠 모델이다. 동일한 시작 weight와 데이터 조건을 최대한 맞추고, 기존 replay의 타깃 coverage 차이를 기록한다. 동일 학습 시간에서 선후공을 교대한 arena 승률과 불확실성, 기존 policy/value 지표, 학습·self-play 처리량을 비교한다. 가능한 경우 여러 seed로 재현한다. 보드 정확도 상승만으로 모델을 승격하지 않는다.

완료 조건:

- [ ] 엔진 규칙과 현재 성 보존 동작을 유지했다.
- [ ] 종국 보드가 모든 활성 self-play 경로에서 수집되고 episode당 한 번 저장된다.
- [ ] 현재 플레이어 관점, 대칭 변환, valid mask가 learner까지 일관되게 전달된다.
- [ ] 작은 선택적 보조 head와 설정 가능한 loss, coverage/빈칸 지표가 동작한다.
- [ ] 기존 replay/checkpoint 및 기존 inference API가 호환된다.
- [ ] warm-start와 정확한 resume의 의미가 분리되어 문서화된다.
- [ ] ONNX 추론에 보조 head가 포함되지 않는다.
- [ ] 관련 테스트와 end-to-end smoke 결과, 미실행 GPU 항목이 기록된다.
- [ ] 별도 실험 설정과 실행/평가 방법을 제공한다. 본격 Runpod 학습 결과는 구현 완료와 구분해 보고한다.

구현 에이전트는 변경 파일, 중요한 설계 결정, 실행한 테스트 결과, 남은 환경 제약을 최종 보고한다. 이 문서는 구현 계획이며 성능 향상이나 GPU 실험 완료를 주장하지 않는다.
