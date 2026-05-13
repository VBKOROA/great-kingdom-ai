# On-Sample Reanalyze Review Actions

## Context

`improvements/PAPER_ALIGNED_REANALYZE_PROPOSAL.md` 기준 구현을 검토한 결과,
전체 테스트는 통과했다.

```text
.venv/bin/python -m pytest
257 passed, 1 skipped
```

다만 `on_sample` 경로의 priority semantics, metrics, phase1+phase2 통합 검증은 더 보강해야
한다.

## 개선할 점

1. **`on_sample` priority sampling 보정**

   현재 `OnSampleReanalyzeDataset`은 priority sampling에 `replay.sample_weights`만 사용한다.
   이 때문에 `priority_value_error_weight`, `priority_policy_kl_weight`,
   `priority_target_age_weight`가 샘플 선택에 실질적으로 반영되지 않는다.

2. **`on_sample` 전용 priority 상태 추가**

   추천 방향은 replay의 `sample_weights`를 priority 저장소로 겸용하지 않는 것이다.

   - `OnSampleReanalyzeDataset` 내부에 별도 `priorities: np.ndarray`를 둔다.
   - 초기값은 replay row별 기본값, 예를 들면 `replay.sample_weights` 또는 `1.0`으로 둔다.
   - priority sampling은 이 별도 priority 배열을 사용한다.
   - sampled batch에 대해 fresh target과 current model prediction/logits를 계산한 뒤,
     `priority_value_error_weight`, `priority_policy_kl_weight`, `priority_target_age_weight`를
     반영해 해당 row들의 priority를 갱신한다.
   - learner loss에 들어가는 `sample_weights`는 기존 replay sample weight와 PER importance
     weight 조합으로 유지하고, priority 값 자체를 loss weight로 직접 섞지 않는다.

   이렇게 하면 on-sample 구조의 한계, 즉 target이 sampling 이후에 만들어진다는 점을 인정하면서도
   다음 sampling부터 fresh error/KL 기반 priority가 반영된다.

3. **`search_reanalyzed` summary 값 수정**

   `on_sample` summary에서 `search_reanalyzed`가 항상 0으로 기록된다. 실제 search refresh 수와
   맞추거나, snapshot의 `search_reanalyzed`와 on-sample의 `policy_reanalyzed`가 어떻게 다른지
   명확히 분리해야 한다.

4. **`policy_reanalyze_ratio`와 `search_reanalyze_fraction/budget` 의미 정리**

   현재 sampled policy refresh에서는 `policy_reanalyze_ratio`로 선택한 row들이
   `refresh_sampled_policies_with_search()`에 들어간다. 이때 `search_reanalyze_fraction`과
   `search_reanalyze_budget`의 의미가 snapshot 모드와 달라질 수 있으므로 설정 해석을 명확히
   해야 한다.

5. **`mcts_root` bootstrap 실패 경로 처리**

   phase1과 phase2를 한 번에 진행하기로 했으므로 `value_bootstrap_source=mcts_root`는 기본
   경로로 검증해야 한다. root value가 없거나 search backend가 지원하지 않는 경우 fail-fast가
   적절한지, 또는 config validation에서 먼저 막아야 하는지 정해야 한다.

6. **`on_sample` instrumentation 보강**

   문서의 mitigation에 있는 지표를 더 채워야 한다.

   - sampled rows per step
   - value eval time
   - search time
   - policy reanalyze ratio actually applied
   - stale policy fallback count
   - bootstrap source used for value targets
   - train step time

7. **legal mask source 일관성 확인**

   training batch 반환 시 `legal_masks_from_features(features)`를 사용하고, search 쪽에서는
   replay에 저장된 `legal_masks`를 사용한다. 두 값이 항상 같다는 전제를 테스트하거나, 불일치 시
   명확히 감지해야 한다.

## 추가해야 할 테스트

1. **`on_sample` priority sampling이 priority config를 반영하는 테스트**

   value error, policy KL, target age가 큰 row의 priority가 실제로 더 커지고 다음 sampling에
   반영되는지 확인한다.

2. **priority weight 0 설정 테스트**

   `priority_value_error_weight=0`, `priority_policy_kl_weight=0`,
   `priority_target_age_weight=0`일 때 별도 priority가 기본값 또는 replay sample weight 기반으로만
   동작하는지 확인한다.

3. **`on_sample`과 snapshot value target row-for-row 비교 확대**

   다음 케이스를 포함한다.

   - episode boundary 직전
   - terminal row
   - player turn이 바뀌는 bootstrap
   - `gamma < 1`
   - `bootstrap_td_steps=0`
   - 여러 episode가 섞인 replay

4. **dynamic horizon snapshot-equivalence 테스트**

   `dynamic_horizon_enabled=true`에서 on-sample value target이 snapshot 경로와 같은 결과를 내는지
   비교한다.

5. **MCTS-root bootstrap snapshot-equivalence 테스트**

   `value_bootstrap_source=mcts_root`에서 snapshot과 on-sample이 같은 bootstrap row를 대상으로
   같은 부호 변환을 적용하는지 확인한다.

6. **실제 sampled search 경로 테스트**

   monkeypatch만이 아니라 Rust core fake 또는 lightweight integration으로
   `refresh_sampled_policies_with_search()` 경로를 태우고, `root_values`,
   `search_reanalyzed`, policy shape를 검증한다.

7. **`policy_reanalyze_ratio` 경계값 테스트**

   - `0.0`: 아무 row도 search refresh하지 않음
   - `1.0`: 전 row refresh
   - 작은 batch에서 `ceil` 동작 확인
   - invalid ratio 예외 확인

8. **`on_sample` summary 테스트**

   training 이후 아래 값들이 실제 sampling 결과와 맞는지 확인한다.

   - `sampled_batches`
   - `sampled_rows`
   - `policy_reanalyzed`
   - `search_reanalyzed`
   - `bootstrap_horizon_counts`
   - `bootstrap_source_counts`

9. **MCTS-root config validation 테스트**

   `reanalyze_mode=on_sample`, `value_bootstrap_source=mcts_root` 조합에서 search backend/root value
   지원이 없을 때 명확한 오류가 나는지 확인한다.

10. **legal mask 일관성 테스트**

    replay에 저장된 `legal_masks`와 `legal_masks_from_features(features)`가 다르면 감지하거나,
    하나의 source of truth를 일관되게 쓰는지 확인한다.

11. **`on_sample` + symmetry augmentation 테스트**

    `sample_arrays()`가 반환한 features, policies, legal masks가 augmentation 이후에도 shape와
    legal policy mask를 유지하는지 확인한다.

12. **priority importance weight 테스트**

    `on_sample`에서 priority sampling 시 returned `sample_weights`가 base replay sample weight와
    PER importance weight를 의도대로 조합하는지 확인한다.

## 현재 개발환경 기준 실행 가능 범위

현재 로컬 개발환경은 GPU 없는 사무용 노트북이며, venv 기준 Python 3.11, CPU-only PyTorch,
import 가능한 `great_kingdom_core` Rust 확장, 사용 가능한 `cargo`가 있는 상태다. 따라서 이
문서의 1차 목표인 구현, 단위 테스트, 작은 규모의 통합 테스트는 로컬에서 진행할 수 있다.

### 로컬 개발환경에서 가능한 작업

- `OnSampleReanalyzeDataset` 전용 priority 배열 추가
- sampled row 기준 priority 갱신 로직 추가
- priority sampling 및 PER importance weight 단위 테스트
- `search_reanalyzed` summary 값을 실제 sampled search 결과와 맞추는 수정
- `policy_reanalyze_ratio` 경계값 및 invalid 설정 테스트
- dynamic horizon, snapshot-equivalence 계열의 소형 replay 테스트
- `value_bootstrap_source=mcts_root` fail-fast 또는 config validation 테스트
- legal mask source 일관성 테스트
- on-sample summary 및 instrumentation 카운터 테스트
- `on_sample` + symmetry augmentation shape/legal mask 테스트
- Rust core를 사용하는 lightweight integration 테스트

### 로컬에서 가능하지만 축소해서 해야 하는 작업

- 실제 sampled search 경로 테스트
- MCTS-root bootstrap snapshot-equivalence 테스트
- 전체 pytest 실행

위 작업들은 CPU-only 환경에서도 가능하지만, search simulation 수, batch size, replay 크기를 작게
잡아야 한다. 로컬 결과는 기능 검증으로만 보고, 처리량이나 큰 search budget 안정성의 근거로
사용하지 않는다.

### Runpod 훈련환경에서 확인해야 하는 작업

- CUDA/GPU 경로 검증
- RTX 3090 24GB 기준 메모리 사용량, throughput, batch size 한계 검증
- 긴 학습 run에서 실제 정책 성능 향상 여부 확인
- 큰 MCTS simulation budget에서의 속도와 안정성 검증
- on-sample priority 변경이 최종 승률, Elo, self-play 품질에 미치는 영향 평가
- Runpod template `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`와 로컬 CPU-only
  PyTorch 환경 차이에서 발생할 수 있는 CUDA/버전 호환성 검증

## 권장 작업 순서

1. `on_sample` 전용 priority 배열과 sampled-row priority update를 추가한다.
2. priority 관련 테스트를 먼저 추가한다.
3. `search_reanalyzed` summary를 실제 값과 맞춘다.
4. dynamic horizon + MCTS-root snapshot-equivalence 테스트를 추가한다.
5. 실제 sampled search 경로 테스트를 추가한다.
6. instrumentation을 보강한다.
