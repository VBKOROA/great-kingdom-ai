# Great Kingdom AI 프로젝트 진행 역사

이 문서는 `docs/` 아래의 기존 문서를 모두 읽고, 프로젝트가 어떤 문제를 어떤 순서로 풀어 왔는지 하나의 흐름으로 정리한 기록이다.

문서들의 작성일이 모두 명시되어 있지는 않으므로, 아래 순서는 각 문서의 의존 관계, 완료 체크리스트, 실험 결과, 현재 README의 상태를 기준으로 재구성한 것이다.

## 1. 출발점: 재현 가능한 규칙 엔진

프로젝트의 첫 기반은 Great Kingdom 규칙을 재현 가능한 엔진으로 고정하는 일이었다.

`rule-spec.md`는 9x9 보드, 중앙 중립 성, 두 플레이어의 40개 성 제한, 패스, 영토, 포위와 파괴, 점수 승패를 명세한다. 특히 다음 규칙이 엔진 구현의 핵심 기준이 됐다.

- 성 연결과 자유 공간은 상하좌우만 사용한다.
- 상대 성 그룹을 하나라도 파괴하면 즉시 승리한다.
- 자기 성이 파괴되는 수는 입력 가능하지만, 상대 성을 먼저 파괴하지 못하면 즉시 패배한다.
- 연속 패스로만 영토 점수를 계산한다.
- Blue는 Orange보다 3칸 이상 앞서야 점수 승리하고, 그 외 점수 상황은 Orange 승리다.
- 상대 영토 안에는 착수할 수 없다.

`manual-cli-test-cases.md`는 이 규칙이 Rust `GameState`와 `great-kingdom-play` CLI에서 제대로 동작하는지 사람이 직접 확인하기 위한 체크리스트로 정리됐다. 초기 상태, 착수와 턴 전환, 중립 성 착수 금지, raw action index, 패스, 포획, 자살수, 판정 우선순위, 대각선 비연결, 보드 가장자리, 영토 착수 금지, 40개 제한, 종료 후 입력 거부 같은 항목을 수동 재현 시퀀스로 검증하게 했다.

이 단계의 산출물은 Rust 규칙 엔진과 Python CLI가 같은 규칙 명세를 공유하는 구조다. 이후 AI 실험은 모두 이 규칙 엔진 위에 올라간다.

## 2. AlphaZero-lite 파이프라인과 초기 검색 백엔드

프로젝트는 Great Kingdom용 AlphaZero-lite 실험 저장소로 확장됐다. 현재 README 기준 구조는 다음과 같다.

- Rust: 규칙 엔진, feature extraction, Gumbel search, PyO3 확장
- Python: self-play, replay buffer, policy-value network 학습, arena 평가, 모델 승격 pipeline
- Runpod: RTX 3090 24GB에서 본격 학습
- 로컬 노트북: CPU 기반 테스트, smoke run, 규칙 검증

초기에는 PUCT MCTS 백엔드와 Gumbel 백엔드가 함께 존재했다. 이후 `remove-mcts-plan.md`에서 MCTS 제거 계획이 세워졌다. 목표는 self-play, arena, pipeline을 Gumbel 기반으로 단순화하면서 Rust rules engine, feature extraction, replay sample 포맷, Python 학습 흐름은 유지하는 것이었다.

MCTS 제거 계획은 다음 순서로 잡혔다.

1. `EvalRequest`를 MCTS 모듈 밖으로 분리한다.
2. Python 검색 백엔드 기본값을 Gumbel로 고정한다.
3. pipeline과 arena 설정에서 `mcts_*`, `search_backend`, `c_puct` 계열을 제거한다.
4. Rust MCTS 구현과 PyO3 binding을 제거한다.
5. 테스트와 public API 이름에서 MCTS 표현을 정리한다.

이 결정으로 프로젝트의 검색 방향은 Gumbel search 단일 축으로 정리됐다.

## 3. Rust ONNX self-play로 데이터 생성 가속

다음 큰 전환은 Python/PyTorch callback 기반 self-play 병목을 줄이기 위해 Rust ONNX Runtime 경로를 만든 것이다.

`rust-onnx-self-play-plan.md`의 목표는 Runpod RTX 3090 환경에서 self-play 추론을 Python/PyTorch 콜백 대신 Rust/ONNX Runtime으로 실행해 데이터 생성 속도를 높이는 것이었다.

기존 구조의 병목 후보는 다음이었다.

- root/leaf evaluation마다 Python 콜백 진입
- NumPy 배열 생성과 PyTorch tensor 변환
- 반복적인 CPU/GPU 전송
- Python orchestration 비용

설계는 역할을 분리했다.

- Python은 PyTorch 학습, checkpoint 저장, ONNX export를 담당한다.
- Rust는 규칙 엔진, Gumbel search, ONNX 모델 추론을 담당한다.
- 기존 Python/PyTorch self-play 경로는 legacy/fallback으로 유지한다.
- 새 `great-kingdom-rust-onnx-pipeline`은 Rust ONNX self-play 결과를 replay buffer에 적재하고 학습을 이어간다.

구현 범위는 단계적으로 완료됐다.

- ONNX export CLI 추가
- FP32, opset 17, dynamic batch axis 기준 PyTorch/ONNX parity test
- Rust `OnnxEvaluator` CPU/CUDA provider 구조
- Python callback 없는 batched self-play
- Rust self-play artifact 저장
- legacy replay/checkpoint/log import
- Rust ONNX 전용 pipeline

현재 README에는 `great-kingdom-export-onnx`와 `great-kingdom-rust-onnx-pipeline` 실행 경로가 정식 사용법으로 반영돼 있다.

## 4. Arena 평가 병렬화

self-play가 Rust ONNX 경로로 이동한 뒤, arena 평가도 병목으로 남았다.

`batched-arena-plan.md`는 기존 `run_arena()`가 게임을 하나씩 순차 실행하고, 매 턴 root model 평가도 batch size 1로 처리하는 문제를 다뤘다. Rust Gumbel search 내부 leaf 평가는 batch화되어 있었지만, 여러 arena 게임 사이의 root/leaf inference는 묶이지 않았다.

해결 방향은 `ArenaConfig.batch_size`를 추가하고, `batch_size > 1`이면 여러 arena game을 동시에 진행하는 batched arena 경로를 사용하는 것이었다.

주요 구현은 완료됐다.

- `ArenaConfig.batch_size`와 validation 추가
- Rust `EvalRequest` optional metadata 추가
- Rust `GumbelArenaBatch` 추가
- active game들의 root/leaf evaluation을 candidate/best 모델별 batch로 나누는 Python 경로 추가
- `run_arena()` dispatch 연결
- Python/Rust 테스트와 Runpod config, README 갱신

README 기준 Runpod RTX 3090에서는 arena `batch_size=20`을 우선 권장하고, CUDA OOM이 발생하면 8 또는 4로 낮추는 운영 방식이 정리돼 있다.

## 5. Gumbel policy target 문제 진단

Rust ONNX pipeline이 동작한 뒤, 학습 품질 문제가 나타났다. 핵심 증상은 policy loss와 policy KL이 잘 내려가지 않는 것이었다.

`training-replay-store-plan.md`는 이 문제를 처음 정리했다. 관찰 결과는 다음과 같았다.

- single-batch overfit은 성공했다.
- 일반 replay 학습은 느렸다.
- replay policy target entropy가 매우 낮고, max probability가 거의 1에 가까웠다.
- 같은 feature에 서로 다른 policy/value target이 붙는 중복 state 충돌이 있었다.
- 기존 replay를 feature 기준으로 aggregate하면 학습 곡선과 arena 결과가 개선됐다.

처음 의심한 직접 원인은 search용 `gumbel_c_visit=50.0`, `gumbel_c_scale=1.0`이 replay policy target logits에도 그대로 들어가 target softmax를 사실상 one-hot에 가깝게 만든다는 점이었다.

이때의 1차 해결책은 `policy_target_temperature`였다.

- search action selection scale은 유지한다.
- replay에 저장하는 policy target softmax에만 temperature를 적용한다.
- 기존 replay와 새 temperature target replay는 섞지 않는다.

해당 문서의 체크리스트는 완료됐고, Runpod smoke에서 target entropy 상승과 raw replay 1000 step train smoke 개선이 확인됐다.

## 6. Target scale 분리와 남은 replay 문제

이후 `training-replay-current-diagnosis.md`에서 temperature만으로는 충분하지 않다는 점이 다시 확인됐다.

`policy_target_temperature=2.0`만 적용한 새 replay는 loss가 내려가기는 했지만 policy KL이 여전히 높았고, replay diagnostics에서는 target이 여전히 날카로운 편이었다.

그래서 action selection용 Gumbel scale과 replay policy target용 scale을 분리했다.

실험 설정은 다음 방향이었다.

```json
{
  "gumbel_c_visit": 50.0,
  "gumbel_c_scale": 1.0,
  "policy_target_c_visit": 5.0,
  "policy_target_c_scale": 0.25,
  "policy_target_temperature": 2.0
}
```

결과적으로 hard target 문제는 크게 완화됐다.

- `policy.entropy.p50`: 약 `0.05`에서 약 `3.16`으로 증가
- `policy.max_probability.p50`: 약 `0.99`에서 약 `0.25`로 감소
- `policy KL`: `2.x`에서 `0.55~0.60` 수준으로 감소

하지만 새로운 판단도 생겼다.

- policy loss는 target entropy가 높으면 자연스럽게 크게 보일 수 있으므로, policy loss보다 policy KL과 arena 결과를 봐야 한다.
- 같은 시작/초반 exact state가 replay에 반복 저장된다.
- 같은 state에 terminal outcome `-1`과 `+1`이 동시에 붙어 value target variance가 크다.
- value loss `0.8~0.9`는 학습 실패가 아니라 Monte Carlo terminal target 구조의 자연스러운 바닥일 수 있다.
- checkpoint를 새 work dir로 복사할 때 optimizer/scheduler state까지 이어받으면 실험 변인이 섞일 수 있다.

이 단계에서 hard policy target은 1순위 병목에서 내려왔고, replay distribution과 value variance가 더 중요한 문제로 올라왔다.

## 7. Replay 개선 방향 재정리

`gumbel-replay-improvement-plan.md`는 위 진단을 바탕으로 다음 개선 순서를 정리했다.

핵심 원칙은 다음이다.

- Arena가 최종 판정이다.
- replay policy target이 root prior를 거의 복사하는 self-imitation 상태인지 직접 계측해야 한다.
- exact-state 중복은 aggregate를 통해 variance reduction으로 활용하되, 방문 빈도 정보를 완전히 버리면 안 된다.
- Completed-Q value blending은 바로 기본 경로에 넣지 않고 별도 ablation으로 격리한다.

실행 우선순위는 다음으로 정리됐다.

1. weight-only bootstrap을 구현한다.
2. target-vs-prior diagnostics를 추가한다.
3. 현재 `5 / 0.25 / T=2` target scale로 arena를 판정한다.
4. 약하면 `10 / 0.5 / T=2` sharpen ablation을 비교한다.
5. count-aware aggregate replay store를 구현한다.
6. aggregate replay store를 Runpod에서 검증한다.
7. 마지막으로 Completed-Q value blending을 ablation한다.

문서 기준으로 weight-only bootstrap, target-vs-prior diagnostics, offline count-aware aggregate 관련 구현은 완료된 상태다. 온라인 replay store 전환과 Completed-Q value blending은 아직 후속 단계로 남아 있다.

## 8. Aggregate replay 실험 결과

`gumbel-aggregate-result.md`는 2026-05-06 run 결과를 정리한다.

비교는 pure Gumbel과 aggregate-only Gumbel 사이에서 이뤄졌다. aggregate-only는 Gumbel search와 policy target 설정은 pure와 같게 두고, 학습 replay에만 exact-state aggregate를 적용한 설정이다.

두 번의 400-game arena에서 aggregate-only가 모두 우세했다.

- Arena 1: aggregate-only 243승 / 400게임, 승률 60.75%
- Arena 2: aggregate-only 262승 / 400게임, 승률 65.5%
- 합산: aggregate-only 505승 / 800게임, 승률 63.125%

이 결과는 target temperature나 target scale 수정 효과가 아니라 replay aggregate 효과로 해석됐다.

따라서 현재 실전 config 반영 방향은 다음과 같다.

- 채택: aggregate replay
- 보류: policy target scale 분리
- 보류: policy target temperature 완화

Runpod 실전 학습 config는 pure Gumbel search/target 설정을 유지하면서 aggregate replay만 켜는 방향으로 조정됐다. 이후 `log_count + cap=null` 추가 ablation도 pure 대비 247승 / 400게임, 승률 61.75%를 기록했으므로, 임의 cap이 없는 `log_count` weighting을 실전값으로 채택했다.

```json
{
  "aggregate_replay": true,
  "aggregate_replay_weight_mode": "log_count",
  "aggregate_replay_weight_cap": null,
  "policy_target_temperature": 1.0,
  "policy_target_c_visit": 50.0,
  "policy_target_c_scale": 1.0
}
```

현재 문서 기준 최종 실험 결론은 `pure Gumbel < aggregate-only Gumbel`이다.

## 9. 현재 성능 병목: Gumbel select hot path

학습 품질 개선과 별개로, Runpod profile에서는 Gumbel search의 최대 성능 병목이 neural eval이 아니라 select 단계로 드러났다.

`select-bottleneck-plan.md`의 관찰 로그 기준, 30 waves 합계에서 대략 다음 비중이 나왔다.

- select: 40.7%
- request: 13.5%
- eval: 22.5%
- backup: 22.8%

`select`가 단독 1위 병목이다. 현재 select 경로는 simulation마다 다음 비용을 낸다.

- root action scheduler 선형 스캔
- root `edge_index_for_action()` 선형 검색
- `GameState::clone()`
- path를 따라 `GameState::apply(action)`
- path `Vec` allocation/grow
- 내부 node 선택 시 여러 임시 `Vec` 생성
- pending leaf 반환 시 leaf `GameState::clone()`

추천 개선안은 selection hot path를 allocation-free에 가깝게 바꾸는 것이다.

2026-05-06에는 1차 최적화로 `select_inner_action_index()` hot path를 기존 reference selector에서
분리했다. 기존 `inner_improved_policy()` / `select_inner_action()`는 테스트와 의미 검증용 reference로
유지하고, search 내부 node 선택 경로만 heap allocation 없는 다중 pass 계산으로 교체했다.

이 변경으로 내부 node 선택 시 매번 생성되던 `InnerEdgeStats`, `prior_probs`, `completed_q`,
`transformed_q`, `logits`, `InnerPolicyEntry` 임시 `Vec` 생성을 제거했다. 선택 수식과 tie-break는 기존
reference selector와 동일하게 유지해야 하므로 parity test를 추가했다.

변경 후 Runpod profile에서는 같은 `active_games=32` 조건에서 전체 wave 시간이 대략 38% 줄었고,
`select` 시간도 약 39% 감소했다. 다만 합산 비중은 여전히 `select`가 약 41%로 가장 컸고, `eval`과
`backup`이 각각 약 26%, 25% 수준으로 뒤를 이었다. 따라서 다음 1순위는 backup보다 위험이 낮은
`GumbelNode::edge_index_for_action()` action lookup table 최적화다.

후속 작업으로 `GumbelNode`에 action lookup table을 추가해 `edge_index_for_action()`의 선형 검색을
O(1) 조회로 교체했다. node 생성 시 `[u16; ACTION_SPACE]` 테이블을 채우고, search hot path는 기존 API를
그대로 호출하므로 선택 의미는 바꾸지 않는다.

action lookup table 적용 후 profile에서는 의미 있는 개선이 확인되지 않았다. 다음 판단을 위해
`GKA_GUMBEL_SELECT_DETAIL=1` 플래그를 추가했고, 기존 `GKA_GUMBEL_PROFILE=1`과 함께 켜면 wave별로
`scheduler_next`, `state_clone`, `search`, `apply`, `inner_select`, `edge_lookup`, `leaf_clone`,
`reserve_path`, `scheduler_reserve`, `terminal_backup`, `complete_reserved` 세부 시간이 출력된다.

남은 우선순위는 다음이다.

1. Runpod select detail profile로 `GameState::clone/apply`, scheduler, path/reserve 중 실제 병목을 확정한다.
2. 병목이 scheduler 재검색이면 scheduler index API로 `next_action()` 이후 `reserve_visit(action)` 재검색을 줄인다.
3. 그래도 backup 비중이 크면 backup 경로를 별도 최적화한다.

기대 효과는 select 시간을 20~50% 줄여 전체 wall time을 약 8~20% 개선하는 것이다.

## 10. 현재 프로젝트 상태 요약

문서 전체를 종합하면 현재 프로젝트는 다음 상태에 있다.

- 규칙 명세와 수동 CLI 검증 기준이 정리돼 있다.
- Rust 규칙 엔진과 Python 학습/평가/CLI 구조가 잡혀 있다.
- 검색 백엔드는 MCTS를 제거하고 Gumbel 중심으로 단순화하는 방향을 택했다.
- Rust ONNX self-play pipeline이 도입되어 Runpod CUDA self-play 경로가 정식화됐다.
- Arena는 `batch_size` 기반 batched 실행 경로가 추가됐다.
- Gumbel policy target이 지나치게 hard한 문제는 temperature와 target scale 분리 실험으로 진단됐다.
- 하지만 가장 최근 aggregate 실험에서는 target scale/temperature보다 exact-state aggregate replay가 더 확실한 strength 개선을 보였다.
- 현재 실전 방향은 pure Gumbel search/target을 유지하면서 count-aware aggregate replay를 채택하는 쪽이다.
- 다음 성능 개선 후보는 Runpod select detail profile로 select 내부 병목을 확정한 뒤 scheduler index API나
  `GameState::clone/apply` 최적화를 검토하는 것이다.

## 11. 남은 과제

문서 기준으로 남은 과제는 다음이다.

1. aggregate replay를 offline artifact 수준에서 충분히 검증한 뒤 online replay store 형태로 정식화한다.
2. target-vs-prior diagnostics를 계속 사용해 search-improved target과 root prior 복사를 구분한다.
3. aggregate-only 설정을 full Runpod config에서 더 긴 학습과 arena로 검증한다.
4. arena가 약해지면 target sharpen ablation을 다시 비교한다.
5. value variance가 명확한 병목이라는 근거가 쌓일 때만 Completed-Q value blending을 별도 ablation으로 진행한다.
6. Runpod select detail profile로 select 내부 병목을 확정하고 후속 최적화를 진행한다.
7. README와 설정 파일은 실험 결론이 바뀔 때마다 현재 기본 경로와 legacy/fallback 경로를 명확히 구분해 갱신한다.

## 12. 참고한 문서

- `docs/rule-spec.md`
- `docs/manual-cli-test-cases.md`
- `docs/remove-mcts-plan.md`
- `docs/rust-onnx-self-play-plan.md`
- `docs/batched-arena-plan.md`
- `docs/training-replay-store-plan.md`
- `docs/training-replay-current-diagnosis.md`
- `docs/gumbel-replay-improvement-plan.md`
- `docs/gumbel-aggregate-result.md`
- `docs/select-bottleneck-plan.md`
