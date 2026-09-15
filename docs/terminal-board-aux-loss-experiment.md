# 종국 보드 예측 auxiliary loss 실험 안내

이 문서는 `docs/terminal-board-aux-loss-plan.md` 구현의 실행 방법과 검증 상태를 기록한다.
이 안내의 명령은 별도 실험 설정과 work directory를 사용한다. 현재 Runpod 기본 설정에도
보조 loss를 활성화했으며, 최신 초기값과 기존 checkpoint 전환 주의점은
[Runpod 초기 설정](runpod-initial-training-config.md)을 따른다.

## 구현 요약

- self-play 단일/배치/Rust ONNX 경로 모두 종료 직후의 9×9 보드를 `GameLog`와 replay
  episode에 수집한다.
- 엔진 절대 색 코드(`0=empty, 1=blue, 2=orange, 3=neutral`)를 episode당 한 번
  `terminal_boards: uint8[E,9,9]` + `terminal_board_present: bool[E]` +
  `terminal_board_encoding="absolute_cells_preserved_v1"`로 저장한다.
- 학습 시 transition의 `player` 관점으로 변환한다(`0 empty, 1 own, 2 opponent, 3 neutral`).
  색 관점 변환과 D4 대칭 변환은 별도 함수이며, 대칭은 feature/policy/legal mask와
  동일한 변환을 종국 타깃에도 적용한다.
- `ModelConfig`의 기본 비활성 필드 `terminal_board_head`, `terminal_board_hidden_channels`로
  선택적 Conv1×1 head를 만들고, `strong_attn_terminal_board` 프리셋을 제공한다.
  `forward(x) -> (policy_logits, value)` 계약과 ONNX 두 출력은 유지된다.
- `terminal_board_loss_weight`(기본 0.0)로 칸 평균 cross entropy 보조 loss를 켠다.
  유효 타깃이 없으면 NaN 없이 0으로 처리하고 coverage 지표를 남긴다.
- 기존 필드 없는 replay/checkpoint는 그대로 로드되며 보조 타깃만 invalid가 된다.
- `--warm-start-terminal-board`로 기존 `strong_attn` weight에서 보조 head만 새로 초기화해
  시작할 수 있다. 이때 optimizer/scheduler/scaler/EMA는 새로 초기화된다.

## 새 데이터 확인

학습을 시작하기 전에 새 self-play shard에 종국 보드가 기록되는지 확인한다.

```bash
.venv/bin/python - <<'PY'
from great_kingdom_ai.replay import TrajectoryReplayStore
store = TrajectoryReplayStore.load("data/runpod/train-strong-attn-terminal-board/replay/trajectory-replay.npz")
print("episodes", store.episode_count, "rows", len(store))
print("terminal boards present", None if store.terminal_board_present is None else int(store.terminal_board_present.sum()))
print("encoding", store.terminal_board_encoding)
PY
```

## 실험 시작

설정 파일: `configs/experiments/terminal-board/{train,actor-v2,learner-v2}.yaml`
실험 work directory: `data/runpod/train-strong-attn-terminal-board`

### A. 새로 초기화

```bash
.venv/bin/great-kingdom-init-async-v2 \
  --train-config configs/experiments/terminal-board/train.yaml \
  --work-dir data/runpod/train-strong-attn-terminal-board \
  --model-preset strong_attn_terminal_board \
  --overwrite
```

### B. 기존 strong_attn checkpoint에서 warm start

`--warm-start-terminal-board`는 기존 checkpoint의 backbone/policy/value weight를 그대로
복원하고 보조 head만 무작위 초기화한다. optimizer/scheduler/scaler/EMA와 step은 새로
시작하므로 정확한 resume와 다르다.

```bash
.venv/bin/great-kingdom-init-async-v2 \
  --train-config configs/experiments/terminal-board/train.yaml \
  --work-dir data/runpod/train-strong-attn-terminal-board \
  --warm-start-terminal-board data/runpod/train-strong-attn/checkpoints/training-latest.pt \
  --overwrite
```

### C. actor / learner

```bash
RAYON_NUM_THREADS=4 GKA_ONNX_BATCH_BUCKETING=1 \
.venv/bin/great-kingdom-actor-v2 \
  --actor-config configs/experiments/terminal-board/actor-v2.yaml \
  --loop

.venv/bin/great-kingdom-learner-v2 \
  --learner-config configs/experiments/terminal-board/learner-v2.yaml \
  --train-config configs/experiments/terminal-board/train.yaml \
  --loop --sleep-seconds 3
```

학습 로그의 `aux=... aux_cov=... aux_acc=...`가 보조 loss와 coverage/정확도다.
`aux_cov`가 0이면 해당 데이터에 종국 보드가 없다는 뜻이므로 데이터 경로를 먼저 확인한다.

## 평가

동일한 시작 weight/데이터 조건에서 baseline(`train-strong-attn`)과 실험군을 arena로
비교한다. 보드 정확도만으로 승격하지 않는다.

```bash
.venv/bin/great-kingdom-evaluate \
  --candidate data/runpod/train-strong-attn-terminal-board/checkpoints/onnx/training-latest.onnx \
  --best data/runpod/train-strong-attn/checkpoints/snapshots/baseline.onnx \
  --report data/runpod/train-strong-attn-terminal-board/reports/arena-vs-baseline.json \
  --config configs/runpod/arena.yaml \
  --games 80 --batch-size 80 --device cuda
```

## 검증 상태

로컬 CPU에서 실행한 항목:

- Rust: 종국 보드가 포획된 성과 마지막 착수를 보존하고, trusted search 착수가 일반
  apply와 동일한 종국 상태를 만든다 (`cargo test --lib rules::tests`).
- Python: self-play 단일/배치 종국 보드 수집, replay NPZ round-trip, legacy NPZ, 신구 혼합
  append, capacity trimming, 알 수 없는 encoding 거부, 관점/D4 변환, dataset gather,
  보조 loss(수작업 계산, 전부 invalid, lambda=0, 잘못된 설정, backward, 감소), 기존
  checkpoint 로드, warm-start, resume, EMA round-trip, ONNX 두 출력/보조 branch 미포함.

Runpod GPU 미실행 항목:

- 본 학습/actor-learner 처리량, AMP 경로, 다중 seed arena 승률.
