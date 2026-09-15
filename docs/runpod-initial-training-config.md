# Runpod self-play 학습 초기 설정

기준일: 2026-09-15

이 문서는 `configs/runpod`의 actor·learner·학습 초기값과 선정 근거를 기록한다.
게임 규칙과 현재 구현을 바탕으로 고른 **실측 전 시작값**이며, 최적 처리량이나 기력 향상을
검증한 설정은 아니다. 실제 실행값은 YAML을 기준으로 한다.

## 1. 환경과 범위

- GPU: Runpod RTX 3090 24GB 한 장, CPU: AMD EPYC 7C13. Pod에 할당된 CPU/RAM은 별도 확인한다.
- Pod template: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` (+Jupyter).
- actor 1개와 learner 1개가 GPU를 공유한다. actor 내부에서 여러 게임을 병렬 처리한다.
- 모델: `strong_attn_terminal_board`, 종국 보드 보조 loss 0.1, ONNX 추론 FP16, learner AMP 사용.
- 운영 경로: `data/runpod/train-strong-attn`.
- 로컬 검증: GPU 없는 노트북, `.venv/bin/python` 사용.
- 규칙: [rule-spec.md](rule-spec.md). 9×9 보드이며 상대 포획·자살로 즉시 승패가 결정된다.
- 종국 보드 보조 loss를 Runpod 기본 설정에도 활성화했다. 구현·warm-start 설명은
  [별도 안내](terminal-board-aux-loss-experiment.md)를 참고한다. 별도 실험 YAML은 최초 실험
  시점의 설정이므로 최신 Runpod 초기값과 학습 스케줄 등이 다르다.

## 2. Actor 초기값

설정: [actor-v2.yaml](../configs/runpod/actor-v2.yaml)

| 항목 | 값 | 의미와 선택 이유 |
|---|---:|---|
| `games` | 256 | 한 shard의 게임 수. 게임 간 병렬성과 새 모델 반영 간격을 절충한다. |
| `rust_self_play_batch_size` | 256 | 한 번에 진행할 게임 수의 상한. |
| `onnx_max_batch_size` | 512 | 한 ONNX 호출의 입력 행 수 상한. learner와 GPU를 공유하는 조건을 고려한다. |
| `self_play.leaf_batch_size` | 4 | 한 게임에서 평가 결과를 받기 전에 선택하는 leaf 수의 상한. |
| `ema_opponent_fraction` | 0.25 | EMA 상대와 대국하는 게임 비율. |

현재 구현은 일반 self-play와 raw/EMA 혼합 대국을 따로 실행한다. 따라서 EMA 모델이 있으면
256게임은 **일반 192게임 → 혼합 64게임**으로 진행된다. 혼합 대국의 추론 요청은 사용할
모델에 따라 다시 나뉜다.

`leaf_batch_size=4`는 GPU 입력이 4개라는 뜻이 아니다. 일반 192게임이 모두 진행 중이면
게임마다 최대 4개의 leaf를 모아 최대 768개를 요청하고, ONNX는 최대 512개씩 처리한다.
종료 상태, 탐색 예산, 중복 leaf 방지 등에 따라 실제 요청은 이보다 작다.

현재 playout cap randomization은 다음과 같다.

| 항목 | 값 |
|---|---:|
| `playout_cap_randomization` | true |
| `playout_cap_full_search_fraction` | 0.25 |
| full simulations / max considered actions | 64 / 16 |
| fast simulations / max considered actions | 16 / 4 |

replay 용량과 재사용률에서 세는 transition은 저장된 학습 행이다. full/fast 선택으로
일부 수순만 학습에 저장되므로 게임 수나 전체 착수 수와 동일하지 않다.

낮은 탐색 예산과 즉시 승패가 갈리는 게임 특성을 고려해 leaf 묶음을 작게 하고, 평가 결과를
다음 탐색에 자주 반영하도록 했다. 대신 추론 호출과 동기화가 늘 수 있다. 이 선택의 기력상
이점은 아직 검증하지 않았다.

이전 `leaf_batch_size=1024`는 `fed4376`에서 2048에서 줄어든 값이다. 당시 ONNX 상한은
4096→1024, 동시 게임 상한은 256→64로 함께 변경됐다. 커밋에는 최적화라는 설명만 있고
측정 근거는 없다. 현재 구현에서는 남은 탐색 예산도 상한이므로 1024는 실질적으로 leaf
배치 제한을 풀어둔 값에 가깝다.

## 3. Learner 초기값

설정: [learner-v2.yaml](../configs/runpod/learner-v2.yaml)

| 항목 | 값 | 선택 이유 |
|---|---:|---|
| `replay_capacity` | 262,144 | 균등 샘플링에서 과거 데이터 다양성과 최신성을 절충한다. |
| `min_replay_transitions` | 16,384 | 배치 512의 32배를 확보한 뒤 학습을 시작한다. |
| `train_reuse_factor` | 4.0 | 새 데이터에 비례한 학습 예산을 제한해 반복 학습과 GPU 경쟁을 조절한다. |

`--loop`에서는 새 transition이 들어올 때 다음 예산을 더한다.

```text
추가 학습 샘플 예산 = 새 transition 수 × train_reuse_factor
실행 step 수 = min(steps, floor(남은 학습 샘플 예산 / batch_size))
```

새 transition 4,096개는 학습 샘플 예산 16,384개, 즉 배치 512에서 32 step에 해당한다.
각 transition을 정확히 네 번 사용하는 방식은 아니다. 전체 replay에서 무작위로 뽑는다.
`train_reuse_factor`에 따른 예산 제어는 연속 실행 `--loop` 경로에 적용된다.

replay가 가득 찬 정상 상태에서, 예산을 모두 소비한다는 가정하에 용량만큼 새 행을 수집하는
동안의 학습량은 약 `262144 × 4 / 512 = 2048 step`이다. 초기 replay 확장기나 예산 적체
상황에는 이 근사를 그대로 적용하지 않는다. Replay는 CPU 메모리와 디스크에 저장되므로
용량을 GPU VRAM만으로 판단하지 않는다.

## 4. 학습 초기값

설정: [train.yaml](../configs/runpod/train.yaml)

| 구분 | 항목 | 값 |
|---|---|---|
| Training chunk | `batch_size` | 512 |
| | `steps` | 128 |
| | `amp` | true |
| | `prefetch_batches` | 2 |
| Optimizer | `optimizer` | adamw |
| | `learning_rate` | 0.0003 |
| | `weight_decay` | 0.01 |
| | `gradient_clip_norm` | 5.0 |
| Loss | `policy_loss_weight` / `value_loss_weight` | 1.0 / 1.0 |
| | `l2_loss_weight` | 0.0 |
| | `terminal_board_loss_weight` | 0.1 |
| | `mask_policy_loss` | true |
| Learning-rate schedule | `lr_schedule` | constant_with_warmup |
| | `lr_warmup_steps` | 512 |
| EMA | `ema_decay` | 0.999 |

- 배치 512로 학습 효율을 확보하고, chunk 상한을 128 step으로 두어 새 데이터 확인과 모델
  배포 간격을 제한한다. 실제 chunk는 예산에 따라 짧아진다. 짧은 chunk는 checkpoint 저장과
  ONNX export 빈도를 늘릴 수 있으므로 전체 처리량으로 평가한다.
- Prefetch 2개는 데이터 준비를 겹치면서 대기 배치 메모리를 제한하려는 시작값이다.
- AdamW와 학습률 3e-4, weight decay 0.01, gradient clipping 5.0은 기존 값을 유지한다.
  별도 L2 loss는 추가하지 않는다. `momentum`과 `nesterov`는 현재 구현에서 SGD 전용이다.
- Policy/value는 1:1로 시작한다. 두 loss의 숫자가 같다는 뜻은 아니며, 이후 지표와 arena로
  조정 필요성을 판단한다. 종국 보드 보조 head를 활성화하고 칸 평균 CE에 0.1을 곱해
  추가한다. 이는 실험 시작값이며 검증된 최적 가중치가 아니다.
- 학습 종료 시점이 정해지지 않은 지속 self-play이므로 512 optimizer step 동안 워밍업한
  뒤 LR 3e-4를 유지한다. 성능 정체 시 감쇠 여부를 별도로 판단한다. StepLR/cosine 전용
  옵션은 현재 schedule에 적용되지 않는다.
- EMA의 과거 가중치 반감기는 `log(0.5) / log(0.999) ≈ 693 step`이다. Chunk 경계에서
  초기화되지 않는다. 안정성과 최신 모델 추종 속도를 절충한 값이며 파라미터 평균 기준이다.
- D4 augmentation 사용, recency/priority sampling 비활성, terminal value target을 유지한다.

### 초기화와 resume 구분

새 학습은 현재 train YAML로 `great-kingdom-init-async-v2`를 실행하면 보조 head가 포함된다.
기존 head 없는 `strong_attn` checkpoint는 YAML만 바꾸고 resume할 수 없다. 로더는 저장된
모델 구조를 복원하므로 보조 loss를 계산할 때 head 누락 오류가 난다. 원본 checkpoint를
출력 경로와 다른 위치에 보관하고, 초기화 CLI의 `--warm-start-terminal-board`에 그 경로를
전달해야 한다. Warm-start는 backbone/policy/value를 복원하고 보조 head를 새로 만들며,
optimizer/scheduler/scaler/EMA와 step을 초기화한다. 기존 출력이 있는 운영 경로를 사용할
때는 actor/learner를 멈추고 원본을 보관한 뒤 명시적으로 `--overwrite`를 사용한다.
이미 보조 head가 있는 checkpoint는 warm-start 대신 기존 resume 경로를 사용한다.

Actor는 재빌드한 최신 Rust 확장을 사용해야 종국 보드를 수집한다. 기존 replay에 타깃이
없어도 읽을 수 있지만 그 행은 보조 loss에서 제외되며 자동 backfill은 없다. 새 shard가
들어온 뒤 `aux_cov`가 0보다 큰지 확인한다. ONNX 출력은 policy/value 두 개를 유지한다.

위 값은 새 학습의 초기 설정이다. `train_checkpoint_mode: resume`에서는 checkpoint의
optimizer·scheduler 상태와 누적 step을 복원하므로 YAML 변경만으로 모든 상태가 초기화되지
않는다. 특히 optimizer의 LR/weight decay가 저장값에서 복원되며, 워밍업은 다시 시작되지
않는다. Scheduler 함수는 현재 설정으로 생성되므로 schedule 변경은 재개 이후 LR에 영향을
줄 수 있다. 기존 학습에서 변경할 때는 시작 로그와 실제 LR을 확인한다.

## 5. 실행과 실측 확인

초기 checkpoint와 raw/EMA ONNX 모델을 준비한 뒤 각각 실행한다.

```bash
.venv/bin/great-kingdom-actor-v2 \
  --actor-config configs/runpod/actor-v2.yaml --loop

.venv/bin/great-kingdom-learner-v2 \
  --learner-config configs/runpod/learner-v2.yaml \
  --train-config configs/runpod/train.yaml --loop --sleep-seconds 3
```

설정 파일을 바꾼 뒤에는 해당 프로세스를 재시작해 적용한다.

실측에서는 시간당 저장 transition 수, learner step 수, 데이터 대기·예산 적체,
checkpoint/export 소요 시간, CPU RAM·GPU 메모리 최대 사용량을 함께 기록한다.
Policy KL/value error와 선후공 교대 arena 결과로 학습 품질을 따로 확인한다.
GPU 사용률만으로 actor 수나 배치를 늘리지 않는다. Actor 추가는 데이터 공급이 부족하고
GPU·CPU·메모리에 여유가 있을 때 비교한다.

## 6. 검증 범위와 참고

로컬 CPU에서 YAML 로딩·설정 유효성, 재사용률 예산 계산, chunk 상한, 작은 모델의
optimizer·EMA 초기화, 512 step 워밍업 후 일정 LR을 확인했다. GPU 동시 실행의 메모리,
AMP 학습 안정성, 처리량과 arena 성능은 미검증이다.

문서화 시점에 Python 전체 테스트 393개, mypy 63개 소스 파일 검사, 변경 테스트 파일의
Ruff 검사와 `git diff --check`가 통과했다. 기존 설정 로딩 테스트의 기대값도 갱신했다.
Runpod 보조 loss 활성화 후에는 실제 `strong_attn_terminal_board` 프리셋으로 CPU 학습
한 step을 실행해 유효 타깃 집계, 유한 loss, 보조 head gradient와 EMA head 존재를 확인했다.

- [ONNX Runtime 성능 튜닝](https://onnxruntime.ai/docs/performance/tune-performance/):
  지연·처리량·메모리를 함께 측정하는 관점의 참고 자료.
- [Revisiting Fundamentals of Experience Replay](https://proceedings.mlr.press/v119/fedus20a.html):
  replay 용량과 재사용률의 영향을 다룬 연구. Q-learning 연구로, 본 게임의 숫자에 대한
  직접적인 근거로 사용하지 않는다.
- [PyTorch AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW):
  optimizer와 weight decay 동작 참고.
