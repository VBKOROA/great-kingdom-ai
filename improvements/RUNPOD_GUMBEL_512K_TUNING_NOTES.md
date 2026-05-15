# Runpod Gumbel 512k 튜닝 메모

날짜: 2026-05-15

이 문서는 `data/runpod/train-v2-gumbel-512k` 실행의 현재 튜닝 상태를 정리한다. 일반
학습 가이드가 아니라, 이번 Runpod 실험에서 관찰한 운영 메모다.

## 현재 결론

현재 가장 나아 보이는 운영점은 다음 조합이다.

```json
{
  "train_reuse_factor": 4.0,
  "gumbel_c_scale": 0.25,
  "policy_target_c_scale": 0.25,
  "policy_target_temperature": 1.0
}
```

`training-latest-20260515-071246.pt`는 더 최신 snapshot이 arena에서 이기기 전까지 현재
peak/best 후보로 취급한다.

현재 관찰된 문제는 target sharpness가 아니다. target은 충분히 부드러워졌다. 더 강한
가설은 replay 과사용 또는 policy over-convergence다. learner가 replay CE/KL은 계속
개선하지만, 실제 arena strength는 떨어질 수 있다.

## 주요 체크포인트

| 체크포인트 | 의미 |
| --- | --- |
| `training-latest-20260515-052801.pt` | 이전 baseline. |
| `training-latest-20260515-060810.pt` | `052801`을 명확히 이긴 더 강한 baseline. |
| `training-latest-20260515-071246.pt` | reuse를 4로 낮춘 뒤 나온 현재 peak 후보. |
| `checkpoints/training-latest.pt` | 계속 움직이는 latest. arena 없이 신뢰하면 안 됨. |

## 이번 튜닝에서 바꾼 것

### Search/target Q transform scale

초기 설정에서는 target이 계속 날카로워졌다. 현재 구조는 vanilla Gumbel MuZero에 가깝게
유지하는 것이 목표라서, selection transform과 policy target transform을 따로 움직이지
않고 같이 낮췄다.

같이 변경한 값:

```json
{
  "gumbel_c_scale": 0.25,
  "policy_target_c_scale": 0.25
}
```

유지한 값:

```json
{
  "gumbel_c_visit": 50.0,
  "policy_target_c_visit": 50.0,
  "policy_target_temperature": 1.0
}
```

효과:

- target entropy 하락이 멈췄다.
- target max probability가 내려갔다.
- target support가 늘었다.
- 이 변경 자체가 바로 arena strength를 망가뜨리지는 않았다.

### Learner replay reuse

기존 `train_reuse_factor`는 현재 actor 데이터 생성 속도와 replay 크기에 비해 너무
공격적이었다.

변경 전:

```json
{
  "train_reuse_factor": 8.0
}
```

변경 후:

```json
{
  "train_reuse_factor": 4.0
}
```

효과:

- reuse를 낮추기 전 latest는 replay target 지표상 `060810`보다 더 잘 맞았지만, arena에서
  `060810`에게 졌다.
- reuse를 4로 낮춘 뒤 latest가 회복했고, `060810`을 여러 번 이겼다.

따라서 문제는 학습 부족이 아니라 replay 과사용 또는 over-convergence였을 가능성이 높다.

## Arena 결과

### Reuse 조정 전

| Candidate | Best | Games | Candidate win rate | 해석 |
| --- | --- | ---: | ---: | --- |
| latest | `060810` | 200 | 44.0% | latest가 `060810`보다 약함. |
| latest | `052801` | 200 | 58.0% | latest가 이전 baseline보다는 강함. |
| `060810` | `052801` | 200 | 63.0% | `060810`이 `052801`보다 명확히 강함. |

당시 대략적인 강도 순서:

```text
060810 > latest > 052801
```

### `train_reuse_factor = 4` 이후

latest versus `060810`:

| Games | Candidate wins | Best wins | Candidate win rate |
| ---: | ---: | ---: | ---: |
| 200 | 107 | 93 | 53.5% |
| 200 | 110 | 90 | 55.0% |
| 200 | 110 | 90 | 55.0% |

합산:

```text
latest 327 - 273 060810
win rate 54.5% over 600 games
```

이 결과 때문에 해당 구간의 snapshot, 즉 `071246` 근처 모델을 새 peak 후보로 볼 수 있었다.

### 추가 학습 후 peak를 지난 구간

이후 latest versus `training-latest-20260515-071246.pt`:

| Candidate | Best | Games | Candidate win rate | 해석 |
| --- | --- | ---: | ---: | --- |
| latest | `071246` | 200 | 46.0% | latest가 local peak를 지난 것으로 보임. |

이 결과가 핵심이다. replay metric만으로는 promote를 판단하면 안 된다.

## 진단 지표

최종 산출물이 EMA이므로 모델 판단은 EMA 지표를 기준으로 본다.

### Reuse 조정 후 건강했던 구간

arena가 좋아지던 시점의 대표 latest 지표:

```text
EMA mismatch              약 0.418
EMA policy CE             약 1.353
EMA KL(target || policy)  약 0.792
EMA entropy               약 1.394
EMA max probability       약 0.535

target entropy            약 0.561
target max probability    약 0.797
target support p50        약 17
```

이 구간은 arena가 좋아지고 있었고, target sharpness도 안정적이어서 괜찮은 상태로 봤다.

### 이후 over-convergence 구간

추가 학습 뒤 latest 지표:

```text
EMA mismatch              0.402
EMA policy CE             1.321
EMA KL(target || policy)  0.719
EMA entropy               1.368
EMA max probability       0.544

target entropy            0.602
target max probability    0.783
target support p50        19
```

replay metric은 개선됐고 target은 오히려 더 부드러워졌지만, arena에서는 `071246`에게
졌다. 따라서 즉각적인 문제는 target이 아니다. learner가 현재 replay target 분포에 너무
수렴하고 있을 가능성이 높다.

## Target Sharpness 해석

현재 target sharpness는 허용 범위다.

최근 target 지표는 좋은 방향으로 움직였다.

```text
target entropy mean       대략 0.48 -> 0.52 -> 0.56 -> 0.60
target max probability    대략 0.83 -> 0.81 -> 0.80 -> 0.78
support p50               대략 14 -> 16 -> 17 -> 19
```

또한 이 게임 규칙상 low-entropy target이 일부 많이 나오는 것은 자연스럽다.

- 상대 성 그룹을 잡으면 즉시 승리하는 수
- 자신의 성 그룹이 파괴되어 즉시 패배하는 자살수
- 성 40개를 모두 써서 pass만 가능한 상태
- pass/endgame 점수 계산으로 사실상 정해지는 상태

따라서 target entropy만 보고 실패 신호로 보면 안 된다. 더 중요한 질문은 forced state가
아닌 일반 state에서도 target이 무너지는지다. 이건 나중에 bucket별 진단으로 따로 확인하는
것이 좋다.

## 현재 가설

핵심 문제는 다음 세 가지의 균형이다.

```text
actor 데이터 생성 속도 + replay capacity + learner update 압력
```

learner 압력이 너무 높으면 모델은 replay target을 더 잘 맞추지만, resulting policy가
너무 confident해지거나 replay 분포에 특화될 수 있다. 이 경우 CE/KL이 좋아져도 arena
strength가 떨어질 수 있다.

`train_reuse_factor = 4`가 `8`보다 arena strength를 개선한 이유도 이쪽으로 해석된다.
다만 `4`에서도 충분히 오래 돌리면 local peak를 지나갈 수 있다.

## 운영 규칙

### Promote 기준

loss, CE, KL, mismatch만 보고 promote하지 않는다.

latest가 현재 best/peak를 arena에서 이길 때만 promote한다.

```text
800+ games에서 52% 이상: 강한 promote 신호
200 games에서 52-55%: 좋은 후보. 가능하면 arena 확장
50-52%: 노이즈 또는 거의 동급
50% 미만: promote 금지
```

### Snapshot 주기

local peak 근처 snapshot을 반드시 남긴다. latest는 peak를 빠르게 지나칠 수 있다.

권장:

- 현재 best 상대로 200-game arena를 이기면 snapshot 저장
- 승률이 threshold 근처면 더 큰 arena 실행
- 새 snapshot이 명확히 이길 때까지 이전 best 유지

### Replay metric은 좋아지는데 arena가 떨어질 때

latest를 계속 믿고 밀지 않는다.

권장 대응:

1. 이전 arena winner를 best로 유지한다.
2. latest와 직전 snapshot을 직접 비교한다.
3. `train_reuse_factor`를 `4`에서 `3`으로 낮추는 것을 고려한다.
4. target/search scale을 다시 건드리기 전에 LR decay를 고려한다.

target entropy가 이미 `0.55-0.60` 근처라면, arena가 떨어졌다는 이유만으로
`gumbel_c_scale`을 더 낮추지 않는다.

## 유용한 명령어

현재 replay sample 기준으로 checkpoint를 재평가한다.

```bash
python scripts/reevaluate_trajectory_policy_targets.py \
  --replay data/runpod/train-v2-gumbel-512k/replay/trajectory-replay.npz \
  --checkpoint data/runpod/train-v2-gumbel-512k/checkpoints/training-latest.pt \
  --eval-rows 16384 \
  --batch-size 1024 \
  --device cuda \
  --compare-ema \
  --pretty
```

latest를 특정 snapshot과 arena 비교한다.

```bash
great-kingdom-evaluate \
  --candidate data/runpod/train-v2-gumbel-512k/checkpoints/training-latest.pt \
  --best data/runpod/train-v2-gumbel-512k/model-snapshots/training-latest-20260515-071246.pt \
  --config configs/runpod/arena.json \
  --onnx-max-batch-size 8192 \
  --onnx-precision fp16 \
  --report data/runpod/train-v2-gumbel-512k/reports/arena/result.json
```

## 다음 권장 액션

`training-latest-20260515-071246.pt`를 현재 best 후보로 둔다.

추가 튜닝을 하기 전에 다음 중 하나를 먼저 한다.

1. `071246` versus `060810` 큰 arena로 best hierarchy를 확정한다.
2. `train_reuse_factor`를 `3`으로 낮추거나 LR decay를 적용한 뒤, 새 latest versus
   `071246` arena를 다시 본다.

현재 우선순위는 target을 더 부드럽게 만드는 것이 아니다. learner over-convergence를
제어하고, arena로 local peak를 잡는 것이다.
