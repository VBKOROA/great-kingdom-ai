# Arena Results 2026-05-18

2026-05-18 Runpod 학습 중 확인한 snapshot arena 결과 기록이다.

## 요약

- `training-latest-20260518-120827.pt`는 `training-latest-20260518-115320.pt`에게
  `46.5%`로 졌다. 이 시점은 regression으로 판단한다.
- 같은 구간의 update pressure 진단에서는 `120827`이 replay target에는 더 잘 맞았다.
  따라서 단순 optimizer/update pressure 문제가 아니라 replay policy target 품질과 arena
  strength의 정렬이 깨진 것으로 판단했다.
- self-play sample target 품질을 올리기 위해 full search simulation을 `128`로 올린 뒤,
  `123841`, `124344`, `125351` snapshot에서 arena strength가 다시 회복됐다.
- `training-latest-20260518-124344.pt`는 regression 확인용 anchor였던
  `training-latest-20260518-115320.pt`에게도 `57.5%`로 이겼다. 현재는 `124344`를 새 strong
  anchor로 보는 것이 타당하다.
- 이후 adjacent snapshot 비교는 `50%` 근처로 얇아졌지만, `training-latest-20260518-130858.pt`가
  `124344`에게 `57.5%`로 이겼다. 따라서 전체 strength는 아직 상승 중이고, 5분 간격 adjacent
  arena는 노이즈에 묻히기 시작한 것으로 본다.

## Arena 결과

| candidate | best/anchor | games | candidate win rate | candidate wins | blue wins | orange wins | avg length | 판단 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `training-latest-20260518-115320.pt` | `training-latest-20260518-111803.pt` | 200 | 67.5% | 135 | 66/100 | 69/100 | 46.675 | 우세 |
| `training-latest-20260518-115822.pt` | `training-latest-20260518-111803.pt` | 200 | 64.5% | 129 | 66/100 | 63/100 | 44.660 | 우세 |
| `training-latest-20260518-120325.pt` | `training-latest-20260518-112306.pt` | 200 | 60.0% | 120 | 61/100 | 59/100 | 49.365 | 우세지만 margin 감소 |
| `training-latest-20260518-120827.pt` | `training-latest-20260518-112808.pt` | 200 | 58.5% | 117 | 52/100 | 65/100 | 47.685 | 우세지만 margin 감소 |
| `training-latest-20260518-120827.pt` | `training-latest-20260518-115320.pt` | 200 | 46.5% | 93 | 44/100 | 49/100 | 48.540 | regression |
| `checkpoints/onnx/training-latest.onnx` | `training-latest-20260518-115320.pt` | 200 | 57.0% | 114 | 55/100 | 59/100 | 48.545 | sim128 이후 회복 신호 |
| `training-latest-20260518-123841.pt` | `training-latest-20260518-113813.pt` | 200 | 66.5% | 133 | 66/100 | 67/100 | 47.410 | 회복 |
| `training-latest-20260518-124344.pt` | `training-latest-20260518-113813.pt` | 200 | 74.5% | 149 | 79/100 | 70/100 | 46.900 | 강한 회복 |
| `training-latest-20260518-124344.pt` | `training-latest-20260518-115320.pt` | 200 | 57.5% | 115 | 62/100 | 53/100 | 48.065 | 이전 peak 돌파 |
| `training-latest-20260518-125351.pt` | `training-latest-20260518-114315.pt` | 200 | 71.0% | 142 | 71/100 | 71/100 | 44.115 | 강한 우세 유지 |
| `training-latest-20260518-125351.pt` | `training-latest-20260518-124344.pt` | 200 | 54.0% | 108 | 60/100 | 48/100 | 50.240 | 얇은 개선 |
| `training-latest-20260518-125854.pt` | `training-latest-20260518-125351.pt` | 200 | 52.5% | 105 | 51/100 | 54/100 | 50.330 | 얇은 개선 |
| `training-latest-20260518-130356.pt` | `training-latest-20260518-125854.pt` | 200 | 50.0% | 100 | 56/100 | 44/100 | 48.270 | adjacent plateau |
| `training-latest-20260518-130858.pt` | `training-latest-20260518-130356.pt` | 200 | 49.5% | 99 | 52/100 | 47/100 | 49.130 | adjacent noise/plateau |
| `training-latest-20260518-130858.pt` | `training-latest-20260518-124344.pt` | 200 | 57.5% | 115 | 58/100 | 57/100 | 49.840 | 누적 strength 상승 |

## 관찰

초반에는 절반 전 snapshot 기준 candidate 승률이 높았지만, 시간이 지날수록 margin이 줄었다.

```text
115320 vs 111803: 67.5%
115822 vs 111803: 64.5%
120325 vs 112306: 60.0%
120827 vs 112808: 58.5%
```

이후 stronger anchor인 `115320`과 직접 비교했을 때 `120827`이 `46.5%`로 졌다.
따라서 단순 plateau가 아니라 실제 strength regression으로 본다.

```text
120827 vs 115320: 46.5%
```

self-play full search simulation을 `128`로 올린 뒤에는 strength가 다시 회복됐다.

```text
latest after sim128 vs 115320: 57.0%
123841 vs 113813: 66.5%
124344 vs 113813: 74.5%
124344 vs 115320: 57.5%
125351 vs 114315: 71.0%
125351 vs 124344: 54.0%
125854 vs 125351: 52.5%
130356 vs 125854: 50.0%
130858 vs 130356: 49.5%
130858 vs 124344: 57.5%
```

5분 전후 adjacent snapshot끼리는 차이가 작아서 `50%` 근처 결과가 나오기 시작했다. 그러나
`130858`을 `124344` anchor에 붙이면 `57.5%`로 양쪽 색상 모두 이겼다. 이후에는 너무 가까운
snapshot끼리 비교하기보다 strong anchor 대비 누적 strength를 보는 편이 정보량이 높다.

## 관련 update pressure 진단

비교:

```text
before: training-latest-20260518-115320.pt, EMA
after:  training-latest-20260518-120827.pt, EMA
```

핵심 지표:

| probe | target_kl_delta_mean | value_mse_delta | value_mae_delta | top1_flip_rate |
| --- | ---: | ---: | ---: | ---: |
| all | -0.1339 | -0.0147 | -0.0140 | 22.5% |
| old | -0.1372 | -0.0162 | -0.0149 | 23.1% |
| recent | -0.1377 | -0.0141 | -0.0136 | 23.5% |

sampling pressure:

```text
train_reuse_factor: 8.0
recent_fraction_config: 0.0
recent_fraction_overweight: 1.0
recent_row_overweight: 1.0
sample_weights: all 1.0
```

해석:

- `120827`은 `115320`보다 replay policy target KL이 전 구간에서 낮다.
- value MSE/MAE도 전 구간에서 개선됐다.
- recency oversampling이나 priority outlier는 확인되지 않았다.
- 그런데 arena에서는 `120827`이 `115320`에게 졌다.

따라서 이 regression은 learner가 replay를 못 맞춘 문제가 아니라, 당시 replay target을 더 잘
맞출수록 arena strength가 떨어지는 문제로 해석한다. 이후 full search simulation을 `128`로 올렸고,
그 뒤 arena 결과가 회복됐다.

## 현재 운영 기준

- `training-latest-20260518-115320.pt`는 regression 확인용 old anchor로 유지한다.
- `training-latest-20260518-124344.pt`는 새 strong anchor로 유지한다.
- `training-latest-20260518-125351.pt`는 절반 전 snapshot 기준 `71.0%`로 강한 우세를 유지했다.
- `training-latest-20260518-130858.pt`는 `124344` 대비 `57.5%`로 이겼으므로 strong anchor 후보로
  유지한다.
- 이후 검증은 5분 간격 adjacent 비교보다 `latest vs 124344` 또는 `latest vs 130858` 중심으로 한다.
- 새 anchor 교체는 strong anchor 대비 `57-60%+` 결과가 반복될 때 한다.
- self-play simulation `128` 변경 후 결과가 좋아지고 있으므로, 당분간 learner parameter는 추가로
  흔들지 않는다.

권장 다음 평가:

```bash
great-kingdom-evaluate \
  --candidate data/runpod/train-v3/checkpoints/onnx/training-latest.onnx \
  --best data/runpod/train-v3/snapshots/training-latest-20260518-130858.pt \
  --config configs/runpod/arena.json \
  --onnx-max-batch-size 2048 \
  --onnx-precision fp16 \
  --report data/runpod/train-v3/reports/arena-latest-vs-130858.json
```
