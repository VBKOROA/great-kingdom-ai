# Update Pressure Tuning Runbook

`training-latest`가 특정 snapshot에게 졌을 때 바로 학습을 갈아엎지 말고,
arena 노이즈인지, raw update가 과한지, EMA가 늦는지, replay sampling이 한쪽으로 쏠리는지
분리해서 본다. 이 문서는 `scripts/diagnose_async_update_pressure.py` 결과를 보고
`configs/runpod/train.json`과 `configs/runpod/learner-v2.json`을 조정하는 기준이다.

## 1. 먼저 패배를 고정한다

학습 중 quick arena는 노이즈가 크다. 같은 snapshot에게 계속 지는지 먼저 확인한다.

```bash
great-kingdom-evaluate \
  --candidate data/runpod/train-v3/checkpoints/onnx/training-latest.onnx \
  --best data/runpod/train-v3/checkpoints/snapshots/<snapshot>.onnx \
  --report data/runpod/train-v3/reports/arena-latest-vs-<snapshot>-80.json \
  --config configs/runpod/arena.json \
  --games 80 \
  --batch-size 80 \
  --device cuda
```

가능하면 같은 비교를 seed만 바꿔 한 번 더 돌린다. `20-40`게임 quick check에서 진 결과만으로는
파라미터를 바꾸지 않는다.

## 2. 진단을 실행한다

비교 방향은 `before=<이긴 snapshot>`, `after=<진 latest>`로 둔다. 이렇게 해야 지표에서
`after`가 학습 후 모델이고, `target_kl_delta_mean`과 value delta가 악화 방향을 그대로 보여준다.

```bash
python scripts/diagnose_async_update_pressure.py \
  --replay data/runpod/train-v3/replay/trajectory-replay.npz \
  --before data/runpod/train-v3/checkpoints/snapshots/<snapshot>.pt \
  --after data/runpod/train-v3/checkpoints/training-latest.pt \
  --train-config configs/runpod/train.json \
  --learner-config configs/runpod/learner-v2.json \
  --rows-per-split 8192 \
  --batch-size 1024 \
  --device cuda \
  --pretty \
  > data/runpod/train-v3/reports/update-pressure-latest-vs-<snapshot>.json
```

최근 learner cycle에서 import된 transition 수를 알고 있으면 같이 넣는다.

```bash
python scripts/diagnose_async_update_pressure.py \
  --before data/runpod/train-v3/checkpoints/snapshots/<snapshot>.pt \
  --after data/runpod/train-v3/checkpoints/training-latest.pt \
  --imported-transitions <last_cycle_imported_transitions> \
  --pretty
```

EMA 문제를 분리하려면 세 번 비교한다.

```bash
# snapshot raw -> latest raw: 실제 optimizer update가 좋아졌는지
python scripts/diagnose_async_update_pressure.py --before <snapshot.pt> --after <latest.pt> \
  --before-weights raw --after-weights raw --pretty

# snapshot raw -> latest ema: actor/export에 실제 사용되는 EMA가 좋은지
python scripts/diagnose_async_update_pressure.py --before <snapshot.pt> --after <latest.pt> \
  --before-weights raw --after-weights ema --pretty

# latest raw -> latest ema: EMA가 raw를 얼마나 따라오고 있는지
python scripts/diagnose_async_update_pressure.py --before <latest.pt> --after <latest.pt> \
  --before-weights raw --after-weights ema --pretty
```

## 3. 먼저 볼 지표

`sampling_pressure`:

- `train_steps`: 이 learner cycle에서 실제로 도는 step 추정치다. async v2에서는 대략
  `min(train.json steps, imported_transitions * train_reuse_factor / batch_size)`다.
- `train_reuse_factor`: 새 transition 1개당 허용하는 학습 sample 수다. 클수록 새 shard 하나가
  모델을 더 많이 민다.
- `recent_fraction_config`, `recent_window_config`: recency sampling 설정이다.
- `natural_recent_fraction`: replay 안에서 최근 window가 차지하는 자연 비율이다.
- `effective_recent_fraction`: batch에서 실제 최근 row가 차지하는 비율이다.
- `recent_fraction_overweight`: 최근 window를 자연 비율보다 몇 배 더 뽑는지다.
- `recent_row_overweight`: 최근 row 1개가 old row 1개보다 몇 배 자주 뽑히는지다.

`probes`는 `all`, `old`, `recent`로 나뉜다.

- `target_kl_delta_mean = after_target_kl.mean - before_target_kl.mean`
  - 음수면 latest가 replay policy target에 더 잘 맞는다.
  - 양수면 latest가 target에서 멀어졌다.
- `value_mse_delta`, `value_mae_delta`
  - 음수면 value head가 terminal target에 더 잘 맞는다.
  - 양수면 value head가 악화됐다.
- `before_to_after_kl`, `after_to_before_kl`
  - checkpoint 사이 정책 분포 변화량이다. p95가 튀면 일부 state에서 큰 정책 점프가 난다.
- `top1_flip_rate`
  - 가장 선호하는 action이 바뀐 비율이다. `old`에서만 높으면 과거 메타 망각 신호다.
- `value_prediction_delta_abs`
  - value 예측이 checkpoint 사이 얼마나 움직였는지다.

절대 기준은 게임 단계와 replay 상태에 따라 달라진다. 판단은 `old`와 `recent`의 차이,
raw와 EMA의 차이, arena 결과가 같은 방향인지로 한다.

## 4. 패턴별 튜닝

### A. old에서만 악화되고 recent는 좋아짐

증상:

- `probes.old.target_kl_delta_mean > 0`
- `probes.old.value_mse_delta > 0` 또는 `top1_flip_rate`가 `recent`보다 뚜렷하게 높음
- `probes.recent.target_kl_delta_mean <= 0`
- `recent_fraction_overweight` 또는 `recent_row_overweight`가 큼

의미: 최신 shard 쪽으로 update pressure가 쏠려서 과거 snapshot이 잡고 있던 메타를 잊고 있다.

조정:

- `train.json`
  - `recent_sample_fraction`을 낮춘다. 이미 `0.0`이면 recency가 원인이 아니다.
  - `recent_sample_window`가 너무 작으면 키운다. 작은 window에 fraction을 주면 같은 최근 row를
    반복해서 본다.
  - `priority_target_age_weight`는 `0.0`으로 유지하거나 낮춘다.
  - `priority_alpha`를 `0.5 -> 0.3-0.4`로 낮춘다.
  - `priority_max_priority`를 `8.0 -> 4.0-6.0`으로 낮춘다.
- `learner-v2.json`
  - `replay_capacity`를 늘려 old coverage를 보존한다. 3090 24GB 기준으로 디스크/RAM 여유가
    있으면 `256000 -> 512000`을 먼저 검토한다.
  - `train_reuse_factor`를 `8.0 -> 4.0-6.0`으로 낮춰 새 shard 한 묶음이 모델을 미는 힘을 줄인다.

검증: 변경 후 같은 snapshot과 latest를 다시 arena로 비교한다. `old` probe의
`target_kl_delta_mean`이 0 근처로 내려가야 한다.

### B. all/old/recent가 모두 악화됨

증상:

- `all`, `old`, `recent`에서 `target_kl_delta_mean > 0`
- `value_mse_delta > 0`
- `before_to_after_kl.p95`, `top1_flip_rate`, `value_prediction_delta_abs.p95`가 같이 큼

의미: 한 cycle의 optimizer update 자체가 과하다. sampling 문제가 아니라 step 크기 문제일
가능성이 높다.

조정:

- `train.json`
  - `learning_rate`를 25-50% 낮춘다. 현재 기준 `0.02 -> 0.01`이 1차 후보이고, 여전히 흔들리면
    `0.005`까지 낮춘다.
  - `lr_warmup_steps`를 늘린다. 현재 `256`이면 `512`를 시험한다.
  - `gradient_clip_norm`은 `5.0`을 유지하고, 큰 점프가 계속되면 `2.0-3.0`을 시험한다.
  - `steps`를 줄인다. 현재 `512`이면 `256-384`를 시험한다.
- `learner-v2.json`
  - `train_reuse_factor`를 낮춘다. 현재 `8.0`이면 `4.0-6.0`을 시험한다.

검증: `before_to_after_kl.p95`와 `top1_flip_rate`가 줄어야 한다. arena는 같은 snapshot에 대해
최소 80게임으로 확인한다.

### C. raw는 좋아졌는데 EMA/export가 짐

증상:

- `raw -> raw` 진단은 `target_kl_delta_mean <= 0`이고 value도 개선된다.
- `raw -> ema` 또는 `latest raw -> latest ema`에서 KL 차이가 크다.
- arena는 `training-latest.onnx`를 쓰고 있고 export는 EMA를 선호한다.

의미: EMA가 raw를 너무 늦게 따라오거나, 반대로 raw가 너무 흔들려 EMA만 버티고 있다.

조정:

- raw가 arena에서도 더 강하면 `ema_decay`를 낮춘다.
  - 현재 `0.997`이면 `0.995`, 더 빠른 추종이 필요하면 `0.99`를 시험한다.
- raw가 불안정하고 EMA만 그나마 낫다면 `ema_decay`를 유지하거나 `0.998`을 시험한다.
- EMA decay는 직접 바꾸기 전에 branch 실험으로 본다.

```bash
python scripts/compare_ema_decay_branches.py \
  --source-checkpoint data/runpod/train-v3/checkpoints/training-latest.pt \
  --replay data/runpod/train-v3/replay/trajectory-replay.npz \
  --train-config configs/runpod/train.json \
  --ema-decays 0.99 0.995 0.997 0.998 \
  --steps 256 \
  --device cuda \
  --eval-device cuda \
  --pretty
```

검증: branch 결과에서 `source_raw_to_branch_ema`가 좋아지고, `branch_raw_to_branch_ema` KL이
과하지 않은 decay를 고른다.

### D. policy는 개선되는데 value가 악화됨

증상:

- `target_kl_delta_mean <= 0`
- `value_mse_delta > 0` 또는 `value_mae_delta > 0`
- arena에서 중후반 판단이 나빠진다.

의미: policy target에는 맞고 있지만 value head가 terminal target을 과하게 흔들고 있다.

조정:

- `train.json`
  - `value_loss_weight`를 `1.0 -> 0.5-0.75`로 낮춘다.
  - `priority_value_error_weight`를 `0.5 -> 0.25` 또는 `0.0`으로 낮춘다.
  - `learning_rate`도 한 단계 낮춘다.
- value만 나쁘고 policy가 크게 좋아진 경우에는 한 번에 많이 바꾸지 않는다. value loss weight와
  learning rate 중 하나를 먼저 바꾼다.

검증: `value_mse_delta`가 0 이하로 내려가고, `policy` 지표가 크게 후퇴하지 않아야 한다.

### E. priority sampling이 너무 날카로움

증상:

- `replay_metadata.sample_weights.p95` 또는 `max`가 높다.
- `before_to_after_kl.p95`는 큰데 mean은 작다.
- 일부 outlier row 때문에 policy가 국소적으로 크게 바뀐다.

의미: priority가 어려운 row를 보는 건 맞지만, 소수 row가 update를 과도하게 지배한다.

조정:

- `train.json`
  - `priority_alpha`: `0.5 -> 0.3-0.4`
  - `priority_beta`: `0.3 -> 0.4-0.6`
  - `priority_max_priority`: `8.0 -> 4.0-6.0`
  - `priority_policy_kl_weight`: `1.0 -> 0.5-0.75`
  - `priority_value_error_weight`: `0.5 -> 0.25`

검증: p95 drift가 내려가고 mean 개선이 유지되는지 본다.

### F. 진단 지표는 좋아졌는데 arena만 짐

증상:

- `target_kl_delta_mean <= 0`, `value_mse_delta <= 0`
- raw/EMA drift도 과하지 않다.
- 그래도 특정 snapshot에게 arena에서 진다.

의미: update pressure 문제가 아닐 수 있다. replay target이 특정 약점을 못 담고 있거나,
평가 설정/색상/탐색 강도 차이가 원인일 수 있다.

다음 확인:

- `scripts/diagnose_color_bias.py`로 색상 편향 확인
- `scripts/diagnose_trajectory_policy_targets.py`로 trajectory policy target 품질 확인
- `scripts/run_candidate_pairwise_matrix.py`로 여러 snapshot에 대한 상성 확인
- arena games를 늘리고, 같은 모델을 candidate/best 위치만 바꿔 평가해 위치 편향 확인

이 경우에는 learning rate나 reuse factor를 더 낮추는 것보다 replay target 품질과 self-play
다양성을 먼저 본다.

## 5. 권장 변경 폭

한 번에 하나의 축만 바꾼다. 여러 값을 동시에 바꾸면 원인을 잃는다.

| 목적 | 1차 변경 | 2차 변경 |
| --- | --- | --- |
| update가 너무 큼 | `learning_rate 0.02 -> 0.01` | `train_reuse_factor 8 -> 4-6`, `steps 512 -> 256-384` |
| old 메타 망각 | `recent_sample_fraction` 낮추기 | `replay_capacity 256000 -> 512000`, priority 완화 |
| EMA가 늦음 | `ema_decay 0.997 -> 0.995` | `0.99` branch 시험 |
| EMA가 raw 노이즈를 못 막음 | `ema_decay 0.997 -> 0.998` | learning rate/reuse도 낮춤 |
| priority outlier 과함 | `priority_alpha 0.5 -> 0.3-0.4` | `priority_max_priority 8 -> 4-6`, `priority_beta 0.4-0.6` |
| value만 악화 | `value_loss_weight 1.0 -> 0.5-0.75` | `priority_value_error_weight 0.5 -> 0.25` |
| 개선이 너무 느림 | `train_reuse_factor 8 -> 12` | `steps` 상향, 단 drift 진단 후 적용 |

## 6. 재검증 루프

1. snapshot과 latest arena 결과를 저장한다.
2. update pressure 진단 JSON을 저장한다.
3. 위 패턴 중 하나만 골라 설정을 바꾼다.
4. learner를 몇 cycle 돌린다.
5. 같은 snapshot에 대해 arena 80게임 이상을 다시 돌린다.
6. 같은 명령으로 update pressure 진단을 다시 저장한다.

결과가 좋아졌다고 판단하려면 arena 승률만 보지 말고 다음을 같이 만족해야 한다.

- `old` probe가 이전보다 덜 망가진다.
- `recent` probe 개선이 완전히 사라지지 않는다.
- raw와 EMA 사이 KL이 해석 가능한 수준으로 줄거나 유지된다.
- 바꾼 파라미터와 지표 변화가 같은 방향이다.

## 7. 빠른 결론 규칙

- latest가 old snapshot에게 지고 `old` probe만 나쁘면 recency/priority/replay capacity를 본다.
- latest가 모든 probe에서 나쁘면 learning rate, steps, train reuse를 줄인다.
- raw는 괜찮고 EMA만 나쁘면 EMA decay branch를 돌린다.
- 진단은 좋은데 arena만 나쁘면 update pressure가 아니라 target 품질, 색상 편향, snapshot 상성 문제를
  본다.
