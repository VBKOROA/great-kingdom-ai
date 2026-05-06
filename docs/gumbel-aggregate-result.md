# Gumbel Aggregate 실험 결과

## 결론

현재 측정 결과에서는 **pure Gumbel < aggregate-only Gumbel**이다.

`aggregate-only`는 Gumbel search와 policy target 설정은 pure와 동일하게 유지하고, 학습 replay에만
exact-state aggregate를 적용한 설정이다.

```text
pure:
  aggregate_replay = false
  policy_target_c_visit == gumbel_c_visit
  policy_target_c_scale == gumbel_c_scale
  policy_target_temperature = 1.0

aggregate-only:
  aggregate_replay = true
  aggregate_replay_weight_mode = sqrt_count
  aggregate_replay_weight_cap = 8.0
  policy_target_c_visit == gumbel_c_visit
  policy_target_c_scale == gumbel_c_scale
  policy_target_temperature = 1.0
```

즉 이번 결과는 target temperature/scale 수정 효과가 아니라, **replay aggregate 효과**로 해석한다.

## 측정 결과

실험 run:

```text
data/ablation/gumbel/20260506-015309
```

### Arena 1

```text
report: data/ablation/gumbel/20260506-015309/reports/aggregate-vs-pure-arena.json
games: 400
candidate: aggregate-only
best: pure
```

결과:

```text
aggregate-only wins: 243 / 400 = 60.75%
pure wins:           157 / 400 = 39.25%
average_game_length: 20.3125
promoted: true
```

색깔별:

```text
aggregate blue:   111 / 200 = 55.5%
aggregate orange: 132 / 200 = 66.0%
```

### Arena 2

Seed window만 바꿔서 추가 측정했다.

```text
report: data/ablation/gumbel/20260506-015309/reports/aggregate-vs-pure-arena-400-seed-91000.json
games: 400
candidate: aggregate-only
best: pure
```

결과:

```text
aggregate-only wins: 262 / 400 = 65.5%
pure wins:           138 / 400 = 34.5%
average_game_length: 19.715
promoted: true
```

색깔별:

```text
aggregate blue:   117 / 200 = 58.5%
aggregate orange: 145 / 200 = 72.5%
```

### 합산

```text
aggregate-only wins: 505 / 800 = 63.125%
pure wins:           295 / 800 = 36.875%
```

두 seed window 모두 aggregate-only가 60% 이상을 기록했으므로, 현재 기준에서는 aggregate replay를
채택할 근거가 충분하다.

## 이전 modified 묶음 결과와 구분

이전에 측정한 `modified` 묶음은 다음 변경을 함께 포함했다.

```text
aggregate replay
policy_target_c_visit / policy_target_c_scale 분리
policy_target_temperature > 1.0
```

그 결과는 seed에 따라 48~52%로 흔들렸고, pure 대비 우위를 확인하지 못했다.

따라서 지금 채택할 수 있는 결론은 다음과 같이 좁힌다.

```text
채택: aggregate replay
보류: policy target scale 분리
보류: policy target temperature 완화
```

## 실전 config 반영

Runpod 실전 학습 config는 pure Gumbel search/target 설정을 유지하면서 aggregate replay만 켜는 방향으로
반영했다.

```text
configs/runpod/pure-gumbel-pipeline.json
```

핵심 설정:

```json
{
  "aggregate_replay": true,
  "aggregate_replay_weight_mode": "sqrt_count",
  "aggregate_replay_weight_cap": 8.0
}
```

Gumbel target은 여전히 pure 설정이다.

```json
{
  "policy_target_temperature": 1.0,
  "policy_target_c_visit": 50.0,
  "policy_target_c_scale": 1.0
}
```

