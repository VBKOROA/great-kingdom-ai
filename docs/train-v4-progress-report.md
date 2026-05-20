# Train v4 Progress Report

작성 시점: 2026-05-20

이 문서는 `arena-anchor-playbook.md`를 근거로 삼지 않는다. train-v4에서 실제 관측한
arena/KOTH/replay 진단 결과와 현재 운영 판단만 기록한다.

## 현재 설정

주요 train-v4 설정:

- `work_dir`: `data/runpod/train-v4`
- `policy_target_c_scale`: `1.0`
- `gumbel_c_scale`: `1.0`
- `learning_rate`: 처음 `0.0125`, 이후 `0.005`로 하향
- `train_reuse_factor`: 처음 `8.0`, 이후 `4.0` 실험
- `ema_decay`: `0.999`
- `priority_enabled`: `true`
- `priority_alpha`: `0.5`
- `priority_beta`: `0.3`
- `priority_max_priority`: `8.0`
- `recent_sample_fraction`: `0.0`
- `recent_sample_window`: `102400`

중요한 점:

- `train_reuse_factor`는 최근 shard만 골라서 학습하는 파라미터가 아니다.
- learner는 imported transition 수에 reuse factor를 곱해 train sample budget을 만들고,
  실제 batch는 replay 전체에서 샘플링한다.
- 현재 `recent_sample_fraction=0.0`이므로 최근 window를 강제로 더 뽑지 않는다.
- priority sampling은 켜져 있지만 `priority_target_age_weight=0.0`이므로 age/recent 기준
  가중은 없다.

## Actor / Learner 데이터 흐름 요약

actor는 shard별로 self-play 결과를 만든다.

- shard output: `trajectory-replay.npz`, `game_logs.json`
- metadata에 `shard_completed` event를 append

continuous learner는 cycle마다 다음처럼 동작한다.

- pending completed shard를 모두 import
- replay에 append 후 capacity에 맞게 유지
- `imported_transitions * train_reuse_factor`만큼 train budget 증가
- `floor(budget / batch_size)`만큼 train step 수행
- `train_config.steps` cap 적용

따라서 한 learner cycle은 `8 actors = 8 shards`가 아니다. cycle 시작 시점에 완료된 shard 수에
따라 학습량이 달라진다.

## LR 0.0125 / Reuse 8 관측

초기 train-v4는 `lr=0.0125`, `reuse=8`로 시작했다.

관측:

- `112022` snapshot은 latest-best 상대로 `32/4` arena에서 `40%`
- 같은 `112022`는 `192/8` arena에서 `47%`
- update-pressure 진단에서 after 모델이 replay target에는 더 가까워졌지만 arena strength는
  떨어졌다.

대표 진단:

- target policy entropy mean: 약 `0.10`
- target policy max mean: 약 `0.96`
- target policy max p50: 거의 `1.0`
- latest-best 대비 top1 flip rate: 약 `26-29%`
- latest-best 대비 policy KL mean: 약 `0.42-0.46`

판단:

- target은 매우 sharp하다.
- 모델은 replay target을 더 잘 따라갔지만 그 변화가 바로 strength 개선으로 이어지지 않았다.
- `policy_target_c_scale=1.0` 자체 실패라기보다, `lr=0.0125`에서 sharp target을 너무 빠르게
  흡수한 쪽이 더 그럴듯했다.

## LR 0.005 하향 후 관측

`learning_rate`를 `0.005`로 낮춘 뒤 강한 후보가 나오기 시작했다.

주요 결과:

- `115036` vs latest-best: `200 games`, candidate `48%`
- `115538` vs latest-best: `400 games`, candidate `65.25%`, promoted
- `115538` confirm vs latest-best: `400 games`, candidate `68.75%`, promoted

그러나 KOTH/matrix에서는 `115538`이 항상 가장 robust하지는 않았다.

예시 KOTH:

- latest-best / `100944`: average `53.5%`
- `115036`: average `53.375%`, worst `44.5%`
- `115538`: average `52.5625%`, worst `42.5%`

판단:

- LR 하향은 좋은 방향이었다.
- 다만 최신 snapshot이 단조롭게 강해지는 상태는 아니었다.
- `115538`은 latest-best 직접전에서는 강했지만 KOTH robustness는 상대적으로 애매했다.

## Reuse 4 관측

이후 `train_reuse_factor=4.0`을 실험했다.

첫 reuse 4 KOTH:

- `124601`: average `62.125%`, best `73%`, worst `55.5%`
- 하지만 latest-best 직접전 `400 games`에서는 candidate `49.25%`
- side split: Blue `57%`, Orange `41.5%`

다음 reuse 4 KOTH:

- `131615`: average `57.375%`, best `78.5%`, worst `47.5%`
- KOTH side split: Blue `48.25%`, Orange `66.5%`

`131615` latest-best 직접전:

- `400 games`
- candidate win rate: `72%`
- Blue: `138/200 = 69%`
- Orange: `150/200 = 75%`
- promoted: true

판단:

- `reuse=4`는 snapshot 간 policy drift를 확실히 줄였다.
- `131113 -> 132117` 진단에서:
  - model-to-model KL mean: 약 `0.0137`
  - top1 flip rate: 약 `9.8%`
  - value prediction delta abs mean: 약 `0.03`
- 이는 `latest-best -> 115538` 비교의 KL `~0.42`, top1 flip `~28-29%`보다 훨씬 작다.
- 따라서 update pressure 완화 효과는 확인됐다.
- 다만 KOTH side split과 latest-best 직접전 side split이 항상 같은 방향으로 나오지는 않는다.
  KOTH의 side skew는 matchup 조합 영향을 크게 받는다.

## Replay 색깔 편향 진단

`scripts/diagnose_color_bias.py`로 replay winner 분포를 확인했다.

전체 replay:

- games: `19270`
- Blue wins: `9635`
- Orange wins: `9635`
- Blue win rate: `50%`
- Orange win rate: `50%`
- winner imbalance: `0.0`

최근 window:

- last 100 games: Blue `38%`, Orange `62%`
- last 500 games: Blue `45.4%`, Orange `54.6%`
- last 1000 games: Blue `46.6%`, Orange `53.4%`
- last 5000 games: Blue `48.86%`, Orange `51.14%`

row percent 기준:

- last 1% rows: Blue `40.4%`, Orange `59.6%`
- last 5% rows: Blue `46.3%`, Orange `53.7%`
- last 10% rows: Blue `46.9%`, Orange `53.1%`

판단:

- 전체 replay에는 색깔 winner 편향이 없다.
- 최근 tail에는 Orange winner가 더 많은 구간이 있었다.
- 하지만 현재 sampler가 최근 tail을 강제로 overweight하지 않으므로, 이것만으로 side skew를
  설명하기는 어렵다.

## Priority Sampling Mass 진단

`sampling_priorities` 기준 실제 sampling mass를 확인했다. 현재 `priority_alpha=0.5`이므로
실제 mass는 대략 `priority ** 0.5`에 비례한다.

전체 sampling mass:

- episode winner Blue: `49.28%`
- episode winner Orange: `50.72%`
- row player Blue: `50.47%`
- row player Orange: `49.53%`
- winning-player rows: `51.14%`
- losing-player rows: `48.86%`

판단:

- priority sampling이 특정 색깔을 크게 더 뽑고 있지는 않다.
- Orange winner mass가 약 `+1.4%p` 정도로, 최근 arena side skew를 설명할 정도는 아니다.

Priority cap 포화:

- priority `>= 7.0`: `48.15%`
- priority `>= 7.5`: `25.30%`
- priority `>= 7.9`: `22.52%`
- priority `>= 7.99`: `22.48%`
- mean: `6.84`
- p50: `6.93`
- p95: `8.0`
- max: `8.0`

판단:

- `priority_max_priority=8.0` cap 포화는 꽤 심하다.
- replay row의 약 22%가 사실상 cap 근처라 상위 priority 구간의 구분력이 떨어졌을 가능성이
  있다.
- 하지만 지금까지의 문제는 학습 부족보다 snapshot 출렁임/side skew에 가까웠다.
- 따라서 `priority_max_priority`를 바로 키우는 것은 추천하지 않는다.
- 안정성 목적이면 `priority_max_priority=8.0` 유지가 낫다.
- priority 실험을 한다면 cap 상향보다 `priority_alpha=0.3`처럼 priority 효과를 완화하는 쪽이
  더 안전한 후보이다.

## 현재 결론

현재까지의 결론:

- `policy_target_c_scale=1.0`은 유지 가능하다.
- `lr=0.0125`는 sharp target에 비해 너무 컸던 것으로 보인다.
- `lr=0.005` 하향은 명확히 좋은 방향이었다.
- `reuse=8`에서도 강한 후보는 나왔지만 snapshot drift가 컸다.
- `reuse=4`는 model-to-model drift를 줄였고, `131615`처럼 latest-best를 확실히 이기는
  후보도 만들었다.
- 최신 snapshot이 항상 1등이 아니어도 설정 실패로 보지 않는다.
- 현재 운영은 latest snapshot promote가 아니라 KOTH/rolling matrix로 후보를 고른 뒤
  latest-best 직접전으로 검증하는 방식이 맞다.

## Promote 기준

candidate를 promote하려면 최소한 다음을 본다.

- latest-best 직접전 `400 games`
- total win rate `> 50%`
- Blue/Orange side win rate 모두 `> 50%`
- 가능하면 seed를 바꾼 confirm run

KOTH/matrix winner만으로 promote하지 않는다.

KOTH는 후보 선별용이고, latest-best 직접전은 promote 검증용이다.

## 다음 운영 기준

현재 기본 운영:

- `lr=0.005`
- `train_reuse_factor=4.0`
- `policy_target_c_scale=1.0`
- `priority_max_priority=8.0`
- `priority_alpha=0.5` 유지

rolling matrix 판단:

- 최근 snapshot이 1등이 아니라는 이유만으로 설정을 바꾸지 않는다.
- 최근 window의 평균 strength와 peak strength를 같이 본다.
- peak가 나오는데 최신 유지가 안 되는 것은 snapshot selection 문제일 수 있다.
- 모든 후보가 약해지고 peak도 사라지면 설정 변경을 검토한다.

설정 변경 후보:

- snapshot 변동성이 다시 커지면 `reuse` 추가 하향을 검토할 수 있다.
- priority cap 포화가 문제가 된다고 판단되면 `priority_alpha=0.3` 실험을 우선 고려한다.
- `priority_max_priority` 상향은 high-error row를 더 강하게 반복 학습시켜 변동성을 키울 수
  있으므로 별도 branch 실험으로만 다룬다.

## 현재 가장 중요한 관찰

`131615`는 latest-best 직접전에서 `72%`를 기록했고, Blue/Orange 모두 `50%`를 크게 넘었다.
따라서 현재 train-v4는 `lr=0.005`, `reuse=4`, `policy_target_c_scale=1.0` 조합에서
실제 strength 개선 후보를 만들고 있다.

남은 문제는 최신 snapshot이 항상 최고가 아니라는 점이다. 이것은 지금 단계에서는 설정 실패보다
snapshot selection / promotion policy 문제로 보는 것이 더 타당하다.
