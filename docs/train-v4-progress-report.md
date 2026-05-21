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
- self-play actor ONNX export: raw weights 사용 (`onnx_prefer_ema=false`)

중요한 점:

- `train_reuse_factor`는 최근 shard만 골라서 학습하는 파라미터가 아니다.
- learner는 imported transition 수에 reuse factor를 곱해 train sample budget을 만들고,
  실제 batch는 replay 전체에서 샘플링한다.
- 현재 `recent_sample_fraction=0.0`이므로 최근 window를 강제로 더 뽑지 않는다.
- priority sampling은 켜져 있지만 `priority_target_age_weight=0.0`이므로 age/recent 기준
  가중은 없다.
- `ema_decay=0.999`는 checkpoint 안의 shadow weights를 유지한다는 뜻이고, actor가 쓰는
  `training-latest.onnx`는 기본적으로 raw training weights에서 export한다.

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

## 2026-05-21 Self-play EMA Actor 비활성화

train-v4 운영에서 self-play actor가 쓰는 ONNX를 EMA weights가 아니라 raw training weights로
export하도록 바꿨다.

변경 내용:

- `configs/runpod/learner-v2.json`: `onnx_prefer_ema=false`
- async v2 learner/factory ONNX export 기본값: `prefer_ema=false`
- EMA model은 checkpoint 내부 shadow weights로 계속 유지
- actor가 읽는 `checkpoints/onnx/training-latest.onnx`만 raw 최신 정책을 반영

판단:

- 현재 구조는 9x9 Gumbel 계열 self-play이고, 저장 샘플은 전체 포지션이 아니라
  `192 simulations / top-k 8` full-search 구간에 집중된다.
- 이 구조에서는 actor prior가 최신 raw policy를 얼마나 빨리 반영하느냐가 중요하다.
- replay buffer 자체가 이미 과거 self-play sample을 포함하므로, actor까지 EMA를 쓰면
  effective policy age가 더 늙는다.
- 특히 top-k 후보군이 좁기 때문에 raw model이 새로 배운 좋은 수나 refutation이 EMA prior에
  늦게 반영되면, search가 애초에 그 후보를 검증하지 못할 수 있다.
- 따라서 현재 기본값은 self-play actor raw, EMA는 checkpoint smoothing/eval/export 후보로
  남기는 쪽이 더 타당하다.

내 반응:

- 이 변경은 현재 train-v4 구조에서는 합리적이다.
- EMA self-play가 Gumbel search를 직접 망가뜨린다기보다, Gumbel search가 의존하는 최신
  policy prior 기반 후보 선별을 늦추는 비용이 더 커 보인다.
- 이미 replay와 priority sampling, full-search 저장 구간이 안정화 역할을 하고 있으므로 actor
  쪽까지 EMA로 늦추는 것은 지나치게 보수적인 기본값일 수 있다.
- 다만 raw actor 전환 후 value loss 장기 폭등, policy entropy 붕괴, 평균 게임 길이 급변,
  self-play 품질 붕괴가 보이면 EMA actor를 다시 비교 후보로 둔다.

## 현재 결론

현재까지의 결론:

- `policy_target_c_scale=1.0`은 유지 가능하다.
- `lr=0.0125`는 sharp target에 비해 너무 컸던 것으로 보인다.
- `lr=0.005` 하향은 명확히 좋은 방향이었다.
- `reuse=8`에서도 강한 후보는 나왔지만 snapshot drift가 컸다.
- `reuse=4`는 model-to-model drift를 줄였고, `131615`처럼 latest-best를 확실히 이기는
  후보도 만들었다.
- self-play actor는 raw latest ONNX를 쓰는 기본값이 더 타당하다.
- EMA는 계속 유지하되, actor 기본 policy가 아니라 smoothing/eval/export 후보로 본다.
- raw actor 전환 후 `045409`가 anchor matrix 1등을 기록했으므로, raw actor가 즉시
  self-play 품질을 붕괴시켰다는 가설은 현재 우선순위가 낮다.
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

- `lr=0.003` 실험 중
- `train_reuse_factor=4.0`
- `policy_target_c_scale=1.0`
- `priority_max_priority=8.0`
- `priority_alpha=0.3`으로 하향
- self-play actor ONNX는 raw weights 사용 (`onnx_prefer_ema=false`)
- EMA shadow weights는 유지

rolling matrix 판단:

- 최근 snapshot이 1등이 아니라는 이유만으로 설정을 바꾸지 않는다.
- 최근 window의 평균 strength와 peak strength를 같이 본다.
- peak가 나오는데 최신 유지가 안 되는 것은 snapshot selection 문제일 수 있다.
- 모든 후보가 약해지고 peak도 사라지면 설정 변경을 검토한다.
- recent matrix는 최신 후보 선별용이고, anchor matrix는 robust 후보 유지용이다.
- anchor pool은 상위 4개를 유지한다.
- recent matrix 1등이 anchor matrix에서 기존 anchor를 못 넘으면 promote 후보로 보지 않는다.

설정 변경 후보:

- 다음 recent winner도 anchor matrix에서 `045409`를 넘지 못하면 먼저 update-pressure 진단을
  수행한다.
- after 모델이 replay target에는 더 가까운데 arena에서 약하면 `learning_rate`를 `0.005`에서
  `0.003` 또는 `0.0025`로 낮추는 것을 우선 검토한다.
- priority cap 포화가 계속 심하면 `priority_alpha=0.3` 유지 결과를 먼저 본다.
- `priority_alpha=0.3`에서도 anchor robustness가 개선되지 않으면 `priority_enabled=false`
  branch를 별도 실험 후보로 둔다.
- learner가 actor보다 앞서는 문제가 명확하면 `train_reuse_factor=2.0`을 검토한다.
- `priority_max_priority` 상향은 high-error row를 더 강하게 반복 학습시켜 변동성을 키울 수
  있으므로 별도 branch 실험으로만 다룬다.

## 2026-05-21 Rolling / Anchor Matrix 업데이트

최근 5개 snapshot matrix에서 두 번의 peak가 관측됐다.

첫 recent matrix:

- `045409`: average `53.75%`, worst `38.33%`, Blue `60.0%`, Orange `47.5%`
- `043839`: average `53.33%`, worst `40.0%`, Blue `45.0%`, Orange `61.67%`

두 번째 recent matrix:

- `050413`: average `57.5%`, worst `50.0%`, Blue `53.33%`, Orange `61.67%`
- `052423`: average `50.83%`, worst `43.33%`, Blue `36.67%`, Orange `65.0%`
- `051920`: average `44.17%`, non-losing `0/4`

판단:

- recent matrix만 보면 `050413`이 가장 좋아 보였다.
- 하지만 latest가 단조롭게 좋아지는 형태는 아니며, `050413` 이후 snapshot들은 다시 흔들렸다.
- 이 패턴은 학습 전체 붕괴라기보다 peak 이후 drift/oscillation으로 보는 것이 더 타당하다.

anchor matrix 결과:

- `045409`: average `59.17%`, worst `55.0%`, winning `4/4`, Blue `55.83%`,
  Orange `62.5%`
- `050413`: average `50.42%`, worst `45.0%`, winning `1/4`, Blue `58.33%`,
  Orange `42.5%`
- `131615`: average `50.42%`, worst `40.0%`, winning `2/4`, Blue `58.33%`,
  Orange `42.5%`
- `124601`: average `47.5%`, worst `31.67%`
- `120543`: average `42.5%`, worst `33.33%`

판단:

- `045409`가 현재 anchor pool의 확실한 1등이다.
- `050413`은 recent peers 상대로는 강했지만 anchor pool에서는 robust winner가 아니었다.
- 현재 anchor top4 유지 기준이면 `120543`을 제거하고 `050413`을 편입할 수 있다.
- promote/direct confirm 후보는 현재 `045409` 하나로 본다.
- `045409` 이후 최신 후보들이 아직 `045409`를 넘지 못하고 있으므로, 다음 window에서도
  신규 후보가 anchor matrix에서 밀리면 update-pressure 진단을 먼저 수행한다.

## 2026-05-21 LR / Priority Follow-up

`045409` 이후 recent winner들이 계속 anchor matrix에서 `045409`를 넘지 못했다.

추가 recent matrix:

- `052925`: average `56.25%`, worst `41.67%`, Blue `55.0%`, Orange `57.5%`
- `053427`: average `55.42%`, worst `51.67%`, Blue `65.0%`, Orange `45.83%`
- `054934`: average `52.08%`, worst `38.33%`

anchor matrix에서는 다시 `045409`가 1등이었다.

- `045409`: average `58.33%`, worst `51.67%`, winning `4/4`, Blue `62.5%`,
  Orange `54.17%`
- `052925`: average `47.92%`, worst `35.0%`, winning `1/4`

`045409` vs `052925` update-pressure 진단:

- EMA 기준 target KL delta mean: all `-0.051`, old `-0.052`, recent `-0.046`
- raw 기준 target KL delta mean: all `-0.041`, old `-0.048`, recent `-0.033`
- raw 기준 top1 flip rate: 약 `26%`
- raw 기준 model-to-model KL mean: 약 `0.12-0.13`
- value 개선은 없거나 매우 작음

판단:

- `052925`는 live replay target에는 `045409`보다 더 가까웠지만, anchor arena에서는 약했다.
- 이 결과는 value collapse라기보다 sharp policy target을 더 맞추는 방향이 strength 개선으로
  이어지지 않은 상황에 가깝다.
- raw actor가 직접 붕괴 원인이라는 가설은 계속 낮은 우선순위다.
- 먼저 `lr=0.003` 구간을 관찰하고, 동시에 priority 효과 완화를 실험한다.

priority cap 재진단:

- priority `>= 7.0`: `48.53%`
- priority `>= 7.5`: `25.37%`
- priority `>= 7.9`: `22.47%`
- priority `>= 7.99`: `22.43%`
- mean: `6.85`
- p50: `6.95`
- p95: `8.0`
- max: `8.0`

판단:

- priority cap 포화는 여전히 심하다.
- replay row 약 22%가 사실상 max priority 근처라 high-priority 구간 구분력이 떨어져 있다.
- cap을 올리는 것은 high-error row를 더 강하게 밀어 변동성을 키울 수 있으므로 우선하지 않는다.
- `priority_alpha`를 `0.5`에서 `0.3`으로 낮춰 priority sampling 효과를 완화했다.
- `priority_alpha=0.3`의 성공 기준은 recent matrix 최고 평균이 아니라 anchor matrix에서
  `045409`에 근접하거나 넘는지, worst win rate와 side balance가 개선되는지이다.

LR comparison matrix:

- `055436`: average `53.0%`, worst `45.0%`, Blue `62.0%`, Orange `44.0%`
- `061445`: average `51.0%`, worst `42.5%`, Blue `51.0%`, Orange `51.0%`
- `060943`: average `50.5%`, worst `47.5%`, Blue `58.0%`, Orange `43.0%`
- `061948`: average `50.0%`, worst `47.5%`, Blue `53.0%`, Orange `47.0%`

해석:

- `0.003` 구간은 강한 peak를 바로 만들지는 못했지만, 일부 후보는 side balance가 나아졌다.
- 다만 anchor matrix에서는 여전히 `045409`가 average `63.67%`로 압도적이었다.
- 따라서 현재까지는 `045409`가 유지 champion이고, `priority_alpha=0.3` 이후 snapshot을 추가
  관찰한다.

## 현재 가장 중요한 관찰

`045409`는 anchor matrix에서 average `59.17%`, worst `55.0%`, Blue/Orange 모두 `50%`
이상을 기록했다. 따라서 raw actor 전환 후에도 train-v4는 robust peak 후보를 만들고 있다.

남은 문제는 `045409` 이후 최신 snapshot들이 그 peak를 계속 넘지 못하고 있다는 점이다. 이것은
지금 단계에서는 raw actor 실패보다 sharp policy target, update pressure, priority sampling 포화,
그리고 snapshot selection 문제로 보는 것이 더 타당하다. 현재는 `lr=0.003`과
`priority_alpha=0.3` 조합의 후속 snapshot을 관찰한다.
