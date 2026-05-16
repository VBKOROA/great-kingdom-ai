# Recent Replay Color Bias 진단

## 목적

2026-05-16 Runpod `train-v2-gumbel-512k-adamw3e4-cscale01-r16` run에서 관찰된 arena 흔들림과
recent replay sampling 문제를 정리한다.

이 문서는 해법 확정 문서가 아니라, 현재 문제의 형태와 다음 실험 기준을 남기기 위한 진단 메모다.

## 현재 증상

최근 checkpoint 간 arena 결과가 전체 승률뿐 아니라 Blue/Orange side별 승률에서 크게 흔들린다.

대표 사례:

```text
075941 vs 073931:
  candidate 35.5%, blue 5/100, orange 66/100

082456 vs 081451:
  candidate 41.5%, blue 83/100, orange 0/100

083501 vs 081451:
  candidate 64.5%, blue 30/100, orange 99/100

102244 vs 081451:
  candidate 52.5%, blue 77/100, orange 28/100

111311 vs 081451:
  candidate 70.0%, blue 43/100, orange 97/100
```

paired seed arena를 적용한 뒤에도 side skew가 사라지지 않았다. 따라서 단순 seed mismatch나 평가 노이즈만으로
보기 어렵다.

## Replay 전체와 최근 구간의 차이

`scripts/diagnose_color_bias.py`로 trajectory replay를 진단했다.

전체 replay 기준:

```text
games: 40717
blue_win_rate: 47.81%
orange_win_rate: 52.19%
winner_imbalance_abs: 4.38%
```

최근 row 기준:

```text
recent 1% rows:
  blue 43.78%, orange 56.22%, imbalance 12.44%

recent 5% rows:
  blue 43.27%, orange 56.73%, imbalance 13.47%

recent 10% rows:
  blue 42.86%, orange 57.14%, imbalance 14.29%

recent 20% rows:
  blue 44.33%, orange 55.67%, imbalance 11.34%

recent 30% rows:
  blue 45.50%, orange 54.50%, imbalance 9.01%

recent 50% rows:
  blue 46.81%, orange 53.19%, imbalance 6.37%
```

즉 replay 전체는 약한 Orange 우세 정도지만, 최근 10~20% 구간은 Orange 승리 데이터가 강하게 치우쳐 있다.

## 현재 해석

현재 문제는 "게임 자체가 항상 한 색에 유리하다"라기보다 다음 조합에 가깝다.

```text
recent replay winner/color skew
+ recent sampling overweight
+ matchup cyclicity
= 특정 side/matchup에 강하게 특화된 checkpoint가 자주 생성됨
```

기존 설정이 예를 들어 다음과 같다면:

```text
replay_capacity ~= 512000 rows
recent_sample_window = 51200 rows
recent_sample_fraction = 0.25
```

최근 10% row가 batch의 25%를 차지한다. 이 경우 최근 row는 균일 샘플링 대비 대략 3배 정도 자주 학습된다.
최근 10%가 Orange 승리 쪽으로 57:43까지 치우쳐 있으면, learner가 이 분포를 강하게 따라갈 수 있다.

## 실험에서 확인된 것

### recent_sample_fraction=0은 과교정이었다

recent oversampling을 완전히 끄면 최신 target density가 부족해지는 쪽으로 무너졌다.

```text
104756 vs latest-best:
  candidate 10.0%, blue 14/100, orange 6/100

105258 vs 081451:
  candidate 24.0%, blue 36/100, orange 12/100
```

따라서 `recent_sample_fraction=0`은 현재 run에서는 기본 해법으로 보기 어렵다.

### recent window 확대는 strength 회복에 도움이 됐다

`recent_sample_fraction=0.25`를 유지하고 `recent_sample_window`를 키우는 실험은 붕괴를 줄였다.

```text
110809 vs latest-best:
  candidate 53.5%, blue 50/100, orange 57/100

111311 vs latest-best:
  candidate 85.5%, blue 96/100, orange 75/100
```

다만 counter-anchor인 `081451` 기준으로는 여전히 side skew가 남았다.

```text
113321 vs 081451:
  candidate 52.5%, blue 27/100, orange 78/100
```

따라서 window 확대는 도움이 되지만, matchup cyclicity 자체를 제거하는 근본 해법은 아니다.

## 현재 권장 기본선

당장 설정만으로 진행할 경우:

```text
learning_rate: 2e-5 근처 유지
recent_sample_fraction: 0.25 유지 또는 0.125 실험
recent_sample_window: 102400 ~ 153600 범위
recent_sample_fraction=0: 기본값으로 사용하지 않음
```

해석 기준:

- `latest-best.pt` 기준 전체 승률을 본다.
- `081451` 같은 counter-anchor도 별도로 본다.
- 전체 승률만 보지 말고 `candidate_blue_wins`, `candidate_orange_wins`를 같이 본다.
- 한쪽 side가 20~30% 이하로 무너지면 promotion 여부와 별개로 matchup skew로 기록한다.

## 더 직접적인 개선 후보

sampling hyperparameter만으로는 최근 winner/color skew를 완전히 제거하기 어렵다.

더 직접적인 후보:

1. recent sampler에서 winner balance를 맞춘다.
   - recent 구간에서 Blue-win episode와 Orange-win episode를 50:50에 가깝게 뽑는다.
   - row 기준으로 구현할 경우 episode length 차이를 함께 고려해야 한다.

2. inverse winner-frequency weighting을 적용한다.
   - recent window 안에서 Blue/Orange winner 빈도의 역수로 sample weight를 조정한다.
   - 지나치게 강한 보정은 실제 메타 신호까지 지울 수 있으므로 cap이 필요하다.

3. arena gate에 side별 조건을 추가한다.
   - 예: overall 승률 조건 외에 `min(candidate_blue_win_rate, candidate_orange_win_rate)` 하한을 둔다.
   - rollback을 하지 않는 구조라면 최소한 promotion/report 판단에서 별도 플래그로 남긴다.

4. opponent/counter anchor pool을 둔다.
   - `latest-best.pt` 단일 비교만으로는 RPS 형태의 matchup cycle을 놓칠 수 있다.
   - `latest-best.pt`, 최근 best 몇 개, 알려진 counter-anchor를 같이 평가한다.

## 현재 결론

현재 run의 핵심 문제는 replay 전체의 color bias라기보다, 최신 replay 구간의 winner/color skew가
recent sampling overweight와 만나면서 특정 side/matchup에 특화된 checkpoint를 만드는 것이다.

`recent_sample_fraction=0`은 이 skew를 피하지만 strength까지 같이 잃었다. 따라서 현재는
`recent_sample_fraction`을 완전히 끄기보다, window를 넓히거나 winner-balanced recent sampling을 추가하는
방향이 더 타당하다.
