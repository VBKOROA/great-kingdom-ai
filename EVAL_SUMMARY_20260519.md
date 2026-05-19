# 2026-05-19 Model Evaluation Summary

## Current Decision

Keep `latest-best.pt` as the current best for now.

The strongest direct promotion candidate is not yet clean enough to replace it:

- `training-latest-20260519-094340.pt` looked good in the narrow top pool, but tied `latest-best.pt` directly.
- `training-latest-20260519-115445.pt` beat `latest-best.pt` clearly, but showed a hard-counter weakness in the narrow matrix.

## Key Arena Results

### `094340` vs `latest-best`

Command result:

```text
candidate: data/runpod/train-v3/snapshots/training-latest-20260519-094340.pt
best: latest-best.pt
games: 200
candidate_wins: 100
best_wins: 100
candidate_win_rate: 0.500
candidate_blue_wins: 48 / 100
candidate_orange_wins: 52 / 100
promoted: false
```

Interpretation:

`094340` is competitive, but this direct evaluation does not justify promotion over `latest-best.pt`.

### `115445` vs `latest-best`

Two repeated evaluations:

```text
Run 1:
candidate_win_rate: 0.605
candidate_wins: 121 / 200
candidate_blue_wins: 64 / 100
candidate_orange_wins: 57 / 100

Run 2:
candidate_win_rate: 0.610
candidate_wins: 122 / 200
candidate_blue_wins: 66 / 100
candidate_orange_wins: 56 / 100
```

Interpretation:

`115445` is clearly strong against `latest-best.pt`, including both colors. However, narrow top-pool matrix results show it has at least one severe bad matchup, so it is risky as a service best.

## Wide Pool Matrix

The wide matrix compared 11 candidates, 10 matches per candidate.

Top rows:

```text
latest-best.pt / 084810:
average_win_rate: 0.5595
blue_win_rate: 0.534
orange_win_rate: 0.585
winning_match_count: 7 / 10
non_losing_match_count: 8 / 10
worst_win_rate: 0.390

094340:
average_win_rate: 0.533
blue_win_rate: 0.516
orange_win_rate: 0.550
winning_match_count: 6 / 10
non_losing_match_count: 7 / 10
worst_win_rate: 0.455
```

Interpretation:

`latest-best.pt` is still the best wide-pool generalist. `094340` has a lower average, but a better worst-case score than `latest-best.pt`.

## Narrow Top Pool Matrix

The narrow top pool had 5 candidates and 4 matches per candidate.

```text
094340:
average_win_rate: 0.580
blue_win_rate: 0.6375
orange_win_rate: 0.5225
winning_match_count: 3 / 4
non_losing_match_count: 3 / 4
worst_win_rate: 0.465

112431:
average_win_rate: 0.5375
blue_win_rate: 0.510
orange_win_rate: 0.565
winning_match_count: 2 / 4
non_losing_match_count: 2 / 4
worst_win_rate: 0.350

115445:
average_win_rate: 0.51125
blue_win_rate: 0.5175
orange_win_rate: 0.505
winning_match_count: 2 / 4
non_losing_match_count: 2 / 4
worst_win_rate: 0.290

latest-best.pt:
average_win_rate: 0.45875
blue_win_rate: 0.4175
orange_win_rate: 0.500
winning_match_count: 2 / 4
non_losing_match_count: 2 / 4
worst_win_rate: 0.320

074740:
average_win_rate: 0.4125
blue_win_rate: 0.470
orange_win_rate: 0.355
winning_match_count: 1 / 4
non_losing_match_count: 1 / 4
worst_win_rate: 0.350
```

Interpretation:

`094340` is the best narrow-pool contender, but it failed to beat `latest-best.pt` in direct 200-game evaluation. `115445` is strong into `latest-best.pt`, but its `0.290` worst matchup is too risky.

## Training State Read

The run does not appear completely broken.

Signals that training may be recovering:

- `115445` beat `latest-best.pt` twice by about 60-61%.
- 09xx and 11xx snapshots are competitive in the top pool.
- The previous bad `113938` result may have been an unstable intermediate snapshot after changing EMA behavior.

Remaining problem:

- Matchup dependence is still high.
- Some candidates have strong direct wins but poor worst-case matrix results.
- `latest-best.pt` remains the safest generalist until a newer model beats it directly and avoids hard-counter failures.

## Current Recommendation

Do not promote yet.

Keep:

- `latest-best.pt` as service best
- `training-latest-20260519-094340.pt` as stable contender
- `training-latest-20260519-115445.pt` as latest-best killer / risky contender
- `training-latest-20260519-112431.pt` as another top-pool contender

Next evaluation should compare new snapshots against:

1. `latest-best.pt`
2. `training-latest-20260519-094340.pt`
3. `training-latest-20260519-115445.pt`

Promotion should require:

- direct win over `latest-best.pt`
- acceptable result against `094340`
- no severe hard-counter result like `worst_win_rate < 0.40`
- both Blue and Orange win rates near or above 50%
