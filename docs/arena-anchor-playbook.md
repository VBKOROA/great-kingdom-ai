# Arena Anchor Playbook

This note records the current arena anchor set and how to use it when judging
new checkpoints. It is based on the 2026-05-19 train-v3 runs discussed during
the `policy_target_c_scale=0.1` transition.

## Current Best

Promote `training-latest-20260519-063201.pt` as the current best candidate.

Observed results:

| Candidate | Opponent | Win Rate | Blue Wins | Orange Wins | Read |
| --- | --- | ---: | ---: | ---: | --- |
| `063201` | `latest-best.pt` at run time | 70.0% | 68/100 | 72/100 | Strong replacement signal |
| `063201` | `042549` | 51.0% | 53/100 | 49/100 | Holds the counter-anchor |
| `063201` | `041042` | 57.0% | 55/100 | 59/100 | Beats old-meta anchor |

The important point is not only the 70% score versus `latest-best.pt`; it is
that the result is strong on both colors and does not collapse against the
known counter-anchor `042549`.

## Anchor Set

Keep these anchors around for a while:

| Anchor | Role | Why It Matters |
| --- | --- | --- |
| `training-latest-20260519-063201.pt` | current best | Best broad result so far; use as primary replacement target. |
| `training-latest-20260519-042549.pt` | counter-anchor | Loses to old `latest-best.pt` but catches several newer checkpoints. Good RPS detector. |
| `training-latest-20260519-041042.pt` | old-meta anchor | Older checkpoint that some newer models beat cleanly and others fail against. |
| previous `latest-best.pt` target | retirement watch | Keep only until `063201` is confirmed over another few snapshots. Record the concrete file path when running. |

Do not treat `042549` as globally stronger than `latest-best.pt`; it lost
`042549 vs latest-best.pt` by 41.0%. Its value is that it detects a specific
counter-meta.

## Promotion Rule

For replacing `latest-best.pt`, prefer a pool check over a single previous-vs-latest
match.

Minimum replacement rule:

```text
candidate vs current best >= 55%
candidate vs 042549       >= 50%
candidate vs 041042       >= 50%
```

Strong replacement rule:

```text
candidate vs current best >= 58%
candidate vs 042549       >= 52%
candidate vs 041042       >= 52%
both colors are not obviously broken
```

If a checkpoint beats current best but loses badly to `042549`, treat it as an
RPS-specialized candidate, not a clean promotion.

## Evaluation Order

For each new snapshot, run matches in this order:

1. `candidate` vs `063201`
2. `candidate` vs `042549`
3. `candidate` vs `041042`
4. `candidate` vs previous `latest-best.pt`, only while retiring that alias

If the first match is below 45%, stop unless debugging a specific regression.
If the first match is 50-55%, run the counter-anchor before deciding.

## Interpreting Side Splits

A good checkpoint should not rely on one color only.

Red flags:

```text
overall win rate looks good, but one side is below 45/100
candidate beats one anchor by 60% and loses another by 45%
candidate blue and orange results point in opposite directions across anchors
```

The replay color-bias diagnosis showed Blue has a mild natural edge in recent
self-play, so do not overreact to small side differences. Treat large one-color
failure as a separate issue from overall strength.

## Replay Context

The current target-sharpness setting is:

```text
search/selection c_scale: 1.0
policy_target_c_scale: 0.1
```

Recent shard diagnostics under `policy_target_c_scale=0.1` looked healthy:

```text
max_probability.p50 ~= 0.90-0.92
support.p50 ~= 28
entropy.mean ~= 0.51-0.53
```

Full replay diagnostics may lag because the 256k replay still contains older,
sharper shards. Prefer recent shard diagnostics when deciding whether the policy
target scale is currently healthy.
