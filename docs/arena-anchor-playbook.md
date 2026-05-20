# Arena Anchor Playbook

This note records the current train-v3 arena promotion rules and anchor set.
It reflects the 2026-05-20 evaluation run where `training-latest-20260520-080021.pt`
became the best service candidate.

## Current Best

Use `training-latest-20260520-080021.pt` as `latest-best.pt`.

Promotion evidence:

| Candidate | Opponent | Runs | Candidate Win Rate | Side Split | Read |
| --- | --- | ---: | --- | --- | --- |
| `080021` | previous `latest-best.pt` / `063442` | 1 | 85.5% | Blue 92, Orange 79 | Clear direct replacement signal. |
| `080021` | `094340` | 1 | 53% | Blue 65, Orange 41 | Passes the stable counter-anchor. |
| `080021` | `115445` | 2 | 42%, 42.5% | Blue 51/38, Orange 33/47 | Weak but repeated above the 40% floor. Keep `115445` as an anchor. |
| `063442` | previous `latest-best.pt` | 2 | 69%, 64% | Blue 75/72, Orange 63/56 | Clear direct replacement signal. |
| `063442` | `115445` | 1 | 76% | Blue 84, Orange 68 | Crushes the latest-best killer / risky anchor. |
| `063442` | `094340` | 3 | 51%, 54.5%, 54% | Blue 45/45/44, Orange 57/64/64 | Slight overall edge, but Blue weakness must remain tracked. |

The main weakness is the `115445` matchup. It is not severe enough to block
promotion because both repeated results stayed above 40%, but `115445` must
stay in the anchor pool.

## Anchor Set

Keep these checkpoints:

| Anchor | Role | Why It Matters |
| --- | --- | --- |
| `latest-best.pt` / `training-latest-20260520-080021.pt` | current best | Best validated service candidate as of 2026-05-20. |
| `training-latest-20260519-094340.pt` | stable contender / counter-anchor | Detects the Blue-side weakness in newer models. |
| `training-latest-20260519-115445.pt` | latest-best killer / risky anchor | Holds `080021` to about 42%. Useful RPS detector. |

Do not keep weak follow-up snapshots as anchors unless they expose a new,
repeatable failure mode.

Rejected snapshots:

| Snapshot | Reason |
| --- | --- |
| `training-latest-20260520-061935.pt` | 37.5% vs old `latest-best`, Orange 17%. Replay fit improved but arena collapsed. |
| `training-latest-20260520-063944.pt` | 27.5% vs `115445`, both colors bad. |
| `training-latest-20260520-065953.pt` | 32% vs `063442/latest-best`; also weak vs `094340` despite beating `115445`. |
| `training-latest-20260520-072504.pt` | 52% vs `063442` overall but Blue 35%, so not promotable. |
| `training-latest-20260520-073007.pt` | 38.5% vs `063442/latest-best`, Blue 29%. Replay fit improved, but arena regressed. |
| `training-latest-20260520-075016.pt` | 46% vs `063442/latest-best`, but Orange 18%. Not promotable despite Blue 74%. |

## Backend Policy

Use PyTorch fp32 arena for normal checkpoint gating and matrix checks.

Current practical reasons:

- Matrix scripts already use PyTorch fp32.
- `great-kingdom-evaluate` with `.pt` inputs and ONNX backend includes temporary ONNX export time.
- PyTorch fp32 has been fast enough for 200-game checks on Runpod.
- A model that is clearly stronger in fp32 usually remains strong in service fp16, but final service validation can still be run with cached ONNX if needed.

If using ONNX for final validation, export/cache ONNX once per checkpoint first.
Do not compare PyTorch runtime with ONNX runtime when `.pt -> .onnx` export time is included.

## Promotion Rule

A candidate must not be judged only against `latest-best.pt`.

Minimum checks:

```text
candidate vs latest-best.pt
candidate vs training-latest-20260519-094340.pt
candidate vs training-latest-20260519-115445.pt
```

Promote only if:

```text
candidate beats latest-best clearly, preferably in repeated 200-game runs
candidate does not lose badly to 094340 or 115445
worst repeated direct result is not below 40%
no anchor exposes a severe side collapse
```

Strong promotion signal:

```text
candidate vs latest-best >= 58% in at least one run and repeats above 55%
candidate vs 094340     >= 50% or only slightly below with no side collapse
candidate vs 115445     >= 50%
```

Red flags:

```text
one side below 40/100, especially if overall is also below 45%
overall win rate swings from win to loss across repeated 200-game runs
candidate beats one anchor by 65%+ but loses another by 40% or worse
replay KL/value improves while arena drops hard
```

## Evaluation Order

For each new snapshot:

1. Run `candidate` vs `latest-best.pt`.
2. If it is below 45%, stop unless debugging.
3. If it is 50% or better, run `candidate` vs `094340`.
4. Run `candidate` vs `115445` before promotion.
5. Repeat the most suspicious matchup if the first result is near 50% or has a bad side split.

Example:

```bash
great-kingdom-evaluate \
  --candidate data/runpod/train-v3/snapshots/training-latest-YYYYMMDD-HHMMSS.pt \
  --best latest-best.pt \
  --report result-vs-best.json \
  --config configs/runpod/arena.json
```

For reproducibility checks, fix both arena seed streams:

```bash
great-kingdom-evaluate \
  --candidate CANDIDATE.pt \
  --best ANCHOR.pt \
  --report result-fixed.json \
  --config configs/runpod/arena.json \
  --seed-start 20260520 \
  --gumbel-seed 700000
```

## Interpreting Diagnostics

Replay-fit diagnostics are useful, but they are not promotion evidence.

Repeated pattern observed on 2026-05-20:

```text
after_target_kl improves
value MAE/MSE improves
arena strength drops or a hard-counter appears
```

Examples:

- `061935` fit the replay better than `latest-best`, but scored only 37.5% in arena.
- `063944` fit the replay better than `115445`, but scored only 27.5% against it.
- `065953` was very close to `063442` by KL, but scored only 32% against `063442/latest-best`.
- `073007` improved target KL and value metrics versus `063442`, but scored only 38.5% with Blue 29%.
- `075016` reached 46% versus `063442`, but did so with Blue 74 and Orange 18, so the aggregate score hid a severe side collapse.

Conclusion:

```text
replay fitting improvement != arena strength improvement
arena anchors are the source of truth for promotion
```

## Replay And Color Notes

The replay color-bias diagnosis did not show a large outcome bias:

```text
overall Blue win rate ~= 50.4%
overall Orange win rate ~= 49.6%
recent windows mostly within a few percentage points
```

Therefore a candidate with Blue 30-40% or Orange 30-40% in arena should be
treated as a model/search weakness, not as a replay color-outcome imbalance.

Self-play game length is shorter than arena game length because PCR records a
fraction of turns. Do not interpret the raw self-play episode length as a direct
arena-length mismatch without accounting for PCR.

## Training Knobs

Current stable direction:

```text
EMA decay: 0.999
train_reuse_factor: 4.0
learning rate: 0.005 with warmup after optimizer bootstrap
```

Observed tuning read:

- `train_reuse_factor=8.0` was too aggressive; it produced replay-fit gains with arena instability.
- `train_reuse_factor=4.0` produced `063442`, the previous best candidate.
- Later snapshots drifted away from `063442`, so learning rate alone was not the clean fix.
- Lowering learning rate alone did not solve the drift; `073007` and `075016` still had side-specific regressions.
- `080021` became the new best after resetting optimizer state for the first train chunk with `bootstrap-once`.
- Prefer optimizer-state reset / bootstrap-once checks before raising actor simulations.

Suggested next experiments:

```text
keep EMA decay at 0.999
keep train_reuse_factor at 4.0
use lr 0.005 with constant warmup unless side collapses return
use bootstrap-once when intentionally clearing optimizer momentum/scheduler state
raise actor simulations only after update-size controls are exhausted
```

When intentionally clearing optimizer momentum once, bootstrap the first
training call from `training-latest.pt` and resume normally after that:

```bash
great-kingdom-learner-v2 \
  --learner-config configs/runpod/learner-v2.json \
  --train-config configs/runpod/train.json \
  --bootstrap-once \
  --train-reuse-factor 4.0 \
  --loop
```

When only changing the learning rate while resuming optimizer state, use the
CLI override so the optimizer actually receives the intended value.

## Operational Checklist

When a candidate looks good:

1. Save the checkpoint as a named snapshot.
2. Evaluate against `latest-best`, `094340`, and `115445`.
3. Repeat any anchor with near-50% result or bad side split.
4. Promote only after no hard-counter result appears.
5. Keep the promoted checkpoint as an anchor for the next run.
6. Keep old anchors until newer models pass them consistently.

For the current run, the next candidate should be judged against:

```text
latest-best.pt                         # currently 080021
training-latest-20260519-094340.pt
training-latest-20260519-115445.pt
```
