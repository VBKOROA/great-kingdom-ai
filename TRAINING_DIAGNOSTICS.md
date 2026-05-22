# Training Diagnostics

MuZero/AlphaZero-style self-play training should not be judged by loss alone.
Targets and data distribution move with the current model, so arena strength is
the primary signal and loss is a health check.

## Primary Signal

Use arena/matrix results to decide whether the model is actually improving.

- Compare current against snapshots far enough apart to beat noise.
- With 5-minute snapshots, useful defaults are:
  - quick: current vs 6 snapshots ago
  - normal: current vs 12 snapshots ago
  - cycle check: current vs 24 and 48 snapshots ago
- Keep a best-so-far reference and compare against it periodically.

If current loses to both 6-snapshot and 12-snapshot references, treat it as a
regression signal. Do not immediately assume the cause is learning rate.

## Loss Patterns

### Stable Loss, Weakening Arena

Pattern:

```text
loss oscillates inside a stable band
arena strength decreases against old snapshots
```

Example:

```text
1.9, 2.1, 1.9, 2.1, ...
```

This is usually not a learning-rate explosion. It more often means the model is
fitting the current replay/search target distribution while losing general
strength.

First suspects:

- `train_reuse_factor` too high
- `recent_sample_fraction` too high
- priority sampling too aggressive
- latest self-play distribution or opening loop being over-amplified

First adjustment candidates:

```json
"train_reuse_factor": 10.0
```

If still weakening:

```json
"recent_sample_fraction": 0.1
```

Then consider softening priority sampling:

```json
"priority_alpha": 0.25,
"priority_beta": 0.25,
"priority_target_age_weight": 0.0
```

### Rising Or Widening Loss, Weakening Arena

Pattern:

```text
loss moving average rises
loss oscillation amplitude grows
policy KL or value loss spikes become larger
arena strength decreases
```

This is more consistent with unstable updates.

First suspect:

- learning rate too high

First adjustment candidate:

```json
"learning_rate": 0.0002
```

If instability remains, try:

```json
"learning_rate": 0.0001
```

## Raw vs EMA Clues

Use raw and EMA behavior to separate causes.

- raw weakens but EMA is stable: update steps are probably too sharp; suspect LR.
- raw and EMA both weaken: data/reuse/recent/priority issue is more likely.
- raw/EMA gap keeps growing: LR or update pressure is probably too high.

## Train Reuse Factor

`train_reuse_factor` is not directly tuned by actor count. It means how many
training samples are budgeted per imported transition:

```text
train_steps = imported_transitions * train_reuse_factor / batch_size
```

More actors increase wall-clock data flow, but they do not change the meaning of
reuse. Lower reuse when the model is learning too much from each data
distribution, not simply because actor count increased.

## Practical Rule

Use this decision table first:

```text
loss stable + arena down
=> reduce reuse/recent/priority pressure first

loss rising or widening + arena down
=> reduce learning rate first

loss stable + arena stable/up
=> do not tune just because loss is not decreasing
```

