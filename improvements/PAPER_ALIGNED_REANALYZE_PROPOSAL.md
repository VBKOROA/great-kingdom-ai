# Paper-Aligned Reanalyze Proposal

## Scope

This proposal uses "paper-aligned" in two layers:

1. **Pipeline alignment**: replay is sampled first, and reanalyze work is applied to the learner
   batch or its prepared context instead of rebuilding a full replay-sized target snapshot every
   iteration.
2. **Target alignment**: EfficientZero-specific target details such as dynamic value horizon and
   MCTS-root bootstrap values can be added after the sampled-batch pipeline is stable.

The immediate recommendation is to implement pipeline alignment first, then layer in target-level
alignment where it fits our game and our Gumbel MCTS stack.

## Current Problem

The current train-v2 pipeline does not match the EfficientZero/LightZero-style reanalyze structure.

Current pipeline:

```text
self-play
  -> append trajectories to replay
  -> build a full replay-sized reanalyze snapshot
  -> refresh value targets for all rows
  -> refresh policy targets with search for a selected subset
  -> train from that snapshot for the current iteration
```

With the current Runpod config, this means a replay capacity around 500k transitions is treated as
the reanalyze unit. Even when `save_target_snapshots` is false, the pipeline still materializes a
large in-memory target snapshot before training.

This differs from EfficientZero/LightZero-style training, where replay is sampled first and
reanalyze is applied to the sampled training batch or its prepared context. In that structure, a
"100% value reanalyze" or "99% policy reanalyze" ratio means the targets used by the learner batch
are refreshed, not that the whole replay buffer is refreshed every iteration.

## Evidence From Reference Implementations

- EfficientZero uses a parallel pipeline with self-play workers, CPU context workers, GPU reanalyze
  workers, and a learner batch queue. Reanalyze happens between replay sampling and learner
  training.
- LightZero's MuZero/EfficientZero buffer samples a batch from replay, prepares the context, then
  computes target value and target policy for that batch.
- The EfficientZero public repo exposes `--revisit_policy_search_rate 0.99`, which is a policy
  target reanalyze rate for sampled training data.
- EfficientZero's supplement also states that policy targets are reanalyzed for 99% of sampled data
  and value targets for 100% of sampled data.

References:

- EfficientZero supplement: `Mastering Atari Games with Limited Data`, Appendix pipeline section.
- LightZero `GameBuffer.sample()` / `_make_batch()` / `_compute_target_policy_reanalyzed()`.
- YeWR/EfficientZero README arguments, especially `--revisit_policy_search_rate 0.99`.

## Current Config Snapshot

As of this proposal, the relevant Runpod config is:

```json
{
  "replay_capacity": 500000,
  "self_play_games": 1000,
  "train_reuse_factor": 2.0,
  "reanalyze_batch_size": 8192,
  "search_reanalyze_fraction": 0.5,
  "search_reanalyze_budget": 65536,
  "search_reanalyze_simulations": 32,
  "search_reanalyze_root_batch_size": 16384,
  "search_reanalyze_leaf_batch_size": 1024,
  "save_target_snapshots": false
}
```

Training config:

```json
{
  "batch_size": 512,
  "recent_sample_fraction": 0.25,
  "recent_sample_window": 80000,
  "priority_enabled": true,
  "priority_value_error_weight": 1.0,
  "priority_policy_kl_weight": 1.0,
  "priority_target_age_weight": 0.0,
  "ema_decay": 0.995
}
```

The current setup improved training loss/KL substantially, but it is still a custom snapshot-based
approximation rather than a paper-aligned reanalyze pipeline.

## Proposed Target Structure

Target pipeline:

```text
self-play
  -> append trajectories to replay
  -> learner asks for a training batch
  -> sample rows from trajectory replay
  -> prepare reanalyze context for those rows
  -> refresh value targets for sampled rows
  -> refresh policy targets for sampled rows according to policy_reanalyze_ratio
  -> train one gradient step
```

The important change is that replay sampling happens before reanalyze.

## Proposed Implementation Plan

### Step 1: Add an on-sample reanalyze dataset

Add a dataset wrapper, tentatively:

```text
OnSampleReanalyzeDataset
```

Responsibilities:

- Wrap `TrajectoryReplayStore`.
- Implement `sample_arrays(batch_size, rng, recent_fraction, recent_window, priority_config)`.
- Sample transition indexes from the raw replay store.
- Build feature/policy/value arrays only for the sampled rows.
- Return a `TrainingArrays`-compatible batch.

This should coexist with the current `ReanalyzeTargetSnapshot` path.

### Step 2: Phase 1 sampled value refresh with snapshot-equivalent semantics

For every sampled row:

- locate its episode and timestep;
- locate the bootstrap timestep `t + bootstrap_td_steps`;
- evaluate only the bootstrap features needed for the sampled batch;
- convert bootstrap value to the sampled transition player's perspective;
- fall back to terminal outcome when the bootstrap index crosses the terminal region.

This preserves the current snapshot target semantics while changing the work unit from
"entire replay" to "sampled learner batch". It is the safest first implementation because the
new path can be checked row-for-row against equivalent rows from full snapshot mode.

### Step 3: Reanalyze policy targets by sampled-batch ratio

Add a config knob:

```json
{
  "policy_reanalyze_ratio": 0.99
}
```

For a sampled training batch of 512 rows:

- about 507 rows get fresh search policy targets;
- the remaining rows keep stored self-play policy targets.

This replaces the current global `search_reanalyze_budget` semantics in paper-aligned mode.

### Step 4: Add an explicit mode switch

Add a pipeline config field:

```json
{
  "reanalyze_mode": "snapshot"
}
```

Allowed values:

- `snapshot`: current behavior.
- `on_sample`: paper-aligned sampled-batch reanalyze.

This keeps rollback easy and allows direct A/B comparison.

### Step 5: Keep Gumbel search exploration semantics consistent

EfficientZero's MuZero-style reanalyze pipeline resamples Dirichlet noise during target search.
We should **not** copy that behavior directly.

Our pipeline uses Gumbel MCTS, and the Gumbel MuZero paper replaces root exploration by noisy prior
perturbation with Gumbel-Top-k action sampling without replacement. The paper explicitly notes that
Gumbel MuZero does not use Dirichlet noise.

Therefore:

- keep **self-play without Dirichlet noise**;
- keep **reanalyze search without Dirichlet noise**;
- document any future Dirichlet experiment as a custom ablation, not as the default paper-aligned
  path.

This avoids target mismatch between the search operator used to create trajectories and the search
operator used to refresh policy targets.

### Step 6: Phase 2 EfficientZero-faithful value targets

After the on-sample pipeline is verified against current snapshot semantics, add an optional target
mode that follows EfficientZero more closely:

- dynamic bootstrap horizon that shrinks for older trajectories;
- bootstrap from reanalyzed MCTS root value rather than only from a refreshed value head;
- explicit instrumentation for horizon distribution and root-value target usage.

This is a separate target-definition change, not a prerequisite for getting the sampled-batch
pipeline right.

### Step 7: Paper-aligned priority preset and beta annealing

The current `priority_policy_kl_weight=1.0` is a custom intervention that worked well for reducing
policy KL, but it is not the cleanest paper-aligned default.

For `on_sample` mode, start with:

```json
{
  "priority_enabled": true,
  "priority_value_error_weight": 1.0,
  "priority_policy_kl_weight": 0.0,
  "priority_target_age_weight": 0.0,
  "priority_alpha": 0.6,
  "priority_beta": 0.4
}
```

Treat `priority_beta=0.4` as the initial value. EfficientZero anneals beta toward `1.0`, so beta
annealing is a useful follow-up once the main reanalyze target changes are in place.

Keep the existing KL-priority setup available as a custom mode, because it may still be useful for
this game.

## Expected Benefits

- Matches EfficientZero/LightZero semantics more closely.
- Makes "100% value reanalyze" feasible with a 500k replay buffer.
- Makes "99% policy reanalyze" mean 99% of the actual learner batch, not 99% of the whole replay.
- Avoids large full-replay target snapshots every iteration.
- Reduces stale-target mismatch for exactly the samples used by SGD.
- Opens the door to overlapping CPU context preparation, GPU search/eval, and learner training.

## Risks

- Per-batch search can become the new learner bottleneck if implemented synchronously.
- Training step latency may become more variable because batch sampling now includes reanalyze work.
- Priority scores based on refreshed value/policy targets are less straightforward because targets
  are no longer precomputed for the whole replay.
- Existing snapshot-level diagnostics need replacements, because there is no full target snapshot
  to inspect.
- The current custom KL-priority setup has empirically helped; removing it may worsen early
  convergence even if it is more paper-aligned.
- EfficientZero-faithful dynamic horizon and MCTS-root value targets would change the target
  definition itself, so they should not be mixed into the first pipeline migration.

## Mitigations

- Keep `snapshot` mode until `on_sample` mode is validated.
- Add instrumentation for sampled-batch reanalyze:
  - sampled rows per step;
  - value eval time;
  - search time;
  - policy reanalyze ratio actually applied;
  - stale policy fallback count;
  - bootstrap source used for value targets;
  - train step time.
- Initially run `on_sample` for short A/B tests using the same replay and checkpoint.
- Consider a small batch queue after the synchronous version works:
  - CPU prepares sampled contexts;
  - GPU reanalyzes targets;
  - learner consumes ready batches.

## Recommended First Experiment

Start with a conservative pipeline-aligned mode:

```json
{
  "reanalyze_mode": "on_sample",
  "policy_reanalyze_ratio": 0.99,
  "bootstrap_td_steps": 8,
  "search_reanalyze_simulations": 32,
  "search_reanalyze_max_considered_actions": 8,
  "search_reanalyze_root_batch_size": 512,
  "search_reanalyze_leaf_batch_size": 1024
}
```

Search exploration policy:

```json
{
  "self_play_dirichlet_noise": false,
  "reanalyze_dirichlet_noise": false
}
```

These booleans are shown here as explicit policy decisions. They do not need to become config knobs
unless we want a dedicated ablation.

Training priority:

```json
{
  "priority_enabled": true,
  "priority_value_error_weight": 1.0,
  "priority_policy_kl_weight": 0.0,
  "priority_target_age_weight": 0.0,
  "recent_sample_fraction": 0.25,
  "recent_sample_window": 80000
}
```

Compare against the current snapshot mode on:

- wall-clock per iteration;
- train loss/policy/value/KL;
- fixed checkpoint arena or fixed evaluation;
- GPU utilization during reanalyze/search;
- number of actually search-refreshed learner samples per iteration.

## Implementation Priority

1. Add `reanalyze_mode` config with no behavior change.
2. Add sampled transition index selection from `TrajectoryReplayStore`.
3. Add sampled value target refresh.
4. Add sampled policy search refresh.
5. Wire `train_from_replay()` to consume `OnSampleReanalyzeDataset`.
6. Add tests comparing sampled-batch targets to equivalent rows from full snapshot mode.
7. Add Runpod config preset for `on_sample`.
8. After the sampled-batch path is stable, add dynamic horizon.
9. Then add reanalyzed MCTS-root bootstrap value targets.
10. After the target-definition upgrades are understood, add beta annealing for prioritized replay.
