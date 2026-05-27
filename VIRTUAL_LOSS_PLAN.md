# Virtual Loss Implementation Plan

## Goal

Add full path-level virtual loss to the Rust Gumbel search backend so batched leaf
selection can keep filling evaluation batches without repeatedly blocking on the
same pending leaf or subtree.

This is primarily a search-throughput change. It should not change replay schema,
training targets, model outputs, or Python-facing result formats.

## Current State

The current implementation has partial protection for batched search:

- Root sequential halving reserves root visits with `RootSequentialHalving::reserve_visit`.
- Pending leaf evaluation is tracked with `GumbelEdge::pending_evaluation`.
- `reserve_path` only marks the final edge as pending.
- Real `visit_count` and `value_sum` are updated only after evaluation completes in
  `backup_path`.

This avoids duplicate leaf evaluation, but it does not make already-pending paths
less attractive during inner node selection. When the same subtree is selected
again, search can return `BlockedPending`, reducing batch fill rate.

## Proposed Design

Keep completed search statistics separate from virtual statistics.

Add virtual fields to `GumbelEdge`:

```rust
virtual_visit_count: u32
virtual_value_sum: f32
```

Use effective stats for selection only:

```rust
effective_visit_count = visit_count + virtual_visit_count
effective_value_sum = value_sum + virtual_value_sum
```

Final public result methods such as `visit_counts()` should continue to return
completed visits only. This avoids changing policy targets and replay semantics.

## Virtual Value Convention

Use a conservative constant virtual loss from the perspective of each edge owner:

```rust
const VIRTUAL_LOSS_VALUE: f32 = -1.0;
```

When reserving a path, apply one virtual visit to every edge on the path. The
stored virtual value must follow the same sign convention as `backup_path`.

When unreserving a path, subtract exactly the same virtual visit/value from every
edge before applying the real backup.

Important: sign handling is the highest-risk part of this change. Add focused
tests before relying on batch performance numbers.

## Files To Change

- `rust/great_kingdom_core/src/gumbel/node.rs`
  - Add virtual stat fields to `GumbelEdge`.
  - Add helper methods for effective visit count, effective value sum, and
    effective mean Q.

- `rust/great_kingdom_core/src/gumbel/search.rs`
  - Change `reserve_path` to reserve the full path, not just the final edge.
  - Change `unreserve_path` to remove full-path virtual stats.
  - Keep `pending_evaluation` on the final edge unless tests show it is no longer
    needed. It remains useful as a hard duplicate-leaf guard.
  - Update `select_inner_action_index` to use effective stats.
  - Ensure `backup_path` updates completed stats only.

- `rust/great_kingdom_core/src/gumbel/selection.rs`
  - If root or inner policy selection uses `InnerEdgeStats`, decide whether it
    needs effective stats during active search.
  - Do not let final policy target generation include virtual stats.

- `rust/great_kingdom_core/src/gumbel/policy.rs`
  - Keep final policy target and root value based on completed stats.
  - If root ranking scores are recomputed while pending work exists, use effective
    stats only for scheduling, not for final result construction.

- `rust/great_kingdom_core/src/gumbel/batch.rs`
  - Verify the active batched search path still reserves, evaluates, unreserves,
    and backs up in that order.

- `rust/great_kingdom_core/src/gumbel/arena_batch.rs`
  - Mirror the same verification for arena batched search.

## Implementation Steps

1. Add virtual stat fields and helper methods on `GumbelEdge`.
2. Add a small helper that computes the signed edge value sequence for a path.
3. Extend `reserve_path` and `unreserve_path` to apply/remove virtual stats across
   the full path.
4. Update inner selection to use effective stats.
5. Add a root-ranking helper for active scheduling if root scheduling should see
   virtual stats.
6. Keep final result generation on completed stats.
7. Add focused Rust unit tests.
8. Run Rust tests first, then the Python Gumbel/self-play tests.

## Test Checklist

Add or update Rust tests for:

- `reserve_path` adds one virtual visit to every edge in the path.
- `unreserve_path` fully restores virtual stats to zero.
- Real `visit_count` is unchanged until `backup_path`.
- `backup_path` after unreserve produces the same completed stats as before.
- `select_inner_action_index` avoids an already virtually reserved high-priority
  edge when another reasonable edge exists.
- Final `visit_counts()` excludes virtual visits.
- Batched evaluator path still returns exactly `simulations` completed visits.

Suggested commands:

```bash
cargo test --manifest-path rust/great_kingdom_core/Cargo.toml gumbel
python -m pytest tests/test_gumbel.py tests/test_self_play.py tests/test_evaluate.py
```

Use the project venv for Python commands.

## Risks

- Incorrect sign handling can bias Q values during pending search.
- Accidentally including virtual visits in final `visit_counts()` would alter
  policy targets.
- Root scheduling already has reserved visits; adding root virtual stats may
  double-count if not separated carefully.
- A virtual loss value of `-1.0` is simple but may be too aggressive. If batch
  diversity improves but playing strength drops, make it configurable or test a
  softer value such as `-0.5`.

## Recommended First Version

Implement virtual stats for inner selection first while keeping final result stats
unchanged. Treat root scheduling carefully because `RootSequentialHalving` already
tracks reserved visits. After correctness tests pass, measure:

- average leaf batch fill rate
- `BlockedPending` count
- self-play throughput
- short arena strength check against the previous implementation
