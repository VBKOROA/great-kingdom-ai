# Full 2D Position Bias Unit Test Plan

## Scope

Target implementation:

- `python/great_kingdom_ai/model.py`
- `Full2DRelativePositionBias`
- `BoardSelfAttentionBlock` usage of `relative_bias`

This plan focuses on unit tests for the full 2D relative position bias mapping used by the `strong_attn` preset. It does not cover training quality or arena strength.

## Current Coverage

Existing tests cover the basic surface:

- `relative_bias_table` shape is `(289, num_heads)` for a 9x9 board.
- `relative_index` shape is `(81, 81)`.
- `relative_index` values are in range `[0, 288]`.
- `forward()` returns shape `[1, heads, 81, 81]`.
- `BoardSelfAttentionBlock` keeps input shape.
- `strong_attn` has two attention blocks.
- `strong_attn` ONNX export matches PyTorch output.

This is enough to catch major shape regressions, but it does not strongly prove that `(query_row, query_col, key_row, key_col)` maps to the intended `(dr, dc)` bias entry.

## Proposed Tests

### 1. Relative Index Matches Expected Formula

Add a test that checks selected known coordinate pairs against the formula:

```text
dr = query_row - key_row
dc = query_col - key_col
index = (dr + board_size - 1) * (2 * board_size - 1) + (dc + board_size - 1)
```

Suggested cases:

- Same cell: `(0, 0) -> (0, 0)` maps to center offset `(0, 0)`.
- Horizontal neighbor: `(0, 0) -> (0, 1)` maps to `(0, -1)`.
- Vertical neighbor: `(1, 0) -> (0, 0)` maps to `(1, 0)`.
- Corner to opposite corner: `(0, 0) -> (8, 8)` maps to `(-8, -8)`.
- Opposite corner back: `(8, 8) -> (0, 0)` maps to `(8, 8)`.
- Mixed interior pair: `(4, 4) -> (5, 3)` maps to `(-1, 1)`.

Expected value:

- This catches swapped row/column bugs.
- This catches sign reversal bugs.
- This catches off-by-one errors in the `board_size - 1` offset.

### 2. Same Relative Offset Reuses Same Index

Add a test that checks multiple coordinate pairs with the same `(dr, dc)` produce identical `relative_index` values.

Example pairs for `(dr=1, dc=-2)`:

- Query `(2, 1)`, key `(1, 3)`
- Query `(5, 4)`, key `(4, 6)`
- Query `(8, 2)`, key `(7, 4)`

Expected value:

- This verifies the bias is relative, not absolute-position dependent.
- It protects against accidentally encoding absolute board cells.

### 3. Different Relative Offsets Do Not Collide

Add a test that iterates all legal query/key cell pairs, computes each `(dr, dc)`, and records the produced index.

Assertions:

- Every `(dr, dc)` maps to exactly one index.
- No index maps to more than one `(dr, dc)`.
- The number of unique offsets is `289`.
- The number of unique indexes used is `289`.

Expected value:

- This proves the full 17x17 relative offset grid is represented without collisions.
- It catches incorrect flattening formulas such as multiplying by `board_size` instead of `2 * board_size - 1`.

### 4. Forward Output Uses Indexed Bias Table Values

Add a test that fills `relative_bias_table` with deterministic values, then checks that `forward()` returns the exact values at selected query/key positions.

Example setup:

```python
table[index, head] = index * 10 + head
```

Assertions:

- `bias.shape == (1, num_heads, 81, 81)`.
- For selected `(head, query_cell, key_cell)`, `bias[0, head, query_cell, key_cell] == table[relative_index[query_cell, key_cell], head]`.

Expected value:

- This proves `forward()` gathers from the intended index matrix.
- This catches wrong `permute` order bugs.
- This catches accidental head/cell dimension swaps.

### 5. Relative Bias Receives Gradient

Add a small gradient test:

```python
bias_module = Full2DRelativePositionBias(num_heads=2, board_size=9)
loss = bias_module().sum()
loss.backward()
```

Assertions:

- `relative_bias_table.grad is not None`.
- Gradient shape equals `relative_bias_table.shape`.
- Gradients are finite.
- Every table row has nonzero gradient for both heads.

Expected value:

- This verifies the bias table participates in optimization.
- Since every relative offset appears at least once on a 9x9 board, all rows should receive gradient under `sum()`.

### 6. Board Size Generalization

Add a parametrized test for smaller board sizes, such as `board_size=1`, `2`, and `3`.

Assertions:

- Table rows equal `(2 * board_size - 1) ** 2`.
- Index shape equals `(board_size * board_size, board_size * board_size)`.
- Unique used indexes equal `(2 * board_size - 1) ** 2`.
- Forward shape equals `(1, heads, board_size * board_size, board_size * board_size)`.

Expected value:

- This keeps the module honest as a generic class.
- It makes edge cases easier to reason about than the full 9x9 board.

## Suggested File

Add the tests to:

```text
tests/test_model.py
```

Reason:

- The existing model and attention tests already live there.
- The new tests are small unit tests, not integration tests.
- Keeping them near the current `Full2DRelativePositionBias` test avoids scattering attention coverage.

## Suggested Test Names

```python
def test_full_2d_relative_position_bias_known_coordinate_mapping() -> None:
    ...

def test_full_2d_relative_position_bias_same_offset_reuses_index() -> None:
    ...

def test_full_2d_relative_position_bias_offsets_do_not_collide() -> None:
    ...

def test_full_2d_relative_position_bias_forward_gathers_table_values() -> None:
    ...

def test_full_2d_relative_position_bias_table_receives_gradient() -> None:
    ...

@pytest.mark.parametrize("board_size", [1, 2, 3, 9])
def test_full_2d_relative_position_bias_board_size_invariants(board_size: int) -> None:
    ...
```

## Verification Command

Run the focused model tests:

```bash
.venv/bin/python -m pytest tests/test_model.py -q
```

If ONNX compatibility is touched later, also run:

```bash
.venv/bin/python -m pytest tests/test_onnx_export.py::test_strong_attn_onnx_runtime_outputs_match_pytorch -q
```

## Priority

Recommended implementation order:

1. Known coordinate mapping.
2. Same offset reuses same index.
3. Offset collision check.
4. Forward gather check.
5. Gradient check.
6. Board size parametrization.

The first four tests give the highest confidence for the current implementation. The gradient and board-size tests are useful but less urgent for the immediate training run.
