# Ordo Snapshot Ranking Script Plan

## Goal

Implement a fast script that ranks recent training snapshots in a checkpoint folder with Ordo.

The script should answer this operational question:

> "Among the latest N snapshots, which checkpoints are strongest, and is the latest model improving?"

This is not a promotion gate. It is a quick ranking/monitoring tool for many snapshots.

## Constraints

- Ignore `docs/`; use the current scripts and Python package structure.
- Keep the implementation testable and split into small functions.
- Run on a CPU office laptop when needed, but support Runpod GPU evaluation.
- Python runs in the project venv.
- Avoid full round-robin by default. For 25 snapshots, full round-robin is 300 pairs and too expensive for frequent checks.
- Snapshot count is a CLI window, not a fixed algorithm requirement. The script should work with any count >= 2.
- Anchors are optional. If no anchor is supplied, rank the selected snapshots relative to each other.

## Existing Code To Reuse

- `scripts/run_pt_folder_matrix_arena.py`
  - already selects `.pt` files from a folder.
  - currently runs full pairwise matrix.
- `scripts/run_candidate_pairwise_matrix.py`
  - has `PairwiseMatchResult`.
  - has `_run_or_load_pairwise_match`.
  - has arena config override helpers.
  - caches per-match arena reports.
- `great_kingdom_ai.evaluate`
  - supports PyTorch checkpoint arena.
  - supports ONNX backend through `run_arena_checkpoints_onnx`.

The new script should not duplicate arena logic. It should mostly add:

- snapshot selection
- sparse pairing generation
- arena report reuse
- Ordo input generation
- Ordo execution
- Ordo output parsing
- ranking summary JSON

## Proposed Script

Add:

```text
scripts/run_snapshot_ordo_ranking.py
```

Default output:

```text
SNAPSHOT_DIR/ordo-ranking-reports/
  pairs/
    pair-0001-snapshot_x-vs-snapshot_y-arena.json
  games.pgn
  ordo.txt
  summary.json
```

## CLI

Proposed command:

```bash
.venv/bin/python scripts/run_snapshot_ordo_ranking.py \
  data/runpod/train-strong-attn/checkpoints/snapshots \
  --max-snapshots 25 \
  --arena-config configs/runpod/arena.json \
  --backend onnx \
  --device cuda \
  --games 8 \
  --pair-offsets 1,2,4,8 \
  --anchor-games 16
```

Important flags:

- `checkpoint_dir`: folder containing snapshot `.pt` files.
- `--glob`: default `*.pt`.
- `--recursive`: optional recursive scan.
- `--max-snapshots`: keep latest N snapshots after sorting; default `25`; omit or set `0` to use all selected snapshots.
- `--anchors`: optional fixed anchor checkpoints always included; no anchors is a valid mode.
- `--pair-offsets`: sparse time-neighbor offsets; default `1,2,4,8`.
- `--games`: games per normal snapshot pair.
- `--anchor-games`: games per snapshot-anchor pair; default same as `--games`.
- `--backend`: `onnx` or `pytorch`; default `onnx` for speed on Runpod.
- `--ordo-bin`: default `ordo`.
- `--force-arena`: rerun arena reports.
- `--force-ordo`: rerun Ordo even if `games.pgn` did not change.
- `--output-dir`: defaults to `CHECKPOINT_DIR/ordo-ranking-reports`.
- `--latest-first` or `--sort-by-mtime`: optional if filename order is not reliable.

## Snapshot Selection

Implement:

```python
def select_snapshots(
    checkpoint_dir: Path,
    *,
    glob: str,
    recursive: bool,
    max_snapshots: int,
    sort_by_mtime: bool,
) -> list[Path]:
    ...
```

Default sort should be filename sort, matching current scripts. `--sort-by-mtime` can be used for timestamped files if names are not stable.

Anchors, when provided, are separate from snapshots:

- snapshots are ranked for recent trend.
- anchors stabilize the Elo origin.
- if an anchor is also in snapshots, dedupe by resolved path.

If no anchors are provided:

- Ordo still ranks the selected snapshots correctly relative to each other.
- `summary.json` should normalize ratings with the oldest selected snapshot at `0 Elo` by default.
- Cross-run absolute Elo offsets may drift, so compare runs through overlapping snapshots and rank/order, not only raw Elo numbers.

## Sparse Pairing

Implement:

```python
@dataclass(frozen=True)
class SnapshotPair:
    candidate: Path
    baseline: Path
    kind: str
    games: int
```

Generate pairs:

1. Neighbor pairs by offsets.
   - For sorted snapshots `S0..S24` and offsets `1,2,4,8`, add:
     - `S_i` vs `S_(i-offset)` for every valid `i`.
   - This gives enough local ordering signal without full round-robin.
2. Optional anchor pairs.
   - If anchors are provided, every snapshot plays every anchor.
   - Use `--anchor-games` because anchors define the shared rating scale.
3. Optional current-best pair.
   - Treat `--current-best PATH` as another anchor.

For 25 snapshots and offsets `1,2,4,8`:

```text
24 + 23 + 21 + 17 = 85 snapshot pairs
```

With 2 optional anchors:

```text
85 + 25 * 2 = 135 total pairs
```

This is much cheaper than full round-robin if the full matrix also needs large games per pair. If that is still too expensive, use offsets `1,3,8` or reduce `--max-snapshots` to 16.

## Arena Execution

Use a small wrapper around existing arena code:

```python
def run_or_load_snapshot_pair(
    pair: SnapshotPair,
    *,
    report_path: Path,
    arena_config: ArenaConfig,
    backend: Literal["onnx", "pytorch"],
    force: bool,
) -> PairwiseMatchResult:
    ...
```

For PyTorch backend, reuse `_run_or_load_pairwise_match`.

For ONNX backend, implement a sibling function using:

```python
run_arena_checkpoints_onnx(
    candidate_checkpoint=pair.candidate,
    best_checkpoint=pair.baseline,
    config=arena_config,
    onnx_max_batch_size=...,
    onnx_precision=...,
)
```

Reason: repeated PyTorch model loading is slow; ONNX arena uses the Rust evaluator path and should be the default on Runpod.

Each pair should write one arena JSON report and be skipped on the next run unless `--force-arena` is set.

## Ordo Input

Convert arena game results to an Ordo-compatible result file. Ordo only needs player names and results for rating fitting.

Great Kingdom sides are Blue and Orange, as defined in `docs/rule-spec.md`:

- Blue is the first player.
- Orange is the second player.
- Code, JSON summaries, logs, and tests should use Blue/Orange terminology.

If the implementation uses PGN because Ordo expects PGN field names, `White` and `Black` must be treated only as PGN transport tags, not as game terminology:

- PGN `White` tag slot = Great Kingdom Blue side.
- PGN `Black` tag slot = Great Kingdom Orange side.
- Blue win = `1-0`.
- Orange win = `0-1`.
- draw/no winner = `1/2-1/2`.

Prefer helper names that preserve the game terms:

```python
def blue_orange_game_to_ordo_pgn(...):
    ...
```

Avoid variable names like `white_model` or `black_model`; use `blue_model` and `orange_model`.

Use stable player IDs:

```python
def model_id(path: Path) -> str:
    return path.stem
```

For each game in an arena report, decide which model occupied the Blue side and which occupied the Orange side:

- if `candidate_player == BLUE`
  - candidate is Blue
  - baseline is Orange
- if `candidate_player == ORANGE`
  - baseline is Blue
  - candidate is Orange

Example PGN transport entry:

```pgn
[Event "great-kingdom-ai snapshot arena"]
[White "snapshot-000120"]
[Black "snapshot-000115"]
[Result "1-0"]

1-0
```

Write:

```text
games.pgn
```

## Ordo Execution

Implement:

```python
def run_ordo(
    *,
    ordo_bin: str,
    pgn_path: Path,
    output_path: Path,
    anchor_name: str | None,
    anchor_elo: float,
) -> None:
    ...
```

Expected base command shape:

```bash
ordo -p games.pgn -o ordo.txt
```

During implementation, verify exact flags with `ordo --help` on the target environment and keep all Ordo-specific flags in this one function.

Anchor behavior:

- If `--ordo-anchor NAME` is given, fix that model near `--ordo-anchor-elo`, default `0`.
- If Ordo binary/config does not support fixed anchor directly, post-normalize all ratings by subtracting the anchor rating and adding `anchor_elo`.
- If no anchor is given, normalize best or oldest snapshot to `0` in `summary.json`, but keep raw Ordo text.

## Ordo Output Parsing

Implement a tolerant parser:

```python
@dataclass(frozen=True)
class OrdoRating:
    name: str
    elo: float
    error: float | None
    games: int | None
```

Parser should:

- ignore header/separator lines.
- identify rows containing model name and numeric Elo.
- keep raw line in the summary for debugging if parsing is imperfect.

If parsing fails, still write `summary.json` with:

- `ordo_output_path`
- `pgn_path`
- `matches`
- `error`

## Summary JSON

Write:

```json
{
  "event": "snapshot_ordo_ranking_summary",
  "checkpoint_dir": "...",
  "snapshots": ["..."],
  "anchors": ["..."],
  "pair_offsets": [1, 2, 4, 8],
  "pair_count": 135,
  "games_total": 1080,
  "backend": "onnx",
  "arena_config": {},
  "pgn_path": ".../games.pgn",
  "ordo_output_path": ".../ordo.txt",
  "ranking": [
    {
      "rank": 1,
      "model": "snapshot-000125",
      "path": ".../snapshot-000125.pt",
      "elo": 42.3,
      "error": 18.7,
      "games": 96,
      "is_snapshot": true,
      "is_anchor": false
    }
  ],
  "latest": {
    "model": "snapshot-000125",
    "rank": 1,
    "elo": 42.3,
    "delta_vs_previous_snapshot": 7.1,
    "delta_vs_best_anchor": 38.2
  }
}
```

## Tests

Add:

```text
tests/test_snapshot_ordo_ranking_script.py
```

Focused tests:

1. `select_snapshots` keeps only latest N by filename sort.
2. `select_snapshots(..., sort_by_mtime=True)` orders by file modification time.
3. pairing offsets produce expected pairs and no duplicates.
4. anchors are deduped from snapshots.
5. arena summary/game JSON converts to correct pseudo-PGN for both colors.
6. draw/no-winner converts to `1/2-1/2`.
7. Ordo output parser handles a representative rating table.
8. parser failure still allows summary writing with an error field.

Do not run real arena in unit tests. Mock/stub arena report loading.

## Implementation Steps

1. Add `scripts/run_snapshot_ordo_ranking.py` with pure helper functions first.
2. Add unit tests for selection, pairing, PGN conversion, and Ordo parsing.
3. Add arena execution wrapper using existing pairwise helper for PyTorch.
4. Add ONNX backend wrapper through `run_arena_checkpoints_onnx`.
5. Add Ordo subprocess function.
6. Add final `summary.json` writer.
7. Run targeted tests:

```bash
.venv/bin/python -m pytest tests/test_snapshot_ordo_ranking_script.py
```

8. Run a tiny smoke test on 3 snapshots with `--games 2`.
9. Run the intended latest-25 command on Runpod.

## Recommended Defaults

For frequent monitoring:

```text
max_snapshots = 25
pair_offsets = 1,2,4,8
games = 8
anchor_games = 16 if anchors are provided
backend = onnx
paired_seeds = true in arena config
```

For a cheaper quick check:

```text
max_snapshots = 16
pair_offsets = 1,3,8
games = 4
anchor_games = 8
```

For a deeper check every few hours:

```text
max_snapshots = 50
pair_offsets = 1,2,4,8,16
games = 8
anchor_games = 24
```

## Operational Notes

- Keep fixed anchors across runs. Otherwise the Elo baseline can drift.
- Do not rely only on latest-vs-previous; adjacent snapshots are highly correlated.
- Use paired seeds to reduce first-player/color bias.
- Keep raw arena reports and raw Ordo output. They make reruns and debugging much cheaper.
- If Ordo is unavailable, fail clearly with the install/check command, but keep `games.pgn` and arena reports.
