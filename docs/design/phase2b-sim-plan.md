# Phase 2B part 2: event simulation (T-B1d) - implementation plan

> Working notes (English). Spec of the task: [phase2b-design.md](phase2b-design.md) sections 2-7 and
> [phase2b-decisions.md](phase2b-decisions.md). Status lives in PLAN.md (owned by the main agent).

## Reuse from the drafts (docs/design/drafts/)

- `workload_stream_draft.py.txt`: `Request`, `Stream` (seeded SeedSequence spawn for lengths, arrivals and
  prefixes; heap of timed arrivals; closed-loop refill; session call scheduling) and `scripted()`. Kept with
  fixes: per-stack overrides, a staggered closed-loop start (below), prefix/session fields in scripted requests.
- `endurance_simulator_draft.py.txt`: the Simulator skeleton (COO occupancy rows, entry arrays, `_write`,
  `_refresh`, `_reclaim`, wave start, summary columns). Rewritten where wrong or slow: free segments come from
  per-partition heaps (no O(S) mask per opening), running requests live in slot arrays (vectorized decode
  advance), reads accrue per entry and are applied at quantum end or before rows move, and space reservations
  include the remaining decode pages of running requests. `VllmTracker` is not merged (see Not done).
  The drafts are deleted once merged.

## Deviations from design section 4 (and the literal/conservative choices)

1. Read-threshold split: the quantum ends at the earliest predicted crossing, but never below
   `rd_slack` (default 1 %) of the shortest threshold in steps. A refresh is then late by at most 1 % of X,
   which bounds the bias of refresh counts to <= 1 % and keeps die segments at X = 1e5 tractable.
2. Requests that complete or arrive inside a quantum are handled at their exact step (sub-intervals); the
   step time stays constant inside the quantum (one-quantum lag, as for M5).
3. Decode pages staged in HBM are written as whole rows per partition at quantum end; at a request's
   completion its staged full pages are written (a final partial row), so host pages equal analytic `P_r`.
   Only the partial last page per (layer, head, HBF tensor) dies in HBM.
4. Pages staged in HBM are not read from HBF (they do not add block reads). Pages written inside a quantum
   start accruing reads in the next quantum.
5. Closed loops start staggered: the i-th initial request gets output `out - round(i out / B)` and the i-th
   initial session starts at call `floor(i calls / B)` (its history written at once). A warm-up device only;
   the first `--warmup-h` is excluded from all rates.
6. Simulation mode needs an explicit `--hours` (no default): without it `configure` raises a ValueError that
   points to `--analytic`. This keeps the existing test `test_cli_validation` unchanged.
7. Multiple stacks: one stack is simulated with concurrency `round(C / stacks)` (rate / stacks for Poisson).
8. `wear ideal` allocates like `dynamic` and reports max PEC = AVGPEC. `remap` is not implemented.
9. Admission reserves the request's whole expected footprint (prefill + decode pages) plus the low watermark
   minus one segment; if that fails after a forced reclaim it admits when the footprint alone fits.

## Scope order

1. Stream (`hbf/workload.py`): closed / Poisson, lengths, prefix groups (Zipf), sessions, `scripted()`.
2. `simulate()`: packed, dedicated, hbm_decode with partitions none/channel/die; then layer_major waves.
3. M5 per quantum (`interference(...)` with the quantum's programs and erases per step).
4. CLI and outputs: meta.json, endurance_timeseries.csv, endurance_segments.csv, endurance_summary.csv
   (analytic column names where they overlap, plus analytic reference columns), report.md (Chinese).
5. Tests below.

## Tests (tests/test_endurance.py), hand-computed

- Stream: same seed reproduces the sequence; Poisson mean inter-arrival within 2 % of 1/rate over 20,000
  draws; prefix hit share and Zipf top-group share; session calls `think_s` apart; scripted order.
- 10: banks 4, ppb 4, 8 segments of 16 pages, T = 10 ms, hbm_decode (no decode writes), 16-page request,
  X = 40 (4 reads per block per step, threshold 10 steps), out = 35: 3 refreshes, 48 refresh pages,
  4 segment openings, 16 block erases. Replay variant: 3 replays, 48 replay pages, 4 openings.
- 11: idle entry retained 50 h, T_ret = 24 h: 2 retention rewrites (2 x 16 pages).
- 12: packed, rows of 4, 10 requests of 16 pages: WA = 1, openings = ceil(160 / 16) = 10.
- 13: two segments; A = 12, B = 4, A dies, C = 16: compaction copies 4 pages, WA = 36 / 32 = 1.125.
- 14: dedicated superblocks, 8B BF16, 10 requests of 1,024 + 338 tokens: 10 superblock erases and
  erase_wa = 262,144 / 43,520 = 6.02 (the design's 5.95 counts the partial last page, 44,032 pages).
- 15: 4 segments, 100 sequential one-segment requests: wear none -> max PEC 100; dynamic -> 25 each.
- 16 (vllm_pool): not done.
- Consistency with `analytic()` (closed loop, packed allocation, no pressure, ConstantClock):
  (a) 8B BF16 32K + 338 out, decode writes on, no refresh: P/E per day within 2 %;
  (b) whole-superblock prompts with decode KV in HBM, X giving 4 refreshes per request: refresh_to_kv and
  P/E per day within 2 %. (c) packed with decode writes and refresh is reported, not asserted tightly (below).
- M5: fifo mode is linear, so the mean extra per step equals interference() at the mean load (within 3 %).

## CLI flags added

`--hours` (required without `--analytic`), `--warmup-h` (1), `--wear none|dynamic|ideal` (dynamic),
`--refresh auto|relocate|replay` (auto: replay for layer_major, else relocate), `--refresh-idle relocate|evict`,
`--reclaim evict|compact`, `--low-watermark` (segments, default 2 + one maximal request), `--report-s` (60),
`--quantum-s` (1), `--rd-slack` (0.01), `--slots` (layer_major), `--align row|none`, `--no-stagger`.
Existing flags reused: `--rd-threshold`, `--rd-hot`, `--retention-h`, `--prefix-reads`, `--m5`, `--seed`, ...

## Not done

- `vllm_pool` simulation (VllmTracker, violations, stranded_frac): the CLI rejects it without `--analytic`.
- `--prefill colocated`, `--victim cost_benefit`, `--pre-open`, `--separate-prefix`, wear `remap`.
