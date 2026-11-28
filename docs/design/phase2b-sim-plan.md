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

## Status (branch of T-B1d, base 1b69e7b)

Implemented and tested (`python3 -m pytest -q -p no:cacheprovider tests/test_endurance.py tests/test_interference.py`:
48 passed in about 6 s; tests/test_golden.py and tests/test_sweep.py pass unchanged):

- `hbf/workload.py`: `Request`, `Stream`, `scripted`, `per_stack`; tests `test_stream_closed_loop_is_seeded_and_staggered`,
  `test_stream_poisson_rate_and_prefix_choice`, `test_stream_sessions_and_script`.
- `hbf/endurance.py`: `SimOptions`, `Simulator`, `simulate`, `CapacityError`; policies packed, dedicated, hbm_decode
  (partitions none/channel/die) and layer_major; tests `test_10_read_disturb_refresh_tiny_device` (3 refreshes, 48
  pages, 4 openings, 16 block erases; replay variant), `test_11_retention_rewrites_idle_entry` (2 rewrites),
  `test_12_packed_without_pressure`, `test_13_compaction_two_segments` (4 pages, WA 1.125),
  `test_14_dedicated_superblocks` (10 erases, erase_wa 6.02; channel: 30 erases, 1.13),
  `test_15_wear_allocation` (100 vs 25 each), `test_layer_major_wave_replays_padding`,
  `test_shared_prefix_sessions_and_eviction`, `test_m5_hook_per_quantum`.
- Consistency with `analytic()`: `test_simulation_matches_analytic_in_steady_state` (packed without refresh and
  hbm_decode with whole-superblock entries: rate, KV write, P/E per day, WA within 1 %, measured 0.02-0.03 %);
  `test_simulation_refresh_with_decode_writes_is_below_analytic`: packed 32K + 2,048, X = 1e5 gives refresh_to_kv
  4.49 against the analytic 5 (-10 %), P/E per day -8 %. Reasons: the closed form counts generated pages as read from
  step 0, and relocated data lands in a partly filled frontier segment whose blocks gain reads more slowly. Other
  measured gaps (not asserted): reasoning refresh_to_kv 2.57 vs 5.0 (generated pages are 60 % of a request);
  agentic call rate -15 % (TPOT grows with the batch, so calls see a B-weighted step time); with M5 on, the
  simulation's request rate drops by TPOT/TPOT' (the closed form does not feed M5 back).
- CLI: `test_cli_simulation`, `test_cli_simulation_validation`; analytic report.md is byte-identical to 1b69e7b,
  the analytic summary only gains `hbm_bytes_per_request` and the corrected `kv_bytes_per_token` (HBF bytes).
- Coordinator requests: `calibrate()` locks `calib-<digest>/.lock` (fcntl) and writes calibration.json atomically.

Measured runtime (this container, F1 Clock, 0.25-0.5 simulated h, extrapolated per simulated day): long32k packed
B=8 about 35 s/day; agentic about 40 s; chat B=8 about 160 s; long32k die B=8 about 130 s; **long32k die B=64 about
2,000 s (22 ms per quantum: too slow, see Next steps 1)**. The 3 h hand check (32K + 2,048, X = 1e5) took 3.9 s.

## Next steps for the local agent (in order)

1. Speed of many partitions. Profiled (long32k, die, B=64, 0.1 h: 12.2 s): 9.7 s are in `_housekeeping` ->
   `_reclaim` -> `_victim`, 23,508 calls in 383 quanta. Cause: the default low watermark is per partition,
   2 + ceil(max request / segment) = 2 + 256 die segments, while a die holds 512; every die that holds a 4 GiB
   request sits below it, so each quantum scans O(S) arrays 64 times and finds no victim. Fix both: (a) skip
   `_reclaim(part)` while nothing changed since its last fruitless call (remember `self.drops` per partition, as
   `_admit_pending` does with `self.blocked`); (b) compute `_victim`'s per-segment copy counts once per
   housekeeping pass, not per partition. Next costs: `Clock.step` (`row_pages` once per partition, 1.0 s) and
   `_admit` (0.7 s). Target: a die-partition day in a few minutes.
2. `vllm_pool` simulation (design-B 4.3, test 16): merge `VllmTracker` from docs/design/drafts/endurance_simulator_draft.py.txt
   (lines 579-680; untested) into hbf/endurance.py; add "vllm_pool" to `SIM_POLICIES`; wear as packed (it reuses
   `_write`), plus a tracker fed with block ids at admission and decode (`_vllm_write` in the draft) that reports
   `violations_skip/overlap/live_erase`, `stranded_frac` (time average) and `legalize_gb`; set `legal` False; remove
   the CLI rejection in `configure`. Test 16 (hand values): allocate ids 0..7, free 2 and 5, allocate 2 with LIFO
   reuse -> 2 overlaps; stranded = 2/8 until the whole range is free. Then delete the simulator draft.
3. Optional flags from design-B: `--prefill colocated` (add input / `prefill_tok_s` serial seconds per admission
   to the quantum), `--victim cost_benefit`, `--pre-open`, `--separate-prefix`, wear `remap`.
4. Open questions to record in REPORT: (a) wear levelling only among free segments lets a few segments cycle when
   idle cached KV fills the device (chat with retain_s 600 and a hot prefix: max PEC/day 12x the mean); a static
   wear-levelling move would fix it; (b) hbm_decode sessions write the previous call's output with the next
   prompt (the closed form never writes it); (c) staged full pages of a finishing request are written (matches
   the closed form's P_r; dropping them would save under 0.1 % for long32k).

Local commands:

```
python3 -m pytest -q tests/test_endurance.py tests/test_interference.py tests/test_golden.py tests/test_sweep.py
python3 -m hbf.endurance --model llama-3.1-8b --format bf16 --workload long32k --concurrency 8 --alloc packed \
    --hours 24 --warmup-h 1 --rd-threshold 1e5,1e6 --output runs/sim-long32k
python3 -m hbf.endurance --model llama-3.1-8b --format bf16 --workload agentic --concurrency 8 --alloc packed \
    --hours 48 --warmup-h 2 --rd-threshold 1e5,1e6 --output runs/sim-agentic
```

Check in endurance_summary.csv: `avg_pec_per_day` and `refresh_to_kv` against `analytic_avg_pec_per_day` and
`analytic_refresh_to_kv` (long32k: within a few % at X = 1e6, where nothing is refreshed; agentic: refresh_to_kv
close, call rate lower as explained above), `stalls` = 0, `free_segments_min` > low watermark, `wall_s`; in
endurance_timeseries.csv that avg_pec grows linearly after the warm-up and free_segments is stable.

## Not done

- `vllm_pool` simulation (VllmTracker, violations, stranded_frac): the CLI rejects it without `--analytic`.
- `--prefill colocated`, `--victim cost_benefit`, `--pre-open`, `--separate-prefix`, wear `remap`.
- Fast path for die partitions at high concurrency (Next steps 1).
