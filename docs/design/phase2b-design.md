# Phase B design: request dimension (M2), endurance (M4), write/refresh interference (M5)

> Working notes of the design discussion (English). The plan and status are in [PLAN.md](../../PLAN.md); parameter sources are in [docs/research/params.md](../research/params.md).

The repository state is HEAD d328067. Since 6868f47, `pipeline()` has gained `kv_format=` and `requests=(per_stack, heads)`, sweep.py is committed, and TensorSpec has the new role `aux`; this design accounts for all three. I edited no repository files and did not run MQSim.

---

## 0. Measurements and facts this design rests on

### 0.1 Phase A probes

Settings for every probe: analysis-only, `--topology ocp --read-plane-ns 0 --coalesce-ns 1 --cache-read --layout striped_contiguous --gpu b200`, dense mode unless noted.

| model | B | ctx | format / mode | pages/step | rounds | est ms | TPOT ms | GPU ms | max_block_reads | wall s |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 8B | 1 | 4K | bf16 | 131,072 | 128 | 0.70 | 3.04 | 2.35 | 128 | 0.2 |
| 8B | 8 | 4K | bf16 | 1,048,576 | 1,024 | 4.64 | 6.99 | 2.36 | 249 | 0.3 |
| 8B | 32 | 4K | bf16 | 4,194,304 | 4,096 | 18.18 | 20.52 | 2.41 | 255 | 1.1 |
| 8B | 64 | 4K | bf16 | 8,388,608 | 8,192 | 36.23 | 38.57 | 2.47 | 256 | 2.6 |
| 8B | 64 | 4K | kivi2 | 1,540,096 | 1,536 | 6.90 | 9.24 | 3.80 | 251 | 0.9 |
| 8B | 8 | 32K | bf16 | 8,388,608 | 8,192 | 36.23 | 38.57 | 2.47 | 256 | 2.6 |
| 8B | 32 | 32K | bf16 | 33,554,432 | 32,768 | 144.51 | 146.85 | 2.83 | 256 | 10.6 |
| 8B | 8 | 32K | fp8 | 4,194,304 | 4,096 | 18.18 | 20.52 | 2.93 | 255 | 1.1 |
| 8B | 8 | 32K | kivi2 | 1,568,768 | 1,568 | 6.97 | 9.32 | 3.84 | 256 | 0.5 |
| 8B | 8 | 32K | bf16, uniform_tokens 5% | 421,888 | 789 | 3.48 | 5.83 | 2.35 | 28 | 1.5 |
| 8B | 8 | 32K | quest, uniform_tokens 5% | 946,176 | 1,314 | 5.79 | 8.14 | 2.35 | 43 | 1.6 |
| 8B | 8 | 32K | bf16, sink_recent 5% | 425,984 | 416 | 1.96 | 4.31 | 2.35 | 13 | 0.3 |
| 70B | 8 | 32K | bf16 | 20,971,520 | 20,480 | 90.56 | 112.28 | 22.33 | 256 | 5.6 |

### 0.2 Findings from the probes

- **F1. TPOT surrogate.** `est_ns ≈ L × 4.0 µs + 4.406 µs × rounds` is within 3% for all 13 points. Example: 70B gives 80×4.0 + 20,480×4.406 = 90.55 ms, measured 90.56. This holds for tR = 4,096 and the HBF command model, where the per-round cost is about 1.0757·tR. Simulated time in M4 uses this surrogate plus `system.evaluate` (section 2).
- **F2. Decode is HBF-bound.** Dense decode on B200 is limited by HBF reads: GPU 2.4 ms against 36–145 ms of HBF for 8B.
  - At B=8, 32K, BF16 the device senses 217 M pages/s; with KIVI-2 it senses 168 M/s.
  - So when HBF-bound, compression raises steps per second and leaves senses per second roughly unchanged.
- **F3. Per-block reads per step.**
  - Dense: `max_block_reads` = pages per block (256), because every page of an active block is sensed once per step.
  - Uniform 5%: 28 per step (mean 12.9, per-step hot factor 2.2). Random selection averages out over many steps; persistent skew is unknown.
  - Quest: 43. sink_recent 5%: 13.
- **F4. Batch emulation is not reusable per request.** With B=8 at 32K, request slot 3 under `layer_contiguous` or `striped_contiguous` touches 32,768 NAND blocks, and every one is shared with other slots. A slot can therefore be rewritten only when all slots die together (wave reuse).
  - Under `vllm_interleaved`, slot 3 lives on only 128 banks: lpa mod 128 ∈ [48, 64). Its 4,096 blocks are not shared, so each request gets its own eighth of the banks.
- **F5. Geometry (8B BF16).**
  - 128 KiB per token; 16 tokens per page; 512 pages (2 MiB) per 16 tokens.
  - A 32K request is 1,048,576 pages = 4 GiB = 4 superblocks. A superblock is one block index across all 1,024 banks: 1 GiB at ppb 256.
  - A request of 1,362 tokens (ShareGPT-like 1,024 in + 338 out) is 44,032 pages = 0.168 superblocks.
- **F6. Inner-loop cost.** A numpy benchmark of one segment-level quantum (bincount read accrual, threshold scan, live recount, least-PEC free pick):
  - 0.035 ms with 512 segments;
  - 0.23 ms with 8,192 segments (channel blocks);
  - 0.72 ms with 32,768 segments (die blocks).

  That is 3–62 s per simulated day at 1 s quanta, before Python overhead per arrival.

### 0.3 Spec facts that drive the design (OCP HBF v0.7.0 text)

- **Write order (§5.4.1 items 4, 5, 9; §11.5.2.2; §11.5.2.6).** Writes are sequential within a NAND block with no skipping. Writing page 0 (WL-0, STR-0) auto-erases the block. A write to an already-written page is an Overlap error. Several blocks may be written concurrently ("can write multiple random NAND blocks simultaneously").
  - Item 4 says "no skipping … within a host channel's allocated global address range". This design reads it per NAND block (open question 1).
- **Replay (§5.7, Table 13 code 0x7).** The spec's address calculation is per host channel. With R1·R2 = 64 banks per channel, a bank's block consists of the channel-local units L2 + k·R5. With R5 = 64 this equals the repo's global map: a block is lpa stride 1024, 256 rows (`device_lpas` in `hbf/pages.py` agrees).
  - A refresh replays the block from page 0 to its last written page.
  - Data being replayed should be read from a host copy (§11.5.1.4).
- **No suspend.** The spec contains no program or erase suspend: "suspend" and "resume" do not occur in the text.
  - The only rule is §11.5: when refresh is interleaved with work, "refresh commands and read commands are not issued to the same die at the same time. Otherwise, some read operations may experience increased latency."
  - So M5's default blocking unit is the die (16 banks). That also matches MQSim, where a chip on `ocp` is an HBF die and a program makes the chip busy.
- **Block size hint (§11.5.1.8).** "The 1024 CSRs support 1MiB, 2MiB, or 4MiB per NAND block capacity." That is ppb ∈ {256, 512, 1024} at 4 KiB pages: 1024 CSRs × 32 bits = 32,768 blocks of 1 MiB = one 32 GiB channel. params.md's 192/256/768 should gain 512 and 1024.
- **No GC (§11.4).** There is no garbage collection. Zone remap (§11.4.1) changes only the logical-to-physical map; the host must invalidate or rewrite the data (§5.2.4.6).
  - Wear registers MAXPEC/AVGPEC (§5.9.2.17–18, Table 33 note 2) report the average, so "ideal wear leveling" means AVGPEC.
- **Retention and read disturb (§9 Table 33, §11.5).**
  - Retention: 24 h power-on at 85 °C; refresh "typically every 24–48 hours".
  - Read disturb: product-specific X; the base die keeps a per-block page-read counter (§4.4). CECC/UECC with "Read Count > Max_Read_Threshold" (§11.5.1.4–5).
- **Write accumulation (§5.4.1 items 6–7, §11.5.2.4–5).** A 4 KiB unit must complete within a host-set time limit (no value given; the BUCSTS 500 ms-unit timeout at 0x2975 is a different register). Pending units per channel are product-specific, with examples 64 or 128.
- **Channel partitioning (§13.3.3).** The spec suggests partitioning channels between weights and KV "from an endurance perspective". §13.2.2 is the "dedicated channel" model.

---

## 1. M2: request dimension (implement first)

### 1.1 Data model (new `hbf/batch.py`)

```python
@dataclass(frozen=True)
class BatchRequest:
    prompt: int                 # tokens written at prefill (full pages; the partial last page stays in HBM)
    generated: int = 0          # tokens decoded so far (full pages flushed; partial pages in HBM staging)
    prefix: int | None = None   # shared-prefix group: members map their first prefix_tokens to the owner's pages
    prefix_tokens: int = 0      # multiple of every HBF tensor's tokens per page (rounded down otherwise)

@dataclass(frozen=True)
class BatchSpec:
    requests: tuple            # BatchRequest per request of ONE stack (analysis batch b)
    allocation: str = "layer_major"   # layer_major | dedicated | packed | vllm_pool
    partition: str = "none"           # none | channel | die   (dedicated/packed only)
    intra: str = "striped_contiguous" # prompt layout inside a request (a LAYOUTS name)
    flush_order: str = "rotate"       # layer | rotate (decode flush page order)
    align: str = "row"                # packed: start every prompt on a 1,024-page row
    block_tokens: int | None = None   # vllm_pool: tokens per vLLM block (default: max tokens per page)
    free_order: str = "fresh"         # vllm_pool: fresh (ids ascending) | shuffled:<seed>
```

Design choices:

- **No change to `Access`.** Requests stay encoded as `kv_head = r*H + h`, as phase A does. This keeps `access.csv` and its header bit-identical, and `iter_layer_pages`/`layer_pages` need no change.
- **Per-request contexts in the generator.** `generate()` gains an optional `requests=` argument: a context per request and H. Each request's windows come from its own context (`TensorSpec.tokens(c_r)`) and its own block counts.
  - The loop order stays (step, layer, r, h), so the RNG draw order equals phase A's (step, layer, kv_head) order when all contexts are equal.
  - `uniform_pages` should build its Access from (layer, head, t, block) directly instead of the `page_access(lpa(...))` round trip. This changes nothing for existing layouts and avoids decoding shared-prefix pages to the wrong owner.
- **One restriction.** `layer_pages` uses `layout.window_starts[t]` as the block origin, so `PageTableLayout` must reject an HBF-tier tensor whose window start depends on the context. Every current preset starts its HBF tensors at 0; KIVI's residual is HBM-tier.
- **New `PageTableLayout(Layout)`.** Its `lpa_array(layer, head, t, block)` looks up a page table built by replaying the allocation history.
  - Storage: int32 per page with a per-request base offset and per-(t) strides, which is 134 MB for B=32 × 32K.
  - It also keeps `history`: an (n, 3) array of (write index, lpa, kind ∈ {prefill, decode, padding}) for the legality checker.
  - `decode()` uses an argsort over the table.
- **Synthetic history for a snapshot.** Request r is admitted at step `a_r = max_g − g_r`, so all requests end at the snapshot.
  - Its prompt pages are written at `a_r` in intra-layout order, padding included.
  - Each flush happens when a page of tensor t completes (every tpp_t tokens). Requests interleave in step order.
- **Pipeline wiring.** `pipeline(requests=...)` takes a `BatchSpec`; the current `(per_stack, heads)` tuple is its `layer_major` special case. New flags in `hbf/cli.py`:
  - `--alloc`, `--partition`, `--contexts LIST|dist:...`, `--generated LIST|N`, `--prefix-tokens`, `--prefix-groups`, `--flush-order`, `--block-tokens`, `--free-order`, `--check-writes`.
  - meta.json gains a `"batch"` block only when it is not the default.
  - `kv_bytes_per_request` becomes the mean over requests.

### 1.2 Address mapping per policy

Notation (static OCP map, consistent with `hbf/pages.py::device_lpas` and spec §5.7):

- host channel = lpa mod 16
- channel-local unit i = lpa // 16; bank in die = i mod 16; die in channel = (i // 16) mod 4
- page row = lpa // 1024; block = row // ppb; page in block = row mod ppb
- **row**: 1,024 consecutive lpa, one page on every bank (4 MiB)
- **superblock j**: rows [j·ppb, (j+1)·ppb), i.e. 1 GiB at ppb 256
- **channel block (ch, j)**: {16·i + ch : i ∈ [64·ppb·j, 64·ppb·(j+1))}, i.e. 64 MiB
- **die block**: 16 banks × ppb, i.e. 16 MiB

Every policy except `layer_major` and `vllm_pool` first maps a page to an intra-request virtual index v:

- prompt pages: `v = intra.lpa_array(layer, h, t, block)` on Shape(L, H, d, prompt_r), base 0, padding included;
- generated pages: `v = span_prompt + k`, where k is the page's rank in flush order;
- `rotate`: in flush g, page (l, h, t) takes in-flush slot `((l + g) mod L)·H·T + h·T + t`;
- `layer`: slot `l·H·T + h·T + t`.

| policy | v (or page) → lpa |
|---|---|
| **(a) `layer_major`** | Today's `LAYOUTS[intra](cfg, Shape(L, H·B, d, ctx))`, exactly the existing code path. All contexts must be equal (ValueError otherwise: it reserves equal slots). |
| **(b) `dedicated`, partition none** | Request r takes whole superblocks s₀, s₁, … from the free pool (wear policy picks); `lpa = s_{v // P_sb}·P_sb + v mod P_sb`, with P_sb = 1024·ppb. Generated pages are appended inside the request's own segments. |
| **(b′) `packed`, partition none** | One global frontier F. Each write burst (a prompt or a flush) takes [F, F+n). Prompts start on a row boundary (`align=row`: at most 1,023 padding pages per request). The frontier jumps to the next free superblock when the current one fills. |
| **(c) `dedicated`/`packed`, partition channel** | Request r gets channel ch_r (least pages, ties to lowest). Channel-local frontier i gives `lpa = 16·i + ch_r`; segments are channel blocks. Partition `die`: die-local k gives `i = (k//16)·64 + die·16 + k mod 16`. This is the spec's "dedicated channel" model (§13.2.2). |
| **(d) `vllm_pool`** | Per-layer pools of `num_blocks × H × Σ_t block_tokens/tpp_t` pages (pool bases multiples of 1024). The vLLM block id comes from the request's block table, allocated from a free queue (`fresh` ascending or `shuffled:<seed>`). `lpa = base + l·pool_pages + id·(H·Σ_t n_t) + h·Σ_t n_t + off_t + (b mod n_t)`, with n_t = block_tokens / tpp_t. This assumes one head per page (HND), as `TensorSpec` already does. |

Shared prefixes: the members' table entries for prefix blocks point at the owner's pages. Phase A deduplicates pages per (step, layer), so this snapshot models "shared prefix read once per step" (cascade attention). Per-sequence multiplicity is handled analytically in M4.

### 1.3 Consequences per policy

Read parallelism is measured by phase A on the snapshot: `iter_layer_pages → analyze_pages → system.evaluate`, unchanged.

| policy | read parallelism (phase A via LayerPages) | write / erase consequence |
|---|---|---|
| (a) `layer_major` | Today's results; balanced with striped layouts. | Every block is shared by all slots (F4), so a single request cannot be replaced: only static-batching waves are legal, and each wave rewrites the whole region (P/E +1 per block, reserved and padding pages included). Decode appends into the reserved regions are illegal (section 1.4), so generated KV must stay in HBM until the wave ends. Prefix sharing is impossible. |
| (b) `dedicated` superblocks | Each request is striped over all 1,024 banks, so dense reads are balanced. Generated pages: with `layer` order, a flush is half a row, so layer l lands on 32 banks; `rotate` spreads it across all banks over 64 flushes. | A request's segments die together, so no compaction is ever needed. Erase amplification = ⌈P/P_sb⌉·P_sb / P: 5.95× at 1,362 tokens, 2.0× at 4K, 1.24× at 32K+338 (ppb 256). |
| (b′) `packed` | Same as (b) for prompts. Generated pages from all requests interleave at the frontier. | Erase amplification ≈ 1, but segments hold data with mixed lifetimes, so the host must evict or compact. Requires a host page table (vLLM already keeps block tables). |
| (c) channel / die partition | Per request: 64 or 16 banks. Per layer: rounds = max over partitions of (pages in partition / banks per partition). With B=8 equal requests on 8 channels the layer takes 2× the striped time; it is balanced only when load is even across 16 channels (needs B ≥ 16; die partitions need B ≥ 64). | 64 MiB or 16 MiB erase unit: 1.12× erase amplification at 1,362 tokens. Writes, erases and refresh are confined to one partition's dies, but a synchronous layer read still waits for the slowest partition. |
| (d) `vllm_pool` | Per-layer pools put block id → bank group (16·id) mod 1024: balls into 64 bins. With fresh ids κ = 1; with shuffled ids κ ≈ 1.4 at B=1 32K and ≈ 1.2 at B=8 (estimate; the snapshot measures it). The block-major variant (phase A's `vllm_interleaved`) maps layer l to 32 banks. | Free-list reuse writes into the middle of blocks that hold other live data, giving Overlap/order errors (§11.5.2.2, §11.5.2.6). The only legal reuse is a whole superblock range of one pool once all its ids are free, so free blocks are stranded. |

### 1.4 Write-legality checker: a cheap, parameter-free result for item 5

`check_write_order(cfg, lpas_in_write_order, live=None)`:

1. Block id and page in block come from `place`/`block_ids`.
2. A stable sort by block keeps time order within each block.
3. Within each block:
   - page 0 is an erase; it is legal only when no live page remains in the block;
   - every other write is legal only when page = previous + 1.
4. Counts:
   - `skip`: §11.5.2.6 sub-code 0x1;
   - `overlap`: §11.5.2.2;
   - `live_erase`: data loss;
   - `open_gaps`: blocks left non-contiguous at the end.

Run it on every snapshot history (`--check-writes`). Expected results:

- `layer_major` and any reserved per-tensor layout with decode appends are illegal (overlaps, or skips at prefill unless the reserved tail is padded);
- `vllm_pool` with shuffled ids has overlaps;
- `dedicated` and `packed` are legal.

---

## 2. Clock: simulated time from phase A (implement second)

```python
@dataclass(frozen=True)
class Clock:
    model: ModelSpec; gpu: GpuSpec; options: SystemOptions; tensors: tuple  # all tiers
    page_bytes: int = 4096; read_ns: int = 4096; banks: int = 1024; stacks: int = 1
    layer_ns: float = 4000.0; round_ns: float = 4406.0    # F1 fit (ocp, HBF flags, tR 4,096)
    balance: float = 1.0                                   # kappa = rounds / ceil(pages/banks), from M2
    read_share: dict = {"data": 1.0, "aux": 1.0, "meta": 1.0}  # pages sensed / stored per step, per role
    hot: float = 1.0                                       # max/mean block reads (persistent skew), swept
    def step(self, contexts, partitions=None) -> StepTime   # tpot_ns, h_ns, gap_ns per layer, pages per layer
```

How `step()` works:

1. For the running set it computes:
   - `pages_l = H · Σ_r Σ_t ⌈w_t(c_r)/tpp_t⌉ · share(role_t)`, using HBF-tier window lengths;
   - `rounds_l = ⌈κ·pages_l/1024⌉`, or for partitions `max_p pages_p/banks_p`;
   - `est = layer_ns + round_ns·rounds`, `bound = rounds·tR`;
   - `burst_bytes`, `hbm_bytes`, `attended_tokens`, `attended_elements`, `dequant_ops` from the tensor specs.
2. It feeds L identical rows into the unchanged `system.evaluate` with `batch = B·stacks` and `analysis_batch = B`, and returns `tpot_hbf_est_ns`.
3. Results are cached by (B, Σpages in 0.5% buckets). `evaluate` costs about 0.3 ms per call.
4. Gaps for M5: `g_l = pre + post + max(0, a_l − h_l)` (serial/stream), or the prefetch timeline's idle intervals.

`calibrate(cfg, model, gpu, fmt, mode, generation, batch_spec)` runs at most three phase A analysis-only pipelines into `<out>/calibration/`:

- two points (B=1 and B=8 at the run's context) refit `layer_ns` and `round_ns` for the actual cfg (tR, bandwidth grade);
- one M2 snapshot of the policy gives κ, `read_share` (pages read / stored per role) and the per-step hot factor.

This takes 1–15 s. `--no-calibrate` uses the F1 constants and is valid only for the HBF command model at tR 4,096. The result is cached by a hash of the inputs, so sweeps reuse it.

Prefill is disaggregated by default: KV arrives already written and costs the decode clock nothing except M5 interference. `--prefill colocated` adds `tokens/prefill_tok_s` of serial time per admission, using the params.md §6 analytic prefill rate at MFU 0.5.

---

## 3. M4a: closed-form steady state (`hbf/endurance.py::analytic`, implement third)

This mode exists for instant curves and as the oracle for the simulator's tests. Assume closed loop with B running requests (or a given rate λ), T = Clock TPOT, capacity C pages, and ppb.

- **Rates and footprint.**
  - Request rate: λ = B / (out·T), or the given λ.
  - Pages per request: `P_r = L·H·Σ_t ⌊w_t(in_uncached+out)/tpp_t⌋`. Partial pages are dropped at the end unless the KV is retained.
  - Host KV write: W_kv = λ·P_r·page.
- **Read-disturb refresh.**
  - Reads per block per step: ρ = ppb · p̄ · h, where p̄ = read share and h = hot factor. For shared prefixes multiply by the number of sharers under per-sequence reads.
  - Refreshes per lifetime: n = ⌊s·ρ/X⌋, where s = steps that read the data (out, plus 1 per prefill hit; Σ out over the calls of a session).
  - Refresh write: λ·n·P_r·page.
  - Key closed form: for dense reads the threshold is **X/ppb steps**, independent of TPOT and of compression. It is 390.6 / 3,906 / 39,063 steps at X = 1e5 / 1e6 / 1e7.
- **Retention.** Live bytes older than T_ret are rewritten every T_ret.
- **Erase amplification.** Dedicated segments: `⌈P_r/P_seg⌉·P_seg/P_r`.
- **Lifetime.** P/E per block per day (ideal = AVGPEC) = `(Σ programs × erase_amp) · 86,400 / C`; days to limit = N_PE / that.

Worked example (8B BF16, B=8, 32K uncached, out 338, T = 38.57 ms):

| quantity | value |
|---|---|
| request rate | 0.614 req/s |
| KV write | about 2.66 GB/s |
| P/E per day | about 418 |
| days to 1e5 (ideal wear leveling) | about 239 |
| decode-only write rate | 27.2 MB/s |
| refresh write if all active data lived past X (X = 1e6) | read BW 0.89 TB/s × 256 / X = 228 MB/s, i.e. 8.4× the decode KV write |
| same, X = 1e5 | 2.28 GB/s |
| does a 338-token request reach X at all? | No when X ≥ 1e5 (338 < 390 steps), so its own KV never needs read-disturb refresh. Agentic histories and shared prefixes do. |

---

## 4. M4b: event simulation (`hbf/endurance.py::simulate`, implement fourth)

### 4.1 State

Everything is struct-of-arrays in numpy, at segment granularity. This is exact for row-aligned placement: a row spans every bank of the segment, so an entry with n pages in a segment puts n/banks_seg pages in each of its blocks.

- **Segments** (S = 512 superblocks, 8,192 channel blocks or 32,768 die blocks, derived from `--capacity-gib` and ppb, not from `Config.op_ratio`):
  - `fill`, `live`, `active_live`, `pec`;
  - `erases[cause]` with cause ∈ {kv, refresh_rd, refresh_ret, compaction, replay, wave};
  - `reads` (block reads since erase, times hot);
  - `oldest_write_t`; `state` ∈ {free, open, full}.
- **Entries** (a KV object: request-private, shared prefix, or session history):
  - `pages`, `readers`, `read_share`, `state` ∈ {active, idle, dead}, `ttl_end`, `written_t`.
- **Occupancy**: COO arrays (entry, segment, pages, write_t) with a valid mask and periodic compaction.
- **Requests**: arrival, input, cached, out, generated, entry ids, admit_t.
- **Workload** (`hbf/workload.py`, seeded): closed (concurrency) or Poisson (rate, max_batch).
  - Input/output lengths: `fixed:` or `lognormal:median=,sigma=`.
  - Prefix groups (count, tokens, share, Zipf popularity).
  - Agentic sessions (calls, new tokens, out per call, think time, initial history).
  - Presets `chat`, `long`, `agentic`, `reasoning`, each carrying its params.md §6 source tag.

### 4.2 One quantum Δ

Δ is adaptive: at most 1 s, split at the earliest predicted read-threshold or retention crossing, because rates are constant inside a quantum.

1. **Arrivals and admissions.** Admit while running < max_batch and the space reserve can be met (otherwise reclaim, otherwise queue).
   - A prefix hit adds no write; readers +1 on the prefix entry; prefill attention adds one read per prefix page.
   - Prefill writes of uncached pages go to the policy's frontier(s).
   - **Auto-erase happens here.** Opening a segment means writing its row 0, which erases every block of it: `pec += 1`, `erases[cause] += 1`, `reads = 0`. Every page-0 write counts, even on a never-written segment (spec-literal; open question 2).
2. **Clock.** `T = Clock.step(contexts) + δ_M5(previous quantum)`; steps = Δ/T; per-request generated += steps.
3. **Decode flushes through HBM staging.** `new_pages_r = P(c_r + Δg) − P(c_r)`, counting full pages per HBF tensor; KIVI tokens reach HBF only after leaving the residual window.
   - Pages are aggregated into whole rows (1,024 pages, or 64/16 under partitions) before programming. This matters for M5.
   - The staging peak is recorded: one partial page per (layer, head, HBF tensor) per running request, which is 2 MiB per request for 8B BF16 and 4 MiB for KIVI, plus up to one row (4 MiB).
4. **Reads.** `seg.reads += bincount(coo.seg, coo.pages × rate[coo.entry] × steps) / banks_seg × hot`, where `rate` = readers × read_share (per-sequence) or min(readers, 1) × read_share (`--prefix-reads once`).
5. **Completions.** A private entry dies, or becomes idle for `retain_s` (prefix cache or session); shared entries lose a reader. When a segment's `live` reaches 0 it is reclaimable. Its erase is deferred to its next row-0 write (step 1).
6. **Read-disturb refresh** (`seg.reads ≥ X`):
   - `relocate` (default for log policies): read the live pages and append them at the frontier (cause refresh_rd); the old segment's live becomes 0.
   - `replay` (static policies, spec §5.7): rewrite rows 0..fill−1 in place, dead pages included, because there is no skipping. That is +1 P/E and `fill` programs. It needs an HBM copy of the segment (1 GiB for a superblock), and the data is served from HBM meanwhile.
   - Idle cached entries can be evicted instead (`--refresh-idle evict`).
7. **Retention** (`now − oldest_write_t ≥ T_ret`, default 24 h): relocate, replay, or evict idle entries (cause refresh_ret). `--retention off` is an ablation.
8. **Reclaim** while free segments < low watermark (default 2 segments plus one maximal request):
   - victim = minimum `active_live` (greedy; `--victim cost_benefit` optional);
   - evict its idle entries first; the recompute cost is logged as future prefill tokens;
   - copy any remaining active pages (cause compaction: reads + programs);
   - `--reclaim compact` copies idle entries too.
9. **Allocation of free segments** by `--wear`:
   - `none`: lowest index first;
   - `dynamic`: least PEC;
   - `ideal`: report AVGPEC;
   - `remap` (static policies): the wave region rotates over zones at zero cost, because zone remap (§11.4.1) is free when the data is already dead.
10. **Accounting.** Bytes per cause, erases per cause, the M5 inputs for the next quantum, and a timeseries row every `--report-s` (default 60 s). The first `--warmup-h` (default 1 h) is excluded from rates.

### 4.3 Policy-specific behaviour

- **`layer_major`**: static-batching waves. The region is built with the phase A Layout of H·B heads at the maximum context. A wave starts when all slots are free, rewrites the whole region (cause wave), keeps generated KV in HBM (staging = B·out·KV per token), and ends at its longest request.
- **`vllm_pool`**: a vLLM-like block manager runs alongside. Its free queue is LRU, with cached blocks kept until evicted. It tracks fill row and live count per (pool, superblock range, bank group) and reports:
  - `violations` by type;
  - `stranded_frac`: free ids inside ranges that still hold live ids;
  - the cost of legalising (emergency replay of the target range).

  Wear numbers for this policy are labelled "not legal as specified".

Runtime estimate: 10–120 s per simulated day for superblock and channel segments, up to about 5 minutes for die segments with quanta of 0.1 s or less. That meets "minutes for days".

---

## 5. M5: write/refresh interference (`hbf/interference.py`, implement fifth; M4 calls it each quantum)

### 5.1 Inputs and wave model

Inputs per step:

- n_p: pages programmed;
- n_e: blocks erased;
- s: banks per write wave (1,024 for row-striped writes; 64 or 16 under partitions; 1 for single-bank replay);
- per-layer h_l and g_l from the Clock;
- tPROG, tBERS;
- D = 64 dies, q = 16 banks per die.

Under all-bank striping every layer reads every die, so **the busiest die sets the delay, not the average**.

- Program waves per step: K = n_p/s, each blocking its dies for tPROG.
- Erase waves: M = n_e/s, each tBERS.
- Device write duty: W_dev = K·tPROG + M·tBERS.
- Idle time per step: G = Σ g_l.

### 5.2 Modes

- **`fifo`** (no suspend, writes not deferred):
  - extra per step = W_dev; per layer = W_dev / L.
- **`gaps`** (no suspend; host or read-priority defers non-preemptive operations to HBF-idle gaps, which is also what §11.5 asks for):
  - c = ⌊g/tPROG⌋ operations per gap are free;
  - of the extra operations n_x = (K − cL)⁺, the first min(n_x, L) cost `(c+1)·tPROG − g` each and the rest cost tPROG each;
  - erases are placed the same way with tBERS;
  - for non-uniform gaps, use the largest gaps first.
- **`suspend`** (product-specific, not in OCP v0.7.0; MQSim T-SUSP):
  - reads preempt; operations progress only in gaps;
  - extra per step = N_s·t_s + (W_dev − G)⁺, where N_s = min(L, ⌈W_dev / g⌉) is the number of layers that start with a suspended operation;
  - the overflow term is unavoidable when writes do not fit in idle time.
- **`random`** (uncoordinated, independent per die, e.g. per-bank replays; per-die busy fraction u):
  - `δ_l = T_op − T_op/(u(D+1))·(1 − (1−u)^{D+1})` (expected maximum residual over D dies);
  - the correlated striped case is `δ_l = u·T_op/2`.
- **Bank-level blocking** (`--block-unit bank`) is an optimistic ablation: D = 1,024 units with q = 1.

### 5.3 Feedback, spikes and the rule it implies

- **Feedback:** `TPOT' = TPOT + extra_step`, evaluated in M4 with one-quantum lag, which converges in 2–3 quanta. Step-driven writes (flushes, read-disturb) scale per step; time-driven writes (retention, arrivals) scale per second.
- **Spikes:** a single erase wave (tBERS 1–10 ms) or an unspread superblock refresh (ppb·tPROG = 25.6 ms + tBERS) is reported as `max_step_extra`, an approximate p99.
- **Rule this implies:** refresh must be spread over the CECC margin, one row per gap.
- **Host trick:** pre-open segments in idle time (write row 0 early) to take erases off the write path (`--pre-open`).

### 5.4 Worked numbers (8B, B=8, 32K)

g = pre + post = 7,864.32 + 60,293.12 = 68,157.44 ns, so G = 2.181 ms. tPROG = 100 µs.

| load | fifo | gaps | suspend (t_s = 5 µs) |
|---|---|---|---|
| decode flushes alone: 256 pages/step = 0.25 waves | 25 µs per step | 8 µs per step | — |
| refresh at X = 1e5: 21.5 waves/step | 2.15 ms (+5.6%) | 0.68 ms (+1.8%) | 0.16 ms (+0.4%) |

Erases at 2.1 superblocks/s average 82–820 µs per step, with 1–10 ms spikes.

Array program bandwidth is 1024 × 4 KiB / tPROG = 55.9 / 41.9 / 21.0 / 12.0 GB/s at tPROG = 75 / 100 / 200 / 350 µs. Write bandwidth is never the limit; stolen read time is.

### 5.5 Phase A hook and MQSim confirmation

**Phase A hook.** `python -m hbf.interference RUN_DIR --write-gb-s X --erases-per-s Y --mode gaps [--suspend-ns N]` reads `system_layers.csv` (pre, post, attn, hbf_est) and writes `interference_steps.csv` with TPOT′. `system.evaluate` stays unchanged.

**MQSim confirmation is feasible on a small topology.** Use `load("4x1", planes=16, plane_rule="independent", read_plane_ns=0, coalesce_ns=1, cache_read=True)`: 64 banks, about one host channel, low memory. Hand-built batches go to `simulate(..., overlap=True)`, as in test_t6.

1. Baseline dense layer: `Shape(1,8,128,8192)` after a prefill.
2. One 64-page row written to fresh LPAs at t₀, with the read arriving at t₀ + 1 µs. Expect about +tPROG minus the overlap; the die blocks, because an MQSim chip is a die.
3. The write arrives mid-layer under PRIORITY_OUT_OF_ORDER. Expect the program to run after the die's reads, delaying the next burst. This confirms the `gaps` premise.
4. K = 4 rows, to check linearity.
5. A T-SUSP micro on a chip-as-plane topology (cache_read off, as `Config` and `simulate()` require). This one tests bank-level blocking only.

Erase cannot be confirmed: MQSim erases only through GC, which the harness asserts against.

---

## 6. Outputs, CLI and sweep (implement sixth)

**CLI:**

```
python -m hbf.endurance --model llama-3.1-8b --gpu b200 --format kivi2 --mode dense \
    --alloc packed --partition none --workload long --concurrency 8 | --rate 0.5 \
    --hours 24 --warmup-h 1 --rd-threshold 1e6 --rd-hot 1 --pe-limit 1e4,1e5,1e6 \
    --retention-h 24 --wear dynamic --refresh relocate --reclaim evict --prefix-reads per_request \
    --program-ns 100000 --erase-ns 1000000 --pages-per-block 256 --capacity-gib 512 --stacks 1 \
    --m5 gaps [--suspend-ns 5000] [--analytic] --output DIR
```

Device flags reuse the names in `hbf/cli.py`. `build_parser()`, `configure(args)`, `run(directory, **config)`, `summarize_run(directory)` and `METRICS` mirror hbf.cli, so the sweep can drive it.

**Files in DIR:**

- `meta.json`, `calibration.json`
- `endurance_timeseries.csv`:
  - t, running, queued, B, Σctx, tpot, tpot_m5
  - bytes per cause: prefill, decode, refresh_rd, refresh_ret, compaction, replay
  - erases per cause, free segments, live / stranded pages, avg/max PEC, max reads/X
- `endurance_segments.csv`: per-segment PEC, erases per cause, live pages
- `endurance_summary.csv`, one row with:
  - throughput: `tokens_per_s`, `mean_batch`, `mean_tpot_ms`, `tpot_m5_ms`, `max_step_extra_ms`
  - write rates: `kv_write_gb_s`, `prefill_gb_s`, `decode_gb_s`, `refresh_rd_gb_s`, `refresh_ret_gb_s`, `compaction_gb_s`, `program_gb_s`, `program_bw_share`
  - amplification: `wa` (programs / host KV), `erase_wa`, `refresh_to_kv`, `erases_per_s`
  - wear: `avg_pec_per_day`, `max_pec_per_day`, `days_to_limit_{avg,max}@N`, `pe_per_mtok`
  - capacity and legality: `stranded_frac`, `evicted_gb`, `recompute_tokens`, `violations_*`, `staging_peak_mib`, `pending_units_per_channel`, `replay_buffer_mib`
- `report.md`: a Chinese section "寿命评估" plus an assumptions list, each assumption with its source tag.

**Sweep.** Add a top-level grid key `"tool": "endurance"` (default `"analysis"`). `load_grid` checks keys against the chosen tool's parser, `run_one` dispatches on `TOOLS[tool] = (build_parser, configure, run, summarize_run, METRICS)`, and the calibration cache lives in `DIR/calibration/`. This is about 30 lines in `hbf/sweep.py`.

---

## 7. Tests with hand-computed values

**`tests/test_batch.py` (M2):**

1. `layer_major` equals phase A bit-identically. For layouts {layer_contiguous, striped_contiguous, vllm_interleaved} × modes {dense, uniform_tokens seed 4, uniform_pages seed 4}, compare Shape(2, 8, 128, 2048) with B=4 through `--alloc layer_major` against today's `--batch 4`: identical `analysis_layers.csv` and `access.csv`. A `PageTableLayout` built by the generic replay of `layer_major` must also give `lpa_array` identical to the Layout for every (l, head, t, block).
2. Equal-context `generate(requests=...)` reproduces today's Access list exactly.
3. `dedicated` with `--pages-per-block 4` (superblock 4,096 pages), L=2, H=8, ctx 2,048 (4,096 pages per request): request r occupies exactly lpa [4096r, 4096r + 4096). A dense layer gives 2,048·B pages and 2B rounds.
4. Channel partition, 8B dims at 32K: one request per channel for B=8 gives 512 rounds per layer against 256 striped; B=16 gives 512 against 512.
5. `vllm_pool` with fresh ids: layer l of one request with 64 blocks covers 1,024 distinct banks (1 round). With all ids ≡ 0 mod 64 it covers 16 banks (64 rounds).
6. Flush order, B=8, 64 flushes, 8B dims: `layer` puts layer l's generated pages on 32 banks; `rotate` puts them on 1,024.
7. `check_write_order`:
   - writes 0, 1, 2 → legal;
   - 0, 2 → one skip;
   - 0, 1, 1 → one overlap;
   - `layer_major` snapshot with generated = 16 → overlaps > 0;
   - `packed` and `dedicated` → 0.

**`tests/test_endurance.py` (M4):**

8. Clock against phase A: within 0.5% of TPOT for (B=8, 32K bf16) 38.57 ms, (B=32, 32K) 146.85, (B=64, 4K kivi2) 9.24, (70B B=8 32K) 112.28.
9. Analytic:
   - steps to X = 1e6 / 256 = 3,906.25;
   - a 32K request with out 4,000 gives 1 refresh = 1,048,576 pages;
   - out 338 at X = 1e5 gives 0;
   - the B=8 closed-loop example gives 418.5 P/E per day ± 0.2%.
10. Tiny device (banks 4, ppb 4, 8 segments of 16 pages), constant TPOT 10 ms, dense, decode writes off, one 16-page request, X = 40: refresh every 10 steps, so out = 35 gives 3 refreshes, 48 refresh pages, 4 segment openings and 16 block erases.
11. Retention: an idle entry alive 50 h with T_ret = 24 h gives 2 rewrites.
12. Packed without pressure: WA = 1; erases = ⌈Σpages / P_seg⌉.
13. Scripted compaction, two segments: A = 12 pages, then B = 4 fill segment 0; A dies; C = 16 needs space; victim segment 0 with live B. Expected: compaction copies 4 pages, WA = (12+4+16+4)/32 = 1.125.
14. Dedicated superblocks: 10 requests of 1,362 tokens give 10 superblock erases and erase_wa 5.95.
15. Wear: 4 segments, 100 sequential one-segment requests. `none` gives max PEC 100 when the lowest free index is always reused; `dynamic` gives 25 each.
16. vLLM script: allocate ids 0..7, free 2 and 5, allocate 2 with LIFO → 2 overlaps; stranded = 2/8 until the whole range is free.

**`tests/test_interference.py` (M5):**

17. fifo, K = 1, tPROG = 100 µs → 100 µs.
18. gaps, g = 68,157.44, one operation per layer → 31,842.56 ns per layer.
19. suspend, t_s = 5 µs, K·tPROG = 2.15 ms ≤ G = 2,181,038 ns → 160 µs per step; with K·tPROG = 3 ms → 160 µs + 818,962 ns.
20. random, D = 64, u = 0.01, T = 100 µs → 26.2055 µs; correlated → 0.5 µs.

All tests run analysis-only (seconds); none runs MQSim.

---

## 8. Critique of the plan

### Unnecessary as specified

- **Per-page or per-block state and per-step events.** Segment-level state is exact for row-aligned placement (F3/F4). Per-block arrays (524,288) per quantum would cost about 30× more for nothing.
- **Running phase A inside the simulation.** Use the Clock surrogate (F1), with a phase A cross-check test.
- **vLLM free-list as a full wear policy.** Its legal form is `packed`. Report violations and stranded capacity only.
- **Zone-remap mechanics.** KV churn makes least-PEC allocation over free segments nearly ideal. Report ideal (AVGPEC) and max bounds; remap matters for static data such as weights.
- **Host compaction as the default reclaim.** Serving systems drop cached KV and recompute it. Compaction is needed only for active data, so make eviction the default.
- **"Write bandwidth vs program bandwidth" as a headline.** It is always under 10% (≤ 2.7 GB/s against 41.9 GB/s). The meaningful number is stolen read time (M5).

### Missing from the plan (added above or flagged)

- Write-order legality checker (§1.4): parameter-free, the strongest item-5 evidence.
- Prefill reads of cached prefixes, and per-sequence shared-prefix multiplicity: read-disturb hot spots. A 64-way-shared prefix reaches X = 1e6 in 61 steps under per-sequence reads.
- Hot/cold stream separation: a hot prefix in a 1 GiB segment drags cold neighbours into each refresh (`--separate-prefix`).
- HBM costs: staging (2–4 MiB per request), replay copies (1 GiB per superblock), and generated KV kept in HBM under `layer_major`.
- Trickle-writing to the device is infeasible. B=8 needs 4,096 open 4 KiB units (256 per channel) against product examples of 64–128 per channel (§5.4.1 item 7), each open for tpp·TPOT = 0.62 s (§5.4.1 item 6). Report it as a table.
- A read-only-HBF baseline policy (`hbm_decode`, H³-style: generated KV never written to HBF).
- Weights on HBF (the spec's §13 use case): weights are read every token, so weight blocks need refresh every X/ppb steps forever. Out of scope (decision 6) but worth flagging in the report.
- Multiple stacks (rate/N per stack), capacity reserved for other data, `--capacity-gib` independent of ppb. `Config` changes capacity when only `--pages-per-block` changes.
- Wear-dependent X and retention (X falls as PEC grows): curve only. Program energy (640 pJ/B, SPLASH): one line.

### Robust results (report as numbers)

- Legality conflicts (spec rules).
- Erase amplification from reclaim granularity (geometry; ppb only, and the spec hints 256/512/1024).
- Dense read-disturb threshold in steps = X/(ppb·p̄·h), independent of TPOT and compression.
- Refresh bytes ∝ senses × ppb / X.
- KV bytes per token and per request, and their scaling with the compression ratio at fixed offered load.
- Decode write ≪ program bandwidth.

### Fragile results (report as curves)

| result | sweep |
|---|---|
| days to limit | N_PE 3e3–1e6, log axis |
| whether and how much refresh happens | X 1e5–1e7 and skew h 1–16 |
| M5 overhead | tPROG 75–350 µs, tBERS 1–10 ms, suspend absent/5/20 µs, blocking unit die/bank |
| compaction WA | capacity headroom, TTL, eviction on/off |
| ppb | 192, 256, 512, 1024 |

Also report the **break-even X** at which refresh writes equal KV writes for each workload.

### Compression framing (item 4)

Report three numbers per format:

1. lifetime at fixed offered load, where the gain ≈ compression ratio minus granularity effects (dedicated segments get worse for small KV);
2. P/E per million output tokens;
3. lifetime at maximum throughput, where bytes per second stay about the same (F2).

Per block, dense read disturb per step is unchanged by compression. Sparse selection divides it by 1/p̄.

---

## 9. Experiment matrix (items 3, 4, 5 write side)

Runtimes are estimated from F6 and the probe wall times; sequential unless "3 jobs" is stated.

| id | question | grid | cost |
|---|---|---|---|
| E3.1 | Item 3, curves | analytic: {8B, 70B} × {chat, long32k, agentic, reasoning} × X {1e5, 1e6, 1e7} × N_PE {1e4, 1e5, 1e6} × ppb {192, 256, 512, 1024} = 288 rows | < 1 min |
| E3.2 | Item 3, simulated | `packed` + evict, 24 h (agentic 48 h): 2 models × 4 workloads × 3 X = 24 runs | ≈ 40 min (≈ 15 min with 3 jobs) |
| E3.3 | Block size | ppb {192, 256, 512, 1024} × {long32k, agentic} = 8 runs | ≈ 12 min |
| E3.4 | Retention | agentic, TTL {1 h, 24 h, 7 d}, 72 h, retention on/off = 6 runs | ≈ 15 min |
| E4.1 | Compression | {bf16, fp8, int4, kivi4, kivi2, kivi2:think=0.5} × {fixed rate, closed B=32} × {long32k, agentic} at X = 1e6 = 24 runs | ≈ 35 min |
| E4.2 | Sparsity | {dense, uniform_tokens 5%, quest} × {bf16, kivi2} × hot {1, 4}, long32k = 12 runs | ≈ 15 min |
| E5.1 | Legality | M2 snapshots, `--check-writes`: policies × generated {0, 338, 2048} ≈ 20 snapshots | ≈ 3 min |
| E5.2 | Read parallelism per policy | M2 + phase A: B {8, 32} × ctx {4K, 32K} × 6 policies × flush order {layer, rotate} ≈ 60 runs at 1–11 s | ≈ 6 min |
| E5.3 | Reclaim granularity | {dedicated-superblock, packed, channel, die} × {chat, long32k, agentic}, 24 h = 12 runs | ≈ 25 min |
| E5.4 | vLLM free list | violations and stranded capacity, 3 workloads × 24 h | ≈ 5 min |
| E5.5 | Staging and pending units | analytic table | instant |
| E5.6 | M5 | loads from E3.2/E4.1 × tPROG {75, 100, 200, 350} × tBERS {1, 3.5, 10} × {fifo, gaps, suspend 5/20 µs, random}, plus refresh-storm spikes (spread vs at once) | instant |
| E5.7 | Hot spots | shared prefix, sharers {1, 8, 64} × {per_request, once} × X: analytic + 6 runs | ≈ 10 min |
| E5.8 (optional) | MQSim M5 confirmation | 5 runs at ≤ 64 planes (§5.5) | < 5 min |

Total: about 2.5 h sequential, about 1 h with 3 jobs.

---

## 10. Implementation order

1. `hbf/batch.py` + `generate(requests=)` + `PageTableLayout` + `check_write_order` + CLI flags; bit-identity tests.
2. `Clock` + `calibrate` with the phase A cross-check.
3. `analytic`.
4. `hbf/workload.py` + `simulate`.
5. `hbf/interference.py`, wired into `simulate`.
6. Output files and report, then the sweep `tool` key.
7. Experiments.
8. Optional MQSim confirmation, run by the main agent.

---

## 11. Open questions for the main agent

1. **§5.4.1 item 4.** Read "no skipping within a host channel's allocated global address range" per NAND block (items 9 and §11.5.2.6), as designed? A literal per-channel reading would allow only one log per channel and forbid `dedicated`.
2. **Erase counting.** Count an auto-erase (P/E +1) on every page-0 write, including never-written blocks? Assume the read counter resets at erase?
3. **Defaults.** Proposed: allocation `packed` + evict, disaggregated prefill, `--prefix-reads per_request` for M4, wear `dynamic`, M5 `gaps`, die-level blocking. Phase A's dedup models "once per step"; say so in the report?
4. **Clock.** Accept the surrogate (fitted est + `system.evaluate`) rather than a precomputed B×ctx TPOT table? A table can be an optional cross-check (`--tpot-table`).
5. **`pipeline(requests=...)`.** May it become a `BatchSpec` (the tuple is its `layer_major` case)? Should `generate()` take `requests=`? Both keep defaults bit-identical.
6. **Block sizes.** Add ppb 512 and 1024 (spec §11.5.1.8 hint) to params.md's sweep, and decouple `--capacity-gib` from `--pages-per-block` for M4?
7. **Extra policies.** Add `hbm_decode` (generated KV never on HBF) as a baseline? Add the weights-on-HBF read-disturb note to the report even though weights stay in HBM?
8. **Budget.** About 1 h with 3 jobs for the matrix; the MQSim M5 confirmation (5 small runs) is for the main agent to run.

### Critical files for implementation

- /home/user/MQSim-HBF/hbf/pages.py
- /home/user/MQSim-HBF/hbf/access.py
- /home/user/MQSim-HBF/hbf/cli.py
- /home/user/MQSim-HBF/hbf/system.py
- /home/user/MQSim-HBF/hbf/sweep.py
- /home/user/MQSim-HBF/hbf/analysis.py
- /home/user/MQSim-HBF/hbf/timing.py
- /home/user/MQSim-HBF/hbf/simulator.py (only for the optional M5 confirmation)
