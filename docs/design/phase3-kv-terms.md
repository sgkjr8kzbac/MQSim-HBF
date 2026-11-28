# Phase 3 design: compressing each term of the KV size on HBF

> Design notes from the planning subagent (read-only review of the code at commit 7c75e05). The main agent's answers and the work split are in [phase3-decisions.md](phase3-decisions.md); the plan and status are in [PLAN.md](../../PLAN.md).

KV size = 2 (K and V) × layers × kv_heads × head_dim × tokens × bytes/element × batch. The simulator already handles plain byte ratios (fp8, int, KIVI, ThinK, static head pruning). This phase adds methods whose effect on HBF is not a byte ratio.

## 0. Recommendation

Organise the work by the KV-size terms, which also gives the report a clear narrative. Rank the methods by HBF mechanism, not by bytes saved. Nearly every method that is not a plain byte ratio acts through one of three levers:

- **Page granularity**: rows per page, selection unit, fused pages.
- **Address aliasing and placement**: per-layer and per-head regions, shared regions, separate write frontiers.
- **Token flow on a write-once device**: tokens that die or get rewritten leave dead pages, and HBF has no GC (§11.4).

Two small abstractions cover most of it without touching the analysis engine:

- **K1**: page-format fields on `TensorSpec`, plus MLA, DSA and fused presets.
- **K2**: a layer/head-group KV spec, compiled to a grouped layout.

Build both first, in parallel (file-disjoint). Write-side work (K4, K7) waits for T-B1d to merge and plugs in through a Geometry duck type.

## 1. Methods per term

"Now" is current support: Y / P(artial) / N, with file:function.

### K/V pair (the factor 2)

- **MLA latent (DeepSeek-V2/V3, Kimi K2): one latent of 512 + 64 dims per token and layer.**
  - Bytes: 1,152 B per token and layer (Llama-8B GQA: 4,096 B).
  - HBF read: one "head" per layer. A 1,152 B row fits 3 per page, so 15.6% of every page is wasted. Splitting into C (512) and R (64) packs pages exactly but costs 2 pages per sparse token. FP8 (656 B) fits 6 per page.
  - Write: in proportion to bytes.
  - Now: P. Bytes are expressible with `--tensors` and `--heads 1`, but `pages.layer_pages` counts `dims` as attended elements (QK + PV over one latent needs 1,088 elements, not 576), and `model.ModelSpec.pre_params` is GQA only.
  - Missing: `attn_dims`, `mla*` presets, an MLA ModelSpec.
- **K or low-rank K in HBM, V on HBF (ShadowKV, InfiniGen).**
  - HBF holds V only. Sparse V chunks of 8 tokens are half a BF16 page: 2× read amplification.
  - Now: Y (`TensorSpec.tier`; `access.generate` reads HBM-tier tensors whole in selection modes). Missing: landmarks and outlier chunks.
- **K-only cache (Slim Attention, MHA only).** Halves bytes and pages, adds d² FLOPs per token. Now: N. Low priority.

### layers

- **Hybrid local/global (Gemma 3 5:1 with W = 1,024; GPT-OSS 1:1 with W = 128; Llama 4 3:1 with 8K chunks).**
  - Read: per-layer rounds differ (e.g. 1 against 32); serial waits only on global layers; `prefetch:d` can hide a global read behind d local layers, because a dense global read is a static selection.
  - Write: local KV on HBF is written once and dies after W tokens (write traffic without capacity; dead pages strand blocks). Local KV in HBM: no HBF writes.
  - Now: N. `TensorSpec.window` applies to every layer (`Layout.__init__`, the layer loop in `access.generate`). Missing: per-layer groups (K2).
- **SSM or linear-attention hybrids (Qwen3-Next 3:1, Jamba 1:7, MiniMax 7:1).**
  - KV exists on 1/4 to 1/8 of the layers; a fixed state lives in HBM (it changes every step). HBF idles on SSM layers, which gives prefetch slack.
  - Now: N. `system._steps` requires a row for every layer and the generator emits none. Missing: groups, state bytes and zero rows (K2).
- **Cross-layer sharing (CLA-2, YOCO, MiniCache, LCKV).**
  - Bytes and writes 1/s (YOCO ≈ 2/L). Consumer layers re-read the owner's pages: bytes read do **not** drop unless the copy is staged in HBM. Block reads per step grow by s, so the read-disturb threshold in steps becomes X/(ppb·s).
  - Now: N (every layer has its own region: `LayerContiguous.lpa`, `StripedContiguous` starts). Missing: aliasing with `reuse=hbf|hbm` (K2).
- **Per-layer budgets (PyramidKV).** Unequal rounds per layer; fewer prefill writes. Now: P (one budget via a smaller `--context`). Missing: per-layer budgets (K2).

### kv_heads

- **DuoAttention, RazorAttention (retrieval heads keep all, streaming heads keep sink + recent).** Bytes r + (1−r)·W/ctx. Streaming windows slide every step: written and then dead on HBF, or kept in HBM. Now: N (`Layout` has uniform heads per layer). Missing: head groups with their own window and tier (K2).
- **HeadKV, Ada-KV (per-head budgets).** Per-(layer, head) regions of unequal size; under GQA a KV head's budget is the union over its query heads. Now: N. Missing: head groups (K2), GQA union (K5).
- **Static head pruning.** A byte ratio. Now: Y (`--heads`).

### head_dim

- **ThinK.** Now: Y (`formats.parse_format`, `think`).
- **Low-rank K (Palu, ShadowKV).** r/d of K plus reconstruction FLOPs. Now: P (a latent tensor through `--tensors`; FLOPs only through `dequant_ops`). Missing: rank presets.
- **Dynamic channel selection (SparQ top-r by |q|; Loki top-r PCA dims).**
  - Token-major pages hold all channels, so scoring reads every page. With g channel-group tensors, SparQ's random channels touch g·(1 − C(d−d/g, r)/C(d, r)) groups: 7.94 of 8 at r = 32, d = 128, no saving. Loki's static prefix of PCA dims reads exactly r/d. The exact phase needs a token-major copy; a dual layout writes K twice (+50% bytes).
  - Now: N (README "读一个张量的部分通道"; `access.generate` reads all non-meta tensors at the same tokens). Missing: channel-group presets, whole-context tensors, a `channel_select` mode (K6).

### tokens

- **StreamingLLM.** Now: Y (`sink_recent`).
- **Quest, InfLLM block retrieval.** Block size against tokens per page (16-token blocks on 2-bit pages ran at 0.62×, REPORT item 2); metadata read whole every step; hot blocks become read-disturb hot spots. Now: P (`quest`, `uniform_tokens`; not modelled: GQA union, reuse between steps, per-block metadata). Missing: union and reuse (K5), fused pages (K1).
- **DSA (DeepSeek-V3.2: lightning indexer + top-2,048 tokens).** With MLA rows (3 tokens per page) each selected token costs about one page: ≈ 2,016 pages per layer at 128K. The indexer cache (132 B per token, read whole every step) is 4,229 pages, about **2/3 of all pages read**. Now: P (`uniform_tokens --select-tokens 1` plus a meta tensor; the indexer's FLOPs are dropped because meta tensors are not attended in `pages.layer_pages`). Missing: `attn_dims` on meta tensors, a preset (K1).
- **Prefill-time eviction (SnapKV, PyramidKV).** A dense read of the compacted kept set; fewer prefill writes. Now: P (uniform budget via `--context`). Missing: groups (K2), density (K5).
- **Decode-time eviction (H2O, TOVA).** Eviction leaves holes; a page dies only when all its tokens are gone. Keeping 25% of tokens at random leaves 1 − 0.75^16 = **99.0% of BF16 pages live** (≈ 100% for 2-bit): no read or space saving without compaction; compaction is write amplification. Now: read side P (`uniform_tokens --select-tokens 1`, but redrawn every step); write side N (`endurance.Geometry.stored_pages` has no dead pages). Missing: static selection (K5), flow model (K4).
- **Evict at flush (HBF-aware variant: decide in HBM staging before writing).** No holes, no dead pages. Now: N. Missing: density (K5), flow (K4).
- **Token merging (CaM, D2O).** Merging before the write behaves like density; after the write it is a rewrite plus holes. Now: N. Missing: K4.

### bits

- **fp8, int, KIVI with HBM residual.** Now: Y.
- **Quantisation params in the same page as data (T-3a).** A kivi2 16-token block costs 4 pages with separate tensors; fused, 1 page at 32 tokens per page (25% slack) or 1.33 pages at 42 tokens per page. Now: N. Missing: fused presets (K1).
- **KVQuant dense + sparse outliers.** Variable-length rows: a separate page set or slack in each page. Now: P (an amortised aux tensor via `--tensors`).
- **Per-layer bits (KVTuner).** Now: N. Missing: groups (K2).
- **Token-level mixed precision (MiKV, ZipCache).** Two compacted tensors holding complementary tokens; a sparse read touches both. Now: N. Missing: density (K5).
- **Age-based re-quantisation (bf16 in HBM → int4 on HBF → int2 on HBF).** Read side is just windows. Each decode token is written twice: 160 + 96 = 256 B against 96 B (2.67× decode writes), and the int4 pages die. Now: read P (two windowed tensors via `--tensors`); write N (differences of `endurance.Geometry.stored_pages` give 0 writes for windows whose lower end is negative; `batch.PageTableLayout` rejects HBF windows that do not start at 0). Missing: flow (K4), placement (K7).
- **VQ (CQ, CommVQ).** A byte ratio. Now: Y (as bits; the codebook is not counted).

### batch

- **Prefix sharing (RadixAttention) with coalesced reads (Hydragen, cascade).** Coalesced: each shared page is read once per step. Per request (PagedAttention): B× reads and B× read disturb. The prefix is written once and lives long (retention refresh). Now: P (endurance only: `endurance.analytic(prefix_reads=)`, `workload.Workload.prefix_*`; read path N because `batch.place_batch` builds a table per request; `pages.layer_pages` already deduplicates LPAs per (step, layer), so aliased pages would be read coalesced). Missing: prefix aliasing (K7).
- **CacheBlend (reuse with about 15% of tokens recomputed).** Patched tokens are scattered, so every shared page is still read; writes −85%. Now: N. Missing: a density patch tensor (K5).

## 2. Abstractions and where they plug in

**A1. Page-format fields on `TensorSpec`** (`hbf/access.py`)
- `attn_dims: int|None`: elements one query head multiplies per stored token. Unset means `dims` for data tensors and 0 for aux and meta tensors.
- `logical_dims: int|None`: BF16-equivalent elements stored; feeds `hbf_elements` (for `effective_gb_s`) and `dequant_ops`.
- `density` (a Fraction, default 1): share of the window's tokens stored, compacted in token order (K5's field).
- Plug points: `pages.layer_pages` (attended, hbf and dequant sums; with K5 the compacted block mapping `first = floor((lo − origin)·ρ) // tpp`); `formats.parse_format` (new presets, optional `q_heads=` keyword for the indexer's `attn_dims`); `endurance.Geometry.demand` (one line, after T-B1d merges).
- `controller.schedule` raises for density ≠ 1: numpy engine only, as `--timing` already is.

**A2. KV spec with groups** (new `hbf/kvspec.py`)
- `KVGroup(name, layers, heads, tensors|format, window=None, tier=None, owner=None, reuse="hbf"|"hbm", budget=None, state_bytes=0)`. `KVSpec(groups)` compiles to flat tensors named `{group}_{tensor}`, presence per (layer, tensor), virtual head ranges per layer (groups covering the same layer split its heads), and an alias map from consumer layer to owner layer.
- `GroupedLayout(Layout)`: a start table [L, T, Hmax] with −1 for absent entries. Consumer layers point at the owner's starts (`reuse=hbf`) or get HBM-tier alias tensors (`reuse=hbm`). Intra order `contiguous` or `striped` with a per-layer gap `units // (T_l·H_l)` (the phase-A gap collides on banks when regions are 64 pages long, i.e. a 1,024-token window).
- `generate_grouped(cfg, layout, mode, …)` follows `batch.generate_batch` (modes dense, uniform_tokens, sink_recent), so `access.py` stays untouched in K2.
- SSM state: an HBM aux pseudo-tensor (`bits=8`, `dims=state_bytes`, window [0, 1)), which gives each SSM layer a row through the existing zero-row path.
- `KVSpec.request_bytes()` replaces `cli.request_bytes` when a spec is given; `KVSpec.geometry()` returns a duck type of `endurance.Geometry` (A3).
- CLI: `--kv-spec PRESET[:k=v]|JSON|FILE`, presets `hybrid`, `ssm`, `cla`, `yoco`, `duo`, `pyramid`. `--format` sets the default format for every group; `kvspec.from_model(model, fmt)` when the model declares `layer_pattern`; `cli.pipeline` pads zero rows for layers without reads (needed with `--timing`).

**A3. Token flow** (in `kvspec.py`)
- `flow(spec, prompt, generated, evict)` returns tokens written at admission, written during decode, died and rewritten. For a window, decode writes are `hi(P+n) − hi(P)` and deaths `lo(P+n) − lo(P)`. Eviction policies: `none | prefill | flush | decode(keep, cluster c, compact θ)`. Live-page fraction under eviction: 1 − (1 − s)^(tpp/c).
- `KVGeometry` adds `written_pages(c0, c1)`, `dead_pages(c0, c1)` and `layer_row_pages(ctx)` (per layer).
- Endurance patch (≤ 30 lines, after T-B1d merges, by the simulation owner): `analytic` uses `written_pages` when present, else the stored-pages difference (existing results stay bit-identical); sim entries receive `died`; `Clock` uses per-layer rows when `layer_row_pages` exists; the endurance CLI gains `--kv-spec`.

**A4. Selection realism** (`access.generate` options, all off by default)
- `select_static`: the same selection every step (eviction state).
- `reuse=p`: blocks selected again from the previous step are served from an HBM cache (HBM reads).
- `query_groups=g`: GQA union, effective fraction 1 − (1 − f)^g.
- `full_tensors=NAMES`: tensors read over the whole context (Loki's K_lo).
- Mode `channel_select`.

**A5. Model presets** (`hbf/model.py`, `hbf/configs/models.json`, `hbf/system.py`)
- New `ModelSpec` fields, all defaulted: `attention` (gqa|mla), `kv_lora_rank`, `rope_dim`, `q_lora_rank`, `v_head_dim`, `layer_pattern` (e.g. `LLLLLG`, `SSSA`), `window`, and MoE fields (`experts`, `top_k`, `shared_experts`, `expert_intermediate`, `dense_layers`).
- `system.linear_ns` split into FLOP-params (active experts) and byte-params (experts touched, E(1 − (1 − k/E)^B)).
- `evaluate` uses per-layer pre/post lists; `timeline` already accepts lists.

**A6. Placement** (`hbf/batch.py`)
- `PageTableLayout` and `prompt_keys` aware of `GroupedLayout`; sliding HBF windows get flush and death events.
- `--frontiers single|class`: short-lived tensors (sliding windows, int4 stages, streaming heads) get their own frontiers and segments.
- Shared prefix: `--prefix-tokens`, `--prefix-groups`, `--prefix-reads coalesced|per_request`.
- The write check reports `stranded_pages`.

**Must stay bit-identical**: the 6 cases in `tests/golden/`; `tests/test_engine.py` parity; `tests/test_system.py` and the endurance calibration within 0.01%; 2–3 re-run points from the phase 2A grids, whose summary rows must diff exactly. Rules: new fields default to "unchanged"; `tensors_meta` drops new fields at their default so `meta.json` stays stable; code branches on presence (no `--kv-spec`, density 1, options off) and never changes the RNG draw order on the default path; new modes are appended to `MODES`.

## 3. Milestones

Tracks: **A** = `access.py`, `formats.py`, `pages.py`; **B** = `kvspec.py`, `cli.py`, `batch.py`; **C** = `model.py`, `system.py`, `models.json`.

| ID | Goal | Files | Depends on | Items |
|---|---|---|---|---|
| K1 | Page formats: `attn_dims`/`logical_dims`; presets `mla[:split,indexer]`, `mla_fp8`, `fused=1[,pow2=1]` for int/kivi | access.py (TensorSpec), formats.py, pages.py, tests/test_formats.py | — | 2, 5 |
| K2 | KV groups + GroupedLayout + `--kv-spec` + zero rows | hbf/kvspec.py (new), cli.py, tests/test_kvspec.py | — | 1, 2, 5 |
| K3 | MLA, hybrid and MoE model presets | model.py, system.py, configs/models.json, tests/test_system.py | parameter research; K2 interface `from_model` | 1, 2 |
| K5 | Density, `select_static`, `reuse`, `query_groups` | access.py (generate), pages.py, controller.py (reject) | K1 | 2, 5 |
| K4 | Token flow + `KVGeometry` + endurance patch | kvspec.py; endurance.py patch only after T-B1d | K2, T-B1d | 3, 4, 5 |
| K6 | Partial-dimension reads: `loki`, `sparq[:dual]`, `full_tensors`, `channel_select` | formats.py, access.py | K5 | 2, 5 |
| K7 | Grouped placement, class frontiers, prefix aliasing, stranded pages | batch.py, tests/test_batch.py | K2, K4 | 3, 5 |

Order: K1 ∥ K2, then K3 ∥ K5, then K4 → K7 (after T-B1d merges), then K6.

### Acceptance tests (`ocp`: 1,024 banks, 4 KiB pages; BF16 unless noted)

**K1**
- `mla`: C = 16 bits × 576 dims, `attn_dims` 1,088, 3 tokens per page. At ctx 32,768 with 1 head: 10,923 pages, `min_rounds` 11.
- `mla:split=1`: C (4 tokens per page) + R (32 tokens per page) = 9,216 pages (−15.6%).
- `mla_fp8` (656 B): 6 tokens per page, 5,462 pages.
- `mla:indexer=1` adds a meta tensor of 132 B (31 tokens per page). At ctx 131,072 the indexer is 4,229 pages; `uniform_tokens --select-tokens 1 --fraction 0.015625` gives data pages ≈ 43,691·(1 − (63/64)³) = 2,016 (±3% over seeds); indexer FLOPs use `attn_dims` = index_heads·128/q_heads.
- Fused kivi2: `kivi2:residual=0` at ctx 4,096 per head is 96 pages separate; `fused=1` makes a 96 B row (42 tokens per page) and 98 pages; `pow2=1` gives 128 pages. Isolated 16-token blocks: 4.0 pages per block separate, 1.0 for pow2, (21 + 7)/21 = 1.333 at 42 tokens per page.
- System: `kv_heads` 1, `q_heads` 16, 3 tokens → attention FLOPs per layer 2·16·3·1,088 = 104,448.
- Regression: golden files and engine parity hold; `attended_elements` unchanged for every existing preset.

**K2**
- Hybrid: shape (6 layers, 8 heads, d 128, 32,768 tokens), `hybrid:pattern=LLLLLG,window=1024,local_tier=hbf`, contiguous, dense: pages per layer 1,024 × 5 and 32,768; rounds 1, 1, 1, 1, 1, 32; 37,888 pages per step. With `local_tier=hbm`: 32,768 HBF pages; `hbm_bytes` 4,194,304 on each local layer; 6 system rows. `kv_bytes_per_request` = 134,217,728 + 20,971,520.
- CLA: `cla:share=2`, 4 layers, 8 heads, ctx 4,096. `reuse=hbf`: 4,096 pages on every layer (16,384 per step), stored 8,192, consumer LPAs equal the owner's, `max_block_reads` = 2× an unshared run with the same stored bytes. `reuse=hbm`: 8,192 HBF pages per step; consumer `hbm_bytes` 16,777,216.
- Duo: `duo:retrieval=0.5,sink=64,recent=256` at 32K: 16,384 HBF pages per layer; `hbm_bytes` 655,360; `attended_tokens` 132,352.
- Pyramid: budgets 8,192, 6,144, 4,096, 2,048 → exactly that many pages per layer.
- SSM: `ssm:pattern=SSSA` gives 4 rows; S rows have 0 pages and `hbm_bytes` = state bytes.
- Without `--kv-spec`, everything is bit-identical.

**K3**
- Parameter totals within 1% of published values: DeepSeek-V2-Lite 15.7 B total / 2.4 B active; Gemma-3-27B; GPT-OSS-120B 116.8 B / 5.1 B; Qwen3-Next-80B-A3B.
- KV bytes per token from `from_model`: DeepSeek-V2-Lite 27·1,152 = 31,104 B; Qwen3-Next 12·2·2·256·2 = 24,576 B; GPT-OSS-120B (global layers only) 18·8·2·64·2 = 36,864 B.
- Experts touched with E = 64, k = 6: 6 at B = 1, toward 64 as B grows.
- TPOT of the existing Llama and Qwen presets unchanged.

**K5**
- Density 1/4 at ctx 32,768: 512 pages per (head, tensor) under a dense read.
- `select_static`: the same page set in steps 0 and 1.
- `reuse=0.8`: step-1 HBF pages ≈ 20% of step 0.
- `query_groups=4`, f = 0.05: selected share 1 − 0.95⁴ = 0.1855.
- Keep 25% with `select_tokens=1`: pages ≈ 0.990 of dense for BF16, ≈ 1.0 for kivi2; aligned clusters of 16: 0.25 for BF16.

**K4**
- Window (−4,096, −128), prompt 32,768: admission writes 3,968 tokens; after +1,000 generated, decode writes 1,000 and 1,000 die; stored stays 3,968.
- kivi4 → kivi2 ladder: 256 B against 96 B per decode token per head (2.67×).
- H2O live-page fraction 1 − 0.75^16 = 0.98998.
- Analytic results for existing presets unchanged.

**K6**
- `loki:rank=32` at 32K: K_lo is 512 pages per head (K is 2,048).
- `sparq:channels=32,groups=8`: mean groups touched 7.94 ± 0.1; `groups=128`: 32 × 16 = 512 scoring pages per head.
- `dual=1` raises HBF bytes per token by 1.5× for K.

**K7**
- Hybrid with local layers on HBF, 1 request, prompt 32,768 + 4,096 generated: dead local pages 5·8·2·4,096/16 = 20,480. `frontiers=single`: all 20,480 stranded; `frontiers=class`: at most one open short-lived segment stranded (small ppb).
- 4 requests sharing an 8,192-token prefix (1 layer, 8 heads): 8,192 prefix pages written once instead of 32,768, and read once per step when coalesced.

## 4. Experiments (analysis-only sweeps; grids `evidence/grids/phase3-*.json`)

| ID | Varies | Outputs | Items | Needs |
|---|---|---|---|---|
| X1 Architectures | Llama-8B, DS-V2-Lite (bf16/fp8), Gemma-3-27B (local in HBM vs HBF), GPT-OSS-20B, Qwen3-Next, Llama-8B + CLA2 (hbf/hbm) × ctx {8K, 32K, 128K} × B {1, 8, 32} × overlap {serial, prefetch:1, 5} | HBF read ms, TPOT, HBF/HBM, wait share, `hbm_fits` | 1, 2 | K2, K3 |
| X2 Granularity | MLA one/split/fp8 × DSA top-k {512, 2,048} × indexer {hbf, hbm} × ctx {32K, 128K}; kivi2 separate vs fused (42/32) × block {16, 32, 128} × fraction {1%, 5%} | pages, rounds, read amplification, speedup vs BF16 | 2, 5 | K1 |
| X3 Heads/layers | Duo r {0.25, 0.5, 0.75}; Pyramid slopes; CLA s {2, 4} × reuse | HBF ms, `hbm_fits_hbf`, `max_block_reads` | 2, 5 | K2 |
| X4 Partial dims | Loki r {16, 32, 64}; SparQ r × g {8, 16, 128} × dual | scoring pages, total pages, bytes written | 2, 5 | K6 |
| X5 Eviction | keep {10%, 25%} × cluster {1, 4, 16} × {bf16, kivi2}; holes vs density (evict at flush) | live pages read, reads saved | 2, 5 | K5 |
| X6 Endurance | SWA on HBF vs HBM; kivi ladder; H2O compaction θ {0, 0.25, 0.5}; prefix coalesced vs per request × workloads | P/E per day, pe_per_mtok, refresh_to_kv, stranded share | 3, 4, 5 | K4, K7, T-B1d |

One optional MQSim confirmation: a hybrid point whose local layers on HBF take about 1 round each (the estimate's overhead was highest for small layers, 1.20× at 8K, B=1).

## 5. Other directions

1. **HBF-aware page formats** (this is K1; do it first). Cheap, answers T-3a, and is expected to reverse item 2's negative result: fused kivi2 should read 0.5–0.67× BF16's pages for 16-token blocks instead of 2×.
2. **Weights on HBF (§13)** as a cheap analytic side study. Dense weights are read in full every step, so the read-disturb threshold is X/ppb steps: 3,906 at X = 1e6, ppb 256, i.e. a refresh every 78 s at 20 ms per step. For 70 GB of weights on a 512 GiB stack that is about 140 device-average P/E per day at X = 1e6 and about 14 at X = 1e7. MoE experts are read less often. A strong "viable only if X ≥ …" curve.
3. **Temporal reuse and GQA union** for retrieval methods (folded into K5); likely moves item 2's sparse numbers by several times.
4. **Real traces** (T-C1; T-C2 adapters now) to replace the random-vs-clustered assumption, which swings live pages between 0.99 and 0.25: kept-token run lengths, overlap between steps, GQA union. When a GPU is available.
5. **Multi-stack interleaving.** Medium priority: MLA and hybrid models shrink KV per request, so B per stack grows; only long single requests (128K+, agentic) leave stacks idle.

## 6. Risks and open questions

1. Row packing: rows may not span pages (the TensorSpec rule), which wastes 15.6% of each page for MLA. Should a row-spanning option exist (a straddling token costs 2 page reads)? Can GPU kernels handle 3, 6 or 42 tokens per page?
2. MLA compute: absorbed decode costs 2·q_heads·1,088 FLOPs per token per layer, about 17× Llama-8B; this shifts the HBF/HBM balance, and `compute_eff` 0.5 is untested for MLA kernels.
3. MoE serving model: large MoE models do not fit one GPU. Proposed DP-attention framing: each GPU owns its own requests' KV; experts are sharded.
4. Density approximation: deterministic compaction stands in for real token sets; X5 must be reported as a range until real traces exist.
5. CLA under dynamic selection: consumers may select different tokens; only dense or static selection is modelled. `reuse=hbm` needs a full-layer HBM copy per request, which must enter `hbm_fits_hbf`.
6. Endurance interface: `written_pages`, `dead_pages`, `layer_row_pages` must be agreed with the T-B1d owner before K4. Today `Clock.step` assumes identical layers.
7. Latent write undercount: `analytic`'s stored-pages difference gives zero writes for HBF tensors whose window starts at a negative offset. No current preset triggers it; K4 must fix it before any sliding HBF window is used.
8. Duplicated generator: `generate_grouped` duplicates `access.generate` logic; unify after K5.
9. Prefix read default: coalesced reads are the current default (implicit in the dedup of `pages.layer_pages`); per-request reads need virtual copies on the same bank or a multiplicity column in `LayerPages`.
10. Model parameters (Gemma 3: 62 layers, 16 KV heads, W = 1,024; GPT-OSS; Qwen3-Next; DeepSeek-V3.2 indexer 132 B; Llama 4) need verification against published configs; decide the default tier for SWA local KV (HBM recommended).
11. `effective_gb_s` for MLA and fused rows: `logical_dims` as proposed, or drop the metric for these formats.
