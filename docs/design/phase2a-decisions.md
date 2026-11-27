# Phase A decisions (main agent, after designer-a's proposal and param-researcher's params.md)

> Working notes of the design discussion (English). The plan and status are in [PLAN.md](../../PLAN.md); parameter sources are in [docs/research/params.md](../research/params.md).

Order: golden tests -> M7 numpy engine -> M0 params + `--timing` -> M3 windows/tiers + `--format` presets -> M1 model/system -> sweep driver -> experiments -> MQSim confirmations.

## Answers to the open questions

1. numpy becomes a project dependency (`dependencies = ["numpy"]`); the existing pure-Python `analyze()` stays as the reference and as `--engine python`.
2. `meta.json` `tensors` may gain `window`/`tier` keys (`tensors_meta` stays `asdict`).
3. Phase A `--batch B` = storage `kv_heads` x B, layer-major (`kv_head = r*H + h`), documented as an emulation; phase B M2 must reproduce it bit-identically as policy `layer_major`.
4. `est` follows `cfg`. The report states the command model in use and warns when a multi-plane topology runs without `--read-plane-ns 0 --coalesce-ns 1 --cache-read` (est then models ONFI commands). The sweep driver's base sets the HBF flags.
5. Report both `est` and `bound`; `est` is primary.
6. Weights in HBM only (`--weights hbm`); `hbf` reserved, raises "not implemented" (spec section 13 puts weights on HBF; phase B/C).
7. `--stacks N` (default 1): requests partitioned across N identical stacks; analysis batch ceil(B/N), compute batch B.
8. Presets from params.md section 3/4/8: GPUs H100 SXM (989 TF BF16 dense, 1,979 FP8, 80 GB, 3.35 TB/s), H200 (989/1,979, 141 GB, 4.8 TB/s), B200 (2,250/4,500, 180 GB, 8 TB/s). Efficiencies are assumptions: compute_eff 0.5, memory_eff 0.8 (params.md nominal), overridable. Models: Llama-3.1-8B, Llama-3.1-70B, Qwen2.5-7B, Qwen3-8B, Qwen3-32B (dims in params.md section 4; every entry has a `source`). Dequant ops per element: assumptions (fp8 1, int/kivi 3), labelled.
9. KIVI residual: fixed R snapshot (documented), R should be a multiple of the body tensor's tokens per page.
10. Quest summary preset: yes (`always` tensor).
11. Prefetch with query-dependent selection: hard error unless `--selection static` is given explicitly.
12. Latent preset: deferred (storage can be expressed with `--tensors`; MLA compute differs).
13. Opt-in per-block read counts: yes, with `--timing` add `max_block_reads` per layer/step (block = (unit, (lpa // units) // pages_per_block)).
14. Yes: commit `evidence/time-calibration.csv` from the round-8 runs and re-run about 10 points on the current binary (sequentially).

## Work split

- Main agent: golden tests, M7 (`hbf/pages.py`, analysis integration), M0 (`hbf/timing.py`, CLI), M3 (`hbf/formats.py`, TensorSpec window/tier, schedule/generators), CLI integration, sweep driver.
- Implementer subagent: M1 only, in new files `hbf/model.py`, `hbf/system.py`, `hbf/configs/models.json`, `hbf/configs/gpus.json`, `tests/test_system.py`. It must not edit any other file. It does not commit.

## M1 input contract (what system.evaluate receives)

`layer_rows`: list of dicts, one per (step, layer) of the HBF analysis of ONE stack's share of the batch (group 1 rows only), with keys:
- `step`, `layer` (ints)
- `est_ns` (int or None), `bound_ns` (int): HBF read time of the layer's KV on one stack
- `burst_bytes` (int): HBF bytes at 64 B granularity (HBM-baseline KV bytes for the HBF tier)
- `hbm_bytes` (int): bytes read from HBM-tier tensors (e.g. KIVI residual) for that layer
- `attended_tokens` (int): sum over (request, KV head) of tokens attended (all tiers, data tensors only)
- `dequant_ops` (float): GPU vector ops for dequantization in that layer
These rows cover analysis batch b = ceil(B / stacks); system.evaluate scales token/byte demand to the full compute batch B (x B/b) for compute and HBM-baseline terms, while HBF time stays per stack (stacks run in parallel).

`meta`: dict with `model` (ModelSpec), `gpu` (GpuSpec), `batch` B, `stacks`, `selection` ("static"|"dynamic"), `kv_format` name.

Outputs: `system_layers` rows and `system_steps` rows (see designer's design below) and a list of assumption strings for the report.

## M1 design (from designer-a, adopted)

ModelSpec: name, layers, hidden, q_heads, kv_heads, head_dim, intermediate, vocab, weight_bits=16, gated_mlp=True, tied_embeddings=False, source.
- pre_params = hidden*(q_heads+2*kv_heads)*head_dim ; post_params = q_heads*head_dim*hidden + (3 if gated else 2)*hidden*intermediate.
- Llama-3.1-8B: pre 25,165,824, post 192,937,984, layer 218,103,808.
GpuSpec: name, tensor_flops (BF16 dense, FLOP/s), vector_ops (FLOP/s for dequant; assumption, e.g. FP32 non-tensor rate), hbm_bytes_s, hbm_capacity, compute_eff, memory_eff, layer_overhead_ns=0, source.

Per layer l with compute batch B, F = tensor_flops*compute_eff, BW = hbm_bytes_s*memory_eff, wb = weight_bits/8:
- pre_l = max(2*B*P_pre / F, P_pre*wb / BW); post_l likewise with P_post.
- attended scaled to B: A_l = attended_tokens * B/b; attention flops = 4*(q_heads/kv_heads)*head_dim*A_l (QK^T and PV; A_l already sums over KV heads) ; a_l = max(flops/F + dequant_ops*B/b / vector_ops, hbm_bytes*B/b / BW).
- h_l = HBF time (est_ns if not None else bound_ns; both TPOTs reported).
Overlap modes:
- serial: T = sum_l (pre_l + max(h_l, a_l) + post_l)  (kv_consume=stream: attention consumes pages as they arrive; `staged` gives h_l + a_l)
- prefetch:d: r_start(l) = max(r_end(l-1), c_end(l-1-d)); r_end(l) = r_start(l) + h_l; c_end(l) = max(c_end(l-1) + pre_l, r_end(l)) + a_l + post_l; rejected for selection="dynamic".
- ideal: T = max(sum h_l, sum (pre_l + a_l + post_l)).
TPOT = T + lm_head (lm_head = max(2*B*hidden*vocab/F, vocab*hidden*wb/BW)) + layers*layer_overhead_ns; tokens/s = B / TPOT.
HBM baseline: same compute, KV from HBM: kv_l = (burst_bytes*B/b + hbm_bytes*B/b)/BW, layer = pre + max(kv, attention compute) + post (serial semantics), plus lm_head.
hbm_fits = weights_bytes + B*kv_bytes_per_request_total <= hbm_capacity (kv bytes per request = full stored KV of one request, given in meta as `kv_bytes_per_request`).
system_steps columns: step, tpot_hbf_est_ns, tpot_hbf_bound_ns, tpot_hbm_ns, tokens_per_s_hbf, tokens_per_s_hbm, hbf_over_hbm, gpu_ns, hbf_wait_share (share of T during which compute waits on HBF), hbm_fits.
CLI for re-evaluation: `python -m hbf.system <run_dir> --model ... --gpu ... --batch ... --stacks ... --overlap serial|prefetch:1|ideal` reading `analysis_layers.csv` and `meta.json` (main agent wires the main CLI).
Tests: hand-computed values above; 3-layer recurrence h=[3,1,2], g=[2,2,2] for serial, prefetch:1, ideal; prefetch rejected for dynamic; hbm_fits boundary; presets parse with sources.

## Amendments after the design discussion (these override the text above)

1. `tpot_hbf_est_ns` is None when any `est_ns` of the step is None; `tpot_hbf_bound_ns` is always reported. No silent fallback.
2. `h_l` is `est_ns` (whole layer, device view) for the est TPOT and `bound_ns` for the bound TPOT. `est_batches_ns` (MQSim comparison) is never used by M1.
3. Scaling: every demand term (`attended_tokens`, `dequant_ops`, `hbm_bytes`, `burst_bytes`) scales by B/b for compute and the HBM baseline; `est_ns` and `bound_ns` do not scale (per stack, stacks in parallel, one GPU drives all N stacks). When B mod N != 0, B/b is an average approximation while h is the busiest stack's time (documented assumption).
4. Dequant uses `vector_ops * compute_eff`.
5. Staged attention term: `a_l_staged = max(flops/F + deq/V, (hbm_bytes + burst_bytes)*B/b / BW)`. Prefetch uses a_l_staged in c_end. Serial with kv_consume="staged" uses h_l + a_l_staged; kv_consume="stream" (default) uses max(h_l, a_l) with a_l counting only hbm_bytes. (Writing the staged data into HBM is not counted; listed as an assumption.)
6. Missing layers: `evaluate()` raises if any layer 0..L-1 of a step is missing from layer_rows (the main agent's analysis emits zero rows for layers without HBF pages under --timing).
7. `hbm_fits`: weights_bytes = layers*(pre+post)*wb + vocab*hidden*wb*(1 if tied else 2); `hbm_fits` = weights + B*kv_bytes_per_request <= capacity (KV-in-HBM baseline); `hbm_fits_hbf` = weights + B*hbm_tier_bytes_per_request + prefetch buffer ((d+1) layers x max-layer burst_bytes*B/b; 0 for serial/ideal) <= capacity. `kv_bytes_per_request` = stored page-granular footprint (main agent computes it).
8. Derived columns: gpu_ns = sum(pre + a + post) + lm_head + overhead; hbf_wait_share: serial sum max(0, h_l - a_l)/T (stream) or sum h_l/T (staged); prefetch sum max(0, r_end(l) - (c_end(l-1) + pre_l))/T; ideal max(0, sum h - sum g)/T.
9. Frozen meta keys (written by the main CLI in meta.json under "system"): model (preset name), gpu (preset name), batch, analysis_batch, stacks, selection, kv_format, kv_bytes_per_request, hbm_tier_bytes_per_request. analysis_layers.csv rows to use: group == "1"; empty fields parse as None. Columns: step, layer, est_ns, bound_ns, burst_bytes, hbm_bytes, attended_tokens, dequant_ops.
10. Assumptions to list in the report: softmax/exp ignored; ThinK-pruned K gives 2*(d_k + head_dim) per (q head, token) instead of 4*head_dim (supported via an optional per-row `attn_dims` column = d_k + d_v; default 2*head_dim); per-query-head selection makes A an upper bound; selection scoring flops not counted; one query token per sequence (no speculative decoding); compute_eff 0.5 is a prefill MFU figure (decode is memory-bound for B < ~185 on H100).
