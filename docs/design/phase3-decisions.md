# Phase 3 decisions (main agent, after the planning subagent's design in phase3-kv-terms.md)

> Working notes (English). The plan and status are in [PLAN.md](../../PLAN.md). Phase 3 is developed on branch `claude/hbf-kv-terms`, which starts from `claude/hbf-kv-study` at 7c75e05; phase 2B work (T-B1d event simulation) lands on `claude/hbf-kv-study` first and is merged forward.

Spot checks of the design's numbers before accepting it: MLA row 576 × 2 B = 1,152 B, 3 per page, 1 − 3,456/4,096 = 15.6% waste; fused kivi2 row 96 B, 42 tokens per page, ⌈4,096/42⌉ = 98 pages; H2O 1 − 0.75^16 = 0.98998; DSA data pages 43,691 × (1 − (63/64)³) = 2,016 and indexer ⌈131,072/31⌉ = 4,229; hybrid test 5 × 1,024 + 32,768 = 37,888; KV bytes per token of DeepSeek-V2-Lite (27 × 1,152), Qwen3-Next (12 × 2 × 2 × 256 × 2) and GPT-OSS-120B global layers (18 × 8 × 2 × 64 × 2). All agree.

## Answers to the open questions (section 6)

1. **Row packing.** Keep the TensorSpec rule (a token row never spans pages). K1 shows the trade-off with `mla:split=1`, `mla_fp8` and `fused=1[,pow2=1]`. Row spanning is not implemented: GPU kernel support for 3, 6 or 42 tokens per page is unknown, and a straddling token costs two reads. Revisit if X2 shows the waste matters.
2. **MLA compute.** Use absorbed decode: per token, layer and query head, QK over 576 dims plus PV over 512 dims = 1,088 multiply-adds. `compute_eff` stays the global assumption; the report shows MLA results at 0.5 and 0.25.
3. **MoE serving.** Accept the DP-attention framing: each GPU owns its requests' KV on its own HBF stacks; expert weights are sharded and a step reads the experts touched by the batch. Expert-parallel communication is not modelled (stated as a limitation). K3 keeps MoE to the linear-layer timing only.
4. **Density approximation.** Report X5 as a range between random and clustered token sets until real traces exist (T-C1).
5. **CLA under dynamic selection.** Only dense and static selection; `reuse=hbm` adds the full-layer HBM copy per request to `hbm_fits_hbf`.
6. **Endurance interface.** `written_pages`, `dead_pages` and `layer_row_pages` are fixed after T-B1d merges, with the simulation's code in hand. K4 starts then.
7. **Write undercount for windows with a negative lower end.** K4 fixes it, with a test, before any sliding HBF window is used in endurance runs.
8. **Duplicated generator.** Accepted for K2; unify `generate_grouped` and `access.generate` after K5.
9. **Prefix reads.** Coalesced (once per step) is the default in K7, matching the existing per-(step, layer) dedup; per-request reads come later through a multiplicity column.
10. **Model parameters.** The K3 owner checks every preset against the published `config.json` and cites the source in `docs/research/params.md`. Default tier for sliding-window local KV: **HBM**; HBF is an experiment variant.
11. **`effective_gb_s`.** Use `logical_dims` as proposed.

## Work split

- K1 (page formats; track A: `access.py` TensorSpec, `formats.py`, `pages.py`, `tests/test_formats.py`): main agent, first.
- K2 (KV groups; track B: new `hbf/kvspec.py`, `cli.py`, `tests/test_kvspec.py`): file-disjoint from K1 and can run in parallel. It only reads the new TensorSpec fields, which default to the old behaviour.
- K3 and K5 after K1/K2; K4 and K7 after T-B1d merges; K6 last.
- Every milestone keeps the phase A golden files, engine parity and the phase 2A summary rows unchanged for existing flags, and ends with a review by a subagent before the next one starts.
