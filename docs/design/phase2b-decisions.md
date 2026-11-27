# Phase B decisions (main agent, after designer-b's design in phase2b-design.md)

> Working notes of the design discussion (English). The plan and status are in [PLAN.md](../../PLAN.md); parameter sources are in [docs/research/params.md](../research/params.md).

Read phase2b-design.md in full; this file answers its open questions and splits the work.

## Answers to the open questions (section 11)

1. §5.4.1 item 4 is read per NAND block (with items 9 and §11.5.2.6), as designed. State the literal per-channel reading as an alternative in the report (it would allow one log per host channel and forbid `dedicated`).
2. Every page-0 write counts as an auto-erase (P/E +1), also on never-written blocks (spec-literal, conservative); the per-block read counter resets at erase. Both are labelled assumptions.
3. Defaults accepted: allocation `packed` + evict, disaggregated prefill, `--prefix-reads per_request` in M4, wear `dynamic`, M5 `gaps`, die-level blocking. The report says that phase A's per-(step, layer) dedup models a shared prefix read once per step.
4. Clock surrogate accepted (fitted est + `system.evaluate`), with the phase A cross-check test; `--tpot-table` optional, not required.
5. Yes: `pipeline(requests=...)` may take a `BatchSpec` (the `(per_stack, heads)` tuple stays accepted as the layer_major case) and `generate()` gains `requests=`; defaults stay bit-identical (golden files).
6. Yes: add ppb 512 and 1024 to the sweeps; M4 takes `--capacity-gib` independent of `--pages-per-block`.
7. Yes: add the `hbm_decode` baseline (generated KV never written to HBF) and the weights-on-HBF read-disturb note in the report.
8. Budget accepted. Scope order if time runs short: M5 analytic, M4a analytic, write-legality checker (E5.1), M2 read parallelism per policy (E5.2), M4b simulation; vllm_pool simulation and the MQSim M5 confirmation last.

## Work split (file-disjoint, in parallel)

- Main agent: M2 = `hbf/batch.py` (BatchSpec, BatchRequest, PageTableLayout, policy address maps, synthetic history, `check_write_order`), `generate(requests=)`, `pipeline(requests=BatchSpec)`, CLI flags in hbf/cli.py, tests/test_batch.py; later the sweep `"tool"` key.
- impl-m4 (subagent): `hbf/interference.py` (M5, section 5 of phase2b-design.md) first, then `hbf/endurance.py` (Clock + calibrate, `analytic`, `simulate`, CLI with build_parser/configure/run/summarize_run/METRICS mirroring hbf.cli), `hbf/workload.py`, `tests/test_interference.py`, `tests/test_endurance.py`. New files only. M4's allocation policies (packed, dedicated, channel/die partition, layer_major waves, vllm_pool, hbm_decode) are segment-level abstractions inside endurance.py and do not depend on hbf/batch.py.

## Interfaces fixed now

- M4 calls phase A only through public functions: `hbf.cli.build_parser`, `hbf.cli.configure`, `hbf.cli.pipeline` (analysis-only, for calibration), `hbf.system.evaluate`, `hbf.model.load_model/load_gpu`, `hbf.formats.parse_format`, `hbf.access.TensorSpec`.
- M5's public function: `interference(step_inputs, mode, t_prog, t_bers, suspend_ns=None, dies=64, banks_per_die=16, block_unit="die") -> dict(extra_step_ns, extra_layer_ns, max_step_extra_ns)`; M4 calls it per quantum.
- M4's report and CSV column names as in phase2b-design.md section 6.
