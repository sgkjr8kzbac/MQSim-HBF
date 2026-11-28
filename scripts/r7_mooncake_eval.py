"""R7 end-to-end Mooncake evaluation: trace stats, small-topology MQSim + rounds check,
and full-scale 24h lifetime extrapolation vs synthetic presets and sparse token selection."""
import csv
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hbf.backends.mqsim.build import build
from hbf.backends.mqsim.parse import block_counts
from hbf.backends.mqsim.run import simulate as mqsim_simulate
from hbf.backends.mqsim.schedule import batches_from_trace
from hbf.backends.rounds.analysis import analyze_trace
from hbf.device.config import load
from hbf.frontend.addrtrace import validate, write_trace
from hbf.frontend.kv.formats import parse_format
from hbf.frontend.timeline import build_timeline
from hbf.frontend.trace.requests import (
    RequestRecord,
    import_mooncake_jsonl,
    load_workload,
    per_stack,
    read_requests_csv,
    summarize_requests,
    workload_from_requests,
    write_requests_csv,
)
from hbf.lifetime.clock import calibrate
from hbf.lifetime.counts import compare_block_counts, counts_from_trace, rates_from_counts
from hbf.lifetime.geometry import Device, Geometry, geometry_for
from hbf.lifetime.sim import SimOptions, simulate as sim_simulate
from hbf.models import load_model
from hbf.registry import REGISTRY, ensure_loaded

MOONCAKE_JSONL = ROOT / ".build" / "traces" / "mooncake" / "conversation_trace.jsonl"
CSV_EXPORT = ROOT / ".build" / "traces" / "mooncake" / "conversation_requests.csv"
SAMPLE_ADDR_DIR = ROOT / ".build" / "traces" / "mooncake" / "r7_sample_addr"
EVAL_CSV = ROOT / "evidence" / "r7-mooncake-eval.csv"


def run_scaled_window_verification(binary, records):
    """Run a real Mooncake request window (with shared prefixes) through timeline -> AddressTrace
    -> small-topology MQSim (T-HBFW1..4) AND rounds backend, verifying 100% block count match."""
    # Pick the first 12 requests belonging to shared multi-turn conversations (>=1024 tok prefix)
    # compressed into a short window so prefix reuse, decode flushes, completions, and RD refreshes
    # all execute cleanly inside a fast 16-bank MQSim run.
    by_pid = {}
    for rec in records:
        if rec.prefix_id is not None and (rec.prefix_tokens or 0) >= 1024 and rec.prompt_tokens <= 12288:
            by_pid.setdefault(rec.prefix_id, []).append(rec)
    picked = []
    for pid, grp in by_pid.items():
        if len(grp) >= 2:
            picked.extend(grp[:3])
        if len(picked) >= 8:
            break
    # Preserve original relative ordering and compact inter-arrival gaps onto a 0.25s window,
    # keeping exact Mooncake prompt/prefix ratios scaled by 1/16 to fit a fast 16-bank topology.
    picked.sort(key=lambda r: r.arrival_s)
    t0 = picked[0].arrival_s
    span = max(picked[-1].arrival_s - t0, 1.0)
    window_recs = []
    for rec in picked:
        rel = (rec.arrival_s - t0) / span * 0.25
        pt = max(32, ((rec.prefix_tokens or 0) // 16 // 16) * 16)
        pr = max(pt + 32, (rec.prompt_tokens // 16 // 16) * 16)
        out = max(16, min(64, rec.output_tokens // 8))
        window_recs.append(RequestRecord(
            arrival_s=round(rel, 4),
            prompt_tokens=pr,
            output_tokens=out,
            prefix_id=rec.prefix_id,
            prefix_tokens=pt,
        ))

    wl = workload_from_requests(
        window_recs, name="mooncake_window", retain_s=1.0, loop=False,
        source="Scaled real Mooncake multi-turn conversation subset",
    )
    cfg = load(
        "16x64", channels=2, chips=2, dies=1, planes=4,
        blocks_per_plane=24, pages_per_block=8,
        aggregate_gb_s=64, plane_rule="independent", coalesce_ns=None, read_plane_ns=None,
    )
    geo = Geometry(2, 2, parse_format("bf16", 128).tensors, cfg.page_bytes)
    steps = 120
    step_s = 0.01

    trace, tl_stats = build_timeline(
        cfg, geo, wl, steps=steps, step_s=step_s, policy="packed",
        x=256, retention_s=0.0, refresh="relocate", low_watermark=2, max_segments=20,
    )
    validate(cfg, trace)

    # Save AddressTrace sample so HBFSim (R9) can also replay the exact same Mooncake trace.
    SAMPLE_ADDR_DIR.mkdir(parents=True, exist_ok=True)
    write_trace(SAMPLE_ADDR_DIR, trace.meta, trace.chunks)

    # 1. Rounds backend analysis on the exact AddressTrace
    layer_rows, step_rows = analyze_trace(cfg, trace.chunks)
    rounds_read_pages = sum(r["pages"] for r in layer_rows)
    rounds_est_ns = sum(r["est_batches_ns"] for r in step_rows if r["est_batches_ns"] is not None)

    # 2. Python per-block counts from AddressTrace
    py_counts, _ = counts_from_trace(cfg, trace)

    # 3. MQSim execution (T-HBFW1..4) on the exact AddressTrace
    batches = batches_from_trace(cfg, trace.chunks)
    with tempfile.TemporaryDirectory(prefix="hbf-r7-mq-") as tmp:
        out_dir = Path(tmp) / "mqsim"
        _, _, mq_steps, mq_stats = mqsim_simulate(cfg, batches, out_dir, binary, hbf_write=True)
        mq_counts = block_counts(out_dir / "block_counts.csv")
        cmp = compare_block_counts(mq_counts, py_counts)
        assert mq_stats["Total_GC_Executions"] == 0, mq_stats
        assert cmp["exact_match"], f"Block count mismatch on Mooncake window: {cmp}"
        mq_rates = rates_from_counts(
            mq_counts,
            total_blocks=20 * cfg.units,
            pages_per_block=cfg.pages_per_block,
            page_bytes=cfg.page_bytes,
            duration_s=tl_stats["duration_s"],
            kv_programs=tl_stats["host_programs"],
            refresh_programs=tl_stats["refresh_programs"],
        )
        mq_step_ns = sum(float(r["mqsim_ns"]) for r in mq_steps if int(r["step"]) >= 0)
        mq_read_pages = sum(int(r["read_pages"]) for r in mq_steps if int(r["step"]) >= 0)

    # Note: py_counts includes prefill/refresh reads in step=-1 or post-ops, while rounds_read_pages
    # counts step >= 0 read pages (identical to mq_read_pages).
    assert rounds_read_pages == mq_read_pages, (rounds_read_pages, mq_read_pages)

    summary = {
        "requests": len(window_recs),
        "completed": tl_stats["completed"],
        "steps": steps,
        "blocks_exact_match": cmp["exact_match"],
        "mqsim_total_reads": cmp["mqsim_totals"][0],
        "mqsim_total_programs": cmp["mqsim_totals"][1],
        "mqsim_total_erases": cmp["mqsim_totals"][2],
        "step_read_pages_rounds": rounds_read_pages,
        "step_read_pages_mqsim": mq_read_pages,
        "rounds_est_ms": round(rounds_est_ns / 1e6, 3),
        "mqsim_step_ms_with_writes": round(mq_step_ns / 1e6, 3),
        "refresh_events_rd": tl_stats["refresh_events_rd"],
        "mqsim_pec_opened_day": round(mq_rates["avg_pec_opened_per_day"], 2),
        "mqsim_wall_s": round(mq_stats["wall_seconds"], 3),
        "sample_addr_dir": str(SAMPLE_ADDR_DIR.relative_to(ROOT)),
    }
    print("[R7 window check]", json.dumps(summary, indent=2))
    return summary


def run_full_scale_eval(recs_conv, recs_none):
    """Full-scale 24h lifetime extrapolation on 512 GiB OCP stack, Llama-3.1-8B BF16, packed,
    comparing Mooncake (with/without prefix sharing + sparse token selection) vs synthetic presets."""
    ensure_loaded()
    model = load_model("llama-3.1-8b")
    geo, _ = geometry_for(model, "bf16")
    dev = Device(banks=1024, pages_per_block=256, page_bytes=4096, capacity_gib=512.0)
    tpot_factory = REGISTRY.get_clock("tpot")
    sys_ev = REGISTRY.get_evaluator("system", required=True)
    b200 = sys_ev.load_gpu("b200")
    sys_opts = sys_ev.SystemOptions(overlap="serial", kv_consume="stream")

    # Calibrate sparse read_share and ratios for uniform_tokens(0.1) and sink_recent(0.1, count=4)
    # on a 16K context (closest power-of-2 to Mooncake's 12K mean prompt).
    with tempfile.TemporaryDirectory(prefix="hbf-r7-calib-") as tmp:
        calib_dir = Path(tmp) / "calib"
        calib_dense = calibrate(
            calib_dir, model.name, "b200", "bf16", mode="dense",
            generation={}, context=16384, device=[], policy="packed", partition="none",
        )
        calib_unif = calibrate(
            calib_dir, model.name, "b200", "bf16", mode="uniform_tokens",
            generation={"fraction": 0.1, "select_tokens": 16, "count": 4, "seed": 1},
            context=16384, device=[], policy="packed", partition="none",
        )
        calib_sink = calibrate(
            calib_dir, model.name, "b200", "bf16", mode="sink_recent",
            generation={"fraction": 0.1, "select_tokens": 16, "count": 4, "seed": 1},
            context=16384, device=[], policy="packed", partition="none",
        )

    def make_clk(calib, selection="static"):
        return tpot_factory(
            model, b200, geo,
            layer_ns=calib["layer_ns"], round_ns=calib["round_ns"],
            balance=calib["balance"], ratios=calib["ratios"],
            selection=selection, kv_format="bf16", options=sys_opts,
        )

    mc_conv_full = workload_from_requests(recs_conv, name="mooncake_prefix_300s", retain_s=300.0, loop=True)
    mc_conv_60 = workload_from_requests(recs_conv, name="mooncake_prefix_60s", retain_s=60.0, loop=True)
    mc_none_full = workload_from_requests(recs_none, name="mooncake_no_prefix", retain_s=0.0, loop=True)

    # Note: Mooncake conversation_trace.jsonl arrives at 3.401 req/s (cluster-level trace, 12K mean prompt).
    # A single dense BF16 HBF stack (1,024 banks, B200) saturates around ~1.47 req/s; sharding across
    # stacks=4 with prefix-aware routing (0.85 req/s per stack) gives the realistic unsaturated regime.
    # Meanwhile, 10% sparse token selection (uniform_tokens / sink_recent) reduces read load enough that
    # a single stack sustains the full 3.401 req/s cluster trace unsaturated!
    scenarios = [
        # (label, source, stacks, prefix_mode, retain_s, token_mode, calib, selection, wl, hours, warmup_h)
        (
            "mooncake_4stack_prefix_300s", "mooncake", 4, "conversation", 300.0, "dense",
            calib_dense, "static", per_stack(mc_conv_full, 4), 24.0, 1.0,
        ),
        (
            "mooncake_4stack_prefix_60s", "mooncake", 4, "conversation", 60.0, "dense",
            calib_dense, "static", per_stack(mc_conv_60, 4), 24.0, 1.0,
        ),
        (
            "mooncake_4stack_no_prefix", "mooncake", 4, "none", 0.0, "dense",
            calib_dense, "static", per_stack(mc_none_full, 4), 24.0, 1.0,
        ),
        (
            "mooncake_4stack_prefix_300s_uniform10", "mooncake", 4, "conversation", 300.0, "uniform_tokens:0.1",
            calib_unif, "dynamic", per_stack(mc_conv_full, 4), 24.0, 1.0,
        ),
        (
            "mooncake_4stack_prefix_300s_sink10", "mooncake", 4, "conversation", 300.0, "sink_recent:0.1",
            calib_sink, "static", per_stack(mc_conv_full, 4), 24.0, 1.0,
        ),
        (
            "mooncake_1stack_prefix_300s_uniform10", "mooncake", 1, "conversation", 300.0, "uniform_tokens:0.1",
            calib_unif, "dynamic", mc_conv_full, 24.0, 1.0,
        ),
        (
            "mooncake_1stack_prefix_300s_sink10", "mooncake", 1, "conversation", 300.0, "sink_recent:0.1",
            calib_sink, "static", mc_conv_full, 24.0, 1.0,
        ),
        (
            "mooncake_1stack_dense_saturated", "mooncake", 1, "conversation", 300.0, "dense",
            calib_dense, "static", mc_conv_full, 4.0, 0.5,
        ),
        (
            "synthetic_chat_c8", "synthetic", 1, "none", 0.0, "dense",
            calib_dense, "static", load_workload("chat", concurrency=8, seed=1), 24.0, 1.0,
        ),
        (
            "synthetic_long32k_c8", "synthetic", 1, "none", 0.0, "dense",
            calib_dense, "static", load_workload("long32k", concurrency=8, seed=1), 24.0, 1.0,
        ),
        (
            "synthetic_agentic_c8", "synthetic", 1, "none", 0.0, "dense",
            calib_dense, "static", load_workload("agentic", concurrency=8, seed=1), 24.0, 1.0,
        ),
    ]

    rows = []
    for (label, source_kind, stacks, prefix_mode, retain_s, token_mode,
         calib, selection, wl, hours, warmup_h) in scenarios:
        clk = make_clk(calib, selection=selection)
        share = float(calib["read_share"])
        for x_val in (1e5, 1e6):
            opts = SimOptions(
                policy="packed", x=x_val, share=share, hours=hours, warmup_h=warmup_h,
                stacks=stacks, m5="gaps",
            )
            sim = sim_simulate(geo, clk, wl, dev, opts, model=model, gpu=b200)
            sr = sim.result
            row = {
                "scenario": label,
                "source": source_kind,
                "stacks": stacks,
                "prefix_mode": prefix_mode,
                "retain_s": retain_s,
                "token_mode": token_mode,
                "read_share": round(share, 4),
                "x": f"{x_val:.0e}",
                "hours": hours,
                "mean_batch": round(sr["mean_batch"], 2),
                "mean_context": round(sr["mean_context"], 1),
                "req_per_s": round(sr["req_per_s"], 3),
                "mean_tpot_ms": round(sr["mean_tpot_ms"], 3),
                "tpot_m5_ms": round(sr["tpot_m5_ms"], 3),
                "kv_write_gb_s": round(sr["kv_write_gb_s"], 4),
                "refresh_rd_gb_s": round(sr["refresh_rd_gb_s"], 4),
                "program_gb_s": round(sr["program_gb_s"], 4),
                "refresh_to_kv": round(sr["refresh_to_kv"], 4),
                "wa": round(sr["wa"], 4),
                "avg_pec_per_day": round(sr["avg_pec_per_day"], 2),
                "max_pec_per_day": round(sr["max_pec_per_day"], 2),
                "days_to_1e4": round(sr["days_to_limit_avg@1e4"], 1),
                "days_to_1e5": round(sr["days_to_limit_avg@1e5"], 1),
                "days_to_1e6": round(sr["days_to_limit_avg@1e6"], 1),
                "refresh_events_rd": sr["refresh_events_rd"],
                "evicted_gb": round(sr["evicted_gb"], 2),
                "stalls": sr["stalls"],
                "completed": sr["completed"],
            }
            rows.append(row)
            print(
                f"[R7 sim] {label:38s} X={row['x']} sh={row['read_share']:.3f} "
                f"B={row['mean_batch']:5.1f} TPOT={row['mean_tpot_ms']:6.2f}ms req/s={row['req_per_s']:5.2f} "
                f"KV={row['kv_write_gb_s']:5.2f}GB/s ref/kv={row['refresh_to_kv']:5.2f} "
                f"PEC/d={row['avg_pec_per_day']:6.1f} d@1e5={row['days_to_1e5']:7.1f} st={row['stalls']}",
                flush=True,
            )
    return rows


def main():
    if not MOONCAKE_JSONL.is_file():
        raise SystemExit(
            f"Missing {MOONCAKE_JSONL}; run scripts/download_mooncake.sh first"
        )
    recs_conv = import_mooncake_jsonl(MOONCAKE_JSONL, prefix_mode="conversation")
    recs_none = import_mooncake_jsonl(MOONCAKE_JSONL, prefix_mode="none")
    stats_conv = summarize_requests(recs_conv)
    print("[R7 Mooncake stats]", json.dumps(stats_conv, indent=2))

    # Verify full 12,031-row unified CSV round-trip
    write_requests_csv(CSV_EXPORT, recs_conv)
    rt = read_requests_csv(CSV_EXPORT)
    assert len(rt) == len(recs_conv) and rt[0] == recs_conv[0] and rt[-1] == recs_conv[-1]
    print(f"[R7 CSV round-trip ok] {len(rt)} rows -> {CSV_EXPORT.relative_to(ROOT)}")

    binary = build()
    run_scaled_window_verification(binary, recs_conv)

    rows = run_full_scale_eval(recs_conv, recs_none)
    EVAL_CSV.parent.mkdir(parents=True, exist_ok=True)
    with EVAL_CSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {EVAL_CSV} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
