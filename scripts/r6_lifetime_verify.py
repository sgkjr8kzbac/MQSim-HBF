"""R6 lifetime verification (R6b: scaled-down MQSim vs Python) and full-scale extrapolation (R6c)."""
import csv
import math
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
from hbf.device.config import load
from hbf.frontend.kv.formats import parse_format
from hbf.frontend.timeline import build_timeline
from hbf.frontend.trace.requests import Workload, load_workload, parse_length
from hbf.lifetime.analytic import analytic
from hbf.lifetime.clock import ReadTimeClock, StepClock, StepTime
from hbf.lifetime.counts import compare_block_counts, counts_from_trace, rates_from_counts
from hbf.lifetime.geometry import GIB, Device, Geometry, geometry_for
from hbf.lifetime.sim import SimOptions, simulate as sim_simulate
from hbf.models import load_model
from hbf.registry import REGISTRY, ensure_loaded

VERIFY_CSV = ROOT / "evidence" / "r6-mqsim-verify.csv"
EXTRAP_CSV = ROOT / "evidence" / "r6-extrapolation.csv"


class _FixedStepClock(StepClock):
    """Deterministic constant step-time clock so R6b compares identical physical timelines."""

    def __init__(self, step_s, layers):
        self.step_s = float(step_s)
        self.layers = int(layers)

    def describe(self):
        return {"clock": "fixed", "step_s": self.step_s, "layers": self.layers}

    def step(self, contexts, partitions=None, partition_banks=None, hbm_tokens=None):
        if not len(contexts):
            return None
        tpot_ns = self.step_s * 1e9
        return StepTime(tpot_ns=tpot_ns, gpu_ns=0.0, hbm_tpot_ns=tpot_ns,
                        h_ns=tpot_ns / self.layers, gaps_ns=(0.0,) * self.layers,
                        read_pages=0.0, rounds=1, batch=len(contexts))


def run_scaled_verification(binary):
    """R6b: run scaled-down long32k and agentic through timeline -> MQSim (T-HBFW1..4) and Python Simulator."""
    cases = [
        # Scaled long32k: 16 banks (2x2x1x4), ppb=8, 8 segments (128 blocks total),
        # 2 layers x 2 heads BF16 (16 tok/page -> 4 pages/16 tokens -> 64 pages/256 tokens).
        # Output=48 tokens -> request lifetime 48 steps; X=256 -> threshold_steps = 256/8 = 32 steps.
        # Window = 240 steps = 5x request lifetime, 7.5x refresh threshold!
        dict(
            name="long32k_scaled",
            channels=2, chips=2, planes=4, blocks_per_plane=10, ppb=8, max_segments=8,
            layers=2, heads=2, head_dim=128,
            workload=Workload(
                name="long32k_scaled",
                input=parse_length("fixed:256"),
                output=parse_length("fixed:48"),
                arrival="closed",
                concurrency=4,
                seed=7,
            ),
            steps=240, step_s=0.01, x=256, retention_s=0.0,
            policy="packed", refresh="relocate", low_watermark=2,
        ),
        # Scaled agentic: 16 banks, ppb=8, 16 segments (256 blocks total),
        # 4 layers x 2 heads BF16 (16 tok -> 16 pages = 1 full row, matching real HBF multi-row calls!).
        # 2 concurrent sessions x 6 calls each, history=128 (8 rows), input=32 (2 rows), output=16 (1 row).
        # Session decode lifetime = 6*16 = 96 steps; X=256 -> threshold = 32 steps -> ~3 refreshes/session!
        # Window = 320 steps -> ~6 session completions & many RD refreshes of persistent history!
        dict(
            name="agentic_scaled",
            channels=2, chips=2, planes=4, blocks_per_plane=18, ppb=8, max_segments=16,
            layers=4, heads=2, head_dim=128,
            workload=Workload(
                name="agentic_scaled",
                input=parse_length("fixed:32"),
                output=parse_length("fixed:16"),
                arrival="closed",
                concurrency=2,
                calls=6,
                history=parse_length("fixed:128"),
                think_s=0.02,
                seed=11,
            ),
            steps=320, step_s=0.01, x=256, retention_s=0.0,
            policy="packed", refresh="relocate", low_watermark=2,
        ),
        # Scaled long32k with in-place replay refresh (exercises T-HBFW2 auto-erase on every refresh!)
        dict(
            name="long32k_scaled_replay",
            channels=2, chips=2, planes=4, blocks_per_plane=10, ppb=8, max_segments=8,
            layers=2, heads=2, head_dim=128,
            workload=Workload(
                name="long32k_scaled_replay",
                input=parse_length("fixed:256"),
                output=parse_length("fixed:48"),
                arrival="closed",
                concurrency=4,
                seed=7,
            ),
            steps=240, step_s=0.01, x=256, retention_s=0.0,
            policy="packed", refresh="replay", low_watermark=2,
        ),
    ]

    rows = []
    with tempfile.TemporaryDirectory(prefix="hbf-r6b-") as tmp:
        tmp_root = Path(tmp)
        for c in cases:
            cfg = load(
                "16x64", channels=c["channels"], chips=c["chips"], dies=1, planes=c["planes"],
                blocks_per_plane=c["blocks_per_plane"], pages_per_block=c["ppb"],
                aggregate_gb_s=64, plane_rule="independent", coalesce_ns=None, read_plane_ns=None,
            )
            fmt = parse_format("bf16", c["head_dim"])
            geo = Geometry(c["layers"], c["heads"], fmt.tensors, cfg.page_bytes)

            # 1. Frontend timeline -> validated AddressTrace
            trace, tl_stats = build_timeline(
                cfg, geo, c["workload"],
                steps=c["steps"], step_s=c["step_s"], policy=c["policy"],
                x=c["x"], retention_s=c["retention_s"], refresh=c["refresh"],
                low_watermark=c["low_watermark"], max_segments=c["max_segments"],
            )
            py_counts, py_by_cls = counts_from_trace(cfg, trace)

            # 2. MQSim (T-HBFW1..4) execution on the exact AddressTrace
            batches = batches_from_trace(cfg, trace.chunks)
            out_dir = tmp_root / c["name"]
            _, _, mq_steps, mq_stats = mqsim_simulate(cfg, batches, out_dir, binary, hbf_write=True)
            mq_counts = block_counts(out_dir / "block_counts.csv")
            cmp_blocks = compare_block_counts(mq_counts, py_counts)
            assert cmp_blocks["exact_match"], f"Block count mismatch on {c['name']}: {cmp_blocks}"

            kv_prog = tl_stats["host_programs"]
            ref_prog = tl_stats["refresh_programs"]
            mq_rates = rates_from_counts(
                mq_counts,
                total_blocks=c["max_segments"] * cfg.units,
                pages_per_block=cfg.pages_per_block,
                page_bytes=cfg.page_bytes,
                duration_s=tl_stats["duration_s"],
                kv_programs=kv_prog,
                refresh_programs=ref_prog,
            )

            # 3. Python segment-level event simulator on the exact same scaled Device, Geometry, Workload
            dev = Device(
                banks=cfg.units,
                pages_per_block=cfg.pages_per_block,
                page_bytes=cfg.page_bytes,
                capacity_gib=c["max_segments"] * cfg.units * cfg.pages_per_block * cfg.page_bytes / GIB,
            )
            clock = _FixedStepClock(c["step_s"], c["layers"])
            opts = SimOptions(
                policy=c["policy"], x=float(c["x"]), retention_s=c["retention_s"],
                hours=tl_stats["duration_s"] / 3600.0, warmup_h=0.0,
                quantum_s=c["step_s"], report_s=max(tl_stats["duration_s"], 1.0),
                rd_slack=0.0, refresh=c["refresh"], low_watermark=c["low_watermark"],
                m5="off",
            )
            sim = sim_simulate(geo, clock, c["workload"], dev, opts)
            sres = sim.result

            pec_rel = (mq_rates["avg_pec_opened_per_day"] - sres["avg_pec_per_day"]) / sres["avg_pec_per_day"]
            raw_pec_rel = (mq_rates["avg_pec_per_day"] - sres["avg_pec_per_day"]) / sres["avg_pec_per_day"]
            ref_ratio_mq = mq_rates["refresh_to_kv"]
            ref_ratio_sim = sres["refresh_to_kv"]
            ref_rel = ((ref_ratio_mq - ref_ratio_sim) / ref_ratio_sim) if ref_ratio_sim else 0.0
            prog_rel = (mq_rates["program_gb_s"] - sres["program_gb_s"]) / sres["program_gb_s"]

            row = {
                "case": c["name"],
                "steps": c["steps"],
                "duration_s": round(tl_stats["duration_s"], 4),
                "segments": c["max_segments"],
                "blocks_exact_match": cmp_blocks["exact_match"],
                "mqsim_reads": cmp_blocks["mqsim_totals"][0],
                "mqsim_programs": cmp_blocks["mqsim_totals"][1],
                "mqsim_erases": cmp_blocks["mqsim_totals"][2],
                "mqsim_completed": tl_stats["completed"],
                "sim_completed": sres["completed"],
                "mqsim_rd_refreshes": tl_stats["refresh_events_rd"],
                "sim_rd_refreshes": sres["refresh_events_rd"],
                "mqsim_pec_opened_day": round(mq_rates["avg_pec_opened_per_day"], 2),
                "mqsim_pec_raw_day": round(mq_rates["avg_pec_per_day"], 2),
                "sim_pec_day": round(sres["avg_pec_per_day"], 2),
                "pec_opened_rel_err": round(pec_rel, 5),
                "pec_raw_rel_err": round(raw_pec_rel, 5),
                "mqsim_refresh_to_kv": round(ref_ratio_mq, 4),
                "sim_refresh_to_kv": round(ref_ratio_sim, 4),
                "refresh_to_kv_rel_err": round(ref_rel, 5),
                "program_gb_s_rel_err": round(prog_rel, 5),
                "mqsim_wall_s": round(mq_stats["wall_seconds"], 3),
            }
            rows.append(row)
            print(f"[R6b] {row['case']:22s} exact={row['blocks_exact_match']} "
                  f"pec(opened/sim)={row['mqsim_pec_opened_day']:.1f}/{row['sim_pec_day']:.1f} "
                  f"({row['pec_opened_rel_err']*100:+.2f}%)  "
                  f"ref/kv={row['mqsim_refresh_to_kv']:.3f}/{row['sim_refresh_to_kv']:.3f} "
                  f"({row['refresh_to_kv_rel_err']*100:+.2f}%)")
    return rows


def run_full_extrapolation():
    """R6c: full-scale Python extrapolation on long32k and agentic (ReadTimeClock + ext/system TPOT Clock)."""
    ensure_loaded()
    model = load_model("llama-3.1-8b")
    geo, _ = geometry_for(model, "bf16")
    dev = Device(banks=1024, pages_per_block=256, page_bytes=4096, capacity_gib=512.0)

    # 1. Core ReadTimeClock (no ext/system): rounds-calibrated HBF read time + GPU compute_ns parameter (0 ms & 10 ms).
    # 2. Extension TPOT Clock (ext/system, B200 GPU).
    tpot_factory = REGISTRY.get_clock("tpot")
    sys_ev = REGISTRY.get_evaluator("system", required=True)
    b200 = sys_ev.load_gpu("b200")

    rows = []
    for wl_name in ("long32k", "agentic"):
        wl = load_workload(wl_name, concurrency=8, seed=1)
        for x_val in (1e5, 1e6):
            # Core ReadTimeClock with compute_ns=0 (HBF read-time bound) and compute_ns=5ms (constant GPU compute term)
            clock_specs = [
                ("core_read_time_0ms", ReadTimeClock(geo, model.layers, compute_ns=0.0)),
                ("core_read_time_5ms", ReadTimeClock(geo, model.layers, compute_ns=5_000_000.0)),
                ("ext_system_b200", tpot_factory(model, b200, geo, kv_format="bf16")),
            ]
            for clock_label, clk in clock_specs:
                an_row = analytic(geo, clk, wl, dev, policy="packed", x_values=(x_val,))[0]
                opts = SimOptions(policy="packed", x=x_val, hours=24.0, warmup_h=1.0, m5="gaps")
                sim = sim_simulate(geo, clk, wl, dev, opts)
                sr = sim.result
                row = {
                    "workload": wl_name,
                    "x": f"{x_val:.0e}",
                    "clock": clock_label,
                    "mean_tpot_ms": round(sr["mean_tpot_ms"], 3),
                    "tpot_m5_ms": round(sr["tpot_m5_ms"], 3),
                    "kv_write_gb_s": round(sr["kv_write_gb_s"], 4),
                    "refresh_to_kv_sim": round(sr["refresh_to_kv"], 4),
                    "refresh_to_kv_analytic": round(an_row["refresh_to_kv"], 4),
                    "avg_pec_day_sim": round(sr["avg_pec_per_day"], 2),
                    "avg_pec_day_analytic": round(an_row["avg_pec_per_day"], 2),
                    "max_pec_day_sim": round(sr["max_pec_per_day"], 2),
                    "days_1e4_avg": round(sr["days_to_limit_avg@1e4"], 1),
                    "days_1e5_avg": round(sr["days_to_limit_avg@1e5"], 1),
                    "days_1e6_avg": round(sr["days_to_limit_avg@1e6"], 1),
                    "refresh_events_rd": sr["refresh_events_rd"],
                    "completed": sr["completed"],
                }
                rows.append(row)
                print(f"[R6c] {wl_name:8s} X={row['x']} {clock_label:20s} "
                      f"TPOT={row['mean_tpot_ms']:6.2f}ms  "
                      f"PEC/d(sim/an)={row['avg_pec_day_sim']:6.1f}/{row['avg_pec_day_analytic']:6.1f}  "
                      f"ref/kv={row['refresh_to_kv_sim']:5.2f}/{row['refresh_to_kv_analytic']:5.2f}  "
                      f"days@1e5={row['days_1e5_avg']:6.1f}")
    return rows


def main():
    binary = build()
    verify_rows = run_scaled_verification(binary)
    VERIFY_CSV.parent.mkdir(parents=True, exist_ok=True)
    with VERIFY_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(verify_rows[0]))
        w.writeheader()
        w.writerows(verify_rows)
    print(f"Wrote {VERIFY_CSV}")

    extrap_rows = run_full_extrapolation()
    with EXTRAP_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(extrap_rows[0]))
        w.writeheader()
        w.writerows(extrap_rows)
    print(f"Wrote {EXTRAP_CSV}")


if __name__ == "__main__":
    main()
