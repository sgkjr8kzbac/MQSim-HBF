"""Milestone R9 3-way comparison: HBFSim vs MQSim vs Rounds/Interference models.

Group A (Phase 2A — 15 confirmation points from evidence/scripts/mqsim_confirm.sh):
  Feeds the identical AddressTrace to:
    1. MQSim (sequentially, one 1,024-plane process at a time)
    2. Rounds model (hbf.backends.rounds.analysis.analyze_trace)
    3. HBFSim (raw-physical, both core-BW-matched and native speed-grade BW)
  Outputs: evidence/r9-phase2a-3way.csv

Group B (R5.3 — 9 write-interference points from scripts/r5_write_interference.py):
  Feeds the identical read-only and with-writes traces to:
    1. MQSim (T-HBFW1..4, write_order=fifo and read_first)
    2. HBFSim default (hbf-die-blocking=false, plane-level blocking)
    3. HBFSim patched (hbf-die-blocking=true, die-level blocking)
    4. interference.py (fifo and gaps)
  Outputs: evidence/r9-write-interference-3way.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hbf.backends.hbfsim.build import build as build_hbfsim
from hbf.backends.hbfsim.convert import (
    batches_to_converted,
    convert_trace,
    hbfsim_geometry,
    physical_coords,
)
from hbf.backends.hbfsim.parse import extract_hbf_delta
from hbf.backends.hbfsim.run import HbfSimSession, simulate_batches, write_config
from hbf.backends.mqsim.build import build as build_mqsim
from hbf.backends.mqsim.parse import completions as mqsim_completions
from hbf.backends.mqsim.run import simulate as simulate_mqsim
from hbf.backends.rounds.analysis import analyze_trace
from hbf.cli import _main_run
from hbf.device.config import load as load_cfg
from hbf.frontend.addrtrace import AddressTrace, read_trace
from hbf.frontend.pages import Batch, Request
from hbf.lifetime.interference import interference

PHASE2A_CSV = ROOT / "evidence" / "r9-phase2a-3way.csv"
INTERF_CSV = ROOT / "evidence" / "r9-write-interference-3way.csv"

# Exact 15 confirmation points from evidence/scripts/mqsim_confirm.sh
PHASE2A_CASES: list[tuple[str, list[str]]] = [
    ("dense-128k-1024", ["--layers", "2", "--context", "131072"]),
    ("dense-128k-384", ["--layers", "2", "--context", "131072", "--aggregate-gb-s", "384"]),
    ("dense-128k-3072", ["--layers", "2", "--context", "131072", "--aggregate-gb-s", "3072"]),
    ("dense-32k-b8", ["--layers", "1", "--heads", "64", "--context", "32768"]),
    ("tokens1-128k-striped", ["--layers", "2", "--context", "131072", "--mode", "uniform_tokens", "--fraction", "0.01", "--layout", "striped_contiguous"]),
    ("tokens5-128k-striped", ["--layers", "2", "--context", "131072", "--mode", "uniform_tokens", "--fraction", "0.05", "--layout", "striped_contiguous"]),
    ("tokens5-128k-layer", ["--layers", "2", "--context", "131072", "--mode", "uniform_tokens", "--fraction", "0.05"]),
    ("kivi2-dense-32k", ["--layers", "2", "--context", "32768", "--format", "kivi2"]),
    ("kivi2-tokens1-16", ["--layers", "2", "--context", "131072", "--format", "kivi2", "--mode", "uniform_tokens", "--fraction", "0.01", "--select-tokens", "16"]),
    ("kivi2-tokens1-128", ["--layers", "2", "--context", "131072", "--format", "kivi2", "--mode", "uniform_tokens", "--fraction", "0.01", "--select-tokens", "128"]),
    ("int4-dense-32k", ["--layers", "2", "--context", "32768", "--format", "int4"]),
    ("quest-tokens5-32k", ["--layers", "2", "--context", "32768", "--format", "quest", "--mode", "uniform_tokens", "--fraction", "0.05"]),
    ("tr2048-dense-8k", ["--layers", "2", "--context", "8192", "--read-ns", "2048"]),
    ("tr8192-tokens5-32k", ["--layers", "2", "--context", "32768", "--mode", "uniform_tokens", "--fraction", "0.05", "--read-ns", "8192"]),
    ("staggered-sink", ["--layers", "2", "--context", "32768", "--mode", "sink_recent", "--fraction", "0.05", "--layout", "striped_staggered"]),
]

# Exact 9 write-interference points from scripts/r5_write_interference.py
R5_POINTS = [
    {"name": "P1_prog_fits_gap", "topology": "ocp", "channels": 2, "chips": 2, "planes": 16,
     "blocks": 8, "ppb": 16, "gb_s": 100, "layers": 4, "t_prog": 100_000, "t_bers": 1_000_000,
     "gap_ns": 200_000, "wave_banks": 64, "prog_waves": 1, "erase_waves": 0},
    {"name": "P2_prog_overflows_gap", "topology": "ocp", "channels": 2, "chips": 2, "planes": 16,
     "blocks": 8, "ppb": 16, "gb_s": 100, "layers": 4, "t_prog": 100_000, "t_bers": 1_000_000,
     "gap_ns": 40_000, "wave_banks": 64, "prog_waves": 1, "erase_waves": 0},
    {"name": "P3_prog_burst_tight", "topology": "ocp", "channels": 2, "chips": 2, "planes": 16,
     "blocks": 8, "ppb": 16, "gb_s": 100, "layers": 4, "t_prog": 100_000, "t_bers": 1_000_000,
     "gap_ns": 10_000, "wave_banks": 64, "prog_waves": 2, "erase_waves": 0, "burst_in_layer0": True},
    {"name": "P4_prog_350us_spread", "topology": "ocp", "channels": 2, "chips": 2, "planes": 16,
     "blocks": 8, "ppb": 16, "gb_s": 100, "layers": 4, "t_prog": 350_000, "t_bers": 3_500_000,
     "gap_ns": 100_000, "wave_banks": 64, "prog_waves": 2, "erase_waves": 0},
    {"name": "P5_erase_fits_gap", "topology": "ocp", "channels": 2, "chips": 2, "planes": 16,
     "blocks": 8, "ppb": 16, "gb_s": 100, "layers": 4, "t_prog": 100_000, "t_bers": 1_000_000,
     "gap_ns": 1_200_000, "wave_banks": 64, "prog_waves": 1, "erase_waves": 1},
    {"name": "P6_erase_overflows_gap", "topology": "ocp", "channels": 2, "chips": 2, "planes": 16,
     "blocks": 8, "ppb": 16, "gb_s": 100, "layers": 4, "t_prog": 100_000, "t_bers": 1_000_000,
     "gap_ns": 400_000, "wave_banks": 64, "prog_waves": 1, "erase_waves": 1},
    {"name": "P7_partial_1die_wave", "topology": "ocp", "channels": 2, "chips": 2, "planes": 16,
     "blocks": 8, "ppb": 16, "gb_s": 100, "layers": 4, "t_prog": 100_000, "t_bers": 1_000_000,
     "gap_ns": 10_000, "wave_banks": 16, "prog_waves": 1, "erase_waves": 0},
    {"name": "P8_full_ocp_64die", "topology": "ocp", "channels": 32, "chips": 2, "planes": 16,
     "blocks": 8, "ppb": 16, "gb_s": 1600, "layers": 4, "t_prog": 100_000, "t_bers": 1_000_000,
     "gap_ns": 50_000, "wave_banks": 1024, "prog_waves": 2, "erase_waves": 0},
    {"name": "P9_p2_coalesce_1ns", "topology": "ocp", "channels": 2, "chips": 2, "planes": 16,
     "blocks": 8, "ppb": 16, "gb_s": 100, "layers": 4, "t_prog": 100_000, "t_bers": 1_000_000,
     "gap_ns": 40_000, "wave_banks": 64, "prog_waves": 1, "erase_waves": 0, "coalesce_ns": 1},
]


def _cfg_from_params_json(params_path: Path):
    p = json.loads(params_path.read_text())
    overrides = {
        k: p[k]
        for k in (
            "channels", "chips", "dies", "planes", "page_bytes", "sector_bytes",
            "pages_per_block", "blocks_per_plane", "read_ns", "program_ns", "erase_ns",
            "op_ratio", "gc_threshold", "aggregate_gb_s", "transfer_rate",
            "pcie_lane_gb_s", "pcie_lanes", "queue_depth", "queue_fetch",
            "batch_pages", "interval_ns", "cmd_ns", "rb_ns", "host_ns",
            "suspend", "suspend_ns", "suspend_threshold_ns", "scheduler",
            "plane_rule", "dbsy_ns", "coalesce_ns", "read_plane_ns",
            "cache_read", "address_map",
        )
        if k in p
    }
    if p.get("onfi") is not None:
        overrides["onfi"] = tuple(p["onfi"])
    return load_cfg("ocp", **overrides)


def run_phase2a(hbfsim_bin: Path, tmp_root: Path) -> list[dict]:
    """Run all 15 Phase 2A confirmation cases across MQSim, Rounds, and HBFSim."""
    rows: list[dict] = []
    print("\n=== Group A: Phase 2A 15-point 3-way comparison ===")
    print(f"{'case':24s} {'pages':>7s} {'rounds':>6s} {'bound_ns':>10s} {'est_ns':>10s} "
          f"{'mqsim_ns':>10s} {'hbf_match':>10s} {'hbf_nat':>10s} {'hbf/mq':>7s}")

    for name, flags in PHASE2A_CASES:
        case_dir = tmp_root / "phase2a" / name
        # 1. Run `hbf check` sequentially (generates AddressTrace + MQSim + Rounds).
        _main_run([*flags, "--output", str(case_dir)], force_mqsim=True)

        cfg = _cfg_from_params_json(case_dir / "parameters.json")
        trace = read_trace(case_dir)

        # Extract step >= 0 read chunks (exclude step == -1 prefill writes) with full resident span.
        step0_chunks = [c for c in trace.chunks if c.step >= 0]
        layout = trace.meta.get("layout") or {}
        res_hi = int(layout.get("base", 0)) + int(layout.get("span", 0))
        if not res_hi:
            max_lpa = max((int(c.lpa.max()) for c in step0_chunks if len(c)), default=0)
            res_hi = max_lpa + 1
        step0_meta = {**trace.meta, "resident": [[0, res_hi]]}
        read_trace_obj = AddressTrace(meta=step0_meta, chunks=step0_chunks)

        # 2. Rounds model directly on the identical read AddressTrace.
        _, r_steps = analyze_trace(cfg, step0_chunks, groups=(1,), protocol=None, timing=True)
        r_step = r_steps[0]

        # 3. MQSim step-0 duration from steps.csv.
        with (case_dir / "steps.csv").open() as f:
            mq_steps = list(csv.DictReader(f))
        mq_s0 = next(r for r in mq_steps if int(r["step"]) == 0)
        mqsim_ns = float(mq_s0["mqsim_ns"])

        # 4a. HBFSim with internal channel BW matched to cfg.aggregate_gb_s.
        geom = hbfsim_geometry(cfg)
        converted = convert_trace(cfg, read_trace_obj, geom=geom)
        _, _, h_steps_m, h_stats_m = simulate_batches(
            cfg,
            converted,
            case_dir / "hbfsim_matched",
            binary=hbfsim_bin,
            die_blocking=False,
            match_core_bw=True,
        )
        hbf_match_ns = float(h_steps_m[0]["hbfsim_ns"])

        # 4b. HBFSim with native speed-grade internal channel BW (101.25 GB/s = 96 GB/s payload/ch).
        _, _, h_steps_n, _ = simulate_batches(
            cfg,
            converted,
            case_dir / "hbfsim_native",
            binary=hbfsim_bin,
            die_blocking=False,
            match_core_bw=False,
        )
        hbf_nat_ns = float(h_steps_n[0]["hbfsim_ns"])

        est_batches_ns = float(r_step["est_batches_ns"])
        bound_ns = float(r_step["bound_ns"])
        pages = int(r_step["pages"])
        rounds = int(r_step["rounds"])
        ratio_match_mq = hbf_match_ns / mqsim_ns if mqsim_ns else 0.0
        ratio_match_bound = hbf_match_ns / bound_ns if bound_ns else 0.0
        ratio_mq_bound = mqsim_ns / bound_ns if bound_ns else 0.0

        row = {
            "case": name,
            "aggregate_gb_s": cfg.aggregate_gb_s,
            "read_ns": cfg.read_ns,
            "speed_grade": geom.speed_grade,
            "hbfsim_stacks": geom.stacks,
            "pages": pages,
            "rounds": rounds,
            "subbatches": len(converted.batches),
            "rounds_bound_ns": round(bound_ns, 2),
            "rounds_est_batches_ns": round(est_batches_ns, 2),
            "mqsim_ns": round(mqsim_ns, 2),
            "hbfsim_matched_ns": round(hbf_match_ns, 2),
            "hbfsim_native_ns": round(hbf_nat_ns, 2),
            "hbfsim_read_buffer_hits": h_stats_m["total_read_buffer_hits"],
            "hbfsim_over_mqsim": round(ratio_match_mq, 4),
            "hbfsim_over_bound": round(ratio_match_bound, 4),
            "mqsim_over_bound": round(ratio_mq_bound, 4),
            "hbfsim_wall_s": round(h_stats_m["wall_seconds"], 3),
        }
        rows.append(row)
        print(
            f"{name:24s} {pages:7d} {rounds:6d} {bound_ns:10.0f} {est_batches_ns:10.0f} "
            f"{mqsim_ns:10.0f} {hbf_match_ns:10.1f} {hbf_nat_ns:10.1f} {ratio_match_mq:7.3f}"
        )

    # Extra deep diagnostic breakdown on dense-128k-1024 (1 layer = 131,072 pages, 16 subbatches vs 1 unsplit)
    _print_dense_diagnostic(hbfsim_bin, tmp_root)
    return rows


def _print_dense_diagnostic(hbfsim_bin: Path, tmp_root: Path) -> None:
    """Decompose HBFSim's dense-128k-1024 read latency stage-by-stage."""
    case_dir = tmp_root / "phase2a" / "dense-128k-1024"
    cfg = _cfg_from_params_json(case_dir / "parameters.json")
    trace = read_trace(case_dir)
    layer0 = next(c for c in trace.chunks if c.step == 0 and c.layer == 0)
    geom = hbfsim_geometry(cfg)
    layout = trace.meta.get("layout") or {}
    res_hi = int(layout.get("base", 0)) + int(layout.get("span", 0))
    conv = convert_trace(cfg, AddressTrace(meta={**trace.meta, "resident": [[0, res_hi]]}, chunks=[layer0]), geom=geom)

    print("\n=== Diagnostic breakdown: dense-128k-1024 layer 0 (131,072 pages = 512 MiB, 128 rounds) ===")
    for label, match_bw, combine_subbatches in [
        ("matched_bw_subbatched(8192)", True, False),
        ("matched_bw_unsplit(131072)", True, True),
        ("native_bw_subbatched(8192)", False, False),
        ("native_bw_unsplit(131072)", False, True),
    ]:
        diag_dir = tmp_root / "diag" / label
        cfg_path = diag_dir / "hbfsim.cfg"
        write_config(cfg, cfg_path, geom=geom, match_core_bw=match_bw)
        with HbfSimSession(
            cfg_path,
            diag_dir,
            binary=hbfsim_bin,
            enable_hbm=False,
            enable_hbf=True,
            static_blocks_per_plane=conv.resident_blocks_per_plane,
        ) as session:
            total_ns = 0.0
            agg = {}
            batches = conv.batches
            if combine_subbatches:
                all_reqs = [r for b in batches for r in b.requests]
                batches = [type(batches[0])(0, 0, 0, 0, tuple(all_reqs))]
            for b_idx, batch in enumerate(batches):
                txs = [(f"d{b_idx}/{i}", "HBF_PHYSICAL", r.op, r.addr, r.bytes, 0.0) for i, r in enumerate(batch.requests)]
                rec = session.run_batch(f"diag_{b_idx}", txs)
                d = extract_hbf_delta(rec)
                total_ns += d["elapsed_ns"]
                hd = d["hbf_delta"]
                for k in (
                    "page_reads", "read_buffer_hits", "read_buffer_misses",
                    "page_read_admission_waited_pages", "page_read_admission_wait_work_ns",
                    "hb_io_data_busy_ns", "ecc_decode_ops", "ecc_decode_queue_wait_ns",
                ):
                    if k in hd:
                        agg[k] = agg.get(k, 0.0) + float(hd[k])
            print(
                f"  {label:30s}: layer0={total_ns:9.1f} ns (2 layers={2*total_ns:10.1f} ns) "
                f"reads={int(agg.get('page_reads', 0))} rb_hits={int(agg.get('read_buffer_hits', 0))} "
                f"q_waits={int(agg.get('page_read_admission_waited_pages', 0))}"
            )


def _mqsim_read_durs(comp_path: Path, read_batches: list[Batch]) -> list[int]:
    by_arrival: dict[int, int] = {}
    for c in mqsim_completions(comp_path):
        if c.op == "R":
            cur = by_arrival.get(c.arrival_ns, 0)
            if c.complete_ns > cur:
                by_arrival[c.arrival_ns] = c.complete_ns
    return [by_arrival[b.arrival_ns] - b.arrival_ns for b in read_batches]


def _run_hbfsim_interference_case(
    hbfsim_bin: Path,
    cfg,
    spec: dict,
    prefill: list[Batch],
    ro_reads: list[Batch],
    ww_events: list[Batch],
    work_dir: Path,
    *,
    die_blocking: bool,
) -> tuple[float, float, int, int]:
    """Run read-only and with-writes step-0 timelines in HBFSim; return (ro_ns, extra_ns, progs, erases)."""
    geom = hbfsim_geometry(cfg)
    cfg_path = work_dir / "hbfsim.cfg"
    # Pre-erased free pool: first write to a clean block skips auto-erase (matching MQSim),
    # whereas rewriting block 1 page 0 after prefill + ZONE_INVALIDATE + ZONE_RESET triggers auto-erase!
    write_config(
        cfg,
        cfg_path,
        geom=geom,
        die_blocking=die_blocking,
        match_core_bw=True,
        initial_free_blocks_erased=True,
    )

    # Convert prefill, read-only, and with-writes batches to physical byte addresses.
    conv_prefill = batches_to_converted(cfg, prefill, geom=geom)
    conv_ro = batches_to_converted(cfg, ro_reads, geom=geom)
    conv_ww = batches_to_converted(cfg, ww_events, geom=geom)

    # Determine which physical blocks (zone_size_blocks=1 -> zone = local block index inside channel)
    # need ZONE_INVALIDATE + ZONE_RESET after prefill so step-0 page-0 rewrites trigger auto-erase.
    erase_waves = spec["erase_waves"]
    wave_banks = spec["wave_banks"]
    ppb = cfg.pages_per_block
    units = cfg.units
    reset_zones: list[tuple[int, int, int]] = []
    if erase_waves > 0:
        seen = set()
        for e in range(erase_waves):
            blk = 1 + e
            for u in range(wave_banks):
                lpa = u + units * (blk * ppb)
                s, ch, d, p, b, _ = physical_coords(cfg, lpa, geom=geom)
                # Zone index within (stack, channel) when zone_size_blocks == 1:
                zone_in_ch = ((d * geom.planes_per_die + p) * geom.blocks_per_plane) + b
                key = (s, ch, zone_in_ch)
                if key not in seen:
                    seen.add(key)
                    reset_zones.append(key)

    def _exec(events_conv, sub_dir: Path, do_reset: bool):
        with HbfSimSession(
            cfg_path,
            sub_dir,
            binary=hbfsim_bin,
            enable_hbm=False,
            enable_hbf=True,
            published_blocks_per_plane=geom.blocks_per_plane,
        ) as session:
            # 1. Execute prefill batches so read blocks (block 0) and erase targets (block 1..) are written.
            for idx, b in enumerate(conv_prefill.batches):
                txs = [(f"pf{idx}/{i}", "HBF_PHYSICAL", r.op, r.addr, r.bytes, 0.0) for i, r in enumerate(b.requests)]
                session.run_batch(f"prefill_{idx}", txs)
            # 2. If rewriting erase blocks in step 0, invalidate & reset their zones (0 ns, 0 erases).
            if do_reset:
                for z_i, (s, ch, z) in enumerate(reset_zones):
                    session.zone_command("ZONE_INVALIDATE", f"zi_{z_i}", stack=s, channel=ch, zone=z)
                    session.zone_command("ZONE_RESET", f"zr_{z_i}", stack=s, channel=ch, zone=z)
            # 3. Execute entire step-0 timeline inside ONE batch with exact relative issue_ns!
            base_t = min(b.arrival_ns for b in events_conv.batches)
            txs = []
            read_tx_layer: dict[str, tuple[int, float]] = {}
            for b_idx, b in enumerate(events_conv.batches):
                rel = float(b.arrival_ns - base_t)
                for i, r in enumerate(b.requests):
                    tid = f"s0_{b_idx}_{i}"
                    txs.append((tid, "HBF_PHYSICAL", r.op, r.addr, r.bytes, rel))
                    if r.op == "R":
                        read_tx_layer[tid] = (b.layer, rel)
            rec = session.run_batch("step0", txs, completions=True)
            origin = float(rec["batch_origin_ns"])
            layer_max: dict[int, float] = {}
            layer_arr: dict[int, float] = {}
            for c in rec["transaction_completions"]:
                info = read_tx_layer.get(c["id"])
                if info is not None:
                    lyr, rel = info
                    arr = origin + rel
                    layer_arr[lyr] = arr
                    fin = float(c["finish_ns"])
                    if fin > layer_max.get(lyr, 0.0):
                        layer_max[lyr] = fin
            durs = [layer_max[l] - layer_arr[l] for l in sorted(layer_max)]
            hd = rec["device_delta"]["hbf"]
            return sum(durs), int(hd["raw_physical_programs"]), int(hd["block_erases"])

    ro_ns, _, _ = _exec(conv_ro, work_dir / "ro", do_reset=False)
    ww_ns, progs, erases = _exec(conv_ww, work_dir / "ww", do_reset=bool(reset_zones))
    extra = ww_ns - ro_ns
    if abs(extra) < 1e-6:
        extra = 0.0
    return ro_ns, extra, progs, erases


def run_r5_interference(mqsim_bin: Path, hbfsim_bin: Path, tmp_root: Path) -> list[dict]:
    """Run all 9 R5.3 write-interference points across MQSim, HBFSim (plane & die), and interference.py."""
    rows: list[dict] = []
    print("\n=== Group B: R5.3 9-point write-interference comparison ===")
    print(f"{'point':24s} {'MQ(fifo)':>10s} {'MQ(rf)':>10s} {'HBF(pln)':>10s} "
          f"{'HBF(die)':>10s} {'an(fifo)':>10s} {'an(gaps)':>10s}")

    for spec in R5_POINTS:
        name = spec["name"]
        cfg = load_cfg(
            spec["topology"],
            channels=spec["channels"],
            chips=spec["chips"],
            dies=1,
            planes=spec["planes"],
            blocks_per_plane=spec["blocks"],
            pages_per_block=spec["ppb"],
            aggregate_gb_s=spec["gb_s"],
            program_ns=spec["t_prog"],
            erase_ns=spec["t_bers"],
            coalesce_ns=spec.get("coalesce_ns", 5_000),
            read_plane_ns=spec.get("read_plane_ns", 0),
            cache_read=spec.get("cache_read", True),
        )
        units = cfg.units
        ppb = cfg.pages_per_block
        layers = spec["layers"]
        gap_ns = spec["gap_ns"]
        wave_banks = spec["wave_banks"]
        prog_waves = spec["prog_waves"]
        erase_waves = spec["erase_waves"]

        # Exact same trace construction as scripts/r5_write_interference.py
        t = 0
        prefill: list[Batch] = []
        for p in range(layers):
            reqs = tuple(Request(u + units * p, "W") for u in range(units))
            prefill.append(Batch(-1, -1, len(prefill), t, reqs))
            t += 10_000_000
        for e in range(erase_waves):
            blk = 1 + e
            reqs = tuple(Request(u + units * (blk * ppb), "W") for u in range(wave_banks))
            prefill.append(Batch(-1, -1, len(prefill), t, reqs))
            t += 10_000_000

        pt_dir = tmp_root / "r5" / name
        probe_batches = prefill + [Batch(0, 0, 0, t, tuple(Request(u, "R") for u in range(units)))]
        simulate_mqsim(cfg, probe_batches, pt_dir / "mq-probe", mqsim_bin, hbf_write=True)
        layer_read_solo_ns = _mqsim_read_durs(pt_dir / "mq-probe" / "completions.csv", [probe_batches[-1]])[0]

        step_start = t + 20_000_000
        ro_reads: list[Batch] = []
        ww_events: list[Batch] = []
        write_slots = []
        next_prog_page = 0
        for e in range(erase_waves):
            blk = 1 + e
            write_slots.append(("E", tuple(Request(u + units * (blk * ppb), "W") for u in range(wave_banks))))
        for _ in range(prog_waves - erase_waves):
            blk = 1 + erase_waves + (next_prog_page // ppb)
            pg = next_prog_page % ppb
            next_prog_page += 1
            write_slots.append(("W", tuple(Request(u + units * (blk * ppb + pg), "W") for u in range(wave_banks))))

        by_layer_writes = {l: [] for l in range(layers)}
        for idx, item in enumerate(write_slots):
            target_l = 0 if spec.get("burst_in_layer0") else (idx % layers)
            by_layer_writes[target_l].append(item)

        window_stride = max(gap_ns + layer_read_solo_ns + 50_000, 5_000_000)
        for l in range(layers):
            read_arrival = step_start + l * window_stride + max(gap_ns, 50_000)
            if by_layer_writes[l]:
                lead = gap_ns if gap_ns > 0 else 200
                w_t = read_arrival - lead
                for k_idx, (_, reqs) in enumerate(by_layer_writes[l]):
                    ww_events.append(Batch(0, l, k_idx, w_t + k_idx * 10, reqs))
            r_batch = Batch(0, l, 100, read_arrival, tuple(Request(u + units * l, "R") for u in range(units)))
            ro_reads.append(r_batch)
            ww_events.append(r_batch)
        ww_events.sort(key=lambda b: b.arrival_ns)

        # 1. MQSim read-only, fifo, read_first (sequential!)
        simulate_mqsim(cfg, prefill + ro_reads, pt_dir / "mq-ro", mqsim_bin, hbf_write=True)
        ro_durs = _mqsim_read_durs(pt_dir / "mq-ro" / "completions.csv", ro_reads)
        simulate_mqsim(cfg, prefill + ww_events, pt_dir / "mq-fifo", mqsim_bin, hbf_write=True, write_order="fifo")
        fifo_durs = _mqsim_read_durs(pt_dir / "mq-fifo" / "completions.csv", ro_reads)
        simulate_mqsim(cfg, prefill + ww_events, pt_dir / "mq-rf", mqsim_bin, hbf_write=True, write_order="read_first")
        rf_durs = _mqsim_read_durs(pt_dir / "mq-rf" / "completions.csv", ro_reads)

        mq_ro_ns = sum(ro_durs)
        mq_fifo_extra = sum(a - b for a, b in zip(fifo_durs, ro_durs))
        mq_rf_extra = sum(a - b for a, b in zip(rf_durs, ro_durs))

        # 2. HBFSim default (plane-level blocking)
        hbf_pln_ro, hbf_pln_extra, hbf_pln_progs, hbf_pln_erases = _run_hbfsim_interference_case(
            hbfsim_bin, cfg, spec, prefill, ro_reads, ww_events, pt_dir / "hbf-plane", die_blocking=False
        )

        # 3. HBFSim patched (die-level blocking)
        hbf_die_ro, hbf_die_extra, hbf_die_progs, hbf_die_erases = _run_hbfsim_interference_case(
            hbfsim_bin, cfg, spec, prefill, ro_reads, ww_events, pt_dir / "hbf-die", die_blocking=True
        )

        # 4. Analytic interference.py
        dies = cfg.channels * cfg.chips * cfg.dies
        step_in = {
            "programs": prog_waves * wave_banks,
            "erases": erase_waves * wave_banks,
            "wave_banks": wave_banks,
            "gaps_ns": [gap_ns] * layers,
        }
        an_fifo = interference(step_in, "fifo", cfg.program_ns, cfg.erase_ns, dies=dies, banks_per_die=cfg.planes)
        an_gaps = interference(step_in, "gaps", cfg.program_ns, cfg.erase_ns, dies=dies, banks_per_die=cfg.planes)

        row = {
            "point": name,
            "dies": dies,
            "planes": cfg.planes,
            "layers": layers,
            "t_prog_ns": cfg.program_ns,
            "t_bers_ns": cfg.erase_ns,
            "gap_ns": gap_ns,
            "wave_banks": wave_banks,
            "prog_waves": prog_waves,
            "erase_waves": erase_waves,
            "mqsim_ro_ns": mq_ro_ns,
            "mqsim_fifo_extra_ns": mq_fifo_extra,
            "mqsim_read_first_extra_ns": mq_rf_extra,
            "hbfsim_plane_ro_ns": round(hbf_pln_ro, 2),
            "hbfsim_plane_extra_ns": round(hbf_pln_extra, 2),
            "hbfsim_die_ro_ns": round(hbf_die_ro, 2),
            "hbfsim_die_extra_ns": round(hbf_die_extra, 2),
            "hbfsim_programs": hbf_pln_progs,
            "hbfsim_erases": hbf_pln_erases,
            "analytic_fifo_ns": round(an_fifo["extra_step_ns"]),
            "analytic_gaps_ns": round(an_gaps["extra_step_ns"]),
        }
        rows.append(row)
        print(
            f"{name:24s} {mq_fifo_extra:10d} {mq_rf_extra:10d} {hbf_pln_extra:10.1f} "
            f"{hbf_die_extra:10.1f} {row['analytic_fifo_ns']:10d} {row['analytic_gaps_ns']:10d}"
        )

    # Extra partial-plane diagnostic: 1 plane of die 0 writes while reads hit planes 1..15 of die 0
    _print_partial_plane_diagnostic(hbfsim_bin, tmp_root)
    return rows


def _print_partial_plane_diagnostic(hbfsim_bin: Path, tmp_root: Path) -> None:
    """Contrast plane-level vs die-level blocking when 1 plane writes and other planes on the same die read."""
    cfg = load_cfg("ocp", channels=2, chips=2, dies=1, planes=16, blocks_per_plane=8, pages_per_block=16,
                   aggregate_gb_s=100, program_ns=100_000, erase_ns=1_000_000)
    units = cfg.units
    ppb = cfg.pages_per_block
    prefill = [Batch(-1, -1, 0, 0, tuple(Request(u, "W") for u in range(units)))]
    # Read ONLY planes 1..15 of die 0 (LPAs 1..15) while plane 0 of die 0 writes block 1 page 0!
    read_b = Batch(0, 0, 1, 20_000_000 + 40_000, tuple(Request(u, "R") for u in range(1, 16)))
    write_b = Batch(0, 0, 0, 20_000_000, (Request(0 + units * ppb, "W"),))
    spec = {"erase_waves": 0, "wave_banks": 1}
    pln_ro, pln_ex, _, _ = _run_hbfsim_interference_case(
        hbfsim_bin, cfg, spec, prefill, [read_b], [write_b, read_b],
        tmp_root / "diag_partial_plane_off", die_blocking=False,
    )
    die_ro, die_ex, _, _ = _run_hbfsim_interference_case(
        hbfsim_bin, cfg, spec, prefill, [read_b], [write_b, read_b],
        tmp_root / "diag_partial_plane_on", die_blocking=True,
    )
    print("\n=== Diagnostic: 1-plane write (die 0 plane 0) vs sparse read on same die (planes 1..15), gap=40 us ===")
    print(f"  HBFSim plane-blocking (default): ro={pln_ro:.1f} ns, extra={pln_ex:+.1f} ns")
    print(f"  HBFSim die-blocking   (patched): ro={die_ro:.1f} ns, extra={die_ex:+.1f} ns")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="R9 HBFSim vs MQSim vs Rounds/Interference comparison")
    parser.add_argument("--only", choices=("all", "phase2a", "r5"), default="all")
    args = parser.parse_args()

    hbfsim_bin = build_hbfsim(target="hbfsim")
    mqsim_bin = build_mqsim()

    with tempfile.TemporaryDirectory(prefix="hbf-r9-compare-") as tmp:
        tmp_root = Path(tmp)
        if args.only in ("all", "phase2a"):
            p2a_rows = run_phase2a(hbfsim_bin, tmp_root)
            _write_csv(PHASE2A_CSV, p2a_rows)
        if args.only in ("all", "r5"):
            r5_rows = run_r5_interference(mqsim_bin, hbfsim_bin, tmp_root)
            _write_csv(INTERF_CSV, r5_rows)


if __name__ == "__main__":
    main()
