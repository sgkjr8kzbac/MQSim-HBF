"""Endurance CLI: analytic curves, event simulation, calibration, and CSV summaries."""
import argparse
import csv
import itertools
import json
import math
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from ..device.config import PARAMETERS
from ..frontend.trace.requests import load_workload, per_stack, presets as workload_presets
from ..util import write_table
from .analytic import analytic
from .clock import Clock, ConstantClock, calibrate, layout_rounds
from .geometry import CAUSES, PARTITIONS, POLICIES, Device, geometry_for, sci
from .interference import MODES as M5_MODES
from .report import report_lines, sim_report_lines
from .sim import (REFRESH, SIM_POLICIES, VICTIMS, WEAR, CapacityError, SimOptions, simulate)

METRICS = ("mean_batch", "mean_context", "mean_tpot_ms", "tpot_m5_ms", "max_step_extra_ms", "req_per_s",
           "tokens_per_s", "pages_per_request", "threshold_steps", "refreshes_per_request", "kv_write_gb_s") + tuple(
    f"{cause}_gb_s" for cause in CAUSES) + (
    "program_gb_s", "program_bw_share", "wa", "erase_amp", "erase_wa", "refresh_to_kv", "erases_per_s", "avg_pec_per_day",
    "max_pec_per_day", "pe_per_mtok", "break_even_x", "staging_peak_mib", "pending_units_per_channel",
    "unit_open_s", "replay_buffer_mib", "legal") + (  # event simulation only (empty for --analytic)
    "evicted_gb", "recompute_tokens", "stalls", "free_segments_min", "stranded_frac", "violations_skip",
    "violations_overlap", "violations_live_erase", "legalize_gb", "analytic_avg_pec_per_day",
    "analytic_refresh_to_kv")
SPARSE_MODES = ("dense", "uniform_tokens", "uniform_pages", "sink_recent")


def _list(text, kind=str, name="value"):
    items = [item.strip() for item in str(text).split(",") if item.strip()]
    if not items:
        raise ValueError(f"--{name} needs at least one value")
    try:
        return [kind(item) if kind is not float else float(item) for item in items]
    except ValueError:
        raise ValueError(f"--{name}: cannot parse {text!r}") from None


def build_parser():
    parser = argparse.ArgumentParser(prog="python -m hbf.endurance", description="Endurance of KV cache stored on HBF")
    parser.add_argument("--analytic", action="store_true",
                        help="closed-form steady state instead of the event simulation (which needs --hours)")
    parser.add_argument("--model", default="llama-3.1-8b", help="model preset(s), comma separated")
    parser.add_argument("--gpu", default="b200", help="GPU preset (hbf/configs/gpus.json) or JSON")
    parser.add_argument("--format", dest="kv_format", default="bf16",
                        help="KV format(s), comma separated (hbf/formats.py presets, e.g. bf16,fp8,kivi2)")
    parser.add_argument("--kv-spec",
                        help="KV group spec (hbf/kvspec.py presets, e.g. hybrid:local_tier=hbf, cla:share=2, ssm)")
    parser.add_argument("--mode", choices=SPARSE_MODES, default="dense",
                        help="read selection of a decode step (non-dense modes take ratios from calibration)")
    parser.add_argument("--fraction", type=float, default=.1, help="selected share (uniform_*, sink_recent)")
    parser.add_argument("--select-tokens", type=int, default=16, help="uniform_tokens block (default 16)")
    parser.add_argument("--count", type=int, default=4, help="sink_recent: sink tokens")
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the request stream (simulation) and of the calibration snapshot's selection")
    parser.add_argument("--alloc", choices=POLICIES, default="packed",
                        help="allocation policy: packed (default), dedicated, layer_major (static waves), "
                             "hbm_decode (generated KV stays in HBM), vllm_pool (not legal as specified)")
    parser.add_argument("--partition", choices=tuple(PARTITIONS), default="none",
                        help="segments per request group: none (1,024-bank superblocks), channel (64), die (16)")
    workload = parser.add_argument_group("workload (presets in hbf/configs/workloads.json)")
    workload.add_argument("--workload", default="long32k", help="preset(s), comma separated: "
                          + ", ".join(workload_presets()))
    workload.add_argument("--concurrency", type=int, default=8,
                          help="closed loop: requests (sessions for agentic) in flight per GPU (default 8)")
    workload.add_argument("--rate", type=float, help="Poisson arrivals per second per GPU instead of a closed loop")
    workload.add_argument("--max-batch", type=int, default=256, help="Poisson: running requests at most")
    workload.add_argument("--input-len", help="override the input length (fixed:N, lognormal:median=,sigma=, ...)")
    workload.add_argument("--output-len", help="override the output length")
    workload.add_argument("--history", help="sessions: override the initial history length")
    workload.add_argument("--calls", type=int, help="sessions: calls per session")
    workload.add_argument("--think-s", type=float, help="sessions: seconds between calls")
    workload.add_argument("--prefix-groups", type=int, help="shared-prefix groups")
    workload.add_argument("--prefix-tokens", type=int, help="tokens per shared prefix")
    workload.add_argument("--prefix-share", type=float, help="share of requests that use a shared prefix")
    workload.add_argument("--zipf", type=float, help="Zipf exponent of the prefix popularity")
    workload.add_argument("--retain-s", type=float, help="idle lifetime of finished KV (prefix cache), seconds")
    wear = parser.add_argument_group("endurance")
    wear.add_argument("--rd-threshold", default="1e6",
                      help="read-disturb refresh threshold X in block reads, comma separated (product specific, "
                           "spec section 11.5; params.md sweeps 1e5, 1e6, 1e7)")
    wear.add_argument("--rd-hot", type=float, default=1.0,
                      help="persistent read skew h: max over mean block reads (unknown; default 1)")
    wear.add_argument("--read-share", type=float,
                      help="pages read per stored page per step (default: 1 for dense, else from calibration)")
    wear.add_argument("--pe-limit", default="1e4,1e5,1e6", help="P/E limits N_PE, comma separated")
    wear.add_argument("--retention-h", type=float, default=24.0,
                      help="retention refresh interval in hours (spec section 9: 24 h; 0 turns it off)")
    wear.add_argument("--prefix-reads", choices=("per_request", "once"), default="per_request",
                      help="a shared prefix is read once per sharing request per step, or once per step")
    device = parser.add_argument_group("device (defaults from hbf/configs/parameters.json and the OCP spec)")
    device.add_argument("--pages-per-block", default="256",
                        help="pages per NAND block, comma separated (spec section 11.5.1.8 hints 256/512/1024 "
                             "at 4 KiB; SPLASH and FlashAccel use 192)")
    device.add_argument("--capacity-gib", type=float, default=512.0, help="KV capacity per stack (default 512)")
    device.add_argument("--stacks", type=int, default=1, help="HBF stacks sharing the GPU's requests")
    device.add_argument("--read-ns", type=int, help="tR for the Clock and calibration (default 4096)")
    device.add_argument("--aggregate-gb-s", type=int, help="core-die channel bandwidth for calibration")
    device.add_argument("--program-ns", type=float, help="tPROG (default 100000, SPLASH)")
    device.add_argument("--erase-ns", type=float, help="tBERS (default 1000000, SPLASH)")
    system = parser.add_argument_group("system model and interference")
    system.add_argument("--overlap", default="serial", help="serial, prefetch:<d> or ideal")
    system.add_argument("--kv-consume", choices=("stream", "staged"), default="stream")
    system.add_argument("--m5", choices=M5_MODES + ("off",), default="gaps",
                        help="write/erase interference mode (hbf/interference.py; default gaps)")
    system.add_argument("--suspend-ns", type=float, help="suspend latency for --m5 suspend (product specific)")
    system.add_argument("--block-unit", choices=("die", "bank"), default="die")
    calib = parser.add_argument_group("clock calibration against phase A (analysis-only pipelines)")
    calib.add_argument("--no-calibrate", action="store_true",
                       help="use the F1 constants (valid for the HBF command model at tR 4,096 ns) and kappa 1")
    calib.add_argument("--calibration-dir", type=Path, help="cache of calibration runs (default OUTPUT/calibration)")
    calib.add_argument("--calib-context", type=int,
                       help="context of the calibration runs (default: the workload's mean context, 1K-32K)")
    sim = parser.add_argument_group("event simulation (without --analytic; design-B section 4)")
    sim.add_argument("--hours", type=float, help="simulated hours (required without --analytic)")
    sim.add_argument("--warmup-h", type=float, default=1.0, help="first hours excluded from the rates (default 1)")
    sim.add_argument("--wear", choices=WEAR, default="dynamic",
                     help="free-segment allocation: none (lowest index), dynamic (least PEC, default), "
                          "ideal (as dynamic; max PEC reported as AVGPEC), remap (cyclic zone remap, "
                          "spec section 11.4.1)")
    sim.add_argument("--refresh", choices=REFRESH, default="auto",
                     help="read-disturb and retention refresh: relocate live pages (auto for packed, dedicated, "
                          "hbm_decode, vllm_pool) or replay the segment in place (spec section 5.7; auto for "
                          "layer_major)")
    sim.add_argument("--refresh-idle", choices=("relocate", "evict"), default="relocate",
                     help="idle cached entries in a refreshed segment: relocate (default) or drop them")
    sim.add_argument("--reclaim", choices=("evict", "compact"), default="evict",
                     help="reclaim victims: evict idle entries, copy active pages (default); compact copies all")
    sim.add_argument("--victim", choices=VICTIMS, default="greedy",
                     help="reclaim victim selection: greedy (fewest pages to copy, default) or cost_benefit "
                          "((1 - u) * age / (1 + u))")
    sim.add_argument("--prefill", choices=("disaggregated", "colocated"), default="disaggregated",
                     help="prefill timing: disaggregated (default, no decode delay) or colocated (adds "
                          "input / prefill_tok_s serial time per admission)")
    sim.add_argument("--pre-open", action="store_true",
                     help="pre-open free segments in idle time (takes non-replay erases off M5 interference)")
    sim.add_argument("--separate-prefix", action="store_true",
                     help="place shared prefixes on their own write frontier per partition (hot/cold separation)")
    sim.add_argument("--block-tokens", type=int,
                     help="vllm_pool: tokens per vLLM block (default: lcm of HBF tensors' tokens per page)")
    sim.add_argument("--free-order", default="fresh",
                     help="vllm_pool: initial free queue order: fresh (ascending ids, default) or shuffled:<seed>")
    sim.add_argument("--vllm-reuse", choices=("fifo", "lifo"), default="fifo",
                     help="vllm_pool: freed block-id reuse order (fifo default, or lifo)")
    sim.add_argument("--low-watermark", type=int,
                     help="free segments per partition kept by reclaim (default 2 + one maximal request)")
    sim.add_argument("--report-s", type=float, default=60.0, help="timeseries interval, simulated s (default 60)")
    sim.add_argument("--quantum-s", type=float, default=1.0, help="longest quantum in seconds (default 1)")
    sim.add_argument("--rd-slack", type=float, default=.01,
                     help="shortest quantum as a share of the fastest segment's threshold in steps "
                          "(default 0.01: a refresh is at most 1%% of X late)")
    sim.add_argument("--slots", type=int,
                     help="running requests at most (layer_major: slots per wave; default --concurrency or "
                          "--max-batch)")
    sim.add_argument("--align", choices=("row", "none"), default="row",
                     help="packed: start each new entry on a row (default)")
    sim.add_argument("--no-stagger", action="store_true",
                     help="start closed-loop requests together (default: staggered over one request lifetime)")
    parser.add_argument("--output", type=Path)
    return parser


def configure(args):
    """Run configuration from parsed arguments (raises ValueError on bad input)."""
    from ..models import load_gpu, load_model
    from ..system import SystemOptions
    if not args.analytic:
        if args.hours is None:
            raise ValueError("The event simulation needs --hours (simulated time); use --analytic for the "
                             "closed-form steady state")
        if args.alloc not in SIM_POLICIES:
            raise ValueError(f"--alloc {args.alloc} is not simulated (only {', '.join(SIM_POLICIES)})")
        if args.hours <= 0 or not 0 <= args.warmup_h < args.hours:
            raise ValueError("--hours must be positive and --warmup-h in [0, --hours)")
        if args.report_s <= 0 or args.quantum_s <= 0 or not 0 <= args.rd_slack < 1:
            raise ValueError("--report-s and --quantum-s must be positive and --rd-slack in [0, 1)")
        if (args.low_watermark is not None and args.low_watermark < 1) or (args.slots is not None and args.slots < 1):
            raise ValueError("--low-watermark and --slots must be positive")
        if args.alloc == "layer_major" and args.refresh == "relocate":
            raise ValueError("layer_major regions are refreshed in place: use --refresh replay or auto")
        if args.block_tokens is not None and args.block_tokens < 1:
            raise ValueError("--block-tokens must be positive")
        if args.free_order != "fresh" and not args.free_order.startswith("shuffled:"):
            raise ValueError("--free-order must be fresh or shuffled:<seed>")
        if args.free_order.startswith("shuffled:"):
            try:
                int(args.free_order[9:])
            except ValueError:
                raise ValueError(f"Invalid --free-order {args.free_order!r}: use fresh or shuffled:<int>") from None
    params = json.loads(PARAMETERS.read_text())
    models = [load_model(name) for name in _list(args.model, name="model")]
    formats = _list(args.kv_format, name="format")
    overrides = {"input": args.input_len, "output": args.output_len, "history": args.history, "calls": args.calls,
                 "think_s": args.think_s, "prefix_groups": args.prefix_groups, "prefix_tokens": args.prefix_tokens,
                 "prefix_share": args.prefix_share, "zipf": args.zipf, "retain_s": args.retain_s,
                 "max_batch": args.max_batch, "concurrency": args.concurrency, "seed": args.seed}
    if args.rate is not None:
        overrides |= {"arrival": "poisson", "rate": args.rate}
    workloads = [load_workload(name, **overrides) for name in _list(args.workload, name="workload")]
    ppbs = _list(args.pages_per_block, int, "pages-per-block")
    xs = _list(args.rd_threshold, float, "rd-threshold")
    limits = _list(args.pe_limit, float, "pe-limit")
    if any(value < 1 for value in ppbs) or any(value <= 0 for value in xs + limits):
        raise ValueError("pages per block, X and P/E limits must be positive")
    if type(args.stacks) is not int or args.stacks < 1 or args.capacity_gib <= 0 or args.rd_hot <= 0:
        raise ValueError("--stacks, --capacity-gib and --rd-hot must be positive")
    if args.retention_h < 0:
        raise ValueError("--retention-h must be non-negative")
    if args.read_share is not None and not 0 < args.read_share <= 1:
        raise ValueError("--read-share must be in (0, 1]")
    if args.partition != "none" and args.alloc not in ("packed", "dedicated", "hbm_decode"):
        raise ValueError("--partition applies to packed, dedicated and hbm_decode")
    if args.m5 == "suspend" and args.suspend_ns is None:
        raise ValueError("--m5 suspend needs --suspend-ns (OCP v0.7.0 has no suspend; the value is product specific)")
    val_ctx = _calib_context(workloads[0], args.calib_context)
    for model in models:
        for fmt in formats:
            geo, _ = geometry_for(model, fmt, kv_spec=args.kv_spec, context=val_ctx)
            if not args.analytic and args.alloc == "vllm_pool":
                if hasattr(geo, "is_uniform") and not geo.is_uniform:
                    raise ValueError(f"--alloc vllm_pool requires uniform layers (not {model.name} with --kv-spec)")
                if args.block_tokens is not None and any(args.block_tokens % tpp for tpp in geo.tpp):
                    raise ValueError(f"--block-tokens ({args.block_tokens}) must be a multiple of every HBF "
                                     f"tensor's tokens per page ({geo.tpp}) for {model.name}/{fmt}")
    device_flags = []
    for flag, value in (("--read-ns", args.read_ns), ("--aggregate-gb-s", args.aggregate_gb_s)):
        if value is not None:
            device_flags += [flag, str(value)]
    generation = {}
    if args.mode != "dense":
        generation = {"fraction": args.fraction, "select_tokens": args.select_tokens, "count": args.count,
                      "seed": args.seed}
    return {"models": models, "gpu": load_gpu(args.gpu), "gpu_name": args.gpu, "formats": formats,
            "kv_spec": args.kv_spec,
            "mode": args.mode, "generation": generation, "policy": args.alloc, "partition": args.partition,
            "workloads": workloads, "pages_per_block": ppbs, "x_values": xs, "pe_limits": limits,
            "hot": args.rd_hot, "read_share": args.read_share,
            "retention_s": args.retention_h * 3600 if args.retention_h else 0.0,
            "prefix_reads": args.prefix_reads, "capacity_gib": args.capacity_gib, "stacks": args.stacks,
            "read_ns": args.read_ns or params["read_ns"], "device_flags": device_flags,
            "t_prog": args.program_ns or params["program_ns"], "t_bers": args.erase_ns or params["erase_ns"],
            "options": SystemOptions(overlap=args.overlap, kv_consume=args.kv_consume), "m5": args.m5,
            "suspend_ns": args.suspend_ns, "block_unit": args.block_unit, "calibrate": not args.no_calibrate,
            "calibration_dir": args.calibration_dir, "calib_context": args.calib_context,
            "analytic": args.analytic, "hours": args.hours, "warmup_h": args.warmup_h, "wear": args.wear,
            "refresh": args.refresh, "refresh_idle": args.refresh_idle, "reclaim": args.reclaim,
            "victim": args.victim, "prefill": args.prefill, "pre_open": args.pre_open,
            "separate_prefix": args.separate_prefix, "block_tokens": args.block_tokens,
            "free_order": args.free_order, "vllm_reuse": args.vllm_reuse,
            "low_watermark": args.low_watermark, "report_s": args.report_s, "quantum_s": args.quantum_s,
            "rd_slack": args.rd_slack, "slots": args.slots, "align": args.align, "stagger": not args.no_stagger}


def _calib_context(workload, given):
    if given:
        return given
    if workload.sessions:
        mean = workload.history.mean() + workload.calls / 2 * (workload.input.mean() + workload.output.mean())
    else:
        mean = workload.input.mean() + workload.prefix_share * workload.prefix_tokens + workload.output.mean() / 2
    return int(min(32768, max(1024, 2 ** math.ceil(math.log2(max(mean, 1))))))


def snapshot_batch(workload, stacks):
    """Running requests per stack for the calibration snapshot (kappa depends on the batch)."""
    if workload.arrival != "closed" or workload.sessions:
        return None
    return max(1, math.ceil(workload.concurrency / stacks))


def make_clock(config, model, kv_format, workload, directory):
    """Clock for one (model, format, workload): calibrated against phase A, or the F1 constants."""
    kv_spec = config.get("kv_spec")
    ctx = _calib_context(workload, config.get("calib_context"))
    geometry, fmt = geometry_for(model, kv_format, kv_spec=kv_spec, context=ctx)
    selection = "static" if config["mode"] in ("dense", "sink_recent") else "dynamic"
    base = {"read_ns": config["read_ns"], "stacks": config["stacks"], "selection": selection,
            "kv_format": fmt.name, "options": config["options"]}
    if not config["calibrate"]:
        rounds_fn = (layout_rounds(model, kv_format)
                     if config["policy"] == "layer_major" and not hasattr(geometry, "spec") else None)
        return Clock(model, config["gpu"], geometry, rounds_fn=rounds_fn, **base), geometry, None
    cache = config["calibration_dir"] or Path(directory) / "calibration"
    calib = calibrate(cache, model.name, config["gpu_name"], kv_format, mode=config["mode"],
                      generation=config["generation"], context=ctx,
                      device=config["device_flags"], policy=config["policy"], partition=config["partition"],
                      snapshot_batch=snapshot_batch(workload, config["stacks"]), kv_spec=kv_spec)
    clock = Clock(model, config["gpu"], geometry, layer_ns=calib["layer_ns"], round_ns=calib["round_ns"],
                  balance=calib["balance"], ratios=calib["ratios"], **base)
    return clock, geometry, calib


def run(directory, **config):
    """Write meta.json, calibration.json, endurance_summary.csv and report.md into ``directory``; returns rows.

    With ``analytic`` false (no ``--analytic``) every (model, format, workload, ppb, X) is simulated
    (``simulate``) and endurance_timeseries.csv and endurance_segments.csv are written too.
    """
    from ..util import write_table
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    analytic_mode = config.get("analytic", True)
    if not analytic_mode:
        rows, calibrations = _simulation_run(directory, config)
    else:
        rows, calibrations = _analytic_rows(directory, config)
    write_table(directory / "endurance_summary.csv", rows)
    meta = {key: value for key, value in config.items()
            if key not in ("models", "gpu", "workloads", "options", "calibration_dir")}
    meta |= {"models": [model.name for model in config["models"]], "gpu": config["gpu_name"],
             "workloads": [{"name": w.name, "description": w.describe(), "source": w.source}
                           for w in config["workloads"]],
             "options": {"overlap": config["options"].overlap, "kv_consume": config["options"].kv_consume},
             "tool": "endurance", "analytic": analytic_mode, "argv": sys.argv}
    (directory / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str) + "\n")
    if calibrations:
        (directory / "calibration.json").write_text(json.dumps(calibrations, indent=2) + "\n")
    lines = report_lines(rows, config, calibrations) if analytic_mode else sim_report_lines(rows, config, calibrations)
    (directory / "report.md").write_text("\n".join(lines) + "\n")
    return rows


def _read_share(config, clock, calib):
    """Pages read per stored page per step: --read-share, the calibration's, or the sparse fraction."""
    share = config["read_share"] or (calib["read_share"] if calib else 1.0)
    if config["mode"] != "dense" and calib is None and config["read_share"] is None:
        share = config["generation"].get("fraction", 1.0)
        clock.ratios = {"pages": share}
    return share


def _simulation_run(directory, config):
    """Simulate every combination; writes the timeseries and segment CSVs, returns (summary rows, calibrations)."""
    from ..util import write_table
    rows, calibrations, series, segments = [], [], [], []
    for model, kv_format, workload in itertools.product(config["models"], config["formats"], config["workloads"]):
        clock, geometry, calib = make_clock(config, model, kv_format, workload, directory)
        if calib is not None:
            calibrations.append({"model": model.name, "format": kv_format, "workload": workload.name} | calib)
        share = _read_share(config, clock, calib)
        share_of_stack = per_stack(workload, config["stacks"])
        for ppb, x in itertools.product(config["pages_per_block"], config["x_values"]):
            device = Device(pages_per_block=ppb, capacity_gib=config["capacity_gib"], partition=config["partition"])
            options = SimOptions(
                policy=config["policy"], x=x, hot=config["hot"], share=share, retention_s=config["retention_s"],
                prefix_reads=config["prefix_reads"], hours=config["hours"], warmup_h=config["warmup_h"],
                quantum_s=config["quantum_s"], report_s=config["report_s"], rd_slack=config["rd_slack"],
                wear=config["wear"], refresh=config["refresh"], refresh_idle=config["refresh_idle"],
                reclaim=config["reclaim"], victim=config.get("victim", "greedy"),
                prefill=config.get("prefill", "disaggregated"), pre_open=config.get("pre_open", False),
                separate_prefix=config.get("separate_prefix", False),
                block_tokens=config.get("block_tokens"), free_order=config.get("free_order", "fresh"),
                vllm_reuse=config.get("vllm_reuse", "fifo"),
                low_watermark=config["low_watermark"], align=config["align"],
                slots=config["slots"], stagger=config["stagger"], stacks=config["stacks"], m5=config["m5"],
                t_prog=config["t_prog"], t_bers=config["t_bers"], suspend_ns=config["suspend_ns"],
                block_unit=config["block_unit"], pe_limits=tuple(config["pe_limits"]))
            sim = simulate(geometry, clock, share_of_stack, device, options, model=model, gpu=config["gpu"])
            try:  # the closed form at the same settings, for comparison
                ref = analytic(geometry, clock, workload, device, policy=config["policy"], x_values=(x,),
                               pe_limits=config["pe_limits"], share=share, hot=config["hot"],
                               retention_s=config["retention_s"], prefix_reads=config["prefix_reads"],
                               t_prog=config["t_prog"], t_bers=config["t_bers"], m5=config["m5"],
                               suspend_ns=config["suspend_ns"], block_unit=config["block_unit"],
                               stacks=config["stacks"], align=config["align"])[0]
            except (ValueError, ZeroDivisionError):
                ref = {}
            key = {"model": model.name, "format": kv_format, "mode": config["mode"], "read_share": share}
            rows.append(key | sim.result | {name: ref.get(name[len("analytic_"):]) for name in
                                            ("analytic_avg_pec_per_day", "analytic_refresh_to_kv",
                                             "analytic_kv_write_gb_s", "analytic_req_per_s")})
            tag = {"model": model.name, "format": kv_format, "workload": workload.name, "pages_per_block": ppb, "x": x}
            series += [tag | row for row in sim.timeseries]
            segments += [tag | row for row in sim.segment_rows()]
    write_table(directory / "endurance_timeseries.csv", series)
    write_table(directory / "endurance_segments.csv", segments)
    return rows, calibrations


def _analytic_rows(directory, config):
    """Closed-form rows of every combination (``analytic``); returns (rows, calibrations)."""
    rows, calibrations = [], []
    for model, kv_format, workload in itertools.product(config["models"], config["formats"], config["workloads"]):
        clock, geometry, calib = make_clock(config, model, kv_format, workload, directory)
        if calib is not None:
            calibrations.append({"model": model.name, "format": kv_format, "workload": workload.name} | calib)
        share = _read_share(config, clock, calib)
        for ppb in config["pages_per_block"]:
            device = Device(pages_per_block=ppb, capacity_gib=config["capacity_gib"], partition=config["partition"])
            for row in analytic(geometry, clock, workload, device, policy=config["policy"],
                                x_values=config["x_values"], pe_limits=config["pe_limits"], share=share,
                                hot=config["hot"], retention_s=config["retention_s"],
                                prefix_reads=config["prefix_reads"], t_prog=config["t_prog"],
                                t_bers=config["t_bers"], m5=config["m5"], suspend_ns=config["suspend_ns"],
                                block_unit=config["block_unit"], stacks=config["stacks"], align=config["align"]):
                rows.append({"model": model.name, "format": kv_format, "mode": config["mode"],
                             "read_share": share} | row)
    return rows, calibrations


def summarize_run(directory, head_dim=None):
    """METRICS of the first row of endurance_summary.csv (``head_dim`` is accepted for hbf.sweep)."""
    path = Path(directory) / "endurance_summary.csv"
    with path.open(newline="") as stream:
        first = next(csv.DictReader(stream), None)
    if first is None:
        return dict.fromkeys(METRICS)

    def parse(text):
        if text in (None, ""):
            return None
        if text in ("True", "False"):
            return text == "True"
        try:
            return float(text)
        except ValueError:
            return text
    return {key: parse(first.get(key)) for key in METRICS}



def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    directory = args.output or Path("runs") / f"endurance-{now}-{uuid4().hex[:8]}"
    try:
        config = configure(args)
        run(directory, **config)
    except (OSError, ValueError, CapacityError) as error:
        parser.error(str(error))
    print(Path(directory).resolve())


if __name__ == "__main__":
    main()
