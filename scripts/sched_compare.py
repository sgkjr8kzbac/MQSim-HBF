#!/usr/bin/env python3
"""Compare our simplified request scheduler (hbf/lifetime/sim) against ServeLoop (vLLM 0.26 FCFS).

Zero modifications to core hbf/ behavior: all observation hooks, ServeLoop adapters, and
HBF wear replay for ServeLoop schedules live exclusively in this evaluation script.

Outputs:
- ``evidence/sched-compare.csv``: full metric table across all scenarios (ample & tight capacity,
  no-prefix & prefix-sharing, default & matched timing, plus ServeLoop cold-HBF & tight-HBM diagnostics).
- ``.build/serveloop/sched_compare_summary.json``: detailed auxiliary data (HBFSim calibration check,
  binned write-rate curves, per-scenario diagnostics) used to draft ``docs/design/sched-compare.md``.
"""
from __future__ import annotations

from collections import defaultdict
import csv
from dataclasses import dataclass
import heapq
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mooncake_to_serveloop import (
    build_serveloop_trace_dict,
    select_mooncake_slice,
)

from hbf.frontend.trace.requests import RequestRecord, scripted
from hbf.lifetime.geometry import CAUSES, DAY, GIB, Device, geometry_for, resolve_prefill_tok_s
from hbf.lifetime.sim.engine import Simulator
from hbf.lifetime.sim.options import ACTIVE, IDLE, OPEN, FULL, PREFIX, PRIVATE, SimOptions
from hbf.models import load_model
from hbf.registry import REGISTRY, load_extensions

OUT_CSV = ROOT / "evidence" / "sched-compare.csv"
AUX_JSON = ROOT / ".build" / "serveloop" / "sched_compare_summary.json"
MOONCAKE_JSONL = ROOT / ".build" / "traces" / "mooncake" / "conversation_trace.jsonl"

# Experiment parameters (documented verbatim in docs/design/sched-compare.md).
SHARD_STACKS = 3        # Prefix-preserving 1/3 shard -> ~1.04 req/s, unsaturated on 1x 512 GiB OCP stack
MAX_REQUESTS = 1200     # ~19.3 minutes of real Mooncake traffic
MAX_BATCH_SLOTS = 64    # Aligned concurrency cap (ours: slots=64, ServeLoop: max_batch_requests=64)
SL_MAX_BATCH_TOKENS = 8192
SL_PREFILL_CHUNK = 2048
PREFIX_TTL_S = 300.0
AMPLE_CAP_GIB = 512.0
TIGHT_CAP_GIB = 120.0
TIGHT_SL_MATCHED_GIB = 64.0
TIGHT_SL_DEFAULT_GIB = 24.0
KV_BYTES_PER_TOKEN = 131072  # Llama-3.1-8B BF16: 32 * 2 * 8 * 128 * 2
PAGE_BYTES = 4096
PPB = 256
BANKS = 1024
# HBM4 effective bandwidth in ServeLoop configs/systems/4hbm-4hbf.cfg:
# 4 stacks * 32 ch * 2 pc * 4 B/pc * 8 GT/s * 0.94 eff = 7,700.48 GB/s (calibrated vs HbfSimExecutor).
HBM4_EFF_BW_B_PER_NS = 4 * 32 * 2 * 4 * 8 * 0.94


def ensure_external_tools() -> tuple[Path, Path]:
    """Ensure pinned ServeLoop checkout and HBFSim binary exist via their setup scripts."""
    sl_dir = Path(
        subprocess.check_output(["bash", str(ROOT / "scripts/serveloop_build.sh")], text=True).strip()
    )
    hbf_build = Path(
        subprocess.check_output(["bash", str(ROOT / "scripts/hbfsim_build.sh")], text=True).strip()
    )
    sim_bin = hbf_build / "hbfsim"
    if str(sl_dir) not in sys.path:
        sys.path.insert(0, str(sl_dir))
    return sl_dir, sim_bin


def _pct(values: list[float] | np.ndarray, q: float) -> float:
    if len(values) == 0:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _write_rate_stats(write_events: list[tuple[float, float]], span_s: float, bin_s: float) -> tuple[float, float, float]:
    """Return (mean_gb_s, peak_gb_s, peak_to_mean) binned over ``bin_s`` seconds (decimal GB/s)."""
    if not write_events or span_s <= 0:
        return 0.0, 0.0, 0.0
    n_bins = max(1, int(math.ceil(span_s / bin_s)))
    bins = np.zeros(n_bins, dtype=np.float64)
    for t_s, n_bytes in write_events:
        idx = min(n_bins - 1, max(0, int(t_s // bin_s)))
        bins[idx] += n_bytes
    rates_gb_s = bins / (bin_s * 1e9)
    mean_r = float(rates_gb_s.mean())
    peak_r = float(rates_gb_s.max())
    p2m = peak_r / mean_r if mean_r > 0 else 0.0
    return mean_r, peak_r, p2m


# ---------------------------------------------------------------------------
# 1. HBFSim calibration check for ServeLoop's default timing (all-hbm)
# ---------------------------------------------------------------------------

def calibrate_serveloop_hbfsim(sl_dir: Path, sim_bin: Path, sel_raw: list[dict], mc_sha: str) -> dict[str, Any]:
    """Run ServeLoop's real HbfSimExecutor on a 6-request Mooncake probe to verify our timing model."""
    from hbfsim_client.simulation_session import ResolvedSystemConfig
    from serveloop.catalog import load_model_any
    from serveloop.contracts import BatchSlice, RequestSpec, RequestTrace, ScheduledBatch, TraceProvenance
    from serveloop.io import run_config_from_dict
    from serveloop.run import _artifact, default_run_config, derive_placement, run_experiment

    model = load_model_any(sl_dir / "models/llama31-8b-w8-kv-bf16.json")
    sys_cfg_path = sl_dir / "configs/systems/4hbm-4hbf.cfg"
    reqs6 = [
        RequestSpec(f"p_{i:02d}", float(i * 5e6), model.model_id, int(sel_raw[i]["input_length"]), 20, None)
        for i in range(6)
    ]
    trace6 = RequestTrace(TraceProvenance("production_trace", "mooncake_calib_6", mc_sha), tuple(reqs6))
    sys_cfg = ResolvedSystemConfig.load([sys_cfg_path]).resolve(sim_bin, enable_hbf=False)
    placement = derive_placement(system=sys_cfg, models={model.model_id: model}, preset="all-hbm")
    policy, timing, prefetch = run_config_from_dict(
        default_run_config(timing="roofline", peak_tflops=989.0, efficiency=0.6, prefetch_depth=1)
    )
    arts = {"simulator": _artifact(sim_bin), "system_configs": [_artifact(sys_cfg_path)]}
    t0 = time.perf_counter()
    exp = run_experiment(
        models={model.model_id: model}, request_trace=trace6, router=None,
        system_config_paths=[sys_cfg_path], placement_spec=placement, policy=policy,
        timing=timing, prefetch_depth=prefetch, simulator=sim_bin, input_artifacts=arts,
    )
    wall_s = time.perf_counter() - t0
    by_kind: dict[str, list[tuple[float, float]]] = defaultdict(list)
    w_per_layer = model.weight_footprint_bytes / model.num_layers
    kv_per_tok_layer = model.kv_bytes_per_token / model.num_layers
    for b in exp["serveloop"]["batches"]:
        actual_ns = float(b["finish_ns"] - b["batch_origin_ns"])
        sb = ScheduledBatch(
            batch_id=b["batch_id"], model_id=model.model_id,
            slices=tuple(
                BatchSlice(s["request_id"], s["token_begin"], s["token_count"],
                           s["context_tokens_before"], s["emits_output"], s["phase"])
                for s in b["schedule"]["slices"]
            ),
            not_before_ns=b["batch_origin_ns"],
        )
        bt = timing.timing_for(model=model, batch=sb)
        kv_r = sum(s.context_tokens_before + s.token_count for s in sb.slices) * kv_per_tok_layer
        kv_w = sum(s.token_count for s in sb.slices) * kv_per_tok_layer
        mem_ns = (w_per_layer + kv_r + kv_w) / HBM4_EFF_BW_B_PER_NS
        est_ns = sum(max(c_ns, mem_ns) for c_ns in bt.layer_ns) + bt.tail_ns
        by_kind[b["kind"]].append((actual_ns, est_ns))
    summary = {"wall_s": round(wall_s, 2), "iterations": exp["serveloop"]["summary"]["iterations"], "kinds": {}}
    for kind, pairs in by_kind.items():
        ma = sum(a for a, _ in pairs) / len(pairs)
        me = sum(e for _, e in pairs) / len(pairs)
        summary["kinds"][kind] = {
            "count": len(pairs),
            "hbfsim_mean_ms": round(ma / 1e6, 3),
            "analytic_mean_ms": round(me / 1e6, 3),
            "rel_err_pct": round(100.0 * (me - ma) / ma, 2),
        }
    return summary


# ---------------------------------------------------------------------------
# 2. Observer subclass of our Simulator (read-only instrumentation; core untouched)
# ---------------------------------------------------------------------------

class ObservedSimulator(Simulator):
    """Subclasses hbf.lifetime.sim.engine.Simulator solely to log per-step & per-request metrics."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obs_batch_sizes: list[int] = []
        self.obs_batch_weights: list[int] = []
        self.obs_prefill_toks: list[int] = []
        self.obs_prefill_iters: int = 0
        self.obs_decode_iters: int = 0
        self.obs_write_events: list[tuple[float, float]] = []  # (t_s, bytes)
        self.obs_alloc_events: int = 0
        self.obs_release_events: int = 0
        self.obs_e2e_s: list[float] = []
        self.obs_tpot_ms: list[float] = []
        self.obs_tpot_weights: list[int] = []
        self.obs_prefix_hit_tokens: int = 0
        self.obs_total_prompt_tokens: int = 0
        self.obs_kv_written_tokens: int = 0
        self.obs_span_end_s: float = 0.0

    def _admit(self, req):
        pre_exists = (req.prefix >= 0 and req.prefix in self.prefix_entry)
        pre_tok = (req.prefix_tokens or self.workload.prefix_tokens) if req.prefix >= 0 else 0
        ok = super()._admit(req)
        if not ok:
            return False
        self.obs_total_prompt_tokens += pre_tok + req.input
        if pre_exists:
            self.obs_prefix_hit_tokens += pre_tok
            pf_toks = req.input
        else:
            pf_toks = pre_tok + req.input
        self.obs_prefill_iters += 1
        self.obs_prefill_toks.append(pf_toks)
        self.obs_batch_sizes.append(self.n_running)
        self.obs_batch_weights.append(1)
        self.obs_kv_written_tokens += pf_toks
        return True

    def _write(self, e, pages, cause, align=False):
        if pages > 0 and cause in ("prefill", "decode"):
            self.obs_write_events.append((self.t, float(pages) * self.dev.page_bytes))
            self.obs_alloc_events += 1
        return super()._write(e, pages, cause, align=align)

    def _kill(self, e, evicted=False):
        self.obs_release_events += 1
        return super()._kill(e, evicted=evicted)

    def _complete(self, slot):
        req = self.sl_req[slot]
        self.obs_e2e_s.append(self.t - req.arrival)
        self.obs_kv_written_tokens += req.output
        self.obs_span_end_s = max(self.obs_span_end_s, self.t)
        return super()._complete(slot)

    def _advance(self, on, k, T):
        if k > 0 and len(on) > 0:
            self.obs_decode_iters += int(k)
            self.obs_batch_sizes.append(int(len(on)))
            self.obs_batch_weights.append(int(k))
            self.obs_tpot_ms.append(float(T) * 1000.0)
            self.obs_tpot_weights.append(int(k))
        return super()._advance(on, k, T)


def _weighted_pct(values: list[float], weights: list[int], q: float) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    order = np.argsort(arr)
    arr, w = arr[order], w[order]
    cdf = np.cumsum(w) / w.sum()
    idx = int(np.searchsorted(cdf, q / 100.0))
    return float(arr[min(idx, len(arr) - 1)])


def _weighted_mean(values: list[float], weights: list[int]) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    return float(np.dot(arr, w) / w.sum())


def run_ours(
    sel_recs: list[RequestRecord],
    *,
    with_prefix: bool,
    cap_gib: float,
    model,
    geo,
    clock,
    gpu,
) -> dict[str, Any]:
    """Run our unmodified Simulator logic via ObservedSimulator at X=1e5 and X=1e6."""
    if with_prefix:
        script_items = []
        pids = set()
        for r in sel_recs:
            pt = r.prefix_tokens if (r.prefix_id is not None and r.prefix_id >= 0) else 0
            pid = r.prefix_id if (r.prefix_id is not None and r.prefix_id >= 0) else -1
            if pid >= 0:
                pids.add(pid)
            script_items.append({
                "arrival": r.arrival_s,
                "input": max(1, r.prompt_tokens - pt),
                "output": r.output_tokens,
                "prefix": pid,
                "prefix_tokens": pt,
            })
        wl = scripted(
            script_items,
            name="mc_prefix",
            retain_s=PREFIX_TTL_S,
            max_batch=MAX_BATCH_SLOTS,
            prefix_groups=max(pids) + 1 if pids else 0,
            prefix_tokens=512 if pids else 0,
            prefix_share=1.0 if pids else 0.0,
        )
    else:
        script_items = [
            {"arrival": r.arrival_s, "input": r.prompt_tokens, "output": r.output_tokens}
            for r in sel_recs
        ]
        wl = scripted(script_items, name="mc_noprefix", retain_s=0.0, max_batch=MAX_BATCH_SLOTS)

    dur_s = sel_recs[-1].arrival_s
    horizon_h = (dur_s + 900.0) / 3600.0

    # Primary run at X = 1e5
    dev = Device(pages_per_block=PPB, capacity_gib=cap_gib)
    opt_1e5 = SimOptions(
        policy="packed", x=1e5, hours=horizon_h, warmup_h=0.0,
        prefill="colocated", m5="gaps", slots=MAX_BATCH_SLOTS,
    )
    sim_1e5 = ObservedSimulator(geo, clock, wl, dev, opt_1e5, model=model, gpu=gpu)
    res_1e5 = sim_1e5.run()

    # Secondary run at X = 1e6 for endurance sensitivity
    opt_1e6 = SimOptions(
        policy="packed", x=1e6, hours=horizon_h, warmup_h=0.0,
        prefill="colocated", m5="gaps", slots=MAX_BATCH_SLOTS,
    )
    sim_1e6 = ObservedSimulator(geo, clock, wl, dev, opt_1e6, model=model, gpu=gpu)
    res_1e6 = sim_1e6.run()

    # Compute exact span from first arrival to last request completion
    span_s = sim_1e5.obs_span_end_s if sim_1e5.obs_span_end_s > 0 else sim_1e5.t
    # Normalize P/E/day to actual active serving span (since horizon_h includes post-drain idle padding!)
    pec_day_1e5 = float(sim_1e5.pec.mean()) * DAY / span_s
    pec_day_1e6 = float(sim_1e6.pec.mean()) * DAY / span_s
    ref_kv_1e5 = float(res_1e5["refresh_to_kv"])
    ref_kv_1e6 = float(res_1e6["refresh_to_kv"])

    total_iters = sim_1e5.obs_prefill_iters + sim_1e5.obs_decode_iters
    kv_write_bytes = sum(b for _, b in sim_1e5.obs_write_events)
    mr10, pr10, p2m10 = _write_rate_stats(sim_1e5.obs_write_events, span_s, 10.0)
    _, _, p2m1 = _write_rate_stats(sim_1e5.obs_write_events, span_s, 1.0)
    # Our scheduler never preempts running requests (0 preemption recompute); evicted/expired
    # idle prefix cache volume is reported symmetrically in ``evicted_gb`` on both sides.
    recomp_toks = 0.0
    recomp_frac = 0.0

    # Also include 0-prefill decode iterations when computing mean prefill tokens over all iters
    pf_mean_all = sum(sim_1e5.obs_prefill_toks) / total_iters if total_iters > 0 else 0.0
    pf_frac = sim_1e5.obs_prefill_iters / total_iters if total_iters > 0 else 0.0

    return {
        "completed": int(sim_1e5.n["completed"]),
        "total_requests": len(sel_recs),
        "span_s": span_s,
        "iterations": total_iters,
        "batch_median": _weighted_pct(sim_1e5.obs_batch_sizes, sim_1e5.obs_batch_weights, 50),
        "batch_p90": _weighted_pct(sim_1e5.obs_batch_sizes, sim_1e5.obs_batch_weights, 90),
        "batch_max": float(max(sim_1e5.obs_batch_sizes)) if sim_1e5.obs_batch_sizes else 0.0,
        "batch_mean": _weighted_mean(sim_1e5.obs_batch_sizes, sim_1e5.obs_batch_weights),
        "prefill_tok_median_pos": _pct(sim_1e5.obs_prefill_toks, 50),
        "prefill_tok_p90_pos": _pct(sim_1e5.obs_prefill_toks, 90),
        "prefill_tok_max": float(max(sim_1e5.obs_prefill_toks)) if sim_1e5.obs_prefill_toks else 0.0,
        "prefill_tok_mean_all": pf_mean_all,
        "prefill_iter_frac": pf_frac,
        "preemptions": 0,
        "stalls": int(sim_1e5.n["stalls"]),
        "recompute_tokens": recomp_toks,
        "recompute_write_frac": recomp_frac,
        "prefix_hit_tokens": sim_1e5.obs_prefix_hit_tokens,
        "prefix_hit_frac": sim_1e5.obs_prefix_hit_tokens / max(1, sim_1e5.obs_total_prompt_tokens),
        "kv_write_gb": kv_write_bytes / 1e9,
        "write_rate_mean_gb_s": mr10,
        "write_rate_peak10_gb_s": pr10,
        "write_peak_to_mean_10s": p2m10,
        "write_peak_to_mean_1s": p2m1,
        "kv_alloc_events": sim_1e5.obs_alloc_events,
        "kv_release_events": sim_1e5.obs_release_events,
        "e2e_p50_s": _pct(sim_1e5.obs_e2e_s, 50),
        "e2e_p90_s": _pct(sim_1e5.obs_e2e_s, 90),
        "e2e_max_s": float(max(sim_1e5.obs_e2e_s)) if sim_1e5.obs_e2e_s else 0.0,
        "e2e_mean_s": float(np.mean(sim_1e5.obs_e2e_s)) if sim_1e5.obs_e2e_s else 0.0,
        "tpot_p50_ms": _weighted_pct(sim_1e5.obs_tpot_ms, sim_1e5.obs_tpot_weights, 50),
        "tpot_p90_ms": _weighted_pct(sim_1e5.obs_tpot_ms, sim_1e5.obs_tpot_weights, 90),
        "tpot_p95_ms": _weighted_pct(sim_1e5.obs_tpot_ms, sim_1e5.obs_tpot_weights, 95),
        "tpot_mean_ms": _weighted_mean(sim_1e5.obs_tpot_ms, sim_1e5.obs_tpot_weights),
        "pec_per_day_x1e5": pec_day_1e5,
        "refresh_to_kv_x1e5": ref_kv_1e5,
        "pec_per_day_x1e6": pec_day_1e6,
        "refresh_to_kv_x1e6": ref_kv_1e6,
        "evicted_gb": float(res_1e5["evicted_gb"]),
        "swap_out_blocks": 0,
        "swap_out_gb": 0.0,
        "wall_s": sim_1e5.wall_s + sim_1e6.wall_s,
    }


# ---------------------------------------------------------------------------
# 3. ServeLoop runner (genuine ServeLoopEngine + ServeLoopPlacement + PrefixCache)
#    + HBF segment-packed wear replay of ServeLoop's exact per-iteration schedule
# ---------------------------------------------------------------------------

def replay_serveloop_on_hbf_wear(
    iter_events: list[dict[str, Any]],
    *,
    cap_gib: float,
    x_threshold: float,
    span_s: float,
    prefix_ttl_s: float,
    seg_banks: int = BANKS,
) -> tuple[float | None, float | None, str]:
    """Replay ServeLoop's per-iteration KV writes/reads/releases/preemptions on our HBF segment wear model.

    Mirrors ``hbf/lifetime/sim/`` packed-segment mechanics:
    - Default ``seg_banks = 1024`` -> segment size ``P = 1024 * 256 = 262,144`` pages (1.0 GiB).
      When ServeLoop packs >98% of a tight pool via 2 MiB paged blocks so 1 GiB superblocks
      cannot find a free segment, callers may retry with ``seg_banks = 16`` (16 MiB die segments,
      same total capacity ``cap_gib``) and record the fallback explicitly in ``wear_status``.
    """
    P = seg_banks * PPB
    S = int(round(cap_gib * (BANKS // seg_banks)))
    if S < 4:
        return None, None, "capacity < 4 segments"

    pages_per_tok = KV_BYTES_PER_TOKEN / PAGE_BYTES  # 32.0 pages/token (exact for Llama-3.1-8B BF16)

    fill = np.zeros(S, dtype=np.float64)
    live = np.zeros(S, dtype=np.float64)
    reads = np.zeros(S, dtype=np.float64)
    pec = np.zeros(S, dtype=np.int64)
    state = np.zeros(S, dtype=np.int8)  # 0=FREE, 1=OPEN, 2=FULL
    free_set = set(range(S))
    heap = [(0, s) for s in range(S)]
    heapq.heapify(heap)
    frontier = -1

    # Active entries: key -> dict(seg -> pages)
    entries: dict[str, dict[int, float]] = {}
    seg_entries: list[set[str]] = [set() for _ in range(S)]
    # Prefix cache entries (if prefix sharing enabled): stored as entries keyed by "pf:{req_or_blk}"
    # For ServeLoop prefix blocks, we track one entry per 512-token-equivalent or live prefix cache bytes.
    kv_programs = 0.0
    ref_programs = 0.0
    comp_programs = 0.0
    reclaim_low_wm = min(max(2, S // 8), 16)

    def _release_if_empty(s: int):
        if s in free_set or state[s] != 2 or live[s] > 1e-6:
            return
        live[s] = 0.0
        fill[s] = 0.0
        reads[s] = 0.0
        state[s] = 0
        free_set.add(s)
        heapq.heappush(heap, (int(pec[s]), s))

    def _close(s: int):
        nonlocal frontier
        state[s] = 2
        if frontier == s:
            frontier = -1
        _release_if_empty(s)

    def _take_free() -> int:
        while heap:
            _, s = heapq.heappop(heap)
            if s in free_set:
                free_set.remove(s)
                return s
        return -1

    def _reclaim_one() -> bool:
        nonlocal comp_programs
        # Pick full segment with fewest live pages that actually has dead space to recover.
        best_s = -1
        best_live = P - 1.0
        for s in range(S):
            if state[s] == 2 and live[s] < best_live:
                best_live = live[s]
                best_s = s
        if best_s < 0:
            return False
        return _relocate_seg(best_s, is_refresh=False)

    def _open_seg() -> int:
        nonlocal frontier
        s = _take_free()
        if s < 0:
            if not _reclaim_one():
                return -1
            if frontier >= 0 and fill[frontier] < P - 1e-9:
                return frontier
            s = _take_free()
            if s < 0:
                return -1
        pec[s] += 1
        state[s] = 1
        fill[s] = 0.0
        reads[s] = 0.0
        frontier = s
        return s

    def _write_pages(ekey: str, n_pages: float, cause: str) -> bool:
        nonlocal frontier, kv_programs, ref_programs, comp_programs
        if n_pages <= 1e-9:
            return True
        if ekey not in entries:
            entries[ekey] = {}
        rem = float(n_pages)
        while rem > 1e-9:
            s = frontier
            if s < 0:
                s = _open_seg()
                if s < 0:
                    return False
            k = min(rem, P - fill[s])
            fill[s] += k
            live[s] += k
            entries[ekey][s] = entries[ekey].get(s, 0.0) + k
            seg_entries[s].add(ekey)
            if cause == "kv":
                kv_programs += k
            elif cause == "refresh":
                ref_programs += k
            else:
                comp_programs += k
            if fill[s] >= P - 1e-9:
                _close(s)
            rem -= k
        return True

    def _kill_entry(ekey: str):
        segs = entries.pop(ekey, None)
        if not segs:
            return
        for s, k in segs.items():
            seg_entries[s].discard(ekey)
            live[s] = max(0.0, live[s] - k)
            if live[s] <= 1e-6:
                _release_if_empty(s)

    def _relocate_seg(s: int, *, is_refresh: bool) -> bool:
        if state[s] == 1:
            _close(s)
        moves: list[tuple[str, float]] = []
        for ekey in list(seg_entries[s]):
            k = entries[ekey].pop(s, 0.0)
            seg_entries[s].discard(ekey)
            if k > 1e-9:
                live[s] = max(0.0, live[s] - k)
                moves.append((ekey, k))
        live[s] = 0.0
        _release_if_empty(s)
        cause = "refresh" if is_refresh else "compaction"
        for ekey, k in moves:
            if not _write_pages(ekey, k, cause):
                return False
        return True

    # Track active read counts and execute events
    for ev in iter_events:
        # 1. Preemptions discard preempted request's private KV pages
        for rid in ev["preempted_ids"]:
            _kill_entry(f"req:{rid}")
        # 2. Prefix cache evictions/expirations shrink shared prefix entry
        if ev["prefix_evicted_pages"] > 0 and "prefix_pool" in entries:
            rem_ev = ev["prefix_evicted_pages"]
            segs = entries["prefix_pool"]
            for s in list(segs):
                if rem_ev <= 1e-9:
                    break
                take = min(rem_ev, segs[s])
                segs[s] -= take
                live[s] = max(0.0, live[s] - take)
                rem_ev -= take
                if segs[s] <= 1e-9:
                    del segs[s]
                    seg_entries[s].discard(ekey_pf := "prefix_pool")
                if live[s] <= 1e-6:
                    _release_if_empty(s)
        # 3. Writes for this iteration (prefill chunks + recompute + decode)
        for rid, new_toks, pf_cached_new_toks in ev["writes"]:
            priv_toks = max(0, new_toks - pf_cached_new_toks)
            if pf_cached_new_toks > 0:
                if not _write_pages("prefix_pool", pf_cached_new_toks * pages_per_tok, "kv"):
                    return None, None, "no free 1 GiB segment during prefix write"
            if priv_toks > 0:
                if not _write_pages(f"req:{rid}", priv_toks * pages_per_tok, "kv"):
                    return None, None, "no free 1 GiB segment during kv write"
        # 4. Reads: every scheduled slice reads its context KV (private + shared prefix hit)
        for rid, ctx_toks, hit_toks in ev["reads"]:
            ekey = f"req:{rid}"
            segs = entries.get(ekey)
            if segs:
                for s, k in segs.items():
                    reads[s] += k / seg_banks
            if hit_toks > 0 and "prefix_pool" in entries:
                pf_segs = entries["prefix_pool"]
                tot_pf = sum(pf_segs.values())
                if tot_pf > 0:
                    hit_pages = hit_toks * pages_per_tok
                    scale = min(1.0, hit_pages / tot_pf)
                    for s, k in pf_segs.items():
                        reads[s] += (k * scale) / seg_banks
        # 5. Completed requests release private KV
        for rid in ev["completed_ids"]:
            _kill_entry(f"req:{rid}")
        # 6. Read-disturb refresh checks
        hot_segs = np.flatnonzero(reads >= x_threshold)
        for s in hot_segs.tolist():
            if live[s] > 1e-6:
                if not _relocate_seg(s, is_refresh=True):
                    return None, None, "no free 1 GiB segment during RD refresh"
            else:
                reads[s] = 0.0
        # 7. Background reclaim if free segments low (bounded and only while making progress)
        if len(free_set) < reclaim_low_wm:
            for _ in range(S):
                if len(free_set) >= reclaim_low_wm:
                    break
                before_free = len(free_set)
                before_dead = sum(P - live[s] for s in range(S) if state[s] == 2)
                if not _reclaim_one():
                    break
                after_dead = sum(P - live[s] for s in range(S) if state[s] == 2)
                if len(free_set) <= before_free and after_dead >= before_dead - 1.0:
                    break

    pec_per_day = float(pec.mean()) * DAY / span_s if span_s > 0 else 0.0
    ref_to_kv = (ref_programs + comp_programs) / kv_programs if kv_programs > 0 else 0.0
    return pec_per_day, ref_to_kv, "ok"


def run_serveloop_scenario(
    sl_dir: Path,
    sel_raw: list[dict[str, Any]],
    mc_sha: str,
    *,
    with_prefix: bool,
    cap_gib: float,
    timing_mode: str,   # "default" or "matched"
    ours_clock=None,
    ours_model=None,
    ours_gpu=None,
    cold_hbf_gib: float = 0.0,
    sim_bin: Path | None = None,
) -> dict[str, Any]:
    """Run ServeLoop's exact scheduler + paged KV + PrefixCache on the Mooncake slice."""
    from serveloop.catalog import load_model_any
    from serveloop.compiler import ServeLoopCompiler
    from serveloop.contracts import (
        BatchSlice,
        RequestSpec,
        RequestTrace,
        RooflineTimingProvider,
        ScheduledBatch,
        SchedulerPolicy,
        TraceProvenance,
    )
    from serveloop.engine import ServeLoopEngine
    from serveloop.placement import KvPlacement, PlacementSpec, ServeLoopPlacement
    from serveloop.prefix import PrefixCache, PrefixEntry

    class FastExpirePrefixCache(PrefixCache):
        """O(1) min-heap TTL expiry over upstream PrefixCache (eliminates O(N_entries) scan per iter)."""

        def __init__(self, pool, capacity_bytes, ttl_ns):
            super().__init__(pool, capacity_bytes, ttl_ns)
            self._exp_heap: list[tuple[float, str]] = []

        def expire(self, now_ns):
            if self.ttl_ns is None:
                return
            while self._exp_heap and self._exp_heap[0][0] <= now_ns:
                exp_ns, key = heapq.heappop(self._exp_heap)
                entry = self.entries.get(key)
                if entry is not None and entry.expires_ns is not None and entry.expires_ns <= now_ns:
                    self.evict(key, expired=True)

        def publish(self, keys, blocks, complete_blocks, now_ns, *, pending=False):
            self.expire(now_ns)
            for ordinal in range(complete_blocks):
                key = keys[ordinal]
                if key in self.entries:
                    entry = self.entries[key]
                    if entry.pending and not pending:
                        exp = None if self.ttl_ns is None else now_ns + self.ttl_ns
                        self.entries[key] = PrefixEntry(entry.blocks, exp, False)
                        if exp is not None:
                            heapq.heappush(self._exp_heap, (exp, key))
                    self.entries.move_to_end(key)
                    continue
                selected = tuple(layer[ordinal] for layer in blocks)
                required = len(selected) * self.pool.block_bytes
                if required > self.capacity_bytes:
                    continue
                while self.bytes + required > self.capacity_bytes:
                    self.evict(next(iter(self.entries)))
                self.pool.retain(selected)
                exp = None if pending or self.ttl_ns is None else now_ns + self.ttl_ns
                self.entries[key] = PrefixEntry(selected, exp, pending)
                if exp is not None:
                    heapq.heappush(self._exp_heap, (exp, key))
                self.bytes += required
                self.stats["inserted_blocks"] += 1

        def install(self, key, blocks, expires_ns):
            super().install(key, blocks, expires_ns)
            if expires_ns is not None:
                heapq.heappush(self._exp_heap, (expires_ns, key))

    t_wall0 = time.perf_counter()
    model = load_model_any(sl_dir / "models/llama31-8b-w8-kv-bf16.json")
    trace_doc = build_serveloop_trace_dict(
        sel_raw,
        source_sha256=mc_sha,
        source_label=f"mooncake_s{SHARD_STACKS}_n{len(sel_raw)}",
        model_id=model.model_id,
        with_prefix=with_prefix,
        vocab_size=model.vocab_size,
    )
    reqs = tuple(
        RequestSpec(
            request_id=r["request_id"],
            arrival_ns=r["arrival_ns"],
            model_id=r["model_id"],
            prompt_tokens=r["prompt_tokens"],
            output_tokens=r["output_tokens"],
            token_ids=None if r["token_ids"] is None else tuple(r["token_ids"]),
            cache_salt=r["cache_salt"],
        )
        for r in trace_doc["requests"]
    )
    trace = RequestTrace(
        provenance=TraceProvenance("production_trace", trace_doc["provenance"]["source"], mc_sha),
        requests=reqs,
    )

    weight_bytes = model.weight_footprint_bytes
    kv_bytes = int(cap_gib * GIB)
    hbm_total = ((weight_bytes + kv_bytes + 2 * GIB) // 4096) * 4096
    reserve = ((hbm_total - weight_bytes - kv_bytes) // 4096) * 4096
    hbf_cap = int(cold_hbf_gib * GIB) if cold_hbf_gib > 0 else 0
    pspec = PlacementSpec(
        hbm_capacity_bytes=hbm_total,
        hbf_capacity_bytes=hbf_cap,
        external_capacity_bytes=0,
        hbm_runtime_reserve_bytes=reserve,
        hbm_model_cache_bytes=0,
        model_weight_tiers={model.model_id: "hbm"},
        object_tier_overrides={},
        kv_block_tokens=16,
        kv_placement=KvPlacement(hot="hbm", cold="hbf" if cold_hbf_gib > 0 else None),
        prefix_cache_bytes=kv_bytes if with_prefix else 0,
        prefix_cache_ttl_ns=PREFIX_TTL_S * 1e9 if with_prefix else None,
    )
    hbf_geom = None
    if cold_hbf_gib > 0:
        from hbfsim_client.simulation_session import ResolvedSystemConfig
        assert sim_bin is not None
        sys_cfg_hbf = ResolvedSystemConfig.load([sl_dir / "configs/systems/4hbm-4hbf.cfg"]).resolve(
            sim_bin, enable_hbf=True
        )
        hbf_geom = sys_cfg_hbf.hbf_geometry
    pl = ServeLoopPlacement(models={model.model_id: model}, spec=pspec, hbf_geometry=hbf_geom, router=None)
    if with_prefix:
        pl._prefix = FastExpirePrefixCache(pl._hot, pspec.prefix_cache_bytes, pspec.prefix_cache_ttl_ns)
        # Skip redundant decode-step re-publication of already-finalized prompt blocks (preserves
        # prefill publication & release_order while avoiding O(prompt_blocks * decode_steps) overhead).
        _orig_pub_complete = pl._publish_complete
        _orig_pub_pending = pl._publish_pending
        _finalized_reqs: set[str] = set()

        def _fast_pub_complete(state, token_end: int, finish_ns: float) -> None:
            rid = state.request.request_id
            if token_end > state.request.prompt_tokens and rid in _finalized_reqs:
                pl._prefix.expire(finish_ns)
                return
            _orig_pub_complete(state, token_end, finish_ns)
            if token_end >= state.request.prompt_tokens:
                _finalized_reqs.add(rid)

        def _fast_pub_pending(state, token_end: int) -> None:
            rid = state.request.request_id
            if token_end > state.request.prompt_tokens and rid in _finalized_reqs:
                return
            _orig_pub_pending(state, token_end)

        pl._publish_complete = _fast_pub_complete
        pl._publish_pending = _fast_pub_pending
    roofline = RooflineTimingProvider(peak_tflops=989.0, efficiency=0.6)
    w_per_layer = model.weight_footprint_bytes / model.num_layers
    kv_per_tok_layer = model.kv_bytes_per_token / model.num_layers

    # Dynamic clock executor implementing ServeLoopExecutor protocol directly over ServeLoopPlacement
    class DynamicExecutor:
        def __init__(self):
            self._frontier = 0.0
            self.alloc_events = 0
            self.release_events = 0
            self.swap_out_blocks = 0
            self.swap_out_bytes = 0

        @property
        def frontier_ns(self) -> float:
            return self._frontier

        @property
        def execution_identity(self):
            return {"kind": f"serveloop_dynamic_{timing_mode}"}

        def admit_request(self, request: RequestSpec) -> int:
            return pl.admit_request(request, now_ns=max(self._frontier, request.arrival_ns))

        def can_reserve(self, slices) -> bool:
            return pl.can_reserve(slices)

        def reserve(self, batch_id: int, slices):
            rec = pl.reserve(batch_id, slices)
            if rec.get("allocated_blocks", 0) > 0:
                self.alloc_events += 1
            return rec

        def preempt_request(self, request_id: str):
            rec = pl.preempt_request(request_id)
            if rec.get("mode") == "recompute":
                self.release_events += 1
            elif rec.get("mode") == "swap_out":
                self.swap_out_blocks += int(rec.get("blocks", 0))
                self.swap_out_bytes += int(rec.get("bytes", 0))
            return rec

        def release_request(self, request_id: str) -> None:
            pl.release_request(request_id)
            self.release_events += 1

        def step_duration_ns(self, sched: ScheduledBatch) -> float:
            if timing_mode == "default":
                bt = roofline.timing_for(model=model, batch=sched)
                kv_r = sum(s.context_tokens_before + s.token_count for s in sched.slices) * kv_per_tok_layer
                kv_w = sum(s.token_count for s in sched.slices) * kv_per_tok_layer
                mem_ns = (w_per_layer + kv_r + kv_w) / HBM4_EFF_BW_B_PER_NS
                return sum(max(c_ns, mem_ns) for c_ns in bt.layer_ns) + bt.tail_ns
            # Matched timing: match our B200 + 1-stack OCP HBF clock per iteration
            dec_ctxs = [s.context_tokens_before + s.token_count for s in sched.slices if s.phase == "decode"]
            decode_ns = float(ours_clock.step(np.array(dec_ctxs, dtype=np.int64)).tpot_ns) if dec_ctxs else 0.0
            pf_slices = [s for s in sched.slices if s.phase == "prefill"]
            pf_ns = 0.0
            if pf_slices:
                pf_toks = sum(s.token_count for s in pf_slices)
                mean_ctx = float(sum(s.context_tokens_before + s.token_count for s in pf_slices) / len(pf_slices))
                pf_rate = resolve_prefill_tok_s(ours_model, ours_gpu, mean_ctx)
                pf_ns = (pf_toks / pf_rate) * 1e9
            return max(1.0, decode_ns + pf_ns)

        def complete_iteration(self, sched: ScheduledBatch) -> tuple[float, float]:
            start_ns = max(self._frontier, sched.not_before_ns)
            dur_ns = self.step_duration_ns(sched)
            finish_ns = start_ns + dur_ns
            pl.discard_unmapped(sched.batch_id)
            pl.complete_batch(sched, finish_ns)
            self._frontier = finish_ns
            return start_ns, finish_ns

    # Subclass ServeLoopEngine ONLY to fix upstream 7d5d7a4 double-preemption bug in _preempt
    # (engine.py:258-262 reset prefill_target to prompt_tokens when a request with emitted >= 1
    # was preempted a second time mid-recomputation).
    class FixedPreemptEngine(ServeLoopEngine):
        def _preempt(self, batch_id: int, state) -> None:
            super()._preempt(batch_id, state)
            if self._preemptions and self._preemptions[-1]["mode"] == "recompute":
                state.prefill_target = state.request.prompt_tokens + state.emitted

    ex = DynamicExecutor()
    comp = ServeLoopCompiler(models={model.model_id: model}, request_trace=trace, router=None,
                             timing=roofline, prefetch_depth=1)
    policy = SchedulerPolicy(
        max_batch_requests=MAX_BATCH_SLOTS,
        max_batch_tokens=SL_MAX_BATCH_TOKENS,
        prefill_chunk_tokens=SL_PREFILL_CHUNK,
    )
    eng = FixedPreemptEngine(models={model.model_id: model}, request_trace=trace,
                             compiler=comp, executor=ex, policy=policy)

    batch_sizes: list[int] = []
    prefill_toks_pos: list[int] = []
    prefill_toks_sum = 0
    prefill_iters = 0
    write_events: list[tuple[float, float]] = []
    iter_events: list[dict[str, Any]] = []
    tpot_ms_all: list[float] = []

    prev_preempt_idx = 0
    prev_pf_inserted = 0
    prev_pf_evicted = 0
    prev_pf_expired = 0
    total_written_tokens = 0

    while eng.unfinished_requests > 0:
        sched, _ = eng.next_schedule()
        start_ns, finish_ns = ex.complete_iteration(sched)
        dur_ms = (finish_ns - start_ns) / 1e6

        # Preemptions that occurred while scheduling this iteration
        new_preempts = eng._preemptions[prev_preempt_idx:]
        prev_preempt_idx = len(eng._preemptions)
        preempted_ids = [p["request_id"] for p in new_preempts if p["mode"] == "recompute"]

        # Track prefix cache deltas after complete_batch
        pf_rcpt = pl.prefix_receipt()
        ins_delta = pf_rcpt["inserted_blocks"] - prev_pf_inserted
        ev_delta = (pf_rcpt["evicted_blocks"] - prev_pf_evicted) + (pf_rcpt["expired_blocks"] - prev_pf_expired)
        prev_pf_inserted = pf_rcpt["inserted_blocks"]
        prev_pf_evicted = pf_rcpt["evicted_blocks"]
        prev_pf_expired = pf_rcpt["expired_blocks"]

        b_size = len(sched.slices)
        batch_sizes.append(b_size)
        pf_toks = 0
        has_decode = False
        iter_written_toks = 0
        writes_list = []
        reads_list = []
        for sl in sched.slices:
            iter_written_toks += sl.token_count
            if sl.phase == "prefill":
                pf_toks += sl.token_count
            else:
                has_decode = True
            st = eng._by_id[sl.request_id]
            reads_list.append((sl.request_id, sl.context_tokens_before + sl.token_count, st.prefix_hit_tokens))
            writes_list.append((sl.request_id, sl.token_count, 0))
        if ins_delta > 0 and writes_list:
            # Attribute newly published prefix blocks (16 tokens/block) to prefix_pool in HBF wear replay
            pf_new_toks = ins_delta * 16
            rid0, t0, _ = writes_list[0]
            writes_list[0] = (rid0, t0, min(t0, pf_new_toks))

        if pf_toks > 0:
            prefill_iters += 1
            prefill_toks_pos.append(pf_toks)
            prefill_toks_sum += pf_toks
        if has_decode:
            tpot_ms_all.append(dur_ms)

        total_written_tokens += iter_written_toks
        write_events.append((start_ns / 1e9, float(iter_written_toks) * KV_BYTES_PER_TOKEN))

        # Apply completion in engine and capture finished request IDs
        before_done = eng.unfinished_requests
        eng.complete(sched, finish_ns)
        completed_ids = []
        if eng.unfinished_requests < before_done:
            for sl in sched.slices:
                if eng._by_id[sl.request_id].complete:
                    completed_ids.append(sl.request_id)

        iter_events.append({
            "preempted_ids": preempted_ids,
            "prefix_evicted_pages": float(ev_delta) * 16.0 * (KV_BYTES_PER_TOKEN / PAGE_BYTES),
            "writes": writes_list,
            "reads": reads_list,
            "completed_ids": completed_ids,
        })

    # Per-request metrics
    e2e_s = []
    req_tpot_ms = []
    prefix_hit_tokens = 0
    total_prompt_tokens = 0
    for st in eng._states:
        total_prompt_tokens += st.request.prompt_tokens
        prefix_hit_tokens += st.prefix_hit_tokens
        if st.completion_ns is not None:
            e2e_s.append((st.completion_ns - st.request.arrival_ns) / 1e9)
            if st.request.output_tokens > 1 and st.first_token_ns is not None:
                req_tpot_ms.append((st.completion_ns - st.first_token_ns) / (st.request.output_tokens - 1) / 1e6)

    span_s = ex.frontier_ns / 1e9
    recomp_toks = float(sum(p["processed_tokens_before"] for p in eng._preemptions if p["mode"] == "recompute"))
    recomp_frac = recomp_toks / total_written_tokens if total_written_tokens > 0 else 0.0
    kv_write_bytes = float(total_written_tokens) * KV_BYTES_PER_TOKEN
    mr10, pr10, p2m10 = _write_rate_stats(write_events, span_s, 10.0)
    _, _, p2m1 = _write_rate_stats(write_events, span_s, 1.0)

    # Replay ServeLoop's exact schedule onto our HBF segment-packed wear model (X=1e5 and X=1e6)
    wear_cap = cap_gib if cold_hbf_gib == 0 else cold_hbf_gib
    pec_1e5, ref_1e5, wear_status = replay_serveloop_on_hbf_wear(
        iter_events, cap_gib=wear_cap, x_threshold=1e5, span_s=span_s,
        prefix_ttl_s=PREFIX_TTL_S if with_prefix else 0.0, seg_banks=BANKS,
    )
    wear_seg_banks = BANKS
    wear_replay_cap = wear_cap
    if pec_1e5 is None:
        # ServeLoop's 2 MiB paged pool can pack >98% of nominal capacity; retry with 16 MiB die
        # segments and +10% flash OP headroom (scaling PEC back to nominal wear_cap blocks).
        wear_seg_banks = 16
        wear_replay_cap = wear_cap * 1.10
        pec_raw, ref_1e5, st2 = replay_serveloop_on_hbf_wear(
            iter_events, cap_gib=wear_replay_cap, x_threshold=1e5, span_s=span_s,
            prefix_ttl_s=PREFIX_TTL_S if with_prefix else 0.0, seg_banks=wear_seg_banks,
        )
        pec_1e5 = None if pec_raw is None else pec_raw * (wear_replay_cap / wear_cap)
        wear_status = f"die_16m_op10 ({wear_status}; die={st2})"
    pec_1e6_raw, ref_1e6, _ = replay_serveloop_on_hbf_wear(
        iter_events, cap_gib=wear_replay_cap, x_threshold=1e6, span_s=span_s,
        prefix_ttl_s=PREFIX_TTL_S if with_prefix else 0.0,
        seg_banks=wear_seg_banks,
    )
    pec_1e6 = None if pec_1e6_raw is None else pec_1e6_raw * (wear_replay_cap / wear_cap)

    pf_rcpt = pl.prefix_receipt()
    wall_s = time.perf_counter() - t_wall0
    return {
        "completed": len(e2e_s),
        "total_requests": len(reqs),
        "span_s": span_s,
        "iterations": eng.iterations,
        "batch_median": _pct(batch_sizes, 50),
        "batch_p90": _pct(batch_sizes, 90),
        "batch_max": float(max(batch_sizes)) if batch_sizes else 0.0,
        "batch_mean": float(np.mean(batch_sizes)) if batch_sizes else 0.0,
        "prefill_tok_median_pos": _pct(prefill_toks_pos, 50),
        "prefill_tok_p90_pos": _pct(prefill_toks_pos, 90),
        "prefill_tok_max": float(max(prefill_toks_pos)) if prefill_toks_pos else 0.0,
        "prefill_tok_mean_all": prefill_toks_sum / eng.iterations if eng.iterations else 0.0,
        "prefill_iter_frac": prefill_iters / eng.iterations if eng.iterations else 0.0,
        "preemptions": len(eng._preemptions),
        "stalls": 0,
        "recompute_tokens": recomp_toks,
        "recompute_write_frac": recomp_frac,
        "prefix_hit_tokens": prefix_hit_tokens,
        "prefix_hit_frac": prefix_hit_tokens / max(1, total_prompt_tokens),
        "kv_write_gb": kv_write_bytes / 1e9,
        "write_rate_mean_gb_s": mr10,
        "write_rate_peak10_gb_s": pr10,
        "write_peak_to_mean_10s": p2m10,
        "write_peak_to_mean_1s": p2m1,
        "kv_alloc_events": ex.alloc_events,
        "kv_release_events": ex.release_events,
        "e2e_p50_s": _pct(e2e_s, 50),
        "e2e_p90_s": _pct(e2e_s, 90),
        "e2e_max_s": float(max(e2e_s)) if e2e_s else 0.0,
        "e2e_mean_s": float(np.mean(e2e_s)) if e2e_s else 0.0,
        "tpot_p50_ms": _pct(tpot_ms_all, 50),
        "tpot_p90_ms": _pct(tpot_ms_all, 90),
        "tpot_p95_ms": _pct(tpot_ms_all, 95),
        "tpot_mean_ms": float(np.mean(tpot_ms_all)) if tpot_ms_all else 0.0,
        "req_tpot_p50_ms": _pct(req_tpot_ms, 50),
        "pec_per_day_x1e5": pec_1e5,
        "refresh_to_kv_x1e5": ref_1e5,
        "pec_per_day_x1e6": pec_1e6,
        "refresh_to_kv_x1e6": ref_1e6,
        "wear_status": wear_status,
        "evicted_gb": (pf_rcpt["evicted_blocks"] + pf_rcpt["expired_blocks"]) * 16 * KV_BYTES_PER_TOKEN / 1e9,
        "swap_out_blocks": ex.swap_out_blocks,
        "swap_out_gb": ex.swap_out_bytes / 1e9,
        "wall_s": wall_s,
    }


def _worker_serveloop(task_args):
    """ProcessPool worker running one ServeLoop scenario (zero MQSim, ~0.3 GB RAM)."""
    sl_dir_str, sim_bin_str, sel_raw, mc_sha, cap_label, cap_gib, pref_label, wp, tmode, cold_hbf = task_args
    cache_dir = ROOT / ".build" / "serveloop" / "cache_v1"
    cache_dir.mkdir(parents=True, exist_ok=True)
    scen_id = f"{cap_label}__{pref_label}__{tmode}"
    cache_file = cache_dir / f"{scen_id}.json"
    if cache_file.is_file():
        return (cap_label, cap_gib, pref_label, wp, tmode, cold_hbf, json.loads(cache_file.read_text()))
    sl_dir = Path(sl_dir_str)
    if str(sl_dir) not in sys.path:
        sys.path.insert(0, str(sl_dir))
    load_extensions()
    model = load_model("llama-3.1-8b")
    geo, _ = geometry_for(model, "bf16", context=16384)
    sys_ev = REGISTRY.get_evaluator("system")
    gpu = sys_ev.load_gpu("b200")
    ours_clock = REGISTRY.get_clock("tpot")(
        model, gpu, geo, read_ns=4096, stacks=1, kv_format="bf16",
        options=sys_ev.SystemOptions(overlap="serial", kv_consume="stream"),
    )
    sl_res = run_serveloop_scenario(
        sl_dir, sel_raw, mc_sha,
        with_prefix=wp, cap_gib=cap_gib, timing_mode=tmode,
        ours_clock=ours_clock, ours_model=model, ours_gpu=gpu,
        cold_hbf_gib=cold_hbf, sim_bin=Path(sim_bin_str),
    )
    cache_file.write_text(json.dumps(sl_res), encoding="utf-8")
    return (cap_label, cap_gib, pref_label, wp, tmode, cold_hbf, sl_res)


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def main() -> int:
    load_extensions()
    sl_dir, sim_bin = ensure_external_tools()
    if not MOONCAKE_JSONL.is_file():
        subprocess.check_call(["bash", str(ROOT / "scripts/download_mooncake.sh")])

    sel_raw, sel_recs, slice_summary = select_mooncake_slice(
        MOONCAKE_JSONL, stacks=SHARD_STACKS, max_requests=MAX_REQUESTS, retain_s=PREFIX_TTL_S
    )
    print(
        f"Mooncake slice: {slice_summary['count']} reqs over {slice_summary['duration_s']:.1f}s "
        f"({slice_summary['rate_req_s']:.2f} req/s), mean_in={slice_summary['mean_prompt']:.0f}, "
        f"max_in={slice_summary['max_prompt']}, mean_out={slice_summary['mean_output']:.0f}"
    )

    # Save converted ServeLoop request files (no-prefix & with-prefix) under .build/serveloop/
    for tag, wp in (("noprefix", False), ("prefix", True)):
        p = ROOT / ".build/serveloop" / f"mooncake_s{SHARD_STACKS}_n{MAX_REQUESTS}_{tag}.json"
        doc = build_serveloop_trace_dict(
            sel_raw, source_sha256=slice_summary["sha256"],
            source_label=f"mooncake_s{SHARD_STACKS}_n{MAX_REQUESTS}_{tag}",
            with_prefix=wp,
        )
        p.write_text(json.dumps(doc), encoding="utf-8")

    # 1. Calibrate ServeLoop default timing against real HbfSimExecutor
    print("\n--- Step A: Calibrating ServeLoop default timing against real HbfSimExecutor ---")
    calib = calibrate_serveloop_hbfsim(sl_dir, sim_bin, sel_raw, slice_summary["sha256"])
    print("HBFSim calibration:", json.dumps(calib))

    # 2. Initialize our B200 + 1-stack OCP HBF clock
    model = load_model("llama-3.1-8b")
    geo, _ = geometry_for(model, "bf16", context=16384)
    sys_ev = REGISTRY.get_evaluator("system")
    gpu = sys_ev.load_gpu("b200")
    ours_clock = REGISTRY.get_clock("tpot")(
        model, gpu, geo, read_ns=4096, stacks=1, kv_format="bf16",
        options=sys_ev.SystemOptions(overlap="serial", kv_consume="stream"),
    )

    # 3. Run the complete comparison matrix
    # Core 2x2x2 matrix: (ample 512 GiB, tight 120 GiB) x (no-prefix, prefix 300s) x (default, matched)
    # Note: our simulator depends on (capacity, prefix), independent of ServeLoop's timing mode, so we
    # run our 4 unique configurations once and pair them against both default-timing and matched-timing ServeLoop.
    ours_cache: dict[tuple[float, bool], dict[str, Any]] = {}
    csv_rows: list[dict[str, Any]] = []
    pair_summaries: list[dict[str, Any]] = []

    print("\n--- Step B: Running our simplified scheduler (4 base configs) ---")
    for cap_label, cap_gib in (("ample_512g", AMPLE_CAP_GIB), ("tight_120g", TIGHT_CAP_GIB)):
        for pref_label, wp in (("no_prefix", False), ("prefix_300s", True)):
            res = run_ours(sel_recs, with_prefix=wp, cap_gib=cap_gib, model=model, geo=geo, clock=ours_clock, gpu=gpu)
            ours_cache[(cap_gib, wp)] = res
            print(
                f"  Ours [{cap_label:10s} | {pref_label:11s}]: done={res['completed']}, stalls={res['stalls']:4d}, "
                f"B_med={res['batch_median']:4.1f}, tpot_p50={res['tpot_p50_ms']:5.2f}ms, "
                f"kv_w={res['kv_write_gb']:6.1f}GB (p2m10={res['write_peak_to_mean_10s']:.2f}), "
                f"PEC/d(1e5)={res['pec_per_day_x1e5']:6.1f}, ref/kv={res['refresh_to_kv_x1e5']:.2f}"
            )

    print("\n--- Step C: Running ServeLoop (8 core configs + 3 tight diagnostics, 4 parallel workers) ---", flush=True)
    scenarios = [
        ("ample_512g", AMPLE_CAP_GIB, "no_prefix", False, "default", 0.0),
        ("ample_512g", AMPLE_CAP_GIB, "prefix_300s", True, "default", 0.0),
        ("ample_512g", AMPLE_CAP_GIB, "no_prefix", False, "matched", 0.0),
        ("ample_512g", AMPLE_CAP_GIB, "prefix_300s", True, "matched", 0.0),
        ("tight_120g", TIGHT_CAP_GIB, "no_prefix", False, "default", 0.0),
        ("tight_120g", TIGHT_CAP_GIB, "prefix_300s", True, "default", 0.0),
        ("tight_120g", TIGHT_CAP_GIB, "no_prefix", False, "matched", 0.0),
        ("tight_120g", TIGHT_CAP_GIB, "prefix_300s", True, "matched", 0.0),
        # Diagnostics: tight preemption regimes for ServeLoop (64 GiB matched, 24 GiB default, 64 GiB HBM + 512 GiB HBF cold)
        ("tight_64g_sl_matched", TIGHT_SL_MATCHED_GIB, "no_prefix", False, "matched", 0.0),
        ("tight_24g_sl_default", TIGHT_SL_DEFAULT_GIB, "no_prefix", False, "default", 0.0),
        ("tight_64g_hbf_cold", TIGHT_SL_MATCHED_GIB, "no_prefix", False, "matched", AMPLE_CAP_GIB),
    ]

    from concurrent.futures import ProcessPoolExecutor, as_completed
    # Schedule slowest scenario (tight_120g prefix matched) first so parallel workers minimize makespan.
    run_order = sorted(
        range(len(scenarios)),
        key=lambda i: 0 if (scenarios[i][0] == "tight_120g" and scenarios[i][3] and scenarios[i][4] == "matched") else i + 1,
    )
    task_map = {
        i: (str(sl_dir), str(sim_bin), sel_raw, slice_summary["sha256"], *scenarios[i])
        for i in range(len(scenarios))
    }
    results_by_idx: dict[int, Any] = {}
    with ProcessPoolExecutor(max_workers=5) as pool:
        futs = {pool.submit(_worker_serveloop, task_map[i]): i for i in run_order}
        for fut in as_completed(futs):
            idx = futs[fut]
            item = fut.result()
            results_by_idx[idx] = item
            cap_label, cap_gib, pref_label, wp, tmode, cold_hbf, sl_res = item
            scen_id = f"{cap_label}__{pref_label}__{tmode}"
            print(
                f"  SL   [{scen_id:38s}]: iters={sl_res['iterations']:6d}, preempt={sl_res['preemptions']:4d}, "
                f"recomp={sl_res['recompute_write_frac']*100:4.1f}%, B_med={sl_res['batch_median']:4.1f}, "
                f"tpot_p50={sl_res['tpot_p50_ms']:5.2f}ms, kv_w={sl_res['kv_write_gb']:6.1f}GB "
                f"(p2m10={sl_res['write_peak_to_mean_10s']:.2f}), "
                f"PEC/d(1e5)={sl_res['pec_per_day_x1e5'] if sl_res['pec_per_day_x1e5'] is not None else -1:6.1f} "
                f"[{sl_res['wear_status']}], wall={sl_res['wall_s']:.1f}s",
                flush=True,
            )
    results_ordered = [results_by_idx[i] for i in range(len(scenarios))]

    for cap_label, cap_gib, pref_label, wp, tmode, cold_hbf, sl_res in results_ordered:
        scen_id = f"{cap_label}__{pref_label}__{tmode}"

        # Append ServeLoop row
        csv_rows.append({
            "scenario": scen_id,
            "capacity_regime": cap_label,
            "capacity_gib": cap_gib,
            "prefix_mode": pref_label,
            "timing_mode": tmode,
            "scheduler": "serveloop" if cold_hbf == 0 else "serveloop_hbf_cold",
            **{k: v for k, v in sl_res.items() if k not in ("req_tpot_p50_ms",)},
        })

        # Pair with our corresponding run if part of the 8 core scenarios
        if (cap_gib, wp) in ours_cache and cold_hbf == 0:
            o_res = ours_cache[(cap_gib, wp)]
            csv_rows.append({
                "scenario": scen_id,
                "capacity_regime": cap_label,
                "capacity_gib": cap_gib,
                "prefix_mode": pref_label,
                "timing_mode": tmode,
                "scheduler": "ours",
                **o_res,
                "wear_status": "ok",
            })
            b_diff = (
                100.0 * (o_res["batch_median"] - sl_res["batch_median"]) / sl_res["batch_median"]
                if sl_res["batch_median"] > 0 else 0.0
            )
            pec_diff_1e5 = (
                100.0 * (o_res["pec_per_day_x1e5"] - sl_res["pec_per_day_x1e5"]) / sl_res["pec_per_day_x1e5"]
                if sl_res["pec_per_day_x1e5"] else None
            )
            pec_diff_1e6 = (
                100.0 * (o_res["pec_per_day_x1e6"] - sl_res["pec_per_day_x1e6"]) / sl_res["pec_per_day_x1e6"]
                if sl_res["pec_per_day_x1e6"] else None
            )
            pair_summaries.append({
                "scenario": scen_id,
                "batch_median_diff_pct": round(b_diff, 2),
                "recompute_write_pct": round(100.0 * sl_res["recompute_write_frac"], 2),
                "pec_diff_x1e5_pct": None if pec_diff_1e5 is None else round(pec_diff_1e5, 2),
                "pec_diff_x1e6_pct": None if pec_diff_1e6 is None else round(pec_diff_1e6, 2),
                "pass_batch_10pct": abs(b_diff) <= 10.0,
                "pass_recomp_5pct": 100.0 * sl_res["recompute_write_frac"] <= 5.0,
                "pass_pec_5pct_x1e5": pec_diff_1e5 is not None and abs(pec_diff_1e5) <= 5.0,
                "pass_pec_5pct_x1e6": pec_diff_1e6 is not None and abs(pec_diff_1e6) <= 5.0,
            })

    # Write CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(csv_rows[0].keys())
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in csv_rows:
            w.writerow({k: _fmt(row.get(k)) for k in fieldnames})

    AUX_JSON.parent.mkdir(parents=True, exist_ok=True)
    AUX_JSON.write_text(json.dumps({
        "slice_summary": slice_summary,
        "hbfsim_calibration": calib,
        "pair_summaries": pair_summaries,
    }, indent=2), encoding="utf-8")

    print(f"\nWrote {OUT_CSV} ({len(csv_rows)} rows) and {AUX_JSON}")
    print("\n--- Criterion Check Summary ---")
    for ps in pair_summaries:
            pec_str = f"{ps['pec_diff_x1e5_pct']:+6.1f}%" if ps['pec_diff_x1e5_pct'] is not None else "   N/A "
            print(
                f"  {ps['scenario']:34s}: B_med_diff={ps['batch_median_diff_pct']:+6.1f}% "
                f"({'PASS' if ps['pass_batch_10pct'] else 'FAIL'}), "
                f"recomp={ps['recompute_write_pct']:4.1f}% "
                f"({'PASS' if ps['pass_recomp_5pct'] else 'FAIL'}), "
                f"PEC_diff(1e5)={pec_str} "
                f"({'PASS' if ps['pass_pec_5pct_x1e5'] else 'FAIL'})"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
