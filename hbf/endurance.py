"""Endurance of KV cache stored on HBF (module M4): Clock and closed-form steady state.

``analytic`` gives steady-state write, refresh and wear rates per stack for a closed-loop
(concurrency) or Poisson (rate) workload, instantly, as curves over X, N_PE and pages per block.
Simulated time comes from a ``Clock``: phase A's per-layer estimate est = layer_ns + round_ns x
rounds (fitted to MQSim, design-B F1, or refitted by ``calibrate`` against phase A runs) fed as
layer rows into the unchanged ``system.evaluate``. The request-level event simulation
(design-B section 4) is not implemented yet.

Spec rules used (OCP HBF v0.7.0):

* Writes are sequential within a NAND block, no skipping (section 5.4.1 items 4 and 9); writing
  page 0 auto-erases the block (item 5). Every page-0 write counts as one P/E cycle (assumption:
  spec-literal and conservative). There is no garbage collection (section 11.4).
* Read disturb: the host refreshes a block after X reads (section 11.5, product specific); the
  base die keeps a per-block page-read counter (section 4.4), assumed to reset at erase.
* Retention: 24 h power-on (section 9 Table 33); refresh "typically every 24-48 hours" (11.5).
* Replay (section 5.7) rewrites a block from page 0 to its last written page in place.

``python -m hbf.endurance --analytic --model llama-3.1-8b --format bf16,kivi2 --workload long32k
--concurrency 8 --rd-threshold 1e5,1e6,1e7 --pe-limit 1e4,1e5,1e6 --output DIR`` writes
endurance_summary.csv and report.md. ``build_parser``, ``configure``, ``run``, ``summarize_run``
and ``METRICS`` mirror ``hbf.cli`` for the sweep driver.
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np

from .interference import MODES as M5_MODES, hbf_gaps, interference
from .workload import load_workload, presets as workload_presets

GIB = 2 ** 30
DAY = 86400.0
CAUSES = ("prefill", "decode", "refresh_rd", "refresh_ret", "compaction", "replay", "wave", "padding")
POLICIES = ("packed", "dedicated", "layer_major", "hbm_decode", "vllm_pool")
PARTITIONS = {"none": 1024, "channel": 64, "die": 16}  # banks per partition (spec section 4.6 Figure 4)
F1 = {"layer_ns": 4000.0, "round_ns": 4406.0}  # design-B F1 fit: ocp, HBF command model, tR = 4,096 ns
HBF_FLAGS = ["--topology", "ocp", "--read-plane-ns", "0", "--coalesce-ns", "1", "--cache-read"]


# --------------------------------------------------------------------------------------------
# KV geometry

class Geometry:
    """Stored KV of one request: pages per context length, per-layer read demand.

    ``stored_pages`` counts full pages only (a partial last page stays in HBM staging);
    ``read_pages`` counts every page a dense step senses (ceil), as phase A does.
    """

    def __init__(self, layers, heads, tensors, page_bytes=4096):
        self.layers, self.heads, self.page_bytes = layers, heads, page_bytes
        self.tensors = tuple(tensors)
        self.hbf = tuple(spec for spec in self.tensors if spec.tier == "hbf")
        self.hbm = tuple(spec for spec in self.tensors if spec.tier == "hbm")
        if not self.hbf:
            raise ValueError("At least one tensor must be stored on HBF")
        self.tpp = tuple(page_bytes // spec.token_bytes for spec in self.hbf)
        if 0 in self.tpp:
            raise ValueError("A token row must fit in one page")
        self.data = tuple(spec for spec in self.tensors if spec.role == "data")

    @property
    def token_bytes(self):
        """KV bytes per token of all layers and heads (all tiers)."""
        return self.layers * self.heads * sum(spec.token_bytes for spec in self.tensors)

    @property
    def hbf_token_bytes(self):
        return self.layers * self.heads * sum(spec.token_bytes for spec in self.hbf)

    @staticmethod
    def window(spec, ctx):
        ctx = np.asarray(ctx, dtype=np.int64)
        lo, hi = spec.window or (None, None)
        lo = np.zeros_like(ctx) if lo is None else (ctx + lo if lo < 0 else np.full_like(ctx, lo))
        hi = ctx.copy() if hi is None else (ctx + hi if hi < 0 else np.full_like(ctx, hi))
        lo = np.clip(lo, 0, ctx)
        return lo, np.clip(hi, lo, ctx)

    def stored_pages(self, ctx):
        """Full HBF pages of a request with ``ctx`` tokens (all layers and heads)."""
        ctx = np.asarray(ctx, dtype=np.int64)
        total = np.zeros_like(ctx)
        for spec, tpp in zip(self.hbf, self.tpp):
            lo, hi = self.window(spec, ctx)
            total += (hi - lo) // tpp
        return self.layers * self.heads * total

    def row_pages(self, ctx, role_share=None):
        """Pages one layer senses per step (ceil), summed over requests, scaled per role."""
        ctx = np.asarray(ctx, dtype=np.int64)
        total = 0.0
        for spec, tpp in zip(self.hbf, self.tpp):
            lo, hi = self.window(spec, ctx)
            share = 1.0 if role_share is None else role_share.get(spec.role, 1.0)
            total += float(np.sum(-(-(hi - lo) // tpp))) * share
        return self.heads * total

    def partial_pages(self):
        """Pages of HBM staging per running request: one partial page per (layer, head, HBF tensor)."""
        return self.layers * self.heads * len(self.hbf)

    def demand(self, ctx, hbm_tokens=None):
        """Per-layer demand of one dense step over requests with ``ctx`` tokens (phase A columns)."""
        ctx = np.asarray(ctx, dtype=np.int64)
        extra = np.zeros_like(ctx) if hbm_tokens is None else np.asarray(hbm_tokens, dtype=np.int64)
        burst = hbm = elements = dequant = 0.0
        for spec in self.tensors:
            lo, hi = self.window(spec, ctx)
            tokens = hi - lo
            if spec.tier == "hbf":
                full = tokens * spec.token_bytes // self.page_bytes
                rest = tokens * spec.token_bytes - full * self.page_bytes
                burst += float(np.sum(full * self.page_bytes + -(-rest // 64) * 64))
            else:
                hbm += float(np.sum(tokens)) * spec.token_bytes
            if spec.role == "data":
                elements += float(np.sum(tokens)) * spec.dims
            dequant += float(np.sum(tokens)) * spec.dims * spec.dequant_ops
        hbm += float(np.sum(extra)) * sum(spec.token_bytes for spec in self.hbf)
        los, his = zip(*(self.window(spec, ctx) for spec in self.data)) if self.data else ((ctx * 0,), (ctx,))
        attended = float(np.sum(np.max(his, axis=0) - np.min(los, axis=0)))
        h = self.heads
        return {"burst_bytes": h * burst, "hbm_bytes": h * hbm, "attended_tokens": h * attended,
                "attended_elements": h * elements, "dequant_ops": h * dequant}


def geometry_for(model, kv_format=None, page_bytes=4096):
    from .formats import parse_format
    fmt = parse_format(kv_format or "bf16", model.head_dim)
    return Geometry(model.layers, model.kv_heads, fmt.tensors, page_bytes), fmt


# --------------------------------------------------------------------------------------------
# Clock

@dataclass(frozen=True)
class StepTime:
    tpot_ns: float
    gpu_ns: float
    hbm_tpot_ns: float
    h_ns: float  # HBF read time of one layer
    gaps_ns: tuple  # HBF-idle time before each layer's read (sums to tpot - layers x h)
    read_pages: float  # pages one layer senses
    rounds: int
    batch: int


class Clock:
    """Simulated decode step time of a running set (per stack), from phase A's fitted estimate.

    est per layer = layer_ns + round_ns x rounds, rounds = ceil(kappa x pages / banks) (or the
    busiest partition), fed as identical layer rows into ``system.evaluate`` with compute batch
    b x stacks and analysis batch b. ``ratios`` scale the dense demand for a sparse mode
    (pages, attended tokens and elements, burst, HBM bytes, dequant), from ``calibrate``.
    ``rounds_fn(b, ctx)``, when given, returns exact rounds for b equal contexts (phase A layout).
    """

    def __init__(self, model, gpu, geometry, options=None, *, read_ns=4096, banks=1024, stacks=1,
                 layer_ns=F1["layer_ns"], round_ns=F1["round_ns"], balance=1.0, ratios=None, selection="static",
                 kv_format="bf16", rounds_fn=None):
        from .system import SystemOptions
        self.model, self.gpu, self.geometry = model, gpu, geometry
        self.options = options or SystemOptions()
        self.read_ns, self.banks, self.stacks = read_ns, banks, stacks
        self.layer_ns, self.round_ns, self.balance = layer_ns, round_ns, balance
        self.ratios = dict(ratios or {})
        self.selection, self.kv_format, self.rounds_fn = selection, kv_format, rounds_fn
        self.cache = {}

    def describe(self):
        return {"layer_ns": self.layer_ns, "round_ns": self.round_ns, "balance": self.balance,
                "ratios": self.ratios, "read_ns": self.read_ns, "banks": self.banks, "stacks": self.stacks,
                "selection": self.selection, "kv_format": self.kv_format, "overlap": self.options.overlap}

    def step(self, contexts, partitions=None, partition_banks=None, hbm_tokens=None):
        """StepTime of one decode step, or None when nothing runs."""
        ctx = np.asarray(contexts, dtype=np.int64)
        b = len(ctx)
        if not b:
            return None
        geo, r = self.geometry, self.ratios
        hbf_ctx = ctx if hbm_tokens is None else ctx - np.asarray(hbm_tokens, dtype=np.int64)
        share = r.get("pages", 1.0)
        pages = geo.row_pages(hbf_ctx) * share
        if partitions is not None and partition_banks:
            parts = np.asarray(partitions)
            loads = [geo.row_pages(hbf_ctx[parts == p]) * share for p in np.unique(parts)]
            rounds = max(math.ceil(self.balance * load / partition_banks - 1e-9) for load in loads)
        elif self.rounds_fn is not None and len(set(ctx.tolist())) == 1 and hbm_tokens is None:
            rounds = self.rounds_fn(b, int(ctx[0]))
        else:
            rounds = math.ceil(self.balance * pages / self.banks - 1e-9)
        d = geo.demand(hbf_ctx, None if hbm_tokens is None else hbm_tokens)
        scale = {"burst_bytes": "burst", "hbm_bytes": "hbm", "attended_tokens": "attended",
                 "attended_elements": "elements", "dequant_ops": "dequant"}
        d = {key: value * r.get(scale[key], 1.0) for key, value in d.items()}
        key = (b, rounds) + tuple(int(round(math.log1p(d[k]) * 200)) for k in sorted(d))
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        from .system import evaluate
        est = self.layer_ns + self.round_ns * rounds if rounds else 0.0
        rows = [{"step": 0, "layer": layer, "est_ns": est, "bound_ns": rounds * self.read_ns} | d
                for layer in range(self.model.layers)]
        meta = {"batch": b * self.stacks, "stacks": self.stacks, "analysis_batch": b, "selection": self.selection,
                "kv_format": self.kv_format}
        result = evaluate(rows, self.model, self.gpu, self.options, meta)
        row, layers = result.steps[0], result.layers
        attn = [x["attn_staged_ns" if self.options.staged else "attn_ns"] for x in layers]
        h = [est] * len(layers)
        gaps = hbf_gaps(h, [x["pre_ns"] for x in layers], attn, [x["post_ns"] for x in layers],
                        self.options.overlap, self.options.staged)
        gaps[0] += max(0.0, row["tpot_hbf_est_ns"] - sum(h) - sum(gaps))
        out = StepTime(row["tpot_hbf_est_ns"], row["gpu_ns"], row["tpot_hbm_ns"], est, tuple(gaps), pages, rounds, b)
        self.cache[key] = out
        return out


class ConstantClock:
    """Fixed step time (tests and ablations): every step takes ``tpot_ns``, HBF idle half of it."""

    def __init__(self, tpot_ns, layers=1, idle_share=.5):
        self.tpot_ns, self.layers, self.idle_share = tpot_ns, layers, idle_share
        self.stacks = 1

    def describe(self):
        return {"constant_tpot_ns": self.tpot_ns}

    def step(self, contexts, partitions=None, partition_banks=None, hbm_tokens=None):
        if not len(contexts):
            return None
        gap = self.tpot_ns * self.idle_share / self.layers
        h = self.tpot_ns * (1 - self.idle_share) / self.layers
        return StepTime(self.tpot_ns, self.tpot_ns, self.tpot_ns, h, (gap,) * self.layers, 0.0, 0, len(contexts))


def layout_rounds(model, kv_format=None, layout="striped_contiguous", pages_per_block=256):
    """rounds_fn for Clock: exact phase A rounds of b equal requests (layer-major, dense) from the layout."""
    from .access import LAYOUTS, Shape
    from .config import load
    from .formats import parse_format
    cfg = load("ocp", pages_per_block=pages_per_block, blocks_per_plane=max(1, 131072 // pages_per_block))
    tensors = parse_format(kv_format or "bf16", model.head_dim).tensors
    cache = {}

    def rounds(b, ctx):
        if (b, ctx) not in cache:
            lay = LAYOUTS[layout](cfg, Shape(model.layers, model.kv_heads * b, model.head_dim, ctx), tensors=tensors)
            layers = [0] if layout.startswith("striped") else range(model.layers)
            worst = 0
            for layer in layers:
                lpas = [lay.lpa_array(layer, np.repeat(np.arange(lay.shape.kv_heads), lay.block_counts[t]),
                                      np.full(lay.shape.kv_heads * lay.block_counts[t], t),
                                      np.tile(np.arange(lay.block_counts[t]), lay.shape.kv_heads))
                        for t in range(len(lay.tensors))]
                worst = max(worst, int(np.bincount(np.concatenate(lpas) % cfg.units).max()))
            cache[b, ctx] = worst
        return cache[b, ctx]
    return rounds


# --------------------------------------------------------------------------------------------
# Calibration against phase A

def _csv(path):
    if not path.is_file():
        return []
    with path.open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row.get("group", "1") == "1"]
    out = []
    for row in rows:
        parsed = {}
        for key, value in row.items():
            try:
                parsed[key] = float(value) if value not in ("", None) else None
            except ValueError:
                parsed[key] = value
        out.append(parsed)
    return out


def phase_a(directory, model, gpu, kv_format, batch, context, *, mode="dense", generation=None, device=(),
            alloc="layer_major", partition="none"):
    """Run one analysis-only phase A pipeline (``hbf.cli``) into ``directory``; returns its CSV rows."""
    from .cli import build_parser, configure, pipeline
    argv = HBF_FLAGS + ["--layout", "striped_contiguous", "--model", model, "--gpu", gpu,
                        "--format", kv_format or "bf16", "--mode", mode, "--context", str(context),
                        "--batch", str(batch), "--analysis-only", "--timing", "--output", str(directory)]
    for key, value in (generation or {}).items():
        argv += [f"--{key.replace('_', '-')}", str(value)]
    argv += list(device)
    if alloc != "layer_major" or partition != "none":
        argv += ["--alloc", alloc, "--partition", partition]
    parser = build_parser()
    args = parser.parse_args(argv)
    run = configure(args)
    saved, sys.argv = sys.argv, ["hbf.cli", *argv]
    try:
        pipeline(directory=directory, binary=None, **run)
    finally:
        sys.argv = saved
    return {"layers": _csv(directory / "analysis_layers.csv"), "steps": _csv(directory / "analysis_steps.csv"),
            "system": _csv(directory / "system_steps.csv")}


def calibration_batches(model, kv_format, context, rounds=(32, 128), limit=256):
    """Batches whose dense layer-major runs read about ``rounds`` rounds per layer (the F1 fit range)."""
    from .model import load_model
    geometry, _ = geometry_for(load_model(model), kv_format)
    pages = geometry.row_pages(np.array([context]))
    low = min(limit - 1, max(1, math.ceil(rounds[0] * 1024 / pages)))
    return low, min(limit, max(low + 1, math.ceil(rounds[1] * 1024 / pages)))


def calibrate(directory, model, gpu="b200", kv_format="bf16", *, mode="dense", generation=None, context=32768,
              batches=None, device=(), policy="packed", partition="none", snapshot_batch=None):
    """Fit the Clock to at most three phase A analysis-only runs, cached by a hash of the inputs.

    Two dense layer-major points (``batches``; default about 32 and 128 rounds per layer, see
    ``calibration_batches``) refit layer_ns and round_ns for the actual device flags (least squares
    of est_ns over rounds per layer). One snapshot of the policy (``--alloc``,
    M2) at ``snapshot_batch`` (default the larger batch) in the run's mode gives kappa = measured rounds over the Clock's
    rounds at kappa 1 (ceil(pages / 1,024), or the busiest partition's), the demand ratios of
    the mode against dense at the same batch, and the per-step hot factor (max block reads over
    ppb x read share). Pages per block do not change rounds or est, so they are fixed at 256.
    Returns the fields for ``Clock`` plus diagnostics.
    """
    import shutil
    batches = tuple(batches or calibration_batches(model, kv_format, context))
    inputs = {"model": model, "gpu": gpu, "format": kv_format, "mode": mode, "generation": generation or {},
              "context": context, "batches": list(batches), "device": list(device), "policy": policy,
              "partition": partition, "snapshot_batch": snapshot_batch, "version": 3}
    digest = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()[:16]
    root = Path(directory) / f"calib-{digest}"
    cached = root / "calibration.json"
    if cached.is_file():
        return json.loads(cached.read_text())
    root.mkdir(parents=True, exist_ok=True)

    def fresh(name):
        target = root / name
        if target.exists():
            shutil.rmtree(target)
        return target

    points, dense = [], {}
    for b in batches:
        dense[b] = phase_a(fresh(f"dense-b{b}"), model, gpu, kv_format, b, context, device=device)
        points += [(row["rounds"], row["est_ns"]) for row in dense[b]["layers"] if row["pages"]]
    rounds = np.array([point[0] for point in points], dtype=float)
    est = np.array([point[1] for point in points], dtype=float)
    round_ns, layer_ns = F1["round_ns"], F1["layer_ns"]
    if len(set(rounds.tolist())) >= 2:
        round_ns, layer_ns = (float(value) for value in np.polyfit(rounds, est, 1))
    if layer_ns < 0 or len(set(rounds.tolist())) < 2:  # keep F1's intercept, fit the slope only
        layer_ns = F1["layer_ns"]
        round_ns = float(np.sum(rounds * (est - layer_ns)) / np.sum(rounds * rounds))
    b = snapshot_batch or max(batches)
    snap_policy = {"hbm_decode": "packed"}.get(policy, policy)
    snap_policy = snap_policy if snap_policy in ("packed", "dedicated", "vllm_pool") else "layer_major"
    snap_part = partition if snap_policy in ("packed", "dedicated") else "none"
    if mode == "dense" and snap_policy == "layer_major" and b in dense:
        snapshot = dense[b]
    else:
        snapshot = phase_a(fresh(f"snapshot-{mode}-{snap_policy}-{snap_part}-b{b}"), model, gpu, kv_format, b,
                           context, mode=mode, generation=generation, device=device, alloc=snap_policy,
                           partition=snap_part)
    kappas = []
    for row in snapshot["layers"]:
        if not row["pages"]:
            continue
        if snap_part == "none":
            ideal = math.ceil(row["pages"] / 1024)
        else:  # requests spread over partitions: the busiest holds ceil(b / parts) of them
            parts = 1024 // PARTITIONS[snap_part]
            ideal = math.ceil(math.ceil(b / parts) * row["pages"] / b / PARTITIONS[snap_part])
        kappas.append(row["rounds"] / ideal)
    def total(rows, key):
        return sum(row.get(key) or 0 for row in rows)
    ratios = {}
    if mode != "dense":
        for name, key in (("pages", "pages"), ("attended", "attended_tokens"), ("elements", "attended_elements"),
                          ("burst", "burst_bytes"), ("hbm", "hbm_bytes"), ("dequant", "dequant_ops")):
            base = total(dense[b]["layers"], key)
            ratios[name] = total(snapshot["layers"], key) / base if base else 1.0
    share = ratios.get("pages", 1.0)
    max_reads = max((row.get("max_block_reads") or 0) for row in snapshot["steps"]) if snapshot["steps"] else 0
    result = {"layer_ns": layer_ns, "round_ns": round_ns, "balance": float(np.mean(kappas)) if kappas else 1.0,
              "ratios": ratios, "read_share": share, "hot_step": max_reads / (256 * share) if share else 1.0,
              "max_block_reads": max_reads,
              "batches": list(batches),
              "phase_a_tpot_ns": {f"b{key}": value["system"][0]["tpot_hbf_est_ns"]
                                  for key, value in dense.items() if value["system"]},
              "snapshot": {"policy": snap_policy, "partition": snap_part, "batch": b,
                           "tpot_ns": snapshot["system"][0]["tpot_hbf_est_ns"] if snapshot["system"] else None},
              "inputs": inputs, "points": len(points)}
    cached.write_text(json.dumps(result, indent=2) + "\n")
    return result


# --------------------------------------------------------------------------------------------
# Closed-form steady state (M4a)

@dataclass(frozen=True)
class Device:
    """One HBF stack for the endurance model. Capacity is independent of pages per block."""
    banks: int = 1024  # spec section 4.6 Figure 4: 16 banks x 4 dies x 16 host channels
    pages_per_block: int = 256
    page_bytes: int = 4096  # spec section 4.1
    capacity_gib: float = 512.0  # spec Table 3: 512 GiB per stack
    partition: str = "none"

    def __post_init__(self):
        if self.partition not in PARTITIONS:
            raise ValueError(f"partition must be one of {', '.join(PARTITIONS)}")
        if self.banks % PARTITIONS[self.partition] and self.partition != "none":
            raise ValueError("banks must be a multiple of the partition size")
        if min(self.banks, self.pages_per_block, self.page_bytes) < 1 or self.capacity_gib <= 0:
            raise ValueError("banks, pages per block, page bytes and capacity must be positive")

    @property
    def capacity_pages(self):
        return int(self.capacity_gib * GIB // self.page_bytes)

    @property
    def segment_banks(self):
        return self.banks if self.partition == "none" else PARTITIONS[self.partition]

    @property
    def parts(self):
        return self.banks // self.segment_banks

    @property
    def segment_pages(self):
        return self.segment_banks * self.pages_per_block


def prefill_tok_s(model, gpu, tokens):
    """Analytic prefill rate: FLOP/token = 2P + 2 L (q_heads d_h) S at the GPU's compute_eff (params.md section 6)."""
    params = model.layers * model.layer_params + model.embedding_params
    return gpu.flops / (2 * params + 2 * model.layers * model.q_heads * model.head_dim * max(tokens, 1))


def threshold_steps(x, ppb, share=1.0, hot=1.0):
    """Decode steps until a block of fully read KV reaches X reads: X / (ppb x share x hot)."""
    return x / (ppb * share * hot)


def erase_amp(pages, segment_pages):
    """Erase amplification of entries that own whole segments: ceil(P / P_seg) P_seg / P."""
    return math.ceil(pages / segment_pages - 1e-9) * segment_pages / pages if pages > 0 else 1.0


def region_pages(model, tensors, slots, prompt, pages_per_block=256, layout="striped_contiguous"):
    """Pages a layer_major wave writes: phase A's layout of H x slots heads at the prompt length, padding included."""
    from .access import LAYOUTS, Shape
    from .config import load
    cfg = load("ocp", pages_per_block=pages_per_block, blocks_per_plane=max(1, 131072 // pages_per_block))
    lay = LAYOUTS[layout](cfg, Shape(model.layers, model.kv_heads * slots, model.head_dim, int(prompt)),
                          tensors=tensors)
    return float(lay.span)


def sci(value):
    """Compact label of a limit or threshold: 1e5, 3e3, 1.5e6; plain below 1,000."""
    if value < 1000:
        return f"{value:g}"
    exponent = math.floor(math.log10(value) + 1e-12)
    return f"{value / 10 ** exponent:g}e{exponent}"


def _pages(geometry, tokens):
    return float(geometry.stored_pages(np.array([int(round(tokens))]))[0])


def analytic(geometry, clock, workload, device=None, *, policy="packed", x_values=(1e6,), pe_limits=(1e4, 1e5, 1e6),
             share=1.0, hot=1.0, retention_s=86400.0, prefix_reads="per_request", t_prog=100000.0,
             t_bers=1000000.0, m5="gaps", suspend_ns=None, block_unit="die", stacks=1, align="row", region=None):
    """Steady-state rows per stack, one per X, for a closed-loop or Poisson workload (design-B section 3).

    * Rate: closed loop keeps ``concurrency / stacks`` requests (sessions for an agentic workload)
      in flight per stack; lambda = B / (out x T), T = Clock TPOT at the mean decode context.
    * Pages per request P_r = L H sum_t floor(w_t / tpp_t) (full pages; partial pages die in HBM).
    * Read disturb: rho = ppb x share x hot block reads per decode step; a page read for s steps is
      refreshed floor(s rho / X) times. For a request all its pages get its own s = out
      (conservative: generated pages are counted as if present from the start). Session data
      written in call k is read in every later call (out steps each, plus one prefill read).
    * Retention: data alive longer than T_ret is rewritten floor(life / T_ret) times.
    * P/E per block per day = programs x erase amplification x 86,400 / C (ideal wear levelling,
      the spec's AVGPEC, section 5.9.2.18).
    ``region`` (layer_major) is the wave's written pages; default phase A's layout via the clock's model.
    """
    device = device or Device()
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {', '.join(POLICIES)}")
    if m5 not in M5_MODES + ("off",):
        raise ValueError(f"m5 must be one of {', '.join(M5_MODES + ('off',))}")
    page, ppb = device.page_bytes, device.pages_per_block
    seg_banks, seg_pages, capacity = device.segment_banks, device.segment_pages, device.capacity_pages
    rho = ppb * share * hot
    w = workload
    n_in, n_out = w.input.mean(), w.output.mean()
    sessions = w.sessions
    calls = w.calls if sessions else 1
    h0 = w.history.mean() if sessions and w.history is not None else 0.0
    prefix = w.prefix_tokens if w.prefix_share else 0
    hbm_out = policy in ("hbm_decode", "layer_major")
    if sessions:
        ctx = h0 + n_in + (calls - 1) / 2 * (n_in + n_out) + n_out / 2
    else:
        ctx = w.prefix_share * prefix + n_in + n_out / 2
    active = w.concurrency / stacks if w.arrival == "closed" else None
    b = max(1, math.ceil(active - 1e-9)) if active is not None else 1

    def step_of(batch):
        parts = (np.arange(batch) % device.parts) if device.parts > 1 else None
        return clock.step(np.full(batch, int(round(ctx))), parts, seg_banks if parts is not None else None,
                          np.full(batch, int(n_out / 2)) if hbm_out else None)

    out_wave = w.output.mean_max(b) if policy == "layer_major" else n_out
    for _ in range(40):
        step = step_of(b)
        tpot = step.tpot_ns / 1e9
        call_s = n_out * tpot
        if active is not None:
            new_b = max(1, round(active * call_s / (call_s + w.think_s))) if sessions else b
        else:
            new_b = max(1, round(w.rate / stacks * calls * call_s))
        if new_b == b:
            break
        b = new_b
        out_wave = w.output.mean_max(b) if policy == "layer_major" else n_out
    if policy == "layer_major":
        calls_per_s = b / (out_wave * tpot)
    elif active is not None:
        calls_per_s = active / (call_s + w.think_s) if sessions else b / call_s
    else:
        calls_per_s = w.rate / stacks * calls
    entry_rate = calls_per_s / calls
    tokens_per_s = calls_per_s * n_out

    # Chunks of one request (or session): (pages, cause, block reads over the chunk's life, life in s).
    chunks = []
    if sessions:
        history = 0.0
        for k in range(calls):
            fresh = h0 + n_in if k == 0 else n_in
            pre = _pages(geometry, history + fresh) - _pages(geometry, history)
            dec = _pages(geometry, history + fresh + n_out) - _pages(geometry, history + fresh)
            later = calls - 1 - k
            reads = (later + 1) * n_out * rho + later * ppb
            life = (later + 1) * (call_s + w.think_s) + w.retain_s
            chunks.append((pre, "prefill", reads, life))
            if not hbm_out:
                chunks.append((dec, "decode", reads - n_out / 2 * rho, life))
            history += fresh + n_out
    else:
        hit = w.prefix_share
        own = hit * (_pages(geometry, prefix + n_in) - _pages(geometry, prefix)) + (1 - hit) * _pages(geometry, n_in)
        full = (hit * (_pages(geometry, prefix + n_in + n_out) - _pages(geometry, prefix))
                + (1 - hit) * _pages(geometry, n_in + n_out))
        life = call_s + w.retain_s
        if policy == "layer_major":
            life = out_wave * tpot
        chunks.append((own, "prefill", n_out * rho, life))
        if not hbm_out and full > own:
            chunks.append((full - own, "decode", n_out * rho, life))
    host_pages = sum(chunk[0] for chunk in chunks)
    if policy == "layer_major":
        if region is None:
            model = getattr(clock, "model", None)
            if model is None:
                raise ValueError("layer_major needs region= or a Clock with a model")
            region = region_pages(model, geometry.tensors, b, max(1, round(w.input.mean_max(b))), ppb)
        per_request = region / b  # each request's share of the wave's region

    rows = []
    for x in x_values:
        rd = ret = numerator = 0.0
        for pages, _, reads, life in chunks:
            rd += math.floor(reads / x + 1e-9) * pages
            numerator += reads * pages
            ret += math.floor(life / retention_s + 1e-9) * pages if retention_s else 0.0
        prefix_rd = prefix_ret = 0.0
        if prefix and w.prefix_groups:
            p_pages = _pages(geometry, prefix)
            weights = 1.0 / np.arange(1, w.prefix_groups + 1) ** w.zipf
            for p in weights / weights.sum():
                readers = b * w.prefix_share * p
                k = readers if prefix_reads == "per_request" else min(1.0, readers)
                per_s = k * rho / tpot + calls_per_s * w.prefix_share * p * ppb  # decode + prefill reads
                prefix_rd += p_pages * per_s / x
                numerator += p_pages * per_s / entry_rate
            prefix_ret = w.prefix_groups * p_pages / retention_s if retention_s else 0.0
        rates = dict.fromkeys(CAUSES, 0.0)
        rates["prefill"] = entry_rate * sum(c[0] for c in chunks if c[1] == "prefill")
        rates["decode"] = entry_rate * sum(c[0] for c in chunks if c[1] == "decode")
        if policy == "layer_major":
            # The whole region is replayed in place (no skipping) once its blocks reach X reads.
            rates["padding"] = entry_rate * (per_request - host_pages)
            rates["replay"] = entry_rate * per_request * (math.floor(n_out * rho / x + 1e-9)
                                                          + (math.floor(out_wave * tpot / retention_s + 1e-9)
                                                             if retention_s else 0))
            amp = erase_amp(region, seg_pages)
            refreshes = rates["replay"] / (entry_rate * per_request)
        else:
            rates["refresh_rd"] = entry_rate * rd + prefix_rd
            rates["refresh_ret"] = entry_rate * ret + prefix_ret
            if align == "row" and policy in ("packed", "hbm_decode", "vllm_pool"):
                rates["padding"] = entry_rate * (seg_banks - 1) / 2  # prompts start on a row
            amp = erase_amp(host_pages, seg_pages) if policy == "dedicated" else 1.0
            refreshes = rd / host_pages if host_pages else 0.0
        programs = sum(rates.values())
        pe_day = programs * amp * DAY / capacity
        kv = rates["prefill"] + rates["decode"]
        m5_row = {"extra_step_ns": 0.0, "max_step_extra_ns": 0.0}
        if m5 != "off":
            m5_row = interference({"programs": programs * tpot, "erases": programs * amp / ppb * tpot,
                                   "wave_banks": seg_banks, "gaps_ns": step.gaps_ns, "step_ns": step.tpot_ns},
                                  m5, t_prog, t_bers, suspend_ns, block_unit=block_unit)
        row = {"workload": w.name, "policy": policy, "partition": device.partition, "pages_per_block": ppb,
               "x": x, "stacks": stacks, "mean_batch": b, "mean_context": ctx, "mean_tpot_ms": tpot * 1e3,
               "tpot_m5_ms": (step.tpot_ns + m5_row["extra_step_ns"]) / 1e6,
               "max_step_extra_ms": m5_row["max_step_extra_ns"] / 1e6, "req_per_s": calls_per_s,
               "tokens_per_s": tokens_per_s, "pages_per_request": host_pages / calls,
               "kv_bytes_per_token": geometry.token_bytes, "threshold_steps": threshold_steps(x, ppb, share, hot),
               "refreshes_per_request": refreshes, "kv_write_gb_s": kv * page / 1e9}
        row |= {f"{cause}_gb_s": rates[cause] * page / 1e9 for cause in CAUSES}
        refresh = rates["refresh_rd"] + rates["refresh_ret"] + rates["replay"]
        row |= {"program_gb_s": programs * page / 1e9,
                "program_bw_share": programs * page / (device.banks * page / (t_prog / 1e9)),
                "wa": programs / kv if kv else None, "erase_amp": amp,
                "erase_wa": programs * amp / kv if kv else None,
                "refresh_to_kv": refresh / kv if kv else None, "erases_per_s": programs * amp / ppb,
                "avg_pec_per_day": pe_day, "max_pec_per_day": pe_day}
        for limit in pe_limits:
            row[f"days_to_limit_avg@{sci(limit)}"] = limit / pe_day if pe_day else math.inf
        for limit in pe_limits:
            row[f"days_to_limit_max@{sci(limit)}"] = limit / pe_day if pe_day else math.inf
        row["pe_per_mtok"] = pe_day / (tokens_per_s * DAY / 1e6) if tokens_per_s else None
        row["break_even_x"] = numerator * entry_rate / kv if kv and policy != "layer_major" else (
            n_out * rho if policy == "layer_major" else None)
        partial = geometry.partial_pages() * page
        row["staging_peak_mib"] = (b * partial + seg_banks * page
                                   + (b * n_out * geometry.token_bytes if hbm_out else 0)) / 2 ** 20
        row["pending_units_per_channel"] = b * geometry.partial_pages() / 16
        row["unit_open_s"] = max(geometry.tpp) * tpot
        row["replay_buffer_mib"] = seg_pages * page / 2 ** 20 if policy == "layer_major" else 0.0
        row["legal"] = policy != "vllm_pool"
        rows.append(row)
    return rows


# --------------------------------------------------------------------------------------------
# Command line, outputs and report

METRICS = ("mean_batch", "mean_context", "mean_tpot_ms", "tpot_m5_ms", "max_step_extra_ms", "req_per_s",
           "tokens_per_s", "pages_per_request", "threshold_steps", "refreshes_per_request", "kv_write_gb_s") + tuple(
    f"{cause}_gb_s" for cause in CAUSES) + (
    "program_gb_s", "program_bw_share", "wa", "erase_amp", "erase_wa", "refresh_to_kv", "erases_per_s", "avg_pec_per_day",
    "max_pec_per_day", "pe_per_mtok", "break_even_x", "staging_peak_mib", "pending_units_per_channel",
    "unit_open_s", "replay_buffer_mib", "legal")
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
    parser = argparse.ArgumentParser(prog="python -m hbf.endurance", description=__doc__.split("\n\n")[0])
    parser.add_argument("--analytic", action="store_true",
                        help="closed-form steady state (the event simulation is not implemented yet)")
    parser.add_argument("--model", default="llama-3.1-8b", help="model preset(s), comma separated")
    parser.add_argument("--gpu", default="b200", help="GPU preset (hbf/configs/gpus.json) or JSON")
    parser.add_argument("--format", dest="kv_format", default="bf16",
                        help="KV format(s), comma separated (hbf/formats.py presets, e.g. bf16,fp8,kivi2)")
    parser.add_argument("--mode", choices=SPARSE_MODES, default="dense",
                        help="read selection of a decode step (non-dense modes take ratios from calibration)")
    parser.add_argument("--fraction", type=float, default=.1, help="selected share (uniform_*, sink_recent)")
    parser.add_argument("--select-tokens", type=int, default=16, help="uniform_tokens block (default 16)")
    parser.add_argument("--count", type=int, default=4, help="sink_recent: sink tokens")
    parser.add_argument("--seed", type=int, default=1, help="seed of the calibration snapshot's selection")
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
    parser.add_argument("--output", type=Path)
    return parser


def configure(args):
    """Run configuration from parsed arguments (raises ValueError on bad input)."""
    from .model import load_gpu, load_model
    from .system import SystemOptions
    if not args.analytic:
        raise ValueError("The event simulation (design-B section 4) is not implemented yet; use --analytic")
    params = json.loads((Path(__file__).resolve().parent / "configs" / "parameters.json").read_text())
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
    if args.read_share is not None and not 0 < args.read_share <= 1:
        raise ValueError("--read-share must be in (0, 1]")
    if args.partition != "none" and args.alloc not in ("packed", "dedicated", "hbm_decode"):
        raise ValueError("--partition applies to packed, dedicated and hbm_decode")
    if args.m5 == "suspend" and args.suspend_ns is None:
        raise ValueError("--m5 suspend needs --suspend-ns (OCP v0.7.0 has no suspend; the value is product specific)")
    for fmt in formats:
        geometry_for(models[0], fmt)  # validates the format name and parameters
    device_flags = []
    for flag, value in (("--read-ns", args.read_ns), ("--aggregate-gb-s", args.aggregate_gb_s)):
        if value is not None:
            device_flags += [flag, str(value)]
    generation = {}
    if args.mode != "dense":
        generation = {"fraction": args.fraction, "select_tokens": args.select_tokens, "count": args.count,
                      "seed": args.seed}
    return {"models": models, "gpu": load_gpu(args.gpu), "gpu_name": args.gpu, "formats": formats,
            "mode": args.mode, "generation": generation, "policy": args.alloc, "partition": args.partition,
            "workloads": workloads, "pages_per_block": ppbs, "x_values": xs, "pe_limits": limits,
            "hot": args.rd_hot, "read_share": args.read_share,
            "retention_s": args.retention_h * 3600 if args.retention_h else 0.0,
            "prefix_reads": args.prefix_reads, "capacity_gib": args.capacity_gib, "stacks": args.stacks,
            "read_ns": args.read_ns or params["read_ns"], "device_flags": device_flags,
            "t_prog": args.program_ns or params["program_ns"], "t_bers": args.erase_ns or params["erase_ns"],
            "options": SystemOptions(overlap=args.overlap, kv_consume=args.kv_consume), "m5": args.m5,
            "suspend_ns": args.suspend_ns, "block_unit": args.block_unit, "calibrate": not args.no_calibrate,
            "calibration_dir": args.calibration_dir, "calib_context": args.calib_context}


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
    geometry, fmt = geometry_for(model, kv_format)
    selection = "static" if config["mode"] in ("dense", "sink_recent") else "dynamic"
    base = {"read_ns": config["read_ns"], "stacks": config["stacks"], "selection": selection,
            "kv_format": fmt.name, "options": config["options"]}
    if not config["calibrate"]:
        rounds_fn = layout_rounds(model, kv_format) if config["policy"] == "layer_major" else None
        return Clock(model, config["gpu"], geometry, rounds_fn=rounds_fn, **base), geometry, None
    cache = config["calibration_dir"] or Path(directory) / "calibration"
    calib = calibrate(cache, model.name, config["gpu_name"], kv_format, mode=config["mode"],
                      generation=config["generation"], context=_calib_context(workload, config["calib_context"]),
                      device=config["device_flags"], policy=config["policy"], partition=config["partition"],
                      snapshot_batch=snapshot_batch(workload, config["stacks"]))
    clock = Clock(model, config["gpu"], geometry, layer_ns=calib["layer_ns"], round_ns=calib["round_ns"],
                  balance=calib["balance"], ratios=calib["ratios"], **base)
    return clock, geometry, calib


def run(directory, **config):
    """Write meta.json, calibration.json, endurance_summary.csv and report.md into ``directory``; returns rows."""
    from .system import write_table
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    rows, calibrations = [], []
    for model, kv_format, workload in itertools.product(config["models"], config["formats"], config["workloads"]):
        clock, geometry, calib = make_clock(config, model, kv_format, workload, directory)
        if calib is not None:
            calibrations.append({"model": model.name, "format": kv_format, "workload": workload.name} | calib)
        share = config["read_share"] or (calib["read_share"] if calib else 1.0)
        if config["mode"] != "dense" and calib is None and config["read_share"] is None:
            share = config["generation"].get("fraction", 1.0)
            clock.ratios = {"pages": share}
        for ppb in config["pages_per_block"]:
            device = Device(pages_per_block=ppb, capacity_gib=config["capacity_gib"], partition=config["partition"])
            for row in analytic(geometry, clock, workload, device, policy=config["policy"],
                                x_values=config["x_values"], pe_limits=config["pe_limits"], share=share,
                                hot=config["hot"], retention_s=config["retention_s"],
                                prefix_reads=config["prefix_reads"], t_prog=config["t_prog"],
                                t_bers=config["t_bers"], m5=config["m5"], suspend_ns=config["suspend_ns"],
                                block_unit=config["block_unit"], stacks=config["stacks"]):
                rows.append({"model": model.name, "format": kv_format, "mode": config["mode"],
                             "read_share": share} | row)
    write_table(directory / "endurance_summary.csv", rows)
    meta = {key: value for key, value in config.items()
            if key not in ("models", "gpu", "workloads", "options", "calibration_dir")}
    meta |= {"models": [model.name for model in config["models"]], "gpu": config["gpu_name"],
             "workloads": [{"name": w.name, "description": w.describe(), "source": w.source}
                           for w in config["workloads"]],
             "options": {"overlap": config["options"].overlap, "kv_consume": config["options"].kv_consume},
             "tool": "endurance", "analytic": True, "argv": sys.argv}
    (directory / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str) + "\n")
    if calibrations:
        (directory / "calibration.json").write_text(json.dumps(calibrations, indent=2) + "\n")
    (directory / "report.md").write_text("\n".join(report_lines(rows, config, calibrations)) + "\n")
    return rows


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


def _fmt(value, digits=2):
    if value is None:
        return "—"
    if isinstance(value, float) and math.isinf(value):
        return "∞"
    if abs(value) >= 1e5 or (0 < abs(value) < 10 ** -digits):
        mantissa, exponent = f"{value:.2e}".split("e")
        return f"{mantissa}e{int(exponent)}"
    return f"{value:,.{digits}f}"


def report_lines(rows, config, calibrations=()):
    """Chinese report section 寿命评估 with every assumption and its source tag."""
    limits = config["pe_limits"]
    policy, partition = config["policy"], config["partition"]
    names = {"packed": "紧凑追加（packed）", "dedicated": "每请求独占段（dedicated）",
             "layer_major": "静态批次波（layer_major）", "hbm_decode": "生成的 KV 留在 HBM（hbm_decode）",
             "vllm_pool": "vLLM 空闲块池（vllm_pool，规范下不合法）"}
    lines = ["# 寿命评估（解析稳态）", "",
             f"分配方式：{names[policy]}，分区 `{partition}`；读取方式 `{config['mode']}`；"
             f"每个 stack 容量 {config['capacity_gib']:g} GiB，{config['stacks']} 个 stack 分担请求，"
             f"tPROG {config['t_prog'] / 1e3:g} µs，tBERS {config['t_bers'] / 1e3:g} µs，"
             f"保持刷新间隔 {config['retention_s'] / 3600:g} h（0 表示关闭），持续读偏斜 h = {config['hot']:g}。", ""]
    for w in config["workloads"]:
        lines.append(f"- 负载 {w.describe()}。来源：{w.source}")
    lines += ["", "下表每行是一个 stack 的稳态（写入速率、P/E 均按每个 stack 计）。P/E 每天按理想磨损均衡（AVGPEC）。"
              "“到 N 的天数”= N_PE / 每天 P/E。阈值步数 = X / (ppb × 读比例 × h)。写放大 = 编程页 / 主机 KV 页；"
              "擦除放大 = 擦除容量 / 编程页（dedicated 与 layer_major 大于 1）。等量 X：读扰刷新写入等于 KV 写入时的 X"
              "（连续近似）。批是每个 stack 平均同时解码的请求数。", "",
              "| 模型 | 格式 | 负载 | ppb | X | 批 | TPOT ms | TPOT′ ms | KV 写 GB/s | 读扰刷新 GB/s | 保持刷新 GB/s "
              "| 重放 GB/s | 编程 GB/s | 刷新/KV | 写放大 | 擦除放大 | P/E 每天 | "
              + " | ".join(f"到 {sci(limit)} 天" for limit in limits) + " | 阈值步数 | 刷新/请求 | 等量 X |",
              "|" + "---|" * 3 + "---:|" * (18 + len(limits))]
    for row in rows:
        cells = [row["model"], row["format"], row["workload"], str(row["pages_per_block"]), sci(row["x"]),
                 str(row["mean_batch"]), _fmt(row["mean_tpot_ms"]), _fmt(row["tpot_m5_ms"]),
                 _fmt(row["kv_write_gb_s"], 3), _fmt(row["refresh_rd_gb_s"], 3), _fmt(row["refresh_ret_gb_s"], 3),
                 _fmt(row["replay_gb_s"], 3), _fmt(row["program_gb_s"], 3), _fmt(row["refresh_to_kv"]),
                 _fmt(row["wa"]), _fmt(row["erase_amp"]), _fmt(row["avg_pec_per_day"], 1)]
        cells += [_fmt(row[f"days_to_limit_avg@{sci(limit)}"], 1) for limit in limits]
        cells += [_fmt(row["threshold_steps"], 1), _fmt(row["refreshes_per_request"]), _fmt(row["break_even_x"], 0)]
        lines.append("| " + " | ".join(cells) + " |")
    first = rows[0] if rows else None
    if first:
        lines += ["", "其他列（见 endurance_summary.csv）：每秒请求与 token、写入按原因拆分、阵列编程带宽占比"
                  f"（例如第一行 {first['program_bw_share']:.4f}）、HBM 暂存峰值 {first['staging_peak_mib']:.1f} MiB、"
                  f"逐页涓流写时每主机通道同时未完成的 4 KiB 单元 {first['pending_units_per_channel']:.0f} 个"
                  f"（每个要开 {first['unit_open_s']:.2f} s；规范第 5.4.1 节第 7 条举例的产品上限是每通道 64 或 128 个）。"]
    if calibrations:
        lines += ["", "## 时钟校准", "",
                  "时钟按 phase A 的每层估算 est = layer_ns + round_ns × 轮数（对 MQSim 校准过的估算），"
                  "两点（稠密 layer-major，批大小使每层约 32 与 128 轮）拟合 layer_ns、round_ns（截距为负时固定为 4,000 ns 只拟合斜率）；策略快照（`--alloc`）给出均衡系数 κ"
                  "（实测轮数 / 理想轮数）和稀疏读取相对稠密的需求比例。", "",
                  "| 模型 | 格式 | 负载 | 拟合批 | layer_ns | round_ns | 快照批 | κ | 读比例 | 单步热度 | 快照 phase A TPOT ms |",
                  "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for c in calibrations:
            tpot = c["snapshot"]["tpot_ns"]
            lines.append(f"| {c['model']} | {c['format']} | {c['workload']} | {c['batches']} | {c['layer_ns']:.0f} | "
                         f"{c['round_ns']:.1f} | {c['snapshot']['batch']} | {c['balance']:.4f} | {c['read_share']:.4f} | "
                         f"{c['hot_step']:.2f} | "
                         f"{_fmt(tpot / 1e6 if tpot else None, 3)} |")
    lines += ["", "## 结论要点", "",
              "- 稠密读取时，一个装满的块每步被读 ppb 次，所以读扰阈值 = X / ppb 步，与 TPOT 和压缩格式无关"
              "（X = 1e5 / 1e6 / 1e7、ppb = 256 时为 390.6 / 3,906 / 39,063 步）。只有活得比这更久的数据（长输出、"
              "会话历史、共享前缀）才需要读扰刷新；稀疏选择把阈值步数放大 1 / 读比例 倍。",
              "- 固定负载下，压缩按压缩比减少 KV 写入和 P/E；在 HBF 带宽受限的最大吞吐下，每秒感测页数基本不变，"
              "压缩省下的是每 token 的 P/E（pe_per_mtok），而不是每秒的写入字节。",
              "- 阵列编程带宽 = 1,024 × 4 KiB / tPROG（tPROG = 100 µs 时 41.9 GB/s），KV 写入远低于它；"
              "真正的代价是编程和擦除占用 die 造成的读延迟（TPOT′，见 hbf/interference.py）。",
              "", "## 假设", ""]
    lines += [f"- {item}" for item in assumptions(config)]
    return lines


def assumptions(config):
    """Assumption list of the endurance model, each with its source tag (params.md tags in brackets)."""
    items = [
        "写入顺序：NAND 块内地址连续、不得跳跃，写页 0（WL-0、STR-0）时自动擦除该块（OCP v0.7.0 第 5.4.1 节第 4、5、9 条）。"
        "第 4 条的“主机通道分配的全局地址范围内不得跳跃”按每个 NAND 块理解（与第 9 条和第 11.5.2.6 节一致）；"
        "若按每个主机通道字面理解，每个主机通道只能有一个追加日志，`dedicated` 不再合法。",
        "每次写页 0 都计一次 P/E，包括从未写过的块（规范字面、偏保守，假设）；块的读计数在擦除时清零（假设）。",
        "没有垃圾回收（第 11.4 节）；空间只能由主机回收（丢弃后重算、搬移或整块重写）。区域重映射（第 11.4.1 节）只改映射。",
        "读扰：主机在一个块被读 X 次后刷新它（第 11.5 节，产品相关；基片按块计读次数，第 4.4 节）。X 扫 1e5–1e7："
        "3D TLC 实测 5.5e4–5.2e5（STRAW 2025 [W]，作为未知 SLC 值的保守替代），经典 SLC 约 1e6（Cai 等 DSN 2015 [W]），"
        "1e7 是猜测 [G]（params.md 第 2、8 节）。",
        "稠密读取每步把每个 KV 页读一次，所以装满数据的块每步读 ρ = ppb × 读比例 × h 次；h（持续偏斜）未知，默认 1 [G]；"
        "稀疏模式的读比例取自校准快照（phase A 同批次的读页数 / 稠密读页数）。",
        "一个请求的所有页按该请求的解码步数 s 计刷新次数 ⌊s·ρ/X⌋（偏保守：生成的页按一开始就在计）；"
        "会话第 k 次调用写入的数据在之后每次调用中被读（每次 out 步，外加一次 prefill 读）。",
        "保持：数据寿命超过 T_ret 时每 T_ret 重写一次；T_ret = 24 h（第 9 节表 33 上电保持 24 h；第 11.5 节建议 24–48 h）。",
        "N_PE 由产品决定（第 9 节表 33 注 2，寄存器 MAXPEC/AVGPEC）；扫 1e4（pSLC 低端 [S]）、1e5（SPLASH、FlashAccel 的 SLC 假设 [P]）、"
        "1e6（FlashAccel 放宽保持后的 10× 假设 [P]）。",
        "每天 P/E = 编程页数 × 擦除放大 × 86,400 / 容量页数，即理想磨损均衡下的平均 PEC（AVGPEC，第 5.9.2.18 节）；"
        "实际最大 PEC 更高。",
        "容量取 --capacity-gib（默认 512 GiB，规范表 3 [P]），与每块页数无关。每块页数：第 11.5.1.8 节提示 1/2/4 MiB 块，"
        "即 4 KiB 页时 256/512/1024 页 [P]；SPLASH、FlashAccel 用 192 [P]。",
        "tPROG = 100 µs、tBERS = 1 ms 取自 SPLASH 表 1 [P]（hbf/configs/parameters.json）；规范未给出。",
        "packed：每个新条目（请求或会话）的 prompt 从一行（1,024 页，分区时 64/16 页）的起点开始，平均补 (行宽 − 1)/2 页填充（假设）；"
        "解码页先在 HBM 暂存，攒满一行再写；擦除放大为 1。",
        "dedicated：每个条目独占整段（超级块 1 GiB，分区时 64/16 MiB），擦除放大 = ⌈P/P_seg⌉·P_seg/P。",
        "layer_major：静态批次波，每波按 phase A 布局（H × B 个 head、波内最长 prompt）整区重写（含保留尾部的填充），"
        "生成的 KV 留在 HBM；读扰或保持到期时整段原地重放（第 5.7 节），需要一段的 HBM 副本。",
        "hbm_decode：生成的 KV 从不写入 HBF（SK hynix H³ 式，[S]）；HBM 暂存随输出长度增长。",
        "vllm_pool：磨损按 packed 计，但空闲块重用会写入仍有有效数据的块中间，违反第 5.4.1 节，规范下不合法。",
        "时钟：每层估算 est = layer_ns + round_ns × 轮数，轮数 = ⌈κ × 每层读页 / 1,024⌉（分区时取最忙分区）；"
        "F1 常数（4,000 ns + 4,406 ns × 轮数）只在 HBF 命令模型、tR = 4,096 ns 下与 MQSim 校准过；"
        "其余计算项、重叠方式和 HBM 基线沿用系统级评估（hbf/system.py）的假设。",
        "闭环：每个 stack 保持 concurrency / stacks 个请求（会话负载为会话数）在解码；prefill 分离（默认），"
        "KV 写入完成后才开始解码，不占解码时间。TPOT 取平均解码上下文下的值。",
        "共享前缀：per_request 时每个共享的请求每步各读一次前缀（序列级），once 时每步只读一次（级联注意力）；"
        "phase A 的按 (step, layer) 去重对应 once。",
        "写入干扰（M5）：默认 gaps 模式（主机把编程、擦除放进 HBF 空闲间隙，符合第 11.5 节“刷新与读不同时发往同一 die”），"
        "以 die 为阻塞单位；OCP v0.7.0 没有挂起（suspend）。只算一遍，不反馈到请求速率。",
        "逐页涓流写需要每个运行请求每 (层, head, HBF 张量) 一个未完成 4 KiB 单元，每个单元要开 tpp × TPOT；"
        "第 5.4.1 节第 6、7 条要求单元在主机设定的时限内写完，产品上限举例为每通道 64 或 128 个，所以按行攒满再写。",
        "权重放在 HBM。若权重放在 HBF（规范第 13 节的用例），权重每 token 都被读，权重块每 X/ppb 步就要刷新一次，不在本模型范围内。",
        "速率按每个 stack 计；N 个 stack 时每个 stack 承担 1/N 的请求，GPU 计算批为 b × N。",
    ]
    return items


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    directory = args.output or Path("runs") / f"endurance-{now}-{uuid4().hex[:8]}"
    try:
        config = configure(args)
        run(directory, **config)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(Path(directory).resolve())


if __name__ == "__main__":
    main()
