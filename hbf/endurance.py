"""Endurance of KV cache stored on HBF (module M4): Clock, closed-form steady state and event simulation.

``analytic`` gives steady-state write, refresh and wear rates per stack for a closed-loop
(concurrency) or Poisson (rate) workload, instantly, as curves over X, N_PE and pages per block.
``simulate`` runs the segment-level event simulation (design-B section 4).
Simulated time comes from a ``Clock``: phase A's per-layer estimate est = layer_ns + round_ns x
rounds (fitted to MQSim, design-B F1, or refitted by ``calibrate`` against phase A runs) fed as
layer rows into the unchanged ``system.evaluate``.

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
import heapq
import itertools
import json
import math
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np

from .interference import MODES as M5_MODES, hbf_gaps, interference
from .workload import Stream, load_workload, per_stack, presets as workload_presets

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
        if spec.window is None:
            hi = np.maximum(ctx, 0)
            return np.zeros_like(hi), hi
        lo, hi = spec.window
        lo = np.zeros_like(ctx) if lo is None else (ctx + lo if lo < 0 else np.full_like(ctx, lo))
        hi = ctx.copy() if hi is None else (ctx + hi if hi < 0 else np.full_like(ctx, hi))
        lo = np.clip(lo, 0, ctx)
        return lo, np.clip(hi, lo, ctx)

    def _tokens(self, spec, ctx):
        if spec.window is None:
            return np.maximum(ctx, 0)
        lo, hi = self.window(spec, ctx)
        return hi - lo

    def stored_pages(self, ctx):
        """Full HBF pages of a request with ``ctx`` tokens (all layers and heads)."""
        ctx = np.asarray(ctx, dtype=np.int64)
        total = np.zeros_like(ctx)
        for spec, tpp in zip(self.hbf, self.tpp):
            total += self._tokens(spec, ctx) // tpp
        return self.layers * self.heads * total

    @staticmethod
    def _raw_bounds(spec, ctx):
        ctx = np.asarray(ctx, dtype=np.int64)
        upper = np.maximum(ctx, 0)
        if spec.window is None:
            return np.zeros_like(upper), upper
        lo_s, hi_s = spec.window
        lo = np.zeros_like(ctx) if lo_s is None else (ctx + lo_s if lo_s < 0 else np.full_like(ctx, lo_s))
        hi = ctx.copy() if hi_s is None else (ctx + hi_s if hi_s < 0 else np.full_like(ctx, hi_s))
        lo = np.clip(lo, 0, upper)
        hi = np.clip(hi, 0, upper)
        return np.minimum(lo, hi), hi

    def written_pages(self, c0, c1):
        """Full HBF pages written as context grows from ``c0`` to ``c1`` (``c1 >= c0``).

        For non-sliding windows (``lo is None`` or ``lo >= 0``) this equals
        ``stored_pages(c1) - stored_pages(c0)``. For sliding windows (``lo < 0``) admission
        (``c0 <= 0``) writes ``(hi(c1) - lo(c1)) // tpp`` while decode (``c0 > 0``) writes every
        page entering ``hi`` even though ``stored_pages`` stays constant.
        """
        c0 = np.asarray(c0, dtype=np.int64)
        c1 = np.asarray(c1, dtype=np.int64)
        total = np.zeros(np.broadcast_shapes(c0.shape, c1.shape), dtype=np.int64)
        for spec, tpp in zip(self.hbf, self.tpp):
            if spec.window is not None and spec.window[0] is not None and spec.window[0] < 0:
                lo0, hi0 = self._raw_bounds(spec, c0)
                lo1, hi1 = self._raw_bounds(spec, c1)
                admit = (hi1 - lo1) // tpp
                grow = np.maximum(0, hi1 // tpp - hi0 // tpp)
                total += np.where(c0 <= 0, admit, grow)
            else:
                total += np.maximum(0, self._tokens(spec, c1) // tpp - self._tokens(spec, c0) // tpp)
        return self.layers * self.heads * total

    def dead_pages(self, c0, c1):
        """Full HBF pages that die (slide out of window) as context grows from ``c0`` to ``c1``."""
        c0 = np.asarray(c0, dtype=np.int64)
        c1 = np.asarray(c1, dtype=np.int64)
        total = np.zeros(np.broadcast_shapes(c0.shape, c1.shape), dtype=np.int64)
        for spec, tpp in zip(self.hbf, self.tpp):
            if spec.window is not None and spec.window[0] is not None and spec.window[0] < 0:
                lo0, _ = self._raw_bounds(spec, c0)
                lo1, _ = self._raw_bounds(spec, c1)
                died = np.maximum(0, lo1 // tpp - lo0 // tpp)
                total += np.where(c0 <= 0, 0, died)
        return self.layers * self.heads * total

    def per_request_row_pages(self, ctx, role_share=None):
        """Pages one layer senses per step (ceil) for each request in ``ctx``, scaled per role."""
        ctx = np.asarray(ctx, dtype=np.int64)
        total = np.zeros(len(ctx), dtype=float)
        for spec, tpp in zip(self.hbf, self.tpp):
            tokens = self._tokens(spec, ctx)
            share = 1.0 if role_share is None else role_share.get(spec.role, 1.0)
            total += (-(-tokens // tpp)) * share
        return self.heads * total

    def row_pages(self, ctx, role_share=None):
        """Pages one layer senses per step (ceil), summed over requests, scaled per role."""
        return float(self.per_request_row_pages(ctx, role_share).sum())

    def hbm_bytes(self, ctx):
        """HBM-tier KV bytes of one request with ``ctx`` tokens (e.g. KIVI's full-precision residual)."""
        ctx = np.asarray(ctx, dtype=np.int64)
        total = np.zeros_like(ctx)
        for spec in self.hbm:
            total += self._tokens(spec, ctx) * spec.token_bytes
        return self.layers * self.heads * total

    def partial_pages(self):
        """Pages of HBM staging per running request: one partial page per (layer, head, HBF tensor)."""
        return self.layers * self.heads * len(self.hbf)

    def demand(self, ctx, hbm_tokens=None):
        """Per-layer demand of one dense step over requests with ``ctx`` tokens (phase A columns)."""
        ctx = np.asarray(ctx, dtype=np.int64)
        extra = np.zeros_like(ctx) if hbm_tokens is None else np.asarray(hbm_tokens, dtype=np.int64)
        burst = hbm = elements = hbf_elements = dequant = 0.0
        for spec in self.tensors:
            tokens = self._tokens(spec, ctx)
            tok_sum = float(np.sum(tokens))
            if spec.tier == "hbf":
                tpp = self.page_bytes // spec.token_bytes
                full_burst = -(-(tpp * spec.token_bytes) // 64) * 64
                full = tokens // tpp
                rem = tokens - full * tpp
                burst += float(np.sum(full * full_burst + -(-(rem * spec.token_bytes) // 64) * 64))
            else:
                hbm += tok_sum * spec.token_bytes
            if spec.attended_dims:
                elements += tok_sum * spec.attended_dims
            if spec.role == "data" and spec.tier == "hbf":
                hbf_elements += tok_sum * spec.logical
            if spec.dequant_ops:
                dequant += tok_sum * spec.logical * spec.dequant_ops
        hbm += float(np.sum(extra)) * sum(spec.token_bytes for spec in self.hbf)
        los, his = zip(*(self.window(spec, ctx) for spec in self.data)) if self.data else ((ctx * 0,), (ctx,))
        attended = float(np.sum(np.max(his, axis=0) - np.min(los, axis=0)))
        h = self.heads
        return {"burst_bytes": h * burst, "hbm_bytes": h * hbm, "attended_tokens": h * attended,
                "attended_elements": h * elements, "hbf_elements": h * hbf_elements, "dequant_ops": h * dequant}


def geometry_for(model, kv_format=None, page_bytes=4096, *, kv_spec=None, context=32768, local_tier="hbm"):
    if kv_spec is None and not model.layer_pattern and model.attention == "gqa":
        from .formats import parse_format
        fmt = parse_format(kv_format or "bf16", model.head_dim, model.q_heads // model.kv_heads)
        return Geometry(model.layers, model.kv_heads, fmt.tensors, page_bytes), fmt
    from .kvspec import KVSpec, from_model, parse_kvspec, resolve_format
    fmt = resolve_format(model, kv_format or "bf16")
    if kv_spec is None:
        spec = from_model(model, fmt, local_tier=local_tier)
    elif isinstance(kv_spec, KVSpec):
        spec = kv_spec
    else:
        spec = parse_kvspec(kv_spec, model.layers, model.kv_heads, context)
    return spec.geometry(model.layers, model.kv_heads, fmt.tensors, page_bytes), fmt


# --------------------------------------------------------------------------------------------
# Clock

@dataclass(frozen=True)
class StepTime:
    tpot_ns: float
    gpu_ns: float
    hbm_tpot_ns: float
    h_ns: float  # HBF read time of one layer (mean across layers for heterogeneous specs)
    gaps_ns: tuple  # HBF-idle time before each layer's read (sums to tpot - layers x h)
    read_pages: float  # pages one layer senses (mean across layers for heterogeneous specs)
    rounds: int
    batch: int
    layer_rounds: tuple = ()


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
        scale = {"burst_bytes": "burst", "hbm_bytes": "hbm", "attended_tokens": "attended",
                 "attended_elements": "elements", "hbf_elements": "elements", "dequant_ops": "dequant"}
        if hasattr(geo, "layer_row_pages"):
            per_req_layers = geo.per_request_layer_row_pages(hbf_ctx) * share
            layer_pages = per_req_layers.sum(axis=1)
            if partitions is not None and partition_banks:
                parts = np.asarray(partitions, dtype=np.int64)
                layer_rounds = tuple(
                    math.ceil(self.balance * float(np.bincount(parts, weights=per_req_layers[l]).max())
                              / partition_banks - 1e-9) if layer_pages[l] > 0 else 0
                    for l in range(self.model.layers)
                )
            else:
                layer_rounds = tuple(
                    math.ceil(self.balance * float(layer_pages[l]) / self.banks - 1e-9) if layer_pages[l] > 0 else 0
                    for l in range(self.model.layers)
                )
            layer_demands = [
                {key: value * r.get(scale[key], 1.0) for key, value in ld.items()}
                for ld in geo.layer_demand(hbf_ctx, None if hbm_tokens is None else hbm_tokens)
            ]
            d_sig = tuple(
                int(round(math.log1p(val) * 200))
                for k in sorted(scale)
                for val in (sum(ld[k] for ld in layer_demands), max(ld[k] for ld in layer_demands))
            )
            key = ("L", b, layer_rounds, d_sig)
            hit = self.cache.get(key)
            if hit is not None:
                return hit
            from .system import evaluate
            h = [self.layer_ns + self.round_ns * r_l if r_l else 0.0 for r_l in layer_rounds]
            rows = [{"step": 0, "layer": l, "est_ns": h[l], "bound_ns": layer_rounds[l] * self.read_ns}
                    | layer_demands[l] for l in range(self.model.layers)]
            meta = {"batch": b * self.stacks, "stacks": self.stacks, "analysis_batch": b, "selection": self.selection,
                    "kv_format": self.kv_format}
            result = evaluate(rows, self.model, self.gpu, self.options, meta)
            row, layers = result.steps[0], result.layers
            attn = [x["attn_staged_ns" if self.options.staged else "attn_ns"] for x in layers]
            gaps = hbf_gaps(h, [x["pre_ns"] for x in layers], attn, [x["post_ns"] for x in layers],
                            self.options.overlap, self.options.staged)
            gaps[0] += max(0.0, row["tpot_hbf_est_ns"] - sum(h) - sum(gaps))
            out = StepTime(row["tpot_hbf_est_ns"], row["gpu_ns"], row["tpot_hbm_ns"],
                           sum(h) / len(h), tuple(gaps), float(layer_pages.mean()),
                           max(layer_rounds) if layer_rounds else 0, b, layer_rounds)
            self.cache[key] = out
            return out
        per_req = geo.per_request_row_pages(hbf_ctx) * share
        pages = float(per_req.sum())
        if partitions is not None and partition_banks:
            parts = np.asarray(partitions, dtype=np.int64)
            max_load = float(np.bincount(parts, weights=per_req).max())
            rounds = math.ceil(self.balance * max_load / partition_banks - 1e-9)
        elif self.rounds_fn is not None and len(set(ctx.tolist())) == 1 and hbm_tokens is None:
            rounds = self.rounds_fn(b, int(ctx[0]))
        else:
            rounds = math.ceil(self.balance * pages / self.banks - 1e-9)
        d = geo.demand(hbf_ctx, None if hbm_tokens is None else hbm_tokens)
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
    tensors = parse_format(kv_format or "bf16", model.head_dim, model.q_heads // model.kv_heads).tensors
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
            alloc="layer_major", partition="none", kv_spec=None):
    """Run one analysis-only phase A pipeline (``hbf.cli``) into ``directory``; returns its CSV rows."""
    from .cli import build_parser, configure, pipeline
    argv = HBF_FLAGS + ["--layout", "striped_contiguous", "--model", model, "--gpu", gpu,
                        "--format", kv_format or "bf16", "--mode", mode, "--context", str(context),
                        "--batch", str(batch), "--analysis-only", "--output", str(directory)]
    if kv_spec:
        argv += ["--kv-spec", kv_spec]
    for key, value in (generation or {}).items():
        argv += [f"--{key.replace('_', '-')}", str(value)]
    argv += list(device)
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


def calibration_batches(model, kv_format, context, rounds=(32, 128), limit=256, kv_spec=None):
    """Batches whose dense layer-major runs read about ``rounds`` rounds per layer (the F1 fit range)."""
    from .model import load_model
    geometry, _ = geometry_for(load_model(model), kv_format, kv_spec=kv_spec, context=context)
    if hasattr(geometry, "layer_row_pages"):
        pages = float(geometry.layer_row_pages(np.array([context])).max())
    else:
        pages = geometry.row_pages(np.array([context]))
    if pages <= 0:
        return 1, 2
    low = min(limit - 1, max(1, math.ceil(rounds[0] * 1024 / pages)))
    return low, min(limit, max(low + 1, math.ceil(rounds[1] * 1024 / pages)))


def calibrate(directory, model, gpu="b200", kv_format="bf16", *, mode="dense", generation=None, context=32768,
              batches=None, device=(), policy="packed", partition="none", snapshot_batch=None, kv_spec=None):
    """Fit the Clock to at most three phase A analysis-only runs, cached by a hash of the inputs.

    Two dense layer-major points (``batches``; default about 32 and 128 rounds per layer, see
    ``calibration_batches``) refit layer_ns and round_ns for the actual device flags (least squares
    of est_ns over rounds per layer). One snapshot of the policy (``--alloc``,
    M2) at ``snapshot_batch`` (default the larger batch) in the run's mode gives kappa = measured rounds over the Clock's
    rounds at kappa 1 (ceil(pages / 1,024), or the busiest partition's), the demand ratios of
    the mode against dense at the same batch, and the per-step hot factor (max block reads over
    ppb x read share). Pages per block do not change rounds or est, so they are fixed at 256.
    Returns the fields for ``Clock`` plus diagnostics. Safe for concurrent processes sharing
    ``directory`` (an exclusive lock per digest).
    """
    batches = tuple(batches or calibration_batches(model, kv_format, context, kv_spec=kv_spec))
    inputs = {"model": model, "gpu": gpu, "format": kv_format, "mode": mode, "generation": generation or {},
              "context": context, "batches": list(batches), "device": list(device), "policy": policy,
              "partition": partition, "snapshot_batch": snapshot_batch, "version": 4}
    if kv_spec:
        inputs["kv_spec"] = kv_spec
    digest = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()[:16]
    root = Path(directory) / f"calib-{digest}"
    cached = root / "calibration.json"
    if cached.is_file():
        return json.loads(cached.read_text())
    root.mkdir(parents=True, exist_ok=True)
    # Sweep workers share the cache: one process fits a digest while the others wait on the lock and
    # then read its result; calibration.json appears atomically (os.replace).
    with _exclusive(root / ".lock"):
        if cached.is_file():
            return json.loads(cached.read_text())
        result = _fit_calibration(root, inputs, model, gpu, kv_format, mode, generation, context, batches, device,
                                  policy, partition, snapshot_batch, kv_spec)
        temporary = root / f".calibration-{os.getpid()}.json"
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        os.replace(temporary, cached)
    return result


@contextmanager
def _exclusive(path):
    """An exclusive advisory lock on ``path`` (fcntl.flock; a no-op where fcntl is missing)."""
    try:
        import fcntl
    except ImportError:  # pragma: no cover - not POSIX
        yield
        return
    with open(path, "a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _fit_calibration(root, inputs, model, gpu, kv_format, mode, generation, context, batches, device, policy,
                     partition, snapshot_batch, kv_spec=None):
    """The phase A runs and fits of ``calibrate`` (called with the cache lock held)."""
    import shutil
    from .model import load_model

    def fresh(name):
        target = root / name
        if target.exists():
            shutil.rmtree(target)
        return target

    points, dense = [], {}
    for b in batches:
        dense[b] = phase_a(fresh(f"dense-b{b}"), model, gpu, kv_format, b, context, device=device, kv_spec=kv_spec)
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
    grouped = bool(kv_spec or load_model(model).layer_pattern)
    snap_policy = {"hbm_decode": "packed"}.get(policy, policy)
    snap_policy = snap_policy if (not grouped and snap_policy in ("packed", "dedicated", "vllm_pool")) else "layer_major"
    snap_part = partition if snap_policy in ("packed", "dedicated") else "none"
    if mode == "dense" and snap_policy == "layer_major" and b in dense:
        snapshot = dense[b]
    else:
        snapshot = phase_a(fresh(f"snapshot-{mode}-{snap_policy}-{snap_part}-b{b}"), model, gpu, kv_format, b,
                           context, mode=mode, generation=generation, device=device, alloc=snap_policy,
                           partition=snap_part, kv_spec=kv_spec)
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
    """Analytic prefill rate: FLOP/token = 2P + attn_flops(S) at the GPU's compute_eff (params.md section 6)."""
    s = max(tokens, 1)
    if not model.experts and not model.layer_pattern and model.attention == "gqa":
        params = model.layers * model.layer_params + model.embedding_params
        return gpu.flops / (2 * params + 2 * model.layers * model.q_heads * model.head_dim * s)
    params = model.active_params if model.experts else model.total_params
    attn_per_tok = 0
    for layer in range(model.layers):
        kind = model.layer_kind(layer)
        if kind == "S":
            continue
        span = min(s, model.window) if kind == "L" else s
        attn_per_tok += model.q_heads * model.default_attn_dims * span
    return gpu.flops / (2 * params + attn_per_tok)


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


def _written(geometry, c0, c1):
    c0_i, c1_i = int(round(c0)), int(round(c1))
    if hasattr(geometry, "written_pages"):
        return float(geometry.written_pages(np.array([c0_i]), np.array([c1_i]))[0])
    return max(0.0, _pages(geometry, c1_i) - _pages(geometry, c0_i))


def _dead(geometry, c0, c1):
    if not hasattr(geometry, "dead_pages"):
        return 0.0
    return float(geometry.dead_pages(np.array([int(round(c0))]), np.array([int(round(c1))]))[0])


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
    read_mult = float(getattr(geometry, "read_multiplier", 1.0))
    rho = ppb * share * hot * read_mult
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
    dead_total = 0.0
    if sessions:
        history = 0.0
        for k in range(calls):
            fresh = h0 + n_in if k == 0 else n_in
            pre = (min(_written(geometry, history, history + fresh), _pages(geometry, history + fresh))
                   if history > 0 else _written(geometry, 0, fresh))
            dec = _written(geometry, history + fresh, history + fresh + n_out)
            if history > 0:
                dead_total += _dead(geometry, history, history + fresh)
            dead_total += _dead(geometry, history + fresh, history + fresh + n_out)
            later = calls - 1 - k
            reads = (later + 1) * n_out * rho + later * ppb * read_mult
            life = (later + 1) * (call_s + w.think_s) + w.retain_s
            chunks.append((pre, "prefill", reads, life))
            if not hbm_out:
                chunks.append((dec, "decode", reads - n_out / 2 * rho, life))
            history += fresh + n_out
    else:
        hit = w.prefix_share
        pre_hit = (min(_written(geometry, prefix, prefix + n_in), _pages(geometry, prefix + n_in))
                   if hit else 0.0)
        own = hit * pre_hit + (1 - hit) * _written(geometry, 0, n_in)
        dec_hit = _written(geometry, prefix + n_in, prefix + n_in + n_out) if hit else 0.0
        dec = hit * dec_hit + (1 - hit) * _written(geometry, n_in, n_in + n_out)
        dead_hit = _dead(geometry, prefix + n_in, prefix + n_in + n_out) if hit else 0.0
        dead_total = hit * dead_hit + (1 - hit) * _dead(geometry, n_in, n_in + n_out)
        life = call_s + w.retain_s
        if policy == "layer_major":
            life = out_wave * tpot
        chunks.append((own, "prefill", n_out * rho, life))
        if not hbm_out and dec > 0:
            chunks.append((dec, "decode", n_out * rho, life))
        if hbm_out:
            dead_total = min(dead_total, own)
    host_pages = sum(chunk[0] for chunk in chunks)
    if policy == "layer_major":
        if region is None:
            if hasattr(geometry, "region_span"):
                region = geometry.region_span(b, max(1, round(w.input.mean_max(b))), ppb)
            else:
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
                per_s = k * rho / tpot + calls_per_s * w.prefix_share * p * ppb * read_mult  # decode + prefill reads
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
               "dead_pages_per_request": dead_total / calls,
               "kv_bytes_per_token": geometry.hbf_token_bytes,
               "hbm_bytes_per_request": float(geometry.hbm_bytes(np.array([int(round(ctx))]))[0]),
               "threshold_steps": threshold_steps(x, ppb, share, hot * read_mult),
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
# vLLM free-list legality tracker

class VllmTracker:
    """Write legality of a vLLM-style per-layer pool on HBF (one layer's pool; all layers are alike).

    Block id i holds pool pages [i P_id, (i + 1) P_id); a page's NAND block is (lpa mod banks,
    (lpa // banks) // ppb) and its position (lpa // banks) mod ppb. A write is legal when it is
    the block's next position (fill), or position 0 of a block without live pages (auto-erase,
    section 5.4.1 item 5). Otherwise it is an overlap (an already-written position, section
    11.5.2.2), a skip (section 11.5.2.6) or a live erase (page 0 of a block that still holds live
    data). Freed ids return to the free queue (``reuse``: fifo as vLLM, or lifo); stranded ids
    are free ids below the fill position of a block that still holds live data.
    """

    def __init__(self, ids, pages_per_id, pages_per_block, banks=1024, order="fresh", reuse="fifo"):
        from collections import deque
        if ids < 1 or pages_per_id < 1 or pages_per_block < 1 or banks < 1:
            raise ValueError("ids, pages_per_id, pages_per_block and banks must be positive")
        self.ids, self.pages_per_id, self.ppb, self.banks = ids, pages_per_id, pages_per_block, banks
        # When pages_per_id divides banks, every id occupies one whole bank-group on a single row
        # and all banks of a group receive identical writes and frees, so tracking one bank per
        # group is exact and pages_per_id times faster.
        self._grouped = (banks % pages_per_id == 0)
        self._scale = pages_per_id if self._grouped else 1
        groups = banks // pages_per_id if self._grouped else banks
        rows = -(-ids * pages_per_id // banks)
        self.nblocks = -(-rows // pages_per_block)
        self.fill = np.zeros(groups * self.nblocks, dtype=np.int64)
        self.live = np.zeros(groups * self.nblocks, dtype=np.int64)
        start = np.arange(ids)
        if order.startswith("shuffled:"):
            try:
                seed = int(order[9:])
            except ValueError:
                raise ValueError(f"Invalid free_order {order!r}: use fresh or shuffled:<int>") from None
            np.random.Generator(np.random.PCG64(seed)).shuffle(start)
        elif order != "fresh":
            raise ValueError("free_order must be fresh or shuffled:<seed>")
        if reuse not in ("fifo", "lifo"):
            raise ValueError("vllm_reuse must be fifo or lifo")
        self.queue, self.reuse = deque(start.tolist()), reuse
        self.used = np.zeros(ids, dtype=bool)
        self.counts = {"skip": 0, "overlap": 0, "live_erase": 0, "erases": 0, "writes": 0, "legalize_pages": 0}

    def _pages(self, ids):
        arr = np.asarray(ids, dtype=np.int64)
        if self._grouped:
            lpa = arr * self.pages_per_id
            row = lpa // self.banks
            return ((lpa % self.banks) // self.pages_per_id) * self.nblocks + row // self.ppb, row % self.ppb
        lpa = (arr[:, None] * self.pages_per_id + np.arange(self.pages_per_id)).ravel()
        row = lpa // self.banks
        return (lpa % self.banks) * self.nblocks + row // self.ppb, row % self.ppb

    def allocate(self, n):
        if n <= 0:
            return []
        if n > len(self.queue):
            raise CapacityError(f"vLLM pool exhausted: need {n} ids, {len(self.queue)} free")
        ids = [self.queue.popleft() for _ in range(n)]
        self.used[ids] = True
        self.write(ids)
        return ids

    def write(self, ids):
        if not len(ids):
            return
        key, pos = self._pages(ids)
        owner = (np.arange(len(ids), dtype=np.int64) if self._grouped
                 else np.repeat(np.arange(len(ids)), self.pages_per_id))
        order = np.argsort(key, kind="stable")
        k, p, o = key[order], pos[order], owner[order]
        first = np.ones(len(k), dtype=bool)
        first[1:] = k[1:] != k[:-1]
        live_before = self.live[k]
        erase = (p == 0) & first & (live_before == 0)
        live_erase = (p == 0) & ~erase
        # Within each block, split erase epochs only on legal erases (live_before == 0), not on
        # live_erase, so fill never shrinks below still-live pages.
        epoch_start = first | erase
        epoch = np.cumsum(epoch_start) - 1
        scale = self.ppb + 2
        val = np.where(first & ~erase, np.maximum(self.fill[k] - 1, p), p)
        keyed = epoch * scale + (val + 1)
        after = np.maximum.accumulate(keyed)
        before = np.empty_like(keyed)
        before[:1] = 0
        before[1:] = after[:-1]
        prev_high = np.where(erase, -1, np.where(first, self.fill[k] - 1, before - epoch * scale - 1))
        legal = erase | ((p != 0) & (p == prev_high + 1))
        overlap = ~legal & ~live_erase & (p <= prev_high)
        skip = ~legal & ~live_erase & (p > prev_high + 1)
        for name, mask in (("overlap", overlap), ("skip", skip), ("live_erase", live_erase)):
            if mask.any():
                self.counts[name] += len(np.unique(o[mask]))
        bad_mask = overlap | live_erase
        if bad_mask.any():  # legalising = replaying the offending blocks up to their fill at violation time
            bad_k = k[bad_mask]
            eff_fill = np.zeros(len(self.fill), dtype=np.int64)
            np.maximum.at(eff_fill, bad_k, np.maximum(self.fill[bad_k], prev_high[bad_mask] + 1))
            self.counts["legalize_pages"] += int(eff_fill[np.unique(bad_k)].sum()) * self._scale
        self.counts["erases"] += int(erase.sum()) * self._scale
        self.counts["writes"] += len(ids)
        ends = np.flatnonzero(np.r_[first[1:], True])
        self.fill[k[ends]] = after[ends] - epoch[ends] * scale
        np.add.at(self.live, k, 1)

    def free(self, ids):
        if not len(ids):
            return
        key, _ = self._pages(ids)
        np.subtract.at(self.live, key, 1)
        self.used[list(ids)] = False
        if self.reuse == "fifo":
            self.queue.extend(ids)
        else:
            self.queue.extendleft(ids)

    def stranded(self):
        """Fraction of all ids that are free but cannot be rewritten legally now."""
        free = np.flatnonzero(~self.used)
        if not len(free):
            return 0.0
        key, pos = self._pages(free)
        stuck = (self.live[key] > 0) & (pos < self.fill[key])
        if self._grouped:
            return float(stuck.sum()) / self.ids
        per_id = stuck.reshape(len(free), self.pages_per_id).any(axis=1)
        return float(per_id.sum()) / self.ids


# --------------------------------------------------------------------------------------------
# Event simulation (M4b)

SIM_POLICIES = ("packed", "dedicated", "hbm_decode", "layer_major", "vllm_pool")
WEAR = ("none", "dynamic", "ideal", "remap")
REFRESH = ("auto", "relocate", "replay")
VICTIMS = ("greedy", "cost_benefit")
CAUSE = {cause: index for index, cause in enumerate(CAUSES)}
EPS = 1e-6
OPEN, FULL = 1, 2  # segment states (0: never written)
ACTIVE, IDLE = 1, 2  # entry states (0: unused)
PRIVATE, PREFIX, SESSION = 0, 1, 2  # entry kinds


class CapacityError(RuntimeError):
    """No free segment for a write or an admission (raise --capacity-gib or lower the load)."""


@dataclass(frozen=True)
class SimOptions:
    """Settings of one event simulation (CLI names in ``build_parser``)."""
    policy: str = "packed"
    x: float = 1e6  # read-disturb threshold in block reads
    hot: float = 1.0  # persistent read skew (max over mean block reads)
    share: float = 1.0  # pages sensed per stored page per decode step
    retention_s: float = 86400.0  # 0: retention refresh off
    prefix_reads: str = "per_request"
    hours: float = 24.0
    warmup_h: float = 1.0
    quantum_s: float = 1.0
    report_s: float = 60.0
    rd_slack: float = 0.01  # shortest quantum as a share of the fastest segment's threshold in steps
    wear: str = "dynamic"
    refresh: str = "auto"  # auto: replay for layer_major, relocate otherwise
    refresh_idle: str = "relocate"  # or evict: idle entries in a refreshed segment are dropped
    reclaim: str = "evict"  # or compact: idle entries in a reclaim victim are copied too
    victim: str = "greedy"  # greedy (fewest pages to copy) or cost_benefit ((1 - u) * age / (1 + u))
    prefill: str = "disaggregated"  # disaggregated or colocated (adds input / prefill_tok_s per admission)
    pre_open: bool = False  # pre-open free segments in idle time (takes non-replay erases off M5)
    separate_prefix: bool = False  # keep shared prefixes on their own write frontier per partition
    low_watermark: int | None = None  # free segments per partition kept by reclaim (default 2 + one request)
    align: str = "row"  # packed: a new entry starts on a row
    slots: int | None = None  # running requests at most (layer_major: slots per wave)
    stagger: bool = True
    stacks: int = 1  # for the summary only (the workload is already one stack's share)
    m5: str = "gaps"
    t_prog: float = 100000.0
    t_bers: float = 1000000.0
    suspend_ns: float | None = None
    block_unit: str = "die"
    pe_limits: tuple = (1e4, 1e5, 1e6)
    block_tokens: int | None = None  # vllm_pool: tokens per vLLM block (default: lcm of HBF tpp)
    free_order: str = "fresh"  # vllm_pool: fresh (ascending ids) or shuffled:<seed>
    vllm_reuse: str = "fifo"  # vllm_pool: fifo (vLLM default) or lifo

    def __post_init__(self):
        if self.policy not in SIM_POLICIES:
            raise ValueError(f"The simulation supports the policies {', '.join(SIM_POLICIES)}")
        if self.wear not in WEAR or self.refresh not in REFRESH:
            raise ValueError(f"wear must be one of {', '.join(WEAR)} and refresh one of {', '.join(REFRESH)}")
        if self.refresh_idle not in ("relocate", "evict") or self.reclaim not in ("evict", "compact"):
            raise ValueError("refresh_idle is relocate or evict; reclaim is evict or compact")
        if self.victim not in VICTIMS:
            raise ValueError(f"victim must be one of {', '.join(VICTIMS)}")
        if self.prefill not in ("disaggregated", "colocated"):
            raise ValueError("prefill must be disaggregated or colocated")
        if self.prefix_reads not in ("per_request", "once") or self.align not in ("row", "none"):
            raise ValueError("prefix_reads is per_request or once; align is row or none")
        if self.m5 not in M5_MODES + ("off",):
            raise ValueError(f"m5 must be one of {', '.join(M5_MODES + ('off',))}")
        if self.m5 == "suspend" and self.suspend_ns is None:
            raise ValueError("m5 suspend needs suspend_ns")
        if min(self.x, self.hot, self.hours, self.quantum_s, self.report_s) <= 0 or not 0 < self.share <= 1:
            raise ValueError("x, hot, hours, quantum and report interval must be positive and share in (0, 1]")
        if self.warmup_h < 0 or self.warmup_h >= self.hours:
            raise ValueError("The warm-up must be non-negative and shorter than the simulated time")
        if self.retention_s < 0 or not 0 <= self.rd_slack < 1:
            raise ValueError("retention_s must be >= 0 and rd_slack in [0, 1)")
        if self.low_watermark is not None and self.low_watermark < 1:
            raise ValueError("low_watermark must be at least 1 segment")
        if self.slots is not None and self.slots < 1:
            raise ValueError("slots must be positive")
        if self.policy == "layer_major" and self.refresh == "relocate":
            raise ValueError("layer_major regions are refreshed in place (replay, spec section 5.7)")
        if self.block_tokens is not None and self.block_tokens < 1:
            raise ValueError("block_tokens must be positive")
        if self.free_order != "fresh" and not self.free_order.startswith("shuffled:"):
            raise ValueError("free_order must be fresh or shuffled:<seed>")
        if self.free_order.startswith("shuffled:"):
            try:
                int(self.free_order[9:])
            except ValueError:
                raise ValueError(f"Invalid free_order {self.free_order!r}: use fresh or shuffled:<int>") from None
        if self.vllm_reuse not in ("fifo", "lifo"):
            raise ValueError("vllm_reuse must be fifo or lifo")


class Simulator:
    """Segment-level event simulation of one HBF stack (design-B section 4); ``simulate`` runs it.

    A segment is one block index across the banks of a partition (a superblock of 1,024 banks, a
    channel block of 64 or a die block of 16). Placement is row-aligned, so an entry with n pages
    in a segment puts n / banks pages in each of its blocks and segment-level counts are exact.

    State (struct of arrays): segments (fill, live, pec, block reads since erase x hot, state,
    owner: a dedicated entry or -2 for a wave region, erases per cause); entries (a request's own
    KV, a shared prefix or a session history: pages, readers, active or idle, idle deadline);
    occupancy rows (entry, segment, pages, oldest write time) in COO arrays; running requests in
    slot arrays (context, end context, staged decode pages, entry, prefix entry, partition).

    One quantum: step time T from the Clock plus the previous quantum's M5 extra; n = quantum / T
    steps, cut at the earliest predicted read-threshold or retention crossing (not below rd_slack
    of the fastest segment's threshold in steps); requests complete and arrive at their exact step
    inside it. Then block reads are applied, staged decode pages are written in whole rows, and
    idle expiry, read-disturb refresh, retention refresh and reclaim run (design-B section 4.2).
    """

    def __init__(self, geometry, clock, workload, device=None, options=None, model=None, gpu=None):
        self.geo, self.clock, self.workload = geometry, clock, workload
        self.dev = d = device or Device()
        self.opt = o = options or SimOptions()
        self.model = model if model is not None else getattr(clock, "model", None)
        self.gpu = gpu if gpu is not None else getattr(clock, "gpu", None)
        if o.policy in ("layer_major", "vllm_pool") and d.partition != "none":
            raise ValueError(f"{o.policy} uses the whole device (no partition)")
        if o.prefill == "colocated" and (self.model is None or self.gpu is None):
            raise ValueError("prefill='colocated' needs model and gpu (from Clock or arguments)")
        if o.refresh == "auto":
            self.refresh = "replay" if o.policy == "layer_major" else "relocate"
        else:
            self.refresh = o.refresh
        self.writes_decode = o.policy in ("packed", "dedicated", "vllm_pool")
        self.hbm_out = o.policy in ("hbm_decode", "layer_major")
        self.parts, self.row, self.P = d.parts, d.segment_banks, d.segment_pages
        self.S = S = d.capacity_pages // self.P
        if S < 2 * self.parts:
            raise ValueError(f"The device holds {S} segments of {self.P} pages: need at least 2 per partition")
        self.read_mult = float(getattr(self.geo, "read_multiplier", 1.0))
        self.read_scale = o.share * o.hot * self.read_mult / self.row
        # segments
        self.fill, self.live, self.reads = np.zeros(S), np.zeros(S), np.zeros(S)
        self.pec = np.zeros(S, dtype=np.int64)
        self.remap_epoch = np.zeros(S, dtype=np.int64)
        self.state = np.zeros(S, dtype=np.int8)
        self.owner = np.full(S, -1, dtype=np.int64)
        self.busy = np.zeros(S, dtype=bool)
        self.part_of = np.arange(S) % self.parts
        self.erases = np.zeros((S, len(CAUSES)), dtype=np.int64)
        self.is_free = np.ones(S, dtype=bool)
        self.heaps = [[self._key(s) for s in range(p, S, self.parts)] for p in range(self.parts)]
        self.free_count = np.array([len(heap) for heap in self.heaps], dtype=np.int64)
        self.part_live = np.zeros(self.parts)
        self.part_pec = np.zeros(self.parts)  # sum of segment PECs per partition (tie-break for new entries)
        self.part_ver = np.zeros(self.parts, dtype=np.int64)
        self.reclaim_blocked = np.full(self.parts, -1, dtype=np.int64)
        self.frontier = [-1] * self.parts
        self.prefix_frontier = [-1] * self.parts
        # entries
        self.e_cap = 0
        self.free_entries = []
        self._grow_entries(256)
        self.e_rows, self.e_owned, self.e_key, self.e_ids = {}, {}, {}, {}
        self.prefix_entry, self.session_entry = {}, {}
        self.protected = set()
        # occupancy rows
        self.c_e = np.zeros(4096, dtype=np.int64)
        self.c_s = np.zeros(4096, dtype=np.int64)
        self.c_n = np.zeros(4096)
        self.c_t = np.zeros(4096)
        self.c_v = np.zeros(4096, dtype=bool)
        self.c_count = self.dead_rows = 0
        # running requests
        w = workload
        self.cap = cap = o.slots or (w.concurrency if w.arrival == "closed" else w.max_batch)
        self.sl_on = np.zeros(cap, dtype=bool)
        self.sl_req = [None] * cap
        self.sl_entry = np.full(cap, -1, dtype=np.int64)
        self.sl_prefix = np.full(cap, -1, dtype=np.int64)
        self.sl_ctx = np.zeros(cap, dtype=np.int64)
        self.sl_end = np.zeros(cap, dtype=np.int64)
        self.sl_gen = np.zeros(cap, dtype=np.int64)
        self.sl_part = np.zeros(cap, dtype=np.int64)
        self.sl_staged = np.zeros(cap)
        self.sl_dead = np.zeros(cap)
        self.free_slots = list(range(cap - 1, -1, -1))
        self.n_running = 0
        self.queue = []
        self.stream = Stream(workload, stagger=o.stagger)
        self.region, self.region_cache = [], {}
        # vLLM free-list tracker (runs alongside packed wear when policy == "vllm_pool")
        self.vllm = None
        self.block_tokens = 0
        self._vllm_residual = 0
        self.vllm_stranded = 0.0
        self.vllm_dirty = False
        if o.policy == "vllm_pool":
            if hasattr(self.geo, "is_uniform") and not self.geo.is_uniform:
                raise ValueError("vllm_pool requires uniform layers (not heterogeneous --kv-spec)")
            bt = o.block_tokens or math.lcm(*self.geo.tpp)
            if any(bt % tpp for tpp in self.geo.tpp):
                raise ValueError("vllm_pool block_tokens must be a multiple of every HBF tensor's tokens per page")
            self.block_tokens = bt
            windows = {spec.window for spec in self.geo.hbf}
            if len(windows) == 1:
                win = next(iter(windows))
                if win is not None and win[0] in (None, 0) and win[1] is not None and win[1] < 0:
                    self._vllm_residual = -win[1]
            per_id = self.geo.heads * sum(bt // tpp for tpp in self.geo.tpp)
            pool_pages = self.S * self.P // self.geo.layers
            n_ids = pool_pages // per_id
            if n_ids < 1:
                raise ValueError("vllm_pool capacity too small for even one block id")
            self.vllm = VllmTracker(n_ids, per_id, d.pages_per_block, d.banks, o.free_order, o.vllm_reuse)
        # clocks and counters
        self.t = 0.0
        self.page_cache = {}
        self.programs = np.zeros(len(CAUSES))
        self.seg_erases = np.zeros(len(CAUSES), dtype=np.int64)
        self.block_erases = 0
        self.m5_erases = 0
        self.acc = dict.fromkeys(("steps", "tokens", "batch_s", "ctx_s", "decode_s", "tpot_s", "m5_s",
                                  "stranded_s", "prefill_s"), 0.0)
        self.n = dict.fromkeys(("completed", "admitted", "refresh_rd", "refresh_ret", "compaction", "evicted_pages",
                                "recompute_tokens", "stalls", "reclaims", "waves", "quanta", "dead_pages"), 0.0)
        self.prefill_dt = 0.0
        self.guard = False
        self.pending_reads = False
        self.drops = 0  # bumps whenever live data dies or moves (retry point of a stalled admission)
        self.blocked = None
        self.m5_extra = self.max_step_extra = self.staging_peak = 0.0
        self.m5_mark = (0.0, 0)
        self.free_min = int(self.free_count.sum())
        self.warm = None
        self.timeseries = []
        self.low_wm = o.low_watermark or min(2 + math.ceil(self._max_request_pages() / self.P - 1e-9),
                                             max(1, S // self.parts // 2))
        self.wall_s = 0.0

    # ---- helpers ---------------------------------------------------------------------------
    def _key(self, s):
        if self.opt.wear == "none":
            return (0, s)
        if self.opt.wear == "remap":
            return (int(self.remap_epoch[s]), s)
        return (int(self.pec[s]), s)

    def pages(self, tokens):
        """Full HBF pages of ``tokens`` tokens of one request (cached)."""
        tokens = int(tokens)
        hit = self.page_cache.get(tokens)
        if hit is None:
            hit = self.page_cache[tokens] = float(self.geo.stored_pages(np.array([tokens]))[0])
        return hit

    def _written(self, c0, c1):
        c0_i, c1_i = int(c0), int(c1)
        if hasattr(self.geo, "written_pages"):
            return float(self.geo.written_pages(np.array([c0_i]), np.array([c1_i]))[0])
        return max(0.0, self.pages(c1_i) - self.pages(c0_i))

    def _dead(self, c0, c1):
        if not hasattr(self.geo, "dead_pages"):
            return 0.0
        return float(self.geo.dead_pages(np.array([int(c0)]), np.array([int(c1)]))[0])

    def _retire_dead(self, e, n):
        """Expire ``n`` oldest live pages of entry ``e`` (sliding window deaths). Returns unretired remainder."""
        if n <= EPS or not self.e_rows.get(e):
            return max(0.0, float(n))
        self._entry_reads(e)
        rows = sorted(self.e_rows[e].values(), key=lambda i: (self.c_t[i], i))
        rem = float(n)
        retired = 0.0
        for i in rows:
            if rem <= EPS:
                break
            avail = float(self.c_n[i])
            if avail <= rem + EPS:
                _, dropped = self._drop(i)
                rem -= dropped
                retired += dropped
            else:
                s = int(self.c_s[i])
                p = self.part_of[s]
                self.c_n[i] -= rem
                self.e_pages[e] -= rem
                self.live[s] -= rem
                self.part_live[p] -= rem
                self.drops += 1
                self.part_ver[p] += 1
                if self.live[s] < EPS:
                    self.live[s] = 0.0
                    self._release(s)
                retired += rem
                rem = 0.0
        self.n["dead_pages"] += retired
        return max(0.0, rem)

    def _max_request_pages(self):
        w = self.workload

        def top(length):
            if length is None:
                return 0.0
            return float(length.get("n")) if length.kind == "fixed" else float(length._grid()[-1])
        if w.arrival == "script":
            tokens = max(int(item["input"]) + int(item["output"]) for item in w.script)
        elif w.sessions:
            tokens = top(w.history) + w.calls * (top(w.input) + top(w.output))
        else:
            tokens = top(w.input) + top(w.output)
        return self.pages(int(tokens + w.prefix_tokens))

    def _grow_entries(self, new):
        fields = (("e_pages", float, 0), ("e_readers", float, 0), ("e_state", np.int8, 0),
                  ("e_ttl", float, math.inf), ("e_kind", np.int8, 0), ("e_part", np.int64, 0),
                  ("e_dseg", np.int64, -1), ("e_tokens", float, 0), ("e_steps", float, 0),
                  ("e_stored", np.int64, 0))  # e_stored: context position up to which the entry is on HBF
        old = self.e_cap
        for name, kind, fill in fields:
            grown = np.full(new, fill, dtype=kind)
            if old:
                grown[:old] = getattr(self, name)
            setattr(self, name, grown)
        self.free_entries.extend(range(new - 1, old - 1, -1))
        self.e_cap = new

    def _new_entry(self, kind, part, tokens):
        if not self.free_entries:
            self._grow_entries(2 * self.e_cap)
        e = self.free_entries.pop()
        self.e_pages[e] = self.e_readers[e] = self.e_steps[e] = 0.0
        self.e_state[e], self.e_ttl[e], self.e_kind[e] = ACTIVE, math.inf, kind
        self.e_part[e], self.e_dseg[e], self.e_tokens[e], self.e_stored[e] = part, -1, tokens, 0
        self.e_rows[e], self.e_owned[e] = {}, []
        return e

    def _program(self, cause, n):
        self.programs[CAUSE[cause]] += n

    # ---- occupancy rows --------------------------------------------------------------------
    def _row(self, e, s, n):
        rows = self.e_rows[e]
        i = rows.get(s)
        if i is not None:  # keeps the older write time (conservative for retention)
            self.c_n[i] += n
            return
        if self.c_count == len(self.c_e):
            if self.dead_rows > self.c_count // 2:
                self._compact_rows()
            else:
                for name in ("c_e", "c_s", "c_n", "c_t", "c_v"):
                    array = getattr(self, name)
                    grown = np.zeros(2 * len(array), dtype=array.dtype)
                    grown[:len(array)] = array
                    setattr(self, name, grown)
        i = self.c_count
        self.c_e[i], self.c_s[i], self.c_n[i], self.c_t[i], self.c_v[i] = e, s, n, self.t, True
        self.c_count += 1
        rows[s] = i

    def _compact_rows(self):
        self._flush_reads()
        keep = np.flatnonzero(self.c_v[:self.c_count])
        for name in ("c_e", "c_s", "c_n", "c_t", "c_v"):
            array = getattr(self, name)
            array[:len(keep)] = array[keep]
        self.c_v[len(keep):] = False
        self.c_count, self.dead_rows = len(keep), 0
        for rows in self.e_rows.values():
            rows.clear()
        for i, (e, s) in enumerate(zip(self.c_e[:len(keep)].tolist(), self.c_s[:len(keep)].tolist())):
            self.e_rows[e][s] = i

    def _drop(self, i):
        """Invalidate occupancy row i; returns (entry, pages)."""
        e, s, n = int(self.c_e[i]), int(self.c_s[i]), float(self.c_n[i])
        self.c_v[i] = False
        self.dead_rows += 1
        self.drops += 1
        p = self.part_of[s]
        self.part_ver[p] += 1
        del self.e_rows[e][s]
        self.e_pages[e] -= n
        self.live[s] -= n
        self.part_live[p] -= n
        if self.live[s] < EPS:
            self.live[s] = 0.0
            self._release(s)
        return e, n

    # ---- segments --------------------------------------------------------------------------
    def _release(self, s):
        """Mark segment s free when it is closed, unowned and holds no live data."""
        if (self.is_free[s] or self.busy[s] or self.state[s] == OPEN or self.owner[s] != -1
                or self.live[s] > EPS):
            return
        self.is_free[s] = True
        p = self.part_of[s]
        heapq.heappush(self.heaps[p], self._key(s))
        self.free_count[p] += 1
        self.drops += 1
        self.part_ver[p] += 1

    def _take(self, part):
        heap = self.heaps[part]
        while heap:
            s = heapq.heappop(heap)[1]
            if self.is_free[s]:
                self.is_free[s] = False
                self.remap_epoch[s] += 1
                self.free_count[part] -= 1
                self.free_min = min(self.free_min, int(self.free_count.sum()))
                return s
        return -1

    def _erase(self, s, cause):
        """Row-0 write of segment s: every block of it is auto-erased (spec section 5.4.1 item 5)."""
        c = CAUSE[cause]
        self.pec[s] += 1
        self.part_pec[self.part_of[s]] += 1
        self.erases[s, c] += 1
        self.seg_erases[c] += 1
        self.block_erases += self.row
        if not self.opt.pre_open or cause == "replay":
            self.m5_erases += self.row
        self.reads[s] = 0.0

    def _open(self, part, cause, owner=-1, prefix=False):
        """Take a free segment of ``part`` and write its row 0 (an erase), reclaiming first when none is free."""
        front = self.prefix_frontier if (prefix and self.opt.separate_prefix) else self.frontier
        s = self._take(part)
        if s < 0 and not self.guard:
            self._reclaim(part, force=True, target=lambda: self.free_count[part] >= 1 or (owner < 0 and front[part] >= 0))
            if owner < 0 and front[part] >= 0:  # compaction opened a frontier: continue there
                return front[part]
            s = self._take(part)
        if s < 0:
            raise CapacityError(f"No free segment in partition {part} at t = {self.t:.1f} s "
                                f"({self.S} segments of {self.P} pages); raise --capacity-gib or --low-watermark")
        self._erase(s, cause)
        self.state[s], self.owner[s], self.fill[s] = OPEN, owner, 0.0
        if owner >= 0:
            self.e_owned[owner].append(s)
        else:
            front[part] = s
        return s

    def _close(self, s):
        self.state[s] = FULL
        p = self.part_of[s]
        if self.frontier[p] == s:
            self.frontier[p] = -1
        if self.prefix_frontier[p] == s:
            self.prefix_frontier[p] = -1
        o = self.owner[s]
        if o >= 0 and self.e_dseg[o] == s:
            self.e_dseg[o] = -1
        self.part_ver[p] += 1
        self._release(s)

    # ---- writes ----------------------------------------------------------------------------
    def _write(self, e, n, cause, align=False):
        """Append n pages of entry e: the partition's frontier, or the entry's own segment (dedicated)."""
        part = int(self.e_part[e])
        dedicated = self.opt.policy == "dedicated"
        use_prefix = self.opt.separate_prefix and self.e_kind[e] == PREFIX
        front = self.prefix_frontier if use_prefix else self.frontier
        while n > EPS:
            if dedicated:
                s = int(self.e_dseg[e])
                if s < 0:
                    s = self._open(part, cause, owner=e)
                    self.e_dseg[e] = s
            else:
                s = front[part]
                if s >= 0 and align and self.opt.align == "row":
                    pad = math.ceil(self.fill[s] / self.row - 1e-9) * self.row - self.fill[s]
                    if pad > EPS:
                        self.fill[s] += pad
                        self._program("padding", pad)
                        if self.fill[s] >= self.P - EPS:
                            self._close(s)
                            s = -1
                if s < 0:
                    s = self._open(part, cause, prefix=use_prefix)
                align = False
            k = min(n, self.P - self.fill[s])
            self.fill[s] += k
            self.live[s] += k
            self.part_live[self.part_of[s]] += k
            self.e_pages[e] += k
            self._row(e, s, k)
            self._program(cause, k)
            n -= k
            if self.fill[s] >= self.P - EPS:
                self._close(s)

    def _read_once(self, e):
        """Prefill attention over a cached entry: every page is read once (no share, no skew)."""
        for s, i in self.e_rows[e].items():
            self.reads[s] += self.c_n[i] * self.read_mult / self.row

    # ---- block reads -----------------------------------------------------------------------
    def _flush_reads(self):
        """Apply the read steps accrued since the last flush to the segments holding each entry."""
        if not self.pending_reads:
            return
        n = self.c_count
        weight = self.c_n[:n] * self.e_steps[self.c_e[:n]]
        weight[~self.c_v[:n]] = 0.0
        self.reads += np.bincount(self.c_s[:n], weight, self.S) * self.read_scale
        self.e_steps[:] = 0.0
        self.pending_reads = False

    def _entry_reads(self, e):
        k = self.e_steps[e]
        if k:
            for s, i in self.e_rows[e].items():
                self.reads[s] += self.c_n[i] * k * self.read_scale
            self.e_steps[e] = 0.0

    def _rates(self):
        """Block reads per decode step per segment for the current running set."""
        on = np.flatnonzero(self.sl_on)
        rate = np.zeros(self.e_cap)
        np.add.at(rate, self.sl_entry[on], 1.0)
        pre = self.sl_prefix[on]
        pre = pre[pre >= 0]
        if len(pre):
            if self.opt.prefix_reads == "per_request":
                np.add.at(rate, pre, 1.0)
            else:
                rate[np.unique(pre)] = 1.0
        n = self.c_count
        weight = self.c_n[:n] * rate[self.c_e[:n]]
        weight[~self.c_v[:n]] = 0.0
        return np.bincount(self.c_s[:n], weight, self.S) * self.read_scale

    # ---- admission and completion ------------------------------------------------------------
    def _partition(self):
        """Partition of a new entry: least live pages; ties to the least worn (wear levelling) or lowest index."""
        if self.parts == 1:
            return 0
        low = np.flatnonzero(self.part_live <= self.part_live.min() + EPS)
        if self.opt.wear == "none":
            return int(low[0])
        return int(low[np.argmin(self.part_pec[low])])

    def _outstanding(self, part):
        """Pages that running requests of ``part`` will still write (remaining decode plus staged)."""
        if not self.writes_decode:
            return 0.0
        on = np.flatnonzero(self.sl_on & (self.sl_part == part))
        if not len(on):
            return 0.0
        geo = self.geo
        if hasattr(geo, "written_pages"):
            rem = geo.written_pages(self.sl_ctx[on], self.sl_end[on]).sum()
        else:
            rem = (geo.stored_pages(self.sl_end[on]) - geo.stored_pages(self.sl_ctx[on])).sum()
        return float(rem + self.sl_staged[on].sum())

    def _frontier_rem(self, part, prefix=False):
        if self.opt.policy == "dedicated":
            return 0.0
        front = self.prefix_frontier if (prefix and self.opt.separate_prefix) else self.frontier
        s = front[part]
        return (self.P - self.fill[s]) if s >= 0 else 0.0

    def _room(self, part, prefix=False):
        return float(self.free_count[part] * self.P) + self._frontier_rem(part, prefix)

    def _segs_needed(self, part, main_pages, prefix_pages=0.0):
        """Free segments of ``part`` needed to write ``main_pages`` and ``prefix_pages``."""
        if self.opt.policy == "dedicated":
            s_main = math.ceil(main_pages / self.P - 1e-9) if main_pages > EPS else 0
            s_pre = math.ceil(prefix_pages / self.P - 1e-9) if prefix_pages > EPS else 0
            return s_main + s_pre
        if self.opt.separate_prefix:
            rem_main = self._frontier_rem(part, prefix=False)
            rem_pre = self._frontier_rem(part, prefix=True)
            s_main = max(0, math.ceil((main_pages - rem_main) / self.P - 1e-9)) if main_pages > EPS else 0
            s_pre = max(0, math.ceil((prefix_pages - rem_pre) / self.P - 1e-9)) if prefix_pages > EPS else 0
            return s_main + s_pre
        total = main_pages + prefix_pages
        rem = self._frontier_rem(part, prefix=False)
        return max(0, math.ceil((total - rem) / self.P - 1e-9)) if total > EPS else 0

    def _space(self, part, need, prefix_need=0.0):
        """Room for ``need`` (and ``prefix_need``) pages plus outstanding writes and reserve, reclaiming when short."""
        want = need + self._outstanding(part)
        reserve = self.low_wm - 1
        if self.free_count[part] >= self._segs_needed(part, want, prefix_need) + reserve:
            return True
        self._reclaim(part, force=True,
                      target=lambda: self.free_count[part] >= self._segs_needed(part, want, prefix_need) + reserve)
        return self.free_count[part] >= self._segs_needed(part, want, prefix_need)

    def _vllm_tokens(self, tokens):
        return max(0, int(tokens) - self._vllm_residual)

    def _vllm_outstanding(self):
        """vLLM block ids that currently running requests will still allocate during decode."""
        if self.vllm is None or not self.writes_decode:
            return 0
        on = np.flatnonzero(self.sl_on)
        if not len(on):
            return 0
        bt, res = self.block_tokens, self._vllm_residual
        end_ids = np.maximum(0, self.sl_end[on] - res) // bt
        ctx_ids = np.maximum(0, self.sl_ctx[on] - res) // bt
        return int((end_ids - ctx_ids).sum())

    def _vllm_ensure(self, need_ids):
        """Evict oldest idle entries when the vLLM free queue lacks ``need_ids`` ids."""
        if self.vllm is None or need_ids <= len(self.vllm.queue):
            return True
        idle = [e for e in np.flatnonzero(self.e_state == IDLE).tolist()
                if e not in self.protected and e in self.e_ids]
        idle.sort(key=lambda e: min((self.c_t[i] for i in self.e_rows[e].values()), default=math.inf))
        for e in idle:
            self._kill(e, evicted=True)
            if need_ids <= len(self.vllm.queue):
                return True
        return need_ids <= len(self.vllm.queue)

    def _vllm_write(self, e, lo, hi):
        """vllm_pool: allocate and write the ids of full vLLM blocks in HBF tokens [lo, hi) of entry e."""
        if self.vllm is None:
            return
        bt = self.block_tokens
        count = self._vllm_tokens(hi) // bt - self._vllm_tokens(lo) // bt
        if count > 0:
            self._vllm_ensure(count)
            self.e_ids.setdefault(e, []).extend(self.vllm.allocate(count))
            self.vllm_dirty = True

    def _update_stranded(self, dt):
        """Integrate time-averaged stranded fraction over ``dt`` seconds (vllm_pool only)."""
        if self.vllm is not None and dt > 0:
            if self.vllm_dirty:
                self.vllm_stranded = self.vllm.stranded()
                self.vllm_dirty = False
            self.acc["stranded_s"] += self.vllm_stranded * dt

    def _admit(self, req):
        """Start a request: prefix and session entries, prefill writes. False when space is short."""
        w = self.workload
        pre_tokens = w.prefix_tokens if req.prefix >= 0 else 0
        start = pre_tokens + req.history + req.input
        end = start + req.output
        sess = self.session_entry.get(req.session, -1) if req.session >= 0 else -1
        pre = self.prefix_entry.get(req.prefix, -1) if req.prefix >= 0 else -1
        new_prefix = req.prefix >= 0 and pre < 0
        # A session's history is on HBF up to e_stored (under hbm_decode the last call's output is not:
        # it is written with this call's prompt); an evicted history is recomputed and rewritten.
        have = int(self.e_stored[sess]) if sess >= 0 else pre_tokens
        own = min(self._written(have, start), self.pages(start)) if have > 0 else self._written(0, start)
        pre_need = self.pages(pre_tokens) if new_prefix else 0.0
        need = own + (self._written(start, end) if self.writes_decode else 0.0)
        if not self.opt.separate_prefix:
            need += pre_need
            pre_need = 0.0
        vllm_need = 0
        if self.vllm is not None:
            bt = self.block_tokens
            if new_prefix:
                vllm_need += self._vllm_tokens(pre_tokens) // bt
            v_hi = self._vllm_tokens(end if self.writes_decode else start)
            vllm_need += max(0, v_hi // bt - self._vllm_tokens(have) // bt)
            vllm_need += self._vllm_outstanding()
        part = int(self.e_part[sess]) if sess >= 0 else self._partition()
        keep = [e for e in (sess, pre) if e >= 0]
        self.protected.update(keep)  # a forced reclaim must not evict what this request reuses
        try:
            if not self._space(part, need, pre_need) or not self._vllm_ensure(vllm_need):
                return False
        finally:
            self.protected.difference_update(keep)
        if req.prefix >= 0:
            if pre >= 0:
                self._read_once(pre)
            else:
                pre = self._new_entry(PREFIX, part, pre_tokens)
                self.prefix_entry[req.prefix], self.e_key[pre] = pre, ("prefix", req.prefix)
                self._write(pre, self.pages(pre_tokens), "prefill", align=True)
                self._vllm_write(pre, 0, pre_tokens)
            self.e_state[pre], self.e_ttl[pre] = ACTIVE, math.inf
            self.e_readers[pre] += 1
        if sess >= 0:
            e = sess
            self._read_once(e)
            if have > 0:
                dead_sess = self._dead(have, start)
                if dead_sess > EPS:
                    self._retire_dead(e, dead_sess)
        else:
            e = self._new_entry(SESSION if req.session >= 0 else PRIVATE, part, 0)
            if req.session >= 0:
                self.session_entry[req.session], self.e_key[e] = e, ("session", req.session)
                self.n["recompute_tokens"] += req.history  # an evicted history is recomputed
        self.e_state[e], self.e_readers[e], self.e_ttl[e] = ACTIVE, 1.0, math.inf
        self.e_tokens[e], self.e_stored[e] = start - pre_tokens, start
        self._write(e, own, "prefill", align=sess < 0)
        self._vllm_write(e, have, start)
        if self.opt.prefill == "colocated":
            prefill_toks = req.input + (pre_tokens if new_prefix else 0) + (req.history if sess < 0 else 0)
            dt = self.opt.stacks * prefill_toks / prefill_tok_s(self.model, self.gpu, start)
            self.prefill_dt += dt
            self.acc["prefill_s"] += dt
        slot = self.free_slots.pop()
        self.sl_on[slot], self.sl_req[slot] = True, req
        self.sl_entry[slot], self.sl_prefix[slot], self.sl_part[slot] = e, pre, part
        self.sl_ctx[slot], self.sl_end[slot], self.sl_gen[slot], self.sl_staged[slot] = start, end, 0, 0.0
        self.sl_dead[slot] = 0.0
        self.n_running += 1
        self.n["admitted"] += 1
        return True

    def _complete(self, slot):
        req, e, pre = self.sl_req[slot], int(self.sl_entry[slot]), int(self.sl_prefix[slot])
        self._entry_reads(e)
        if self.sl_staged[slot] > EPS:  # the request's last staged full pages (a partial row)
            self._write(e, float(self.sl_staged[slot]), "decode")
        if self.sl_dead[slot] > EPS:
            self._retire_dead(e, float(self.sl_dead[slot]))
            self.sl_dead[slot] = 0.0
        if self.writes_decode:
            self.e_stored[e] = self.sl_end[slot]
            self.e_tokens[e] = self.sl_end[slot] - (self.workload.prefix_tokens if pre >= 0 else 0)
        self.sl_on[slot], self.sl_req[slot], self.sl_staged[slot], self.sl_dead[slot] = False, None, 0.0, 0.0
        self.free_slots.append(slot)
        self.n_running -= 1
        self.n["completed"] += 1
        w = self.workload
        if pre >= 0:
            self.e_readers[pre] -= 1
            if self.e_readers[pre] <= 0:
                if w.retain_s > 0:
                    self._idle(pre, self.t + w.retain_s)
                else:
                    self._kill(pre)
        if req.session >= 0 and not req.last:
            self._idle(e, math.inf)
        elif req.retain_s > 0:
            self._idle(e, self.t + req.retain_s)
        else:
            self._kill(e)
        self.stream.call_done(req, self.t)

    def _idle(self, e, ttl):
        self.e_state[e], self.e_readers[e], self.e_ttl[e] = IDLE, 0.0, ttl
        self.part_ver[self.e_part[e]] += 1

    def _kill(self, e, evicted=False):
        self._entry_reads(e)
        if evicted:
            self.n["evicted_pages"] += self.e_pages[e]
            self.n["recompute_tokens"] += self.e_tokens[e]
        for i in list(self.e_rows[e].values()):
            self._drop(i)
        for s in self.e_owned.pop(e, ()):
            if self.owner[s] == e:
                self.owner[s] = -1
                if self.state[s] == OPEN:
                    self.state[s] = FULL
                self._release(s)
        if self.vllm is not None and e in self.e_ids:
            self.vllm.free(self.e_ids.pop(e))
            self.vllm_dirty = True
        key = self.e_key.pop(e, None)
        if key is not None:
            table = self.prefix_entry if key[0] == "prefix" else self.session_entry
            if table.get(key[1]) == e:
                del table[key[1]]
        del self.e_rows[e]
        self.e_state[e], self.e_readers[e], self.e_pages[e], self.e_dseg[e] = 0, 0.0, 0.0, -1
        self.e_ttl[e], self.e_steps[e] = math.inf, 0.0
        self.free_entries.append(e)

    def _admit_pending(self):
        self.queue.extend(self.stream.arrived(self.t))
        w = self.workload
        if self.opt.policy == "layer_major":
            if not self.n_running:
                if self.region:
                    self._end_wave()
                self._start_wave()
            return
        if w.arrival == "closed" and not w.sessions:
            while self.n_running + len(self.queue) < self.cap:
                self.queue.append(self.stream.closed_new(self.t))
        if self.blocked == self.drops:  # nothing died or moved since the last failed admission
            return
        while self.queue and self.free_slots:
            if not self._admit(self.queue[0]):
                self.n["stalls"] += 1
                self.blocked = self.drops
                return
            self.queue.pop(0)
        self.blocked = None

    # ---- layer-major waves -------------------------------------------------------------------
    def _region_span(self, prompt):
        """Pages a wave writes: phase A's striped layout of H x slots heads at the longest prompt."""
        if prompt not in self.region_cache:
            if hasattr(self.geo, "region_span"):
                span = self.geo.region_span(self.cap, prompt, self.dev.pages_per_block)
            elif self.model is not None:
                span = region_pages(self.model, self.geo.tensors, self.cap, prompt, self.dev.pages_per_block)
            else:  # no model (tests): every per-head stream padded to whole pages
                per = sum(int(-(-(hi - lo)[0] // tpp)) for spec, tpp in zip(self.geo.hbf, self.geo.tpp)
                          for lo, hi in [self.geo.window(spec, np.array([prompt]))])
                span = float(self.cap * self.geo.layers * self.geo.heads * per)
            self.region_cache[prompt] = span
        return self.region_cache[prompt]

    def _start_wave(self):
        w = self.workload
        reqs = self.queue[:self.cap]
        del self.queue[:len(reqs)]
        while len(reqs) < self.cap and w.arrival == "closed" and not w.sessions:
            reqs.append(self.stream.closed_new(self.t))
        if not reqs:
            return
        prompts = [(w.prefix_tokens if r.prefix >= 0 else 0) + r.history + r.input for r in reqs]
        span = self._region_span(max(prompts))
        need = math.ceil(span / self.P - 1e-9)
        if self.free_count[0] < need:
            self._reclaim(0, force=True, target=lambda: self.free_count[0] >= need)
        if self.free_count[0] < need:
            raise CapacityError(f"A wave region of {span:.0f} pages needs {need} segments, "
                                f"{self.free_count[0]} are free")
        self.region = [self._take(0) for _ in range(need)]
        fills, left = [], span
        for s in self.region:
            self._erase(s, "wave")
            self.state[s], self.owner[s] = FULL, -2
            self.fill[s] = min(float(self.P), left)
            fills.append(self.fill[s])
            left -= self.fill[s]
        real = 0.0
        for req, prompt in zip(reqs, prompts):
            pages = self.pages(prompt)
            real += pages
            e = self._new_entry(PRIVATE, 0, prompt)
            for s, f in zip(self.region, fills):
                k = pages * f / span
                self.live[s] += k
                self.part_live[0] += k
                self.e_pages[e] += k
                self._row(e, s, k)
            self.e_readers[e] = 1.0
            slot = self.free_slots.pop()
            self.sl_on[slot], self.sl_req[slot], self.sl_entry[slot], self.sl_prefix[slot] = True, req, e, -1
            self.sl_ctx[slot], self.sl_end[slot], self.sl_gen[slot] = prompt, prompt + req.output, 0
            self.sl_part[slot], self.sl_staged[slot], self.sl_dead[slot] = 0, 0.0, 0.0
            self.n_running += 1
            self.n["admitted"] += 1
            if self.opt.prefill == "colocated":
                dt = self.opt.stacks * prompt / prefill_tok_s(self.model, self.gpu, prompt)
                self.prefill_dt += dt
                self.acc["prefill_s"] += dt
        self._program("prefill", real)
        self._program("padding", span - real)
        self.n["waves"] += 1

    def _end_wave(self):
        for s in self.region:
            self.owner[s] = -1
            self._release(s)
        self.region = []

    # ---- refresh and reclaim -----------------------------------------------------------------
    def _refresh(self, s, cause):
        self.n[cause] += 1
        if self.refresh == "replay":
            self._replay(s)
        else:
            self._relocate(s, cause)

    def _replay(self, s):
        """Rewrite rows 0 .. fill - 1 of segment s in place, dead pages included (spec section 5.7)."""
        self._flush_reads()
        self._erase(s, "replay")
        self._program("replay", self.fill[s])
        n = self.c_count
        self.c_t[:n][self.c_v[:n] & (self.c_s[:n] == s)] = self.t

    def _relocate(self, s, cause):
        """Move the live pages of segment s (refresh or compaction); idle entries may be evicted instead.

        The pages go to the partition's frontier, or to the owner's own segment (dedicated). Segment s
        is free afterwards; its erase is counted at its next row-0 write.
        """
        self._flush_reads()
        self.busy[s] = True
        o = int(self.owner[s])
        if o >= 0:
            self.e_owned[o].remove(s)
            self.owner[s] = -1
            if self.e_dseg[o] == s:
                self.e_dseg[o] = -1
        if self.state[s] == OPEN:
            self._close(s)
        evict = self.opt.reclaim == "evict" if cause == "compaction" else self.opt.refresh_idle == "evict"
        n = self.c_count
        moves = []
        for i in np.flatnonzero(self.c_v[:n] & (self.c_s[:n] == s)).tolist():
            if not self.c_v[i]:
                continue
            e = int(self.c_e[i])
            if evict and self.e_state[e] == IDLE and e not in self.protected:
                self._kill(e, evicted=True)
                continue
            moves.append(self._drop(i))
        moved = {e for e, _ in moves}
        self.protected.update(moved)
        try:
            for e, pages in moves:
                self._write(e, pages, cause)
        finally:
            self.protected.difference_update(moved)
        self.busy[s] = False
        self.reads[s] = 0.0
        self._release(s)

    def _victim_state(self):
        """Per-segment copy counts and idle-owner mask for victim selection (shared across partitions)."""
        n = self.c_count
        e = self.c_e[:n]
        v = self.c_v[:n]
        if self.opt.reclaim == "compact":
            keep = (self.e_state[e] > 0) & v
        else:
            keep = (self.e_state[e] == ACTIVE) & v
            if self.protected:
                keep |= np.isin(e, list(self.protected)) & v
        if self.opt.separate_prefix:
            is_pre = self.e_kind[e] == PREFIX
            copy_main = np.bincount(self.c_s[:n][keep & ~is_pre], self.c_n[:n][keep & ~is_pre], self.S)
            copy_pre = np.bincount(self.c_s[:n][keep & is_pre], self.c_n[:n][keep & is_pre], self.S)
            copy = copy_main + copy_pre
        else:
            copy_main = copy_pre = None
            copy = np.bincount(self.c_s[:n][keep], self.c_n[:n][keep], self.S)
        owner_idle = np.zeros(self.S, dtype=bool)
        owned = self.owner >= 0
        if owned.any():
            owner_idle[owned] = (self.e_state[self.owner[owned]] == IDLE)
            if self.protected:
                owner_idle[owned] &= ~np.isin(self.owner[owned], list(self.protected))
            copy[owner_idle] = 0.0
            if self.opt.separate_prefix:
                copy_main[owner_idle] = 0.0
                copy_pre[owner_idle] = 0.0
        return [copy, owner_idle, None, copy_main, copy_pre]

    def _oldest_times(self, state):
        if state[2] is None:
            n = self.c_count
            v = self.c_v[:n]
            oldest = np.full(self.S, math.inf)
            np.minimum.at(oldest, self.c_s[:n][v], self.c_t[:n][v])
            state[2] = oldest
        return state[2]

    def _victim(self, part, state=None):
        """Reclaim victim of ``part``: fewest pages to copy (greedy) or highest (1 - u) * age / (1 + u)."""
        if state is None:
            state = self._victim_state()
        copy, owner_idle, _, copy_main, copy_pre = state
        sl = slice(part, self.S, self.parts)
        cand = (~self.busy[sl] & ~self.is_free[sl] & (self.live[sl] > EPS)
                & (((self.state[sl] == FULL) & (self.owner[sl] == -1)) | owner_idle[sl]))
        local = np.flatnonzero(cand)
        if not len(local):
            return -1
        index = part + local * self.parts
        if self.opt.separate_prefix:
            rem_main = self._frontier_rem(part, prefix=False)
            rem_pre = self._frontier_rem(part, prefix=True)
            cm, cp = copy_main[index], copy_pre[index]
            segs_main = np.where(cm > EPS, np.maximum(0, np.ceil((cm - rem_main) / self.P - 1e-9)), 0)
            segs_pre = np.where(cp > EPS, np.maximum(0, np.ceil((cp - rem_pre) / self.P - 1e-9)), 0)
            fits = (copy[index] < self.P - EPS) & (segs_main + segs_pre <= self.free_count[part])
            index = index[fits]
            if not len(index):
                return -1
        else:
            room = self._room(part)
            if self.opt.victim == "cost_benefit":
                index = index[(copy[index] < self.P - EPS) & (copy[index] <= room + EPS)]
                if not len(index):
                    return -1
            else:
                c_idx = copy[index]
                c_min = float(c_idx.min())
                if c_min >= self.P - EPS or c_min > room + EPS:
                    return -1
                least = index[c_idx <= c_min + EPS]
                if len(least) > 1:  # ties: the segment with the oldest data (least recently written first)
                    oldest = self._oldest_times(state)
                    least = least[np.argsort(oldest[least], kind="stable")]
                return int(least[0])
        if self.opt.victim == "cost_benefit":
            oldest = self._oldest_times(state)
            u = copy[index] / self.P
            age = np.maximum(self.t - oldest[index], 1e-6)
            score = (1.0 - u) * age / (1.0 + u)
            best_score = float(score.max())
            top = index[score >= best_score - 1e-12 * max(1.0, best_score)]
            if len(top) > 1:
                top = top[np.argsort(copy[top], kind="stable")]
            return int(top[0])
        c_idx = copy[index]
        c_min = float(c_idx.min())
        least = index[c_idx <= c_min + EPS]
        if len(least) > 1:
            oldest = self._oldest_times(state)
            least = least[np.argsort(oldest[least], kind="stable")]
        return int(least[0])

    def _reclaim(self, part, force=False, target=None, state=None):
        """Free segments of ``part``: while fewer than the low watermark, or (forced) until ``target`` holds.
        Returns True if any segment was reclaimed."""
        if self.guard:
            return False
        if not force and not self.protected and self.reclaim_blocked[part] == self.part_ver[part]:
            return False
        self.guard = True
        changed = False
        try:
            for _ in range(self.S):
                if force:
                    if target is not None and (target() if callable(target) else self._room(part) >= target):
                        break
                elif self.free_count[part] >= self.low_wm:
                    break
                victim = self._victim(part, state)
                if victim < 0:
                    if not self.protected:
                        self.reclaim_blocked[part] = self.part_ver[part]
                    break
                changed = True
                state = None  # partition state changed; recompute if another victim is needed
                self.n["reclaims"] += 1
                owner = int(self.owner[victim])
                if owner >= 0:  # dedicated: an idle owner is evicted with all its segments
                    self._kill(owner, evicted=True)
                else:
                    self.n["compaction"] += 1
                    self._relocate(victim, "compaction")
        finally:
            self.guard = False
        return changed

    # ---- one quantum ---------------------------------------------------------------------------
    def _next_retention(self):
        if not self.opt.retention_s:
            return math.inf
        n = self.c_count
        valid = self.c_v[:n]
        return float(self.c_t[:n][valid].min()) + self.opt.retention_s if valid.any() else math.inf

    def _next_ttl(self):
        idle = self.e_state == IDLE
        return float(self.e_ttl[idle].min()) if idle.any() else math.inf

    def _clock_step(self):
        on = np.flatnonzero(self.sl_on)
        parts = self.sl_part[on] if self.parts > 1 else None
        hbm = self.sl_gen[on] if self.hbm_out else None
        return self.clock.step(self.sl_ctx[on], parts, self.row if parts is not None else None, hbm)

    def _limit(self, n, T, horizon, warm):
        """Cut the quantum at the earliest predicted threshold crossing, retention deadline, warm-up end or horizon."""
        o, t = self.opt, self.t
        rates = self._rates()
        read = rates > 0
        if read.any():
            rate = rates[read]
            floor = max(1, int(o.rd_slack * o.x / rate.max()))
            first = float(((o.x - self.reads[read]) / rate).min())
            n = min(n, max(floor, math.ceil(first - 1e-9)))
        for deadline in (self._next_retention(), warm if self.warm is None else math.inf, horizon):
            if deadline < math.inf:
                n = min(n, max(1, math.ceil((deadline - t) / T - 1e-9)))
        return n

    def _advance(self, on, k, T):
        ctx0 = self.sl_ctx[on]
        ctx1 = ctx0 + k
        dt = k * T
        self._update_stranded(dt)
        if self.writes_decode:
            if hasattr(self.geo, "written_pages"):
                self.sl_staged[on] += self.geo.written_pages(ctx0, ctx1)
            else:
                self.sl_staged[on] += self.geo.stored_pages(ctx1) - self.geo.stored_pages(ctx0)
        if hasattr(self.geo, "dead_pages"):
            dead = self.geo.dead_pages(ctx0, ctx1)
            if dead.any():
                for idx in np.flatnonzero(dead).tolist():
                    slot = int(on[idx])
                    rem = self._retire_dead(int(self.sl_entry[slot]), float(dead[idx]))
                    if rem > EPS:
                        self.sl_dead[slot] += rem
        if self.vllm is not None:
            for idx, slot in enumerate(on.tolist()):
                self._vllm_write(int(self.sl_entry[slot]), int(ctx0[idx]), int(ctx1[idx]))
        np.add.at(self.e_steps, self.sl_entry[on], k)
        pre = self.sl_prefix[on]
        pre = pre[pre >= 0]
        if len(pre):
            if self.opt.prefix_reads == "per_request":
                np.add.at(self.e_steps, pre, k)
            else:
                self.e_steps[np.unique(pre)] += k
        self.pending_reads = True
        self.sl_ctx[on] = ctx1
        self.sl_gen[on] += k
        self.acc["batch_s"] += len(on) * dt
        self.acc["ctx_s"] += float(ctx0.sum() + ctx1.sum()) / 2 * dt
        self.acc["tokens"] += len(on) * k

    def _consume_prefill(self):
        """Advance clock by serial colocated-prefill time accumulated during admissions."""
        if self.prefill_dt > 0:
            dt = self.prefill_dt
            self.prefill_dt = 0.0
            self._update_stranded(dt)
            self.t += dt
            return dt
        return 0.0

    def _quantum(self, n, T, stop=math.inf):
        """Run n decode steps of length T; requests complete and arrive at their exact step. Returns steps run."""
        t0, done, prefill_q = self.t, 0, 0.0
        splits = self.opt.policy != "layer_major"
        while done < n:
            on = np.flatnonzero(self.sl_on)
            if not len(on):
                break
            k = max(1, min(n - done, int((self.sl_end[on] - self.sl_ctx[on]).min())))
            if splits and self.free_slots and self.blocked != self.drops:
                arrival = self.stream.next_time()
                if arrival < t0 + n * T + prefill_q:
                    k = min(k, max(1, math.ceil((arrival - t0 - prefill_q) / T - 1e-9) - done))
            self._advance(on, k, T)
            done += k
            self.t = t0 + done * T + prefill_q
            finished = on[self.sl_ctx[on] >= self.sl_end[on]]
            for slot in finished.tolist():
                self._complete(slot)
            if len(finished) or (self.free_slots and self.stream.next_time() <= self.t + 1e-12):
                self._admit_pending()
                prefill_q += self._consume_prefill()
                if self.t >= stop - 1e-9:
                    break
        return done

    def _flush_decode(self):
        """Write staged decode pages in whole rows (per partition for packed, per request for dedicated)."""
        if not self.writes_decode:
            return
        on = np.flatnonzero(self.sl_on & (self.sl_staged > EPS))
        if not len(on):
            return
        row = self.row
        if self.opt.policy == "dedicated":
            for slot in on.tolist():
                whole = math.floor(self.sl_staged[slot] / row + 1e-9) * row
                if whole:
                    self.sl_staged[slot] -= whole
                    e = int(self.sl_entry[slot])
                    self._write(e, whole, "decode")
                    if self.sl_dead[slot] > EPS:
                        self.sl_dead[slot] = self._retire_dead(e, float(self.sl_dead[slot]))
            return
        for part in np.unique(self.sl_part[on]).tolist():
            group = on[self.sl_part[on] == part]
            total = float(self.sl_staged[group].sum())
            whole = math.floor(total / row + 1e-9) * row
            if whole <= 0:
                continue
            share = whole / total
            for slot in group.tolist():
                pages = float(self.sl_staged[slot]) * share
                self.sl_staged[slot] -= pages
                e = int(self.sl_entry[slot])
                self._write(e, pages, "decode")
                if self.sl_dead[slot] > EPS:
                    self.sl_dead[slot] = self._retire_dead(e, float(self.sl_dead[slot]))

    def _housekeeping(self):
        """Idle expiry, read-disturb refresh, retention refresh and background reclaim."""
        o = self.opt
        for e in np.flatnonzero((self.e_state == IDLE) & (self.e_ttl <= self.t + 1e-9)).tolist():
            if self.e_state[e] == IDLE:
                self._kill(e)
        self._flush_reads()
        for s in np.flatnonzero(self.reads >= o.x * (1 - 1e-9)).tolist():
            if self.reads[s] < o.x * (1 - 1e-9):
                continue
            if self.live[s] > EPS:
                self._refresh(s, "refresh_rd")
            elif self.state[s] == OPEN and self.owner[s] == -1:
                self._close(s)  # blocks past X take no new data
        if o.retention_s:
            n = self.c_count
            due = self.c_v[:n] & (self.c_t[:n] <= self.t - o.retention_s + 1e-6)
            for s in np.unique(self.c_s[:n][due]).tolist():
                if self.live[s] > EPS:
                    self._refresh(s, "refresh_ret")
        if o.policy != "layer_major":
            need = [part for part in range(self.parts)
                    if self.free_count[part] < self.low_wm and self.reclaim_blocked[part] != self.part_ver[part]]
            if need:
                state = self._victim_state()
                for part in need:
                    if self.free_count[part] < self.low_wm:
                        if state is None:
                            state = self._victim_state()
                        if self._reclaim(part, state=state):
                            state = None

    def _m5(self, step, steps):
        programs, erases = float(self.programs.sum()), self.m5_erases
        if self.opt.m5 != "off":
            o = self.opt
            load = {"programs": (programs - self.m5_mark[0]) / steps, "erases": (erases - self.m5_mark[1]) / steps,
                    "wave_banks": self.row, "gaps_ns": step.gaps_ns, "step_ns": step.tpot_ns}
            result = interference(load, o.m5, o.t_prog, o.t_bers, o.suspend_ns, block_unit=o.block_unit)
            self.m5_extra = result["extra_step_ns"]
            if self.warm is not None:
                self.max_step_extra = max(self.max_step_extra, result["max_step_extra_ns"])
        self.m5_mark = (programs, erases)

    def _staging(self):
        on = np.flatnonzero(self.sl_on)
        pages = float(self.sl_staged[on].sum()) + len(on) * self.geo.partial_pages()
        held = float(self.sl_gen[on].sum()) * self.geo.hbf_token_bytes if self.hbm_out else 0.0
        self.staging_peak = max(self.staging_peak, pages * self.dev.page_bytes + held)

    def run(self):
        """Simulate ``hours``; returns the summary row (rates over the time after the warm-up)."""
        o = self.opt
        wall = time.perf_counter()
        horizon, warm = o.hours * 3600, o.warmup_h * 3600
        next_report, step, stuck = 0.0, None, 0
        if warm <= 0:
            self.warm = self._snapshot()
        self._admit_pending()
        self._consume_prefill()
        while self.t < horizon - 1e-9:
            if self.warm is None and self.t >= warm - 1e-9:
                self.warm = self._snapshot()
            if self.t >= next_report - 1e-9:
                self._record(step)
                next_report = (math.floor(self.t / o.report_s + 1e-9) + 1) * o.report_s
            if not self.n_running:
                if self.queue and self.blocked == self.drops and self._next_ttl() == math.inf:
                    raise CapacityError(f"The next request does not fit even after reclaim (t = {self.t:.1f} s, "
                                        f"{self.S} segments of {self.P} pages); raise --capacity-gib")
                events = [self.stream.next_time(), horizon, next_report, self._next_retention(), self._next_ttl()]
                if self.warm is None:
                    events.append(warm)
                before = self.t
                self.t = max(min(events), self.t)
                self._update_stranded(self.t - before)
                self._housekeeping()
                self.m5_mark = (float(self.programs.sum()), self.m5_erases)
                self._admit_pending()
                self._consume_prefill()
                stuck = stuck + 1 if self.t <= before and not self.n_running else 0
                if stuck > 1000:
                    raise RuntimeError(f"The simulation makes no progress at t = {self.t:.3f} s")
                continue
            step = self._clock_step()
            T = (step.tpot_ns + self.m5_extra) / 1e9
            n = self._limit(max(1, int(o.quantum_s / T + 1e-9)), T, horizon, warm)
            stop = min(horizon, warm) if self.warm is None else horizon
            steps = self._quantum(n, T, stop)
            self.n["quanta"] += 1
            self._staging()
            self._flush_reads()  # before the flush: staged pages were read from HBM, not HBF
            self._flush_decode()
            self._housekeeping()
            if steps:
                self.acc["steps"] += steps
                self.acc["decode_s"] += steps * T
                self.acc["tpot_s"] += steps * step.tpot_ns / 1e9
                self.acc["m5_s"] += steps * self.m5_extra / 1e9
                self._m5(step, steps)
            self._admit_pending()
            self._consume_prefill()
        if self.warm is None:
            self.warm = self._snapshot()
        self._record(step)
        self.wall_s = time.perf_counter() - wall
        return self.summary()

    # ---- outputs -------------------------------------------------------------------------------
    def _snapshot(self):
        return {"t": self.t, "programs": self.programs.copy(), "seg_erases": self.seg_erases.copy(),
                "block_erases": self.block_erases, "pec": self.pec.copy(), "acc": dict(self.acc), "n": dict(self.n),
                "vllm": dict(self.vllm.counts) if self.vllm is not None else {}}

    def _record(self, step):
        page, on = self.dev.page_bytes, np.flatnonzero(self.sl_on)
        idle = float(self.e_pages[self.e_state == IDLE].sum())
        stranded_frac = 0.0
        stranded_pages = 0.0
        if self.vllm is not None:
            if self.vllm_dirty:
                self.vllm_stranded = self.vllm.stranded()
                self.vllm_dirty = False
            stranded_frac = self.vllm_stranded
            stranded_pages = stranded_frac * self.vllm.ids * self.vllm.pages_per_id * self.geo.layers
        row = {"t_s": self.t, "running": self.n_running, "queued": len(self.queue),
               "sum_context": int(self.sl_ctx[on].sum()),
               "tpot_ms": step.tpot_ns / 1e6 if step else None,
               "tpot_m5_ms": (step.tpot_ns + self.m5_extra) / 1e6 if step else None}
        row |= {f"{cause}_gb": self.programs[i] * page / 1e9 for i, cause in enumerate(CAUSES)}
        row |= {f"erases_{cause}": int(self.seg_erases[i]) for i, cause in enumerate(CAUSES)}
        row |= {"block_erases": self.block_erases, "free_segments": int(self.free_count.sum()),
                "live_pages": float(self.live.sum()), "idle_pages": idle, "stranded_pages": stranded_pages,
                "stranded_frac": stranded_frac, "avg_pec": float(self.pec.mean()), "max_pec": int(self.pec.max()),
                "max_reads_over_x": float(self.reads.max() / self.opt.x)}
        self.timeseries.append(row)

    def summary(self):
        o, d, w0 = self.opt, self.dev, self.warm
        span = max(self.t - w0["t"], 1e-9)
        page, ppb = d.page_bytes, d.pages_per_block
        prog = self.programs - w0["programs"]
        host = prog[CAUSE["prefill"]] + prog[CAUSE["decode"]]
        total = float(prog.sum())
        erased = (self.block_erases - w0["block_erases"]) * ppb
        days = span / DAY
        dpec = self.pec - w0["pec"]
        avg_day = float(dpec.mean()) / days
        max_day = avg_day if o.wear == "ideal" else float(dpec.max()) / days
        acc = {key: self.acc[key] - w0["acc"][key] for key in self.acc}
        n = {key: self.n[key] - w0["n"][key] for key in self.n}
        steps, tokens = acc["steps"], acc["tokens"]
        mean_ctx = acc["ctx_s"] / acc["batch_s"] if acc["batch_s"] else 0.0
        tpot = acc["tpot_s"] / steps if steps else None
        refresh = prog[CAUSE["refresh_rd"]] + prog[CAUSE["refresh_ret"]] + prog[CAUSE["replay"]]
        vllm = ({key: self.vllm.counts[key] - w0["vllm"].get(key, 0) for key in self.vllm.counts}
                if self.vllm is not None else {})
        row = {"workload": self.workload.name, "policy": o.policy, "partition": d.partition, "pages_per_block": ppb,
               "x": o.x, "stacks": o.stacks, "mean_batch": acc["batch_s"] / span, "mean_context": mean_ctx,
               "mean_tpot_ms": tpot * 1e3 if steps else None,
               "tpot_m5_ms": (acc["tpot_s"] + acc["m5_s"]) / steps * 1e3 if steps else None,
               "max_step_extra_ms": self.max_step_extra / 1e6, "req_per_s": n["completed"] / span,
               "tokens_per_s": tokens / span, "pages_per_request": host / n["completed"] if n["completed"] else None,
               "dead_pages_per_request": n["dead_pages"] / n["completed"] if n["completed"] else None,
               "kv_bytes_per_token": self.geo.hbf_token_bytes,
               "hbm_bytes_per_request": float(self.geo.hbm_bytes(np.array([int(round(mean_ctx))]))[0]),
               "threshold_steps": threshold_steps(o.x, ppb, o.share, o.hot * self.read_mult),
               "refreshes_per_request": (prog[CAUSE["refresh_rd"]] + prog[CAUSE["replay"]]) / host if host else None,
               "kv_write_gb_s": host * page / span / 1e9}
        row |= {f"{cause}_gb_s": prog[i] * page / span / 1e9 for i, cause in enumerate(CAUSES)}
        row |= {"program_gb_s": total * page / span / 1e9,
                "program_bw_share": total * page / span / (d.banks * page / (o.t_prog / 1e9)),
                "wa": total / host if host else None, "erase_amp": erased / total if total else None,
                "erase_wa": erased / host if host else None, "refresh_to_kv": refresh / host if host else None,
                "erases_per_s": erased / ppb / span, "avg_pec_per_day": avg_day, "max_pec_per_day": max_day}
        for limit in o.pe_limits:
            row[f"days_to_limit_avg@{sci(limit)}"] = limit / avg_day if avg_day else math.inf
        for limit in o.pe_limits:
            row[f"days_to_limit_max@{sci(limit)}"] = limit / max_day if max_day else math.inf
        row["pe_per_mtok"] = avg_day / (tokens / span * DAY / 1e6) if tokens else None
        row["break_even_x"] = None
        row["staging_peak_mib"] = self.staging_peak / 2 ** 20
        row["pending_units_per_channel"] = row["mean_batch"] * self.geo.partial_pages() / 16
        row["unit_open_s"] = max(self.geo.tpp) * tpot if steps else None
        row["replay_buffer_mib"] = self.P * page / 2 ** 20 if self.refresh == "replay" else 0.0
        row["legal"] = o.policy != "vllm_pool"
        row |= {"hours": o.hours, "warmup_h": o.warmup_h, "measured_h": span / 3600, "segments": self.S,
                "segment_pages": self.P, "low_watermark": self.low_wm, "decode_steps": int(steps),
                "completed": int(n["completed"]),
                "refresh_events_rd": int(n["refresh_rd"]), "refresh_events_ret": int(n["refresh_ret"]),
                "compaction_events": int(n["compaction"]), "waves": int(n["waves"]),
                "segment_erases": int((self.seg_erases - w0["seg_erases"]).sum()),
                "free_segments_min": self.free_min, "evicted_gb": n["evicted_pages"] * page / 1e9,
                "recompute_tokens": n["recompute_tokens"], "stalls": int(n["stalls"]),
                "stranded_frac": acc["stranded_s"] / span if self.vllm is not None else 0.0,
                "violations_skip": int(vllm.get("skip", 0)),
                "violations_overlap": int(vllm.get("overlap", 0)),
                "violations_live_erase": int(vllm.get("live_erase", 0)),
                "legalize_gb": vllm.get("legalize_pages", 0) * page * self.geo.layers / 1e9,
                "prefill_s": acc["prefill_s"], "quanta": int(self.n["quanta"]), "wall_s": self.wall_s}
        return row

    def segment_rows(self):
        rows = []
        for s in range(self.S):
            row = {"segment": s, "partition": int(self.part_of[s]), "state": int(self.state[s]),
                   "pec": int(self.pec[s]), "live_pages": float(self.live[s]), "fill_pages": float(self.fill[s]),
                   "reads": float(self.reads[s])}
            row |= {f"erases_{cause}": int(self.erases[s, i]) for i, cause in enumerate(CAUSES)}
            rows.append(row)
        return rows


def simulate(geometry, clock, workload, device=None, options=None, model=None, gpu=None):
    """Run the event simulation of one stack; returns the Simulator with ``result`` (the summary row),
    ``timeseries`` and ``segment_rows()``. ``workload`` is one stack's share (``workload.per_stack``)."""
    sim = Simulator(geometry, clock, workload, device, options, model, gpu)
    sim.result = sim.run()
    return sim


# --------------------------------------------------------------------------------------------
# Command line, outputs and report

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
    parser = argparse.ArgumentParser(prog="python -m hbf.endurance", description=__doc__.split("\n\n")[0])
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
    from .model import load_gpu, load_model
    from .system import SystemOptions
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
    from .system import write_table
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
    from .system import write_table
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
    kv_spec_note = f"；KV 规格 `{config['kv_spec']}`" if config.get("kv_spec") else ""
    lines = ["# 寿命评估（解析稳态）", "",
             f"分配方式：{names[policy]}，分区 `{partition}`；读取方式 `{config['mode']}`{kv_spec_note}；"
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
    lines += _calibration_lines(calibrations)
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


def _calibration_lines(calibrations):
    if not calibrations:
        return []
    lines = ["", "## 时钟校准", "",
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
    return lines


SIM_NAMES = {"packed": "紧凑追加（packed）", "dedicated": "每请求独占段（dedicated）",
             "layer_major": "静态批次波（layer_major）", "hbm_decode": "生成的 KV 留在 HBM（hbm_decode）",
             "vllm_pool": "vLLM 空闲块池（vllm_pool，规范下不合法）"}


def sim_report_lines(rows, config, calibrations=()):
    """Chinese report section 寿命评估 of an event simulation, with every assumption and its source tag."""
    limits = config["pe_limits"]
    extras = []
    if config.get("kv_spec"):
        extras.append(f"KV 规格 `{config['kv_spec']}`")
    if config.get("victim", "greedy") != "greedy":
        extras.append(f"回收选段 `{config['victim']}`")
    if config.get("prefill", "disaggregated") != "disaggregated":
        extras.append(f"prefill `{config['prefill']}`")
    if config.get("pre_open"):
        extras.append("预开段 `--pre-open`")
    if config.get("separate_prefix"):
        extras.append("前缀独立前沿 `--separate-prefix`")
    if config["policy"] == "vllm_pool":
        extras.append(f"vLLM 重用 `{config.get('vllm_reuse', 'fifo')}`、空闲顺序 `{config.get('free_order', 'fresh')}`")
    extra_note = ("；" + "，".join(extras)) if extras else ""
    lines = ["# 寿命评估（事件模拟）", "",
             f"分配方式：{SIM_NAMES[config['policy']]}，分区 `{config['partition']}`；读取方式 `{config['mode']}`；"
             f"每个 stack 容量 {config['capacity_gib']:g} GiB，{config['stacks']} 个 stack 分担请求（模拟其中一个）；"
             f"tPROG {config['t_prog'] / 1e3:g} µs，tBERS {config['t_bers'] / 1e3:g} µs；"
             f"保持刷新间隔 {config['retention_s'] / 3600:g} h（0 表示关闭），持续读偏斜 h = {config['hot']:g}；"
             f"模拟 {config['hours']:g} h，前 {config['warmup_h']:g} h 预热不计入速率；"
             f"磨损分配 `{config['wear']}`，刷新 `{config['refresh']}`，空闲条目刷新时 `{config['refresh_idle']}`，"
             f"回收 `{config['reclaim']}`，写入干扰 `{config['m5']}`（以 {config['block_unit']} 为阻塞单位）"
             f"{extra_note}。", ""]
    for w in config["workloads"]:
        lines.append(f"- 负载 {w.describe()}。来源：{w.source}")
    lines += ["", "下表每行是一次模拟（一个 stack，预热之后的速率）。P/E 每天取各段 PEC 的增量：平均即 AVGPEC，"
              "最大即 MAXPEC（第 5.9.2.17–18 节）；“到 N 的天数”= N_PE / 每天 P/E。写放大 = 编程页 / 主机 KV 页；"
              "擦除放大 = 擦除的页 / 主机 KV 页。TPOT′ 含写入干扰（M5，滞后一个量子反馈到步时）。"
              "“解析”两列是同一设置下 `--analytic` 的值，用来对照。", "",
              "| 模型 | 格式 | 负载 | ppb | X | 批 | TPOT ms | TPOT′ ms | 最大单步拖慢 ms | KV 写 GB/s | 读扰刷新 GB/s "
              "| 保持刷新 GB/s | 搬移 GB/s | 重放 GB/s | 编程 GB/s | 刷新/KV | 写放大 | 擦除放大 | P/E 每天（平均） "
              "| P/E 每天（最大） | " + " | ".join(f"到 {sci(limit)} 天（最大）" for limit in limits)
              + " | 驱逐 GB | 解析 P/E 每天 | 解析 刷新/KV |",
              "|" + "---|" * 3 + "---:|" * (20 + len(limits))]
    for row in rows:
        cells = [row["model"], row["format"], row["workload"], str(row["pages_per_block"]), sci(row["x"]),
                 _fmt(row["mean_batch"]), _fmt(row["mean_tpot_ms"]), _fmt(row["tpot_m5_ms"]),
                 _fmt(row["max_step_extra_ms"]), _fmt(row["kv_write_gb_s"], 3), _fmt(row["refresh_rd_gb_s"], 3),
                 _fmt(row["refresh_ret_gb_s"], 3), _fmt(row["compaction_gb_s"], 3), _fmt(row["replay_gb_s"], 3),
                 _fmt(row["program_gb_s"], 3), _fmt(row["refresh_to_kv"]), _fmt(row["wa"]), _fmt(row["erase_wa"]),
                 _fmt(row["avg_pec_per_day"], 1), _fmt(row["max_pec_per_day"], 1)]
        cells += [_fmt(row[f"days_to_limit_max@{sci(limit)}"], 1) for limit in limits]
        cells += [_fmt(row["evicted_gb"], 1), _fmt(row["analytic_avg_pec_per_day"], 1),
                  _fmt(row["analytic_refresh_to_kv"])]
        lines.append("| " + " | ".join(cells) + " |")
    first = rows[0] if rows else None
    if first:
        lines += ["", f"每次模拟 {first['segments']} 个段（每段 {first['segment_pages']} 页），回收低水位 "
                  f"{first['low_watermark']} 段；第一行用了 {first['quanta']} 个量子、{first['wall_s']:.1f} s。"
                  "时间序列（每 --report-s 一行：运行与排队请求、TPOT、各原因累计写入与擦除、空闲段、有效页、平均与最大 PEC、"
                  "最大读次数 / X）在 endurance_timeseries.csv，每段的 PEC、各原因擦除次数和有效页在 endurance_segments.csv。"
                  f"HBM 暂存峰值 {first['staging_peak_mib']:.1f} MiB。"]
    lines += ["", "## 与解析模型的差别", "",
              "- 解析模型把生成的页当作从第一步起就在、每页按请求的全部解码步数计读次数；模拟按页的实际写入时间计，"
              "所以长输出（reasoning）的读扰刷新低于解析值。",
              "- 模拟按块计读：被搬移的数据落在正在填充的前沿段里，块中页少、每步读次数少，到 X 的时间更长；"
              "数据在到 X 前死去时它积累的读次数作废。整段由一个条目占满时两者一致（见 tests/test_endurance.py）。",
              "- 模拟里写入干扰会拖慢步时、降低请求速率；批大小和上下文随时间波动，TPOT 随批增大，"
              "所以按请求平均的步时高于平均批下的 TPOT（会话负载约低 10–15 % 的请求速率）。",
              "- 只在空闲段之间做磨损均衡：容量被空闲缓存占满时只有少数空闲段在轮转，最大 PEC 可以远高于平均值。"]
    lines += _calibration_lines(calibrations)
    lines += ["", "## 假设", ""]
    lines += [f"- {item}" for item in assumptions(config)]
    return lines


def assumptions(config):
    """Assumption list of the endurance model, each with its source tag (params.md tags in brackets).

    The closed form's list is unchanged; for an event simulation (``analytic`` false) the items that
    describe the closed form's shortcuts are replaced and the simulation's own items are appended.
    """
    items = _analytic_assumptions()
    if config.get("analytic", True):
        return items
    replace_items = {
        "一个请求的所有页": "读扰按块计：一个段（分区内所有 bank 的同一块号）里的条目每步各读一次自己的页，块读次数 = "
                    "Σ 条目页数 × 读比例 × h / 段的 bank 数；块读次数达到 X 时刷新整段。命中共享前缀或会话历史时，"
                    "缓存的每页在 prefill 读一次（不乘读比例和 h，与解析模型一致）。",
        "每天 P/E = 编程页数": "每天 P/E 取预热后各段 PEC 的增量除以天数：平均即理想磨损均衡下的 AVGPEC，"
                         "最大即 MAXPEC（第 5.9.2.17–18 节）；wear ideal 时最大按平均报告。",
        "packed：每个新条目": "packed：每个新条目（请求、前缀或会话）的 prompt 从前沿的下一行起点开始（--align row，填充按实际计）；"
                         "解码页先在 HBM 暂存，每个量子结束时按分区攒成整行写入；请求结束时剩下的整页一起写入"
                         "（与解析模型的每请求页数一致），只有每 (层, head, 张量) 最后不满的一页留在 HBM 丢弃。",
        "闭环：每个 stack": "闭环：每个 stack 保持 round(concurrency / stacks) 个请求（会话负载为会话数）；"
                       "请求在准确的解码步完成并立即补上，prefill 分离，KV 写入完成后开始解码。启动时前 B 个请求的输出依次缩短"
                       "（第 i 个解码 out − round(i·out/B) 个 token），会话从第 ⌊i·calls/B⌋ 次调用开始，使完成时间错开；"
                       "预热时间不计入速率（假设）。TPOT 每个量子按当时运行请求的上下文计算。",
        "写入干扰（M5）": "写入干扰（M5）：默认 gaps 模式（主机把编程、擦除放进 HBF 空闲间隙，符合第 11.5 节“刷新与读不同时发往同一 die”），"
                     "以 die 为阻塞单位；OCP v0.7.0 没有挂起（suspend）。每个量子按该量子的编程页数和块擦除数（每步平均）计算一次，"
                     "加到下一个量子的步时上（滞后一个量子），所以会降低请求速率；最大单步拖慢取预热后各量子的最大值。",
        "速率按每个 stack 计": "速率按每个 stack 计；N 个 stack 时只模拟一个 stack（并发 round(C/N) 或到达率 λ/N），GPU 计算批为 b × N。",
        "vllm_pool：": None,
    }
    out = []
    for item in items:
        prefix = next((key for key in replace_items if item.startswith(key)), None)
        if prefix is None:
            out.append(item)
        elif replace_items[prefix] is not None:
            out.append(replace_items[prefix])
    out += [
        "段级状态（设计 B 第 4.1 节）：段 = 分区内所有 bank 的同一块号（ppb 256 时超级块 1 GiB、通道块 64 MiB、die 块 16 MiB）；"
        "按行对齐放置时一个条目在段内每个块里的页数相同，所以按段计数是精确的（假设：主机按行写入）。",
        f"量子：每个量子至多 {config.get('quantum_s', 1.0):g} s，在最早的预计读扰阈值或保持到期处截断，但不短于最快的段阈值步数的 "
        f"{config.get('rd_slack', .01):g}（刷新最多晚这么多的 X，计数偏差不超过这个比例）；量子内步时不变，"
        "请求在准确的步完成和到达。暂存在 HBM 的页不计 HBF 读次数，量子内新写入的页从下一个量子开始计读。",
        "读扰与保持刷新：relocate（日志式分配的默认）把段内有效页写到前沿（dedicated 写到条目自己的新段），旧段在下次写第 0 行时才擦除；"
        "replay（layer_major 默认，第 5.7 节）原地重写第 0 行到最后写入的行，包括无效页和填充页（不得跳跃），需要一段的 HBM 副本。"
        "--refresh-idle evict 时段内的空闲缓存条目直接丢弃（以后重算）。",
        "回收：某分区空闲段少于低水位（默认 2 段 + 一个最大请求的段数）时，选需要搬移的页最少的段（贪心；相同时选数据最老的；"
        "`--victim cost_benefit` 按 (1−u)·age/(1+u) 选段），"
        "先驱逐其中的空闲条目（整条丢弃，记为重算 token），再把活跃页搬到前沿（compaction）；--reclaim compact 时空闲条目也搬移。"
        "准入时为请求预留全部预计页数（prefill + 解码）和低水位减一段；做不到时先强制回收，仍放不下就排队。",
        "磨损分配：none 取编号最小的空闲段，dynamic 取 PEC 最小的空闲段，ideal 同 dynamic，"
        "remap 按区域循环轮转（第 11.4.1 节区域重映射，无额外写入）。只在空闲段之间均衡，"
        "不搬移冷数据（没有静态磨损均衡），所以容量被空闲缓存占满时最大 PEC 会远高于平均。"
        "新条目放在有效页最少的分区，相同时选磨损最少的分区（wear none 时选编号最小的）。",
        "hbm_decode 的会话：上一次调用生成的 KV 不写入 HBF，由下一次调用的 prompt 一起写入（偏保守）。"
        "被驱逐的会话历史在下一次调用时整段重算并重写。",
        "vllm_pool：磨损按 packed 计，同时由 VllmTracker 跟踪每层块池的块号分配与释放，统计写序违规"
        "（skip/overlap/live_erase）、时间平均滞留比例（stranded_frac）与合法化所需重放写入量（legalize_gb）；"
        "该策略在规范下不合法（legal = False）。",
        "可选开关：`--prefill colocated` 在每次准入时串行计入 input / prefill_tok_s 的 prefill 时间；"
        "`--pre-open` 在空闲时预先打开段（非重放擦除不计入 M5 干扰，P/E 照计）；"
        "`--separate-prefix` 把共享前缀写到独立前沿以实现冷热分离。",
    ]
    return out


def _analytic_assumptions():
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
    except (OSError, ValueError, CapacityError) as error:
        parser.error(str(error))
    print(Path(directory).resolve())


if __name__ == "__main__":
    main()
