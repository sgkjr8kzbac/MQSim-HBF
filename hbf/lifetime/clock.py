"""Step-time Clock interface and rounds-backend calibration."""
import hashlib
import json
import math
import os
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry import F1, PARTITIONS, geometry_for
from .interference import hbf_gaps


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


class StepClock(ABC):
    """Core clock interface for lifetime models: maps active request contexts to ``StepTime``."""

    stacks: int = 1

    @abstractmethod
    def describe(self):
        """Return a JSON-serialisable summary of clock parameters."""

    @abstractmethod
    def step(self, contexts, partitions=None, partition_banks=None, hbm_tokens=None):
        """Return ``StepTime`` of one decode step, or ``None`` when ``contexts`` is empty."""


class ReadTimeClock(StepClock):
    """Core HBF clock: step time = HBF read time (rounds model) + constant compute term.

    Depends only on HBF geometry and read-time parameters (no GPU roofline or ``hbf.system``).
    Each layer's read time is ``layer_ns + round_ns * rounds`` when ``rounds > 0``, else ``0``.
    The constant ``compute_ns`` is spread evenly across layer idle gaps so that
    ``tpot_ns == sum(h) + compute_ns``.
    """

    def __init__(self, geometry, layers, *, read_ns=4096, banks=1024, stacks=1,
                 layer_ns=F1["layer_ns"], round_ns=F1["round_ns"], balance=1.0,
                 ratios=None, rounds_fn=None, compute_ns=0.0):
        self.geometry = geometry
        self.layers = layers
        self.read_ns, self.banks, self.stacks = read_ns, banks, stacks
        self.layer_ns, self.round_ns, self.balance = layer_ns, round_ns, balance
        self.ratios = dict(ratios or {})
        self.rounds_fn = rounds_fn
        self.compute_ns = float(compute_ns)
        self.cache = {}

    def describe(self):
        return {"clock": "read_time", "layer_ns": self.layer_ns, "round_ns": self.round_ns,
                "balance": self.balance, "ratios": self.ratios, "read_ns": self.read_ns,
                "banks": self.banks, "stacks": self.stacks, "compute_ns": self.compute_ns}

    def _rounds_for(self, ctx, hbf_ctx, partitions, partition_banks):
        geo = self.geometry
        share = self.ratios.get("pages", 1.0)
        if hasattr(geo, "layer_row_pages"):
            per_req_layers = geo.per_request_layer_row_pages(hbf_ctx) * share
            layer_pages = per_req_layers.sum(axis=1)
            if partitions is not None and partition_banks:
                parts = np.asarray(partitions, dtype=np.int64)
                layer_rounds = tuple(
                    math.ceil(self.balance * float(np.bincount(parts, weights=per_req_layers[l]).max())
                              / partition_banks - 1e-9) if layer_pages[l] > 0 else 0
                    for l in range(self.layers)
                )
            else:
                layer_rounds = tuple(
                    math.ceil(self.balance * float(layer_pages[l]) / self.banks - 1e-9) if layer_pages[l] > 0 else 0
                    for l in range(self.layers)
                )
            return layer_rounds, float(layer_pages.mean())
        per_req = geo.per_request_row_pages(hbf_ctx) * share
        pages = float(per_req.sum())
        if partitions is not None and partition_banks:
            parts = np.asarray(partitions, dtype=np.int64)
            max_load = float(np.bincount(parts, weights=per_req).max())
            rounds = math.ceil(self.balance * max_load / partition_banks - 1e-9)
        elif self.rounds_fn is not None and len(set(ctx.tolist())) == 1 and len(ctx) == len(hbf_ctx) and np.array_equal(ctx, hbf_ctx):
            rounds = self.rounds_fn(len(ctx), int(ctx[0]))
        else:
            rounds = math.ceil(self.balance * pages / self.banks - 1e-9)
        return (rounds,) * self.layers, pages

    def step(self, contexts, partitions=None, partition_banks=None, hbm_tokens=None):
        """StepTime of one decode step, or None when nothing runs."""
        ctx = np.asarray(contexts, dtype=np.int64)
        b = len(ctx)
        if not b:
            return None
        hbf_ctx = ctx if hbm_tokens is None else ctx - np.asarray(hbm_tokens, dtype=np.int64)
        layer_rounds, mean_pages = self._rounds_for(ctx, hbf_ctx, partitions, partition_banks)
        hetero = hasattr(self.geometry, "layer_row_pages")
        key = (b, layer_rounds, round(mean_pages, 6))
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        h = [self.layer_ns + self.round_ns * r_l if r_l else 0.0 for r_l in layer_rounds]
        read_total = sum(h)
        tpot = read_total + self.compute_ns
        gap = self.compute_ns / self.layers if self.layers else 0.0
        gaps = (gap,) * self.layers
        out = StepTime(
            tpot, self.compute_ns, self.compute_ns,
            read_total / self.layers if self.layers else 0.0,
            gaps, mean_pages,
            max(layer_rounds) if layer_rounds else 0, b,
            layer_rounds if hetero else (),
        )
        self.cache[key] = out
        return out


class Clock(StepClock):
    """Simulated decode step time of a running set (per stack), combining HBF reads with GPU TPOT.

    est per layer = layer_ns + round_ns x rounds, rounds = ceil(kappa x pages / banks) (or the
    busiest partition), fed as layer rows into ``system.evaluate`` with compute batch
    b x stacks and analysis batch b. ``ratios`` scale the dense demand for a sparse mode
    (pages, attended tokens and elements, burst, HBM bytes, dequant), from ``calibrate``.
    ``rounds_fn(b, ctx)``, when given, returns exact rounds for b equal contexts (phase A layout).
    """

    def __init__(self, model, gpu, geometry, options=None, *, read_ns=4096, banks=1024, stacks=1,
                 layer_ns=F1["layer_ns"], round_ns=F1["round_ns"], balance=1.0, ratios=None, selection="static",
                 kv_format="bf16", rounds_fn=None):
        from ..system import SystemOptions
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
            from ..system import evaluate
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
        from ..system import evaluate
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


class ConstantClock(StepClock):
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
    from ..device.config import load
    from ..frontend.kv.formats import parse_format
    from ..frontend.kv.layouts import LAYOUTS
    from ..frontend.kv.tensors import Shape
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
# Calibration directly against frontend + rounds backend (no CLI or CSV files)

_DEVICE_INT_FLAGS = {
    "--read-ns": "read_ns",
    "--program-ns": "program_ns",
    "--erase-ns": "erase_ns",
    "--aggregate-gb-s": "aggregate_gb_s",
    "--pages-per-block": "pages_per_block",
    "--blocks-per-plane": "blocks_per_plane",
    "--planes": "planes",
    "--read-plane-ns": "read_plane_ns",
    "--coalesce-ns": "coalesce_ns",
}


def _parse_device_overrides(device):
    """Parse ``device`` flag sequence or mapping into ``(topology, kwargs)`` for ``device.config.load``."""
    topology = "ocp"
    overrides = {}
    if isinstance(device, dict):
        overrides = dict(device)
        topology = overrides.pop("topology", topology)
        return topology, overrides
    tokens = list(device or ())
    i = 0
    while i < len(tokens):
        flag = tokens[i]
        if flag == "--topology":
            topology = tokens[i + 1]
            i += 2
        elif flag == "--plane-rule":
            overrides["plane_rule"] = tokens[i + 1]
            i += 2
        elif flag == "--cache-read":
            overrides["cache_read"] = True
            i += 1
        elif flag in _DEVICE_INT_FLAGS:
            overrides[_DEVICE_INT_FLAGS[flag]] = int(tokens[i + 1])
            i += 2
        else:
            raise ValueError(f"Unsupported device override flag for calibration: {flag!r}")
    return topology, overrides


def phase_a(directory, model, gpu, kv_format, batch, context, *, mode="dense", generation=None, device=(),
            alloc="layer_major", partition="none", kv_spec=None):
    """Run frontend trace generation + rounds backend directly in memory (no ``hbf.cli`` or CSVs)."""
    from ..backends.rounds.analysis import analyze_trace
    from ..device.config import load
    from ..frontend.addrtrace import iter_trace
    from ..frontend.kv.layouts import LAYOUTS
    from ..frontend.kv.placement import BatchRequest, BatchSpec, PageTableLayout, generate_batch
    from ..frontend.kv.tensors import Shape, default_tensors
    from ..frontend.trace.synthetic import generate
    from ..kvspec import GroupedLayout, from_model, generate_grouped, parse_kvspec, resolve_format
    from ..models import load_model

    topology, overrides = _parse_device_overrides(device)
    cfg = load(topology, **overrides)
    model_spec = load_model(model)
    fmt = resolve_format(model_spec, kv_format or "bf16")
    tensors = fmt.tensors
    shape = Shape(model_spec.layers, model_spec.kv_heads * batch, model_spec.head_dim, context)

    kv_spec_obj = None
    if kv_spec is not None:
        kv_spec_obj = parse_kvspec(kv_spec, model_spec.layers, model_spec.kv_heads, context)
    elif model_spec.layer_pattern:
        kv_spec_obj = from_model(model_spec, fmt)

    gen = {
        "steps": 1, "seed": 1, "fraction": 0.1, "count": 4, "unit": 0,
        "hot_fraction": 0.5, "hot_units": 1, "select_tokens": 16,
    } | dict(generation or {})
    if mode in ("uniform_tokens", "uniform_pages", "sink_recent", "channel_select") and fmt.full_tensors:
        gen.setdefault("full_tensors", list(fmt.full_tensors))

    if alloc != "layer_major" and kv_spec_obj is None:
        spec = BatchSpec(tuple(BatchRequest(context, 0) for _ in range(batch)),
                         alloc, partition, "striped_contiguous")
        per_request = Shape(model_spec.layers, model_spec.kv_heads, model_spec.head_dim, context)
        layout = PageTableLayout(cfg, per_request, spec, tensors=tensors)
        accesses = list(generate_batch(cfg, layout, mode, **{key: gen[key] for key in (
            "steps", "seed", "fraction", "count", "select_tokens")}))
    elif kv_spec_obj is not None:
        layout = GroupedLayout(cfg, shape, kv_spec_obj, tensors or default_tensors(shape),
                               requests=batch, intra="striped")
        accesses = list(generate_grouped(cfg, layout, mode, **{key: gen[key] for key in (
            "steps", "seed", "fraction", "count", "select_tokens")}))
    else:
        layout = LAYOUTS["striped_contiguous"](cfg, shape, tensors=tensors)
        accesses = list(generate(cfg, layout, mode, **gen))

    _, chunk_stream = iter_trace(cfg, layout, accesses, prefill=False, expected_steps=(0,),
                                 layout_name="striped_contiguous")
    layer_rows, step_rows = analyze_trace(cfg, chunk_stream, (1,), None, timing=True)
    attn_dims = fmt.attn_dims if fmt else 2 * shape.head_dim
    for row in layer_rows:
        row["attn_dims"] = attn_dims

    read_ns = sum(row["est_ns"] for row in layer_rows if row.get("est_ns") is not None)
    tpot_ns = None
    if gpu is not None:
        from ..models import load_gpu
        from ..system import SystemOptions, evaluate
        gpu_spec = load_gpu(gpu)
        selection = "static" if mode in ("dense", "sink_recent") else "dynamic"
        meta = {"model": model_spec.name, "gpu": gpu_spec.name, "batch": batch, "stacks": 1,
                "analysis_batch": batch, "selection": selection, "kv_format": fmt.name,
                "overlap": "serial", "kv_consume": "stream"}
        tpot_ns = evaluate(layer_rows, model_spec, gpu_spec, SystemOptions(), meta).steps[0]["tpot_hbf_est_ns"]
    return {"layers": layer_rows, "steps": step_rows, "read_ns": read_ns, "tpot_ns": tpot_ns}


def calibration_batches(model, kv_format, context, rounds=(32, 128), limit=256, kv_spec=None):
    """Batches whose dense layer-major runs read about ``rounds`` rounds per layer (the F1 fit range)."""
    from ..models import load_model
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
    """Fit the Clock directly against the rounds backend, cached by a hash of the inputs.

    Two dense layer-major points (``batches``; default about 32 and 128 rounds per layer, see
    ``calibration_batches``) refit layer_ns and round_ns for the actual device flags (least squares
    of est_ns over rounds per layer). One snapshot of the policy (``--alloc``,
    M2) at ``snapshot_batch`` (default the larger batch) in the run's mode gives kappa = measured rounds over the Clock's
    rounds at kappa 1 (ceil(pages / 1,024), or the busiest partition's), the demand ratios of
    the mode against dense at the same batch, and the per-step hot factor (max block reads over
    ppb x read share). Pages per block do not change rounds or est, so they are fixed at 256.
    Returns the fields for ``Clock`` / ``ReadTimeClock`` plus diagnostics. Safe for concurrent
    processes sharing ``directory`` (an exclusive lock per digest).
    """
    batches = tuple(batches or calibration_batches(model, kv_format, context, kv_spec=kv_spec))
    inputs = {"model": model, "gpu": gpu, "format": kv_format, "mode": mode, "generation": generation or {},
              "context": context, "batches": list(batches), "device": list(device), "policy": policy,
              "partition": partition, "snapshot_batch": snapshot_batch, "version": 5}
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
    """The rounds-backend runs and fits of ``calibrate`` (called with the cache lock held)."""
    from ..models import load_model

    points, dense = [], {}
    for b in batches:
        dense[b] = phase_a(root, model, gpu, kv_format, b, context, device=device, kv_spec=kv_spec)
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
        snapshot = phase_a(root, model, gpu, kv_format, b, context, mode=mode, generation=generation,
                           device=device, alloc=snap_policy, partition=snap_part, kv_spec=kv_spec)
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
    result = {
        "layer_ns": layer_ns, "round_ns": round_ns, "balance": float(np.mean(kappas)) if kappas else 1.0,
        "ratios": ratios, "read_share": share, "hot_step": max_reads / (256 * share) if share else 1.0,
        "max_block_reads": max_reads,
        "batches": list(batches),
        "phase_a_read_ns": {f"b{key}": value["read_ns"] for key, value in dense.items()},
        "phase_a_tpot_ns": {f"b{key}": value["tpot_ns"] for key, value in dense.items()
                            if value["tpot_ns"] is not None},
        "snapshot": {"policy": snap_policy, "partition": snap_part, "batch": b,
                     "read_ns": snapshot["read_ns"], "tpot_ns": snapshot["tpot_ns"]},
        "inputs": inputs, "points": len(points),
    }
    return result
