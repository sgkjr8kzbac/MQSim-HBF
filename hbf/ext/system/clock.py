"""GPU TPOT step clock registered by ``hbf.ext.system`` into ``registry.clocks['tpot']``."""
import math

import numpy as np

from ...lifetime.clock import StepClock, StepTime
from ...lifetime.geometry import F1
from ...lifetime.interference import hbf_gaps
from .evaluate import SystemOptions, evaluate


class Clock(StepClock):
    """Simulated decode step time of a running set (per stack), combining HBF reads with GPU TPOT.

    est per layer = layer_ns + round_ns x rounds, rounds = ceil(kappa x pages / banks) (or the
    busiest partition), fed as layer rows into ``evaluate`` with compute batch b x stacks and
    analysis batch b. ``ratios`` scale the dense demand for a sparse mode, from ``calibrate``.
    ``rounds_fn(b, ctx)``, when given, returns exact rounds for b equal contexts.
    """

    def __init__(self, model, gpu, geometry, options=None, *, read_ns=4096, banks=1024, stacks=1,
                 layer_ns=F1["layer_ns"], round_ns=F1["round_ns"], balance=1.0, ratios=None, selection="static",
                 kv_format="bf16", rounds_fn=None):
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
