"""KV cache geometry, HBF device parameters, and endurance helper formulas."""
import math
from dataclasses import dataclass

import numpy as np

GIB = 2 ** 30
DAY = 86400.0
CAUSES = ("prefill", "decode", "refresh_rd", "refresh_ret", "compaction", "replay", "wave", "padding")
POLICIES = ("packed", "dedicated", "layer_major", "hbm_decode", "vllm_pool")
PARTITIONS = {"none": 1024, "channel": 64, "die": 16}  # banks per partition (spec section 4.6 Figure 4)
F1 = {"layer_ns": 4000.0, "round_ns": 4406.0}  # design-B F1 fit: ocp, HBF command model, tR = 4,096 ns
HBF_FLAGS = ["--topology", "ocp", "--read-plane-ns", "0", "--coalesce-ns", "1", "--cache-read"]


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
        from ..frontend.kv.formats import parse_format
        fmt = parse_format(kv_format or "bf16", model.head_dim, model.q_heads // model.kv_heads)
        return Geometry(model.layers, model.kv_heads, fmt.tensors, page_bytes), fmt
    from ..registry import REGISTRY
    hook = REGISTRY.get_hook("kvspec", required=True)
    return hook.geometry_for(model, kv_format, page_bytes, kv_spec=kv_spec, context=context, local_tier=local_tier)



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


def resolve_prefill_tok_s(model, gpu, tokens, *, prefill_rate=None):
    """Prefill throughput (tok/s): explicit ``prefill_rate`` (float or callable) or registered GPU hook."""
    if prefill_rate is not None:
        rate = prefill_rate(tokens) if callable(prefill_rate) else prefill_rate
        if rate <= 0:
            raise ValueError("prefill_rate must be positive")
        return float(rate)
    from ..registry import REGISTRY
    fn = REGISTRY.get_prefill_rate("gpu_roofline", required=True)
    return fn(model, gpu, tokens)


def threshold_steps(x, ppb, share=1.0, hot=1.0):
    """Decode steps until a block of fully read KV reaches X reads: X / (ppb x share x hot)."""
    return x / (ppb * share * hot)


def erase_amp(pages, segment_pages):
    """Erase amplification of entries that own whole segments: ceil(P / P_seg) P_seg / P."""
    return math.ceil(pages / segment_pages - 1e-9) * segment_pages / pages if pages > 0 else 1.0


def region_pages(model, tensors, slots, prompt, pages_per_block=256, layout="striped_contiguous"):
    """Pages a layer_major wave writes: phase A's layout of H x slots heads at the prompt length, padding included."""
    from ..device.config import load
    from ..frontend.kv.layouts import LAYOUTS
    from ..frontend.kv.tensors import Shape
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

