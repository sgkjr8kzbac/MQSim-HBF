"""Request-level KV placement (phase B, M2; phase 3, T-K7): where each request's pages live, and in
what order they were written.

Phase A emulates a batch layer-major: request r's head h is KV head ``r * H + h`` of one large
layout, so every NAND block holds pages of every request. Real serving admits and retires
requests one at a time and appends generated tokens; on HBF a NAND block must be written in page
order from page 0, and writing page 0 erases it (OCP HBF v0.7.0 section 5.4.1 items 4, 5 and 9;
overlap and order errors, section 11.5.2). This module builds a *snapshot*: B requests with their
prompts and the tokens generated so far, each request admitted so that all of them end at the
snapshot step, and places them by one of these allocation policies:

* ``layer_major`` - phase A's emulation: one ``LAYOUTS[intra]`` region per (layer, request, head,
  tensor), all requests reserved at the largest context. Its history writes each request's own
  pages, not the layout's padding (phase A's ``--prefill`` writes the whole span in address order).
* ``dedicated`` - each request takes whole segments (``partition``: superblocks = one block index
  on every bank, or channel blocks / die blocks) and writes its prompt span, then its generated
  pages, in its own segments.
* ``packed`` - one write frontier per partition: every prompt (aligned to a row of the
  partition) and every decode flush takes the next pages.

With ``align="stagger"`` (default) the k-th prompt of a partition starts k * banks / n pages into
its row (n requests per partition), so that the same layer of different requests starts on
different banks; ``row`` starts every prompt at bank 0 of a row.
* ``vllm_pool`` - vLLM-style: per-layer pools of fixed-size blocks of ``block_tokens`` tokens,
  allocated from a free queue (``fresh`` ascending or ``shuffled:<seed>``); one page per head
  and tensor (HND).

Phase 3 (T-K7) adds:

* ``kv_spec`` - grouped placement (per-layer tensor presence, per-request head ranges, cross-layer
  aliases) and sliding HBF windows (``lo < 0``) whose aged-out pages die during decode and are
  counted as ``dead_pages`` and ``stranded_pages`` (dead pages in segments that still hold live
  pages, which cannot be erased without GC, OCP section 11.4).
* ``frontiers="class"`` - separate write frontiers and segments for short-lived tensors (sliding
  windows with ``lo < 0``), long-lived tensors and shared prefixes in ``packed`` / ``dedicated``.
* ``prefix_tokens`` / ``prefix_groups`` / ``prefix_reads`` - shared prefix pages written once per
  prefix group and aliased across sharing requests; ``coalesced`` reads each shared page once per
  step (via ``pages.layer_pages`` deduplication), while ``per_request`` maps extra sharing requests
  to virtual copies on the same banks.

Prompts are written whole at admission, padding of the intra layout included; a partial last page
stays in HBM (staging) and is written when it fills, like every generated page, at the end of the
request's write stream. Snapshot reads cover each request's live full pages; the staged tail (under
one page per head and tensor) is left out.
"""
from dataclasses import dataclass, field, replace
from random import Random

import numpy as np

from ...backends.rounds.pages import place
from ..trace.synthetic import share
from ..trace.tokens import Access
from .layouts import LAYOUTS, Layout
from .tensors import Shape, default_tensors

ALLOCATIONS = ("layer_major", "dedicated", "packed", "vllm_pool")
PARTITIONS = ("none", "channel", "die")
FLUSH_ORDERS = ("layer", "rotate")
ALIGNS = ("stagger", "row")
FRONTIERS = ("single", "class")
PREFIX_READS = ("coalesced", "per_request")
OCP_BANKS_PER_DIE = 16
OCP_DIES_PER_HOST = 4


@dataclass(frozen=True)
class BatchRequest:
    prompt: int
    generated: int = 0

    @property
    def context(self):
        return self.prompt + self.generated


@dataclass(frozen=True)
class BatchSpec:
    """B requests of one stack and how they are placed (see the module docstring)."""
    requests: tuple
    allocation: str = "layer_major"
    partition: str = "none"
    intra: str = "striped_contiguous"
    flush_order: str = "rotate"
    block_tokens: int | None = None
    free_order: str = "fresh"
    align: str = "stagger"
    frontiers: str = "single"
    prefix_tokens: int = 0
    prefix_groups: int = 1
    prefix_reads: str = "coalesced"
    kv_spec: object | None = None

    def __post_init__(self):
        if not self.requests or any(type(r.prompt) is not int or r.prompt <= 0 or type(r.generated) is not int
                                    or r.generated < 0 for r in self.requests):
            raise ValueError("A batch needs requests with a positive prompt and nonnegative generated tokens")
        if self.allocation not in ALLOCATIONS or self.partition not in PARTITIONS:
            raise ValueError(f"allocation must be one of {ALLOCATIONS} and partition one of {PARTITIONS}")
        if self.flush_order not in FLUSH_ORDERS or self.intra not in LAYOUTS:
            raise ValueError(f"flush_order must be one of {FLUSH_ORDERS} and intra a layout name")
        if self.partition != "none" and self.allocation not in ("dedicated", "packed"):
            raise ValueError("partition applies to the dedicated and packed allocations")
        if self.align not in ALIGNS:
            raise ValueError(f"align must be one of {ALIGNS}")
        if not (self.free_order == "fresh" or self.free_order.startswith("shuffled:")
                and self.free_order[9:].isdigit()):
            raise ValueError("free_order must be fresh or shuffled:<seed>")
        if self.frontiers not in FRONTIERS:
            raise ValueError(f"frontiers must be one of {FRONTIERS}")
        if self.frontiers == "class" and self.allocation not in ("dedicated", "packed"):
            raise ValueError("frontiers='class' applies to the dedicated and packed allocations")
        if type(self.prefix_tokens) is not int or self.prefix_tokens < 0:
            raise ValueError("prefix_tokens must be a nonnegative integer")
        if type(self.prefix_groups) is not int or self.prefix_groups < 1:
            raise ValueError("prefix_groups must be a positive integer")
        if self.prefix_reads not in PREFIX_READS:
            raise ValueError(f"prefix_reads must be one of {PREFIX_READS}")

    @property
    def default(self):
        """Phase A's ``--batch`` emulation: layer-major, all requests at one context, nothing generated."""
        return (self.allocation == "layer_major" and len({r.context for r in self.requests}) == 1
                and all(r.generated == 0 for r in self.requests)
                and self.frontiers == "single" and self.prefix_tokens == 0
                and self.kv_spec is None)


def partition_geometry(cfg, partition):
    """(partitions, banks per partition, map from partition-local unit k and partition p to LPA row
    offset): the address space of one partition is LPAs ``lpa(p, k)`` for k = 0, 1, ...; a row of
    the partition is its banks once each."""
    units = cfg.units
    if partition == "none":
        return 1, units, lambda p, k: k
    if cfg.address_map != "ocp":
        raise ValueError("Channel and die partitions follow the OCP address map (--topology ocp)")
    hosts = cfg.host_channels
    if partition == "channel":
        return hosts, units // hosts, lambda p, k: hosts * k + p
    banks = OCP_BANKS_PER_DIE

    def die_lpa(p, k):  # p = host channel * 4 + die; k counts the die's banks row by row
        host, die = divmod(p, OCP_DIES_PER_HOST)
        local = (k // banks) * banks * OCP_DIES_PER_HOST + die * banks + k % banks
        return hosts * local + host
    return hosts * OCP_DIES_PER_HOST, banks, die_lpa


def _clamp_window(window, context):
    lo, hi = window or (None, None)
    lo = 0 if lo is None else max(0, lo + context) if lo < 0 else min(lo, context)
    hi = context if hi is None else max(0, hi + context) if hi < 0 else min(hi, context)
    return lo, hi


def _window_at(own, group, context):
    a, b = _clamp_window(own, context)
    if group is None:
        return a, b
    c, d = _clamp_window(group, context)
    return max(a, c), min(b, d)


def _is_short_lived(own, group=None):
    """True if a tensor's window slides forward with decode (negative lower bound ``lo < 0``)."""
    lo1 = (own or (None, None))[0]
    lo2 = (group or (None, None))[0]
    return (lo1 is not None and lo1 < 0) or (lo2 is not None and lo2 < 0)


def _compose_window(own, group):
    """Best-effort single (lo, hi) tuple combining tensor and group windows for metadata."""
    if group is None:
        return own
    if own is None:
        return group
    lo1, hi1 = own
    lo2, hi2 = group
    if lo1 is None:
        lo = lo2
    elif lo2 is None:
        lo = lo1
    elif lo1 < 0 and lo2 < 0:
        lo = max(lo1, lo2)
    elif lo1 >= 0 and lo2 >= 0:
        lo = max(lo1, lo2)
    else:
        lo = lo1 if lo1 < 0 else lo2
    if hi1 is None:
        hi = hi2
    elif hi2 is None:
        hi = hi1
    elif hi1 < 0 and hi2 < 0:
        hi = min(hi1, hi2)
    elif hi1 >= 0 and hi2 >= 0:
        hi = min(hi1, hi2)
    else:
        hi = hi1 if hi1 < 0 else hi2
    return (lo, hi)


def _peak_live_owners(alias):
    """Peak number of simultaneously live aliased owners during layer-by-layer execution."""
    spans = {}
    for consumer, owner in alias:
        lo, hi = spans.get(owner, (owner, owner))
        spans[owner] = (min(lo, consumer), max(hi, consumer))
    return max((sum(lo <= point <= hi for lo, hi in spans.values()) for point, _ in spans.values()), default=0)


def _striped_gap(units, counts):
    """Inter-region gap for striped placement of one layer without bank collisions."""
    if sum(counts) <= units:
        return 0
    base = max(units // len(counts), 1)
    target = min(len(counts), units)

    def distinct(gap):
        pos, seen = 0, set()
        for n, count in enumerate(counts):
            pos += gap if n else 0
            seen.add(pos % units)
            pos += count
        return len(seen)

    if distinct(base) == target:
        return base
    for candidate in (0, base + 1, max(base - 1, 1)):
        if distinct(candidate) == target:
            return candidate
    return max((0, base, base + 1), key=distinct)


@dataclass(frozen=True)
class _TensorPlacement:
    """Placement metadata for one flat HBF tensor."""
    spec: object
    read_layers: tuple
    stored_layers: tuple
    heads: tuple
    alias: dict  # consumer layer -> owner layer
    own_window: tuple | None
    group_window: tuple | None
    short_lived: bool


def _compile_tensors(shape, tensors, kv_spec):
    """Compile uniform tensors or a ``KVSpec`` into flat tensors and per-tensor placement metadata."""
    base_tensors = tuple(tensors) if tensors is not None else default_tensors(shape)
    if kv_spec is None:
        every = tuple(range(shape.layers))
        all_heads = tuple(range(shape.kv_heads))
        hbf = tuple(s for s in base_tensors if s.tier == "hbf")
        hbm = tuple(s for s in base_tensors if s.tier == "hbm")
        hbf_meta = tuple(_TensorPlacement(s, every, every, all_heads, {}, s.window, None,
                                          _is_short_lived(s.window)) for s in hbf)
        hbm_meta = {s.name: (every, every, all_heads, s.window, None) for s in hbm}
        return base_tensors, hbf, hbm, hbf_meta, hbm_meta, {}

    flat, hbf_meta, hbm_meta, copy_owners = [], [], {}, {}
    for group in kv_spec.groups:
        if max(group.layers) >= shape.layers:
            raise ValueError(f"Group {group.name} names layer {max(group.layers)} of a {shape.layers}-layer model")
        h_lo, h_hi = group.heads or (0, shape.kv_heads)
        if not 0 <= h_lo < h_hi <= shape.kv_heads:
            raise ValueError(f"Group {group.name}: heads [{h_lo}, {h_hi}) outside the {shape.kv_heads} heads per request")
        heads = tuple(range(h_lo, h_hi))
        owner = dict(group.alias)
        stored = tuple(l for l in group.layers if l not in owner)
        group_window = (0, group.budget) if group.budget is not None else group.window
        for base_spec in group.tensors or base_tensors:
            tier = group.tier or base_spec.tier
            composed = _compose_window(base_spec.window, group_window)
            spec_ = replace(base_spec, name=f"{group.name}_{base_spec.name}", window=composed, tier=tier)
            if tier == "hbm":
                flat.append(spec_)
                hbm_meta[spec_.name] = (group.layers, stored, heads, base_spec.window, group_window)
                continue
            short = _is_short_lived(base_spec.window, group_window)
            if group.reuse == "hbf" or not owner:
                flat.append(spec_)
                hbf_meta.append(_TensorPlacement(spec_, group.layers, stored, heads, owner,
                                                 base_spec.window, group_window, short))
            else:
                flat.append(spec_)
                hbf_meta.append(_TensorPlacement(spec_, stored, stored, heads, {},
                                                 base_spec.window, group_window, short))
                copy = replace(spec_, name=f"{spec_.name}_copy", tier="hbm")
                flat.append(copy)
                hbm_meta[copy.name] = (tuple(owner), (), heads, base_spec.window, group_window)
                copy_owners[copy.name] = _peak_live_owners(group.alias)
    names = [s.name for s in flat]
    if not flat or len(set(names)) != len(names):
        raise ValueError("A KV spec needs uniquely named tensors")
    all_tensors = tuple(flat)
    hbf = tuple(m.spec for m in hbf_meta)
    hbm = tuple(s for s in all_tensors if s.tier == "hbm")
    return all_tensors, hbf, hbm, tuple(hbf_meta), hbm_meta, copy_owners


def segment_ids(cfg, partition, lpas):
    """Segment index of each LPA for ``partition`` (superblock, channel block or die block)."""
    lpas = np.asarray(lpas, dtype=np.int64)
    if not len(lpas):
        return np.zeros(0, dtype=np.int64)
    placed = place(cfg, lpas)
    blk = placed.offset // cfg.pages_per_block
    if partition == "none":
        return blk.astype(np.int64)
    if partition == "channel":
        return (blk * cfg.host_channels + placed.host).astype(np.int64)
    return (blk * (cfg.host_channels * OCP_DIES_PER_HOST) + placed.die).astype(np.int64)


def stranded_stats(cfg, live_lpas, dead_lpas, partition="none"):
    """Count (dead_pages, stranded_pages, reclaimable_segments) for a set of live and dead LPAs.

    A dead page is *stranded* when its segment still holds at least one live page (so the segment
    cannot be erased without GC, OCP section 11.4). Segments with dead pages and zero live pages
    are reclaimable.
    """
    dead = np.unique(np.asarray(dead_lpas, dtype=np.int64))
    live = np.unique(np.asarray(live_lpas, dtype=np.int64))
    if len(live):
        dead = np.setdiff1d(dead, live, assume_unique=True)
    if not len(dead):
        return 0, 0, 0
    dead_segs = segment_ids(cfg, partition, dead)
    if len(live):
        live_segs = np.unique(segment_ids(cfg, partition, live))
        stranded_mask = np.isin(dead_segs, live_segs)
    else:
        stranded_mask = np.zeros(len(dead), dtype=bool)
    stranded = int(np.count_nonzero(stranded_mask))
    reclaimable = int(len(np.unique(dead_segs[~stranded_mask])))
    return int(len(dead)), stranded, reclaimable


@dataclass
class Snapshot:
    """Page table and write history of a placed batch.

    ``table[r, layer, h, t, block]`` is the LPA of request r's live page (-1: none or dead);
    ``history`` lists the LPAs in write order (prefill, padding and decode writes) and ``kinds``
    how many of each. Sliding HBF windows record aged-out LPAs in ``dead`` and report
    ``dead_pages``, ``stranded_pages`` and ``reclaimable_segments``.
    """
    table: np.ndarray
    full_pages: dict  # (request, tensor) -> live full pages per (layer, head) at the snapshot
    history: list = field(default_factory=list)
    kinds: dict = field(default_factory=lambda: {"prefill": 0, "padding": 0, "decode": 0})
    block_ranges: dict = field(default_factory=dict)  # (request, tensor) -> (lo_block, hi_block) live at snapshot
    dead: list = field(default_factory=list)
    dead_pages: int = 0
    stranded_pages: int = 0
    reclaimable_segments: int = 0
    prefix_pages: int = 0
    prefix_blocks: np.ndarray | None = None
    prefix_ranks: np.ndarray | None = None
    partition: str = "none"

    def write(self, lpas, kind):
        lpas = np.asarray(lpas, dtype=np.int64)
        self.history.append(lpas)
        self.kinds[kind] += len(lpas)

    def written(self):
        return np.concatenate(self.history) if self.history else np.zeros(0, dtype=np.int64)

    def kill(self, lpas):
        lpas = np.asarray(lpas, dtype=np.int64)
        if len(lpas):
            self.dead.append(lpas)

    def dead_lpas(self):
        return np.concatenate(self.dead) if self.dead else np.zeros(0, dtype=np.int64)

    def live_lpas(self):
        live = self.table[self.table >= 0]
        return np.unique(live.astype(np.int64)) if live.size else np.zeros(0, dtype=np.int64)


class PageTableLayout(Layout):
    """A ``Layout`` whose LPAs come from a placed batch: KV head ``r * H + h`` is request r's head h."""

    def __init__(self, cfg, shape, spec, tensors=None):
        """``shape`` is one request's (layers, KV heads, head_dim); its context is ignored."""
        if spec.default:
            raise ValueError("Use the layout directly for phase A's layer-major batch")
        self.spec, self.heads = spec, shape.kv_heads
        self._per_request_shape = shape
        contexts = [r.context for r in spec.requests]
        max_ctx = max(contexts)
        all_tensors, hbf_tensors, hbm_tensors, hbf_meta, hbm_meta, copy_owners = _compile_tensors(
            shape, tensors, spec.kv_spec)
        if not hbf_tensors:
            raise ValueError("At least one tensor must be stored on HBF")
        names = [s.name for s in all_tensors]
        if len(set(names)) != len(names):
            raise ValueError("Tensor names must be unique and nonempty")
        if any(s.density != 1 for s in all_tensors):
            raise ValueError("PageTableLayout does not yet support tensor density < 1")
        self.base = 0
        self.all_tensors = all_tensors
        self.tensors = hbf_tensors
        self.hbm_tensors = hbm_tensors
        self.specs = {s.name: s for s in all_tensors}
        self.index = {s.name: t for t, s in enumerate(hbf_tensors)}
        self.tokens_per_page = tuple(cfg.page_bytes // s.token_bytes for s in hbf_tensors)
        if 0 in self.tokens_per_page:
            raise ValueError("A token row must fit in one page")
        self.shape = Shape(shape.layers, shape.kv_heads * len(spec.requests), shape.head_dim, max_ctx)
        self.windows = {s.name: (0, max_ctx) for s in all_tensors}
        self.window_starts = (0,) * len(hbf_tensors)
        self._hbf_meta = hbf_meta
        self._hbm_meta = hbm_meta
        self._copy_owners = copy_owners
        self._present = {}
        for m in hbf_meta:
            hset = frozenset(m.heads)
            for layer in m.read_layers:
                self._present[layer, m.spec.name] = hset
        for name, (read_layers, _, heads, _, _) in hbm_meta.items():
            hset = frozenset(heads)
            for layer in read_layers:
                self._present[layer, name] = hset

        self.snapshot = place_batch(cfg, shape, spec, self.tensors, self.tokens_per_page, meta=hbf_meta)
        table = self.snapshot.table
        self.block_counts = tuple(int(table.shape[4]) for _ in hbf_tensors)
        self.pages = len(self.snapshot.live_lpas())
        real_span = int(table.max()) + 1 if table.size and table.max() >= 0 else 0
        written = self.snapshot.written()
        if len(written):
            real_span = max(real_span, int(written.max()) + 1)
        self._real_span = real_span
        self._prefix_stride = 0
        self.span = real_span
        if spec.prefix_reads == "per_request" and spec.prefix_tokens > 0 and self.snapshot.prefix_pages > 0:
            max_rank = int(self.snapshot.prefix_ranks.max()) if self.snapshot.prefix_ranks is not None else 0
            if max_rank > 0:
                self._prefix_stride = -(-real_span // cfg.units) * cfg.units
                self.span = real_span + max_rank * self._prefix_stride
        if self.span > 0:
            cfg.lba(self.base + self.span - 1)
        self._keys = None

    def reads(self, layer, name, head):
        if self.spec.kv_spec is None:
            return True
        return (head % self.heads) in self._present.get((layer, name), ())

    def request_window(self, r, t):
        """Tokens [lo, hi) of HBF tensor t that request r holds in live full pages."""
        lo_b, hi_b = self.snapshot.block_ranges.get((r, t), (0, self.snapshot.full_pages[r, t]))
        tpp = self.tokens(t)
        return lo_b * tpp, hi_b * tpp

    def hbm_window(self, r, name):
        """Tokens [lo, hi) of HBM tensor ``name`` for request r."""
        context = self.spec.requests[r].context
        if name in self._hbm_meta:
            _, _, _, own, group = self._hbm_meta[name]
            return _window_at(own, group, context)
        return _clamp_window(self.specs[name].window, context)

    def request_bytes(self, cfg):
        """Stored KV per request (all tiers; HBF pages deduplicated across aliases/prefixes) and HBM footprint."""
        count = len(self.spec.requests)
        hbf = len(self.snapshot.live_lpas()) * cfg.page_bytes
        hbm_stored = hbm_copies = 0
        for s in self.hbm_tensors:
            _, stored_layers, heads, own, group = self._hbm_meta[s.name]
            for request in self.spec.requests:
                lo, hi = _window_at(own, group, request.context)
                width = max(0, hi - lo)
                size = len(heads) * width * s.token_bytes
                if s.name in self._copy_owners:
                    hbm_copies += self._copy_owners[s.name] * size
                else:
                    hbm_stored += len(stored_layers) * size
        return {"kv_bytes_per_request": round((hbf + hbm_stored) / count),
                "hbm_tier_bytes_per_request": round((hbm_stored + hbm_copies) / count)}

    def lpa(self, layer, head, t, block):
        return int(self.lpa_array(layer, np.array([head]), np.array([t]), np.array([block]))[0])

    def lpa_array(self, layer, head, t, block):
        r, h = np.divmod(np.asarray(head, dtype=np.int64), self.heads)
        t_arr = np.asarray(t, dtype=np.int64)
        block = np.asarray(block, dtype=np.int64)
        table = self.snapshot.table
        inside = (block >= 0) & (block < table.shape[4])
        lpa = np.full(len(block), -1, dtype=np.int64)
        lpa[inside] = table[r[inside], layer, h[inside], t_arr[inside], block[inside]]
        if (lpa < 0).any():
            bad = int(np.flatnonzero(lpa < 0)[0])
            raise ValueError(f"Request {int(r[bad])} has no page {int(block[bad])} of tensor {int(t_arr[bad])} "
                             f"in layer {layer}")
        if self._prefix_stride:
            ranks = self.snapshot.prefix_ranks[r]
            is_pref = (ranks > 0) & (block < self.snapshot.prefix_blocks[r, t_arr])
            if is_pref.any():
                lpa = np.where(is_pref, lpa + ranks * self._prefix_stride, lpa)
        return self.base + lpa

    def decode(self, lpa):
        if self._keys is None:
            table = self.snapshot.table
            index = np.argwhere(table >= 0)
            self._keys = {}
            for coords, val in zip(map(tuple, index.tolist()), table[table >= 0].tolist()):
                self._keys.setdefault(val, coords)
        offset = lpa - self.base
        if self._prefix_stride and offset >= self._real_span:
            rank, phys = divmod(offset, self._prefix_stride)
            owner_r, layer, h, t, block = self._keys[phys]
            group = owner_r % self.spec.prefix_groups
            members = [idx for idx in range(len(self.spec.requests)) if idx % self.spec.prefix_groups == group]
            r = members[rank]
            return layer, r * self.heads + h, t, block
        r, layer, h, t, block = self._keys[offset]
        return layer, r * self.heads + h, t, block


def block_range(own_window, group_window, context, tokens_per_page):
    """Live full block range ``(lo_block, hi_block)`` of an HBF tensor at ``context``."""
    lo, hi = _window_at(own_window, group_window, context)
    if lo >= hi:
        return 0, 0
    lo_b, hi_b = lo // tokens_per_page, hi // tokens_per_page
    if lo_b >= hi_b:
        return hi_b, hi_b
    return lo_b, hi_b


def full_pages(spec_t, context, tokens_per_page, group_window=None):
    """Live full pages of an HBF tensor when the request holds ``context`` tokens (0 if empty)."""
    lo_b, hi_b = block_range(spec_t.window, group_window, context, tokens_per_page)
    return hi_b - lo_b


def fill_context(spec_t, tokens, group_window=None):
    """Smallest context at which the tensor's window upper end reaches ``tokens``."""
    req = tokens
    for win in (spec_t.window, group_window):
        if win is None:
            continue
        hi = win[1]
        if hi is None:
            continue
        if hi < 0:
            req = max(req, tokens - hi)
        elif tokens > hi:
            return float("inf")
    return req


def death_context(spec_t, tokens, group_window=None):
    """Smallest context at which a sliding tensor's window lower end reaches ``tokens`` (inf if non-sliding)."""
    best = float("inf")
    for win in (spec_t.window, group_window):
        if win is None:
            continue
        lo = win[0]
        if lo is not None and lo < 0:
            best = min(best, tokens - lo)
    return best


def _as_range(value):
    return (0, value) if type(value) is int else value


def flush_events(spec, layers, heads, tokens_per_page, full_prompt, full_final, tensors, *, meta=None):
    """Decode flushes of every request: (time, request, tensor, block, layer order).

    Request r is admitted at step a_r = max generated - generated_r. Page k of tensor t fills when
    the tensor's window upper end reaches (k + 1) * tokens_per_page tokens (for a KIVI body
    ``[0, -R]`` that is R tokens later than the context); all stored (layer, head) of that tensor
    flush together, earlier tensors before later ones when both fill at the same step. ``rotate``
    starts the g-th flush of a tensor at stored layer g mod L_t, ``layer`` at its first stored layer.
    """
    longest = max(r.generated for r in spec.requests)
    events = []
    for r, request in enumerate(spec.requests):
        admitted = longest - request.generated
        pending = []
        for t in range(len(tokens_per_page)):
            p_lo, p_hi = _as_range(full_prompt[r, t])
            f_lo, f_hi = _as_range(full_final[r, t])
            gwin = meta[t].group_window if meta is not None else None
            stored = meta[t].stored_layers if meta is not None else tuple(range(layers))
            if not stored or f_hi <= p_hi:
                continue
            for k in range(p_hi, f_hi):
                when = fill_context(tensors[t], (k + 1) * tokens_per_page[t], gwin) - request.prompt
                if 0 < when <= request.generated:
                    pending.append((when, t, k, p_hi, stored))
        pending.sort(key=lambda item: (item[0], item[1], item[2]))
        for when, t, k, p_hi, stored in pending:
            first = (k - p_hi) % len(stored) if spec.flush_order == "rotate" else 0
            events.append((admitted + when, r, t, k, np.roll(np.asarray(stored, dtype=np.int64), -first)))
    return sorted(events, key=lambda event: (event[0], event[1]))


def death_events(spec, tokens_per_page, prompt_ranges, final_ranges, tensors, *, meta=None):
    """Page-death events of sliding HBF windows: (time, request, tensor, block)."""
    longest = max(r.generated for r in spec.requests)
    events = []
    for r, request in enumerate(spec.requests):
        if request.generated == 0:
            continue
        admitted = longest - request.generated
        for t in range(len(tokens_per_page)):
            gwin = meta[t].group_window if meta is not None else None
            if not _is_short_lived(tensors[t].window, gwin):
                continue
            p_lo, _ = _as_range(prompt_ranges[r, t])
            f_lo, f_hi = _as_range(final_ranges[r, t])
            dead_end = min(f_lo, f_hi)
            for k in range(p_lo, dead_end):
                when = death_context(tensors[t], (k + 1) * tokens_per_page[t], gwin) - request.prompt
                if 0 < when <= request.generated:
                    events.append((admitted + when, r, t, k))
    return sorted(events, key=lambda event: (event[0], event[1], event[2], event[3]))


def _flush_keys(event, heads):
    """(layer, head) arrays of one flush in write order: layer by layer, heads within."""
    _, _, _, _, order = event
    head_arr = np.arange(heads, dtype=np.int64) if type(heads) is int else np.asarray(heads, dtype=np.int64)
    return np.repeat(order, len(head_arr)), np.tile(head_arr, len(order))


def place_batch(cfg, shape, spec, tensors, tokens_per_page, *, meta=None):
    """Page table and write history of ``spec`` (one stack)."""
    if meta is None:
        _, hbf, _, meta, _, _ = _compile_tensors(shape, tensors, spec.kv_spec)
        if spec.kv_spec is not None:
            tensors = hbf
            tokens_per_page = tuple(cfg.page_bytes // s.token_bytes for s in tensors)
    count = len(spec.requests)
    prompt_ranges, final_ranges, live_counts = {}, {}, {}
    for r, request in enumerate(spec.requests):
        for t, spec_t in enumerate(tensors):
            gwin = meta[t].group_window
            pr = block_range(meta[t].own_window, gwin, request.prompt, tokens_per_page[t])
            fr = block_range(meta[t].own_window, gwin, request.context, tokens_per_page[t])
            prompt_ranges[r, t] = pr
            final_ranges[r, t] = fr
            live_counts[r, t] = fr[1] - fr[0]
    longest_blocks = max((hi for _, hi in final_ranges.values()), default=0)
    if cfg.capacity_sectors // cfg.spp >= 2**31:
        raise ValueError("The page table stores LPAs as int32")
    table = np.full((count, shape.layers, shape.kv_heads, len(tensors), longest_blocks), -1, dtype=np.int32)
    prefix_blocks = np.zeros((count, len(tensors)), dtype=np.int64)
    prefix_ranks = np.zeros(count, dtype=np.int64)
    group_seen = {}
    for r, request in enumerate(spec.requests):
        g = r % spec.prefix_groups
        prefix_ranks[r] = group_seen.get(g, 0)
        group_seen[g] = prefix_ranks[r] + 1
        if spec.prefix_tokens > 0:
            shared = min(spec.prefix_tokens, request.prompt)
            for t in range(len(tensors)):
                p_lo, p_hi = prompt_ranges[r, t]
                if not meta[t].short_lived and p_lo == 0 and p_hi > 0:
                    prefix_blocks[r, t] = min(p_hi, shared // tokens_per_page[t])
    snapshot = Snapshot(table, live_counts, block_ranges=dict(final_ranges),
                        prefix_blocks=prefix_blocks, prefix_ranks=prefix_ranks,
                        partition=spec.partition)
    longest = max(r.generated for r in spec.requests)
    timeline = [(longest - request.generated, 0, r, None) for r, request in enumerate(spec.requests)]
    timeline += [(event[0], 1, event[1], event) for event in
                 flush_events(spec, shape.layers, shape.kv_heads, tokens_per_page,
                              prompt_ranges, final_ranges, tensors, meta=meta)]
    timeline += [(event[0], 2, event[1], event) for event in
                 death_events(spec, tokens_per_page, prompt_ranges, final_ranges, tensors, meta=meta)]
    timeline.sort(key=lambda item: item[:3])
    placer = {"layer_major": LayerMajor, "vllm_pool": VllmPool}.get(spec.allocation, Segments)
    placer(cfg, shape, spec, tensors, tokens_per_page, snapshot, prompt_ranges, final_ranges, meta).run(timeline)
    dead_lpas = snapshot.dead_lpas()
    if len(dead_lpas):
        dead_n, stranded_n, reclaimable_n = stranded_stats(cfg, snapshot.live_lpas(), dead_lpas, spec.partition)
        snapshot.dead_pages = dead_n
        snapshot.stranded_pages = stranded_n
        snapshot.reclaimable_segments = reclaimable_n
    return snapshot


class Placer:
    def __init__(self, cfg, shape, spec, tensors, tokens_per_page, snapshot, prompt_ranges, final_ranges, meta):
        self.cfg, self.shape, self.spec, self.tensors = cfg, shape, spec, tensors
        self.tokens_per_page, self.snapshot = tokens_per_page, snapshot
        self.prompt_ranges, self.final_ranges, self.meta = prompt_ranges, final_ranges, meta
        self.full_prompt = {key: hi - lo for key, (lo, hi) in prompt_ranges.items()}
        self.full_final = {key: hi - lo for key, (lo, hi) in final_ranges.items()}
        self.prefix_table = {}
        self.group_prefix = {}
        if spec.prefix_tokens > 0:
            for g in range(spec.prefix_groups):
                members = [r for r in range(len(spec.requests)) if r % spec.prefix_groups == g]
                if members:
                    self.group_prefix[g] = [int(snapshot.prefix_blocks[members, t].max()) for t in range(len(tensors))]

    def run(self, timeline):
        for _, kind, r, event in timeline:
            if kind == 0:
                self.admit(r)
            elif kind == 1:
                t = event[2]
                layers, heads = _flush_keys(event, self.meta[t].heads)
                self.flush(r, t, event[3], layers, heads)
            else:
                self.die(r, event[2], event[3])

    def prompt_keys(self, r, *, tensors=None, block_starts=None, block_ends=None):
        """(layer, head, tensor, block) arrays of request r's stored prompt pages."""
        parts = []
        t_list = range(len(self.tensors)) if tensors is None else tensors
        for t in t_list:
            p_lo, p_hi = self.prompt_ranges[r, t]
            lo = p_lo if block_starts is None else block_starts[t]
            hi = p_hi if block_ends is None else block_ends[t]
            if hi > lo and self.meta[t].stored_layers and self.meta[t].heads:
                layer, head, block = np.meshgrid(
                    np.asarray(self.meta[t].stored_layers, dtype=np.int64),
                    np.asarray(self.meta[t].heads, dtype=np.int64),
                    np.arange(lo, hi, dtype=np.int64), indexing="ij")
                parts.append((layer.ravel(), head.ravel(), np.full(layer.size, t, dtype=np.int64), block.ravel()))
        if not parts:
            return (np.zeros(0, dtype=np.int64),) * 4
        return tuple(np.concatenate(column) for column in zip(*parts))

    def _alias_copy(self, r, t, blocks):
        """Copy owner-layer LPAs to aliased consumer layers for tensor ``t`` and block slice/indices."""
        alias = self.meta[t].alias
        if not alias:
            return
        heads = np.asarray(self.meta[t].heads, dtype=np.int64)
        for consumer, owner in alias.items():
            self.snapshot.table[r, consumer, heads, t, blocks] = self.snapshot.table[r, owner, heads, t, blocks]

    def record(self, r, layer, head, t, block, lpa):
        self.snapshot.table[r, layer, head, t, block] = lpa
        if self.spec.kv_spec is not None and len(t):
            for t_val in np.unique(t).tolist():
                alias = self.meta[t_val].alias
                if not alias:
                    continue
                mask = t == t_val
                for consumer, owner in alias.items():
                    omask = mask & (layer == owner)
                    if omask.any():
                        self.snapshot.table[r, consumer, head[omask], t_val, block[omask]] = lpa[omask]

    def _alias_prefix_to_request(self, r, g):
        """Populate request r's page table with its prefix group's shared LPAs."""
        pref = self.prefix_table.get(g)
        if pref is None:
            return
        for t in range(len(self.tensors)):
            n_blk = int(self.snapshot.prefix_blocks[r, t])
            if n_blk <= 0:
                continue
            arr = pref.get(t)
            if arr is None:
                continue
            heads = np.asarray(self.meta[t].heads, dtype=np.int64)
            for layer in self.meta[t].stored_layers:
                self.snapshot.table[r, layer, heads, t, :n_blk] = arr[layer, heads, :n_blk]
            self._alias_copy(r, t, slice(0, n_blk))

    def die(self, r, t, block):
        """Mark block ``block`` of request r's tensor ``t`` dead and clear it from the live page table."""
        m = self.meta[t]
        stored = np.asarray(m.stored_layers, dtype=np.int64)
        heads = np.asarray(m.heads, dtype=np.int64)
        if len(stored) and len(heads):
            lpas = self.snapshot.table[r, np.ix_(stored, heads)[0], np.ix_(stored, heads)[1], t, block].ravel()
            lpas = lpas[lpas >= 0]
            if len(lpas):
                self.snapshot.kill(lpas)
        read = np.asarray(m.read_layers, dtype=np.int64)
        if len(read) and len(heads):
            self.snapshot.table[r, np.ix_(read, heads)[0], np.ix_(read, heads)[1], t, block] = -1


class LayerMajor(Placer):
    """Phase A's regions: request r's head h is KV head r * H + h of one layout at the largest context."""

    def run(self, timeline):
        self._simple = (self.spec.kv_spec is None
                        and not any(m.short_lived for m in self.meta))
        if self._simple:
            self.region = LAYOUTS[self.spec.intra](self.cfg, Shape(
                self.shape.layers, self.shape.kv_heads * len(self.spec.requests), self.shape.head_dim,
                max(r.context for r in self.spec.requests)), tensors=self.tensors)
        else:
            self._build_grouped_starts()
        self._prefix_owner = {}
        if self.spec.prefix_tokens > 0:
            for g in range(self.spec.prefix_groups):
                members = [idx for idx in range(len(self.spec.requests)) if idx % self.spec.prefix_groups == g]
                if members:
                    self._prefix_owner[g] = max(members, key=lambda idx: self.spec.requests[idx].prompt)
        super().run(timeline)

    def _build_grouped_starts(self):
        count = len(self.spec.requests)
        total_heads = self.shape.kv_heads * count
        starts = np.full((self.shape.layers, len(self.tensors), total_heads), -1, dtype=np.int64)
        min_lo = [min((self.prompt_ranges[r, t][0] for r in range(count)), default=0)
                  for t in range(len(self.tensors))]
        max_hi = [max((self.final_ranges[r, t][1] for r in range(count)), default=0)
                  for t in range(len(self.tensors))]
        counts = [max(0, hi - lo) for lo, hi in zip(min_lo, max_hi)]
        cursor = 0
        for layer in range(self.shape.layers):
            entries = [(t, r * self.shape.kv_heads + h)
                       for t, m in enumerate(self.meta) if layer in m.stored_layers and counts[t] > 0
                       for r in range(count) for h in m.heads]
            if not entries:
                continue
            striped = self.spec.intra == "striped_contiguous"
            gap = _striped_gap(self.cfg.units, [counts[t] for t, _ in entries]) if striped else 0
            if striped:
                cursor += -cursor % self.cfg.units
            for n, (t, head) in enumerate(entries):
                start = cursor + (gap if n else 0)
                starts[layer, t, head] = start
                cursor = start + counts[t]
        self._starts = starts
        self._min_lo = np.asarray(min_lo, dtype=np.int64)

    def lpas(self, r, layer, head, t, block):
        if self._simple:
            out = np.empty(len(layer), dtype=np.int64)
            for value in np.unique(layer):
                mask = layer == value
                out[mask] = self.region.lpa_array(int(value), r * self.shape.kv_heads + head[mask], t[mask], block[mask])
            return out
        return self._starts[layer, t, r * self.shape.kv_heads + head] + (block - self._min_lo[t])

    def _write_prefix(self, r, g):
        if g in self.prefix_table or g not in self.group_prefix:
            return
        counts = self.group_prefix[g]
        active = [t for t, n in enumerate(counts) if n > 0]
        if not active:
            self.prefix_table[g] = {}
            return
        starts = {t: 0 for t in active}
        ends = {t: counts[t] for t in active}
        owner_r = self._prefix_owner.get(g, r)
        layer, head, t, block = self.prompt_keys(owner_r, tensors=active, block_starts=starts, block_ends=ends)
        lpa = self.lpas(owner_r, layer, head, t, block)
        pref = {tv: np.full((self.shape.layers, self.shape.kv_heads, counts[tv]), -1, dtype=np.int32) for tv in active}
        for tv in active:
            mask = t == tv
            pref[tv][layer[mask], head[mask], block[mask]] = lpa[mask]
        self.prefix_table[g] = pref
        self.snapshot.prefix_pages += len(lpa)
        self.snapshot.write(np.sort(lpa), "prefill")

    def admit(self, r):
        if self.spec.prefix_tokens > 0:
            g = r % self.spec.prefix_groups
            self._write_prefix(r, g)
            self._alias_prefix_to_request(r, g)
            starts = {t: max(self.prompt_ranges[r, t][0], int(self.snapshot.prefix_blocks[r, t]))
                      for t in range(len(self.tensors))}
            ends = {t: self.prompt_ranges[r, t][1] for t in range(len(self.tensors))}
            layer, head, t, block = self.prompt_keys(r, block_starts=starts, block_ends=ends)
        else:
            layer, head, t, block = self.prompt_keys(r)
        if not len(layer):
            return
        lpa = self.lpas(r, layer, head, t, block)
        self.record(r, layer, head, t, block, lpa)
        self.snapshot.write(np.sort(lpa), "prefill")

    def flush(self, r, t, block, layer, head):
        t_arr, b_arr = np.full(len(layer), t, dtype=np.int64), np.full(len(layer), block, dtype=np.int64)
        lpa = self.lpas(r, layer, head, t_arr, b_arr)
        self.record(r, layer, head, t_arr, b_arr, lpa)
        self.snapshot.write(lpa, "decode")


class _PartitionCfg:
    """View of a Config restricted to one partition's banks for intra-request striping and alignment."""

    def __init__(self, cfg, banks):
        self.page_bytes = cfg.page_bytes
        self.units = banks
        self._cfg = cfg

    def lba(self, pa):
        return self._cfg.lba(pa)


class Segments(Placer):
    """dedicated and packed allocations in the address space of a partition (see the module docstring)."""

    def run(self, timeline):
        self.parts, self.banks, self.to_lpa = partition_geometry(self.cfg, self.spec.partition)
        self.intra_cfg = self.cfg if self.banks == self.cfg.units else _PartitionCfg(self.cfg, self.banks)
        self.segment = self.banks * self.cfg.pages_per_block
        self.load = [0] * self.parts
        self.owner, self.frontier, self.segments, self.used, self.admitted = {}, {}, {}, {}, {}
        self.next_segment = [0] * self.parts
        self.prefix_admitted = {}
        super().run(timeline)

    def _class_of(self, t):
        if self.spec.frontiers == "single":
            return "single"
        return "short" if self.meta[t].short_lived else "long"

    def _frontier_key(self, r, cls):
        if self.spec.allocation == "packed":
            return ("packed", self.owner[r], cls)
        return ("dedicated", r, cls)

    def _alloc(self, key, p, n):
        """Partition-local page indices of the next n pages of frontier ``key`` in partition ``p``."""
        if self.spec.allocation == "packed" and self.spec.frontiers == "single":
            start = self.frontier.get(p, 0)
            self.frontier[p] = start + n
            return np.arange(start, start + n, dtype=np.int64)
        used0 = self.used.get(key, 0)
        segments = self.segments.setdefault(key, [])
        while len(segments) * self.segment < used0 + n:
            segments.append(self.next_segment[p])
            self.next_segment[p] += 1
        used = used0 + np.arange(n, dtype=np.int64)
        self.used[key] = used0 + n
        return np.asarray(segments, dtype=np.int64)[used // self.segment] * self.segment + used % self.segment

    def _align_frontier(self, key, p, offset):
        """Align frontier ``key`` to a row of ``self.banks`` plus ``offset``, writing padding if needed."""
        if self.spec.allocation == "packed" and self.spec.frontiers == "single":
            start = self.frontier.get(p, 0)
            aligned = -(-start // self.banks) * self.banks + offset
            if aligned > start:
                self.snapshot.write(self.to_lpa(p, np.arange(start, aligned, dtype=np.int64)), "padding")
            self.frontier[p] = aligned
            return
        start = self.used.get(key, 0)
        aligned = -(-start // self.banks) * self.banks + offset
        pad = aligned - start
        if pad > 0:
            self.snapshot.write(self.to_lpa(p, self._alloc(key, p, pad)), "padding")

    def _intra_slots(self, active, block_starts, block_ends, layer, head, t, block):
        """Map (layer, head, t, block) to intra-span slot indices and total span."""
        if self.spec.kv_spec is None:
            intra_tensors = tuple(
                replace(self.tensors[tv], window=(0, (block_ends[tv] - block_starts[tv]) * self.tokens_per_page[tv]))
                for tv in active)
            intra_context = max((block_ends[tv] - block_starts[tv]) * self.tokens_per_page[tv] for tv in active)
            intra = LAYOUTS[self.spec.intra](self.intra_cfg, Shape(
                self.shape.layers, self.shape.kv_heads, self.shape.head_dim, intra_context), tensors=intra_tensors)
            t_map = np.empty(len(self.tensors), dtype=np.int64)
            b_off = np.zeros(len(self.tensors), dtype=np.int64)
            for i, orig in enumerate(active):
                t_map[orig] = i
                b_off[orig] = block_starts[orig]
            slots = np.empty(len(layer), dtype=np.int64)
            for value in np.unique(layer):
                mask = layer == value
                slots[mask] = intra.lpa_array(int(value), head[mask], t_map[t[mask]], block[mask] - b_off[t[mask]])
            return intra.span, slots

        counts = {tv: block_ends[tv] - block_starts[tv] for tv in active}
        starts_tbl = np.full((self.shape.layers, len(self.tensors), self.shape.kv_heads), -1, dtype=np.int64)
        cursor = 0
        striped = self.spec.intra == "striped_contiguous"
        for lyr in range(self.shape.layers):
            entries = [(tv, h) for tv in active if lyr in self.meta[tv].stored_layers for h in self.meta[tv].heads]
            if not entries:
                continue
            gap = _striped_gap(self.banks, [counts[tv] for tv, _ in entries]) if striped else 0
            if striped:
                cursor += -cursor % self.banks
            for n, (tv, h) in enumerate(entries):
                start = cursor + (gap if n else 0)
                starts_tbl[lyr, tv, h] = start
                cursor = start + counts[tv]
        b_off = np.zeros(len(self.tensors), dtype=np.int64)
        for tv in active:
            b_off[tv] = block_starts[tv]
        slots = starts_tbl[layer, t, head] + (block - b_off[t])
        return cursor, slots

    def _place_span(self, key, p, active, block_starts, block_ends, offset, r_for_keys):
        """Align frontier ``key``, place ``active`` tensors' prompt blocks, and return keys + mapped LPAs."""
        layer, head, t, block = self.prompt_keys(r_for_keys, tensors=active,
                                                 block_starts=block_starts, block_ends=block_ends)
        if not len(layer):
            return layer, head, t, block, np.zeros(0, dtype=np.int64)
        span, slots = self._intra_slots(active, block_starts, block_ends, layer, head, t, block)
        self._align_frontier(key, p, offset)
        self.load[p] += span
        lpa = self.to_lpa(p, self._alloc(key, p, span))
        filled = np.zeros(span, dtype=bool)
        filled[slots] = True
        self.snapshot.kinds["prefill"] += int(filled.sum())
        self.snapshot.kinds["padding"] += int((~filled).sum())
        self.snapshot.history.append(lpa)
        return layer, head, t, block, lpa[slots]

    def _write_prefix(self, r, g):
        if g in self.prefix_table or g not in self.group_prefix:
            return
        counts = self.group_prefix[g]
        active = [t for t, n in enumerate(counts) if n > 0]
        if not active:
            self.prefix_table[g] = {}
            return
        p = min(range(self.parts), key=lambda q: (self.load[q], q))
        idx = self.prefix_admitted.get(p, 0)
        self.prefix_admitted[p] = idx + 1
        share_g = -(-self.spec.prefix_groups // self.parts)
        offset = idx * max(self.banks // share_g, 1) % self.banks if self.spec.align == "stagger" else 0
        cls = "prefix" if self.spec.frontiers == "class" else "single"
        key = ("packed", p, cls) if self.spec.allocation == "packed" else ("prefix", g)
        starts = {t: 0 for t in active}
        ends = {t: counts[t] for t in active}
        layer, head, t, block, mapped = self._place_span(key, p, active, starts, ends, offset, r)
        pref = {tv: np.full((self.shape.layers, self.shape.kv_heads, counts[tv]), -1, dtype=np.int32) for tv in active}
        for tv in active:
            mask = t == tv
            pref[tv][layer[mask], head[mask], block[mask]] = mapped[mask]
        self.prefix_table[g] = pref
        self.snapshot.prefix_pages += len(mapped)

    def admit(self, r):
        if self.spec.prefix_tokens > 0:
            g = r % self.spec.prefix_groups
            self._write_prefix(r, g)
            self._alias_prefix_to_request(r, g)
        p = min(range(self.parts), key=lambda q: (self.load[q], q))
        self.owner[r] = p
        starts = {t: max(self.prompt_ranges[r, t][0], int(self.snapshot.prefix_blocks[r, t]))
                  for t in range(len(self.tensors))}
        ends = {t: self.prompt_ranges[r, t][1] for t in range(len(self.tensors))}
        active_all = [t for t in range(len(self.tensors)) if ends[t] > starts[t]]
        if not active_all:
            return
        index = self.admitted.get(p, 0)
        self.admitted[p] = index + 1
        share_r = -(-len(self.spec.requests) // self.parts)
        offset = index * max(self.banks // share_r, 1) % self.banks if self.spec.align == "stagger" else 0
        classes = ("single",) if self.spec.frontiers == "single" else ("long", "short")
        for cls in classes:
            active = [t for t in active_all if self._class_of(t) == cls]
            if not active:
                continue
            key = self._frontier_key(r, cls)
            layer, head, t, block, mapped = self._place_span(key, p, active, starts, ends, offset, r)
            self.record(r, layer, head, t, block, mapped)

    def flush(self, r, t, block, layer, head):
        p = self.owner[r]
        cls = self._class_of(t)
        key = self._frontier_key(r, cls)
        self.load[p] += len(layer)
        lpa = self.to_lpa(p, self._alloc(key, p, len(layer)))
        self.record(r, layer, head, np.full(len(layer), t, dtype=np.int64),
                    np.full(len(layer), block, dtype=np.int64), lpa)
        self.snapshot.write(lpa, "decode")


class VllmPool(Placer):
    """Per-layer pools of vLLM blocks: block id i of layer l holds pages
    base_l + i * (H * sum n_t) + h * sum n_t + off_t + (b mod n_t), n_t = block_tokens / tokens_per_page[t]."""

    def run(self, timeline):
        tpp = self.tokens_per_page
        block_tokens = self.spec.block_tokens or max(tpp)
        if any(block_tokens % tokens for tokens in tpp):
            raise ValueError(f"block_tokens {block_tokens} must be a multiple of every tensor's tokens per page {tpp}")
        self.block_tokens, self.per = block_tokens, [block_tokens // tokens for tokens in tpp]
        self.offsets = np.concatenate(([0], np.cumsum(self.per)[:-1])).astype(np.int64)
        self.stride = sum(self.per)
        tensors = range(len(self.tensors))
        self.pref_vblocks = {}
        if self.spec.prefix_tokens > 0:
            for g, counts in self.group_prefix.items():
                max_pref_tok = max((counts[t] * tpp[t] for t in tensors), default=0)
                self.pref_vblocks[g] = max_pref_tok // block_tokens
        req_blocks = []
        self.req_pref_vblocks = []
        for r, request in enumerate(self.spec.requests):
            need = -(-max((self.final_ranges[r, t][1] * tpp[t] for t in tensors), default=0) // block_tokens)
            if self.spec.prefix_tokens > 0:
                g = r % self.spec.prefix_groups
                r_vb = min(self.pref_vblocks.get(g, 0), min(self.spec.prefix_tokens, request.prompt) // block_tokens)
            else:
                r_vb = 0
            self.req_pref_vblocks.append(r_vb)
            req_blocks.append(max(0, need - min(need, r_vb)))
        self.blocks = req_blocks
        total = sum(self.pref_vblocks.values()) + sum(self.blocks)
        self.pool = -(-total * self.shape.kv_heads * self.stride // self.cfg.units) * self.cfg.units
        self.ids = list(range(total))
        if self.spec.free_order != "fresh":
            Random(int(self.spec.free_order[9:])).shuffle(self.ids)
        self.owned = {}
        self.prefix_ids = {}
        super().run(timeline)

    def _lpas_from_ids(self, ids, layer, head, t, block):
        tpp = np.asarray(self.tokens_per_page, dtype=np.int64)[t]
        per = np.asarray(self.per, dtype=np.int64)[t]
        index = np.asarray(ids, dtype=np.int64)[block * tpp // self.block_tokens]
        return (layer * self.pool + index * self.shape.kv_heads * self.stride + head * self.stride
                + self.offsets[t] + block % per)

    def lpas(self, r, layer, head, t, block):
        return self._lpas_from_ids(self.owned[r], layer, head, t, block)

    def _write_prefix(self, r, g):
        if g in self.prefix_table or g not in self.group_prefix:
            return
        vb = self.pref_vblocks.get(g, 0)
        ids = [self.ids.pop(0) for _ in range(vb)]
        self.prefix_ids[g] = ids
        if not vb:
            self.prefix_table[g] = {}
            return
        counts = [min(self.group_prefix[g][t], vb * self.per[t]) for t in range(len(self.tensors))]
        active = [t for t, n in enumerate(counts) if n > 0]
        if not active:
            self.prefix_table[g] = {}
            return
        starts = {t: 0 for t in active}
        ends = {t: counts[t] for t in active}
        layer, head, t, block = self.prompt_keys(r, tensors=active, block_starts=starts, block_ends=ends)
        lpa = self._lpas_from_ids(ids, layer, head, t, block)
        pref = {tv: np.full((self.shape.layers, self.shape.kv_heads, counts[tv]), -1, dtype=np.int32) for tv in active}
        for tv in active:
            mask = t == tv
            pref[tv][layer[mask], head[mask], block[mask]] = lpa[mask]
        self.prefix_table[g] = pref
        self.snapshot.prefix_pages += len(lpa)
        self.snapshot.write(np.sort(lpa), "prefill")

    def admit(self, r):
        if self.spec.prefix_tokens > 0:
            g = r % self.spec.prefix_groups
            self._write_prefix(r, g)
            r_vb = self.req_pref_vblocks[r]
            for t in range(len(self.tensors)):
                self.snapshot.prefix_blocks[r, t] = min(int(self.snapshot.prefix_blocks[r, t]), r_vb * self.per[t])
            self._alias_prefix_to_request(r, g)
            self.owned[r] = list(self.prefix_ids.get(g, [])[:r_vb]) + [self.ids.pop(0) for _ in range(self.blocks[r])]
            starts = {t: max(self.prompt_ranges[r, t][0], int(self.snapshot.prefix_blocks[r, t]))
                      for t in range(len(self.tensors))}
            ends = {t: self.prompt_ranges[r, t][1] for t in range(len(self.tensors))}
            layer, head, t, block = self.prompt_keys(r, block_starts=starts, block_ends=ends)
        else:
            self.owned[r] = [self.ids.pop(0) for _ in range(self.blocks[r])]
            layer, head, t, block = self.prompt_keys(r)
        if not len(layer):
            return
        lpa = self.lpas(r, layer, head, t, block)
        self.record(r, layer, head, t, block, lpa)
        self.snapshot.write(np.sort(lpa), "prefill")

    def flush(self, r, t, block, layer, head):
        t_arr, b_arr = np.full(len(layer), t, dtype=np.int64), np.full(len(layer), block, dtype=np.int64)
        lpa = self.lpas(r, layer, head, t_arr, b_arr)
        self.record(r, layer, head, t_arr, b_arr, lpa)
        self.snapshot.write(lpa, "decode")


def check_write_order(cfg, lpas, *, dead_lpas=None, live_lpas=None, partition="none", snapshot=None):
    """Count writes that break the per-block write rules of OCP HBF section 5.4.1.

    In each NAND block (MQSim unit, static offset // pages per block) pages must be written in
    order from page 0 without skipping (items 4 and 9). Writing page 0 auto-erases the block
    (item 5): if the block already holds pages, they are lost (``live_erases``; all data of a
    snapshot is live) and a new erase epoch starts. Within an epoch, relative to the highest page
    written so far:

    * ``skips`` - a page above the next one (section 11.5.2.6, sub-code 0x1);
    * ``backward`` - a page below the highest one that was not written in this epoch (also an
      order violation of section 11.5.2.6: the block cannot go back);
    * ``overwrites`` - a page written again in the same epoch (overlap, section 11.5.2.2).

    When ``snapshot`` or ``dead_lpas`` is supplied, also counts ``dead_pages`` and
    ``stranded_pages`` (dead pages in segments that still hold live pages).
    Returns these counts, the writes and blocks, and ``legal`` (all order/erase counts zero).
    """
    if isinstance(lpas, Snapshot):
        snapshot = lpas
        lpas = snapshot.written()
    if snapshot is not None:
        if dead_lpas is None:
            dead_lpas = snapshot.dead_lpas()
        if live_lpas is None:
            live_lpas = snapshot.live_lpas()
        partition = snapshot.partition
    lpas = np.asarray(lpas, dtype=np.int64)
    dead_n = stranded_n = 0
    if dead_lpas is not None and len(dead_lpas):
        if live_lpas is None:
            live_lpas = np.setdiff1d(np.unique(lpas), np.unique(np.asarray(dead_lpas, dtype=np.int64)))
        dead_n, stranded_n, _ = stranded_stats(cfg, live_lpas, dead_lpas, partition)
    empty = {"writes": 0, "blocks": 0, "skips": 0, "backward": 0, "overwrites": 0, "live_erases": 0,
             "dead_pages": dead_n, "stranded_pages": stranded_n, "legal": True}
    if not len(lpas):
        return empty
    placed = place(cfg, lpas)
    block = placed.unit + cfg.units * (placed.offset // cfg.pages_per_block)
    page = (placed.offset % cfg.pages_per_block).astype(np.int64)
    del placed
    order = np.argsort(block, kind="stable")  # time order kept within each block
    block, page = block[order], page[order]
    del order
    first = np.empty(len(block), dtype=bool)
    first[:1] = True
    np.not_equal(block[1:], block[:-1], out=first[1:])
    erase = (page == 0) & ~first
    epoch = np.cumsum(first | (page == 0)) - 1  # a block's writes split at every page-0 write
    del block
    scale = cfg.pages_per_block + 1
    keyed = epoch * scale + page
    before = np.empty_like(keyed)
    before[:1] = -1
    np.maximum.accumulate(keyed[:-1], out=before[1:])
    start = np.empty(len(epoch), dtype=bool)
    start[:1] = True
    np.not_equal(epoch[1:], epoch[:-1], out=start[1:])
    highest = np.where(start, -1, before - epoch * scale)  # highest page earlier in this epoch
    del before
    skips = int(np.count_nonzero(page > highest + 1))
    below = page <= highest
    # A repeated (epoch, page) is an overwrite (it is always below the highest page); the other writes
    # below the highest page go back to a page the block has not had in this epoch.
    overwrites = len(keyed) - len(np.unique(keyed))
    backward = int(np.count_nonzero(below)) - overwrites
    erases = int(np.count_nonzero(erase))
    return {"writes": len(lpas), "blocks": int(np.count_nonzero(first)), "skips": skips, "backward": backward,
            "overwrites": overwrites, "live_erases": erases,
            "dead_pages": dead_n, "stranded_pages": stranded_n,
            "legal": skips == backward == overwrites == erases == 0}


def generate_batch(cfg, layout, mode, *, steps=1, seed=1, fraction=.1, count=4, select_tokens=16):
    """Reads of a placed batch: ``dense``, ``uniform_tokens`` or ``sink_recent`` per request, over each
    request's live full pages (see ``generate``; the draw order is step, layer, request, head)."""
    if mode not in ("dense", "uniform_tokens", "sink_recent"):
        raise ValueError("A placed batch supports the dense, uniform_tokens and sink_recent modes")
    rng = Random(seed)
    heads, spec = layout.heads, layout.spec
    grouped = spec.kv_spec is not None
    always = [spec_t for spec_t in layout.all_tensors if spec_t.role == "meta" or spec_t.tier == "hbm"]
    for step in range(steps):
        for layer in range(layout.shape.layers):
            for r, request in enumerate(spec.requests):
                context = request.context
                windows = {spec_t.name: layout.request_window(r, layout.index[spec_t.name])
                           for spec_t in layout.tensors}
                for h in range(heads):
                    head = r * heads + h
                    if grouped:
                        hbf_specs = [s for s in layout.tensors if s.role != "meta" and layout.reads(layer, s.name, head)]
                        always_specs = [s for s in always if layout.reads(layer, s.name, head)]
                        if not hbf_specs and not always_specs:
                            continue
                    else:
                        hbf_specs = [s for s in layout.tensors if s.role != "meta"]
                        always_specs = always

                    def clipped(spec_t, lo, hi):
                        lo, hi = max(lo, windows[spec_t.name][0]), min(hi, windows[spec_t.name][1])
                        return [Access(step, layer, head, spec_t.name, "R", lo, hi, spec_t.bits)] if lo < hi else []

                    if mode == "dense":
                        for spec_t in hbf_specs:
                            yield from clipped(spec_t, 0, context)
                    elif mode == "uniform_tokens":
                        blocks = -(-context // select_tokens)
                        for block in sorted(rng.sample(range(blocks), share(blocks, fraction))):
                            lo, hi = block * select_tokens, min((block + 1) * select_tokens, context)
                            for spec_t in hbf_specs:
                                yield from clipped(spec_t, lo, hi)
                    else:
                        recent = max(1, round(context * fraction))
                        sink = min(count, context - recent)
                        for spec_t in hbf_specs:
                            if sink > 0:
                                yield from clipped(spec_t, 0, sink)
                            yield from clipped(spec_t, context - recent, context)
                    for spec_t in always_specs:
                        if spec_t.tier == "hbm":
                            lo, hi = layout.hbm_window(r, spec_t.name)
                        else:
                            lo, hi = windows[spec_t.name]
                        if lo < hi:
                            yield Access(step, layer, head, spec_t.name, "R", lo, hi, spec_t.bits)
