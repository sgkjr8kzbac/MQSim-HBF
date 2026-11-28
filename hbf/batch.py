"""Request-level KV placement (phase B, M2): where each request's pages live, and in what order
they were written.

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

Prompts are written whole at admission, padding of the intra layout included; a partial last page
stays in HBM (staging) and is written when it fills, like every generated page, at the end of the
request's write stream. Snapshot reads cover each request's full pages; the staged tail (under
one page per head and tensor) is left out.
"""
from dataclasses import dataclass, field
from random import Random

import numpy as np

from .access import LAYOUTS, Access, Layout, Shape, share
from .pages import place

ALLOCATIONS = ("layer_major", "dedicated", "packed", "vllm_pool")
PARTITIONS = ("none", "channel", "die")
FLUSH_ORDERS = ("layer", "rotate")
ALIGNS = ("stagger", "row")
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

    @property
    def default(self):
        """Phase A's ``--batch`` emulation: layer-major, all requests at one context, nothing generated."""
        return (self.allocation == "layer_major" and len({r.context for r in self.requests}) == 1
                and all(r.generated == 0 for r in self.requests))


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


@dataclass
class Snapshot:
    """Page table and write history of a placed batch.

    ``table[r, layer, h, t, block]`` is the LPA of request r's page (-1: none); ``history`` lists
    the LPAs in write order (prefill, padding and decode writes) and ``kinds`` how many of each.
    """
    table: np.ndarray
    full_pages: dict  # (request, tensor) -> full pages per (layer, head) at the snapshot
    history: list = field(default_factory=list)
    kinds: dict = field(default_factory=lambda: {"prefill": 0, "padding": 0, "decode": 0})

    def write(self, lpas, kind):
        lpas = np.asarray(lpas, dtype=np.int64)
        self.history.append(lpas)
        self.kinds[kind] += len(lpas)

    def written(self):
        return np.concatenate(self.history) if self.history else np.zeros(0, dtype=np.int64)


class PageTableLayout(Layout):
    """A ``Layout`` whose LPAs come from a placed batch: KV head ``r * H + h`` is request r's head h."""

    def __init__(self, cfg, shape, spec, tensors=None):
        """``shape`` is one request's (layers, KV heads, head_dim); its context is ignored."""
        if spec.default:
            raise ValueError("Use the layout directly for phase A's layer-major batch")
        self.spec, self.heads = spec, shape.kv_heads
        contexts = [r.context for r in spec.requests]
        # The base check covers one request at the largest context; the placed table is checked below.
        Layout.__init__(self, cfg, Shape(shape.layers, shape.kv_heads, shape.head_dim, max(contexts)), tensors=tensors)
        self.shape = Shape(shape.layers, shape.kv_heads * len(spec.requests), shape.head_dim, max(contexts))
        if any(self.window_starts):
            raise ValueError("Per-request contexts need HBF tensors that start at token 0")
        self.windows = {name: (0, self.shape.context) if self.specs[name].tier == "hbm" else window
                        for name, window in self.windows.items()}
        self.snapshot = place_batch(cfg, shape, spec, self.tensors, self.tokens_per_page)
        table = self.snapshot.table
        self.span = int(table.max()) + 1 if table.size and table.max() >= 0 else 0
        written = self.snapshot.written()
        if len(written):
            self.span = max(self.span, int(written.max()) + 1)
            cfg.lba(self.base + self.span - 1)
        self._keys = None

    def request_window(self, r, t):
        """Tokens [0, hi) of HBF tensor t that request r holds in full pages."""
        return 0, self.snapshot.full_pages[r, t] * self.tokens(t)

    def lpa(self, layer, head, t, block):
        return int(self.lpa_array(layer, np.array([head]), np.array([t]), np.array([block]))[0])

    def lpa_array(self, layer, head, t, block):
        r, h = np.divmod(np.asarray(head, dtype=np.int64), self.heads)
        block = np.asarray(block, dtype=np.int64)
        table = self.snapshot.table
        inside = block < table.shape[4]
        lpa = np.full(len(block), -1, dtype=np.int64)
        lpa[inside] = table[r[inside], layer, h[inside], np.asarray(t)[inside], block[inside]]
        if (lpa < 0).any():
            bad = int(np.flatnonzero(lpa < 0)[0])
            raise ValueError(f"Request {int(r[bad])} has no page {int(block[bad])} of tensor {int(np.asarray(t)[bad])} "
                             f"in layer {layer}")
        return self.base + lpa

    def decode(self, lpa):
        if self._keys is None:
            table = self.snapshot.table
            index = np.argwhere(table >= 0)
            self._keys = dict(zip(table[table >= 0].tolist(), map(tuple, index.tolist())))
        r, layer, h, t, block = self._keys[lpa - self.base]
        return layer, r * self.heads + h, t, block


def full_pages(spec_t, context, tokens_per_page):
    """Full pages of an HBF tensor when the request holds ``context`` tokens (0 if its window is empty)."""
    try:
        return spec_t.tokens(context)[1] // tokens_per_page
    except ValueError:  # e.g. a KIVI body before the context exceeds the residual window
        return 0


def fill_context(spec_t, tokens):
    """Smallest context at which the tensor's window holds ``tokens`` tokens (window starting at 0)."""
    hi = (spec_t.window or (None, None))[1]
    if hi is None:
        return tokens
    if hi < 0:
        return tokens - hi
    return tokens if tokens <= hi else float("inf")


def flush_events(spec, layers, heads, tokens_per_page, full_prompt, full_final, tensors):
    """Decode flushes of every request: (time, request, tensor, block, layer order).

    Request r is admitted at step a_r = max generated - generated_r. Page k of tensor t fills when
    the tensor's window reaches (k + 1) * tokens_per_page tokens (for a KIVI body ``[0, -R]`` that is
    R tokens later than the context); all (layer, head) of that tensor flush together, K's pages before
    V's when both fill at the same step. ``rotate`` starts the g-th flush of a tensor at layer g mod L,
    ``layer`` at layer 0.
    """
    longest = max(r.generated for r in spec.requests)
    events = []
    for r, request in enumerate(spec.requests):
        admitted = longest - request.generated
        pending = sorted((fill_context(tensors[t], (k + 1) * tokens_per_page[t]) - request.prompt, t, k)
                         for t in range(len(tokens_per_page)) for k in range(full_prompt[r, t], full_final[r, t]))
        for when, t, k in pending:
            first = (k - full_prompt[r, t]) % layers if spec.flush_order == "rotate" else 0
            events.append((admitted + when, r, t, k, np.roll(np.arange(layers), -first)))
    return sorted(events, key=lambda event: (event[0], event[1]))


def _flush_keys(event, heads):
    """(layer, head) arrays of one flush in write order: layer by layer, heads within."""
    _, _, _, _, order = event
    return np.repeat(order, heads), np.tile(np.arange(heads), len(order))


def place_batch(cfg, shape, spec, tensors, tokens_per_page):
    """Page table and write history of ``spec`` (one stack)."""
    count = len(spec.requests)
    full_prompt, full_final = {}, {}
    for r, request in enumerate(spec.requests):
        for t, spec_t in enumerate(tensors):
            full_prompt[r, t] = full_pages(spec_t, request.prompt, tokens_per_page[t])
            full_final[r, t] = full_pages(spec_t, request.context, tokens_per_page[t])
    longest_blocks = max(full_final.values())
    if cfg.capacity_sectors // cfg.spp >= 2**31:
        raise ValueError("The page table stores LPAs as int32")
    table = np.full((count, shape.layers, shape.kv_heads, len(tensors), longest_blocks), -1, dtype=np.int32)
    snapshot = Snapshot(table, dict(full_final))
    longest = max(r.generated for r in spec.requests)
    timeline = [(longest - request.generated, 0, r, None) for r, request in enumerate(spec.requests)]
    timeline += [(event[0], 1, event[1], event) for event in
                 flush_events(spec, shape.layers, shape.kv_heads, tokens_per_page, full_prompt, full_final, tensors)]
    timeline.sort(key=lambda item: item[:3])
    placer = {"layer_major": LayerMajor, "vllm_pool": VllmPool}.get(spec.allocation, Segments)
    placer(cfg, shape, spec, tensors, tokens_per_page, snapshot, full_prompt, full_final).run(timeline)
    return snapshot


class Placer:
    def __init__(self, cfg, shape, spec, tensors, tokens_per_page, snapshot, full_prompt, full_final):
        self.cfg, self.shape, self.spec, self.tensors = cfg, shape, spec, tensors
        self.tokens_per_page, self.snapshot = tokens_per_page, snapshot
        self.full_prompt, self.full_final = full_prompt, full_final

    def run(self, timeline):
        for _, kind, r, event in timeline:
            if kind == 0:
                self.admit(r)
            else:
                layers, heads = _flush_keys(event, self.shape.kv_heads)
                self.flush(r, event[2], event[3], layers, heads)

    def prompt_keys(self, r):
        """(layer, head, tensor, block) arrays of request r's full prompt pages."""
        parts = []
        for t in range(len(self.tensors)):
            blocks = self.full_prompt[r, t]
            if blocks:
                layer, head, block = np.meshgrid(np.arange(self.shape.layers), np.arange(self.shape.kv_heads),
                                                 np.arange(blocks), indexing="ij")
                parts.append((layer.ravel(), head.ravel(), np.full(layer.size, t), block.ravel()))
        if not parts:
            return (np.zeros(0, dtype=np.int64),) * 4
        return tuple(np.concatenate(column) for column in zip(*parts))

    def record(self, r, layer, head, t, block, lpa):
        self.snapshot.table[r, layer, head, t, block] = lpa


class LayerMajor(Placer):
    """Phase A's regions: request r's head h is KV head r * H + h of one layout at the largest context."""

    def run(self, timeline):
        self.region = LAYOUTS[self.spec.intra](self.cfg, Shape(
            self.shape.layers, self.shape.kv_heads * len(self.spec.requests), self.shape.head_dim,
            max(r.context for r in self.spec.requests)), tensors=self.tensors)
        super().run(timeline)

    def lpas(self, r, layer, head, t, block):
        out = np.empty(len(layer), dtype=np.int64)
        for value in np.unique(layer):
            mask = layer == value
            out[mask] = self.region.lpa_array(int(value), r * self.shape.kv_heads + head[mask], t[mask], block[mask])
        return out

    def admit(self, r):
        layer, head, t, block = self.prompt_keys(r)
        lpa = self.lpas(r, layer, head, t, block)
        self.record(r, layer, head, t, block, lpa)
        self.snapshot.write(np.sort(lpa), "prefill")

    def flush(self, r, t, block, layer, head):
        t, block = np.full(len(layer), t), np.full(len(layer), block)
        lpa = self.lpas(r, layer, head, t, block)
        self.record(r, layer, head, t, block, lpa)
        self.snapshot.write(lpa, "decode")


class Segments(Placer):
    """dedicated and packed allocations in the address space of a partition (see the module docstring)."""

    def run(self, timeline):
        self.parts, self.banks, self.to_lpa = partition_geometry(self.cfg, self.spec.partition)
        self.segment = self.banks * self.cfg.pages_per_block
        self.load = [0] * self.parts
        self.owner, self.frontier, self.segments, self.used, self.admitted = {}, {}, {}, {}, {}
        self.next_segment = [0] * self.parts
        super().run(timeline)

    def local(self, r, n):
        """Partition-local page indices of the next n pages of request r's write stream."""
        p = self.owner[r]
        if self.spec.allocation == "packed":
            start = self.frontier.get(p, 0)
            self.frontier[p] = start + n
            return np.arange(start, start + n, dtype=np.int64)
        used = self.used[r] + np.arange(n, dtype=np.int64)
        segments = self.segments[r]
        while len(segments) * self.segment < self.used[r] + n:
            segments.append(self.next_segment[p])
            self.next_segment[p] += 1
        self.used[r] += n
        return np.asarray(segments, dtype=np.int64)[used // self.segment] * self.segment + used % self.segment

    def admit(self, r):
        request = self.spec.requests[r]
        p = min(range(self.parts), key=lambda q: (self.load[q], q))
        self.owner[r] = p
        intra = LAYOUTS[self.spec.intra](self.cfg, Shape(self.shape.layers, self.shape.kv_heads, self.shape.head_dim,
                                                         request.prompt), tensors=self.tensors)
        # Every prompt starts on a row of the partition; with ``stagger`` the k-th request of the
        # partition then skips k * banks / n pages (n requests per partition, written as padding), so
        # that layer l of different requests does not start on the same banks.
        index = self.admitted.get(p, 0)
        self.admitted[p] = index + 1
        share = -(-len(self.spec.requests) // self.parts)
        offset = index * max(self.banks // share, 1) % self.banks if self.spec.align == "stagger" else 0
        if self.spec.allocation == "packed":
            start = self.frontier.get(p, 0)
            aligned = -(-start // self.banks) * self.banks + offset
            if aligned > start:
                self.snapshot.write(self.to_lpa(p, np.arange(start, aligned, dtype=np.int64)), "padding")
            self.frontier[p] = aligned
        else:
            self.segments[r], self.used[r] = [], 0
            if offset:
                self.snapshot.write(self.to_lpa(p, self.local(r, offset)), "padding")
        self.load[p] += intra.span
        local = self.local(r, intra.span)
        lpa = self.to_lpa(p, local)
        layer, head, t, block = self.prompt_keys(r)
        slots = np.empty(len(layer), dtype=np.int64)
        for value in np.unique(layer):
            mask = layer == value
            slots[mask] = intra.lpa_array(int(value), head[mask], t[mask], block[mask])
        self.record(r, layer, head, t, block, lpa[slots])
        filled = np.zeros(intra.span, dtype=bool)
        filled[slots] = True
        self.snapshot.kinds["prefill"] += int(filled.sum())
        self.snapshot.kinds["padding"] += int((~filled).sum())
        self.snapshot.history.append(lpa)

    def flush(self, r, t, block, layer, head):
        self.load[self.owner[r]] += len(layer)
        lpa = self.to_lpa(self.owner[r], self.local(r, len(layer)))
        self.record(r, layer, head, np.full(len(layer), t), np.full(len(layer), block), lpa)
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
        self.blocks = [-(-max(self.full_final[r, t] * tpp[t] for t in tensors) // block_tokens)
                       for r in range(len(self.spec.requests))]
        total = sum(self.blocks)
        self.pool = -(-total * self.shape.kv_heads * self.stride // self.cfg.units) * self.cfg.units
        self.ids = list(range(total))
        if self.spec.free_order != "fresh":
            Random(int(self.spec.free_order[9:])).shuffle(self.ids)
        self.owned = {}
        super().run(timeline)

    def lpas(self, r, layer, head, t, block):
        tpp = np.asarray(self.tokens_per_page, dtype=np.int64)[t]
        per = np.asarray(self.per, dtype=np.int64)[t]
        index = np.asarray(self.owned[r], dtype=np.int64)[block * tpp // self.block_tokens]
        return (layer * self.pool + index * self.shape.kv_heads * self.stride + head * self.stride
                + self.offsets[t] + block % per)

    def admit(self, r):
        self.owned[r] = [self.ids.pop(0) for _ in range(self.blocks[r])]
        layer, head, t, block = self.prompt_keys(r)
        lpa = self.lpas(r, layer, head, t, block)
        self.record(r, layer, head, t, block, lpa)
        self.snapshot.write(np.sort(lpa), "prefill")

    def flush(self, r, t, block, layer, head):
        t, block = np.full(len(layer), t), np.full(len(layer), block)
        lpa = self.lpas(r, layer, head, t, block)
        self.record(r, layer, head, t, block, lpa)
        self.snapshot.write(lpa, "decode")


def check_write_order(cfg, lpas):
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

    Returns these counts, the writes and blocks, and ``legal`` (all counts zero).
    """
    lpas = np.asarray(lpas, dtype=np.int64)
    empty = {"writes": 0, "blocks": 0, "skips": 0, "backward": 0, "overwrites": 0, "live_erases": 0, "legal": True}
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
            "legal": skips == backward == overwrites == erases == 0}


def generate_batch(cfg, layout, mode, *, steps=1, seed=1, fraction=.1, count=4, select_tokens=16):
    """Reads of a placed batch: ``dense``, ``uniform_tokens`` or ``sink_recent`` per request, over each
    request's full pages (see ``generate``; the draw order is step, layer, request, head)."""
    if mode not in ("dense", "uniform_tokens", "sink_recent"):
        raise ValueError("A placed batch supports the dense, uniform_tokens and sink_recent modes")
    rng = Random(seed)
    heads, spec = layout.heads, layout.spec
    always = [spec_t for spec_t in layout.all_tensors if spec_t.role == "meta" or spec_t.tier == "hbm"]
    for step in range(steps):
        for layer in range(layout.shape.layers):
            for r, request in enumerate(spec.requests):
                context = request.context
                windows = {spec_t.name: layout.request_window(r, layout.index[spec_t.name])
                           for spec_t in layout.tensors}
                for h in range(heads):
                    head = r * heads + h

                    def clipped(spec_t, lo, hi):
                        lo, hi = max(lo, windows[spec_t.name][0]), min(hi, windows[spec_t.name][1])
                        return [Access(step, layer, head, spec_t.name, "R", lo, hi, spec_t.bits)] if lo < hi else []

                    if mode == "dense":
                        for spec_t in layout.tensors:
                            if spec_t.role != "meta":
                                yield from clipped(spec_t, 0, context)
                    elif mode == "uniform_tokens":
                        blocks = -(-context // select_tokens)
                        for block in sorted(rng.sample(range(blocks), share(blocks, fraction))):
                            lo, hi = block * select_tokens, min((block + 1) * select_tokens, context)
                            for spec_t in layout.tensors:
                                if spec_t.role != "meta":
                                    yield from clipped(spec_t, lo, hi)
                    else:
                        recent = max(1, round(context * fraction))
                        sink = min(count, context - recent)
                        for spec_t in layout.tensors:
                            if spec_t.role != "meta":
                                if sink > 0:
                                    yield from clipped(spec_t, 0, sink)
                                yield from clipped(spec_t, context - recent, context)
                    for spec_t in always:
                        if spec_t.tier == "hbm":
                            lo, hi = spec_t.tokens(context)
                        else:
                            lo, hi = windows[spec_t.name]
                        if lo < hi:
                            yield Access(step, layer, head, spec_t.name, "R", lo, hi, spec_t.bits)
