"""Address trace: the sole interface between the Python frontend and all backends (MQSim and rounds).

An address trace has three parts (docs/design/restructure.md section 3.1):
- ``addr.npz``: per-page records (step, layer, subbatch, request, op, lpa, useful_bytes,
  burst_bytes, burst_runs, cls). No arrival timestamps (backends assign arrival times in
  (step, layer, subbatch) order). Within the same (step, layer), writes precede reads.
- ``layers.npz``: per-(step, layer) side table covering every (step, layer), including layers
  with zero HBF pages (hbm_bytes, attended_tokens, attended_elements, hbf_elements, dequant_ops,
  tensor_tokens).
- ``addr.json``: device config digest, layout metadata, generation config, protocol options, and
  ``resident`` LPA intervals already stored in HBF at run start.
"""
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path

import numpy as np

from ..device.address import block_and_page, block_page, place
from ..util import dump
from .pages import LayerPages, iter_layer_pages, prefill_batches, schedule

VALID_OPS = frozenset(("R", "W", "D"))
VALID_CLASSES = frozenset(("kv", "prefill", "flush", "refresh", "relocate"))

EMPTY_I64 = np.zeros(0, dtype=np.int64)
EMPTY_OP = np.zeros(0, dtype="U1")
EMPTY_CLS = np.zeros(0, dtype="U8")


@dataclass
class LayerChunk:
    """One (step, layer) slice of an address trace (page arrays + side-table demand row)."""
    step: int
    layer: int
    subbatch: np.ndarray = field(default_factory=lambda: EMPTY_I64)
    request: np.ndarray = field(default_factory=lambda: EMPTY_I64)
    op: np.ndarray = field(default_factory=lambda: EMPTY_OP)
    lpa: np.ndarray = field(default_factory=lambda: EMPTY_I64)
    useful_bytes: np.ndarray = field(default_factory=lambda: EMPTY_I64)
    burst_bytes: np.ndarray = field(default_factory=lambda: EMPTY_I64)
    burst_runs: np.ndarray = field(default_factory=lambda: EMPTY_I64)
    cls: np.ndarray = field(default_factory=lambda: EMPTY_CLS)
    hbm_bytes: int = 0
    attended_tokens: int = 0
    attended_elements: int = 0
    hbf_elements: int = 0
    dequant_ops: float = 0.0
    tensor_tokens: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.lpa)

    def read_pages(self):
        """Return a ``LayerPages`` view of the read (op == 'R') records in this chunk."""
        if len(self.lpa) and (self.op == "R").all():
            lpa, useful, burst, runs = self.lpa, self.useful_bytes, self.burst_bytes, self.burst_runs
        else:
            mask = self.op == "R"
            lpa = self.lpa[mask]
            useful = self.useful_bytes[mask]
            burst = self.burst_bytes[mask]
            runs = self.burst_runs[mask]
        return LayerPages(
            self.step, self.layer, lpa, np.zeros(len(lpa), dtype=np.int64),
            useful, burst, runs,
            hbm_bytes=self.hbm_bytes, attended_tokens=self.attended_tokens,
            tensor_tokens=self.tensor_tokens, dequant_ops=self.dequant_ops,
            attended_elements=self.attended_elements, hbf_elements=self.hbf_elements,
        )


@dataclass
class AddressTrace:
    """Complete in-memory address trace (metadata + ordered list of ``LayerChunk``s)."""
    meta: dict
    chunks: list[LayerChunk]


def device_digest(cfg):
    """Stable SHA-256 digest of a device ``Config``."""
    payload = json.dumps(asdict(cfg), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def make_meta(cfg, *, layout_meta=None, config_meta=None, protocol=None, resident=()):
    return {
        "device_sha256": device_digest(cfg),
        "layout": layout_meta,
        "config": config_meta,
        "protocol": None if protocol is None else asdict(protocol),
        "resident": [[int(lo), int(hi)] for lo, hi in resident],
    }


class TraceValidator:
    """Streaming validator for address-trace invariants (section 3.1, §5.4.1, §5.4.3).

    Checks:
    1. Every LPA is within ``[0, capacity_pages)``.
    2. Every read (``op == 'R'``) hits a page that is either in ``resident`` (and not released)
       or currently written (OCP §5.4.3, Table 12 error 0x7).
    3. Every write (``op == 'W'``) obeys sequential in-block write order (§5.4.1) with liveness:
       writing page 0 of a block requires all pages of that block to be either released (``D``)
       or never written; writing page ``p > 0`` requires ``p == next_page`` of that block.
    """

    def __init__(self, cfg, resident=()):
        self.cfg = cfg
        self.cap = cfg.capacity_sectors // cfg.spp
        self.ppb = cfg.pages_per_block
        self.resident = tuple((int(lo), int(hi)) for lo, hi in resident)
        # Per-block state for blocks touched by W or D (lazily initialised from resident intervals):
        # block_id -> [next_page, live_set]
        self._blocks = {}

    def _in_resident(self, lpa):
        for lo, hi in self.resident:
            if lo <= lpa < hi:
                return True
        return False

    def _resident_pages_in_block(self, unit, block):
        """Return set of pages [0..ppb) of (unit, block) that lie in ``self.resident``."""
        if not self.resident:
            return set()
        pages = set()
        for p in range(self.ppb):
            lpa = self.cfg.unit_lpa(unit, block * self.ppb + p)
            if self._in_resident(lpa):
                pages.add(p)
        return pages

    def _get_block(self, unit, block):
        key = (unit, block)
        state = self._blocks.get(key)
        if state is None:
            live = self._resident_pages_in_block(unit, block)
            nxt = (max(live) + 1) if live else 0
            state = [nxt, live]
            self._blocks[key] = state
        return state

    def feed(self, chunk: LayerChunk):
        n = len(chunk.lpa)
        if n == 0:
            return
        lpas = chunk.lpa
        if int(lpas.min()) < 0 or int(lpas.max()) >= self.cap:
            bad = int(lpas[(lpas < 0) | (lpas >= self.cap)][0])
            raise ValueError(f"Address trace LPA {bad} out of range [0, {self.cap})")
        ops = chunk.op
        # Fast path: pure-read chunk where every LPA lies inside an untouched resident interval.
        if (ops == "R").all() and not self._blocks and self.resident:
            lo_min, hi_max = int(lpas.min()), int(lpas.max())
            if any(lo <= lo_min and hi_max < hi for lo, hi in self.resident):
                return
        # Fast path: pure-write chunk that writes contiguous LPAs starting on block page 0 on clean blocks.
        if (ops == "W").all() and not self.resident and not self._blocks:
            placed = place(self.cfg, lpas)
            blks, pgs = block_and_page(self.cfg, placed)
            # Validate sequential order per block via numpy when every block starts at page 0.
            order = np.lexsort((np.arange(n, dtype=np.int64), blks))
            sb, sp = blks[order], pgs[order]
            starts = np.empty(n, dtype=bool)
            starts[0] = True
            starts[1:] = sb[1:] != sb[:-1]
            idx = np.flatnonzero(starts)
            counts = np.diff(np.append(idx, n))
            expected_pg = np.arange(n, dtype=np.int64) - np.repeat(idx, counts)
            if (sp == expected_pg).all():
                u_arr = placed.unit[order]
                b_in_plane = (placed.offset[order] // self.ppb)
                for start_i, cnt in zip(idx.tolist(), counts.tolist()):
                    u = int(u_arr[start_i])
                    b = int(b_in_plane[start_i])
                    self._blocks[u, b] = [cnt, set(range(cnt))]
                return
        # General element-wise validation (handles mixed W/R/D, auto-erase, liveness).
        for op, lpa in zip(ops.tolist(), lpas.tolist()):
            if op not in VALID_OPS:
                raise ValueError(f"Invalid address trace op {op!r}")
            unit, block, page = block_page(self.cfg, lpa)
            if op == "R":
                state = self._blocks.get((unit, block))
                if state is not None:
                    live = page in state[1]
                else:
                    live = self._in_resident(lpa)
                if not live:
                    raise ValueError(f"Read of unwritten page LPA {lpa} (unit {unit}, block {block}, page {page}; "
                                     "OCP section 5.4.3 / Table 12 error 0x7)")
            elif op == "D":
                state = self._get_block(unit, block)
                state[1].discard(page)
            else:  # "W"
                state = self._get_block(unit, block)
                nxt, live = state
                if page == 0:
                    if live:
                        raise ValueError(f"Write to page 0 of block ({unit}, {block}) would erase "
                                         f"{len(live)} live pages (OCP section 5.4.1)")
                    state[0] = 1
                    state[1] = {0}
                else:
                    if page != nxt:
                        raise ValueError(f"Out-of-order write to ({unit}, {block}, page {page}); "
                                         f"expected page {nxt} (OCP section 5.4.1)")
                    state[0] = page + 1
                    live.add(page)


def validate(cfg, trace: AddressTrace):
    """Validate an ``AddressTrace`` against ``cfg``; raises ``ValueError`` on any violation."""
    validator = TraceValidator(cfg, trace.meta.get("resident", ()))
    for chunk in trace.chunks:
        validator.feed(chunk)


def iter_trace(cfg, layout, accesses, *, prefill=False, emit_placement_writes=False, expected_steps=(),
               protocol=None, layout_name=None, config_meta=None):
    """Build ``(meta, chunk_iterator)`` streaming by ``(step, layer)`` from logical accesses.

    * Read-only runs without prefill set ``resident = [(layout.base, layout.base + layout.span)]``.
    * Runs with prefill, placement write history (``emit_placement_writes``), or explicit write
      accesses set ``resident = []`` and emit ``op='W'`` records (before any ``op='R'`` records
      within the same ``(step, layer)``).
    * Every ``(step, layer)`` in ``expected_steps x range(layout.shape.layers)`` (plus any step
      seen in ``accesses``) yields a ``LayerChunk``, even when it has zero HBF pages.
    """
    accesses = list(accesses)
    use_placement_writes = bool(emit_placement_writes and hasattr(layout, "snapshot"))
    has_writes = prefill or use_placement_writes or any(a.op == "W" for a in accesses)
    resident = () if has_writes else ((layout.base, layout.base + layout.span),)
    meta = make_meta(
        cfg,
        layout_meta={"name": layout_name, "base": layout.base, "span": layout.span,
                     "layers": layout.shape.layers, "kv_heads": layout.shape.kv_heads,
                     "head_dim": layout.shape.head_dim, "context": layout.shape.context},
        config_meta=config_meta,
        protocol=protocol,
        resident=resident,
    )

    def _chunks():
        limit = cfg.batch_pages
        # 1a. Placement snapshot write history (prefill/padding + decode flush + aged-out releases).
        if use_placement_writes:
            snap = layout.snapshot
            w_lpas_parts, w_cls_parts = [], []
            for arr, kind in zip(snap.history, snap.history_kinds):
                if len(arr):
                    w_lpas_parts.append(arr)
                    tag = "flush" if kind == "decode" else "prefill"
                    w_cls_parts.append(np.full(len(arr), tag, dtype="U8"))
            if w_lpas_parts:
                w_lpas = np.concatenate(w_lpas_parts)
                w_cls = np.concatenate(w_cls_parts)
                n_w = len(w_lpas)
                w_sb = np.arange(n_w, dtype=np.int64) // limit
                w_op = np.full(n_w, "W", dtype="U1")
                dead = snap.dead_lpas()
                if len(dead):
                    next_sb = int(w_sb[-1]) + 1
                    n_d = len(dead)
                    lpas = np.concatenate([w_lpas, dead])
                    sb = np.concatenate([w_sb, next_sb + np.arange(n_d, dtype=np.int64) // limit])
                    op = np.concatenate([w_op, np.full(n_d, "D", dtype="U1")])
                    cls_arr = np.concatenate([w_cls, np.full(n_d, "kv", dtype="U8")])
                else:
                    lpas, sb, op, cls_arr = w_lpas, w_sb, w_op, w_cls
                n = len(lpas)
                yield LayerChunk(
                    step=-1, layer=-1,
                    subbatch=sb,
                    request=np.zeros(n, dtype=np.int64),
                    op=op,
                    lpa=lpas,
                    useful_bytes=np.full(n, cfg.page_bytes, dtype=np.int64),
                    burst_bytes=np.full(n, cfg.page_bytes, dtype=np.int64),
                    burst_runs=np.ones(n, dtype=np.int64),
                    cls=cls_arr,
                )
        # 1b. Prefill writes at step -1, layer -1 (if requested).
        elif prefill:
            if any(a.op == "W" for a in accesses):
                raise ValueError("--prefill writes the whole layout; the accesses must not contain writes")
            p_batches = prefill_batches(cfg, layout)
            lpas = np.arange(layout.base, layout.base + layout.span, dtype=np.int64)
            n = len(lpas)
            yield LayerChunk(
                step=-1, layer=-1,
                subbatch=np.arange(n, dtype=np.int64) // limit,
                request=np.zeros(n, dtype=np.int64),
                op=np.full(n, "W", dtype="U1"),
                lpa=lpas,
                useful_bytes=np.full(n, cfg.page_bytes, dtype=np.int64),
                burst_bytes=np.full(n, cfg.page_bytes, dtype=np.int64),
                burst_runs=np.ones(n, dtype=np.int64),
                cls=np.full(n, "prefill", dtype="U8"),
            )
            del p_batches

        # 2. Explicit writes (e.g. write_then_read at step -1), scheduled via schedule().
        write_accesses = [a for a in accesses if a.op == "W"]
        writes_by_sl = {}
        if write_accesses:
            for batch in schedule(cfg, layout, write_accesses):
                writes_by_sl.setdefault((batch.step, batch.layer), []).append(batch)

        # 3. Read pages by (step, layer) via iter_layer_pages (which validates all accesses).
        reads_by_sl = {}
        for lp in iter_layer_pages(cfg, layout, accesses):
            reads_by_sl[lp.step, lp.layer] = lp

        # Determine all (step, layer) keys in sorted order.
        seen_steps = {s for s, _ in reads_by_sl} | {s for s, _ in writes_by_sl}
        read_steps = sorted(set(expected_steps) | {s for s in seen_steps if s >= 0})
        write_only_keys = sorted(k for k in writes_by_sl if k[0] < 0)

        def _emit_key(step, layer, include_empty_layer):
            w_batches = writes_by_sl.pop((step, layer), ())
            lp = reads_by_sl.pop((step, layer), None)
            if not w_batches and lp is None and not include_empty_layer:
                return None
            parts_sb, parts_op, parts_lpa, parts_u, parts_b, parts_r, parts_cls = [], [], [], [], [], [], []
            sb_offset = 0
            if w_batches:
                for idx, batch in enumerate(w_batches):
                    m = len(batch.requests)
                    parts_sb.append(np.full(m, idx, dtype=np.int64))
                    parts_op.append(np.full(m, "W", dtype="U1"))
                    parts_lpa.append(np.fromiter((req.lpa for req in batch.requests), dtype=np.int64, count=m))
                    parts_u.append(np.fromiter((req.useful_bytes for req in batch.requests), dtype=np.int64, count=m))
                    parts_b.append(np.fromiter((req.burst_bytes for req in batch.requests), dtype=np.int64, count=m))
                    parts_r.append(np.fromiter((req.burst_runs for req in batch.requests), dtype=np.int64, count=m))
                    parts_cls.append(np.full(m, "prefill", dtype="U8"))
                sb_offset = len(w_batches)
            if lp is not None and len(lp):
                m = len(lp)
                parts_sb.append(sb_offset + np.arange(m, dtype=np.int64) // limit)
                parts_op.append(np.full(m, "R", dtype="U1"))
                parts_lpa.append(lp.lpa)
                parts_u.append(lp.useful)
                parts_b.append(lp.burst)
                parts_r.append(lp.runs)
                parts_cls.append(np.full(m, "kv", dtype="U8"))
            if parts_lpa:
                sb = parts_sb[0] if len(parts_sb) == 1 else np.concatenate(parts_sb)
                op = parts_op[0] if len(parts_op) == 1 else np.concatenate(parts_op)
                lpa = parts_lpa[0] if len(parts_lpa) == 1 else np.concatenate(parts_lpa)
                u = parts_u[0] if len(parts_u) == 1 else np.concatenate(parts_u)
                b = parts_b[0] if len(parts_b) == 1 else np.concatenate(parts_b)
                r = parts_r[0] if len(parts_r) == 1 else np.concatenate(parts_r)
                cls = parts_cls[0] if len(parts_cls) == 1 else np.concatenate(parts_cls)
                req_ids = np.zeros(len(lpa), dtype=np.int64)
            else:
                sb = req_ids = lpa = u = b = r = EMPTY_I64
                op = EMPTY_OP
                cls = EMPTY_CLS
            return LayerChunk(
                step=step, layer=layer, subbatch=sb, request=req_ids, op=op, lpa=lpa,
                useful_bytes=u, burst_bytes=b, burst_runs=r, cls=cls,
                hbm_bytes=lp.hbm_bytes if lp is not None else 0,
                attended_tokens=lp.attended_tokens if lp is not None else 0,
                attended_elements=lp.attended_elements if lp is not None else 0,
                hbf_elements=lp.hbf_elements if lp is not None else 0,
                dequant_ops=lp.dequant_ops if lp is not None else 0.0,
                tensor_tokens=lp.tensor_tokens if lp is not None else {},
            )

        for step, layer in write_only_keys:
            chunk = _emit_key(step, layer, include_empty_layer=False)
            if chunk is not None:
                yield chunk
        for step in read_steps:
            for layer in range(layout.shape.layers):
                chunk = _emit_key(step, layer, include_empty_layer=True)
                if chunk is not None:
                    yield chunk

    return meta, _chunks()


def write_trace(directory: Path, meta: dict, chunks):
    """Write ``addr.json``, ``addr.npz`` and ``layers.npz`` into ``directory``."""
    directory = Path(directory)
    chunks = list(chunks)
    dump(directory / "addr.json", meta)

    page_chunks = [c for c in chunks if len(c)]
    if page_chunks:
        step = np.concatenate([np.full(len(c), c.step, dtype=np.int64) for c in page_chunks])
        layer = np.concatenate([np.full(len(c), c.layer, dtype=np.int64) for c in page_chunks])
        subbatch = np.concatenate([c.subbatch for c in page_chunks])
        request = np.concatenate([c.request for c in page_chunks])
        op = np.concatenate([c.op for c in page_chunks])
        lpa = np.concatenate([c.lpa for c in page_chunks])
        useful = np.concatenate([c.useful_bytes for c in page_chunks])
        burst = np.concatenate([c.burst_bytes for c in page_chunks])
        runs = np.concatenate([c.burst_runs for c in page_chunks])
        cls = np.concatenate([c.cls for c in page_chunks])
    else:
        step = layer = subbatch = request = lpa = useful = burst = runs = EMPTY_I64
        op = EMPTY_OP
        cls = EMPTY_CLS
    np.savez(
        directory / "addr.npz",
        step=step, layer=layer, subbatch=subbatch, request=request, op=op, lpa=lpa,
        useful_bytes=useful, burst_bytes=burst, burst_runs=runs, cls=cls,
    )

    layer_chunks = [c for c in chunks if c.step >= 0]
    np.savez(
        directory / "layers.npz",
        step=np.array([c.step for c in layer_chunks], dtype=np.int64),
        layer=np.array([c.layer for c in layer_chunks], dtype=np.int64),
        hbm_bytes=np.array([c.hbm_bytes for c in layer_chunks], dtype=np.int64),
        attended_tokens=np.array([c.attended_tokens for c in layer_chunks], dtype=np.int64),
        attended_elements=np.array([c.attended_elements for c in layer_chunks], dtype=np.int64),
        hbf_elements=np.array([c.hbf_elements for c in layer_chunks], dtype=np.int64),
        dequant_ops=np.array([c.dequant_ops for c in layer_chunks], dtype=np.float64),
        tensor_tokens=np.array([json.dumps(c.tensor_tokens, sort_keys=True) for c in layer_chunks], dtype="U"),
    )


def read_trace(directory: Path) -> AddressTrace:
    """Read ``addr.json``, ``addr.npz`` and ``layers.npz`` from ``directory`` into an ``AddressTrace``."""
    directory = Path(directory)
    meta = json.loads((directory / "addr.json").read_text())
    addr = np.load(directory / "addr.npz")
    layers = np.load(directory / "layers.npz")

    # Group page records by (step, layer).
    p_step = addr["step"]
    p_layer = addr["layer"]
    by_sl = {}
    if len(p_step):
        keys = p_step * (int(p_layer.max()) + 2) + (p_layer + 1)
        change = np.empty(len(keys), dtype=bool)
        change[0] = True
        change[1:] = keys[1:] != keys[:-1]
        starts = np.flatnonzero(change)
        ends = np.append(starts[1:], len(keys))
        for s_i, e_i in zip(starts.tolist(), ends.tolist()):
            sl = (int(p_step[s_i]), int(p_layer[s_i]))
            by_sl[sl] = slice(s_i, e_i)

    # Index side-table rows by (step, layer).
    side = {}
    for i in range(len(layers["step"])):
        sl = (int(layers["step"][i]), int(layers["layer"][i]))
        side[sl] = (
            int(layers["hbm_bytes"][i]),
            int(layers["attended_tokens"][i]),
            int(layers["attended_elements"][i]),
            int(layers["hbf_elements"][i]),
            float(layers["dequant_ops"][i]),
            json.loads(str(layers["tensor_tokens"][i])),
        )

    all_keys = sorted(set(by_sl) | set(side))
    chunks = []
    for sl in all_keys:
        slc = by_sl.get(sl)
        sd = side.get(sl)
        if slc is not None:
            kw = dict(
                subbatch=addr["subbatch"][slc], request=addr["request"][slc],
                op=addr["op"][slc], lpa=addr["lpa"][slc],
                useful_bytes=addr["useful_bytes"][slc], burst_bytes=addr["burst_bytes"][slc],
                burst_runs=addr["burst_runs"][slc], cls=addr["cls"][slc],
            )
        else:
            kw = {}
        if sd is not None:
            hbm_b, att_t, att_e, hbf_e, dq, tt = sd
        else:
            hbm_b = att_t = att_e = hbf_e = 0
            dq = 0.0
            tt = {}
        chunks.append(LayerChunk(
            step=sl[0], layer=sl[1], **kw,
            hbm_bytes=hbm_b, attended_tokens=att_t, attended_elements=att_e,
            hbf_elements=hbf_e, dequant_ops=dq, tensor_tokens=tt,
        ))
    return AddressTrace(meta=meta, chunks=chunks)
