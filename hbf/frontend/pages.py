"""Frontend token-to-page lowering: deduplicated per-layer page arrays and scalar sub-batches.

Turns logical token ``Access`` records and a ``Layout`` into deduplicated 4 KiB page accesses
before they are packaged into an ``AddressTrace`` (``hbf/frontend/addrtrace.py``):

* ``iter_layer_pages`` / ``layer_pages``: vectorised numpy path yielding ``LayerPages`` per
  ``(step, layer)`` with useful bytes, 64 B burst bytes/runs, and HBM/attention/dequant demand.
* ``schedule`` / ``prefill_batches`` / ``schedule_run``: scalar sub-batch lowering used for
  prefill writes (OCP §5.4.1 block page-0 alignment) and explicit write accesses.
"""
from array import array
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace

import numpy as np

BURST_BYTES = 64


@dataclass(frozen=True)
class Request:
    lpa: int
    op: str = "R"
    useful_bytes: int = 4096
    burst_bytes: int = 4096
    burst_runs: int = 1  # contiguous runs of wanted 64 B units: one burst read command each


@dataclass(frozen=True)
class Batch:
    step: int
    layer: int
    subbatch: int
    arrival_ns: int
    requests: tuple[Request, ...]


@dataclass
class LayerPages:
    """Read pages of one (step, layer), sorted by LPA (the order ``schedule`` gives requests).

    ``useful``, ``burst`` and ``runs`` equal ``Request.useful_bytes``, ``burst_bytes`` and
    ``burst_runs``. The remaining fields are the demand the system model needs: bytes read from
    HBM-tier tensors, tokens attended (union over the data tensors of each head), attended
    elements (tokens x dims summed over the data tensors: the K and V elements attention
    multiplies) and the part of them read from HBF, tokens read per tensor and dequantization
    operations.
    """
    step: int
    layer: int
    lpa: np.ndarray
    tensor: np.ndarray
    useful: np.ndarray
    burst: np.ndarray
    runs: np.ndarray
    hbm_bytes: int = 0
    attended_tokens: int = 0
    tensor_tokens: dict = field(default_factory=dict)
    dequant_ops: float = 0.0
    attended_elements: int = 0
    hbf_elements: int = 0

    def __len__(self):
        return len(self.lpa)


def union_size(intervals):
    end = total = 0
    for lo, hi in sorted(intervals):
        total += max(0, hi - max(lo, end))
        end = max(hi, end)
    return total


def burst_chunks(intervals, start, token_bytes, burst=BURST_BYTES):
    return [((lo - start) * token_bytes // burst, -(-(hi - start) * token_bytes // burst))
            for lo, hi in intervals]


def burst_bytes(intervals, start, token_bytes, burst=BURST_BYTES):
    """Bytes moved when a page read returns only the 64 B bursts covering the wanted tokens.

    The OCP HBF base-die spec v0.7.0 (section 4.1) allows host read bursts of 64 B to 4 KiB in
    multiples of 64 B within one NAND page, so the host link can carry less than a page. Between the
    base die and the core dies the minimum is still 4 KiB (section 4.6, device logical unit), so
    this counts host-side bytes only; sensing and the NAND-side transfer stay whole pages.
    """
    return union_size(burst_chunks(intervals, start, token_bytes, burst)) * burst


def burst_runs(intervals, start, token_bytes, burst=BURST_BYTES):
    """Contiguous runs of wanted 64 B units in a page; adjacent or overlapping units merge."""
    runs, end = 0, None
    for lo, hi in sorted(burst_chunks(intervals, start, token_bytes, burst)):
        if end is None or lo > end:
            runs += 1
        end = hi if end is None else max(end, hi)
    return runs


def schedule(cfg, layout, accesses):
    """Deduplicate logical token accesses into sub-batches of ``Request``s (writes before reads)."""
    if any(spec.density != 1 for spec in layout.all_tensors):
        raise ValueError("Tensor density < 1 is supported by the numpy analysis engine only")
    logical = defaultdict(dict)
    for access in accesses:
        access.validate(layout)
        if access.op == "W" and access.step != -1:
            raise ValueError("Phase 0 only accepts prefill writes")
        if access.op == "H" or layout.specs[access.kind].tier != "hbf":
            continue  # HBM-tier tensors and HBM-cached hits have no HBF pages
        group = logical[access.step, access.layer, access.op]
        t = layout.tensor_index(access.kind)
        tokens = layout.tokens(t)
        blocks, first = layout.token_blocks(t, access.token_start, access.token_end)
        for block in blocks:
            start = first + block * tokens
            lpa = layout.lpa(access.layer, access.kv_head, t, block)
            group.setdefault(lpa, (t, start, []))[2].append((
                max(access.token_start, start), min(access.token_end, start + tokens)
            ))
    batches = []
    serials = Counter()
    limit = cfg.batch_pages
    for (step, layer, op), pages in sorted(logical.items(), key=lambda item: (*item[0][:2], item[0][2] == "R")):
        requests = []
        for lpa, (t, start, intervals) in sorted(pages.items()):
            useful = union_size(intervals) * layout.token_bytes(t)
            filled = min(start + layout.tokens(t), layout.windows[layout.tensors[t].name][1]) - start
            if op == "W" and useful != filled * layout.token_bytes(t):
                raise ValueError("Partial-page prefill write would cause an update read")
            token = layout.token_bytes(t)
            requests.append(Request(lpa, op, useful, burst_bytes(intervals, start, token),
                                    burst_runs(intervals, start, token)))
        for start in range(0, len(requests), limit):
            serial = serials[step, layer]
            serials[step, layer] += 1
            batches.append(Batch(step, layer, serial, len(batches) * cfg.interval_ns,
                                 tuple(requests[start:start + limit])))
    return batches


def prefill_batches(cfg, layout):
    """Write every LPA of the layout's span once, in address order, as step -1 (layer -1).

    The OCP HBF spec maps an address to a fixed bank, die, block and page and requires the host to
    write a block's 4 KiB addresses consecutively from page 0 without skipping (section 5.4.1), so
    the span is written whole, padding pages of striped layouts included. MQSim's data write
    frontier uses blocks 0, 3, 4, ... (blocks 1 and 2 are reserved for translation and GC write
    frontiers, Flash_Block_Manager_Base.cpp:53-57), so prior to static-write FTL support MQSim's
    physical block ID does not equal the static block ID; aligning ``base`` to block page 0
    (``units * pages_per_block``) keeps each plane's in-block page offset equal to its static page
    offset.
    """
    if layout.base % (cfg.units * cfg.pages_per_block):
        raise ValueError("A prefill needs a layout base aligned to block page 0 "
                         "(a multiple of units * pages_per_block); otherwise MQSim's in-block page "
                         "offsets differ from the static offsets")
    lpas = range(layout.base, layout.base + layout.span)
    limit = cfg.batch_pages
    return [Batch(-1, -1, index, 0, tuple(Request(lpa, "W") for lpa in lpas[start:start + limit]))
            for index, start in enumerate(range(0, len(lpas), limit))]


def schedule_run(cfg, layout, accesses, prefill=False):
    """Batches of one run: an optional whole-layout prefill, then the scheduled accesses."""
    accesses = list(accesses)
    if prefill and any(access.op == "W" for access in accesses):
        raise ValueError("--prefill writes the whole layout; the accesses must not contain writes")
    batches = (prefill_batches(cfg, layout) if prefill else []) + schedule(cfg, layout, accesses)
    return [replace(batch, arrival_ns=index * cfg.interval_ns) for index, batch in enumerate(batches)]


def _union(group, lo, hi, scale):
    """Per-group union length and run count of [lo, hi) intervals sorted by (group, lo).

    Offsetting every interval by ``group * scale`` (scale > any hi) keeps groups apart, so one
    running maximum over all intervals gives each interval the end of the intervals before it in
    its own group. A run starts where an interval begins after that end (adjacent ones merge, as
    ``burst_runs`` does).
    """
    key_lo, key_hi = group * scale + lo, group * scale + hi
    before = np.empty_like(key_hi)
    before[:1] = -1
    np.maximum.accumulate(key_hi[:-1], out=before[1:])
    count = group[-1] + 1 if len(group) else 0
    length = np.bincount(group, np.maximum(0, key_hi - np.maximum(key_lo, before)), count)
    runs = np.bincount(group, key_lo > before, count)
    return length.astype(np.int64), runs.astype(np.int64)


def per_tensor_tokens(head, g, lo, hi, context, count):
    """Tokens read per tensor, summed over heads (unions within each (head, tensor))."""
    if not len(head):
        return np.zeros(count, dtype=np.int64)
    key = head * count + g
    order = np.lexsort((lo, key))
    key, lo, hi = key[order], lo[order], hi[order]
    new = np.empty(len(key), dtype=bool)
    new[:1] = True
    np.not_equal(key[1:], key[:-1], out=new[1:])
    tokens, _ = _union(np.cumsum(new) - 1, lo, hi, context + 1)
    return np.bincount(key[new] % count, tokens, count).astype(np.int64)


def attended_tokens(head, lo, hi, context):
    """Sum over KV heads of the tokens read by any tensor of the head (the tokens attention covers)."""
    if not len(head):
        return 0
    order = np.lexsort((lo, head))
    head, lo, hi = head[order], lo[order], hi[order]
    new = np.empty(len(head), dtype=bool)
    new[:1] = True
    np.not_equal(head[1:], head[:-1], out=new[1:])
    tokens, _ = _union(np.cumsum(new) - 1, lo, hi, context + 1)
    return int(tokens.sum())


def layer_pages(cfg, layout, step, layer, rows, cached=None):
    """Pages of one (step, layer) from an (n, 4) int64 array of (head, tensor, token_start, token_end),
    where tensor indexes ``layout.all_tensors``. HBM-tier rows (and HBM-cached hits flagged by
    ``cached``) become ``hbm_bytes``; every row counts towards the tokens read per tensor
    (``tensor_tokens``, for dequantization) and data rows towards ``attended_tokens``.
    ``attended_elements`` sums tokens x ``attended_dims`` (data tensors and any tensor with
    ``attn_dims``), ``hbf_elements`` tokens x ``logical`` of HBF data tensors read from HBF.
    Tensors with ``density < 1`` map logical tokens into compacted storage coordinates."""
    specs = layout.all_tensors
    head, g, lo, hi = rows.T
    has_density = any(spec.density != 1 for spec in specs)
    if has_density:
        w_lo = np.asarray([layout.windows[spec.name][0] for spec in specs], dtype=np.int64)[g]
        num = np.asarray([spec.density.numerator for spec in specs], dtype=np.int64)[g]
        den = np.asarray([spec.density.denominator for spec in specs], dtype=np.int64)[g]
        # Combining density < 1 with select_tokens=1 aliases 1/density logical tokens per compacted token.
        c_lo = ((lo - w_lo) * num) // den
        c_hi = ((hi - 1 - w_lo) * num) // den + 1
        eff_lo, eff_hi = w_lo + c_lo, w_lo + c_hi
    else:
        eff_lo, eff_hi = lo, hi
    data = np.asarray([spec.role == "data" for spec in specs])[g]
    # attended_tokens assumes uniform density across data tensors on the same window;
    # attended_elements uses per-tensor counts and is authoritative for system.py.
    attended = attended_tokens(head[data], eff_lo[data], eff_hi[data], layout.shape.context)
    tensor_tokens = per_tensor_tokens(head, g, eff_lo, eff_hi, layout.shape.context, len(specs))
    hbm = np.asarray([spec.tier == "hbm" for spec in specs])
    is_cached = None if cached is None else np.asarray(cached, dtype=bool)
    hbf_cached = None if is_cached is None else (is_cached & ~hbm[g])
    if hbf_cached is not None and hbf_cached.any():
        hbf_read = ~hbm[g] & ~is_cached
        hbf_tokens = per_tensor_tokens(head[hbf_read], g[hbf_read], eff_lo[hbf_read], eff_hi[hbf_read],
                                       layout.shape.context, len(specs))
        cached_tokens = tensor_tokens - hbf_tokens
        hbm_bytes = (sum(tensor_tokens[i] * specs[i].token_bytes for i in np.flatnonzero(hbm).tolist())
                     + sum(cached_tokens[i] * specs[i].token_bytes for i in np.flatnonzero(~hbm).tolist()))
    else:
        hbm_bytes = sum(tensor_tokens[i] * specs[i].token_bytes for i in np.flatnonzero(hbm).tolist())
        hbf_tokens = tensor_tokens
    dequant = sum(tensor_tokens[i] * specs[i].logical * specs[i].dequant_ops for i in range(len(specs)))
    elements = sum(int(tensor_tokens[i]) * specs[i].attended_dims for i in range(len(specs))
                   if specs[i].attended_dims)
    hbf_elements = sum(int(hbf_tokens[i]) * specs[i].logical for i in range(len(specs))
                       if specs[i].role == "data" and specs[i].tier == "hbf")
    hbf_index = np.asarray([layout.index.get(spec.name, -1) for spec in specs], dtype=np.int64)
    keep = ~hbm[g] if is_cached is None else (~hbm[g] & ~is_cached)
    head, t = head[keep], hbf_index[g[keep]]
    tpp = np.asarray(layout.tokens_per_page, dtype=np.int64)[t]
    origin = np.asarray(layout.window_starts, dtype=np.int64)[t]
    if has_density:
        rel_lo, rel_hi = c_lo[keep], c_hi[keep]
        first, last = rel_lo // tpp, (rel_hi - 1) // tpp
    else:
        lo_h, hi_h = lo[keep], hi[keep]
        first, last = (lo_h - origin) // tpp, (hi_h - 1 - origin) // tpp
    count = last - first + 1
    index = np.repeat(np.arange(len(t)), count)
    within = np.arange(int(count.sum()), dtype=np.int64) - np.repeat(np.cumsum(count) - count, count)
    block = first[index] + within
    tpp = tpp[index]
    if has_density:
        start = block * tpp
        lo = np.maximum(rel_lo[index], start) - start
        hi = np.minimum(rel_hi[index], start + tpp) - start
    else:
        start = origin[index] + block * tpp
        lo = np.maximum(lo_h[index], start) - start
        hi = np.minimum(hi_h[index], start + tpp) - start
    t, head = t[index], head[index]
    lpa = layout.lpa_array(layer, head, t, block)
    order = np.lexsort((lo, lpa))
    lpa, t, lo, hi = lpa[order], t[order], lo[order], hi[order]
    new = np.empty(len(lpa), dtype=bool)
    new[:1] = True
    np.not_equal(lpa[1:], lpa[:-1], out=new[1:])
    group = np.cumsum(new) - 1
    tokens, _ = _union(group, lo, hi, int(tpp.max()) + 1 if len(tpp) else 1)
    token_bytes = np.asarray([spec.token_bytes for spec in layout.tensors], dtype=np.int64)[t]
    chunk_lo = lo * token_bytes // BURST_BYTES
    chunk_hi = -(-(hi * token_bytes) // BURST_BYTES)
    chunks, runs = _union(group, chunk_lo, chunk_hi, cfg.page_bytes // BURST_BYTES + 1)
    return LayerPages(step, layer, lpa[new], t[new], tokens * token_bytes[new], chunks * BURST_BYTES, runs,
                      hbm_bytes=int(hbm_bytes), attended_tokens=attended,
                      tensor_tokens={specs[i].name: int(tensor_tokens[i]) for i in range(len(specs))},
                      dequant_ops=float(dequant), attended_elements=elements, hbf_elements=hbf_elements)


def iter_layer_pages(cfg, layout, accesses):
    """``LayerPages`` of every (step, layer) with reads, in (step, layer) order.

    Accesses are validated as ``schedule`` validates them; writes go through ``schedule`` itself
    (its prefill and partial-page checks) and are not part of the read pages.
    """
    groups = defaultdict(lambda: array("q"))
    cached_groups = defaultdict(lambda: array("b"))
    has_cached = False
    writes = []
    index = {spec.name: g for g, spec in enumerate(layout.all_tensors)}
    for access in accesses:
        access.validate(layout)
        if access.op == "W":
            writes.append(access)
            continue
        key = (access.step, access.layer)
        groups[key].extend((access.kv_head, index[access.kind], access.token_start, access.token_end))
        is_cached = access.op == "H"
        if is_cached:
            has_cached = True
        cached_groups[key].append(is_cached)
    if writes:
        schedule(cfg, layout, writes)
    for step, layer in sorted(groups):
        rows = np.frombuffer(groups.pop((step, layer)), dtype=np.int64).reshape(-1, 4)
        cached = np.frombuffer(cached_groups.pop((step, layer)), dtype=bool) if has_cached else None
        yield layer_pages(cfg, layout, step, layer, rows, cached=cached)
