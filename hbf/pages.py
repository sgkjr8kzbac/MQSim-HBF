"""Vectorised page sets: numpy counterparts of ``controller.schedule`` and ``analysis.locate``.

``iter_layer_pages`` turns logical accesses into the deduplicated read pages of every (step,
layer), as ``schedule`` does, but as numpy arrays instead of ``Request`` objects: a 32-layer 128K
dense step has four million pages. ``place`` gives each page's die, plane, offset, core-die
channel and host channel, as ``analysis.locate``, ``Config.device_lpa`` and
``Config.host_channel`` do one LPA at a time. Tests check both against the scalar code.
"""
from array import array
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from .config import OCP_DIES_PER_HOST_CHANNEL
from .controller import BURST_BYTES, schedule


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


@dataclass
class Placed:
    """Location of each page: HBF die and plane (``analysis.locate``), static offset, MQSim channel
    (``device_lpa % channels``), host channel and MQSim unit. ``host`` is ``lpa % host_channels``;
    ``Config.host_channel`` also rejects cwdp channel counts that are not a multiple of 16, which
    callers that use ``host`` must check (``check_host_channels``)."""
    die: np.ndarray
    plane: np.ndarray
    offset: np.ndarray
    channel: np.ndarray
    host: np.ndarray
    unit: np.ndarray


def device_lpas(cfg, lpa):
    """``Config.device_lpa`` for an array of LPAs."""
    lpa = np.asarray(lpa, dtype=np.int64)
    if cfg.address_map == "cwdp":
        return lpa
    hosts, banks = cfg.host_channels, cfg.planes
    host, local = lpa % hosts, lpa // hosts
    bank, die, offset = local % banks, local // banks % OCP_DIES_PER_HOST_CHANNEL, local // (
        banks * OCP_DIES_PER_HOST_CHANNEL)
    channel, chip = 2 * host + die % 2, die // 2
    return channel + cfg.channels * (chip + cfg.chips * (bank + banks * offset))


def place(cfg, lpa):
    """Static locations of an array of LPAs (offset ``lpa // units``, as ``locate`` without offsets)."""
    lpa = np.asarray(lpa, dtype=np.int64)
    if len(lpa) and (lpa.min() < 0 or lpa.max() >= cfg.capacity_sectors // cfg.spp):
        bad = lpa[(lpa < 0) | (lpa >= cfg.capacity_sectors // cfg.spp)][0]
        raise ValueError(f"LPA {bad} exceeds logical capacity")
    device = device_lpas(cfg, lpa)
    channels, chips, dies, planes = cfg.geometry
    channel, rest = device % channels, device // channels
    chip, rest = rest % chips, rest // chips
    die, rest = rest % dies, rest // dies
    plane = rest % planes
    if dies == planes == 1:
        hbf_die, hbf_plane = channel, chip
    else:
        hbf_die, hbf_plane = (channel * chips + chip) * dies + die, plane
    return Placed(hbf_die, hbf_plane, lpa // cfg.units, channel, lpa % cfg.host_channels, device % cfg.units)


def check_host_channels(cfg):
    if cfg.address_map == "cwdp" and cfg.channels % cfg.host_channels:
        raise ValueError("Host channels need a channel count that is a multiple of 16")


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
