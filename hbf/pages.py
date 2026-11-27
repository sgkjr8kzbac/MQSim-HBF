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
    ``burst_runs``. The remaining fields describe reads that do not become HBF pages (HBM-tier
    tensors) and the attended tokens; they stay at their defaults until tensors declare a tier.
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


def layer_pages(cfg, layout, step, layer, rows):
    """Pages of one (step, layer) from an (n, 4) int64 array of (head, tensor, token_start, token_end)."""
    head, t, lo, hi = rows.T
    tpp = np.asarray(layout.tokens_per_page, dtype=np.int64)[t]
    first, last = lo // tpp, (hi - 1) // tpp
    count = last - first + 1
    index = np.repeat(np.arange(len(rows)), count)
    within = np.arange(int(count.sum()), dtype=np.int64) - np.repeat(np.cumsum(count) - count, count)
    block = first[index] + within
    tpp = tpp[index]
    start = block * tpp
    lo = np.maximum(lo[index], start) - start
    hi = np.minimum(hi[index], start + tpp) - start
    t, head = t[index], head[index]
    lpa = layout.lpa_array(layer, head, t, block)
    order = np.lexsort((lo, lpa))
    lpa, t, lo, hi = lpa[order], t[order], lo[order], hi[order]
    new = np.empty(len(lpa), dtype=bool)
    new[:1] = True
    np.not_equal(lpa[1:], lpa[:-1], out=new[1:])
    group = np.cumsum(new) - 1
    tokens, _ = _union(group, lo, hi, int(tpp.max()) + 1 if len(tpp) else 1)
    token_bytes = np.asarray([layout.token_bytes(i) for i in range(len(layout.tensors))], dtype=np.int64)[t]
    chunk_lo = lo * token_bytes // BURST_BYTES
    chunk_hi = -(-(hi * token_bytes) // BURST_BYTES)
    chunks, runs = _union(group, chunk_lo, chunk_hi, cfg.page_bytes // BURST_BYTES + 1)
    page_t = t[new]
    return LayerPages(step, layer, lpa[new], page_t, tokens * token_bytes[new], chunks * BURST_BYTES, runs)


def iter_layer_pages(cfg, layout, accesses):
    """``LayerPages`` of every (step, layer) with reads, in (step, layer) order.

    Accesses are validated as ``schedule`` validates them; writes go through ``schedule`` itself
    (its prefill and partial-page checks) and are not part of the read pages.
    """
    groups = defaultdict(lambda: array("q"))
    writes = []
    for access in accesses:
        access.validate(layout)
        if access.op == "W":
            writes.append(access)
            continue
        groups[access.step, access.layer].extend(
            (access.kv_head, layout.tensor_index(access.kind), access.token_start, access.token_end))
    if writes:
        schedule(cfg, layout, writes)
    for step, layer in sorted(groups):
        rows = np.frombuffer(groups.pop((step, layer)), dtype=np.int64).reshape(-1, 4)
        yield layer_pages(cfg, layout, step, layer, rows)
