"""Convert AddressTrace / Batch sequences into HBFSim raw-physical requests and replay traces.

Physical byte address encoding matches HBFSim's ``HbfController::decode``
(``src/host/hbf_controller.cpp:1473``):

    ppn = ((((stack * C + channel) * D + die) * P + plane) * B + block) * PPB + page
    byte_addr = ppn * page_bytes

Coordinates ``(host_channel, die, plane, block, page)`` are derived exclusively from
``hbf/device/address.py`` (OCP HBF v0.7.0 section 4.6 mapping).
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path

import numpy as np

from ...device.address import OCP_DIES_PER_HOST_CHANNEL, device_lpa, place

# OCP HBF v0.7.0 Table 2/4 limits enforced by HBFSim's HbfDevice constructor.
SPEED_GRADES = {
    1: {"max_channels": 8, "max_dies": 8, "payload_gb_s_per_ch": 48.0},
    2: {"max_channels": 16, "max_dies": 16, "payload_gb_s_per_ch": 96.0},
    3: {"max_channels": 16, "max_dies": 16, "payload_gb_s_per_ch": 192.0},
}


@dataclass(frozen=True)
class HbfSimGeometry:
    """Resolved HBFSim physical geometry matching a ``hbf.device.config.Config``."""
    stacks: int
    channels_per_stack: int
    dies_per_channel: int
    planes_per_die: int
    blocks_per_plane: int
    pages_per_block: int
    page_bytes: int
    speed_grade: int

    @property
    def total_channels(self) -> int:
        return self.stacks * self.channels_per_stack

    @property
    def total_dies(self) -> int:
        return self.total_channels * self.dies_per_channel

    @property
    def total_planes(self) -> int:
        return self.total_dies * self.planes_per_die

    @property
    def total_blocks(self) -> int:
        return self.total_planes * self.blocks_per_plane

    @property
    def total_pages(self) -> int:
        return self.total_blocks * self.pages_per_block

    @property
    def capacity_bytes(self) -> int:
        return self.total_pages * self.page_bytes


@dataclass(frozen=True)
class ConvertedRequest:
    """One page-granular HBFSim raw-physical request with preserved class metadata."""
    lpa: int
    op: str
    addr: int
    bytes: int
    cls: str = "kv"
    useful_bytes: int = 4096
    burst_bytes: int = 4096
    burst_runs: int = 1


@dataclass(frozen=True)
class ConvertedBatch:
    """One sub-batch of HBFSim raw-physical requests with arrival timestamp."""
    step: int
    layer: int
    subbatch: int
    arrival_ns: int
    requests: tuple[ConvertedRequest, ...]


@dataclass(frozen=True)
class ConvertedTrace:
    """Complete converted raw-physical trace ready for HBFSim execution."""
    geometry: HbfSimGeometry
    batches: tuple[ConvertedBatch, ...]
    resident: tuple[tuple[int, int], ...]
    resident_blocks_per_plane: int
    dropped_releases: int = 0
    cls_counts: dict[str, int] = None


def infer_speed_grade(cfg) -> int:
    """Select OCP speed grade (1/2/3) closest to ``cfg.aggregate_gb_s``."""
    gb_s = cfg.aggregate_gb_s
    if gb_s <= 512:
        return 1
    if gb_s >= 2048:
        return 3
    return 2


def hbfsim_geometry(cfg, speed_grade: int | None = None) -> HbfSimGeometry:
    """Map ``hbf.device.config.Config`` onto a valid HBFSim physical geometry.

    For ``address_map == "ocp"``, ``total_channels = cfg.host_channels``,
    ``dies_per_channel = 4``, and ``planes_per_die = cfg.planes``. Because upstream
    HBFSim requires ``channels_per_stack * dies_per_channel <= grade.max_dies``,
    ``total_channels`` is factored into ``stacks * channels_per_stack`` so that
    ``stack * channels_per_stack + channel == host_channel`` preserves exact §4.6
    coordinates and identical byte addresses.
    """
    grade = infer_speed_grade(cfg) if speed_grade is None else int(speed_grade)
    if grade not in SPEED_GRADES:
        raise ValueError(f"Unsupported HBF speed grade {grade}; expected 1, 2, or 3")
    limits = SPEED_GRADES[grade]

    if cfg.address_map == "ocp":
        total_ch = cfg.host_channels
        dies_per_ch = OCP_DIES_PER_HOST_CHANNEL
        planes_per_die = cfg.planes
    elif cfg.dies == 1 and cfg.planes == 1:
        total_ch = cfg.channels
        dies_per_ch = 1
        planes_per_die = cfg.chips
    else:
        total_ch = cfg.channels
        dies_per_ch = cfg.chips * cfg.dies
        planes_per_die = cfg.planes

    if dies_per_ch > 4 or planes_per_die > 16:
        raise ValueError(
            f"Topology (dies_per_ch={dies_per_ch}, planes_per_die={planes_per_die}) "
            "exceeds HBFSim OCP limits (<=4 dies/ch, <=16 planes/die)"
        )
    max_ch_per_stack = min(limits["max_channels"], limits["max_dies"] // dies_per_ch)
    if max_ch_per_stack < 1:
        raise ValueError(f"Speed grade {grade} cannot fit {dies_per_ch} dies/channel")

    # Pick the largest channels_per_stack dividing total_ch within HBFSim's grade limit.
    ch_per_stack = 1
    for candidate in range(min(total_ch, max_ch_per_stack), 0, -1):
        if total_ch % candidate == 0:
            ch_per_stack = candidate
            break
    stacks = total_ch // ch_per_stack
    return HbfSimGeometry(
        stacks=stacks,
        channels_per_stack=ch_per_stack,
        dies_per_channel=dies_per_ch,
        planes_per_die=planes_per_die,
        blocks_per_plane=cfg.blocks_per_plane,
        pages_per_block=cfg.pages_per_block,
        page_bytes=cfg.page_bytes,
        speed_grade=grade,
    )


def encode_physical_address(
    geom: HbfSimGeometry,
    stack: int,
    channel: int,
    die: int,
    plane: int,
    block: int,
    page: int,
    offset: int = 0,
) -> int:
    """Encode ``(stack, channel, die, plane, block, page, offset)`` to HBFSim byte address."""
    ppn = (
        ((((stack * geom.channels_per_stack + channel) * geom.dies_per_channel + die)
          * geom.planes_per_die + plane)
         * geom.blocks_per_plane + block)
        * geom.pages_per_block + page
    )
    return ppn * geom.page_bytes + offset


def decode_physical_address(geom: HbfSimGeometry, byte_addr: int) -> tuple[int, int, int, int, int, int]:
    """Decode HBFSim physical byte address into ``(stack, channel, die, plane, block, page)``."""
    unit = byte_addr // geom.page_bytes
    page = unit % geom.pages_per_block
    unit //= geom.pages_per_block
    block = unit % geom.blocks_per_plane
    unit //= geom.blocks_per_plane
    plane = unit % geom.planes_per_die
    unit //= geom.planes_per_die
    die = unit % geom.dies_per_channel
    unit //= geom.dies_per_channel
    channel = unit % geom.channels_per_stack
    unit //= geom.channels_per_stack
    stack = unit % geom.stacks
    return stack, channel, die, plane, block, page


def physical_coords(cfg, lpa: int, *, geom: HbfSimGeometry | None = None) -> tuple[int, int, int, int, int, int]:
    """Scalar ``(stack, channel, die, plane, block, page)`` of host ``lpa`` via ``device/address.py``."""
    geom = hbfsim_geometry(cfg) if geom is None else geom
    if cfg.address_map == "ocp":
        dev = device_lpa(cfg, lpa)
        core_ch = dev % cfg.channels
        chip = (dev // cfg.channels) % cfg.chips
        plane = (dev // (cfg.channels * cfg.chips)) % cfg.planes
        host_ch = core_ch // 2
        die = (core_ch % 2) + 2 * chip
    else:
        ch, chip, die_idx, plane = cfg.physical(lpa)
        if cfg.dies == 1 and cfg.planes == 1:
            host_ch, die, plane = ch, 0, chip
        else:
            host_ch, die = ch, chip * cfg.dies + die_idx
    stack, channel = divmod(host_ch, geom.channels_per_stack)
    block, page = divmod(lpa // cfg.units, cfg.pages_per_block)
    return stack, channel, die, plane, block, page


def physical_coords_array(cfg, lpas, *, geom: HbfSimGeometry | None = None):
    """Vectorized ``(stack, channel, die, plane, block, page, byte_addr)`` via ``device/address.py``."""
    geom = hbfsim_geometry(cfg) if geom is None else geom
    lpa_arr = np.asarray(lpas, dtype=np.int64)
    placed = place(cfg, lpa_arr)
    if cfg.address_map == "ocp":
        host_ch = placed.host
        # placed.channel is core-die channel; placed.die is (core_ch * chips + chip)
        core_ch = placed.channel
        chip = placed.die % cfg.chips
        die = (core_ch % 2) + 2 * chip
        plane = placed.plane
    else:
        if cfg.dies == 1 and cfg.planes == 1:
            host_ch = placed.die
            die = np.zeros_like(host_ch)
            plane = placed.plane
        else:
            host_ch = placed.channel
            die = placed.die % (cfg.chips * cfg.dies)
            plane = placed.plane
    stack = host_ch // geom.channels_per_stack
    channel = host_ch % geom.channels_per_stack
    block = placed.offset // cfg.pages_per_block
    page = placed.offset % cfg.pages_per_block
    ppn = (
        ((((stack * geom.channels_per_stack + channel) * geom.dies_per_channel + die)
          * geom.planes_per_die + plane)
         * geom.blocks_per_plane + block)
        * geom.pages_per_block + page
    )
    byte_addr = ppn * geom.page_bytes
    return stack, channel, die, plane, block, page, byte_addr


def resident_block_count(cfg, resident) -> int:
    """Minimum prefix of blocks per plane ``[0, N)`` covering all ``resident`` LPA intervals."""
    if not resident:
        return 0
    max_hi = max(int(hi) for _, hi in resident)
    if max_hi <= 0:
        return 0
    max_offset = (max_hi - 1) // cfg.units
    return int(max_offset // cfg.pages_per_block) + 1


def convert_trace(cfg, trace, *, geom: HbfSimGeometry | None = None, interval_ns: int | None = None) -> ConvertedTrace:
    """Convert an ``AddressTrace`` into a ``ConvertedTrace`` for HBFSim raw-physical execution.

    * ``op == 'D'`` release records are dropped (raw-physical mode rejects logical invalidation).
    * Arrival timestamps are assigned in ``(step, layer, subbatch)`` order spaced by
      ``interval_ns`` (default ``cfg.interval_ns``), matching MQSim's batch schedule.
    * ``cls`` is preserved on every request for write/erase attribution.
    """
    geom = hbfsim_geometry(cfg) if geom is None else geom
    step_interval = cfg.interval_ns if interval_ns is None else int(interval_ns)
    resident = tuple((int(lo), int(hi)) for lo, hi in trace.meta.get("resident", ()))
    batches = []
    dropped = 0
    cls_counts: dict[str, int] = {}

    for chunk in trace.chunks:
        if not len(chunk):
            continue
        ops = chunk.op.tolist()
        sbs = chunk.subbatch.tolist()
        lpas = chunk.lpa
        _, _, _, _, _, _, addrs = physical_coords_array(cfg, lpas, geom=geom)
        lpa_list = lpas.tolist()
        addr_list = addrs.tolist()
        useful = chunk.useful_bytes.tolist()
        burst = chunk.burst_bytes.tolist()
        runs = chunk.burst_runs.tolist()
        cls_list = chunk.cls.tolist()

        grouped: dict[int, list[ConvertedRequest]] = {}
        order: list[int] = []
        for sb, op, lpa, addr, u, b, r, c in zip(sbs, ops, lpa_list, addr_list, useful, burst, runs, cls_list):
            if op == "D":
                dropped += 1
                continue
            if op not in ("R", "W"):
                raise ValueError(f"Unsupported address trace op {op!r}")
            cls_counts[c] = cls_counts.get(c, 0) + 1
            if sb not in grouped:
                grouped[sb] = []
                order.append(sb)
            grouped[sb].append(
                ConvertedRequest(
                    lpa=lpa,
                    op=op,
                    addr=addr,
                    bytes=cfg.page_bytes,
                    cls=c,
                    useful_bytes=u,
                    burst_bytes=b,
                    burst_runs=r,
                )
            )
        for sb in order:
            batches.append(
                ConvertedBatch(
                    step=chunk.step,
                    layer=chunk.layer,
                    subbatch=sb,
                    arrival_ns=len(batches) * step_interval,
                    requests=tuple(grouped[sb]),
                )
            )

    return ConvertedTrace(
        geometry=geom,
        batches=tuple(batches),
        resident=resident,
        resident_blocks_per_plane=resident_block_count(cfg, resident),
        dropped_releases=dropped,
        cls_counts=cls_counts,
    )


def batches_to_converted(cfg, batches, *, geom: HbfSimGeometry | None = None, resident=()) -> ConvertedTrace:
    """Convert explicit ``Batch`` objects (preserving their exact ``arrival_ns``) to ``ConvertedTrace``."""
    geom = hbfsim_geometry(cfg) if geom is None else geom
    resident_t = tuple((int(lo), int(hi)) for lo, hi in resident)
    out_batches = []
    cls_counts: dict[str, int] = {}
    for batch in batches:
        lpas = np.array([req.lpa for req in batch.requests], dtype=np.int64)
        _, _, _, _, _, _, addrs = physical_coords_array(cfg, lpas, geom=geom)
        reqs = []
        for req, addr in zip(batch.requests, addrs.tolist()):
            if req.op == "D":
                continue
            tag = "prefill" if req.op == "W" else "kv"
            cls_counts[tag] = cls_counts.get(tag, 0) + 1
            reqs.append(
                ConvertedRequest(
                    lpa=req.lpa,
                    op=req.op,
                    addr=addr,
                    bytes=cfg.page_bytes,
                    cls=tag,
                    useful_bytes=req.useful_bytes,
                    burst_bytes=req.burst_bytes,
                    burst_runs=req.burst_runs,
                )
            )
        if reqs:
            out_batches.append(
                ConvertedBatch(
                    step=batch.step,
                    layer=batch.layer,
                    subbatch=batch.subbatch,
                    arrival_ns=batch.arrival_ns,
                    requests=tuple(reqs),
                )
            )
    return ConvertedTrace(
        geometry=geom,
        batches=tuple(out_batches),
        resident=resident_t,
        resident_blocks_per_plane=resident_block_count(cfg, resident_t),
        dropped_releases=0,
        cls_counts=cls_counts,
    )


# Semantic class mapping for hbfsim-reference Ramulator-compatible text traces
# (HBFSim docs/guides/workload-replay.md section 2).
_REPLAY_SEMANTIC_KIND = {
    "kv": "shared_context",
    "prefill": "generated_context",
    "flush": "generated_context",
    "refresh": "generated_context",
    "relocate": "generated_context",
}


def write_replay_trace(path: Path, converted: ConvertedTrace) -> Path:
    """Write a Ramulator-compatible text trace (`0xADDR R|W bytes=... kind=... layer=...`)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        stream.write("# HBFSim raw-physical replay trace generated by hbf.backends.hbfsim.convert\n")
        for batch in converted.batches:
            layer_idx = max(0, batch.layer)
            for req in batch.requests:
                kind = _REPLAY_SEMANTIC_KIND.get(req.cls, "unknown")
                stream.write(
                    f"0x{req.addr:x} {req.op} bytes={req.bytes} kind={kind} layer={layer_idx}\n"
                )
    return path
