"""Pure-Python reference analysis engine (``analyze`` and ``analyze_group``).

Kept under tests/ as an independent oracle for the vectorised numpy engine in ``hbf.analysis``.
"""
from collections import Counter, defaultdict
from math import ceil

from hbf.analysis import (check_group, check_protocol, locate, placement, plane_geometry,
                          protocol_rounds, summarize)

PLACEMENTS = ("static", "mqsim")


def analyze_group(cfg, requests, group=1, offsets=None):
    """Round metrics of the reads of one (step, layer), or of one sub-batch."""
    check_group(cfg, group)
    dies, per_die = plane_geometry(cfg)
    reads = [req for req in requests if req.op == "R"]
    if len({req.lpa for req in reads}) != len(reads):
        raise ValueError("Requests of one layer must be deduplicated pages")
    needed = defaultdict(set)
    planes = set()
    for req in reads:
        die, plane, offset = locate(cfg, req.lpa, offsets)
        planes.add((die, plane))
        needed[die, plane // group].add(offset)
    pages = len(reads)
    total = dies * per_die
    rounds = max(map(len, needed.values()), default=0)
    per_channel = Counter(cfg.device_lpa(req.lpa) % cfg.channels for req in reads)
    channel_rounds = ceil(max(per_channel.values(), default=0) * cfg.transfer_ns / cfg.read_ns)
    best = ceil(pages / total)
    senses = group * sum(map(len, needed.values()))
    useful = sum(req.useful_bytes for req in reads)
    bursts = sum(req.burst_bytes for req in reads)
    return {
        "group": group, "planes": total, "pages": pages, "active_planes": len(planes),
        "rounds": rounds, "min_rounds": best, "channel_rounds": channel_rounds,
        "utilization": pages / (rounds * total) if rounds else None,
        "balance_loss": rounds / best if rounds else None,
        "sensed_pages": senses, "sense_amplification": senses / pages if pages else None,
        "useful_bytes": useful, "page_bytes": pages * cfg.page_bytes, "burst_bytes": bursts,
        "read_amplification": pages * cfg.page_bytes / useful if useful else None,
        "transfer_amplification": bursts / useful if useful else None,
    }


def analyze(cfg, batches, groups=(1,), mode="static", protocol=None):
    """Per-layer rows and per-step totals for each plane organization in ``groups``."""
    for group in groups:
        check_group(cfg, group)
    if mode not in PLACEMENTS:
        raise ValueError(f"placement must be one of {PLACEMENTS}")
    if protocol is not None:
        check_protocol(cfg, protocol)
    offsets = placement(cfg, batches) if mode == "mqsim" else None
    parts = defaultdict(list)
    for batch in batches:
        part = [req for req in batch.requests if req.op == "R"]
        if part:
            parts[batch.step, batch.layer].append(part)
    layers = []
    for group in groups:
        for (step, layer), chunks in sorted(parts.items()):
            requests = [req for chunk in chunks for req in chunk]
            row = {"step": step, "layer": layer, **analyze_group(cfg, requests, group, offsets)}
            row["subbatches"] = len(chunks)
            row["batch_rounds"] = sum(analyze_group(cfg, chunk, group, offsets)["rounds"] for chunk in chunks)
            if protocol is not None:
                limited = [protocol_rounds(cfg, chunk, group, offsets, protocol) for chunk in chunks]
                row["commands"] = sum(item[2] for item in limited)
                row["credit_rounds"] = sum(item[1] for item in limited)
                row["link_rounds"] = sum(item[3] for item in limited)
                row["protocol_rounds"] = sum(max(item[0], item[3]) for item in limited)
                row["protocol_utilization"] = row["pages"] / (row["protocol_rounds"] * row["planes"])
            layers.append(row)
    return layers, summarize(layers, protocol)
