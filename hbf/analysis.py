"""Mechanism analysis: read rounds and plane utilization of each layer, without running MQSim.

A decode step reads its layers one after another, so every (step, layer) group of reads is
treated as an independent burst. In one *read round* each plane senses one page. The number of
rounds a group needs depends on how its pages land on the planes and on the plane organization:

* ``group = 1`` - independent planes. Every plane senses any block and page in a round, as the
  OCP HBF spec (per-bank sense ordering) and SPLASH assume. A plane holding ``L`` wanted pages
  needs ``L`` rounds.
* ``group = g > 1`` - lock-step groups of ``g`` planes in one die (FlashAccel's megaplanes,
  and MQSim's own multi-plane rule). All planes of a group sense the same offset in a round, so
  a group needs one round per distinct offset it must read, and every plane of the group senses
  in that round whether it holds a wanted page or not.

Pages are placed by MQSim's CWDP static mapping (``Config.physical``). The *offset* of a page in
its plane is ``lpa // units``, the order in which a contiguous prefill writes it there.
"""
from collections import defaultdict
from math import ceil


def plane_geometry(cfg):
    """HBF dies and planes per die of a topology.

    With one die and one plane per MQSim chip (the chip-as-plane setup), a channel is one HBF die
    and its chips are that die's planes. Otherwise MQSim's own (channel, chip, die) is the die.
    """
    if cfg.dies == cfg.planes == 1:
        return cfg.channels, cfg.chips
    return cfg.channels * cfg.chips * cfg.dies, cfg.planes


def locate(cfg, lpa):
    """(die, plane in die, offset in plane) of an LPA."""
    channel, chip, die, plane = cfg.physical(lpa)
    offset = lpa // cfg.units
    if cfg.dies == cfg.planes == 1:
        return channel, chip, offset
    return (channel * cfg.chips + chip) * cfg.dies + die, plane, offset


def analyze_group(cfg, requests, group=1):
    """Round metrics of the reads of one (step, layer)."""
    dies, per_die = plane_geometry(cfg)
    if type(group) is not int or group < 1 or per_die % group:
        raise ValueError(f"Lock-step group of {group} planes must divide {per_die} planes per die")
    reads = [req for req in requests if req.op == "R"]
    if len({req.lpa for req in reads}) != len(reads):
        raise ValueError("Requests of one layer must be deduplicated pages")
    offsets = defaultdict(set)
    planes = set()
    for req in reads:
        die, plane, offset = locate(cfg, req.lpa)
        planes.add((die, plane))
        offsets[die, plane // group].add(offset)
    pages = len(reads)
    total = dies * per_die
    rounds = max(map(len, offsets.values()), default=0)
    best = ceil(pages / total)
    senses = group * sum(map(len, offsets.values()))
    useful = sum(req.useful_bytes for req in reads)
    bursts = sum(req.burst_bytes for req in reads)
    return {
        "group": group, "planes": total, "pages": pages, "active_planes": len(planes),
        "rounds": rounds, "min_rounds": best,
        "utilization": pages / (rounds * total) if rounds else None,
        "balance_loss": rounds / best if rounds else None,
        "sensed_pages": senses, "sense_amplification": senses / pages if pages else None,
        "useful_bytes": useful, "page_bytes": pages * cfg.page_bytes, "burst_bytes": bursts,
        "read_amplification": pages * cfg.page_bytes / useful if useful else None,
        "transfer_amplification": bursts / useful if useful else None,
    }


def analyze(cfg, batches, groups=(1,)):
    """Per-layer rows and per-step totals for each plane organization in ``groups``."""
    reads = defaultdict(list)
    for batch in batches:
        reads[batch.step, batch.layer].extend(req for req in batch.requests if req.op == "R")
    layers, steps = [], []
    for group in groups:
        totals = {}
        for (step, layer), requests in sorted(reads.items()):
            if not requests:
                continue
            row = {"step": step, "layer": layer, **analyze_group(cfg, requests, group)}
            layers.append(row)
            total = totals.setdefault(step, {"group": group, "step": step, "planes": row["planes"],
                                             "layers": 0, "pages": 0,
                                             "rounds": 0, "min_rounds": 0, "sensed_pages": 0,
                                             "useful_bytes": 0, "burst_bytes": 0})
            total["layers"] += 1
            for key in ("pages", "rounds", "min_rounds", "sensed_pages", "useful_bytes", "burst_bytes"):
                total[key] += row[key]
        for total in totals.values():
            total["utilization"] = total["pages"] / (total["rounds"] * total["planes"])
            total["balance_loss"] = total["rounds"] / total["min_rounds"]
            total["sense_amplification"] = total["sensed_pages"] / total["pages"]
            steps.append(total)
    return layers, steps
