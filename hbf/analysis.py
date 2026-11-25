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

Pages are placed on planes by MQSim's CWDP static mapping (``Config.physical``). The *offset* of
a page inside its plane has two models:

* ``static`` (default) - the OCP HBF spec: the address alone decides bank, die, block and page
  (section 4.6, Figure 4; consecutive writes without skipping, section 5.4.1; no page-level FTL,
  section 11.4), so the offset is ``lpa // units``. The data is assumed to be resident.
* ``mqsim`` - MQSim's allocation (``placement``): each plane fills its pages in trace order, at
  every write and at the first read of an LPA that was never written. With ``--prefill`` (whole
  span in address order) it equals ``static``; without it, sparse reads get compact first-touch
  offsets that the spec does not allow (it answers such reads with error 0x7).

A layer larger than ``Config.batch_pages`` is split into sub-batches that MQSim serves one after
another; ``batch_rounds`` sums the rounds of the sub-batches and is the figure to compare with
MQSim, while ``rounds`` treats the whole layer as one burst.
"""
from collections import Counter, defaultdict
from math import ceil


def plane_geometry(cfg):
    """HBF dies and planes per die of a topology.

    With one die and one plane per MQSim chip (the chip-as-plane setup), a channel is one HBF die
    and its chips are that die's planes. Otherwise MQSim's own (channel, chip, die) is the die.
    """
    if cfg.dies == cfg.planes == 1:
        return cfg.channels, cfg.chips
    return cfg.channels * cfg.chips * cfg.dies, cfg.planes


def locate(cfg, lpa, offsets=None):
    """(die, plane in die, offset in plane) of an LPA.

    ``offsets`` maps LPAs to their offset in the plane (see ``placement``). Without it the offset
    is ``lpa // units``, the rank of the LPA among its plane's LPAs when a layout that starts at
    LPA 0 is written in LPA order.
    """
    channel, chip, die, plane = cfg.physical(lpa)
    offset = lpa // cfg.units if offsets is None else offsets[lpa]
    if cfg.dies == cfg.planes == 1:
        return channel, chip, offset
    return (channel * cfg.chips + chip) * cfg.dies + die, plane, offset


def placement(cfg, batches):
    """Offset of every LPA in its plane as MQSim allocates it.

    MQSim takes the next page of the plane's write frontier at every write (out of place) and at
    the first read of an unwritten LPA (online_create_entry_for_reads), in trace order. The
    schedule only accepts prefill writes, which precede the reads of their layer, so these final
    offsets are also the ones that every read sees.
    """
    offsets, used = {}, Counter()
    for batch in batches:
        for req in batch.requests:
            if req.op == "W" or req.lpa not in offsets:
                unit = req.lpa % cfg.units
                offsets[req.lpa] = used[unit]
                used[unit] += 1
    return offsets


def check_group(cfg, group):
    _, per_die = plane_geometry(cfg)
    if type(group) is not int or group < 1 or per_die % group:
        raise ValueError(f"Lock-step group of {group} planes must divide {per_die} planes per die")


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
    best = ceil(pages / total)
    senses = group * sum(map(len, needed.values()))
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


PLACEMENTS = ("static", "mqsim")


def analyze(cfg, batches, groups=(1,), mode="static"):
    """Per-layer rows and per-step totals for each plane organization in ``groups``."""
    for group in groups:
        check_group(cfg, group)
    if mode not in PLACEMENTS:
        raise ValueError(f"placement must be one of {PLACEMENTS}")
    offsets = placement(cfg, batches) if mode == "mqsim" else None
    parts = defaultdict(list)
    for batch in batches:
        part = [req for req in batch.requests if req.op == "R"]
        if part:
            parts[batch.step, batch.layer].append(part)
    keys = ("pages", "rounds", "batch_rounds", "min_rounds", "sensed_pages", "useful_bytes", "burst_bytes")
    layers, steps = [], []
    for group in groups:
        totals = {}
        for (step, layer), chunks in sorted(parts.items()):
            requests = [req for chunk in chunks for req in chunk]
            row = {"step": step, "layer": layer, **analyze_group(cfg, requests, group, offsets)}
            row["subbatches"] = len(chunks)
            row["batch_rounds"] = sum(analyze_group(cfg, chunk, group, offsets)["rounds"] for chunk in chunks)
            layers.append(row)
            total = totals.setdefault(step, {"group": group, "step": step, "planes": row["planes"],
                                             "layers": 0, **dict.fromkeys(keys, 0)})
            total["layers"] += 1
            for key in keys:
                total[key] += row[key]
        for total in totals.values():
            total["utilization"] = total["pages"] / (total["rounds"] * total["planes"])
            total["balance_loss"] = total["rounds"] / total["min_rounds"]
            total["sense_amplification"] = total["sensed_pages"] / total["pages"]
            steps.append(total)
    return layers, steps
