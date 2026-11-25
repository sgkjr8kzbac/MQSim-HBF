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
from dataclasses import dataclass
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
                unit = cfg.unit(req.lpa)
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


BURST_BYTES = 64
MOCS_VALUES = (1,) + tuple(2 ** k for k in range(8, 15))  # OCP register: 1 or 256 ... 16,384
READ_UNITS = ("page", "wanted")


@dataclass(frozen=True)
class Protocol:
    """Host protocol limits on read concurrency (OCP HBF v0.7.0).

    * ``mocs`` - Maximum Outstanding Commands Selected: commands the base die command queue of one
      host channel accepts at once (capability register: 1 or a power of two from 256 to 16,384).
    * ``read_burst`` - burst reads supported (product option, BCS bit). Without them every read is
      one 64 B command, so a whole 4 KiB page takes 64 commands (section 5.3.1, item 1); with them
      one command reads up to 64 consecutive 64 B units of a page.
    * ``read_unit`` - ``page`` reads the whole 4 KiB page; ``wanted`` reads only the 64 B units
      that hold wanted tokens (one burst command per contiguous run when bursts are supported).

    The sensing is a whole page either way; later 64 B reads of a sensed page hit the bank's
    cache buffer (section 5.3.1, items 5a and 8), so they cost commands but no extra round.
    """
    mocs: int = 256
    read_burst: bool = False
    read_unit: str = "page"

    def __post_init__(self):
        if self.mocs not in MOCS_VALUES:
            raise ValueError(f"MOCS must be one of {MOCS_VALUES} (OCP capability register)")
        if self.read_unit not in READ_UNITS or type(self.read_burst) is not bool:
            raise ValueError(f"read_unit must be one of {READ_UNITS}")

    def commands(self, cfg, req):
        """Read commands one page request needs."""
        if self.read_unit == "page":
            return 1 if self.read_burst else cfg.page_bytes // BURST_BYTES
        return req.burst_runs if self.read_burst else req.burst_bytes // BURST_BYTES


def protocol_rounds(cfg, reads, group, offsets, protocol):
    """Read rounds when each host channel can hold at most ``mocs`` outstanding commands.

    A round senses at most one offset per lock-step group (one page per bank for ``group = 1``) and,
    per host channel, issues at most ``mocs`` commands; a page's commands are counted in the round
    that senses it, as their credits stay held until the data is returned. Rounds are scheduled
    greedily per host channel, longest group queue first, which reaches the lower bound
    max(longest queue, ceil(commands / mocs)) when every page needs the same number of commands.
    Returns (rounds, credit-bound rounds, total commands).
    """
    queues = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    total = 0
    for req in reads:
        die, plane, offset = locate(cfg, req.lpa, offsets)
        cost = protocol.commands(cfg, req)
        queues[cfg.host_channel(req.lpa)][die, plane // group][offset] += cost
        total += cost
    rounds = credit = 0
    for groups in queues.values():
        pending = {key: sorted(items.items()) for key, items in groups.items()}
        commands = sum(cost for items in pending.values() for _, cost in items)
        credit = max(credit, -(-commands // protocol.mocs))
        count = 0
        while pending:
            budget = protocol.mocs
            issued = False
            for key in sorted(pending, key=lambda k: -len(pending[k])):
                cost = pending[key][0][1]
                if cost <= budget or not issued:  # a lone item may exceed a tiny MOCS
                    budget -= cost
                    issued = True
                    pending[key].pop(0)
                    if not pending[key]:
                        del pending[key]
                if budget <= 0:
                    break
            count += 1
        rounds = max(rounds, count)
    return rounds, credit, total


PLACEMENTS = ("static", "mqsim")


def analyze(cfg, batches, groups=(1,), mode="static", protocol=None):
    """Per-layer rows and per-step totals for each plane organization in ``groups``.

    With a ``Protocol`` the rows also give the read commands, the rounds that the host channels'
    command credits alone force, and the rounds under both bank and credit limits.
    """
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
    if protocol is not None:
        keys += ("commands", "credit_rounds", "protocol_rounds")
    layers, steps = [], []
    for group in groups:
        totals = {}
        for (step, layer), chunks in sorted(parts.items()):
            requests = [req for chunk in chunks for req in chunk]
            row = {"step": step, "layer": layer, **analyze_group(cfg, requests, group, offsets)}
            row["subbatches"] = len(chunks)
            row["batch_rounds"] = sum(analyze_group(cfg, chunk, group, offsets)["rounds"] for chunk in chunks)
            if protocol is not None:
                limited = [protocol_rounds(cfg, chunk, group, offsets, protocol) for chunk in chunks]
                row["commands"] = sum(item[2] for item in limited)
                row["credit_rounds"] = sum(item[1] for item in limited)
                row["protocol_rounds"] = sum(item[0] for item in limited)
                row["protocol_utilization"] = row["pages"] / (row["protocol_rounds"] * row["planes"])
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
            if protocol is not None:
                total["protocol_utilization"] = total["pages"] / (total["protocol_rounds"] * total["planes"])
            steps.append(total)
    return layers, steps
