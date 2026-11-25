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
HOST_ORDERS = ("fcfs", "trigger")
GROUP_SENSES = ("arrived", "whole")
# System read throughput per UCIe module and direction, OCP sections 5.1.2.1-5.1.2.3 (GB/s).
UCIE_GRADES = {1: 48, 2: 94, 3: 188}


@dataclass(frozen=True)
class Protocol:
    """Host protocol limits on read concurrency (OCP HBF v0.7.0).

    * ``mocs`` - Maximum Outstanding Commands Selected of a host channel's base-die command queue
      (capability register of the Base Die UCIe Channel, section 5.9.2.1: 1 or a power of two
      from 256 to 16,384).
    * ``read_burst`` - burst reads supported (product option, BCS bit). Without them every read is
      one 64 B command, so a whole 4 KiB page takes 64 commands (section 5.3.1, item 1); with them
      one command reads a contiguous run of up to 64 units of 64 B inside one page (section 4.1).
    * ``read_unit`` - ``page`` reads the whole 4 KiB page; ``wanted`` reads only the 64 B units
      that hold wanted tokens.
    * ``host_order`` - ``fcfs`` queues all commands of a page at once, as HBF-EventLab's default
      host does, so the admitted ones hold credits while the page is sensed (commands that did not
      fit follow as cache hits); ``trigger`` first sends one 64 B
      read per page to start the sense and the rest after the data reached the bank's cache
      buffer, where they are served at once as cache hits (section 5.3.1, items 5a, 7 and 8).
      The spec allows both; ``trigger`` needs only one credit per page during the sense.
    * ``group_sense`` - lock-step groups only (``group > 1``), a modelling assumption that neither
      the spec nor HBF-EventLab settles: ``arrived`` (default) senses the pages of an offset whose
      commands have arrived, so pages whose commands did not fit wait for another group sense of
      that offset; ``whole`` senses every plane of the group at that offset as soon as one command
      arrives (a megaplane), so the other pages are cache hits and a ``trigger`` host needs one
      command per offset. With independent planes both give the same rounds.
    * ``ucie_grade`` - optional speed grade 1-3 for a link bound: a host channel moves at most
      48, 94 or 188 GB/s of read data, so a round of tR carries at most that much.
    """
    mocs: int = 256
    read_burst: bool = False
    read_unit: str = "page"
    host_order: str = "fcfs"
    ucie_grade: int | None = None
    group_sense: str = "arrived"

    def __post_init__(self):
        if self.mocs not in MOCS_VALUES:
            raise ValueError(f"MOCS must be one of {MOCS_VALUES} (OCP capability register)")
        if self.read_unit not in READ_UNITS or type(self.read_burst) is not bool:
            raise ValueError(f"read_unit must be one of {READ_UNITS}")
        if self.host_order not in HOST_ORDERS:
            raise ValueError(f"host_order must be one of {HOST_ORDERS}")
        if self.group_sense not in GROUP_SENSES:
            raise ValueError(f"group_sense must be one of {GROUP_SENSES}")
        if self.ucie_grade is not None and self.ucie_grade not in UCIE_GRADES:
            raise ValueError("ucie_grade must be 1, 2 or 3 (OCP section 5.1.2)")

    def commands(self, cfg, req):
        """Read commands one page request needs in total."""
        if self.read_unit == "page":
            return 1 if self.read_burst else cfg.page_bytes // BURST_BYTES
        return req.burst_runs if self.read_burst else req.burst_bytes // BURST_BYTES

    def sensing_credits(self, cfg, req):
        """Credits the page holds while its sense is pending."""
        return 1 if self.host_order == "trigger" else self.commands(cfg, req)

    def read_bytes(self, cfg, req):
        return cfg.page_bytes if self.read_unit == "page" else req.burst_bytes


def protocol_rounds(cfg, reads, group, offsets, protocol):
    """Read rounds when each host channel can hold at most ``mocs`` outstanding commands.

    A round senses at most one offset per lock-step group (one page per bank for ``group = 1``) and,
    per host channel, admits at most ``mocs`` credits. A page whose commands are admitted, even only
    some of them, is sensed in that round and takes min(its sensing credits, credits left); the
    commands that did not fit enter after the sense and are served from the bank's cache buffer
    (section 5.3.1, items 5a and 7), which takes no sensing round. HBF-EventLab confirms it for
    single pages: with MOCS = 1, eight pages of one host channel took eight senses and 504 cache
    hits (33.2 us), not 512 rounds; with MOCS 256, seven pages of 40 commands each took one wave.
    Lock-step groups follow ``Protocol.group_sense``. Cache hits are taken as free: their round
    trips (63 per page at MOCS 1) are not counted.

    Rounds are scheduled greedily per host channel, longest group queue first: with equal credits
    ``c`` per item this reaches max(longest queue, ceil(items / ceil(MOCS / c))); otherwise it is an
    achievable upper bound. The model treats a round as one tR and releases all credits at its end,
    so it is approximate: HBF-EventLab, which frees each credit when its response is delivered,
    fits slightly more pages per wave under ``fcfs``.

    Returns (rounds, credit-bound rounds, total commands, link-bound rounds). The credit bound is
    ceil(units / k) for the units a credit admits (pages, or whole offsets with ``whole``), where k
    is one more than the number of the cheapest units whose credits sum to less than MOCS.
    """
    whole = group > 1 and protocol.group_sense == "whole"
    queues = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    commands = 0
    data = defaultdict(int)
    for req in reads:
        die, plane, offset = locate(cfg, req.lpa, offsets)
        host = cfg.host_channel(req.lpa)
        queues[host][die, plane // group][offset].append(
            protocol.commands(cfg, req) if whole else protocol.sensing_credits(cfg, req))
        commands += protocol.commands(cfg, req)
        data[host] += protocol.read_bytes(cfg, req)
    rounds = credit = 0
    for groups in queues.values():
        for items in groups.values():
            for offset, pages in items.items():
                if whole:  # one sense covers the offset: a single unit (one trigger command)
                    items[offset] = [1 if protocol.host_order == "trigger" else sum(pages)]
        costs = sorted(cost for items in groups.values() for pages in items.values() for cost in pages)
        spent = per_round = 0
        while per_round < len(costs) and spent < protocol.mocs:
            spent += costs[per_round]
            per_round += 1
        credit = max(credit, -(-len(costs) // per_round))
        pending = {key: [list(pages) for _, pages in sorted(items.items())] for key, items in groups.items()}
        count = 0
        while pending:
            budget = protocol.mocs
            for key in sorted(pending, key=lambda k: -len(pending[k])):
                if budget == 0:
                    break
                pages = pending[key][0]  # this round senses the head offset of the group
                while pages and budget:
                    budget -= min(pages.pop(0), budget)  # sensed; commands that did not fit are cache hits
                if not pages:
                    pending[key].pop(0)
                    if not pending[key]:
                        del pending[key]
            count += 1
        rounds = max(rounds, count)
    link = 0
    if protocol.ucie_grade is not None and data:
        per_round = UCIE_GRADES[protocol.ucie_grade] * cfg.read_ns  # GB/s x ns = bytes
        link = -(-max(data.values()) // per_round)
    return rounds, credit, commands, link


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
        keys += ("commands", "credit_rounds", "link_rounds", "protocol_rounds")
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
                row["link_rounds"] = sum(item[3] for item in limited)
                row["protocol_rounds"] = sum(max(item[0], item[3]) for item in limited)
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
