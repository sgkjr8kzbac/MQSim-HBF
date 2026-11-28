"""MQSim backend batch conversion, unwritten-read accounting, and per-batch load metrics."""
from collections import Counter, defaultdict

from ...device.address import block_page
from ...frontend.pages import Batch, Request


def batches_from_trace(cfg, chunks):
    """Convert an address trace's ``LayerChunk`` sequence into MQSim ``Batch`` objects.

    Release records (``op == 'D'``) do not issue device commands and are skipped. Arrival
    timestamps are assigned strictly increasing by ``cfg.interval_ns`` in ``(step, layer, subbatch)``
    order.
    """
    batches = []
    for chunk in chunks:
        if not len(chunk):
            continue
        ops = chunk.op.tolist()
        sbs = chunk.subbatch.tolist()
        lpas = chunk.lpa.tolist()
        useful = chunk.useful_bytes.tolist()
        burst = chunk.burst_bytes.tolist()
        runs = chunk.burst_runs.tolist()
        grouped = {}
        order = []
        for sb, op, lpa, u, b, r in zip(sbs, ops, lpas, useful, burst, runs):
            if op == "D":
                continue
            if sb not in grouped:
                grouped[sb] = []
                order.append(sb)
            grouped[sb].append(Request(lpa, op, u, b, r))
        for sb in order:
            batches.append(Batch(chunk.step, chunk.layer, sb, len(batches) * cfg.interval_ns,
                                 tuple(grouped[sb])))
    return batches


def unwritten_reads(cfg, batches):
    """Reads of LPAs that are not currently written; the OCP spec answers them with error 0x7
    (Table 12). Tracks written pages per (unit, block) and resets a block to {0} whenever its
    page 0 is written (section 5.4.1 auto-erase). Reads in a batch are checked before writes in
    the same batch update block state.
    """
    written = defaultdict(set)
    count = 0
    for batch in batches:
        for req in batch.requests:
            if req.op == "R":
                unit, block, page = block_page(cfg, req.lpa)
                if page not in written.get((unit, block), ()):
                    count += 1
        for req in batch.requests:
            if req.op == "W":
                unit, block, page = block_page(cfg, req.lpa)
                if page == 0:
                    written[unit, block] = {0}
                else:
                    written[unit, block].add(page)
    return count


def load_counts(cfg, requests):
    units, channels = Counter(), Counter()
    for req in requests:
        if req.op == "R":
            units[cfg.unit(req.lpa)] += 1
            channels[cfg.device_lpa(req.lpa) % cfg.channels] += 1
    return units, channels


def metrics(cfg, requests):
    requests = tuple(requests)
    units, channels = load_counts(cfg, requests)
    reads = sum(units.values())
    useful = sum(req.useful_bytes for req in requests if req.op == "R")
    unit_max = max(units.values(), default=0)
    channel_max = max(channels.values(), default=0)
    byte_count = reads * cfg.page_bytes
    bus_ns = cfg.read_command_ns + cfg.transfer_ns
    cycle = bus_ns + cfg.read_ns + cfg.rb_ns
    mq = max(unit_max * cycle, channel_max * bus_ns) + cfg.host_ns if reads else 0
    if cfg.planes > 1:
        # The estimate charges one read command per page. Multi-plane commands share one command
        # among a die's planes, so it does not apply; the mechanism analysis gives the rounds.
        mq = None
    ideal = max(unit_max * cfg.read_ns, byte_count / cfg.aggregate_gb_s)
    return {
        "read_pages": reads, "useful_bytes": useful, "read_amplification": byte_count / useful if useful else None,
        "active_units": len(units), "max_unit_pages": unit_max,
        "unit_imbalance": unit_max * cfg.units / reads if reads else None,
        "max_channel_pages": channel_max, "mq_est_ns": mq, "ideal_ns": ideal, "estimate_gap_ns": None if mq is None else mq - ideal,
    }
