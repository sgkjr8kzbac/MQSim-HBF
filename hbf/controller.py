"""Deduplicate token ranges into page I/O and schedule open-loop sub-batches."""
from collections import Counter, defaultdict
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Request:
    lpa: int
    op: str = "R"
    useful_bytes: int = 4096
    burst_bytes: int = 4096


@dataclass(frozen=True)
class Batch:
    step: int
    layer: int
    subbatch: int
    arrival_ns: int
    requests: tuple[Request, ...]


BURST_BYTES = 64


def union_size(intervals):
    end = total = 0
    for lo, hi in sorted(intervals):
        total += max(0, hi - max(lo, end))
        end = max(hi, end)
    return total


def burst_bytes(intervals, start, token_bytes, burst=BURST_BYTES):
    """Bytes moved when a page read returns only the 64 B bursts covering the wanted tokens.

    The OCP HBF base-die spec v0.7.0 (section 4.1) allows read bursts of 64 B to 4 KiB in
    multiples of 64 B within one NAND page, so the transfer can be smaller than the page while
    the sensing is always a whole page.
    """
    chunks = [((lo - start) * token_bytes // burst, -(-(hi - start) * token_bytes // burst))
              for lo, hi in intervals]
    return union_size(chunks) * burst


def schedule(cfg, layout, accesses):
    logical = defaultdict(dict)
    for access in accesses:
        access.validate(layout)
        if access.op == "W" and access.step != -1:
            raise ValueError("Phase 0 only accepts prefill writes")
        group = logical[access.step, access.layer, access.op]
        t = layout.tensor_index(access.kind)
        tokens = layout.tokens(t)
        for block in range(access.token_start // tokens, (access.token_end - 1) // tokens + 1):
            start = block * tokens
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
            filled = min(start + layout.tokens(t), layout.shape.context) - start
            if op == "W" and useful != filled * layout.token_bytes(t):
                raise ValueError("Partial-page prefill write would cause an update read")
            requests.append(Request(lpa, op, useful, burst_bytes(intervals, start, layout.token_bytes(t))))
        for start in range(0, len(requests), limit):
            serial = serials[step, layer]
            serials[step, layer] += 1
            batches.append(Batch(step, layer, serial, len(batches) * cfg.interval_ns,
                                 tuple(requests[start:start + limit])))
    return batches


def prefill_batches(cfg, layout):
    """Write every LPA of the layout's span once, in address order, as step -1 (layer -1).

    The OCP HBF spec maps an address to a fixed bank, die, block and page and requires the host to
    write a block's 4 KiB addresses consecutively without skipping (section 5.4.1), so the span is
    written whole, padding pages of striped layouts included. MQSim then places page k of each
    plane at offset k, which equals the spec's static offset ``lpa // units`` for a layout at a
    base that is a multiple of the unit count.
    """
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


def unwritten_reads(batches):
    """Reads of LPAs that no earlier batch wrote; the OCP spec answers them with error 0x7."""
    written, count = set(), 0
    for batch in batches:
        count += sum(req.op == "R" and req.lpa not in written for req in batch.requests)
        written.update(req.lpa for req in batch.requests if req.op == "W")
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
    ideal = max(unit_max * cfg.read_ns, byte_count / cfg.aggregate_gb_s)
    return {
        "read_pages": reads, "useful_bytes": useful, "read_amplification": byte_count / useful if useful else None,
        "active_units": len(units), "max_unit_pages": unit_max,
        "unit_imbalance": unit_max * cfg.units / reads if reads else None,
        "max_channel_pages": channel_max, "mq_est_ns": mq, "ideal_ns": ideal, "estimate_gap_ns": mq - ideal,
    }
