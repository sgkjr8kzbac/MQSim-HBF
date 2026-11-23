"""Deduplicate token ranges into page I/O and schedule open-loop sub-batches."""
from collections import Counter, defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class Request:
    lpa: int
    op: str = "R"
    useful_bytes: int = 4096


@dataclass(frozen=True)
class Batch:
    step: int
    layer: int
    subbatch: int
    arrival_ns: int
    requests: tuple[Request, ...]


def union_size(intervals):
    end = total = 0
    for lo, hi in sorted(intervals):
        total += max(0, hi - max(lo, end))
        end = max(hi, end)
    return total


def schedule(cfg, layout, accesses):
    logical = defaultdict(dict)
    for access in accesses:
        access.validate(layout.shape)
        if access.op == "W" and access.step != -1:
            raise ValueError("Phase 0 only accepts prefill writes")
        group = logical[access.step, access.layer, access.op]
        for block in range(access.token_start // layout.tokens, (access.token_end - 1) // layout.tokens + 1):
            start = block * layout.tokens
            lpa = layout.lpa(access.layer, access.kv_head, "KV".index(access.kind), block)
            group.setdefault(lpa, []).append((
                max(access.token_start, start), min(access.token_end, start + layout.tokens)
            ))
    batches = []
    serials = Counter()
    limit = cfg.batch_pages
    for (step, layer, op), pages in sorted(logical.items(), key=lambda item: (*item[0][:2], item[0][2] == "R")):
        requests = []
        for lpa, intervals in sorted(pages.items()):
            useful = union_size(intervals) * layout.token_bytes
            if op == "W" and useful != layout.tokens * layout.token_bytes:
                raise ValueError("Partial-page prefill write would cause an update read")
            requests.append(Request(lpa, op, useful))
        for start in range(0, len(requests), limit):
            serial = serials[step, layer]
            serials[step, layer] += 1
            batches.append(Batch(step, layer, serial, len(batches) * cfg.interval_ns,
                                 tuple(requests[start:start + limit])))
    return batches


def load_counts(cfg, requests):
    units, channels = Counter(), Counter()
    for req in requests:
        if req.op == "R":
            units[req.lpa % cfg.units] += 1
            channels[req.lpa % cfg.channels] += 1
    return units, channels


def metrics(cfg, requests):
    requests = tuple(requests)
    units, channels = load_counts(cfg, requests)
    reads = sum(units.values())
    useful = sum(req.useful_bytes for req in requests if req.op == "R")
    unit_max = max(units.values(), default=0)
    channel_max = max(channels.values(), default=0)
    byte_count = reads * cfg.page_bytes
    cycle = cfg.cmd_ns + cfg.read_ns + cfg.rb_ns + cfg.transfer_ns
    mq = max(unit_max * cycle, channel_max * (cfg.cmd_ns + cfg.transfer_ns)) + cfg.host_ns if reads else 0
    ideal = max(unit_max * cfg.read_ns, byte_count / cfg.aggregate_gb_s)
    return {
        "read_pages": reads, "useful_bytes": useful, "read_amplification": byte_count / useful if useful else None,
        "active_units": len(units), "max_unit_pages": unit_max,
        "unit_imbalance": unit_max * cfg.units / reads if reads else None,
        "max_channel_pages": channel_max, "mq_est_ns": mq, "ideal_ns": ideal, "estimate_gap_ns": mq - ideal,
    }
