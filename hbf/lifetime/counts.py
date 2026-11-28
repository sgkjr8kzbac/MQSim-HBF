"""Compute endurance rates and block-level comparisons from MQSim T-HBFW4 counts (R6b)."""
from collections import Counter
import math

from ..device.address import device_lpa
from .geometry import DAY, sci


def _phys_block(cfg, lpa):
    """Return ``(channel, chip, die, plane, block, page)`` matching MQSim T-HBFW1/T-HBFW4."""
    dev = device_lpa(cfg, lpa)
    ch = dev % cfg.channels
    chip = (dev // cfg.channels) % cfg.chips
    die = (dev // (cfg.channels * cfg.chips)) % cfg.dies
    plane = (dev // (cfg.channels * cfg.chips * cfg.dies)) % cfg.planes
    offset = dev // cfg.units
    block, page = divmod(offset, cfg.pages_per_block)
    return (ch, chip, die, plane, block), page


def counts_from_trace(cfg, trace):
    """Count per-block ``(reads, programs, erases)`` and per-class page ops directly from an ``AddressTrace``."""
    reads, programs, erases = Counter(), Counter(), Counter()
    written_blocks = set()
    by_class = Counter()
    for chunk in trace.chunks:
        if not len(chunk):
            continue
        for op, lpa, cls_name in zip(chunk.op.tolist(), chunk.lpa.tolist(), chunk.cls.tolist()):
            if op == "D":
                by_class[("D", cls_name)] += 1
                continue
            key, page = _phys_block(cfg, lpa)
            by_class[(op, cls_name)] += 1
            if op == "R":
                reads[key] += 1
            elif op == "W":
                if page == 0 and key in written_blocks:
                    erases[key] += 1
                written_blocks.add(key)
                programs[key] += 1
    keys = set(reads) | set(programs) | set(erases)
    block_map = {k: (reads[k], programs[k], erases[k]) for k in keys}
    return block_map, dict(by_class)


def compare_block_counts(mqsim_counts, python_counts):
    """Compare MQSim ``T-HBFW4`` block counts against Python reference counts block-for-block."""
    all_keys = sorted(set(mqsim_counts) | set(python_counts))
    mismatches = []
    for k in all_keys:
        mv = mqsim_counts.get(k, (0, 0, 0))
        pv = python_counts.get(k, (0, 0, 0))
        if mv != pv:
            mismatches.append((k, mv, pv))
    return {
        "exact_match": len(mismatches) == 0,
        "blocks_compared": len(all_keys),
        "mismatched_blocks": len(mismatches),
        "sample_mismatches": mismatches[:5],
        "mqsim_totals": tuple(sum(v[i] for v in mqsim_counts.values()) for i in range(3)),
        "python_totals": tuple(sum(v[i] for v in python_counts.values()) for i in range(3)),
    }


def rates_from_counts(counts, *, total_blocks, pages_per_block, page_bytes, duration_s,
                      kv_programs=None, refresh_programs=None, pe_limits=(1e4, 1e5, 1e6)):
    """Derive write/erase rates and daily P/E cycles from per-block ``(reads, programs, erases)`` counts.

    Reports both raw flash erases (``avg_pec_per_day`` from ``T-HBFW4`` ``erases``, where fresh
    blocks start erased) and open-counted P/E cycles (``avg_pec_opened_per_day`` = ``erases +
    blocks_written``, matching ``Simulator._open``'s charge-on-open convention when ``warmup == 0``).
    """
    if total_blocks < 1 or pages_per_block < 1 or duration_s <= 0:
        raise ValueError("total_blocks, pages_per_block and duration_s must be positive")
    total_reads = sum(v[0] for v in counts.values())
    total_programs = sum(v[1] for v in counts.values())
    total_erases = sum(v[2] for v in counts.values())
    blocks_written = sum(1 for v in counts.values() if v[1] > 0)
    max_erases = max((v[2] for v in counts.values()), default=0)
    max_opened = max((v[2] + (1 if v[1] > 0 else 0) for v in counts.values()), default=0)
    days = duration_s / DAY
    avg_pec_day = (total_erases / total_blocks) / days
    max_pec_day = max_erases / days
    avg_opened_day = ((total_erases + blocks_written) / total_blocks) / days
    max_opened_day = max_opened / days
    avg_prog_day = (total_programs / (total_blocks * pages_per_block)) / days

    row = {
        "total_reads": total_reads,
        "total_programs": total_programs,
        "total_erases": total_erases,
        "blocks_written": blocks_written,
        "total_blocks": total_blocks,
        "duration_s": duration_s,
        "program_gb_s": total_programs * page_bytes / duration_s / 1e9,
        "erases_per_s": total_erases / duration_s,
        "avg_pec_per_day": avg_pec_day,
        "max_pec_per_day": max_pec_day,
        "avg_pec_opened_per_day": avg_opened_day,
        "max_pec_opened_per_day": max_opened_day,
        "avg_pec_from_programs_per_day": avg_prog_day,
        "refresh_to_kv": (refresh_programs / kv_programs) if (kv_programs and refresh_programs is not None) else None,
    }
    for limit in pe_limits:
        row[f"days_to_limit_avg@{sci(limit)}"] = limit / avg_pec_day if avg_pec_day > 0 else math.inf
        row[f"days_to_limit_opened@{sci(limit)}"] = limit / avg_opened_day if avg_opened_day > 0 else math.inf
    return row
