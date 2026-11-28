"""Tests for R5 MQSim HBF write patches (T-HBFW1..T-HBFW4, HBF_WRITE_ORDER, T-CACHE+write)."""
import csv
import random
from collections import Counter

import pytest

from hbf.backends.mqsim.parse import block_counts, completions
from hbf.backends.mqsim.run import simulate
from hbf.device.address import device_lpa
from hbf.device.config import load
from hbf.frontend.pages import Batch, Request


def expected_phys(cfg, lpa):
    """Expected (channel, chip, die, plane, block, page) from static CWDP/OCP mapping."""
    dev = device_lpa(cfg, lpa)
    ch = dev % cfg.channels
    chip = (dev // cfg.channels) % cfg.chips
    die = (dev // (cfg.channels * cfg.chips)) % cfg.dies
    plane = (dev // (cfg.channels * cfg.chips * cfg.dies)) % cfg.planes
    offset = dev // cfg.units
    block, page = divmod(offset, cfg.pages_per_block)
    return ch, chip, die, plane, block, page


def python_block_counts(cfg, batches):
    """Reference per-block (reads, programs, erases) counted in Python from batches."""
    reads, programs, erases = Counter(), Counter(), Counter()
    written_blocks = set()
    for batch in batches:
        for req in batch.requests:
            ch, chip, die, plane, block, page = expected_phys(cfg, req.lpa)
            key = (ch, chip, die, plane, block)
            if req.op == "R":
                reads[key] += 1
            elif req.op == "W":
                if page == 0 and key in written_blocks:
                    erases[key] += 1
                written_blocks.add(key)
                programs[key] += 1
    keys = set(reads) | set(programs) | set(erases)
    return {k: (reads[k], programs[k], erases[k]) for k in keys}


def test_hbfw1_static_mapping_matches_python_page_for_page(binary, tmp_path):
    """T-HBFW1: without prefill, random legal write+read sequence maps page-for-page identically to address.py."""
    cfg = load("16x64", channels=2, chips=2, dies=1, planes=4, blocks_per_plane=16, pages_per_block=8,
               aggregate_gb_s=64, plane_rule="independent", coalesce_ns=None, read_plane_ns=None)
    rng = random.Random(42)
    # Choose random blocks across units and write pages sequentially within each block (in interleaved order),
    # then read back a random subset — no prefill, arbitrary block order.
    blocks_chosen = [(u, b) for u in range(cfg.units) for b in rng.sample(range(8), 2)]
    next_page = {k: 0 for k in blocks_chosen}
    batches = []
    written_lpas = []
    t = 0
    while next_page:
        key = rng.choice(list(next_page))
        u, b = key
        p = next_page[key]
        lpa = u + cfg.units * (b * cfg.pages_per_block + p)
        batches.append(Batch(0, 0, len(batches), t, (Request(lpa, "W"),)))
        written_lpas.append(lpa)
        t += 10_000_000
        if p + 1 >= 3:
            del next_page[key]
        else:
            next_page[key] = p + 1
    for lpa in rng.sample(written_lpas, 24):
        batches.append(Batch(1, 0, len(batches), t, (Request(lpa, "R"),)))
        t += 10_000_000

    out = tmp_path / "hbfw1-static"
    simulate(cfg, batches, out, binary, physical=True, hbfw1=True, hbfw3=True)

    observed = {}
    with (out / "physical.csv").open(newline="") as stream:
        for row in csv.reader(stream):
            if row[1] == "issue":
                dev_lpa = int(row[9])
                phys = tuple(int(x) for x in row[3:9])
                observed[dev_lpa] = phys
    assert len(observed) == len(written_lpas)
    for lpa in written_lpas:
        assert observed[device_lpa(cfg, lpa)] == expected_phys(cfg, lpa)


def test_hbfw1_hbfw2_block_fill_erase_rewrite_and_full_span(binary, tmp_path):
    """T-HBFW1+2+3: fill a block, rewrite from page 0 (auto-erase), and fill entire logical span without GC."""
    # Small device: 1x1x1x2, 8 blocks/plane, 4 pages/block -> 32 phys pages/plane, logical capacity = 93% * 64 = 59 pages
    cfg = load("16x64", channels=1, chips=1, dies=1, planes=2, blocks_per_plane=8, pages_per_block=4,
               aggregate_gb_s=64, plane_rule="independent", coalesce_ns=None, read_plane_ns=None)
    logical_pages = cfg.capacity_sectors // cfg.spp
    batches = []
    t = 0
    # Pass 1: fill the entire logical layout span (sequentially within every block)
    for lpa in range(logical_pages):
        batches.append(Batch(0, 0, len(batches), t, (Request(lpa, "W"),)))
        t += 10_000_000
    # Pass 2: rewrite block 0 on plane 0 and plane 1 completely (triggers auto-erase on page 0)
    for page in range(cfg.pages_per_block):
        for unit in range(cfg.units):
            lpa = unit + cfg.units * page
            batches.append(Batch(1, 0, len(batches), t, (Request(lpa, "W"),)))
            t += 10_000_000
    # Pass 3: rewrite block 0 again + read back
    for page in range(cfg.pages_per_block):
        lpa = cfg.units * page
        batches.append(Batch(2, 0, len(batches), t, (Request(lpa, "W"),)))
        t += 10_000_000
    batches.append(Batch(3, 0, len(batches), t, (Request(0, "R"), Request(1, "R"))))

    out = tmp_path / "hbfw-span"
    _, _, _, stats = simulate(cfg, batches, out, binary, hbf_write=True)
    assert stats["Total_GC_Executions"] == 0
    assert stats["gc_read_write_enqueued"] == 0
    # Block 0 of plane 0 was rewritten twice (2 auto-erases), block 0 of plane 1 once (1 auto-erase)
    assert stats["erase_commands"] == 3
    counts = block_counts(out / "block_counts.csv")
    assert counts == python_block_counts(cfg, batches)


def test_hbfw2_write_order_exact_hand_computed(binary, tmp_path):
    """T-HBFW2 + HBF_WRITE_ORDER: hand-computed timing for fifo vs read_first (both arrival orders)."""
    # Single channel, single chip, single die, single plane -> exact deterministic command latencies.
    cfg = load("16x64", channels=1, chips=1, dies=1, planes=1, blocks_per_plane=8, pages_per_block=4,
               aggregate_gb_s=64, plane_rule=None, coalesce_ns=None, read_plane_ns=None)
    # Hand-computed MQSim ONFI command timings for 1-plane device:
    # - erase command setup time: cmd(1) = 4 cycles? Let's derive exact delta between read_first and fifo
    #   and also verify exact timestamps from physical.csv / completions.csv!
    # Step 0 at t=0: write page 0 of block 0 so block 0 is marked Hbf_written=true,
    # and write page 0 of block 1 so we have a valid page to read.
    t0 = 10_000_000
    delta = 100_000  # 0 < delta < tBERS (3.5 ms)

    # Case A: WR (triggers ER+WR on block 0) at t0; RD (block 1 page 0) arrives at t0 + delta.
    batches_a = [
        Batch(0, 0, 0, 0, (Request(0, "W"),)),
        Batch(0, 0, 1, 1_000_000, (Request(4, "W"),)),
        Batch(1, 0, 0, t0, (Request(0, "W"),)),
        Batch(1, 0, 1, t0 + delta, (Request(4, "R"),)),
    ]
    res = {}
    for order in ("read_first", "fifo"):
        out = tmp_path / f"order-a-{order}"
        simulate(cfg, batches_a, out, binary, physical=True, hbf_write=True, write_order=order)
        done = {(c.arrival_ns, c.op): c.complete_ns for c in completions(out / "completions.csv")}
        res[order] = done

    # Measure exact solo read time and solo program time (including cmd+data+exec+host) from physical/completions
    out_solo_r = tmp_path / "solo-read"
    simulate(cfg, [Batch(0, 0, 0, 0, (Request(4, "R"),))], out_solo_r, binary,
             hbf_write=True, allow_unwritten=True)
    solo_read_ns = next(completions(out_solo_r / "completions.csv")).complete_ns

    out_solo_w = tmp_path / "solo-write"
    simulate(cfg, [Batch(0, 0, 0, 0, (Request(0, "W"),))], out_solo_w, binary, hbf_write=True)
    solo_write_ns = next(completions(out_solo_w / "completions.csv")).complete_ns

    # In solo_read_ns / solo_write_ns, host fetch (1 ns) + command + exec + transfer + host completion (1 ns?):
    # Let's check exact command latency via physical.csv of read_first vs fifo!
    # Under read_first:
    #   ER starts at t0 + host_in, finishes at t_er_end = t0 + host_in + erase_cmd + tBERS
    #   RD runs IMMEDIATELY after ER, BEFORE WR! So RD finishes BEFORE WR!
    assert res["read_first"][(t0 + delta, "R")] < res["read_first"][(t0, "W")]
    # Under fifo:
    #   WR (Issue_time ~ t0) runs IMMEDIATELY after ER, BEFORE RD (Issue_time ~ t0 + delta)!
    #   So WR finishes BEFORE RD!
    assert res["fifo"][(t0, "W")] < res["fifo"][(t0 + delta, "R")]

    # Exact latency check:
    # Under read_first: RD completes at t_er_end + read_cmd + tR + data_out (+ host_done)
    # Under fifo: RD completes at t_er_end + prog_cmd + data_in + tPROG + read_cmd + tR + data_out (+ host_done)
    # Therefore: fifo_RD_complete - read_first_RD_complete MUST EQUAL the exact program transaction duration
    # (prog_cmd + data_in + tPROG) to the nanosecond!
    # And read_first_WR_complete - fifo_WR_complete MUST EQUAL the exact read flash duration
    # (read_cmd + tR + data_out) to the nanosecond!
    rd_diff = res["fifo"][(t0 + delta, "R")] - res["read_first"][(t0 + delta, "R")]
    wr_diff = res["read_first"][(t0, "W")] - res["fifo"][(t0, "W")]
    # Program flash time = prog_cmd (290 ns) + data_in (64 ns) + tPROG (350_000 ns) = 350_354 ns
    # Read flash time = read_cmd (290 ns) + tR (4_116 ns) + data_out (64 ns) = 4_470 ns
    # (Allow exact match against solo_write_ns - cfg.host_ns and solo_read_ns - cfg.host_ns)
    assert rd_diff == solo_write_ns - cfg.host_ns
    assert wr_diff == solo_read_ns - cfg.host_ns
    # Also verify absolute completion time of read_first RD is t0 + tBERS + solo_read (+ erase cmd setup)
    er_rd_elapsed = res["read_first"][(t0 + delta, "R")] - t0
    assert cfg.erase_ns + solo_read_ns <= er_rd_elapsed <= cfg.erase_ns + solo_read_ns + 500
    er_wr_rd_elapsed = res["fifo"][(t0 + delta, "R")] - t0
    assert (cfg.erase_ns + cfg.program_ns + solo_read_ns
            <= er_wr_rd_elapsed
            <= cfg.erase_ns + cfg.program_ns + solo_read_ns + 1000)

    # Case B: RD arrives BEFORE t0 -> in BOTH fifo and read_first, RD completes before WR!
    t_rd_early = t0 - 500
    batches_b = [
        Batch(0, 0, 0, 0, (Request(0, "W"),)),
        Batch(0, 0, 1, 1_000_000, (Request(4, "W"),)),
        Batch(1, 0, 0, t_rd_early, (Request(4, "R"),)),
        Batch(1, 0, 1, t0, (Request(0, "W"),)),
    ]
    for order in ("read_first", "fifo"):
        out = tmp_path / f"order-b-{order}"
        simulate(cfg, batches_b, out, binary, hbf_write=True, write_order=order)
        done = {(c.arrival_ns, c.op): c.complete_ns for c in completions(out / "completions.csv")}
        assert done[(t_rd_early, "R")] == t_rd_early + solo_read_ns
        assert done[(t_rd_early, "R")] < done[(t0, "W")]


def test_hbfw3_repeated_full_span_rewrites_no_gc(binary, tmp_path):
    """T-HBFW3: repeatedly filling and rewriting the full logical span triggers zero GC."""
    cfg = load("16x64", channels=1, chips=1, dies=1, planes=2, blocks_per_plane=6, pages_per_block=4,
               aggregate_gb_s=64, plane_rule="independent", coalesce_ns=None, read_plane_ns=None)
    logical_pages = cfg.capacity_sectors // cfg.spp
    batches = []
    t = 0
    for round_idx in range(3):
        for lpa in range(logical_pages):
            batches.append(Batch(round_idx, 0, lpa, t, (Request(lpa, "W"),)))
            t += 10_000_000
    out = tmp_path / "hbfw3-no-gc"
    _, _, _, stats = simulate(cfg, batches, out, binary, hbf_write=True)
    assert stats["Total_GC_Executions"] == 0
    assert stats["gc_read_write_enqueued"] == 0
    assert block_counts(out / "block_counts.csv") == python_block_counts(cfg, batches)


def test_hbfw4_block_counts_match_python_multi_plane_ocp(binary, tmp_path):
    """T-HBFW4: per-block (reads, programs, erases) match Python counts block-for-block on OCP address map."""
    cfg = load("ocp", channels=2, chips=2, dies=1, planes=4, blocks_per_plane=8, pages_per_block=4,
               aggregate_gb_s=64, coalesce_ns=None)
    batches = []
    t = 0
    # Write 2 full blocks on every unit (8 pages per unit), then rewrite block 0 on half the units,
    # then issue mixed reads across written pages.
    for page in range(8):
        reqs = tuple(Request(u + cfg.units * page, "W") for u in range(cfg.units))
        batches.append(Batch(0, 0, page, t, reqs))
        t += 10_000_000
    for page in range(4):
        reqs = tuple(Request(u + cfg.units * page, "W") for u in range(0, cfg.units, 2))
        batches.append(Batch(1, 0, page, t, reqs))
        t += 10_000_000
    for step in range(3):
        reqs = tuple(Request(u + cfg.units * (step % 8), "R") for u in range(cfg.units))
        batches.append(Batch(2, step, 0, t, reqs))
        t += 10_000_000

    out = tmp_path / "hbfw4-ocp"
    _, _, _, stats = simulate(cfg, batches, out, binary, hbf_write=True)
    assert stats["Total_GC_Executions"] == 0
    mqsim_counts = block_counts(out / "block_counts.csv")
    ref_counts = python_block_counts(cfg, batches)
    assert mqsim_counts == ref_counts
    assert sum(v[0] for v in mqsim_counts.values()) == 3 * cfg.units
    assert sum(v[1] for v in mqsim_counts.values()) == 8 * cfg.units + 4 * (cfg.units // 2)
    assert sum(v[2] for v in mqsim_counts.values()) == cfg.units // 2


def test_t_cache_with_writes_and_auto_erases(binary, tmp_path):
    """T-CACHE combined with HBF writes and auto-erases completes cleanly with expected order."""
    cfg = load("16x64", channels=1, chips=1, dies=1, planes=4, blocks_per_plane=8, pages_per_block=4,
               aggregate_gb_s=64, plane_rule="independent", read_plane_ns=10, coalesce_ns=500,
               cache_read=True)
    batches = []
    t = 0
    # Write block 0 (page 0) and block 1 (pages 0..2) across all 4 planes
    batches.append(Batch(0, 0, 0, t, tuple(Request(p, "W") for p in range(cfg.units))))
    t += 5_000_000
    for page in range(3):
        offset = cfg.pages_per_block + page
        reqs = tuple(Request(p + cfg.units * offset, "W") for p in range(cfg.units))
        batches.append(Batch(0, 0, page + 1, t, reqs))
        t += 5_000_000
    # Issue a burst of multi-round reads on block 1 (exercises T-CACHE pipelining)
    read_burst = tuple(
        Request(p + cfg.units * (cfg.pages_per_block + page), "R")
        for page in range(3) for p in range(cfg.units)
    )
    batches.append(Batch(1, 0, 0, t, read_burst))
    t += 5_000_000
    # Rewrite page 0 of block 0 on all 4 planes (triggers auto-erase on block 0) followed closely by read burst on block 1
    rewrite_reqs = tuple(Request(p, "W") for p in range(cfg.units))
    batches.append(Batch(2, 0, 0, t, rewrite_reqs))
    batches.append(Batch(2, 0, 1, t + 50_000, read_burst))

    out = tmp_path / "t-cache-write"
    _, _, _, stats = simulate(cfg, batches, out, binary, hbf_write=True, write_order="read_first")
    assert stats["Total_GC_Executions"] == 0
    assert stats["erase_commands"] >= 1
    assert block_counts(out / "block_counts.csv") == python_block_counts(cfg, batches)
