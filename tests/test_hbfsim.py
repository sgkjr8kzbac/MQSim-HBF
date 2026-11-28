"""Acceptance and unit tests for the HBFSim backend (Milestone R9).

Covers:
1. Build provenance & config loading (`hbfsim.commit`, `hbfsim-ocp.cfg`, `meta.json`).
2. Step 4a: Read from initial physical image succeeds; read of unwritten page fails with status 0x7.
3. Step 4b: Sequential fill of a block + rewrite of page 0 increments P/E count by 1;
   out-of-order write returns status 0x6.
4. Step 4c: Random LPAs match `hbf/device/address.py` coordinates page-for-page via both
   HBFSim's HBF-physical address heatmap and decoded physical paths.
5. Step 4d: `hbf/backends/hbfsim` never cross-imports `mqsim` or `rounds`.
6. Step 6: Die-blocking switch (`hbf-die-blocking=false` vs `true`):
   - OFF: read to another plane on the same die is NOT blocked by program/erase.
   - ON: read to another plane on the same die IS blocked by `tPROG` and `tBERS`.
"""
from __future__ import annotations

import ast
import json
import random
from pathlib import Path

import numpy as np
import pytest

from hbf.backends.hbfsim import (
    HBFSIM_OCP_CFG,
    PIN,
    HbfSimSession,
    build,
    convert_trace,
    decode_physical_address,
    encode_physical_address,
    hbfsim_geometry,
    parse_live_heatmap,
    physical_coords,
    physical_coords_array,
    simulate_trace,
    write_config,
)
from hbf.backends.hbfsim.run import HbfSimError
from hbf.device.address import place
from hbf.device.config import ROOT, load
from hbf.frontend.addrtrace import AddressTrace, LayerChunk, make_meta


@pytest.fixture(scope="module")
def hbfsim_bin() -> Path:
    return build(target="hbfsim")


@pytest.fixture(scope="module")
def small_ocp_cfg():
    # 1 host channel x 4 dies x 16 planes = 64 planes, 8 blocks/plane, 8 pages/block
    return load(
        "ocp",
        channels=2,
        chips=2,
        dies=1,
        planes=16,
        blocks_per_plane=8,
        pages_per_block=8,
        aggregate_gb_s=100,
        read_ns=4096,
        program_ns=100_000,
        erase_ns=1_000_000,
    )


def test_hbfsim_ocp_cfg_parses_with_describe_system(hbfsim_bin):
    """`hbf/configs/hbfsim-ocp.cfg` resolves cleanly via `hbfsim --describe-system`."""
    import subprocess

    assert HBFSIM_OCP_CFG.is_file()
    assert len(PIN) == 40
    proc = subprocess.run(
        [
            str(hbfsim_bin),
            "--describe-system",
            "--system-config",
            str(HBFSIM_OCP_CFG),
            "--enable-hbm",
            "false",
            "--enable-hbf",
            "true",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    info = json.loads(proc.stdout.strip())
    vals = info["values"]
    assert info["schema"]["name"] == "hbfsim.resolved_system"
    assert vals["hbf-mapping-mode"] == "raw-physical"
    total_planes = (
        int(vals["hbf-stacks"])
        * int(vals["hbf-channels"])
        * int(vals["hbf-dies-per-channel"])
        * int(vals["hbf-planes-per-die"])
    )
    assert total_planes == 1024
    assert int(vals["hbf-blocks-per-plane"]) == 512
    assert int(vals["hbf-pages-per-block"]) == 256
    assert int(vals["hbf-page-size"]) == 4096


def test_initial_image_read_and_unwritten_page_error(hbfsim_bin, small_ocp_cfg, tmp_path):
    """Step 4a: pre-populated physical page reads cleanly; unwritten page fails with status 0x7."""
    cfg_path = tmp_path / "small.cfg"
    geom = write_config(small_ocp_cfg, cfg_path)

    # Block 0 on every plane is static pre-populated; blocks 1..7 are raw-physical (erased).
    page_in_static = encode_physical_address(geom, 0, 0, 0, 0, block=0, page=0)
    page_unwritten = encode_physical_address(geom, 0, 0, 0, 0, block=1, page=0)

    with HbfSimSession(
        cfg_path,
        tmp_path / "run",
        binary=hbfsim_bin,
        enable_hbm=False,
        enable_hbf=True,
        static_blocks_per_plane=1,
        published_blocks_per_plane=geom.blocks_per_plane - 1,
    ) as session:
        ok = session.run_batch(
            "read-static",
            [("r0", "HBF_PHYSICAL", "R", page_in_static, 4096, 0.0)],
            completions=True,
        )
        assert ok["result"] == "pass"
        assert ok["device_delta"]["hbf"]["page_reads"] == 1
        assert ok["elapsed_ns"] > small_ocp_cfg.read_ns

        with pytest.raises(HbfSimError, match="0x7"):
            session.run_batch(
                "read-unwritten",
                [("r1", "HBF_PHYSICAL", "R", page_unwritten, 4096, 0.0)],
            )


def test_sequential_fill_rewrite_pe_and_out_of_order_0x6(hbfsim_bin, small_ocp_cfg, tmp_path):
    """Step 4b: filling pages 0..PPB-1 then rewriting page 0 increments P/E by 1;
    out-of-order write returns status 0x6."""
    cfg_path = tmp_path / "seq.cfg"
    geom = write_config(small_ocp_cfg, cfg_path, initial_free_blocks_erased=True)
    ppb = geom.pages_per_block

    with HbfSimSession(
        cfg_path,
        tmp_path / "run",
        binary=hbfsim_bin,
        enable_hbm=False,
        enable_hbf=True,
        published_blocks_per_plane=geom.blocks_per_plane,
    ) as session:
        # 1. Out-of-order write (page 2 before page 0) via ZONE_WRITE_BATCH -> exact OCP status 0x6!
        bad_addr = encode_physical_address(geom, 0, 0, 0, 0, block=0, page=2)
        ooo = session.zone_write_batch("ooo-test", [("bad0", 0.0, bad_addr, 4096)])
        assert ooo["commands"][0]["status"] == 0x6
        assert ooo["commands"][0]["additional_status"] == 0x1

        # 2. Sequentially write all pages 0..PPB-1 of block 0 on (stack=0, ch=0, die=0, plane=0).
        # With initial_free_blocks_erased=True, the initial fill incurs 0 erases.
        fill_txs = [
            (
                f"w{p}",
                "HBF_PHYSICAL",
                "W",
                encode_physical_address(geom, 0, 0, 0, 0, block=0, page=p),
                4096,
                float(p * 200_000),
            )
            for p in range(ppb)
        ]
        fill_res = session.run_batch("fill-block0", fill_txs)
        assert fill_res["device_delta"]["hbf"]["raw_physical_programs"] == ppb
        wear_before = session.wear_snapshot("after-fill")
        assert wear_before["block_erase_counts"][0] == 0
        assert wear_before["workload_erases"] == 0

        # 3. Host invalidates and resets zone 0 (no media erase yet), then rewrites page 0 ->
        # triggers page-0 auto-erase and increments P/E count of block 0 by exactly 1!
        session.zone_command("ZONE_INVALIDATE", "inv0", stack=0, channel=0, zone=0)
        session.zone_command("ZONE_RESET", "rst0", stack=0, channel=0, zone=0)
        wear_after_reset = session.wear_snapshot("after-reset")
        assert wear_after_reset["block_erase_counts"][0] == 0

        p0_addr = encode_physical_address(geom, 0, 0, 0, 0, block=0, page=0)
        rewrite = session.run_batch(
            "rewrite-p0",
            [("rw0", "HBF_PHYSICAL", "W", p0_addr, 4096, 0.0)],
        )
        assert rewrite["device_delta"]["hbf"]["block_erases"] == 1
        assert rewrite["elapsed_ns"] >= small_ocp_cfg.erase_ns + small_ocp_cfg.program_ns

        wear_after = session.wear_snapshot("after-rewrite")
        assert wear_after["block_erase_counts"][0] == 1
        assert wear_after["workload_erases"] == 1


def test_random_lpas_match_address_py_via_heatmap_and_paths(hbfsim_bin, small_ocp_cfg, tmp_path):
    """Step 4c: random LPAs match `hbf/device/address.py` page-for-page via HBFSim's
    HBF-physical address heatmap and decoded physical resource paths."""
    rng = random.Random(2026)
    geom = hbfsim_geometry(small_ocp_cfg)
    max_lpa = small_ocp_cfg.capacity_sectors // small_ocp_cfg.spp
    sample_lpas = sorted(rng.sample(range(max_lpa), 128))
    lpa_arr = np.array(sample_lpas, dtype=np.int64)

    # Verify scalar vs vectorized coordinate agreement with hbf/device/address.py.
    placed = place(small_ocp_cfg, lpa_arr)
    s_arr, c_arr, d_arr, p_arr, b_arr, pg_arr, addr_arr = physical_coords_array(
        small_ocp_cfg, lpa_arr, geom=geom
    )
    expected_block_bytes = [0] * geom.total_blocks
    expected_paths = []
    for i, lpa in enumerate(sample_lpas):
        coords = physical_coords(small_ocp_cfg, lpa, geom=geom)
        assert coords == (
            int(s_arr[i]),
            int(c_arr[i]),
            int(d_arr[i]),
            int(p_arr[i]),
            int(b_arr[i]),
            int(pg_arr[i]),
        )
        assert decode_physical_address(geom, int(addr_arr[i])) == coords
        # Cross-check against device/address.py Placed directly:
        s, c, d, p, b, pg = coords
        assert s * geom.channels_per_stack + c == int(placed.host[i])
        assert p == int(placed.plane[i])
        assert b * small_ocp_cfg.pages_per_block + pg == int(placed.offset[i])

        global_block = (
            (((s * geom.channels_per_stack + c) * geom.dies_per_channel + d)
             * geom.planes_per_die + p)
            * geom.blocks_per_plane + b
        )
        expected_block_bytes[global_block] += small_ocp_cfg.page_bytes
        expected_paths.append(f"stack{s}/ch{c}/die{d}/plane{p}/block{b}/page{pg}")

    # Build an AddressTrace with resident full capacity and run through simulate_trace
    # with 1 heatmap bin per physical block (bins == geom.total_blocks)!
    n = len(sample_lpas)
    chunk = LayerChunk(
        step=0,
        layer=0,
        subbatch=np.zeros(n, dtype=np.int64),
        request=np.zeros(n, dtype=np.int64),
        op=np.full(n, "R", dtype="U1"),
        lpa=lpa_arr,
        useful_bytes=np.full(n, 4096, dtype=np.int64),
        burst_bytes=np.full(n, 4096, dtype=np.int64),
        burst_runs=np.ones(n, dtype=np.int64),
        cls=np.full(n, "kv", dtype="U8"),
    )
    trace = AddressTrace(
        meta=make_meta(small_ocp_cfg, resident=((0, max_lpa),)),
        chunks=[chunk],
    )

    run_dir = tmp_path / "heatmap-run"
    batch_rows, _, _, stats = simulate_trace(
        small_ocp_cfg,
        trace,
        run_dir,
        binary=hbfsim_bin,
        heatmap_bins=geom.total_blocks,
        collect_completions=True,
    )
    assert len(batch_rows) == 1
    assert batch_rows[0]["media_page_reads"] == n

    # 1. Verify HBF-physical address heatmap matches expected_block_bytes bin-for-bin!
    header, events = parse_live_heatmap(run_dir / "heatmap.jsonl")
    assert header["bins"] == geom.total_blocks
    batch_events = [e for e in events if e["kind"] == "batch"]
    assert len(batch_events) == 1
    assert batch_events[0]["read"] == expected_block_bytes

    # 2. Verify exact page-in-block coordinate on a sample of random LPAs:
    # writing pages 0..pg of each LPA's block makes that exact LPA readable while
    # pg+1 (if < PPB) fails with erased-page status 0x7.
    cfg_p = tmp_path / "page_check.cfg"
    write_config(small_ocp_cfg, cfg_p, initial_free_blocks_erased=True)
    probe_lpas = [lpa for lpa in sample_lpas if (lpa // small_ocp_cfg.units) % small_ocp_cfg.pages_per_block < small_ocp_cfg.pages_per_block - 1][:16]
    # Keep distinct physical blocks so writes don't collide.
    seen_blocks = set()
    distinct_probes = []
    for lpa in probe_lpas:
        coords = physical_coords(small_ocp_cfg, lpa, geom=geom)
        if coords[:5] not in seen_blocks:
            seen_blocks.add(coords[:5])
            distinct_probes.append((lpa, coords))
    with HbfSimSession(
        cfg_p,
        tmp_path / "page-check-run",
        binary=hbfsim_bin,
        enable_hbm=False,
        enable_hbf=True,
        published_blocks_per_plane=geom.blocks_per_plane,
    ) as session:
        for idx, (lpa, (s, c, d, p, b, pg)) in enumerate(distinct_probes):
            w_txs = [
                (
                    f"w{idx}_{k}",
                    "HBF_PHYSICAL",
                    "W",
                    encode_physical_address(geom, s, c, d, p, b, k),
                    4096,
                    float(k * 200_000),
                )
                for k in range(pg + 1)
            ]
            session.run_batch(f"fill_{idx}", w_txs)
            # Reading the exact LPA's encoded address succeeds:
            target_addr = encode_physical_address(geom, s, c, d, p, b, pg)
            ok = session.run_batch(
                f"rd_ok_{idx}", [(f"rok_{idx}", "HBF_PHYSICAL", "R", target_addr, 4096, 0.0)]
            )
            assert ok["device_delta"]["hbf"]["page_reads"] == 1
        # Reading the very next page (pg + 1) fails closed with erased-page status 0x7:
        s, c, d, p, b, pg = distinct_probes[0][1]
        next_addr = encode_physical_address(geom, s, c, d, p, b, pg + 1)
        with pytest.raises(HbfSimError, match="0x7"):
            session.run_batch("rd_bad_0", [("rbad_0", "HBF_PHYSICAL", "R", next_addr, 4096, 0.0)])

    # 3. Verify meta.json recorded hbfsim_commit.
    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["hbfsim_commit"] == PIN


def test_die_blocking_switch_program_and_erase(hbfsim_bin, small_ocp_cfg, tmp_path):
    """Step 6: with hbf-die-blocking=false, a read on plane 1 of die 0 is NOT blocked by
    a program/erase on plane 0 of die 0; with hbf-die-blocking=true, it IS blocked by tPROG/tBERS."""
    geom = hbfsim_geometry(small_ocp_cfg)
    p0_blk1_pg0 = encode_physical_address(geom, 0, 0, 0, 0, block=1, page=0)
    p1_blk0_pg0 = encode_physical_address(geom, 0, 0, 0, 1, block=0, page=0)
    die1_p0_blk0_pg0 = encode_physical_address(geom, 0, 0, 1, 0, block=0, page=0)

    def _run_case(die_blocking: bool, trigger_erase: bool, tag: str):
        cfg_path = tmp_path / f"{tag}.cfg"
        write_config(
            small_ocp_cfg,
            cfg_path,
            die_blocking=die_blocking,
            initial_free_blocks_erased=not trigger_erase,
        )
        with HbfSimSession(
            cfg_path,
            tmp_path / tag,
            binary=hbfsim_bin,
            enable_hbm=False,
            enable_hbf=True,
            static_blocks_per_plane=1,
            published_blocks_per_plane=geom.blocks_per_plane - 1,
        ) as session:
            # Write on die 0 plane 0 at t=0; read die 0 plane 1 and die 1 plane 0 at t=1000 ns.
            res = session.run_batch(
                "interference",
                [
                    ("w0", "HBF_PHYSICAL", "W", p0_blk1_pg0, 4096, 0.0),
                    ("r_same_die", "HBF_PHYSICAL", "R", p1_blk0_pg0, 4096, 1000.0),
                    ("r_other_die", "HBF_PHYSICAL", "R", die1_p0_blk0_pg0, 4096, 1000.0),
                ],
                completions=True,
            )
            by_id = {c["id"]: c for c in res["transaction_completions"]}
            return (
                by_id["r_same_die"]["finish_ns"] - 1000.0,
                by_id["r_other_die"]["finish_ns"] - 1000.0,
                res["device_delta"]["hbf"]["block_erases"],
            )

    # Program-only case (tPROG = 100 us, 0 erases)
    same_off, other_off, er_off = _run_case(False, False, "prog-off")
    same_on, other_on, er_on = _run_case(True, False, "prog-on")
    assert er_off == 0 and er_on == 0
    assert same_off < 10_000.0, f"Expected unblocked read (<10 us), got {same_off}"
    assert same_on > small_ocp_cfg.program_ns, f"Expected die-blocked read (>tPROG), got {same_on}"
    assert other_on < 10_000.0, f"Other die should never be blocked, got {other_on}"

    # Auto-erase + program case (tBERS = 1 ms + tPROG = 100 us)
    same_e_off, _, er_e_off = _run_case(False, True, "erase-off")
    same_e_on, other_e_on, er_e_on = _run_case(True, True, "erase-on")
    assert er_e_off == 1 and er_e_on == 1
    assert same_e_off < 10_000.0
    assert same_e_on > small_ocp_cfg.erase_ns + small_ocp_cfg.program_ns
    assert other_e_on < 10_000.0


def test_convert_drops_d_and_preserves_cls(small_ocp_cfg, tmp_path):
    """Step 2: D records emit no device commands; cls metadata is preserved for attribution."""
    from hbf.backends.hbfsim import write_replay_trace

    chunk = LayerChunk(
        step=0,
        layer=0,
        subbatch=np.array([0, 0, 0, 1], dtype=np.int64),
        request=np.zeros(4, dtype=np.int64),
        op=np.array(["W", "R", "D", "W"], dtype="U1"),
        lpa=np.array([0, 0, 0, 1], dtype=np.int64),
        useful_bytes=np.full(4, 4096, dtype=np.int64),
        burst_bytes=np.full(4, 4096, dtype=np.int64),
        burst_runs=np.ones(4, dtype=np.int64),
        cls=np.array(["prefill", "kv", "kv", "flush"], dtype="U8"),
    )
    trace = AddressTrace(meta=make_meta(small_ocp_cfg, resident=()), chunks=[chunk])
    conv = convert_trace(small_ocp_cfg, trace)
    assert conv.dropped_releases == 1
    assert len(conv.batches) == 2
    assert conv.batches[0].arrival_ns == 0
    assert conv.batches[1].arrival_ns == small_ocp_cfg.interval_ns
    assert [(r.op, r.cls) for r in conv.batches[0].requests] == [("W", "prefill"), ("R", "kv")]
    assert [(r.op, r.cls) for r in conv.batches[1].requests] == [("W", "flush")]
    assert conv.cls_counts == {"prefill": 1, "kv": 1, "flush": 1}

    replay = write_replay_trace(tmp_path / "replay.trace", conv)
    lines = [ln for ln in replay.read_text().splitlines() if ln and not ln.startswith("#")]
    assert len(lines) == 3
    assert " W " in lines[0] and " R " in lines[1] and " W " in lines[2]


def test_hbfsim_backend_does_not_cross_import():
    """hbf/backends/hbfsim must never import from hbf.backends.mqsim or hbf.backends.rounds."""
    hbfsim_dir = ROOT / "hbf" / "backends" / "hbfsim"
    for path in sorted(hbfsim_dir.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                parts = (node.module or "").split(".")
                assert "mqsim" not in parts and "rounds" not in parts, f"{path} cross-imports {node.module}"
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    parts = alias.name.split(".")
                    assert "mqsim" not in parts and "rounds" not in parts, f"{path} cross-imports {alias.name}"
