"""Tests for R7: unified request-level traces, Mooncake importer, CLI, timeline + MQSim/rounds."""
import json
from pathlib import Path

import pytest

from hbf.backends.mqsim.parse import block_counts
from hbf.backends.mqsim.run import simulate as mqsim_simulate
from hbf.backends.mqsim.schedule import batches_from_trace
from hbf.backends.rounds.analysis import analyze_trace
from hbf.device.config import load
from hbf.frontend.addrtrace import validate
from hbf.frontend.kv.formats import parse_format
from hbf.frontend.timeline import build_timeline
from hbf.frontend.trace.requests import (
    RequestRecord,
    Stream,
    import_mooncake_jsonl,
    load_requests,
    per_stack,
    read_requests_csv,
    summarize_requests,
    workload_from_requests,
    write_requests_csv,
)
from hbf.lifetime.cli import main as lifetime_main
from hbf.lifetime.clock import ReadTimeClock
from hbf.lifetime.counts import compare_block_counts, counts_from_trace
from hbf.lifetime.geometry import Device, Geometry, geometry_for
from hbf.lifetime.sim import SimOptions, simulate as sim_simulate
from hbf.models import load_model


def _write_mooncake_fixture(path: Path) -> Path:
    """Create a deterministic inline Mooncake JSONL fixture (no network required)."""
    rows = [
        # Conv A turn 1 & 2 share [0, 10, 11, 12] (2048 tok), turn 3 shares [0, 10, 11, 12, 13]
        {"timestamp": 1000, "input_length": 2048, "output_length": 32, "hash_ids": [0, 10, 11, 12]},
        {"timestamp": 2000, "input_length": 2560, "output_length": 40, "hash_ids": [0, 10, 11, 12, 13]},
        {"timestamp": 3500, "input_length": 3000, "output_length": 48, "hash_ids": [0, 10, 11, 12, 13, 14]},
        # Standalone request sharing only universal root [0] (512 tok)
        {"timestamp": 4000, "input_length": 1200, "output_length": 24, "hash_ids": [0, 20, 21]},
        # Conv B turn 1 & 2 share [0, 30, 31] (1536 tok)
        {"timestamp": 5000, "input_length": 1536, "output_length": 36, "hash_ids": [0, 30, 31]},
        {"timestamp": 6000, "input_length": 2000, "output_length": 28, "hash_ids": [0, 30, 31, 32]},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def test_requests_csv_roundtrip_3_4_and_5_columns(tmp_path):
    """Canonical 3-col, 4-col, and 5-col CSVs round-trip cleanly and reject malformed input."""
    recs_3 = [
        RequestRecord(arrival_s=0.0, prompt_tokens=1024, output_tokens=64),
        RequestRecord(arrival_s=1.5, prompt_tokens=2048, output_tokens=128),
    ]
    p3 = write_requests_csv(tmp_path / "req3.csv", recs_3)
    assert p3.read_text().splitlines()[0] == "arrival_s,prompt_tokens,output_tokens"
    assert read_requests_csv(p3) == recs_3

    recs_4 = [
        RequestRecord(arrival_s=0.0, prompt_tokens=1024, output_tokens=64, prefix_id=0),
        RequestRecord(arrival_s=0.5, prompt_tokens=2048, output_tokens=128, prefix_id=None),
        RequestRecord(arrival_s=1.2, prompt_tokens=1536, output_tokens=32, prefix_id=0),
    ]
    p4 = write_requests_csv(tmp_path / "req4.csv", recs_4)
    assert p4.read_text().splitlines()[0] == "arrival_s,prompt_tokens,output_tokens,prefix_id"
    assert read_requests_csv(p4) == recs_4

    recs_5 = [
        RequestRecord(arrival_s=0.0, prompt_tokens=2048, output_tokens=64, prefix_id=0, prefix_tokens=1024),
        RequestRecord(arrival_s=1.0, prompt_tokens=3072, output_tokens=96, prefix_id=0, prefix_tokens=1024),
    ]
    p5 = write_requests_csv(tmp_path / "req5.csv", recs_5)
    assert p5.read_text().splitlines()[0] == "arrival_s,prompt_tokens,output_tokens,prefix_id,prefix_tokens"
    assert read_requests_csv(p5) == recs_5

    # Unsorted CSV is stably sorted by arrival_s
    unsorted = tmp_path / "unsorted.csv"
    unsorted.write_text("arrival_s,prompt_tokens,output_tokens\n2.0,512,16\n0.5,256,8\n", encoding="utf-8")
    loaded = read_requests_csv(unsorted)
    assert [r.arrival_s for r in loaded] == [0.5, 2.0]

    # Reject bad header and invalid token counts
    bad = tmp_path / "bad.csv"
    bad.write_text("arrival_s,prompt_tokens\n0.0,100\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_requests_csv(bad)

    bad_tok = tmp_path / "bad_tok.csv"
    bad_tok.write_text("arrival_s,prompt_tokens,output_tokens\n0.0,0,10\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_requests_csv(bad_tok)


def test_import_mooncake_jsonl_prefix_modes_and_windowing(tmp_path):
    """Mooncake JSONL importer shifts timestamps, groups hash_ids prefixes, and supports slicing."""
    jsonl = _write_mooncake_fixture(tmp_path / "mooncake.jsonl")

    conv = import_mooncake_jsonl(jsonl, block_tokens=512, prefix_mode="conversation")
    assert len(conv) == 6
    assert conv[0].arrival_s == pytest.approx(0.0)
    assert conv[-1].arrival_s == pytest.approx(5.0)
    # Conv A (indices 0,1,2) all share prefix_id 0 with 4*512 = 2048 tokens
    assert conv[0].prefix_id == conv[1].prefix_id == conv[2].prefix_id == 0
    assert conv[0].prefix_tokens == conv[1].prefix_tokens == conv[2].prefix_tokens == 2048
    # Standalone (index 3) shares universal root [0] -> 512 tokens
    assert conv[3].prefix_tokens == 512
    # Conv B (indices 4,5) share [0, 30, 31] -> 1536 tokens
    assert conv[4].prefix_id == conv[5].prefix_id
    assert conv[4].prefix_id != conv[0].prefix_id
    assert conv[4].prefix_tokens == 1536

    summary = summarize_requests(conv)
    assert summary["count"] == 6
    assert summary["prefix_groups"] == 3
    assert summary["causal_hit_rate"] == pytest.approx(3 / 6)

    longest = import_mooncake_jsonl(jsonl, block_tokens=512, prefix_mode="longest")
    # Turn 2 & 3 share 5 blocks = 2560 tokens under longest mode
    assert longest[1].prefix_id == longest[2].prefix_id
    assert longest[1].prefix_tokens == 2560

    no_pre = import_mooncake_jsonl(jsonl, block_tokens=512, prefix_mode="none")
    assert all(r.prefix_id is None and r.prefix_tokens is None for r in no_pre)

    # Time window slicing + auto loader (rel_s = 0.0, 1.0, 2.5, 3.0, 4.0, 5.0 -> [1.0, 4.5) has 4)
    sliced = load_requests(jsonl, fmt="auto", start_s=1.0, duration_s=3.5)
    assert len(sliced) == 4
    assert sliced[0].arrival_s == pytest.approx(0.0)


def test_workload_from_requests_looping_and_prefix_sharing(tmp_path):
    """workload_from_requests produces a looping script workload and reduces prefill writes via prefix sharing."""
    jsonl = _write_mooncake_fixture(tmp_path / "mooncake.jsonl")
    recs_shared = import_mooncake_jsonl(jsonl, prefix_mode="conversation")
    recs_plain = import_mooncake_jsonl(jsonl, prefix_mode="none")

    wl_shared = workload_from_requests(recs_shared, name="mc_shared", retain_s=10.0, loop=True, loop_s=6.0)
    wl_plain = workload_from_requests(recs_plain, name="mc_plain", retain_s=0.0, loop=True, loop_s=6.0)

    # Verify Stream looping yields multiple cycles
    stream = Stream(wl_shared)
    c0 = stream.arrived(5.9)
    c1 = stream.arrived(11.9)
    assert len(c0) == 6
    assert len(c1) == 6
    assert c1[0].arrival == pytest.approx(6.0)
    assert c0[0].prefix == 0 and c0[0].prefix_tokens == 2048 and c0[0].input == 0
    assert c0[1].prefix == 0 and c0[1].prefix_tokens == 2048 and c0[1].input == 512

    # Simulator with prefix sharing writes strictly fewer KV pages than without prefix sharing
    model = load_model("llama-3.1-8b")
    geo, _ = geometry_for(model, "bf16")
    dev = Device(banks=1024, pages_per_block=256, page_bytes=4096, capacity_gib=512.0)
    clk = ReadTimeClock(geo, model.layers, compute_ns=2_000_000.0)
    opts = SimOptions(policy="packed", x=1e6, hours=0.05, warmup_h=0.0, quantum_s=0.5, m5="off")

    res_shared = sim_simulate(geo, clk, wl_shared, dev, opts).result
    res_plain = sim_simulate(geo, clk, wl_plain, dev, opts).result
    assert res_shared["completed"] >= 100
    assert res_plain["completed"] >= 100
    assert res_shared["prefill_gb_s"] < res_plain["prefill_gb_s"] * 0.75

    # Prefix-aware per_stack sharding keeps conversation group 0 intact on stack 0
    sharded = per_stack(wl_shared, 2)
    g0 = [item for item in sharded.script if item.get("prefix") == 0]
    assert len(g0) == 3
    assert len(sharded.script) < len(wl_shared.script)


def test_r7_timeline_mqsim_and_rounds_on_imported_requests(binary, tmp_path):
    """Imported Mooncake requests -> build_timeline -> AddressTrace -> small-topology MQSim + rounds match."""
    jsonl = _write_mooncake_fixture(tmp_path / "mooncake.jsonl")
    # Compact fixture into a short 0.5s window with scaled tokens for fast small-topology MQSim check
    recs = [
        RequestRecord(arrival_s=0.00, prompt_tokens=128, output_tokens=16, prefix_id=0, prefix_tokens=64),
        RequestRecord(arrival_s=0.05, prompt_tokens=160, output_tokens=16, prefix_id=0, prefix_tokens=64),
        RequestRecord(arrival_s=0.10, prompt_tokens=192, output_tokens=16, prefix_id=0, prefix_tokens=64),
        RequestRecord(arrival_s=0.15, prompt_tokens=96, output_tokens=12, prefix_id=1, prefix_tokens=32),
        RequestRecord(arrival_s=0.20, prompt_tokens=128, output_tokens=12, prefix_id=1, prefix_tokens=32),
    ]
    csv_path = write_requests_csv(tmp_path / "compact.csv", recs)
    loaded = read_requests_csv(csv_path)
    wl = workload_from_requests(loaded, name="compact_mc", retain_s=1.0, loop=False)

    cfg = load(
        "16x64", channels=2, chips=2, dies=1, planes=4,
        blocks_per_plane=12, pages_per_block=8,
        aggregate_gb_s=64, plane_rule="independent", coalesce_ns=None, read_plane_ns=None,
    )
    geo = Geometry(2, 2, parse_format("bf16", 128).tensors, cfg.page_bytes)
    trace, tl_stats = build_timeline(
        cfg, geo, wl, steps=60, step_s=0.01, policy="packed",
        x=256, retention_s=0.0, refresh="relocate", low_watermark=2, max_segments=10,
    )
    validate(cfg, trace)
    assert tl_stats["completed"] == 5

    # Verify rounds backend consumes the exact same AddressTrace cleanly
    layer_rows, step_rows = analyze_trace(cfg, trace.chunks)
    assert len(step_rows) == 60
    rounds_read_pages = sum(r["pages"] for r in layer_rows)
    assert rounds_read_pages > 0

    # Verify MQSim (T-HBFW1..4) matches Python trace block counts 100% block-for-block
    py_counts, _ = counts_from_trace(cfg, trace)
    batches = batches_from_trace(cfg, trace.chunks)
    out_dir = tmp_path / "mqsim"
    _, _, _, mq_stats = mqsim_simulate(cfg, batches, out_dir, binary, hbf_write=True)
    assert mq_stats["Total_GC_Executions"] == 0
    mq_counts = block_counts(out_dir / "block_counts.csv")
    cmp = compare_block_counts(mq_counts, py_counts)
    assert cmp["exact_match"], cmp
    # Also verify Mooncake JSONL fixture itself builds a valid timeline
    mc_recs = import_mooncake_jsonl(jsonl)
    mc_wl = workload_from_requests(mc_recs, name="mc_fixture", retain_s=10.0, loop=False)
    cfg_big = load(
        "16x64", channels=2, chips=2, dies=1, planes=4,
        blocks_per_plane=40, pages_per_block=16,
        aggregate_gb_s=64, plane_rule="independent", coalesce_ns=None, read_plane_ns=None,
    )
    geo_1 = Geometry(1, 1, parse_format("bf16", 128).tensors, cfg_big.page_bytes)
    mc_trace, _ = build_timeline(cfg_big, geo_1, mc_wl, steps=10, step_s=0.1, x=1000, max_segments=32)
    validate(cfg_big, mc_trace)


def test_lifetime_cli_requests_csv_dense_sparse_and_toml(tmp_path):
    """hbf lifetime --requests-csv works for dense, sparse (uniform_tokens), and TOML configs."""
    jsonl = _write_mooncake_fixture(tmp_path / "mooncake.jsonl")
    recs = import_mooncake_jsonl(jsonl)
    csv_path = write_requests_csv(tmp_path / "mc.csv", recs)

    out_dense = tmp_path / "run_dense"
    lifetime_main([
        "--requests-csv", str(csv_path),
        "--requests-retain-s", "30",
        "--hours", "0.05", "--warmup-h", "0.0",
        "--rd-threshold", "1e6",
        "--no-calibrate",
        "--output", str(out_dense),
    ])
    cfg_dense = json.loads((out_dense / "config.json").read_text())
    assert cfg_dense["trace"]["requests_csv"] == str(csv_path)
    assert (out_dense / "endurance_summary.csv").is_file()

    # Combine real request trace with synthetic sparse token selection (uniform_tokens)
    out_sparse = tmp_path / "run_sparse"
    lifetime_main([
        "--requests-csv", str(jsonl),
        "--requests-retain-s", "30",
        "--mode", "uniform_tokens", "--fraction", "0.1",
        "--hours", "0.05", "--warmup-h", "0.0",
        "--rd-threshold", "1e5",
        "--no-calibrate",
        "--output", str(out_sparse),
    ])
    summary_sparse = (out_sparse / "endurance_summary.csv").read_text()
    assert "uniform_tokens" in summary_sparse

    # TOML config specifying [trace] requests_csv
    toml_path = tmp_path / "mooncake.toml"
    toml_path.write_text(
        f'[trace]\nrequests_csv = "{csv_path}"\nrequests_retain_s = 30\nmode = "dense"\n'
        '[lifetime]\nhours = 0.05\nwarmup_h = 0.0\nrd_threshold = "1e6"\nno_calibrate = true\n',
        encoding="utf-8",
    )
    out_toml = tmp_path / "run_toml"
    lifetime_main([str(toml_path), "--output", str(out_toml)])
    assert (out_toml / "endurance_summary.csv").is_file()
