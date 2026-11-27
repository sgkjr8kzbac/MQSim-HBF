"""Endurance model (hbf/endurance.py, M4): Clock against phase A and the closed-form steady state.

Hand-computed values from design-B sections 3 and 7 (tests 8 and 9). Phase A runs here are small
analysis-only pipelines (seconds); nothing runs MQSim.
"""
import csv
import json
import math

import numpy as np
import pytest

from hbf.endurance import (METRICS, Clock, ConstantClock, Device, analytic, build_parser, calibrate,
                           calibration_batches, configure, erase_amp, geometry_for, layout_rounds, main, sci,
                           summarize_run, threshold_steps)
from hbf.model import load_gpu, load_model
from hbf.workload import Workload, load_workload, parse_length, presets

B200 = load_gpu("b200")
LLAMA8 = load_model("llama-3.1-8b")
SUPERBLOCK = 1024 * 256  # pages of one block index across all banks at ppb 256


def no_m5(**kwargs):
    return {"m5": "off"} | kwargs


# ---- test 8: Clock against phase A ---------------------------------------------------------------

@pytest.mark.parametrize("model,fmt,batch,context,tpot_ms", [
    ("llama-3.1-8b", "bf16", 8, 32768, 38.57),
    ("llama-3.1-8b", "bf16", 32, 32768, 146.85),
    ("llama-3.1-8b", "kivi2", 64, 4096, 9.24),
    ("llama-3.1-70b", "bf16", 8, 32768, 112.28),
])
def test_8_clock_matches_phase_a(model, fmt, batch, context, tpot_ms):
    """Phase A TPOTs of design-B section 0.1 (ocp, HBF command model, striped_contiguous, B200)."""
    spec = load_model(model)
    geometry, _ = geometry_for(spec, fmt)
    clock = Clock(spec, B200, geometry, kv_format=fmt, rounds_fn=layout_rounds(spec, fmt))
    step = clock.step([context] * batch)
    assert step.tpot_ns / 1e6 == pytest.approx(tpot_ms, rel=5e-3)
    assert sum(step.gaps_ns) == pytest.approx(step.tpot_ns - spec.layers * step.h_ns)


def test_clock_balance_and_partitions():
    geometry, _ = geometry_for(LLAMA8, "bf16")
    clock = Clock(LLAMA8, B200, geometry)
    # 8 requests at 32K: 262,144 pages per layer, 256 rounds; with kappa 1 equal to the layout's rounds.
    assert clock.step([32768] * 8).rounds == 256
    assert Clock(LLAMA8, B200, geometry, balance=1.5).step([32768] * 8).rounds == 384
    # Channel partitions, one request per channel (design-B test 4): 512 rounds against 256 striped.
    parts = np.arange(8)
    assert clock.step([32768] * 8, parts, 64).rounds == 512
    assert clock.step([32768] * 16, np.arange(16), 64).rounds == 512
    # Generated tokens held in HBM leave the HBF pages unchanged.
    assert clock.step([32768 + 512] * 8, hbm_tokens=[512] * 8).rounds == 256
    assert clock.step([]) is None


def test_8_calibrate_against_small_phase_a(tmp_path):
    """The calibrated Clock reproduces the phase A runs it was fitted to; results are cached."""
    assert calibration_batches("llama-3.1-8b", "bf16", 32768) == (1, 4)
    assert calibration_batches("llama-3.1-8b", "bf16", 1024) == (32, 128)
    calib = calibrate(tmp_path, "llama-3.1-8b", "b200", "bf16", context=1024, batches=(8, 32), policy="packed",
                      snapshot_batch=8)
    assert calib["layer_ns"] > 0 and 4000 < calib["round_ns"] < 4800
    assert calib["balance"] >= 1 and calib["snapshot"]["policy"] == "packed"
    geometry, _ = geometry_for(LLAMA8, "bf16")
    clock = Clock(LLAMA8, B200, geometry, layer_ns=calib["layer_ns"], round_ns=calib["round_ns"])
    assert clock.step([1024] * 32).tpot_ns == pytest.approx(calib["phase_a_tpot_ns"]["b32"], rel=5e-3)
    # The packed snapshot at B = 8 is reproduced through kappa (a per-layer mean, so within 2 % at ~20 rounds).
    packed = Clock(LLAMA8, B200, geometry, layer_ns=calib["layer_ns"], round_ns=calib["round_ns"],
                   balance=calib["balance"])
    assert packed.step([1024] * 8).tpot_ns == pytest.approx(calib["snapshot"]["tpot_ns"], rel=2e-2)
    runs = sorted(path.name for path in next(tmp_path.iterdir()).iterdir())
    again = calibrate(tmp_path, "llama-3.1-8b", "b200", "bf16", context=1024, batches=(8, 32), policy="packed",
                      snapshot_batch=8)
    assert again == calib and sorted(path.name for path in next(tmp_path.iterdir()).iterdir()) == runs


def test_calibrate_sparse_ratios(tmp_path):
    calib = calibrate(tmp_path, "llama-3.1-8b", "b200", "bf16", mode="uniform_tokens",
                      generation={"fraction": .05, "select_tokens": 16}, context=2048, batches=(8, 32),
                      policy="layer_major")
    # BF16 pages hold 16 tokens = one selected block: pages and attended tokens scale by about 5 %.
    assert calib["ratios"]["pages"] == pytest.approx(.05, abs=.01)
    assert calib["ratios"]["attended"] == pytest.approx(calib["ratios"]["pages"], rel=1e-6)
    assert calib["read_share"] == calib["ratios"]["pages"]


# ---- test 9: closed form --------------------------------------------------------------------------

def test_9_threshold_and_refresh_counts():
    assert threshold_steps(1e6, 256) == 3906.25
    assert [threshold_steps(x, 256) for x in (1e5, 1e7)] == [390.625, 39062.5]
    assert threshold_steps(1e6, 256, share=.05, hot=2) == pytest.approx(39062.5)
    geometry, _ = geometry_for(LLAMA8, "bf16")
    clock = ConstantClock(38.57e6)
    # A 32K request (28,768 + 4,000 tokens) is read for 4,000 steps: one refresh of all its pages at X = 1e6.
    row = analytic(geometry, clock, Workload(input="fixed:28768", output="fixed:4000"), x_values=(1e6,), **no_m5())[0]
    assert row["pages_per_request"] == 1_048_576
    assert row["refreshes_per_request"] * row["pages_per_request"] == 1_048_576
    # 338 output tokens never reach X = 1e5 (338 < 390.6 steps).
    row = analytic(geometry, clock, Workload(input="fixed:32768", output="fixed:338"), x_values=(1e5,), **no_m5())[0]
    assert row["refreshes_per_request"] == 0 and row["refresh_rd_gb_s"] == 0
    assert row["break_even_x"] == pytest.approx(338 * 256)


def test_9_closed_loop_worked_example():
    """8B BF16, B = 8, 32K uncached, 338 out, T = 38.57 ms (design-B section 3)."""
    geometry, _ = geometry_for(LLAMA8, "bf16")
    w = Workload(input="fixed:32768", output="fixed:338", concurrency=8)
    row = analytic(geometry, ConstantClock(38.57e6), w, x_values=(1e6,), align="none", **no_m5())[0]
    assert row["req_per_s"] == pytest.approx(8 / (338 * .03857))  # 0.614 req/s
    assert row["pages_per_request"] == 32 * 8 * 2 * (33106 // 16) == 1_059_328
    assert row["kv_write_gb_s"] == pytest.approx(2.66, abs=.01)
    assert row["avg_pec_per_day"] == pytest.approx(418.5, rel=2e-3)
    assert row["days_to_limit_avg@1e5"] == pytest.approx(239, abs=1)
    assert row["decode_gb_s"] * 1e3 == pytest.approx(27.0, abs=.3)  # 27.2 MB/s with partial pages counted
    assert row["wa"] == 1 and row["erase_wa"] == 1
    # Row-aligned prompts add (1,024 - 1) / 2 padding pages per request on average.
    aligned = analytic(geometry, ConstantClock(38.57e6), w, x_values=(1e6,), **no_m5())[0]
    assert aligned["padding_gb_s"] == pytest.approx(row["req_per_s"] * 511.5 * 4096 / 1e9)
    assert aligned["avg_pec_per_day"] == pytest.approx(418.5, rel=2e-3)


def test_compression_scales_writes_at_fixed_rate():
    rows = {}
    for fmt in ("bf16", "fp8", "kivi2"):
        geometry, _ = geometry_for(LLAMA8, fmt)
        w = Workload(input="fixed:32768", output="fixed:338", arrival="poisson", rate=.5)
        rows[fmt] = analytic(geometry, ConstantClock(40e6), w, x_values=(1e6,), align="none", **no_m5())[0]
    assert rows["fp8"]["kv_write_gb_s"] == pytest.approx(rows["bf16"]["kv_write_gb_s"] / 2, rel=2e-3)
    ratio = rows["bf16"]["avg_pec_per_day"] / rows["kivi2"]["avg_pec_per_day"]
    assert 4.5 < ratio < 5.5  # KIVI-2 G32 with a 128-token BF16 residual in HBM: about 5.3x per token
    assert rows["kivi2"]["pe_per_mtok"] == pytest.approx(rows["bf16"]["pe_per_mtok"] / ratio)


def test_dedicated_erase_amplification():
    geometry, _ = geometry_for(LLAMA8, "bf16")
    w = Workload(input="fixed:1024", output="fixed:338", concurrency=8)
    pages = 32 * 8 * 2 * (1362 // 16)  # 43,520 full pages; the partial 16-token page dies in HBM
    assert erase_amp(pages, SUPERBLOCK) == pytest.approx(SUPERBLOCK / pages)  # 6.02 (5.95 with the partial page)
    packed = analytic(geometry, ConstantClock(10e6), w, x_values=(1e6,), align="none", **no_m5())[0]
    dedicated = analytic(geometry, ConstantClock(10e6), w, policy="dedicated", x_values=(1e6,), **no_m5())[0]
    assert dedicated["erase_wa"] == pytest.approx(SUPERBLOCK / pages)
    assert dedicated["avg_pec_per_day"] == pytest.approx(packed["avg_pec_per_day"] * SUPERBLOCK / pages)
    channel = analytic(geometry, ConstantClock(10e6), w, Device(partition="channel"), policy="dedicated",
                       x_values=(1e6,), **no_m5())[0]
    assert channel["erase_wa"] == pytest.approx(math.ceil(pages / 16384) * 16384 / pages)  # 1.13


def test_hbm_decode_and_layer_major():
    geometry, _ = geometry_for(LLAMA8, "bf16")
    w = Workload(input="fixed:32768", output="fixed:4000", concurrency=8)
    clock = ConstantClock(40e6)
    hbm = analytic(geometry, clock, w, policy="hbm_decode", x_values=(1e6,), **no_m5())[0]
    assert hbm["decode_gb_s"] == 0
    assert hbm["staging_peak_mib"] > 8 * 4000 * 131072 / 2 ** 20  # generated KV stays in HBM
    region = 8 * 1_048_576 + 3 * 1024  # a wave's region with some padding
    wave = analytic(geometry, clock, w, policy="layer_major", region=region, x_values=(1e6,), **no_m5())[0]
    per_s = 8 / (4000 * .04)  # requests per second: one wave of 8 every 4,000 steps
    assert wave["prefill_gb_s"] == pytest.approx(per_s * 1_048_576 * 4096 / 1e9)
    assert wave["padding_gb_s"] == pytest.approx(per_s * 3 * 1024 / 8 * 4096 / 1e9)
    # 4,000 x 256 reads >= X = 1e6: the whole region is replayed once per wave.
    assert wave["replay_gb_s"] == pytest.approx(per_s * region / 8 * 4096 / 1e9)
    assert wave["refresh_rd_gb_s"] == 0 and wave["replay_buffer_mib"] == 1024


def test_retention_and_sessions():
    geometry, _ = geometry_for(LLAMA8, "bf16")
    clock = ConstantClock(40e6)
    # An entry kept 50 h after its request is rewritten twice at T_ret = 24 h.
    w = Workload(input="fixed:4096", output="fixed:16", concurrency=1, retain_s=50 * 3600)
    row = analytic(geometry, clock, w, x_values=(1e7,), **no_m5())[0]
    pages = 32 * 8 * 2 * (4112 // 16)
    assert row["refresh_ret_gb_s"] == pytest.approx(2 * row["req_per_s"] * pages * 4096 / 1e9)
    off = analytic(geometry, clock, w, x_values=(1e7,), retention_s=0, **no_m5())[0]
    assert off["refresh_ret_gb_s"] == 0
    # Two-call session, 0 think time: call 0 writes 1,024 + 32 tokens and decodes 32; call 1 appends 32 + 32.
    s = Workload(input="fixed:32", output="fixed:32", calls=2, history="fixed:1024", concurrency=1)
    # Block reads: call 0's prompt 2 x 32 x 256 + 256 (call 1's prefill) = 16,640, its decode pages 16 steps
    # fewer = 12,544; call 1's prompt 32 x 256 = 8,192, its decode pages 4,096. X = 10,000 refreshes call 0 once.
    x = 10_000
    row = analytic(geometry, clock, s, x_values=(x,), **no_m5())[0]
    sessions_per_s = 1 / (2 * 32 * .04)
    first = 32 * 8 * 2 * (1056 // 16)
    expected = (first + 32 * 8 * 2 * 2) * sessions_per_s  # call 0's prompt and decode pages are refreshed once
    assert row["refresh_rd_gb_s"] == pytest.approx(expected * 4096 / 1e9)
    assert row["req_per_s"] == pytest.approx(2 * sessions_per_s)


def test_shared_prefix_reads():
    geometry, _ = geometry_for(LLAMA8, "bf16")
    w = Workload(input="fixed:512", output="fixed:256", concurrency=64, prefix_groups=1, prefix_tokens=4096,
                 prefix_share=1.0)
    per_request = analytic(geometry, ConstantClock(20e6), w, x_values=(1e6,), **no_m5())[0]
    once = analytic(geometry, ConstantClock(20e6), w, x_values=(1e6,), prefix_reads="once", **no_m5())[0]
    assert per_request["refresh_rd_gb_s"] > 30 * once["refresh_rd_gb_s"] > 0
    assert per_request["refresh_ret_gb_s"] > 0  # the shared prefix is resident: rewritten every 24 h


def test_m5_raises_tpot():
    geometry, _ = geometry_for(LLAMA8, "bf16")
    clock = Clock(LLAMA8, B200, geometry)
    row = analytic(geometry, clock, load_workload("long32k"), x_values=(1e6,))[0]
    assert row["tpot_m5_ms"] > row["mean_tpot_ms"] and row["max_step_extra_ms"] > 0
    assert row["program_bw_share"] == pytest.approx(row["program_gb_s"] / (1024 * 4096 / 1e-4 / 1e9))


# ---- workloads -----------------------------------------------------------------------------------

def test_lengths_and_presets():
    assert parse_length("fixed:5").mean() == 5 and parse_length(7).mean() == 7 and str(parse_length("9")) == "fixed:9"
    wide = parse_length("lognormal:median=1000,sigma=0.5")
    assert wide.mean() == pytest.approx(1000 * math.exp(.125), rel=2e-3)
    assert wide.mean_max(8) > wide.mean() and parse_length("fixed:3").mean_max(8) == 3
    assert parse_length("lognormal:median=1000,sigma=2,max=1500").mean() < 1500
    assert parse_length("uniform:lo=10,hi=20").mean() == pytest.approx(15, abs=.01)
    for bad in ("fixed:0", "normal:mu=1", "lognormal:median=5", "uniform:lo=1,hi=2,foo=3"):
        with pytest.raises(ValueError):
            parse_length(bad)
    for name, spec in presets().items():
        workload = load_workload(name)
        assert workload.source and ("[W]" in workload.source or "[G]" in workload.source)
        assert workload.input.mean() > 0 and workload.output.mean() > 0
    agentic = load_workload("agentic")
    assert agentic.sessions and agentic.calls == 64 and load_workload("long").name == "long32k"
    with pytest.raises(ValueError):
        load_workload("long32k", bogus=1)
    with pytest.raises(ValueError):
        Workload(arrival="poisson")


# ---- command line --------------------------------------------------------------------------------

def test_cli_analytic(tmp_path):
    out = tmp_path / "run"
    main(["--analytic", "--no-calibrate", "--format", "bf16,kivi2", "--workload", "long32k",
          "--rd-threshold", "1e5,1e6,1e7", "--pe-limit", "1e4,1e5,1e6", "--pages-per-block", "256,512",
          "--output", str(out)])
    with (out / "endurance_summary.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2 * 3 * 2
    assert {"days_to_limit_avg@1e4", "days_to_limit_avg@1e5", "days_to_limit_avg@1e6"} <= set(rows[0])
    bf16 = [row for row in rows if row["format"] == "bf16" and row["pages_per_block"] == "256"]
    assert [float(row["threshold_steps"]) for row in bf16] == [390.625, 3906.25, 39062.5]
    assert float(rows[0]["mean_tpot_ms"]) == pytest.approx(38.85, abs=.05)
    report = (out / "report.md").read_text()
    assert "# 寿命评估" in report and "## 假设" in report and "第 5.4.1 节" in report and "[P]" in report
    meta = json.loads((out / "meta.json").read_text())
    assert meta["tool"] == "endurance" and meta["analytic"] is True
    summary = summarize_run(out)
    assert set(summary) == set(METRICS) and summary["mean_batch"] == 8


def test_cli_validation():
    parser = build_parser()
    with pytest.raises(ValueError, match="--analytic"):
        configure(parser.parse_args([]))
    with pytest.raises(ValueError, match="suspend"):
        configure(parser.parse_args(["--analytic", "--m5", "suspend"]))
    with pytest.raises(ValueError, match="partition"):
        configure(parser.parse_args(["--analytic", "--alloc", "layer_major", "--partition", "die"]))
    with pytest.raises(ValueError):
        configure(parser.parse_args(["--analytic", "--format", "nosuch"]))
    assert sci(1e5) == "1e5" and sci(3e3) == "3e3" and sci(256) == "256"
