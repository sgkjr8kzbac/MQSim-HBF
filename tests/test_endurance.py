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
# Event simulation (design-B section 4, tests 10-15) and the seeded request stream.
from hbf.access import TensorSpec
from hbf.endurance import CAUSE, GIB, CapacityError, Geometry, SimOptions, simulate
from hbf.interference import interference
from hbf.workload import Stream, scripted

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


# ---- request stream (hbf/workload.py) ------------------------------------------------------------

def test_stream_closed_loop_is_seeded_and_staggered():
    w = load_workload("chat", concurrency=4, seed=7)
    a, b = Stream(w), Stream(w)
    first = [(r.input, r.output) for r in (a.closed_new(0.0) for _ in range(20))]
    assert first == [(r.input, r.output) for r in (b.closed_new(0.0) for _ in range(20))]
    other = Stream(load_workload("chat", concurrency=4, seed=8))
    assert first != [(r.input, r.output) for r in (other.closed_new(0.0) for _ in range(20))]
    # The i-th of the first B requests decodes out - round(i out / B) tokens (staggered start).
    fixed = Stream(load_workload("long32k", concurrency=8))
    assert [fixed.closed_new(0.0).output for _ in range(10)] == [338, 296, 254, 211, 169, 127, 84, 42, 338, 338]
    plain = Stream(load_workload("long32k", concurrency=8), stagger=False)
    assert {plain.closed_new(0.0).output for _ in range(8)} == {338}
    assert Stream(Workload(arrival="poisson", rate=1.0)).closed_new(0.0) is None


def test_stream_poisson_rate_and_prefix_choice():
    w = Workload(arrival="poisson", rate=10.0, prefix_groups=4, prefix_tokens=64, prefix_share=.5, seed=3)
    stream = Stream(w)
    requests = stream.arrived(2000.0)
    assert len(requests) == pytest.approx(20000, rel=.02) and stream.next_time() > 2000
    assert np.diff([r.arrival for r in requests]).mean() == pytest.approx(.1, rel=.02)
    groups = np.array([r.prefix for r in requests])
    assert (groups >= 0).mean() == pytest.approx(.5, abs=.02)
    # Zipf 1 over 4 groups: the first takes 1 / (1 + 1/2 + 1/3 + 1/4) = 12/25 of the hits.
    assert (groups[groups >= 0] == 0).mean() == pytest.approx(12 / 25, abs=.02)


def test_stream_sessions_and_script():
    w = Workload(input="fixed:10", output="fixed:5", calls=3, history="fixed:100", think_s=2.0, concurrency=2)
    stream = Stream(w)
    first = stream.arrived(0.0)
    # Session 1 starts at call floor(1 x 3 / 2) = 1, with call 0's 100 + 10 + 5 tokens written at once.
    assert [(r.session, r.call, r.input, r.history, r.last) for r in first] == [
        (0, 0, 110, 0, False), (1, 1, 125, 0, False)]
    stream.call_done(first[0], 7.0)
    assert stream.next_time() == 9.0
    call = stream.arrived(9.0)[0]
    assert (call.session, call.call, call.history, call.input, call.last) == (0, 1, 115, 10, False)
    stream.call_done(first[1], 8.0)
    last = stream.arrived(10.0)[0]
    assert (last.session, last.call, last.history, last.last) == (1, 2, 130, True)
    stream.call_done(last, 12.0)  # closed loop: a new session replaces the finished one at once
    new = stream.arrived(12.0)[0]
    assert (new.session, new.call, new.input, new.history) == (2, 0, 110, 0)
    script = Stream(scripted([{"arrival": 2.0, "input": 3, "output": 1}, {"arrival": 1.0, "input": 5, "output": 2}]))
    assert [r.input for r in script.arrived(5.0)] == [5, 3]
    with pytest.raises(ValueError):
        scripted([])


# ---- event simulation (design-B section 4, tests 10-15) -------------------------------------------

ONE_PAGE = Geometry(1, 1, (TensorSpec("kv", 8, 4096),))  # one layer and head; a token fills a 4 KiB page
TEN_MS = ConstantClock(10e6)


def tiny(segments, banks=4, ppb=4, partition="none"):
    """A toy stack of ``segments`` segments of banks x ppb pages."""
    return Device(banks=banks, pages_per_block=ppb, capacity_gib=segments * banks * ppb * 4096 / GIB,
                  partition=partition)


def sim_opts(**kwargs):
    return SimOptions(**({"m5": "off", "warmup_h": 0.0, "hours": .01, "retention_s": 0.0, "report_s": 3600.0}
                         | kwargs))


def programs(sim, cause):
    return sim.programs[CAUSE[cause]]


def test_10_read_disturb_refresh_tiny_device():
    """Banks 4, ppb 4, 8 segments of 16 pages; a 16-page request puts 4 pages in each block, so a dense
    step reads every block 4 times and X = 40 is reached every 10 steps. 35 steps give 3 refreshes of
    16 pages, 4 segment openings (the prefill and 3 relocations) and 16 block erases."""
    w = scripted([{"arrival": 0, "input": 16, "output": 35}])
    sim = simulate(ONE_PAGE, TEN_MS, w, tiny(8), sim_opts(policy="hbm_decode", x=40))
    r = sim.result
    assert r["threshold_steps"] == 10 and r["decode_steps"] == 35 and r["completed"] == 1
    assert r["refresh_events_rd"] == 3 and programs(sim, "refresh_rd") == 48 and r["refresh_to_kv"] == 3
    assert r["segment_erases"] == 4 and sim.block_erases == 16
    # In-place replay instead (spec section 5.7): the same segment is erased and rewritten 3 times.
    replay = simulate(ONE_PAGE, TEN_MS, w, tiny(8), sim_opts(policy="hbm_decode", x=40, refresh="replay"))
    assert replay.result["refresh_events_rd"] == 3 and programs(replay, "replay") == 48
    assert replay.result["segment_erases"] == 4 and replay.pec.max() == 4


def test_11_retention_rewrites_idle_entry():
    """An entry kept 50 h after its request is rewritten at 24 h and 48 h (T_ret = 24 h)."""
    w = scripted([{"arrival": 0, "input": 16, "output": 1, "retain_s": 50 * 3600}])
    sim = simulate(ONE_PAGE, TEN_MS, w, tiny(8), sim_opts(policy="hbm_decode", x=1e9, hours=60,
                                                           retention_s=24 * 3600))
    assert sim.result["refresh_events_ret"] == 2 and programs(sim, "refresh_ret") == 32
    assert sim.result["segment_erases"] == 3 and sim.live.sum() == 0  # dead after 50 h


def test_12_packed_without_pressure():
    """No capacity pressure: WA = 1 and segment erases = ceil(written pages / segment pages)."""
    w = scripted([{"arrival": .05 * i, "input": 16, "output": 3} for i in range(10)])
    sim = simulate(ONE_PAGE, TEN_MS, w, tiny(64), sim_opts(policy="hbm_decode", x=1e9))
    assert sim.result["wa"] == 1 and sim.result["segment_erases"] == math.ceil(160 / 16)
    # With decode writes: 12 prompt + 20 generated pages per request, row-aligned (rows of 4 pages).
    w = scripted([{"arrival": .5 * i, "input": 12, "output": 20} for i in range(10)])
    sim = simulate(ONE_PAGE, TEN_MS, w, tiny(64), sim_opts(policy="packed", x=1e9))
    assert programs(sim, "prefill") == 120 and programs(sim, "decode") == 200 and programs(sim, "padding") == 0
    assert sim.result["wa"] == 1 and sim.result["segment_erases"] == math.ceil(320 / 16)


def test_13_compaction_two_segments():
    """A = 12 and B = 4 pages fill segment 0; A dies; C = 16 needs space, so the victim segment 0 (live B)
    is compacted first: 4 pages copied, WA = (12 + 4 + 16 + 4) / 32 = 1.125."""
    w = scripted([{"arrival": 0, "input": 12, "output": 1}, {"arrival": 0, "input": 4, "output": 100},
                  {"arrival": .1, "input": 16, "output": 1}])
    sim = simulate(ONE_PAGE, TEN_MS, w, tiny(2), sim_opts(policy="hbm_decode", x=1e9, low_watermark=2))
    assert programs(sim, "compaction") == 4 and sim.result["wa"] == pytest.approx(1.125)
    assert sim.result["compaction_events"] == 1 and sim.pec.tolist() == [2, 1]


def test_14_dedicated_superblocks():
    """10 requests of 1,024 + 338 tokens (8B BF16) each own a superblock: 10 erases and erase_wa =
    262,144 / 43,520 = 6.02 (design-B's 5.95 counts the partial last page, 44,032 pages; it stays in HBM)."""
    geometry, _ = geometry_for(LLAMA8, "bf16")
    w = scripted([{"arrival": 10.0 * i, "input": 1024, "output": 338} for i in range(10)])
    sim = simulate(geometry, TEN_MS, w, Device(), sim_opts(policy="dedicated", x=1e9, hours=.05))
    r = sim.result
    assert r["segment_erases"] == 10 and r["erase_wa"] == pytest.approx(SUPERBLOCK / 43_520)
    assert r["pages_per_request"] == 43_520 and r["wa"] == 1
    # Channel partition: 64 MiB segments, 3 per request (erase_wa 1.13), one fresh channel per request.
    channel = simulate(geometry, TEN_MS, w, Device(partition="channel"),
                       sim_opts(policy="dedicated", x=1e9, hours=.05))
    assert channel.result["segment_erases"] == 30
    assert channel.result["erase_wa"] == pytest.approx(3 * 16384 / 43_520)
    assert len({int(s) % 16 for s in np.flatnonzero(channel.pec)}) == 10


@pytest.mark.parametrize("wear,expected", [("none", [100, 0, 0, 0]), ("dynamic", [25, 25, 25, 25])])
def test_15_wear_allocation(wear, expected):
    """4 segments, 100 sequential one-segment requests: the lowest free index reuses segment 0 (max PEC
    100); least-PEC allocation spreads the erases (25 each)."""
    w = scripted([{"arrival": float(i), "input": 16, "output": 1} for i in range(100)])
    sim = simulate(ONE_PAGE, TEN_MS, w, tiny(4), sim_opts(policy="hbm_decode", x=1e9, hours=.03, wear=wear))
    assert sim.pec.tolist() == expected and sim.result["segment_erases"] == 100


def test_layer_major_wave_replays_padding():
    """A static wave of 2 slots is laid out at its longest prompt (8 tokens): 16 pages, 4 of them padding.
    Its blocks see 12 live pages / 4 banks = 3 reads per step, so X = 15 replays the whole region in place
    every 5 steps, padding included (no skipping, spec section 5.7): 12 steps give 2 replays of 16 pages."""
    w = scripted([{"arrival": 0, "input": 8, "output": 12}, {"arrival": 0, "input": 4, "output": 12}])
    sim = simulate(ONE_PAGE, TEN_MS, w, tiny(8), sim_opts(policy="layer_major", x=15, slots=2))
    r = sim.result
    assert r["waves"] == 1 and programs(sim, "prefill") == 12 and programs(sim, "padding") == 4
    assert r["refresh_events_rd"] == 2 and programs(sim, "replay") == 32
    assert sim.seg_erases[CAUSE["wave"]] == 1 and sim.seg_erases[CAUSE["replay"]] == 2
    assert r["replay_buffer_mib"] == 16 * 4096 / 2 ** 20


def test_shared_prefix_sessions_and_eviction():
    # A shared prefix is written once; the second request reads it once at prefill (8 pages / 4 banks = 2
    # reads per block), then each step reads the prefix per request (2 x 8) plus both own 4-page entries.
    w = scripted([{"arrival": 0, "input": 4, "output": 2, "prefix": 0}] * 2, prefix_groups=1, prefix_tokens=8)
    sim = simulate(ONE_PAGE, TEN_MS, w, tiny(8), sim_opts(policy="hbm_decode", x=1e9))
    assert programs(sim, "prefill") == 16 and sim.reads.max() == 2 + 2 * (16 + 8) / 4
    once = simulate(ONE_PAGE, TEN_MS, w, tiny(8), sim_opts(policy="hbm_decode", x=1e9, prefix_reads="once"))
    assert once.reads.max() == 2 + 2 * (8 + 8) / 4
    # A session: call 0 writes history + input (12 pages) and its 4 generated pages; call 1 (0.5 s later)
    # appends only its 4 new input pages and reads the 16-page history once at prefill.
    s = Workload(input="fixed:4", output="fixed:4", calls=2, history="fixed:8", think_s=.5, concurrency=1)
    sim = simulate(ONE_PAGE, TEN_MS, s, tiny(16), sim_opts(policy="packed", x=1e9, hours=.57 / 3600))
    assert programs(sim, "prefill") == 16 and programs(sim, "decode") == 4 and sim.result["recompute_tokens"] == 0
    # Idle entries (kept 100 s) fill a 4-segment device; with a low watermark of 1 the oldest is evicted
    # whenever the last free segment is taken: requests 3..7 evict entries 0..4 (5 x 16 pages).
    w = scripted([{"arrival": float(i), "input": 16, "output": 1, "retain_s": 100.0} for i in range(8)])
    sim = simulate(ONE_PAGE, TEN_MS, w, tiny(4), sim_opts(policy="hbm_decode", x=1e9, hours=7.5 / 3600,
                                                           low_watermark=1))
    assert sim.result["evicted_gb"] == pytest.approx(80 * 4096 / 1e9) and sim.result["recompute_tokens"] == 80
    with pytest.raises(CapacityError):
        simulate(ONE_PAGE, TEN_MS, scripted([{"arrival": 0, "input": 80, "output": 1}]), tiny(4), sim_opts())


def test_simulation_matches_analytic_in_steady_state():
    """Closed loop B = 8, 8B BF16, packed allocation, 512 GiB (no pressure), retention and M5 off.

    (a) 32K + 338 tokens with decode writes, no refresh (338 < X / ppb = 3,906 steps): request rate, KV
    write and P/E per day within 1 % of analytic() (measured: 0.02 %).
    (b) Decode KV in HBM (hbm_decode), 32K prompts = 4 whole superblocks, X = 3e4 (117.2 steps):
    floor(512 x 256 / 3e4) = 4 refreshes per request in both models; P/E per day within 1 %."""
    geometry, _ = geometry_for(LLAMA8, "bf16")
    cases = ((Workload(input="fixed:32768", output="fixed:338", concurrency=8), 38.57e6, "packed", 1e6),
             (Workload(input="fixed:32768", output="fixed:512", concurrency=8), 100e6, "hbm_decode", 3e4))
    for w, tpot, policy, x in cases:
        clock = ConstantClock(tpot)
        sim = simulate(geometry, clock, w, Device(), SimOptions(policy=policy, x=x, hours=1.25, warmup_h=.25,
                                                                 m5="off", retention_s=0))
        ref = analytic(geometry, clock, w, Device(), policy=policy, x_values=(x,), retention_s=0, m5="off")[0]
        r = sim.result
        for key in ("req_per_s", "kv_write_gb_s", "avg_pec_per_day", "wa"):
            assert r[key] == pytest.approx(ref[key], rel=1e-2), key
        assert ref["refresh_to_kv"] == (0 if policy == "packed" else 4)
        assert r["refresh_to_kv"] == pytest.approx(ref["refresh_to_kv"], rel=1e-2)  # window edges: 3.996
        assert r["mean_batch"] == pytest.approx(8) and r["mean_context"] == pytest.approx(ref["mean_context"], rel=1e-3)


def test_simulation_refresh_with_decode_writes_is_below_analytic():
    """Packed with decode writes and refresh: 32K + 2,048 tokens, X = 1e5 (390.6 steps).

    analytic() gives floor(2,048 x 256 / 1e5) = 5 refreshes of every page, counting generated pages as if
    present from the first step. The simulation writes them as they are produced and refreshes relocated
    data when its (partly filled) destination segment reaches X, so it lands between 4 and 5 (4.49 measured
    over 2.5 h, see docs/design/phase2b-sim-plan.md)."""
    geometry, _ = geometry_for(LLAMA8, "bf16")
    w = Workload(input="fixed:32768", output="fixed:2048", concurrency=8)
    clock = ConstantClock(40e6)
    sim = simulate(geometry, clock, w, Device(), SimOptions(x=1e5, hours=1.5, warmup_h=.5, m5="off", retention_s=0))
    ref = analytic(geometry, clock, w, Device(), x_values=(1e5,), retention_s=0, m5="off")[0]
    assert ref["refresh_to_kv"] == 5 and 4 < sim.result["refresh_to_kv"] < 5
    assert sim.result["kv_write_gb_s"] == pytest.approx(ref["kv_write_gb_s"], rel=1e-2)
    assert .8 * ref["avg_pec_per_day"] < sim.result["avg_pec_per_day"] < ref["avg_pec_per_day"]


def test_m5_hook_per_quantum(monkeypatch):
    """interference() is called once per quantum with that quantum's programs and block erases per step;
    over the run the loads times the steps add up to every program and erase (no idle time here). Its
    extra delays the next quantum: TPOT' > TPOT, gaps hides part of fifo's delay, off adds nothing."""
    from hbf import endurance
    geometry, _ = geometry_for(LLAMA8, "bf16")
    w = Workload(input="fixed:32768", output="fixed:338", concurrency=8)
    clock = ConstantClock(38.57e6, layers=32)
    extra = {}
    for mode in ("fifo", "gaps", "off"):
        sim = endurance.Simulator(geometry, clock, w, Device(), SimOptions(hours=.25, warmup_h=.05, m5=mode,
                                                                            retention_s=0))
        calls = []

        def spy(load, *args, **kwargs):
            calls.append((load["programs"], load["erases"], sim.acc["steps"]))
            return interference(load, *args, **kwargs)
        monkeypatch.setattr(endurance, "interference", spy)
        r = sim.run()
        extra[mode] = (r["tpot_m5_ms"] - r["mean_tpot_ms"]) * 1e6
        if mode == "off":
            assert not calls and extra[mode] == 0
            continue
        steps = np.diff([0.0] + [call[2] for call in calls])
        assert len(calls) == sim.n["quanta"]
        assert np.dot([call[0] for call in calls], steps) == pytest.approx(sim.m5_mark[0], rel=1e-9)
        assert np.dot([call[1] for call in calls], steps) == pytest.approx(sim.m5_mark[1], rel=1e-9)
        assert r["max_step_extra_ms"] * 1e6 >= extra[mode] > 0
    assert extra["gaps"] < extra["fifo"]
