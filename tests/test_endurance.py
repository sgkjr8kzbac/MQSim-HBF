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
from hbf.endurance import (CAUSE, GIB, CapacityError, Geometry, SimOptions, VllmTracker, prefill_tok_s, simulate)
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
    assert replay.result["refreshes_per_request"] == pytest.approx(3.0)
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
    assert r["refreshes_per_request"] == pytest.approx(32 / 12)
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


def test_cli_simulation(tmp_path):
    out = tmp_path / "sim"
    main(["--no-calibrate", "--workload", "long32k", "--concurrency", "2", "--capacity-gib", "64", "--hours", ".05",
          "--warmup-h", ".01", "--report-s", "30", "--rd-threshold", "1e5", "--output", str(out)])
    with (out / "endurance_summary.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1 and rows[0]["policy"] == "packed" and float(rows[0]["mean_batch"]) == pytest.approx(2)
    assert float(rows[0]["analytic_avg_pec_per_day"]) > 0 and rows[0]["segments"] == "64"
    with (out / "endurance_timeseries.csv").open(newline="") as stream:
        series = list(csv.DictReader(stream))
    assert len(series) == 7 and {"t_s", "running", "tpot_m5_ms", "refresh_rd_gb", "free_segments"} <= set(series[0])
    with (out / "endurance_segments.csv").open(newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 64
    summary = summarize_run(out)
    assert set(summary) == set(METRICS) and summary["mean_batch"] == pytest.approx(2) and summary["stalls"] == 0
    meta = json.loads((out / "meta.json").read_text())
    assert meta["tool"] == "endurance" and meta["analytic"] is False and meta["hours"] == .05
    report = (out / "report.md").read_text()
    assert "# 寿命评估（事件模拟）" in report and "## 假设" in report and "第 5.7 节" in report
    assert "## 与解析模型的差别" in report


def test_cli_simulation_validation(tmp_path):
    parser = build_parser()
    with pytest.raises(ValueError, match="--hours"):
        configure(parser.parse_args(["--no-calibrate"]))
    with pytest.raises(ValueError, match="partition"):
        configure(parser.parse_args(["--hours", "2", "--alloc", "vllm_pool", "--partition", "die"]))
    with pytest.raises(ValueError, match="free-order"):
        configure(parser.parse_args(["--hours", "2", "--alloc", "vllm_pool", "--free-order", "bogus"]))
    with pytest.raises(ValueError, match="block-tokens"):
        configure(parser.parse_args(["--hours", "2", "--alloc", "vllm_pool", "--block-tokens", "0"]))
    with pytest.raises(ValueError, match="multiple"):
        configure(parser.parse_args(["--hours", "2", "--alloc", "vllm_pool", "--block-tokens", "10"]))
    with pytest.raises(ValueError, match="retention-h"):
        configure(parser.parse_args(["--hours", "2", "--retention-h", "-1"]))
    odd = tmp_path / "odd.json"
    odd.write_text(json.dumps({"name": "odd", "layers": 2, "hidden": 384, "q_heads": 6, "kv_heads": 1,
                               "head_dim": 64, "intermediate": 512, "vocab": 1000}))
    with pytest.raises(ValueError, match="multiple of q_heads"):
        configure(parser.parse_args(["--analytic", "--model", f"llama-3.1-8b,{odd}", "--format", "mla:indexer=1"]))
    with pytest.raises(ValueError, match="replay"):
        configure(parser.parse_args(["--hours", "2", "--alloc", "layer_major", "--refresh", "relocate"]))
    with pytest.raises(ValueError, match="warmup"):
        configure(parser.parse_args(["--hours", "1"]))  # the default warm-up is 1 h
    config = configure(parser.parse_args(["--hours", "2", "--wear", "remap", "--no-stagger",
                                          "--alloc", "vllm_pool", "--vllm-reuse", "lifo",
                                          "--prefill", "colocated", "--victim", "cost_benefit",
                                          "--pre-open", "--separate-prefix"]))
    assert config["analytic"] is False and config["wear"] == "remap" and config["stagger"] is False
    assert config["policy"] == "vllm_pool" and config["vllm_reuse"] == "lifo"
    assert config["prefill"] == "colocated" and config["victim"] == "cost_benefit"
    assert config["pre_open"] is True and config["separate_prefix"] is True
    assert {"models", "formats", "workloads", "pages_per_block", "x_values"} <= set(config)


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


# ---- test 16: vllm_pool tracker and simulation (design-B section 4.3, test 16) --------------------

def test_16_vllm_pool_tracker_and_simulation(tmp_path):
    """Allocate ids 0..7, free 2 and 5, allocate 2 with LIFO -> 2 overlaps; stranded = 2/8 until the
    whole range is free. Also checks VllmTracker edge cases and full Simulator / CLI integration."""
    tracker = VllmTracker(ids=8, pages_per_id=1, pages_per_block=8, banks=1, order="fresh", reuse="lifo")
    ids = tracker.allocate(8)
    assert ids == list(range(8)) and tracker.counts["overlap"] == 0 and tracker.counts["erases"] == 1
    assert tracker.stranded() == 0.0
    tracker.free([2, 5])
    assert tracker.stranded() == pytest.approx(2 / 8)
    reused = tracker.allocate(2)
    assert reused == [5, 2]
    assert tracker.counts["overlap"] == 2 and tracker.counts["skip"] == 0 and tracker.counts["live_erase"] == 0
    assert tracker.counts["legalize_pages"] == 8
    assert tracker.stranded() == 0.0
    tracker.free([2, 5])
    assert tracker.stranded() == pytest.approx(2 / 8)
    tracker.free([0, 1, 3, 4, 6, 7])
    assert tracker.stranded() == 0.0  # whole range is free
    # Consecutive writes into an already-filled block are both overlaps (not legal); legal append after overlap.
    t2 = VllmTracker(ids=8, pages_per_id=1, pages_per_block=8, banks=1, order="fresh", reuse="fifo")
    t2.allocate(8)
    t2.free([5, 6])
    assert t2.allocate(2) == [5, 6] and t2.counts["overlap"] == 2
    t3 = VllmTracker(ids=8, pages_per_id=1, pages_per_block=8, banks=1, order="fresh", reuse="fifo")
    t3.allocate(4)
    t3.free([1])
    t3.write([1, 4])  # 1 is overlap (< fill=4), 4 is legal append (== fill=4)
    assert t3.counts["overlap"] == 1 and t3.counts["skip"] == 0 and t3.counts["live_erase"] == 0
    # Grouped fast path (banks % pages_per_id == 0) matches ungrouped path on identical operations.
    g = VllmTracker(ids=16, pages_per_id=4, pages_per_block=4, banks=8, order="shuffled:5", reuse="lifo")
    u = VllmTracker(ids=16, pages_per_id=4, pages_per_block=4, banks=8, order="shuffled:5", reuse="lifo")
    u._grouped, u._scale = False, 1
    u.fill, u.live = np.zeros(8 * u.nblocks, dtype=np.int64), np.zeros(8 * u.nblocks, dtype=np.int64)
    for fn, arg in (("allocate", 6), ("free", None), ("allocate", 4)):
        if fn == "allocate":
            a, b = g.allocate(arg), u.allocate(arg)
            assert a == b
            saved = a[:3]
        else:
            g.free(saved)
            u.free(saved)
        assert g.stranded() == pytest.approx(u.stranded())
        assert g.counts == u.counts
    # End-to-end Simulator: 2 segments of 8 pages (16 ids, banks=1, ppb=8), 8 tokens per page so <8 decode
    # steps add no new block ids. Requests 0..7 arrive at t=0 (1 page each -> ids 0..7). Requests 2 and 5
    # finish after 1 step (t=0.01s), leaving 2/16 stranded for [0.01, 0.05) = 0.04s. At t=0.05s two new
    # 1-page requests arrive and reuse [5, 2] (LIFO) -> 2 overlaps, stranded drops to 0. Horizon = 0.10s.
    eight_tpp = Geometry(1, 1, (TensorSpec("kv", 16, 256),))  # 512 B/tok -> 8 tokens/page
    reqs = [{"arrival": 0.0, "input": 8, "output": 1 if i in (2, 5) else 7} for i in range(8)]
    reqs += [{"arrival": 0.05, "input": 8, "output": 2}, {"arrival": 0.05, "input": 8, "output": 2}]
    sim = simulate(eight_tpp, TEN_MS, scripted(reqs), tiny(2, banks=1, ppb=8),
                   sim_opts(policy="vllm_pool", x=1e9, hours=0.1 / 3600, low_watermark=1, vllm_reuse="lifo",
                            align="none"))
    r = sim.result
    assert r["legal"] is False and r["violations_overlap"] == 2 and r["violations_skip"] == 0
    assert r["violations_live_erase"] == 0 and r["legalize_gb"] > 0
    # 2/16 stranded over 0.04s out of 0.10s -> time average = 0.05.
    assert r["stranded_frac"] == pytest.approx((2 / 16) * (0.04 / 0.10), rel=1e-6)
    # CLI end-to-end check for vllm_pool (and report header echo of non-default options).
    out = tmp_path / "vllm"
    main(["--no-calibrate", "--alloc", "vllm_pool", "--vllm-reuse", "lifo", "--workload", "long32k",
          "--concurrency", "2", "--capacity-gib", "64", "--hours", ".05", "--warmup-h", ".01",
          "--report-s", "30", "--victim", "cost_benefit", "--pre-open", "--separate-prefix",
          "--prefill", "colocated", "--output", str(out)])
    with (out / "endurance_summary.csv").open(newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row["policy"] == "vllm_pool" and row["legal"] == "False"
    assert float(row["violations_overlap"]) > 0 and float(row["legalize_gb"]) > 0
    report = (out / "report.md").read_text()
    assert "vLLM 空闲块池" in report and "vLLM 重用 `lifo`" in report
    assert "回收选段 `cost_benefit`" in report and "prefill `colocated`" in report
    assert "`--pre-open`" in report and "`--separate-prefix`" in report


def test_vllm_tracker_live_erase_and_legalize_edge_cases():
    """(a) write([0, 1]) on a live block preserves fill (no shrinkage) and counts live_erase + overlap;
    (b) intra-batch violation [2, 1] on an empty block charges effective fill;
    (c) ungrouped multi-block IDs charge only offending blocks."""
    # (a) Block 0 filled 0..7; free 0 and 1 (live=6, fill=8); allocate(2) writes [0, 1] -> live_erase=1, overlap=1, fill stays 8.
    t = VllmTracker(ids=8, pages_per_id=1, pages_per_block=8, banks=1, order="fresh", reuse="fifo")
    t.allocate(8)
    t.free([0, 1])
    assert t.allocate(2) == [0, 1]
    assert t.counts["live_erase"] == 1 and t.counts["overlap"] == 1 and t.counts["legalize_pages"] == 8
    assert t.fill[0] == 8
    t.free([3])
    assert t.stranded() == pytest.approx(1 / 8)
    # (b) Empty block: write([2, 1]) -> skip=1 (pos 2), overlap=1 (pos 1 <= 2), legalize_pages = 3.
    t2 = VllmTracker(ids=8, pages_per_id=1, pages_per_block=8, banks=1, order="fresh", reuse="fifo")
    t2.write([2, 1])
    assert t2.counts["skip"] == 1 and t2.counts["overlap"] == 1 and t2.counts["legalize_pages"] == 3
    assert t2.fill[0] == 3
    # (c) Ungrouped multi-block ID (banks=3, pages_per_id=2, ppb=4 -> _grouped=False).
    # ID 0 -> (bank=0, pos=0), (bank=1, pos=0). ID 1 -> (bank=2, pos=0), (bank=0, pos=1).
    # ID 2 -> (bank=1, pos=1), (bank=2, pos=1).
    # Allocate IDs 0, 1, 2 (fills: bank0=2, bank1=2, bank2=2; live=2 each).
    # Free ID 1 (bank2 live=1, bank0 live=1) and ID 2 (bank1 live=1, bank2 live=0!).
    # Now write ID 1: (bank=2, pos=0) has live=0 -> LEGAL erase (must NOT be charged to legalize_pages),
    # while (bank=0, pos=1) has fill=2, live=1 -> OVERLAP (charges only bank 0's fill=2).
    t3 = VllmTracker(ids=6, pages_per_id=2, pages_per_block=4, banks=3, order="fresh", reuse="fifo")
    assert not t3._grouped
    t3.allocate(3)
    t3.free([1, 2])
    assert t3.live[2 * t3.nblocks] == 0 and t3.fill[2 * t3.nblocks] == 2
    t3.write([1])
    assert t3.counts["overlap"] == 1 and t3.counts["live_erase"] == 0
    assert t3.counts["legalize_pages"] == 2  # only bank 0 (fill=2), not legally erased bank 2


def test_vllm_pool_outstanding_ids_and_kivi_prefix_residual():
    """Admission reserves outstanding decode IDs of running requests; KIVI residual is subtracted once."""
    # 2 segments of 8 pages = 16 IDs (1 tok/page, bt=1). Req 0 needs 2 prefill + 6 decode = 8 IDs;
    # Req 1 needs 2 prefill + 6 decode = 8 IDs; Req 2 needs 2 + 6 = 8 IDs -> must stall (not crash during decode).
    reqs = [{"arrival": 0.0, "input": 2, "output": 6} for _ in range(3)]
    sim = simulate(ONE_PAGE, TEN_MS, scripted(reqs), tiny(2, banks=1, ppb=8),
                   sim_opts(policy="vllm_pool", x=1e9, hours=0.2 / 3600, low_watermark=1, align="none"))
    assert sim.result["completed"] == 3 and sim.result["stalls"] >= 1
    # KIVI-2 with residual=128 (tpp=(128, 256, 128, 256), default block_tokens=256) and shared prefix=512,
    # input=512, output=128: total HBF tokens = 512 + 512 + 128 - 128 = 1024 -> 4 vLLM blocks
    # (not 3 from double-subtracting residual on both prefix and own tokens).
    geo, _ = geometry_for(LLAMA8, "kivi2")
    w = scripted([{"arrival": 0.0, "input": 512, "output": 128, "prefix": 0}], prefix_groups=1, prefix_tokens=512)
    ksim = simulate(geo, TEN_MS, w, Device(capacity_gib=64),
                    sim_opts(policy="vllm_pool", x=1e9, hours=0.05))
    assert ksim.vllm.counts["writes"] == 4


def test_separate_prefix_tight_capacity_reclaim():
    """--separate-prefix accounts for both frontiers during admission and reclaim under tight capacity."""
    # 3 segments of 16 pages, low_watermark=1: prefix 0 (8p, retain=100s) on prefix frontier, private (8p)
    # on main frontier; subsequent requests with a second prefix force eviction/reclaim without CapacityError.
    reqs = [
        {"arrival": 0.0, "input": 8, "output": 2, "prefix": 0, "retain_s": 100.0},
        {"arrival": 0.05, "input": 8, "output": 2, "prefix": 0, "retain_s": 100.0},
        {"arrival": 0.10, "input": 8, "output": 2, "prefix": 1, "retain_s": 100.0},
        {"arrival": 0.15, "input": 8, "output": 2, "prefix": 1, "retain_s": 100.0},
        {"arrival": 0.20, "input": 8, "output": 2, "prefix": 0, "retain_s": 0.0},
    ]
    w = scripted(reqs, prefix_groups=2, prefix_tokens=8, retain_s=100.0)
    sim = simulate(ONE_PAGE, TEN_MS, w, tiny(3),
                   sim_opts(policy="hbm_decode", x=1e9, hours=0.5 / 3600, low_watermark=1, separate_prefix=True))
    assert sim.result["completed"] == 5 and sim.result["evicted_gb"] > 0


def test_dedicated_evicts_open_idle_segments():
    """Idle dedicated entries with <1 full segment (state == OPEN) are evicted cleanly under tight capacity."""
    reqs = [{"arrival": 0.1 * i, "input": 4, "output": 1, "retain_s": 100.0} for i in range(6)]
    sim = simulate(ONE_PAGE, TEN_MS, scripted(reqs), tiny(3),
                   sim_opts(policy="dedicated", x=1e9, hours=1.0 / 3600, low_watermark=1))
    assert sim.result["completed"] == 6 and sim.result["evicted_gb"] > 0


def test_session_short_think_time_no_retroactive_reads():
    """A session resuming inside the same quantum under hbm_decode flushes call 0's reads before call 1 writes."""
    # Call 0: history 4 + input 4 = 8 pages, decodes 4 steps -> 8 * 4 / 4 = 8 block reads.
    # think_s = 0.02s (< quantum_s = 1.0s). Call 1: prefill reads existing 8 pages once (+8/4 = 2),
    # writes 8 new pages (call 0's 4 output + call 1's 4 input -> 16 pages total), decodes 4 steps (+16*4/4 = 16).
    # Exact total block reads on seg 0 = 8 + 2 + 16 = 26 (without flushing in _complete, call 0's 4 steps would
    # retroactively hit call 1's 8 new pages -> 34).
    s = Workload(input="fixed:4", output="fixed:4", calls=2, history="fixed:4", think_s=.02, concurrency=1)
    sim = simulate(ONE_PAGE, TEN_MS, s, tiny(8),
                   sim_opts(policy="hbm_decode", x=1e9, hours=0.11 / 3600, quantum_s=1.0, align="none"))
    assert sim.reads[0] == pytest.approx(26.0)


# ---- optional flags from design-B (phase2b-sim-plan.md next step 3) -------------------------------

def test_prefill_colocated_adds_serial_admission_time():
    geometry, _ = geometry_for(LLAMA8, "bf16")
    w = Workload(input="fixed:4096", output="fixed:32", concurrency=1)
    clock = ConstantClock(10e6)
    dis = simulate(geometry, clock, w, Device(capacity_gib=64),
                   sim_opts(hours=10.0 / 3600, prefill="disaggregated", stagger=False), model=LLAMA8, gpu=B200)
    col = simulate(geometry, clock, w, Device(capacity_gib=64),
                   sim_opts(hours=10.0 / 3600, prefill="colocated", stagger=False), model=LLAMA8, gpu=B200)
    per_admit = 4096 / prefill_tok_s(LLAMA8, B200, 4096)
    assert col.result["prefill_s"] == pytest.approx(col.n["admitted"] * per_admit, rel=1e-9)
    assert dis.result["prefill_s"] == 0.0
    assert col.result["req_per_s"] < dis.result["req_per_s"]
    # Multi-stack scales serial prefill time by stacks; new prefix tokens are charged too.
    wp = scripted([{"arrival": 0.0, "input": 1024, "output": 4, "prefix": 0}], prefix_groups=1, prefix_tokens=2048)
    s2 = simulate(geometry, clock, wp, Device(capacity_gib=64),
                  sim_opts(hours=1.0 / 3600, prefill="colocated", stacks=2), model=LLAMA8, gpu=B200)
    expected_p = 2 * (1024 + 2048) / prefill_tok_s(LLAMA8, B200, 3072)
    assert s2.result["prefill_s"] == pytest.approx(expected_p, rel=1e-9)
    # Warm-up boundary: _quantum breaks when colocated prefill crosses warm, keeping snapshot close to warm.
    w_warm = Workload(input="fixed:4096", output="fixed:2", concurrency=1)
    sw = simulate(geometry, clock, w_warm, Device(capacity_gib=64),
                  sim_opts(hours=2.0 / 3600, warmup_h=0.5 / 3600, prefill="colocated", stagger=False),
                  model=LLAMA8, gpu=B200)
    assert sw.warm["t"] == pytest.approx(0.5, abs=0.1)
    with pytest.raises(ValueError, match="model and gpu"):
        simulate(geometry, clock, w, Device(capacity_gib=64), sim_opts(prefill="colocated"))


def test_victim_cost_benefit_prefers_older_cold_segment():
    """Segment 0 (written at t=0): 8 active + 8 dead (u=0.5, age=100s -> score 33.3).
    Segment 1 (written at t=99s): 4 active + 12 dead (u=0.25, age=1s -> score 0.6).
    At t=100s, an 8-page request opens segment 2 (room=8, free=0 < low_watermark=1) triggering reclaim:
    greedy picks segment 1 (4 pages copied); cost_benefit picks segment 0 (8 pages copied)."""
    reqs = [
        {"arrival": 0.0, "input": 8, "output": 20000},   # active in seg 0
        {"arrival": 0.0, "input": 8, "output": 1},       # dies at t=0.01s in seg 0
        {"arrival": 99.0, "input": 4, "output": 20000},  # active in seg 1
        {"arrival": 99.0, "input": 12, "output": 1},     # dies at t=99.01s in seg 1
        {"arrival": 100.0, "input": 8, "output": 100},   # opens seg 2 (room=8, free=0 -> triggers reclaim)
    ]
    greedy = simulate(ONE_PAGE, TEN_MS, scripted(reqs), tiny(3),
                      sim_opts(policy="hbm_decode", x=1e9, hours=101 / 3600, low_watermark=1, victim="greedy"))
    cb = simulate(ONE_PAGE, TEN_MS, scripted(reqs), tiny(3),
                  sim_opts(policy="hbm_decode", x=1e9, hours=101 / 3600, low_watermark=1, victim="cost_benefit"))
    assert programs(greedy, "compaction") == 4
    assert programs(cb, "compaction") == 8


def test_pre_open_removes_non_replay_erases_from_m5():
    geometry, _ = geometry_for(LLAMA8, "bf16")
    w = Workload(input="fixed:32768", output="fixed:338", concurrency=8)
    clock = ConstantClock(38.57e6, layers=32)
    base = simulate(geometry, clock, w, Device(),
                    SimOptions(hours=.2, warmup_h=.05, m5="fifo", t_bers=10e6, retention_s=0, pre_open=False))
    pre = simulate(geometry, clock, w, Device(),
                   SimOptions(hours=.2, warmup_h=.05, m5="fifo", t_bers=10e6, retention_s=0, pre_open=True))
    assert pre.result["tpot_m5_ms"] < base.result["tpot_m5_ms"]
    assert pre.result["max_step_extra_ms"] < base.result["max_step_extra_ms"]
    assert pre.m5_erases == 0 and pre.block_erases > 0
    # In-place replay erases cannot be pre-opened and still hit M5.
    rw = scripted([{"arrival": 0, "input": 16, "output": 35}])
    rep = simulate(ONE_PAGE, TEN_MS, rw, tiny(8),
                   sim_opts(policy="hbm_decode", x=40, refresh="replay", m5="fifo", pre_open=True))
    assert rep.m5_erases == 3 * 4  # 3 replays x 4 banks; initial open erase was pre-opened


def test_separate_prefix_avoids_dragging_cold_neighbors():
    """Hot 8-page prefix (2 readers -> 4 block reads/step) + cold 8-page private entry (1 reader -> 2 block
    reads/step) in 16-page segments at X=40 over 15 steps:
    - shared frontier: both share seg 0 (6 reads/step -> refreshes at step 7 and 14, copying 16 pages each = 32+);
    - separate_prefix: prefix owns seg 0 (4 reads/step -> 1 refresh of 8 pages at step 10), private stays on seg 1
      (2 reads/step -> 30 < 40, 0 refreshes)."""
    reqs = [{"arrival": 0, "input": 8, "output": 15, "prefix": 0},
            {"arrival": 0, "input": 0, "output": 15, "prefix": 0}]
    # Note: input must be >= 1 in Workload/scripted; use input=1 for second reader (1/4 = 0.25 reads/step).
    reqs[1]["input"] = 1
    w = scripted(reqs, prefix_groups=1, prefix_tokens=8)
    shared = simulate(ONE_PAGE, TEN_MS, w, tiny(8),
                      sim_opts(policy="hbm_decode", x=40, separate_prefix=False, align="none"))
    sep = simulate(ONE_PAGE, TEN_MS, w, tiny(8),
                   sim_opts(policy="hbm_decode", x=40, separate_prefix=True, align="none"))
    assert programs(sep, "refresh_rd") == 8  # only the 8-page prefix refreshes once at step 10
    assert programs(shared, "refresh_rd") > programs(sep, "refresh_rd")


def test_wear_remap_rotates_static_waves():
    w = scripted([{"arrival": float(i), "input": 16, "output": 1} for i in range(100)])
    none = simulate(ONE_PAGE, TEN_MS, w, tiny(4),
                    sim_opts(policy="layer_major", x=1e9, hours=.03, slots=1, wear="none"))
    remap = simulate(ONE_PAGE, TEN_MS, w, tiny(4),
                     sim_opts(policy="layer_major", x=1e9, hours=.03, slots=1, wear="remap"))
    assert none.pec.tolist() == [100, 0, 0, 0]
    assert remap.pec.tolist() == [25, 25, 25, 25]
    assert remap.result["wa"] == none.result["wa"]
    # Wave 0 suffers 2 replays (seg 0 reaches PEC=3); waves 1..3 have 0 replays (segs 1..3 PEC=1).
    # Wave 4: remap rotates back to seg 0 ([4, 1, 1, 1]), while dynamic picks least-PEC seg 1 ([3, 2, 1, 1]).
    asym = scripted([{"arrival": 0.0, "input": 16, "output": 25}]
                    + [{"arrival": float(i), "input": 16, "output": 1} for i in range(1, 5)])
    r_remap = simulate(ONE_PAGE, TEN_MS, asym, tiny(4),
                       sim_opts(policy="layer_major", x=40, hours=.01, slots=1, wear="remap"))
    r_dyn = simulate(ONE_PAGE, TEN_MS, asym, tiny(4),
                     sim_opts(policy="layer_major", x=40, hours=.01, slots=1, wear="dynamic"))
    assert r_remap.pec.tolist() == [4, 1, 1, 1]
    assert r_dyn.pec.tolist() == [3, 2, 1, 1]


# ---- Part 2: K1 page formats in hbf.endurance (MLA, fused int/KIVI) -------------------------------

def test_k1_formats_demand_matches_phase_a_and_kv_bytes():
    """Geometry.demand uses attended_dims and logical_dims identically to pages.layer_pages;
    kv_bytes_per_token counts HBF bytes only; analytic and simulate work on MLA and fused formats."""
    from hbf.access import LAYOUTS, Shape, generate
    from hbf.config import load
    from hbf.formats import parse_format
    from hbf.model import ModelSpec
    from hbf.pages import iter_layer_pages

    cfg = load("ocp")
    mla_model = ModelSpec(name="mla-tiny", layers=4, hidden=2048, q_heads=16, kv_heads=1, head_dim=128,
                          intermediate=4096, vocab=32000)
    for model, fmt_name, ctx in (
        (mla_model, "mla", 2048),
        (mla_model, "mla:split=1", 2048),
        (mla_model, "mla_fp8", 2048),
        (mla_model, "mla:indexer=1", 2048),
        (mla_model, "mla:indexer_hbm=1", 2048),
        (LLAMA8, "int4:fused=1", 2048),
        (LLAMA8, "kivi2:fused=1", 2048),
        (LLAMA8, "kivi2:fused=1,pow2=1", 2048),
    ):
        geo, fmt = geometry_for(model, fmt_name)
        layout = LAYOUTS["layer_contiguous"](cfg, Shape(1, model.kv_heads, model.head_dim, ctx), tensors=fmt.tensors)
        (pages,) = iter_layer_pages(cfg, layout, generate(cfg, layout, "dense"))
        d = geo.demand([ctx])
        assert d["attended_elements"] == pages.attended_elements, fmt_name
        assert d["hbf_elements"] == pages.hbf_elements, fmt_name
        assert d["dequant_ops"] == pytest.approx(pages.dequant_ops), fmt_name
        assert d["burst_bytes"] == int(pages.burst.sum()), fmt_name
        assert d["hbm_bytes"] == pages.hbm_bytes, fmt_name
        assert d["attended_tokens"] == pages.attended_tokens, fmt_name
    # kv_bytes_per_token counts HBF bytes only (excludes KIVI HBM residual and HBM DSA indexer).
    kivi_geo, _ = geometry_for(LLAMA8, "kivi2:fused=1")
    assert kivi_geo.hbf_token_bytes == 32 * 8 * 96  # fused K+V+scales = 96 B/head; residual in HBM excluded
    assert kivi_geo.token_bytes > kivi_geo.hbf_token_bytes
    idx_hbm_geo, _ = geometry_for(mla_model, "mla:indexer_hbm=1")
    idx_hbf_geo, _ = geometry_for(mla_model, "mla:indexer=1")
    assert idx_hbm_geo.hbf_token_bytes == 4 * 1 * 1152
    assert idx_hbf_geo.hbf_token_bytes == 4 * 1 * (1152 + 132)
    # Analytic and simulation run end-to-end on MLA, mla_fp8, int4:fused=1, kivi2:fused=1.
    for model, fmt_name in ((mla_model, "mla"), (mla_model, "mla_fp8"),
                            (LLAMA8, "int4:fused=1"), (LLAMA8, "kivi2:fused=1")):
        geo, fmt = geometry_for(model, fmt_name)
        clock = Clock(model, B200, geo, kv_format=fmt.name)
        w = Workload(input="fixed:2048", output="fixed:128", concurrency=4)
        a = analytic(geo, clock, w, Device(capacity_gib=64), x_values=(1e6,), **no_m5())[0]
        s = simulate(geo, clock, w, Device(capacity_gib=64),
                     sim_opts(hours=.05, warmup_h=.01, x=1e6), model=model, gpu=B200).result
        assert a["kv_bytes_per_token"] == geo.hbf_token_bytes
        assert s["kv_bytes_per_token"] == geo.hbf_token_bytes
        assert a["avg_pec_per_day"] > 0 and s["avg_pec_per_day"] > 0
        assert s["kv_write_gb_s"] == pytest.approx(a["kv_write_gb_s"], rel=.05), fmt_name
