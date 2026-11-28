"""Tests for R6: frontend/timeline.py, lifetime/counts.py, scaled MQSim vs Python verification, and extrapolation."""
import pytest

from hbf.backends.mqsim.parse import block_counts
from hbf.backends.mqsim.run import simulate as mqsim_simulate
from hbf.backends.mqsim.schedule import batches_from_trace
from hbf.device.config import load
from hbf.frontend.addrtrace import validate
from hbf.frontend.kv.formats import parse_format
from hbf.frontend.timeline import build_timeline
from hbf.frontend.trace.requests import Workload, load_workload, parse_length
from hbf.lifetime.clock import ReadTimeClock, StepClock, StepTime
from hbf.lifetime.counts import compare_block_counts, counts_from_trace, rates_from_counts
from hbf.lifetime.geometry import GIB, Device, Geometry, geometry_for
from hbf.lifetime.sim import SimOptions, simulate as sim_simulate
from hbf.models import load_model
from hbf.registry import REGISTRY, ensure_loaded


class _FixedClock(StepClock):
    def __init__(self, step_s, layers):
        self.step_s, self.layers = float(step_s), int(layers)

    def describe(self):
        return {"clock": "fixed", "step_s": self.step_s, "layers": self.layers}

    def step(self, contexts, partitions=None, partition_banks=None, hbm_tokens=None):
        if not len(contexts):
            return None
        tpot_ns = self.step_s * 1e9
        return StepTime(tpot_ns=tpot_ns, gpu_ns=0.0, hbm_tpot_ns=tpot_ns,
                        h_ns=tpot_ns / self.layers, gaps_ns=(0.0,) * self.layers,
                        read_pages=0.0, rounds=1, batch=len(contexts))


def test_timeline_policies_and_retention_validate():
    """R6a: packed, dedicated, relocate, replay, and retention refresh all produce valid AddressTraces."""
    cfg = load("16x64", channels=1, chips=1, dies=1, planes=4, blocks_per_plane=10, pages_per_block=4,
               aggregate_gb_s=64, plane_rule="independent", coalesce_ns=None, read_plane_ns=None)
    geo = Geometry(2, 2, parse_format("bf16", 128).tensors, cfg.page_bytes)
    wl = Workload(name="t", input=parse_length("fixed:64"), output=parse_length("fixed:24"),
                  arrival="closed", concurrency=2, seed=3)

    for policy, refresh, ret_s in (
        ("packed", "relocate", 0.0),
        ("packed", "replay", 0.0),
        ("dedicated", "relocate", 0.0),
        ("packed", "relocate", 0.10),
    ):
        trace, stats = build_timeline(
            cfg, geo, wl, steps=60, step_s=0.01, policy=policy,
            x=64, retention_s=ret_s, refresh=refresh, max_segments=8,
        )
        validate(cfg, trace)
        assert stats["completed"] >= 2
        assert stats["total_programs"] > 0
        if ret_s > 0:
            assert stats["refresh_events_ret"] >= 1


@pytest.mark.parametrize("case", [
    dict(name="long32k_relocate", layers=2, heads=2, blocks=10, segs=8, steps=240,
         wl=Workload(name="long32k_s", input=parse_length("fixed:256"), output=parse_length("fixed:48"),
                     arrival="closed", concurrency=4, seed=7),
         refresh="relocate"),
    dict(name="agentic_relocate", layers=4, heads=2, blocks=18, segs=16, steps=320,
         wl=Workload(name="agentic_s", input=parse_length("fixed:32"), output=parse_length("fixed:16"),
                     arrival="closed", concurrency=2, calls=6, history=parse_length("fixed:128"),
                     think_s=0.02, seed=11),
         refresh="relocate"),
    dict(name="long32k_replay", layers=2, heads=2, blocks=10, segs=8, steps=240,
         wl=Workload(name="long32k_r", input=parse_length("fixed:256"), output=parse_length("fixed:48"),
                     arrival="closed", concurrency=4, seed=7),
         refresh="replay"),
])
def test_r6b_scaled_mqsim_matches_python_simulator(binary, tmp_path, case):
    """R6b: MQSim T-HBFW4 counts match timeline block-for-block and match Python Simulator within 5%."""
    cfg = load("16x64", channels=2, chips=2, dies=1, planes=4,
               blocks_per_plane=case["blocks"], pages_per_block=8,
               aggregate_gb_s=64, plane_rule="independent", coalesce_ns=None, read_plane_ns=None)
    geo = Geometry(case["layers"], case["heads"], parse_format("bf16", 128).tensors, cfg.page_bytes)
    step_s = 0.01

    trace, tl_stats = build_timeline(
        cfg, geo, case["wl"], steps=case["steps"], step_s=step_s, policy="packed",
        x=256, retention_s=0.0, refresh=case["refresh"], low_watermark=2, max_segments=case["segs"],
    )
    py_counts, _ = counts_from_trace(cfg, trace)
    batches = batches_from_trace(cfg, trace.chunks)
    out = tmp_path / case["name"]
    _, _, _, mq_stats = mqsim_simulate(cfg, batches, out, binary, hbf_write=True)
    assert mq_stats["Total_GC_Executions"] == 0

    mq_counts = block_counts(out / "block_counts.csv")
    cmp_res = compare_block_counts(mq_counts, py_counts)
    assert cmp_res["exact_match"], cmp_res

    mq_rates = rates_from_counts(
        mq_counts,
        total_blocks=case["segs"] * cfg.units,
        pages_per_block=cfg.pages_per_block,
        page_bytes=cfg.page_bytes,
        duration_s=tl_stats["duration_s"],
        kv_programs=tl_stats["host_programs"],
        refresh_programs=tl_stats["refresh_programs"],
    )
    dev = Device(
        banks=cfg.units, pages_per_block=cfg.pages_per_block, page_bytes=cfg.page_bytes,
        capacity_gib=case["segs"] * cfg.units * cfg.pages_per_block * cfg.page_bytes / GIB,
    )
    opts = SimOptions(
        policy="packed", x=256.0, retention_s=0.0,
        hours=tl_stats["duration_s"] / 3600.0, warmup_h=0.0,
        quantum_s=step_s, report_s=max(tl_stats["duration_s"], 1.0),
        rd_slack=0.0, refresh=case["refresh"], low_watermark=2, m5="off",
    )
    sim = sim_simulate(geo, _FixedClock(step_s, case["layers"]), case["wl"], dev, opts)
    sres = sim.result

    pec_rel = abs(mq_rates["avg_pec_opened_per_day"] - sres["avg_pec_per_day"]) / sres["avg_pec_per_day"]
    ref_rel = abs(mq_rates["refresh_to_kv"] - sres["refresh_to_kv"]) / sres["refresh_to_kv"]
    assert pec_rel < 0.05, f"{case['name']} PEC error {pec_rel:.2%}"
    assert ref_rel < 0.05, f"{case['name']} refresh_to_kv error {ref_rel:.2%}"


def test_r6c_extrapolation_core_clock_and_ext_tpot():
    """R6c: full-scale extrapolation works with both core ReadTimeClock and ext/system TPOT Clock."""
    ensure_loaded()
    model = load_model("llama-3.1-8b")
    geo, _ = geometry_for(model, "bf16")
    dev = Device(banks=1024, pages_per_block=256, page_bytes=4096, capacity_gib=512.0)
    wl = load_workload("long32k", concurrency=8, seed=1)

    core_clk = ReadTimeClock(geo, model.layers, compute_ns=2_500_000.0)
    tpot_clk = REGISTRY.get_clock("tpot")(
        model, REGISTRY.get_evaluator("system", required=True).load_gpu("b200"), geo, kv_format="bf16"
    )
    opts = SimOptions(policy="packed", x=1e5, hours=1.0, warmup_h=0.2, m5="gaps")
    r_core = sim_simulate(geo, core_clk, wl, dev, opts).result
    r_ext = sim_simulate(geo, tpot_clk, wl, dev, opts).result

    assert 300 < r_core["avg_pec_per_day"] < 500
    assert 300 < r_ext["avg_pec_per_day"] < 500
    assert abs(r_core["avg_pec_per_day"] - r_ext["avg_pec_per_day"]) / r_ext["avg_pec_per_day"] < 0.10
