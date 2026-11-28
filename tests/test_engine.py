"""The numpy analysis engine against the pure-Python reference (schedule + analyze)."""
import itertools

import pytest

from hbf.frontend.kv.layouts import LAYOUTS
from hbf.frontend.kv.tensors import Shape, TensorSpec
from hbf.frontend.trace.synthetic import generate
from hbf.backends.rounds.analysis import Protocol, analyze_accesses, locate
from hbf.device.config import load
from hbf.device.address import place
from hbf.frontend.pages import iter_layer_pages, schedule
from tests.reference import analyze

TWO_BIT = (TensorSpec("K", 2, 128), TensorSpec("V", 2, 128))
KIVI_LIKE = (TensorSpec("K", 2, 128), TensorSpec("K_scale", 16, 8), TensorSpec("V", 2, 128), TensorSpec("V_scale", 16, 8))
ODD = (TensorSpec("K", 3, 128), TensorSpec("V", 16, 100))
TOPOLOGIES = {"ocp": {}, "ocp-8ch": {}, "16x64": {}, "256x4": {}, "4x1": {"planes": 16, "plane_rule": "independent"}}
MODES = ("dense", "uniform_pages", "uniform_tokens", "sink_recent", "hot_unit", "write_then_read")


def workload(topology, layout_name, mode, tensors, batch_pages=8192):
    cfg = load(topology, batch_pages=batch_pages, **TOPOLOGIES[topology])
    layout = LAYOUTS[layout_name](cfg, Shape(2, 4, 128, 3000), tensors=tensors)
    accesses = list(generate(cfg, layout, mode, steps=2, fraction=.07, seed=3, count=40, select_tokens=24))
    return cfg, layout, accesses


@pytest.mark.parametrize("topology,layout_name,tensors", list(itertools.product(
    TOPOLOGIES, LAYOUTS, (None, KIVI_LIKE, ODD))))
def test_layer_pages_equal_scheduled_requests(topology, layout_name, tensors):
    for mode in MODES:
        try:
            cfg, layout, accesses = workload(topology, layout_name, mode, tensors)
        except ValueError:  # vllm_interleaved needs equal tokens per page; unit modes may not fit
            continue
        expected = {}
        for batch in schedule(cfg, layout, accesses):
            expected.setdefault((batch.step, batch.layer), []).extend(r for r in batch.requests if r.op == "R")
        expected = {key: value for key, value in expected.items() if value}
        for pages in iter_layer_pages(cfg, layout, accesses):
            requests = expected.pop((pages.step, pages.layer))
            assert pages.lpa.tolist() == [r.lpa for r in requests]
            assert pages.useful.tolist() == [r.useful_bytes for r in requests]
            assert pages.burst.tolist() == [r.burst_bytes for r in requests]
            assert pages.runs.tolist() == [r.burst_runs for r in requests]
            placed = place(cfg, pages.lpa)
            assert list(zip(placed.die.tolist(), placed.plane.tolist(), placed.offset.tolist())) == [
                locate(cfg, r.lpa) for r in requests]
            assert placed.channel.tolist() == [cfg.device_lpa(r.lpa) % cfg.channels for r in requests]
            assert placed.unit.tolist() == [cfg.unit(r.lpa) for r in requests]
        assert not expected


PROTOCOLS = (None, Protocol(256), Protocol(256, host_order="trigger"), Protocol(256, read_unit="wanted"),
             Protocol(1024, read_burst=True, read_unit="wanted"), Protocol(ucie_grade=2),
             Protocol(256, group_sense="whole"), Protocol(1))


@pytest.mark.parametrize("topology,mode,batch_pages", list(itertools.product(
    TOPOLOGIES, ("dense", "uniform_pages", "uniform_tokens", "sink_recent"), (8192, 300))))
def test_numpy_engine_rows_equal_reference(topology, mode, batch_pages):
    groups = (1, 4, 16) if topology in ("ocp", "ocp-8ch", "4x1") else (1, 4)
    for layout_name, tensors in (("layer_contiguous", None), ("vllm_interleaved", None),
                                 ("striped_staggered", KIVI_LIKE), ("striped_contiguous", TWO_BIT)):
        cfg, layout, accesses = workload(topology, layout_name, mode, tensors, batch_pages)
        protocols = PROTOCOLS + ((Protocol(ucie_grade=1),) if cfg.host_channels <= 8 else ())
        for protocol in protocols:
            expected = analyze(cfg, schedule(cfg, layout, accesses), groups, protocol=protocol)
            actual = analyze_accesses(cfg, layout, accesses, groups, protocol=protocol, timing=False)
            for want, got in zip(expected, actual):
                assert len(want) == len(got)
                for row, other in zip(want, got):
                    assert list(row) == list(other)
                    assert all(row[k] == other[k] and type(row[k]) is type(other[k]) for k in row), (row, other)


def test_removed_cli_flags_are_rejected():
    from hbf.cli import build_run_parser
    parser = build_run_parser()
    for flag in (["--engine", "python"], ["--timing"], ["--placement", "mqsim"], ["--dbsy-ns", "0"]):
        with pytest.raises(SystemExit):
            parser.parse_args(flag)
