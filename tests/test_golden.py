"""Byte-identical analysis outputs: new engines and opt-in features must not change default results.

The golden files under tests/golden/ were written by the pure-Python analysis before the numpy engine,
the timing estimate and the compression formats existed. Regenerate them only on purpose:
``HBF_REGEN_GOLDEN=1 pytest tests/test_golden.py``.
"""
import os
from pathlib import Path

import pytest

from hbf.access import Shape, TensorSpec
from hbf.analysis import Protocol
from hbf.cli import pipeline
from hbf.config import load

GOLDEN = Path(__file__).parent / "golden"
FILES = ("analysis_layers.csv", "analysis_steps.csv", "report.md")
TWO_BIT = (TensorSpec("K", 2, 128), TensorSpec("V", 2, 128))
KIVI_LIKE = (TensorSpec("K", 2, 128), TensorSpec("K_scale", 16, 8), TensorSpec("V", 2, 128), TensorSpec("V_scale", 16, 8))

CASES = {
    "ocp-striped-dense": dict(topology="ocp", layout="striped_contiguous", mode="dense", shape=Shape(2, 8, 128, 4096),
                              groups=(1, 16)),
    "ocp-tokens-fcfs": dict(topology="ocp", layout="layer_contiguous", mode="uniform_tokens", shape=Shape(2, 8, 128, 8192),
                            generation={"fraction": .05, "seed": 3, "steps": 2},
                            protocol=Protocol(256, ucie_grade=2)),
    "16x64-vllm-sink": dict(topology="16x64", layout="vllm_interleaved", mode="sink_recent", shape=Shape(2, 4, 128, 4096),
                            generation={"fraction": .05, "count": 4}, batch_pages=100),
    "ocp-staggered-2bit-trigger": dict(topology="ocp", layout="striped_staggered", mode="uniform_pages",
                                       shape=Shape(2, 8, 128, 16384), tensors=TWO_BIT,
                                       generation={"fraction": .1, "seed": 5},
                                       protocol=Protocol(256, read_unit="wanted", host_order="trigger")),
    "16x64-kivi-dense": dict(topology="16x64", layout="layer_contiguous", mode="dense", shape=Shape(2, 4, 128, 2048),
                             tensors=KIVI_LIKE, groups=(1, 4)),
    "ocp-tokens-whole": dict(topology="ocp", layout="layer_contiguous", mode="uniform_tokens", shape=Shape(1, 8, 128, 8192),
                             generation={"fraction": .1, "seed": 7, "select_tokens": 32}, groups=(1, 16),
                             protocol=Protocol(1024, read_burst=True, read_unit="wanted", group_sense="whole")),
}


def run_case(case, directory):
    cfg = load(case["topology"], **({"batch_pages": case["batch_pages"]} if "batch_pages" in case else {}))
    pipeline(cfg, case["shape"], case["layout"], directory, None, mode=case["mode"],
             generation=case.get("generation"), tensors=case.get("tensors"),
             plane_groups=case.get("groups", (1,)), protocol=case.get("protocol"))


@pytest.mark.parametrize("name", sorted(CASES))
def test_analysis_outputs_are_unchanged(name, tmp_path):
    directory = tmp_path / name
    run_case(CASES[name], directory)
    expected = GOLDEN / name
    if os.environ.get("HBF_REGEN_GOLDEN"):
        expected.mkdir(parents=True, exist_ok=True)
        for file in FILES:
            (expected / file).write_bytes((directory / file).read_bytes())
    for file in FILES:
        assert (directory / file).read_bytes() == (expected / file).read_bytes(), f"{name}/{file} changed"
