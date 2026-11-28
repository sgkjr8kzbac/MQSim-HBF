"""Read-time estimates (--timing): hand-computed values and MQSim-measured layers."""
import csv
from pathlib import Path

import pytest

from hbf.access import LAYOUTS, Shape, generate
from hbf.analysis import Protocol, analyze_accesses
from hbf.cli import main, pipeline
from hbf.config import load

HBF_COMMANDS = {"read_plane_ns": 0, "coalesce_ns": 1, "cache_read": True}
EVIDENCE = Path(__file__).parents[1] / "evidence" / "time-calibration.csv"


def rows(cfg, shape, mode="dense", groups=(1,), protocol=None, layout="layer_contiguous", **generation):
    layout = LAYOUTS[layout](cfg, shape)
    return analyze_accesses(cfg, layout, list(generate(cfg, layout, mode, **generation)), groups,
                            protocol=protocol, timing=True)


def test_one_plane_queue_matches_mqsim():
    # 13 pages on plane 0 of a 16-plane die: 13 rounds of cmd 290 + tR 4096 + R/B 20, one transfer
    # (4 channels at 1,024 GB/s: 16 ns), the 1 ns window and 1 ns host time. MQSim (round 8, e19b): 57,296.
    cfg = load("4x1", planes=16, plane_rule="independent", **HBF_COMMANDS)
    (row,), (step,) = rows(cfg, Shape(1, 1, 128, 8192), "same_unit", count=13, unit=0)
    assert row["rounds"] == 13
    assert row["est_ns"] == row["est_batches_ns"] == 1 + 13 * (290 + 4096 + 20) + 16 + 1 == 57_296
    assert row["bound_ns"] == 13 * 4096 and row["bottleneck"] == "bank"
    assert step["est_ns"] == 57_296 and step["max_block_reads"] == 13


@pytest.mark.parametrize("cache,expected,mqsim", [(True, 37_593, 37_624), (False, 47_001, 47_032)])
def test_dense_layer_at_grade_three(cache, expected, mqsim):
    cfg = load("ocp", aggregate_gb_s=3072, read_plane_ns=0, coalesce_ns=1000, cache_read=cache)
    (row,), _ = rows(cfg, Shape(1, 8, 128, 8192))
    assert row["est_ns"] == expected
    assert abs(expected / mqsim - 1) < .001  # MQSim, round 8 (e2_dense_3072_on/off)


def test_channel_bound_and_breakdown_add_up():
    cfg = load("ocp", aggregate_gb_s=384, **HBF_COMMANDS)
    (row,), _ = rows(cfg, Shape(1, 8, 128, 8192))
    assert cfg.transfer_ns == 340 and row["rounds"] == 8
    assert row["bound_ns"] == 256 * 340 == 87_040 and row["bottleneck"] == "channel"
    parts = ("t_balanced_ns", "t_imbalance_ns", "t_protocol_ns", "t_channel_ns", "t_overhead_ns")
    assert sum(row[key] for key in parts) == row["est_ns"]


def test_breakdown_with_credit_limits():
    cfg = load("ocp", **HBF_COMMANDS)
    cfg_8ch = load("ocp-8ch", **HBF_COMMANDS)
    shape = Shape(2, 8, 128, 8192)
    for target, protocol in ((cfg, Protocol(256)), (cfg_8ch, Protocol(256, ucie_grade=1)),
                             (cfg, Protocol(256, ucie_grade=2)), (cfg, Protocol(256, host_order="trigger"))):
        layers, steps = rows(target, shape, "uniform_tokens", protocol=protocol, fraction=.05, seed=4)
        for row in layers:
            parts = ("t_balanced_ns", "t_imbalance_ns", "t_protocol_ns", "t_channel_ns", "t_overhead_ns")
            assert sum(row[key] for key in parts) == row["est_ns"] >= row["bound_ns"]
            assert row["t_protocol_ns"] == (row["protocol_rounds"] - row["rounds"]) * target.read_ns or row["subbatches"] > 1
        assert steps[0]["est_ns"] == sum(row["est_ns"] for row in layers)
    (row, _), _ = rows(cfg, shape, "dense", protocol=Protocol(256))
    assert row["bottleneck"] == "credit" and row["protocol_rounds"] > row["rounds"]


def test_read_time_parameter_changes_channel_rounds(tmp_path):
    main(["--topology", "ocp", "--layers", "1", "--context", "8192", "--analysis-only", "--timing",
          "--read-ns", "8192", "--output", str(tmp_path / "r")])
    (row,) = csv.DictReader((tmp_path / "r" / "analysis_layers.csv").open())
    assert row["channel_rounds"] == "4"  # ceil(256 pages x 128 ns / 8,192 ns)
    assert row["bound_ns"] == str(8 * 8192)
    assert "未经验证" in (tmp_path / "r" / "report.md").read_text()


def test_chip_as_plane_has_no_estimate():
    (row,), (step,) = rows(load("16x64"), Shape(1, 8, 128, 8192))
    assert row["est_ns"] is None and step["est_ns"] is None and row["bound_ns"] == 8 * 4096


def test_lockstep_group_estimate_counts_offsets():
    cfg = load("4x1", planes=16, plane_rule="lockstep", **HBF_COMMANDS)
    (one, wide), _ = rows(cfg, Shape(1, 1, 128, 8192), "distinct_units", groups=(1, 16), count=4)
    # CWDP puts units 0-3 on the four channels: one page and one round per channel either way.
    assert one["rounds"] == wide["rounds"] == 1 and one["est_ns"] == wide["est_ns"] == 1 + 4406 + 16 + 1
    layout = LAYOUTS["layer_contiguous"](cfg, Shape(1, 1, 128, 8192))
    lpas = [cfg.inverse((0, 0, 0, plane), offset) for plane, offset in ((0, 0), (1, 0), (2, 1), (3, 1))]
    accesses = [layout.page_access(lpa) for lpa in lpas]
    (one, wide), _ = analyze_accesses(cfg, layout, accesses, (1, 16), timing=True)
    # Independent: one round of four planes. Lock-step: offsets 0 and 1 are two rounds of two pages.
    assert (one["rounds"], wide["rounds"]) == (1, 2)
    assert one["est_ns"] == 1 + 4406 + 4 * 16 + 1
    assert wide["est_ns"] == 1 + 4406 + max(4406, 2 * 16) + 2 * 16 + 1


def test_block_reads_accumulate_over_the_layers_of_a_step():
    # Dense 8K BF16 on ocp: 8 layers x 2 x 8 x 512 pages = 8 x 8,192 pages; 8,192 pages fill offsets 0-7
    # of every bank and one block holds 256 offsets, so a block sees 8 reads per layer.
    cfg = load("ocp")
    _, (step,) = rows(cfg, Shape(8, 8, 128, 8192))
    assert step["max_block_reads"] == 64


def test_timing_needs_numpy_engine(tmp_path):
    with pytest.raises(ValueError, match="numpy engine"):
        pipeline(load("ocp"), Shape(1, 8, 128, 1024), "layer_contiguous", tmp_path / "x", None, engine="python",
                 timing=True)
    with pytest.raises(SystemExit):
        main(["--topology", "ocp", "--analysis-only", "--timing", "--engine", "python"])


def test_calibration_evidence_within_five_percent():
    measured = list(csv.DictReader(EVIDENCE.open()))
    validated = [row for row in measured if row["hbf_command_model"] == "True"]
    assert len(validated) >= 40
    assert max(abs(float(row["error"])) for row in validated) < .05
    assert all(int(row["est_batches_ns"]) <= int(row["mqsim_ns"]) * 1.01 for row in validated)
