"""Mechanism analysis: read rounds, plane utilization and the 64 B transfer view."""
import csv

import pytest

from hbf.access import Access, LAYOUTS, Shape, TensorSpec, generate
from hbf.analysis import analyze, analyze_group, locate, placement, plane_geometry
from hbf.cli import main, pipeline
from hbf.config import load, parse_topology
from hbf.controller import Batch, Request, burst_bytes, metrics, schedule


def tiny():
    # One die (channel) of four planes (chips): plane = lpa % 4, offset = lpa // 4.
    return load("16x64", channels=1, chips=4, aggregate_gb_s=64)


def test_locate_and_geometry():
    cfg = load("16x64")
    assert plane_geometry(cfg) == (16, 64)
    assert locate(cfg, 0) == (0, 0, 0)
    assert locate(cfg, 17) == (1, 1, 0)
    assert locate(cfg, 1024 + 16 * 63 + 15) == (15, 63, 1)
    real = load("16x64", chips=1, planes=64)
    assert plane_geometry(real) == (16, 64)
    assert locate(real, 16 * 5 + 3) == (3, 5, 0)


def test_hand_computed_lock_step_rounds():
    cfg = tiny()
    reads = [Request(lpa) for lpa in (0, 1, 5, 6)]
    # planes: 0 -> {0}, 1 -> {0, 1}, 2 -> {1}
    independent = analyze_group(cfg, reads, 1)
    assert (independent["rounds"], independent["min_rounds"]) == (2, 1)
    assert independent["utilization"] == 4 / (2 * 4)
    assert independent["sensed_pages"] == 4 and independent["active_planes"] == 3
    pairs = analyze_group(cfg, reads, 2)
    # group {0,1} needs offsets {0,1}, group {2,3} needs offset {1}
    assert pairs["rounds"] == 2 and pairs["sensed_pages"] == 2 * 3
    whole = analyze_group(cfg, reads, 4)
    assert whole["rounds"] == 2 and whole["sense_amplification"] == 8 / 4
    with pytest.raises(ValueError, match="must divide"):
        analyze_group(cfg, reads, 3)
    with pytest.raises(ValueError, match="deduplicated"):
        analyze_group(cfg, reads + [Request(0)], 1)


@pytest.mark.parametrize("topology", ["16x64", "64x16", "256x4"])
def test_independent_rounds_match_unit_load(topology):
    cfg = load(topology)
    shape = Shape(2, 8, 128, 8192)
    layout = LAYOUTS["layer_contiguous"](cfg, shape)
    batches = schedule(cfg, layout, generate(cfg, layout, "uniform_pages", fraction=.05, seed=3))
    layers, steps = analyze(cfg, batches)
    for row in layers:
        requests = [req for b in batches if b.layer == row["layer"] for req in b.requests]
        assert row["rounds"] == metrics(cfg, requests)["max_unit_pages"]
        assert row["min_rounds"] <= row["rounds"]
    assert steps[0]["rounds"] == sum(row["rounds"] for row in layers)


def test_dense_contiguous_is_perfectly_balanced():
    cfg = load("16x64")
    shape = Shape(2, 8, 128, 32768)
    layout = LAYOUTS["layer_contiguous"](cfg, shape)
    layers, _ = analyze(cfg, schedule(cfg, layout, generate(cfg, layout, "dense")), (1, 64))
    for row in layers:
        assert row["pages"] == 8 * 2 * 2048
        assert row["rounds"] == row["min_rounds"] == 32 and row["utilization"] == 1
        assert row["sense_amplification"] == 1


def test_vllm_layout_uses_one_plane_in_32():
    cfg = load("16x64")
    shape = Shape(32, 8, 128, 4096)
    layout = LAYOUTS["vllm_interleaved"](cfg, shape)
    layers, _ = analyze(cfg, schedule(cfg, layout, generate(cfg, layout, "dense")))
    for row in layers:
        assert row["active_planes"] == 32 and row["utilization"] == 32 / 1024


def test_32k_contiguous_puts_same_block_of_every_head_on_one_plane():
    cfg = load("16x64")
    shape = Shape(1, 8, 128, 32768)
    layout = LAYOUTS["layer_contiguous"](cfg, shape)
    accesses = [Access(0, 0, head, kind, "R", 0, 16) for head in range(8) for kind in ("K", "V")]
    (row,), _ = analyze(cfg, schedule(cfg, layout, accesses))
    assert (row["pages"], row["active_planes"], row["rounds"], row["min_rounds"]) == (16, 1, 16, 1)


def test_burst_bytes_follow_64_byte_granularity():
    assert burst_bytes([(3, 5)], 0, 256) == 512
    assert burst_bytes([(1, 2)], 0, 32) == 64
    assert burst_bytes([(1, 3)], 0, 32) == 128
    assert burst_bytes([(128, 129), (129, 130)], 128, 32) == 64
    assert burst_bytes([(129, 130), (130, 131)], 128, 32) == 128
    assert burst_bytes([(0, 128)], 0, 32) == 4096
    cfg = load("16x64")
    shape = Shape(1, 1, 128, 1024)
    layout = LAYOUTS["layer_contiguous"](cfg, shape, tensors=(TensorSpec("K", 2, 128),))
    accesses = [Access(0, 0, 0, "K", "R", 1, 2, 2), Access(0, 0, 0, "K", "R", 200, 202, 2)]
    (row,), _ = analyze(cfg, schedule(cfg, layout, accesses))
    assert row["pages"] == 2 and row["useful_bytes"] == 3 * 32 and row["burst_bytes"] == 64 + 64
    assert row["read_amplification"] == 2 * 4096 / 96


def test_analysis_only_pipeline_skips_mqsim(tmp_path):
    cfg = load("16x16")
    assert plane_geometry(cfg) == (16, 16)
    shape = Shape(2, 8, 128, 4096)
    assert pipeline(cfg, shape, "layer_contiguous", tmp_path / "run", None, mode="uniform_pages",
                    generation={"fraction": .02}, plane_groups=(1, 16)) is None
    assert not (tmp_path / "run" / "trace.txt").exists()
    rows = list(csv.DictReader((tmp_path / "run" / "analysis_steps.csv").open()))
    assert [row["group"] for row in rows] == ["1", "16"]
    assert int(rows[1]["rounds"]) >= int(rows[0]["rounds"])
    assert "机制分析" in (tmp_path / "run" / "report.md").read_text()


def test_parse_topology():
    assert parse_topology("16x16") == (16, 16)
    assert parse_topology("64x16", {"64x16": [64, 16]}) == (64, 16)
    with pytest.raises(ValueError, match="neither named"):
        parse_topology("sixteen")


def test_batch_rounds_add_up_sub_batches():
    cfg = load("16x64", batch_pages=1024)
    shape = Shape(1, 8, 128, 32768)
    layout = LAYOUTS["layer_contiguous"](cfg, shape)
    batches = schedule(cfg, layout, generate(cfg, layout, "uniform_pages", fraction=.25, seed=2))
    (row,), _ = analyze(cfg, batches)
    assert row["subbatches"] == len(batches) > 1
    per_batch = [analyze_group(cfg, b.requests, 1, placement(cfg, batches))["rounds"] for b in batches]
    assert row["batch_rounds"] == sum(per_batch) > row["rounds"]


def test_first_touch_offsets_without_writes():
    cfg = tiny()
    # Without a write, MQSim gives each plane's pages compact offsets in first-read order.
    batches = [Batch(0, 0, 0, 0, (Request(8), Request(13))), Batch(1, 0, 0, 10, (Request(0), Request(9)))]
    offsets = placement(cfg, batches)
    assert offsets == {8: 0, 13: 0, 0: 1, 9: 1}
    writes = [Batch(-1, 0, 0, 0, tuple(Request(lpa, "W") for lpa in range(16)))] + batches
    assert placement(cfg, writes) == {lpa: lpa // 4 for lpa in range(16)}


def test_prefill_rejects_user_writes_and_cli_checks_groups(tmp_path, monkeypatch):
    cfg = load("16x64")
    with pytest.raises(ValueError, match="must not contain writes"):
        pipeline(cfg, Shape(1, 1, 128, 64), "layer_contiguous", tmp_path / "w", None,
                 mode="write_then_read", prefill_layout=True)
    monkeypatch.setattr("sys.argv", ["hbf", "--analysis-only", "--plane-group", "3",
                                     "--output", str(tmp_path / "g")])
    with pytest.raises(SystemExit):
        main()
    assert not (tmp_path / "g").exists()


def test_striped_layout_phases_and_round_trip():
    cfg = load("16x64")
    shape = Shape(2, 4, 128, 1000)
    tensors = (TensorSpec("K", 16, 128), TensorSpec("K_scale", 16, 2), TensorSpec("V", 16, 128))
    layout = LAYOUTS["striped_contiguous"](cfg, shape, base=5, tensors=tensors)
    seen = set()
    for layer in range(2):
        for t in range(3):
            for head in range(4):
                first = layout.lpa(layer, head, t, 0)
                assert first % cfg.units == (t * 1024 // 3 + head * 1024 // 12) % 1024
                for block in range(layout.blocks(t)):
                    lpa = layout.lpa(layer, head, t, block)
                    assert lpa not in seen and layout.decode(lpa) == (layer, head, t, block)
                    seen.add(lpa)
    assert len(seen) == layout.pages and max(seen) == 5 + layout.span - 1
    with pytest.raises(ValueError, match="padding"):
        layout.decode(layout.lpa(0, 0, 0, layout.blocks(0) - 1) + 1)


def test_striping_removes_the_same_block_collision():
    cfg = load("16x64")
    shape = Shape(2, 8, 128, 32768)
    rows = {}
    for name in ("layer_contiguous", "striped_contiguous"):
        layout = LAYOUTS[name](cfg, shape)
        batches = schedule(cfg, layout, generate(cfg, layout, "sink_recent", count=4, fraction=1 / 64))
        rows[name], _ = analyze(cfg, batches)
    for contiguous, striped in zip(rows["layer_contiguous"], rows["striped_contiguous"]):
        # 16 regions x (1 sink page + 32 recent pages); contiguous stacks the same blocks on 33 planes.
        assert contiguous["pages"] == striped["pages"] == 16 * 33
        assert (contiguous["rounds"], contiguous["active_planes"]) == (16, 33)
        assert striped["rounds"] == striped["min_rounds"] == 1
    layout = LAYOUTS["striped_contiguous"](cfg, Shape(1, 8, 128, 32768))
    (row,), _ = analyze(cfg, schedule(cfg, layout, generate(cfg, layout, "dense")))
    assert row["utilization"] == 1


def test_striped_layout_runs_in_mqsim(binary, tmp_path):
    cfg = load("16x1", planes=64, blocks_per_plane=64, plane_rule="independent", coalesce_ns=1000)
    _, layers, _, stats = pipeline(cfg, Shape(2, 8, 128, 4096), "striped_contiguous", tmp_path / "s", binary,
                                   mode="sink_recent", generation={"count": 4, "fraction": .05},
                                   prefill_layout=True)
    reads = [row for row in layers if row["step"] == 0]
    assert len(reads) == 2 and all(row["read_pages"] == 16 * (1 + 13) for row in reads)
