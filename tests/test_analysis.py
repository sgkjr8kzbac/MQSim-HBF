"""Mechanism analysis: read rounds, plane utilization and the 64 B transfer view."""
import csv
from collections import Counter

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
    assert parse_topology("16x16") == {"channels": 16, "chips": 16}
    assert parse_topology("64x16", {"64x16": [64, 16]}) == {"channels": 64, "chips": 16}
    assert load("ocp").geometry == (32, 2, 1, 16) and load("ocp").address_map == "ocp"
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


def test_ocp_address_map_follows_figure_4():
    """4 KiB units: host channel = g mod 16; inside it bank = L mod 16, die = L / 16 mod 4 (section 4.6).

    Figure 4: dies 0 and 2 sit on core-die channel 0, dies 1 and 3 on core-die channel 1; unit 64 of a
    host channel returns to die 0, bank 0, one page further.
    """
    cfg = load("ocp")
    assert cfg.units == 1024 and cfg.host_channels == 16 and plane_geometry(cfg) == (64, 16)
    host = 3
    expect = {0: (0, 0, 0), 15: (0, 0, 15), 16: (1, 0, 0), 31: (1, 0, 15), 32: (0, 1, 0), 48: (1, 1, 0),
              63: (1, 1, 15), 64: (0, 0, 0)}
    for local, (core, chip, bank) in expect.items():
        lpa = host + 16 * local
        assert cfg.physical(lpa) == (2 * host + core, chip, 0, bank)
        assert locate(cfg, lpa)[2] == local // 64
    span = range(3 * cfg.units)
    assert sorted(cfg.device_lpa(lpa) for lpa in span) == list(span)
    assert all(cfg.host_lpa(cfg.device_lpa(lpa)) == lpa for lpa in span)
    assert all(cfg.unit(cfg.unit_lpa(unit, 2)) == unit for unit in range(cfg.units))
    layout = LAYOUTS["layer_contiguous"](cfg, Shape(1, 8, 128, 8192))
    rows, _ = analyze(cfg, schedule(cfg, layout, generate(cfg, layout, "dense")), (1, 16))
    assert [(row["group"], row["utilization"], row["rounds"]) for row in rows] == [(1, 1, 8), (16, 1, 8)]
    with pytest.raises(ValueError, match="must divide"):
        analyze_group(cfg, [Request(0)], 32)
    with pytest.raises(ValueError, match="Figure 4"):
        load("ocp", chips=4)


def test_ocp_topology_runs_in_mqsim(binary, tmp_path):
    cfg = load("ocp", blocks_per_plane=64, coalesce_ns=1000)
    path = tmp_path / "ocp"
    _, layers, _, stats = pipeline(cfg, Shape(1, 8, 128, 4096), "layer_contiguous", path, binary,
                                   mode="sink_recent", generation={"count": 4, "fraction": .05},
                                   prefill_layout=True, physical=True, plane_groups=(1, 16))
    for row in csv.reader((path / "physical.csv").open()):
        if row[1] == "issue":
            channel, chip, die, plane, block, page, device = map(int, row[3:10])
            lpa = cfg.host_lpa(device)
            assert cfg.physical(lpa) == (channel, chip, die, plane)
            assert block * cfg.pages_per_block + page == lpa // cfg.units  # the spec's static offset
    analysis = {row["group"]: row for row in csv.DictReader((path / "analysis_layers.csv").open())}
    commands = stats["Issued_Flash_Read_CMD"] + stats["Issued_Flash_Multiplane_Read_CMD"]
    assert commands >= int(analysis["1"]["rounds"]) and layers[-1]["read_pages"] == 8 * 2 * (1 + 13)


def test_protocol_commands_per_page():
    from hbf.analysis import Protocol
    cfg = load("ocp")
    partial = Request(0, useful_bytes=96, burst_bytes=192, burst_runs=2)
    assert Protocol(256).commands(cfg, partial) == 64  # whole page, 64 B commands (section 5.3.1)
    assert Protocol(256, read_burst=True).commands(cfg, partial) == 1
    assert Protocol(256, read_unit="wanted").commands(cfg, partial) == 3
    assert Protocol(256, read_burst=True, read_unit="wanted").commands(cfg, partial) == 2
    for bad in (0, 128, 300, 32768):
        with pytest.raises(ValueError, match="MOCS"):
            Protocol(bad)
    shape = Shape(1, 1, 128, 1024)
    layout = LAYOUTS["layer_contiguous"](cfg, shape, tensors=(TensorSpec("K", 2, 128),))
    (request,) = schedule(cfg, layout, [Access(0, 0, 0, "K", "R", 1, 3, 2), Access(0, 0, 0, "K", "R", 5, 6, 2)])[0].requests
    # 32 B tokens: tokens 1-2 are bytes 32-96 (units 0-1), token 5 is bytes 160-192 (unit 2): one run.
    assert (request.burst_bytes, request.burst_runs) == (192, 1)


def test_credit_rounds_match_hbf_eventlab_probe():
    """HBF-EventLab, ocp geometry, 4 KiB interleave, 64 B reads: 128 pages (lpa 0-127, 8 per host
    channel on 8 banks) took two sensing rounds at MOCS 256 (8.4 us) and one at MOCS 1024 (4.5 us)."""
    from hbf.analysis import Protocol, protocol_rounds
    cfg = load("ocp")
    reads = [Request(lpa) for lpa in range(128)]
    assert analyze_group(cfg, reads)["rounds"] == 1
    assert protocol_rounds(cfg, reads, 1, None, Protocol(256))[:2] == (2, 2)
    assert protocol_rounds(cfg, reads, 1, None, Protocol(1024))[:2] == (1, 1)
    assert protocol_rounds(cfg, reads, 1, None, Protocol(256, read_burst=True))[:2] == (1, 1)


@pytest.mark.parametrize("seed", range(5))
def test_greedy_credit_schedule_reaches_the_bound(seed):
    """Uniform commands per page: rounds = max(longest bank queue, ceil(pages / floor(MOCS / c)))."""
    import random
    from hbf.analysis import Protocol, protocol_rounds
    rng = random.Random(seed)
    cfg = load("ocp")
    reads = [Request(lpa) for lpa in rng.sample(range(16 * 1024), 3000)]
    for protocol in (Protocol(256), Protocol(512), Protocol(16384), Protocol(256, read_burst=True)):
        per_page = protocol.commands(cfg, reads[0])
        bound = 0
        for host in range(cfg.host_channels):
            mine = [req for req in reads if cfg.host_channel(req.lpa) == host]
            banks = Counter(locate(cfg, req.lpa)[:2] for req in mine)
            bound = max(bound, max(banks.values()), -(-len(mine) // (protocol.mocs // per_page)))
        rounds, credit, commands = protocol_rounds(cfg, reads, 1, None, protocol)
        assert rounds == bound and commands == 3000 * per_page and credit <= rounds
    lockstep, _, _ = protocol_rounds(cfg, reads, 16, None, Protocol(16384))
    assert lockstep == analyze_group(cfg, reads, 16)["rounds"]  # ample credits: only the groups limit


def test_protocol_columns_in_pipeline(tmp_path):
    from hbf.analysis import Protocol
    cfg = load("ocp")
    pipeline(cfg, Shape(2, 8, 128, 8192), "layer_contiguous", tmp_path / "p", None, mode="uniform_pages",
             generation={"fraction": .05}, plane_groups=(1, 16), protocol=Protocol(256))
    rows = list(csv.DictReader((tmp_path / "p" / "analysis_steps.csv").open()))
    for row in rows:
        assert int(row["protocol_rounds"]) >= max(int(row["rounds"]), int(row["credit_rounds"]))
        assert int(row["commands"]) == 64 * int(row["pages"])
    assert "协议约束" in (tmp_path / "p" / "report.md").read_text()
