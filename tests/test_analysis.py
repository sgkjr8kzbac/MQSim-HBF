"""Mechanism analysis: read rounds, plane utilization and the 64 B transfer view."""
import csv
from collections import Counter

import pytest

from hbf.frontend.kv.layouts import LAYOUTS
from hbf.frontend.kv.tensors import Shape, TensorSpec
from hbf.frontend.trace.synthetic import generate
from hbf.frontend.trace.tokens import Access
from hbf.backends.rounds.analysis import locate, placement, plane_geometry
from tests.reference import analyze, analyze_group
from hbf.cli import main, pipeline
from hbf.device.config import load, parse_topology
from hbf.backends.mqsim.schedule import metrics
from hbf.frontend.pages import Batch, Request, burst_bytes, schedule


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
    monkeypatch.setattr("sys.argv", ["hbf", "run", "--analysis-only", "--plane-group", "3",
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
    end = None
    for layer in range(2):
        for t in range(3):
            for head in range(4):
                first = layout.lpa(layer, head, t, 0)
                if (t, head) == (0, 0):
                    assert first % cfg.units == 0  # each layer starts on phase 0
                else:
                    assert first == end + 1024 // 12  # a fixed gap of P / (T * H) pages
                end = layout.lpa(layer, head, t, layout.blocks(t) - 1) + 1
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
    # SPLASH's phases when every region is a multiple of P long: heads P/16 apart, V half a device on.
    assert [layout.lpa(0, h, t, 0) % 1024 for t in range(2) for h in range(8)] == list(range(0, 1024, 64))


@pytest.mark.parametrize("bits, shape", [((16, 16), Shape(2, 8, 128, 32768)), ((2, 2), Shape(2, 8, 128, 32768)),
                                         ((2, 16), Shape(2, 8, 128, 8192)), ((16, 16), Shape(2, 32, 128, 4096)),
                                         ((2, 16), Shape(2, 32, 128, 32768))])
def test_striped_padding_is_bounded_and_dense_reads_stay_balanced(bits, shape):
    """Phases that ignored region length padded short regions up to fixed phases (4.2x the pages for
    2-bit 32K, 16x for 2-bit with 32 heads) and left KIVI's dense reads twice the minimum. With gaps
    a layer pads less than 2P pages and a dense layer needs at most two rounds above the minimum."""
    cfg = load("ocp")
    tensors = (TensorSpec("K", bits[0], 128), TensorSpec("V", bits[1], 128))
    for name in ("striped_contiguous", "striped_staggered"):
        layout = LAYOUTS[name](cfg, shape, tensors=tensors)
        assert layout.span < layout.pages + 2 * shape.layers * 1024
        (row, *_), _ = analyze(cfg, schedule(cfg, layout, generate(cfg, layout, "dense")))
        assert row["rounds"] <= row["min_rounds"] + 2


def test_staggered_stripes_spread_host_channels():
    cfg = load("ocp")
    shape = Shape(1, 8, 128, 32768)
    hosts = {}
    for name in ("striped_contiguous", "striped_staggered"):
        layout = LAYOUTS[name](cfg, shape)
        hosts[name] = {cfg.host_channel(layout.lpa(0, h, t, 0)) for t in range(2) for h in range(8)}
    assert hosts["striped_contiguous"] == {0} and len(hosts["striped_staggered"]) == 16


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
    from hbf.backends.rounds.analysis import Protocol
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
    from hbf.backends.rounds.analysis import Protocol, protocol_rounds
    cfg = load("ocp")
    reads = [Request(lpa) for lpa in range(128)]
    assert analyze_group(cfg, reads)["rounds"] == 1
    assert protocol_rounds(cfg, reads, 1, None, Protocol(256))[:2] == (2, 2)
    assert protocol_rounds(cfg, reads, 1, None, Protocol(1024))[:2] == (1, 1)
    assert protocol_rounds(cfg, reads, 1, None, Protocol(256, read_burst=True))[:2] == (1, 1)
    # Trigger-first order (one 64 B read starts the sense): HBF-EventLab 4.31 us, one round.
    assert protocol_rounds(cfg, reads, 1, None, Protocol(256, host_order="trigger"))[0] == 1


def test_credit_rounds_against_hbf_eventlab_four_pages_per_bank():
    """32 pages on 8 banks of host channel 0 (4 per bank), MOCS 256, sense 4.096 us.

    HBF-EventLab sensed in 7 waves of up to 5 pages with the natural order (it frees each credit
    when its response is delivered), 4 waves (the bank bound) with trigger-first order or MOCS 1024.
    The round model holds credits for a whole round, so under fcfs it is one round pessimistic.
    """
    from hbf.backends.rounds.analysis import Protocol, protocol_rounds
    cfg = load("ocp")
    reads = [Request(16 * (bank + 64 * k)) for bank in range(8) for k in range(4)]
    assert {cfg.host_channel(req.lpa) for req in reads} == {0} and analyze_group(cfg, reads)["rounds"] == 4
    assert protocol_rounds(cfg, reads, 1, None, Protocol(256))[0] == 8
    assert protocol_rounds(cfg, reads, 1, None, Protocol(256, host_order="trigger"))[0] == 4
    assert protocol_rounds(cfg, reads, 1, None, Protocol(1024))[0] == 4


@pytest.mark.parametrize("seed", range(5))
def test_greedy_credit_schedule_reaches_the_bound(seed):
    """Uniform commands per page: rounds = max(longest bank queue, ceil(pages / ceil(MOCS / c)))."""
    import random
    from hbf.backends.rounds.analysis import Protocol, protocol_rounds
    rng = random.Random(seed)
    cfg = load("ocp")
    reads = [Request(lpa) for lpa in rng.sample(range(16 * 1024), 3000)]
    forty = [Request(req.lpa, useful_bytes=2560, burst_bytes=2560, burst_runs=1) for req in reads]
    for protocol, pages in ((Protocol(256), reads), (Protocol(512), reads), (Protocol(16384), reads),
                            (Protocol(256, read_burst=True), reads), (Protocol(1), reads),
                            (Protocol(256, read_unit="wanted"), forty)):  # 40 commands: MOCS / c is not whole
        per_page = protocol.commands(cfg, pages[0])
        bound = 0
        for host in range(cfg.host_channels):
            mine = [req for req in pages if cfg.host_channel(req.lpa) == host]
            banks = Counter(locate(cfg, req.lpa)[:2] for req in mine)
            bound = max(bound, max(banks.values()), -(-len(mine) // -(-protocol.mocs // per_page)))
        rounds, credit, commands, _ = protocol_rounds(cfg, pages, 1, None, protocol)
        assert rounds == bound and commands == 3000 * per_page and credit <= rounds
    lockstep, _, _, _ = protocol_rounds(cfg, reads, 16, None, Protocol(16384))
    assert lockstep == analyze_group(cfg, reads, 16)["rounds"]  # ample credits: only the groups limit


def test_protocol_columns_in_pipeline(tmp_path):
    from hbf.backends.rounds.analysis import Protocol
    cfg = load("ocp")
    pipeline(cfg, Shape(2, 8, 128, 8192), "layer_contiguous", tmp_path / "p", None, mode="uniform_pages",
             generation={"fraction": .05}, plane_groups=(1, 16), protocol=Protocol(256))
    rows = list(csv.DictReader((tmp_path / "p" / "analysis_steps.csv").open()))
    for row in rows:
        assert int(row["protocol_rounds"]) >= max(int(row["rounds"]), int(row["credit_rounds"]),
                                                  int(row["link_rounds"]))
        assert int(row["commands"]) == 64 * int(row["pages"])
    assert "协议约束" in (tmp_path / "p" / "report.md").read_text()


def test_items_larger_than_mocs_start_their_sense_once():
    """A page whose commands do not all fit is sensed once; the rest are cache hits.

    HBF-EventLab, ocp geometry, MOCS 1, burst4096 prefill, 64 B reads: eight pages on eight banks
    of host channel 0 finished in 33.17 us with 8 senses and 504 cache hits (8 x 4.096 us plus
    hits), against 8.31 us (two rounds) at MOCS 256. With MOCS 256, seven pages of 2,560 B (40
    commands each, 280 in all) took one wave of 4.22 us, thirteen took 8.31 us.
    """
    from hbf.backends.rounds.analysis import Protocol, protocol_rounds
    cfg = load("ocp")
    eight = [Request(16 * bank) for bank in range(8)]
    assert protocol_rounds(cfg, eight, 1, None, Protocol(1))[:2] == (8, 8)
    assert protocol_rounds(cfg, eight, 1, None, Protocol(256))[:2] == (2, 2)
    mocs1 = protocol_rounds(cfg, [Request(lpa) for lpa in range(128)], 1, None, Protocol(1))
    assert mocs1[:2] == (8, 8)  # 8 pages per host channel, one sense at a time
    wanted = Protocol(256, read_unit="wanted")
    for banks, expected in ((7, 1), (13, 2)):
        pages = [Request(16 * bank, useful_bytes=2560, burst_bytes=2560) for bank in range(banks)]
        assert protocol_rounds(cfg, pages, 1, None, wanted)[0] == expected


def test_lock_step_group_sense_is_an_explicit_assumption():
    """16 pages at one lock-step offset need 1,024 commands; MOCS 256 admits four pages per round.

    ``arrived`` senses only those four, so the offset is sensed four times; ``whole`` senses the
    whole group at the first command, and then a trigger host needs one command per offset.
    """
    from hbf.backends.rounds.analysis import Protocol, protocol_rounds
    cfg = load("ocp")
    die0 = [Request(16 * bank) for bank in range(16)]  # host channel 0, die 0, banks 0-15, offset 0
    rounds = {(sense, order, mocs): protocol_rounds(cfg, die0, 16, None, Protocol(mocs, host_order=order,
                                                                                  group_sense=sense))[:3]
              for sense in ("arrived", "whole") for order in ("fcfs", "trigger") for mocs in (1, 256)}
    assert rounds["arrived", "fcfs", 256] == (4, 4, 1024)
    assert rounds["arrived", "trigger", 256][:2] == (1, 1)
    assert rounds["arrived", "fcfs", 1][:2] == rounds["arrived", "trigger", 1][:2] == (16, 16)
    assert {rounds["whole", order, mocs][:2] for order in ("fcfs", "trigger") for mocs in (1, 256)} == {(1, 1)}
    assert Protocol().group_sense == "arrived"
    with pytest.raises(ValueError, match="group_sense"):
        Protocol(group_sense="all")
    reads = [Request(lpa) for lpa in range(0, 4096, 3)]
    for protocol in (Protocol(1), Protocol(256)):  # independent planes: the option has no effect
        assert protocol_rounds(cfg, reads, 1, None, protocol) == protocol_rounds(
            cfg, reads, 1, None, Protocol(protocol.mocs, group_sense="whole"))


@pytest.mark.parametrize("seed", range(5))
def test_credit_bound_never_exceeds_rounds(seed):
    import random
    from hbf.backends.rounds.analysis import Protocol, protocol_rounds
    rng = random.Random(seed)
    cfg = load("ocp")
    reads = []
    for lpa in rng.sample(range(16 * 1024), 2000):
        wanted = rng.randint(1, 64)
        reads.append(Request(lpa, useful_bytes=64 * wanted, burst_bytes=64 * wanted, burst_runs=rng.randint(1, wanted)))
    for group in (1, 4, 16):
        for protocol in (Protocol(1), Protocol(256), Protocol(256, read_unit="wanted"),
                         Protocol(256, read_burst=True, read_unit="wanted"), Protocol(1, host_order="trigger"),
                         Protocol(256, group_sense="whole"), Protocol(1, host_order="trigger", group_sense="whole")):
            rounds, credit, _, _ = protocol_rounds(cfg, reads, group, None, protocol)
            assert analyze_group(cfg, reads, group)["rounds"] <= rounds and credit <= rounds


def test_cwdp_topologies_keep_sixteen_host_channels():
    from hbf.backends.rounds.analysis import Protocol, protocol_rounds
    for topology in ("16x64", "64x16", "256x4"):
        cfg = load(topology)
        assert cfg.host_channels == 16 and {cfg.host_channel(lpa) for lpa in range(64)} == set(range(16))
    reads = [Request(lpa) for lpa in range(1024)]
    for topology in ("16x64", "64x16", "256x4"):
        assert protocol_rounds(load(topology), reads, 1, None, Protocol(256))[:2] == (16, 16)
    assert load("16x1", planes=64).host_channels == 16 and load("16x64", channels=1, chips=4).host_channels == 1


def test_link_bound_follows_ucie_grade():
    """A dense round of 64 pages per host channel moves 256 KiB: 5.46 us at grade 1 (48 GB/s)."""
    from hbf.backends.rounds.analysis import Protocol, analyze_accesses, _protocol_part, protocol_rounds
    from hbf.cli import build_run_parser, configure, topology_name
    from hbf.device.address import place
    from hbf.frontend.pages import iter_layer_pages
    cfg8 = load("ocp-8ch")
    assert (cfg8.channels, cfg8.chips, cfg8.planes, cfg8.units, cfg8.host_channels, cfg8.aggregate_gb_s,
            cfg8.width) == (16, 2, 16, 512, 8, 512, 32)
    assert topology_name(load("ocp")) == "ocp" and topology_name(cfg8) == "ocp-8ch"
    reads8 = [Request(lpa) for lpa in range(512 * 2)]  # every bank twice, 128 pages per host channel
    link = {grade: protocol_rounds(cfg8, reads8, 1, None, Protocol(16384, ucie_grade=grade))[3] for grade in (1, 2, 3)}
    # per host channel: 128 pages = 512 KiB; one tR carries 48/94/188 GB/s x 4,096 ns
    assert link == {1: 3, 2: 2, 3: 1}
    # OCP section 5.1.2.1: speed grade 1 allows at most 8 host channels; 16-channel ocp must reject it.
    cfg16 = load("ocp")
    reads16 = [Request(lpa) for lpa in range(1024 * 2)]
    assert {g: protocol_rounds(cfg16, reads16, 1, None, Protocol(16384, ucie_grade=g))[3] for g in (2, 3)} == {2: 2, 3: 1}
    grade1 = Protocol(16384, ucie_grade=1)
    with pytest.raises(ValueError, match="at most 8 host channels"):
        protocol_rounds(cfg16, reads16, 1, None, grade1)
    layout = LAYOUTS["layer_contiguous"](cfg16, Shape(1, 2, 128, 128))
    accesses = list(generate(cfg16, layout, "dense"))
    batches = schedule(cfg16, layout, accesses)
    with pytest.raises(ValueError, match="at most 8 host channels"):
        analyze(cfg16, batches, protocol=grade1)
    with pytest.raises(ValueError, match="at most 8 host channels"):
        analyze_accesses(cfg16, layout, accesses, protocol=grade1)
    layer = next(iter_layer_pages(cfg16, layout, accesses))
    with pytest.raises(ValueError, match="at most 8 host channels"):
        _protocol_part(cfg16, layer, place(cfg16, layer.lpa), 1, grade1, slice(None))
    with pytest.raises(ValueError, match="at most 8 host channels"):
        configure(build_run_parser().parse_args(["--topology", "ocp", "--ucie-grade", "1"]))
    configure(build_run_parser().parse_args(["--topology", "ocp-8ch", "--ucie-grade", "1"]))
    with pytest.raises(ValueError, match="ucie_grade"):
        Protocol(256, ucie_grade=4)


def test_uniform_tokens_reads_the_same_tokens_of_every_tensor():
    """Quest-style selection: K and V of a head cover the same token blocks, whatever their format."""
    cfg = load("ocp")
    shape = Shape(2, 2, 128, 4096)
    kivi = (TensorSpec("K", 2, 128), TensorSpec("V", 16, 128))
    layout = LAYOUTS["layer_contiguous"](cfg, shape, tensors=kivi)
    accesses = list(generate(cfg, layout, "uniform_tokens", fraction=.05, seed=3, select_tokens=16))
    chosen = {}
    for access in accesses:
        chosen.setdefault((access.layer, access.kv_head, access.kind), set()).add((access.token_start, access.token_end))
    for layer in range(2):
        for head in range(2):
            assert chosen[layer, head, "K"] == chosen[layer, head, "V"]
            assert len(chosen[layer, head, "K"]) == 13  # ceil(256 blocks x 5%)
    batches = schedule(cfg, layout, accesses)
    reads = [req for batch in batches for req in batch.requests]
    k_pages = {req.lpa for req in reads if layout.decode(req.lpa)[2] == 0}
    tokens = 13 * 16 * 2 * 2  # blocks x tokens x heads x layers, for each tensor
    assert sum(req.useful_bytes for req in reads) == tokens * 32 + tokens * 256
    assert len(reads) - len(k_pages) == 13 * 2 * 2  # one 16-token V page per block
    assert len(k_pages) < 13 * 2 * 2  # 128-token K pages are shared by blocks
    # A last partial block, blocks not aligned with pages, and the vLLM layout.
    odd = Shape(1, 2, 128, 1000)
    for name, block in (("layer_contiguous", 24), ("vllm_interleaved", 24), ("layer_contiguous", 7)):
        layout = LAYOUTS[name](cfg, odd)
        accesses = list(generate(cfg, layout, "uniform_tokens", fraction=1, select_tokens=block))
        assert {(a.token_start, a.token_end) for a in accesses} == {
            (lo, min(lo + block, 1000)) for lo in range(0, 1000, block)}
        (batch,) = schedule(cfg, layout, accesses)
        assert len(batch.requests) == 2 * 2 * 63  # every page of both tensors and heads, once
        assert sum(req.useful_bytes for req in batch.requests) == 2 * 2 * 1000 * 256
    with pytest.raises(ValueError, match="generator"):
        list(generate(cfg, layout, "uniform_tokens", select_tokens=0))


def test_unit_modes_reject_pages_beyond_the_layout():
    cfg = load("ocp")
    layout = LAYOUTS["layer_contiguous"](cfg, Shape(1, 8, 128, 8192))
    with pytest.raises(ValueError, match="beyond the layout"):
        list(generate(cfg, layout, "same_unit", count=9))


def test_per_page_estimate_is_withheld_for_multi_plane_dies(tmp_path):
    assert metrics(load("ocp"), [Request(0)])["mq_est_ns"] is None
    assert metrics(load("64x16"), [Request(0)])["mq_est_ns"] > load("64x16").read_ns  # chip as plane
    cfg = load("ocp", batch_pages=100)
    pipeline(cfg, Shape(1, 8, 128, 8192), "layer_contiguous", tmp_path / "b", None)
    report = (tmp_path / "b" / "report.md").read_text()
    assert "| 子批轮数 |" in report and "| 1 | 0 | 1 | 8192 | 8 | 82 | 8 |" in report


def test_generator_and_cli_edge_cases(tmp_path, monkeypatch):
    from hbf.frontend.trace.synthetic import share
    from hbf.frontend.pages import prefill_batches
    assert share(100, .07) == 7 and share(100, .071) == 8 and share(3, 1) == 3
    cfg = load("ocp")
    shape = Shape(1, 1, 128, 1600)
    for mode in ("uniform_pages", "uniform_tokens"):  # 100 blocks x 7 % is 7 blocks, not 8
        layout = LAYOUTS["layer_contiguous"](cfg, shape)
        accesses = list(generate(cfg, layout, mode, fraction=.07, select_tokens=16))
        assert len({(a.kind, a.token_start) for a in accesses}) == 2 * 7
    striped = LAYOUTS["striped_contiguous"](cfg, Shape(1, 8, 128, 4096))
    pages = [a for a in generate(cfg, striped, "same_unit", count=3, unit=0)]  # skips padding pages
    assert len({striped.lpa(a.layer, a.kv_head, striped.tensor_index(a.kind), a.token_start // 16)
                for a in pages}) == 3
    for bad_base in (100, cfg.units):
        shifted = LAYOUTS["layer_contiguous"](cfg, Shape(1, 1, 128, 64), base=bad_base)
        with pytest.raises(ValueError, match="block page 0"):
            prefill_batches(cfg, shifted)
    aligned = LAYOUTS["layer_contiguous"](cfg, Shape(1, 1, 128, 64), base=cfg.units * cfg.pages_per_block)
    assert prefill_batches(cfg, aligned)[0].requests[0].lpa == cfg.units * cfg.pages_per_block
    with pytest.raises(ValueError, match="multiple of the unit count"):
        cfg.unit_lpa(512, 0, 512)
    # A link grade alone sets no credit limit: rounds stay at the bank bound.
    monkeypatch.setattr("sys.argv", ["hbf", "run", "--analysis-only", "--topology", "ocp", "--context", "32768",
                                     "--ucie-grade", "3", "--output", str(tmp_path / "u")])
    main()
    (row,) = csv.DictReader((tmp_path / "u" / "analysis_steps.csv").open())
    assert row["protocol_rounds"] == row["rounds"] == "32" and "MOCS = 不限" in (tmp_path / "u" / "report.md").read_text()
    monkeypatch.setattr("sys.argv", ["hbf", "run", "--analysis-only", "--layers", "0", "--output", str(tmp_path / "z")])
    with pytest.raises(SystemExit):
        main()
    with pytest.raises(ValueError):  # a generator error leaves no output directory behind
        pipeline(cfg, Shape(1, 8, 128, 8192), "layer_contiguous", tmp_path / "x", None, mode="same_unit",
                 generation={"count": 99})
    assert not (tmp_path / "x").exists() and not (tmp_path / "z").exists()


def test_core_channel_bound_follows_bandwidth():
    """Dense 8K layer on ocp: 256 pages per core-die channel. MQSim took 96.5, 42.3, 39.0 and 37.6 us
    (23.6, 10.3, 9.5 and 9.2 tR) at 384, 1,024, 1,536 and 3,072 GB/s with cache read, so the
    channel bound, not the 8 sensing rounds, decides at 384 GB/s."""
    bounds = {}
    for grade in (384, 1024, 1536, 3072):
        cfg = load("ocp", aggregate_gb_s=grade)
        layout = LAYOUTS["layer_contiguous"](cfg, Shape(1, 8, 128, 8192))
        (row,), (step,) = analyze(cfg, schedule(cfg, layout, generate(cfg, layout, "dense")))
        assert row["rounds"] == 8 and step["channel_rounds"] == row["channel_rounds"]
        bounds[grade] = row["channel_rounds"]
    assert bounds == {384: 22, 1024: 8, 1536: 6, 3072: 3}


def test_topology_preset_may_override_device_fields(tmp_path, monkeypatch):
    import json
    from hbf.device import config as cfg_mod
    params = json.loads(cfg_mod.PARAMETERS.read_text())
    params["topologies"]["half"] = {"channels": 16, "chips": 2, "planes": 16, "address_map": "ocp",
                                    "plane_rule": "independent", "aggregate_gb_s": 512}
    custom = tmp_path / "parameters.json"
    custom.write_text(json.dumps(params))
    monkeypatch.setattr(cfg_mod, "PARAMETERS", custom)
    cfg = cfg_mod.load("half")
    assert cfg.aggregate_gb_s == 512 and cfg.width == 32 and cfg.host_channels == 8
    assert cfg_mod.load("half", aggregate_gb_s=256).aggregate_gb_s == 256


def test_unwritten_reads_resets_block_on_page_zero_write():
    from hbf.backends.mqsim.schedule import unwritten_reads
    cfg = load("ocp", pages_per_block=4)
    block0 = [cfg.unit_lpa(0, p) for p in range(4)]
    other = [cfg.unit_lpa(1, p) for p in range(4)]
    batches = [
        Batch(-1, -1, 0, 0, tuple(Request(lpa, "W") for lpa in block0 + other)),
        Batch(0, 0, 0, cfg.interval_ns, (Request(block0[1], "R"), Request(block0[0], "W"))),
        Batch(1, 0, 0, 2 * cfg.interval_ns, (Request(block0[0], "R"), Request(block0[1], "R"), Request(other[1], "R"))),
    ]
    # Batch 1 reads page 1 before batch 1's page-0 write erases block 0; batch 2's read of page 1 is unwritten,
    # while page 0 of block 0 (rewritten) and page 1 of unit 1 (untouched) remain written.
    assert unwritten_reads(cfg, batches) == 1
