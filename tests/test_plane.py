"""T-PLANE: real die/plane hierarchy with multi-plane reads in MQSim."""
import csv
from collections import defaultdict

import pytest

from hbf.frontend.kv.layouts import LAYOUTS
from hbf.frontend.kv.tensors import Shape
from hbf.frontend.trace.tokens import read_access
from hbf.backends.rounds.analysis import locate, placement
from hbf.cli import pipeline
from hbf.device.config import load
from hbf.frontend.pages import Batch, Request, schedule_run
from hbf.backends.mqsim.parse import completions
from hbf.backends.mqsim.run import simulate


def die_rounds(cfg, requests, group, placed=None):
    """Sum over dies of the rounds each die needs (independent: group 1; lock-step: whole die)."""
    offsets = defaultdict(set)
    for req in requests:
        die, plane, offset = locate(cfg, req.lpa, placed)
        offsets[die, plane // group].add(offset)
    rounds = defaultdict(int)
    for (die, _), values in offsets.items():
        rounds[die] = max(rounds[die], len(values))
    return sum(rounds.values()), len(rounds)


@pytest.mark.parametrize("planes, dbsy", [(4, None), (8, None), (8, 0), (64, 100)])
def test_multiplane_command_time(binary, evidence, planes, dbsy):
    """One die: the first page goes out alone, the rest as one multi-plane command."""
    cfg = load("16x64", channels=1, chips=1, planes=planes, aggregate_gb_s=64,
               plane_rule="independent", dbsy_ns=dbsy)
    batches = [Batch(0, 0, 0, 0, tuple(Request(p) for p in range(planes)))]
    rows, _, _, stats = simulate(cfg, batches, evidence / f"t-plane-cmd-{planes}-{dbsy}", binary,
                                 allow_unwritten=True)
    sense, xfer, rest = cfg.read_ns + cfg.rb_ns, cfg.transfer_ns, planes - 1
    expected = (cfg.host_ns + cfg.multiplane_read_command_ns(1) + sense + xfer
                + cfg.multiplane_read_command_ns(rest) + sense + rest * xfer)
    assert rows[0]["mqsim_ns"] == expected
    assert (stats["Issued_Flash_Read_CMD"], stats["Issued_Flash_Multiplane_Read_CMD"]) == (1, 1)
    if planes == 4 and dbsy is None:
        assert expected == 10_369  # upstream table entries 1-4 are unchanged


def test_plane_rule_is_noop_without_multiplane(binary, evidence):
    batches = [Batch(0, 0, 0, 0, tuple(Request(p) for p in (0, 4, 8, 1, 2))),
               Batch(1, 0, 0, 1_000_000, tuple(Request(p) for p in (3, 5)))]
    runs = []
    for rule in (None, "independent", "lockstep"):
        cfg = load("16x64", channels=1, chips=4, aggregate_gb_s=64, plane_rule=rule)
        path = evidence / f"t-plane-noop-{rule}"
        simulate(cfg, batches, path, binary)
        runs.append(sorted(completions(path / "completions.csv"), key=lambda c: (c.arrival_ns, c.start_lba)))
    assert runs[0] == runs[1] == runs[2]


@pytest.fixture(scope="module")
def hierarchy_runs(binary, evidence):
    """16 dies x 64 planes, 5% of the pages of 8 heads, after a full prefill."""
    shape = Shape(1, 8, 128, 8192)
    runs = {}
    for rule in ("independent", "lockstep", None):
        cfg = load("16x64", chips=1, planes=64, blocks_per_plane=64, plane_rule=rule)
        path = evidence / f"t-plane-hierarchy-{rule}"
        _, layers, _, stats = pipeline(cfg, shape, "layer_contiguous", path, binary, mode="uniform_pages",
                                       generation={"fraction": .05, "seed": 7}, plane_groups=(1, 64),
                                       prefill_layout=True)
        analysis = list(csv.DictReader((path / "analysis_layers.csv").open()))
        runs[rule] = cfg, path, layers, stats, analysis
    return runs


def test_hierarchy_commands_follow_rounds(hierarchy_runs):
    for rule, group in (("independent", 1), ("lockstep", 64)):
        cfg, path, layers, stats, analysis = hierarchy_runs[rule]
        layout = LAYOUTS["layer_contiguous"](cfg, Shape(1, 8, 128, 8192))
        batches = schedule_run(cfg, layout, read_access(path / "access.csv"), prefill=True)
        reads = [req for batch in batches for req in batch.requests if req.op == "R"]
        rounds, dies = die_rounds(cfg, reads, group)
        commands = stats["Issued_Flash_Read_CMD"] + stats["Issued_Flash_Multiplane_Read_CMD"]
        # At most one extra command per die: the first page that reaches an idle die goes out alone.
        assert rounds <= commands <= rounds + dies, (rule, rounds, commands, dies)
        # Prefill programs one multi-plane command per die and offset.
        assert stats["Issued_Flash_Multiplane_Program_CMD"] > 0
    independent, lockstep = hierarchy_runs["independent"], hierarchy_runs["lockstep"]
    assert int(independent[4][0]["rounds"]) < int(independent[4][1]["rounds"])
    read_ns = {rule: next(row["mqsim_ns"] for row in run[2] if row["step"] == 0)
               for rule, run in hierarchy_runs.items()}
    assert read_ns["independent"] < read_ns["lockstep"]
    # Upstream compares page indices only; after a prefill each plane fills block 0 first,
    # so it behaves like lockstep here.
    assert read_ns[None] == read_ns["lockstep"]


@pytest.mark.parametrize("overrides, message", [
    ({"planes": 65}, "at most 64"),
    ({"plane_rule": "free"}, "plane_rule"),
    ({"dbsy_ns": 0}, "t_DBSY"),
    ({"read_plane_ns": 10}, "per-plane read command step"),
    ({"coalesce_ns": 0}, "coalescing window"),
    ({"coalesce_ns": 100, "scheduler": "OUT_OF_ORDER"}, "coalescing window"),
])
def test_plane_config_validation(overrides, message):
    with pytest.raises(ValueError, match=message):
        load("16x64", chips=1, **overrides)


READ_COMMANDS = {"48", "50"}  # CMD_READ_PAGE, CMD_READ_PAGE_MULTIPLANE


@pytest.mark.parametrize("layout_name, shape, mode", [
    ("vllm_interleaved", Shape(3, 5, 128, 256), "dense"),
    ("layer_contiguous", Shape(3, 5, 128, 1000), "uniform_pages"),
    ("vllm_interleaved", Shape(3, 5, 128, 1000), "uniform_pages"),
    ("striped_contiguous", Shape(2, 3, 128, 1000), "sink_recent"),
])
@pytest.mark.parametrize("rule, group", [("independent", 1), ("lockstep", 8)])
@pytest.mark.parametrize("coalesce", [None, 1000])
def test_mqsim_follows_static_placement(binary, evidence, layout_name, shape, mode, rule, group, coalesce):
    """After the whole-span prefill MQSim places pages where the OCP spec puts them.

    Per layer and die, MQSim then issues the rounds of the static analysis plus at most one lone
    first page; with the T-CMD coalescing window the counts are equal.
    """
    cfg = load("2x1", planes=8, blocks_per_plane=16, plane_rule=rule, aggregate_gb_s=128,
               coalesce_ns=coalesce)
    path = evidence / f"t-plane-align-{layout_name}-{shape.context}-{rule}-{coalesce}"
    pipeline(cfg, shape, layout_name, path, binary, mode=mode, physical=True, prefill_layout=True,
             generation={"fraction": .3, "seed": 5, "count": 4})
    layout = LAYOUTS[layout_name](cfg, shape)
    batches = schedule_run(cfg, layout, read_access(path / "access.csv"), prefill=True)
    offsets = placement(cfg, batches)
    assert all(offset == lpa // cfg.units for lpa, offset in offsets.items())
    written = {(int(row[3]), int(row[6]), int(row[7]), int(row[8])): int(row[9])
               for row in csv.reader((path / "physical.csv").open()) if row[1] == "issue" and row[2] not in READ_COMMANDS}
    for (channel, plane, block, page), lpa in written.items():
        assert block * cfg.pages_per_block + page == lpa // cfg.units  # the spec's static offset
    owner = {req.lpa: batch for batch in batches for req in batch.requests if req.op == "R"}
    issued = defaultdict(set)
    for time, event, command, channel, *_, lpa in csv.reader((path / "physical.csv").open()):
        if event == "issue" and command in READ_COMMANDS:
            batch = owner[int(lpa)]
            issued[batch.step, batch.layer, int(channel)].add(int(time))
    for batch in batches:
        needed = defaultdict(set)
        for req in batch.requests:
            if req.op == "R":
                die, plane, offset = locate(cfg, req.lpa)
                needed[die, plane // group].add(offset)
        rounds = defaultdict(int)
        for (die, _), values in needed.items():
            rounds[die] = max(rounds[die], len(values))
        for die, count in rounds.items():
            commands = len(issued[batch.step, batch.layer, die])
            assert commands == count if coalesce else count <= commands <= count + 1


@pytest.mark.parametrize("planes, step, window", [(8, 10, 500), (64, 0, 2000), (64, 10, 300), (4, None, 1000)])
def test_coalesced_command_time(binary, evidence, planes, step, window):
    """T-CMD: after the window the whole die leaves as one command of cmd(1) + (n-1) * step."""
    cfg = load("16x64", channels=1, chips=1, planes=planes, aggregate_gb_s=64, plane_rule="independent",
               read_plane_ns=step, coalesce_ns=window)
    batches = [Batch(0, 0, 0, 0, tuple(Request(p) for p in range(planes)))]
    rows, _, _, stats = simulate(cfg, batches, evidence / f"t-cmd-{planes}-{step}-{window}", binary,
                                 allow_unwritten=True)
    expected = (window + cfg.multiplane_read_command_ns(planes) + cfg.read_ns + cfg.rb_ns
                + planes * cfg.transfer_ns + cfg.host_ns)
    assert rows[0]["mqsim_ns"] == expected
    assert (stats["Issued_Flash_Read_CMD"], stats["Issued_Flash_Multiplane_Read_CMD"]) == (0, 1)
    if step is not None:
        assert cfg.multiplane_read_command_ns(planes) == 290 + (planes - 1) * step


def test_patch_series_lists_every_patch():
    from hbf.backends.mqsim.build import patch_series
    names = [path.name for path in patch_series()]
    assert names[-4:] == ["T-HBFW1.patch", "T-HBFW2.patch", "T-HBFW3.patch", "T-HBFW4.patch"] and len(names) == len(set(names)) == 14


def cache_cfg(**overrides):
    return load("16x64", **({"channels": 1, "chips": 1, "planes": 64, "aggregate_gb_s": 64,
                             "plane_rule": "independent", "read_plane_ns": 10, "coalesce_ns": 500}
                            | overrides))


@pytest.mark.parametrize("name, overrides, pages, plain, cached", [
    # One plane: the next sensing hides only the 64 ns transfer of the previous page.
    ("chip", {"planes": 1, "plane_rule": None, "read_plane_ns": None, "coalesce_ns": None}, 4, 17_881, 17_689),
    # 64 planes, sensing-bound: 500 + 920 + 4 x 4116 + 3 x 920 + 64 x 64 + 1.
    ("sense-bound", {}, 256, 37_029, 24_741),
    # 64 planes at 16 GB/s, transfer-bound: the third sensing waits until the first batch left the cache.
    ("transfer-bound", {"aggregate_gb_s": 16}, 192, 64_761, 56_529),
])
def test_cache_read_timing(binary, evidence, name, overrides, pages, plain, cached):
    observed = []
    for cache in (False, True):
        cfg = cache_cfg(cache_read=cache, **overrides)
        batches = [Batch(0, 0, 0, 0, tuple(Request(p) for p in range(pages)))]
        path = evidence / f"t-cache-{name}-{cache}"
        rows, _, _, stats = simulate(cfg, batches, path, binary, physical=True, allow_unwritten=True)
        observed.append(rows[0]["mqsim_ns"])
        assert stats["Issued_Flash_Read_CMD"] + stats["Issued_Flash_Multiplane_Read_CMD"] == pages // cfg.planes
    assert observed == [plain, cached]
    if name == "transfer-bound":
        issues = sorted({int(row[0]) for row in csv.reader((path / "physical.csv").open()) if row[1] == "issue"})
        assert issues == [500, 5_536, 22_840]


def test_cache_read_keeps_rounds(binary, evidence):
    """Cache read changes when rounds start, not how many there are."""
    shape = Shape(2, 8, 128, 8192)
    counts = {}
    for cache in (False, True):
        cfg = load("16x1", planes=64, blocks_per_plane=64, plane_rule="lockstep", read_plane_ns=10,
                   coalesce_ns=1000, cache_read=cache)
        path = evidence / f"t-cache-rounds-{cache}"
        _, layers, _, stats = pipeline(cfg, shape, "layer_contiguous", path, binary, mode="uniform_pages",
                                       generation={"fraction": .1, "seed": 3}, prefill_layout=True)
        batches = schedule_run(cfg, LAYOUTS["layer_contiguous"](cfg, shape), read_access(path / "access.csv"),
                               prefill=True)
        rounds = sum(die_rounds(cfg, [r for r in b.requests if r.op == "R"], 64)[0]
                     for b in batches if any(r.op == "R" for r in b.requests))
        counts[cache] = (stats["Issued_Flash_Read_CMD"] + stats["Issued_Flash_Multiplane_Read_CMD"], rounds,
                         sum(row["mqsim_ns"] for row in layers if row["step"] == 0))
    assert counts[False][0] == counts[False][1] == counts[True][0]
    assert counts[True][2] < counts[False][2]


def test_cache_read_does_not_starve_idle_chips(binary, evidence):
    """Chip as plane with 64 chips per channel: cache read must not be slower than without it.

    The cache-priority pass used to admit only cache-ready chips, so idle chips waited until all
    queued data left the channel: a dense 8K layer on 16x64 took 200,778 ns with cache read and
    184,612 ns without it. Idle chips may now be issued in that pass too.
    """
    shape = Shape(1, 8, 128, 8192)
    latency = {}
    for cache in (False, True):
        cfg = load("16x64", blocks_per_plane=64, cache_read=cache)
        _, layers, _, _ = pipeline(cfg, shape, "layer_contiguous", evidence / f"t-cache-idle-{cache}", binary)
        latency[cache] = layers[0]["mqsim_ns"]
    assert latency[True] <= latency[False]


def test_cache_read_rejects_suspension():
    with pytest.raises(ValueError, match="T-CACHE"):
        load("16x64", cache_read=True, suspend="PROGRAM")


def physical_invariants(path, cfg):
    """Sensing on a die never overlaps and data never leaves before its sensing ended."""
    sense_end, sensing, errors, ready = defaultdict(list), defaultdict(list), [], 0
    for time, event, command, channel, chip, die, plane, block, page, lpa in csv.reader((path / "physical.csv").open()):
        if command not in READ_COMMANDS:
            continue
        time, key = int(time), (channel, chip, die, plane, block, page, lpa)
        ready += event == "ready"
        if event == "cmd_done":
            sensing[channel, chip, die].append((time, time + cfg.read_ns + cfg.rb_ns))
            sense_end[key].append(time + cfg.read_ns + cfg.rb_ns)
        elif event == "data_start" and (not sense_end[key] or time < sense_end[key].pop(0)):
            errors.append(("data before sensing ended", key, time))
    for die, spans in sensing.items():
        spans = sorted(set(spans))
        errors += [("overlapping sensing", die, a, b) for a, b in zip(spans, spans[1:]) if b[0] < a[1]]
    return errors, ready


@pytest.mark.parametrize("seed", range(6))
def test_cache_read_random_mixed_workloads(binary, evidence, seed):
    """Random reads and writes, overlapping batches, several chips and dies, with and without T-CACHE."""
    import random
    rng = random.Random(seed)
    topology, extra = [("1x4", {}), ("1x2", {"dies": 2, "planes": 4, "plane_rule": "independent"}),
                       ("2x2", {"planes": 16, "plane_rule": "lockstep", "read_plane_ns": 10, "coalesce_ns": 300}),
                       ("1x4", {"dies": 2, "planes": 8, "plane_rule": "independent", "coalesce_ns": 200}),
                       ("1x1", {"dies": 4, "planes": 2, "plane_rule": "independent"})][seed % 5]
    batches, time = [], 0
    for index in range(rng.randint(3, 8)):
        time += rng.choice([1, 300, 2000, 20000])
        lpas = rng.sample(range(600), rng.randint(1, 100))
        batches.append(Batch(0, index, 0, time, tuple(Request(l, "W" if rng.random() < .2 else "R") for l in lpas)))
    reads = sum(req.op == "R" for batch in batches for req in batch.requests)
    for cache in (False, True):
        cfg = load(topology, blocks_per_plane=32, aggregate_gb_s=16, cache_read=cache, **extra)
        path = evidence / f"t-cache-fuzz-{seed}-{cache}"
        simulate(cfg, batches, path, binary, physical=True, overlap=True, timeout=120, allow_unwritten=True)
        errors, ready = physical_invariants(path, cfg)
        assert errors == [] and ready == reads


def test_cache_read_priority_and_window(binary, evidence):
    """Only cache-ready reads go ahead of queued data out, and nothing leaves before the window closes."""
    cfg = load("1x4", planes=64, plane_rule="independent", read_plane_ns=10, coalesce_ns=500,
               blocks_per_plane=64)
    reads = lambda lpas, op="R": tuple(Request(l, op) for l in lpas)
    mixed = [Batch(0, 0, 0, 0, reads(range(1024))), Batch(0, 1, 0, 700, reads(range(2048, 2100), "W")),
             Batch(0, 2, 0, 900, reads(range(1024, 2048)))]
    last = {}
    for cache in (False, True):
        rows, _, _, _ = simulate(load("1x4", planes=64, plane_rule="independent", read_plane_ns=10,
                                      coalesce_ns=500, blocks_per_plane=64, cache_read=cache),
                                 mixed, evidence / f"t-cache-mixed-{cache}", binary, overlap=True,
                                 allow_unwritten=True)
        last[cache] = rows[-1]["complete_ns"]
    assert last[True] < last[False]  # writes no longer cut in front of other chips' data out
    single = load("1x1", planes=64, plane_rule="independent", read_plane_ns=10, coalesce_ns=500,
                  blocks_per_plane=64, aggregate_gb_s=16, cache_read=True)
    path = evidence / "t-cache-window"
    simulate(single, [Batch(0, 0, 0, 0, reads(range(64))), Batch(0, 1, 0, 6000, reads(range(64, 128)))],
             path, binary, physical=True, overlap=True, allow_unwritten=True)
    issues = sorted({int(row[0]) for row in csv.reader((path / "physical.csv").open()) if row[1] == "issue"})
    assert issues[0] == 500 and issues[1] >= 6000 + 500


def test_multiplane_runs_reject_unwritten_reads(binary, tmp_path):
    """The OCP spec answers reads of unwritten pages with error 0x7 (Table 12)."""
    cfg = load("2x1", planes=8, blocks_per_plane=16, plane_rule="independent", aggregate_gb_s=128)
    with pytest.raises(ValueError, match="never written"):
        pipeline(cfg, Shape(1, 2, 128, 256), "layer_contiguous", tmp_path / "no-prefill", binary,
                 mode="uniform_pages", prefill_layout=False)
    one_plane = load("2x8", blocks_per_plane=16, aggregate_gb_s=128)
    _, _, _, stats = pipeline(one_plane, Shape(1, 2, 128, 256), "layer_contiguous", tmp_path / "chips", binary,
                              mode="uniform_pages")
    assert stats["unwritten_reads"] > 0  # allowed with one plane per chip, where the offset has no effect


def test_default_cli_runs_compliant_ocp_with_auto_prefill(binary, tmp_path):
    """R2 acceptance: running with no device/topology/prefill/timing flags gives compliant ocp results."""
    import json
    from hbf.cli import main
    out_a = tmp_path / "default-analysis"
    main(["run", "--analysis-only", "--layers", "1", "--heads", "2", "--context", "256", "--output", str(out_a)])
    assert (out_a / "analysis_layers.csv").is_file()
    report_a = (out_a / "report.md").read_text()
    assert "拓扑 `ocp`" in report_a and "T-CACHE 开" in report_a and "## 读时间估计" in report_a
    out_m = tmp_path / "default-mqsim"
    main(["run", "--layers", "1", "--heads", "2", "--context", "256", "--blocks-per-plane", "32", "--output", str(out_m)])
    meta = json.loads((out_m / "meta.json").read_text())
    stats = json.loads((out_m / "statistics.json").read_text())
    assert meta["prefill"] and stats["unwritten_reads"] == 0 and stats["Issued_Flash_Multiplane_Program_CMD"] > 0
    assert (out_m / "layers.csv").is_file() and (out_m / "analysis_layers.csv").is_file()


def test_independent_rule_does_not_combine_mismatched_page_writes(binary, evidence):
    """OCP section 11.2.2: multi-plane writes on a die require the same page number.

    Batch 0 writes only plane 0 (advancing its write frontier to page 1 while plane 1 stays at
    page 0). Batch 1 writes both plane 0 and plane 1 together with a coalescing window: because
    their page indices differ (1 vs 0), T-PLANE independent must not merge them into a multi-plane
    program command.
    """
    cfg = load("16x64", channels=1, chips=1, planes=2, aggregate_gb_s=64,
               plane_rule="independent", coalesce_ns=500)
    mismatched = [
        Batch(-1, 0, 0, 0, (Request(0, "W"),)),
        Batch(-1, 0, 1, cfg.interval_ns, (Request(2, "W"), Request(1, "W"))),
    ]
    _, _, _, stats = simulate(cfg, mismatched, evidence / "t-plane-write-mismatch", binary)
    assert stats["Issued_Flash_Multiplane_Program_CMD"] == 0
    assert stats["Issued_Flash_Program_CMD"] == 3
    # Positive control: both planes at page 0 do combine into one multi-plane program command.
    matched = [Batch(-1, 0, 0, 0, (Request(0, "W"), Request(1, "W")))]
    _, _, _, stats_ok = simulate(cfg, matched, evidence / "t-plane-write-match", binary)
    assert stats_ok["Issued_Flash_Multiplane_Program_CMD"] == 1
    assert stats_ok["Issued_Flash_Program_CMD"] == 0
