"""T-PLANE: real die/plane hierarchy with multi-plane reads in MQSim."""
import csv
from collections import defaultdict

import pytest

from hbf.access import LAYOUTS, Shape, read_access
from hbf.analysis import locate, placement
from hbf.cli import pipeline
from hbf.config import load
from hbf.controller import Batch, Request, schedule
from hbf.simulator import completions, simulate


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
    rows, _, _, stats = simulate(cfg, batches, evidence / f"t-plane-cmd-{planes}-{dbsy}", binary)
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
        batches = schedule(cfg, layout, read_access(path / "access.csv"))
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


@pytest.mark.parametrize("layout_name, shape, mode, prefill", [
    ("vllm_interleaved", Shape(3, 5, 128, 256), "dense", True),
    ("layer_contiguous", Shape(3, 5, 128, 1000), "uniform_pages", True),
    ("vllm_interleaved", Shape(3, 5, 128, 1000), "uniform_pages", False),
])
@pytest.mark.parametrize("rule, group", [("independent", 1), ("lockstep", 8)])
@pytest.mark.parametrize("coalesce", [None, 1000])
def test_analysis_offsets_follow_mqsim_allocation(binary, evidence, layout_name, shape, mode, prefill,
                                                 rule, group, coalesce):
    """Per layer and die, MQSim issues the analysed rounds plus at most one lone first page.

    With the T-CMD coalescing window the lone first page disappears and the counts are equal.
    """
    cfg = load("2x1", planes=8, blocks_per_plane=16, plane_rule=rule, aggregate_gb_s=128,
               coalesce_ns=coalesce)
    path = evidence / f"t-plane-align-{layout_name}-{shape.context}-{prefill}-{rule}-{coalesce}"
    pipeline(cfg, shape, layout_name, path, binary, mode=mode, generation={"fraction": .3, "seed": 5},
             physical=True, prefill_layout=prefill)
    batches = schedule(cfg, LAYOUTS[layout_name](cfg, shape), read_access(path / "access.csv"))
    offsets = placement(cfg, batches)
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
                die, plane, offset = locate(cfg, req.lpa, offsets)
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
    rows, _, _, stats = simulate(cfg, batches, evidence / f"t-cmd-{planes}-{step}-{window}", binary)
    expected = (window + cfg.multiplane_read_command_ns(planes) + cfg.read_ns + cfg.rb_ns
                + planes * cfg.transfer_ns + cfg.host_ns)
    assert rows[0]["mqsim_ns"] == expected
    assert (stats["Issued_Flash_Read_CMD"], stats["Issued_Flash_Multiplane_Read_CMD"]) == (0, 1)
    if step is not None:
        assert cfg.multiplane_read_command_ns(planes) == 290 + (planes - 1) * step


def test_patch_series_lists_every_patch():
    from hbf.simulator import patch_series
    names = [path.name for path in patch_series()]
    assert names[-3:] == ["T-SUSP.patch", "T-CMD.patch", "T-CACHE.patch"] and len(names) == len(set(names)) == 9


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
        rows, _, _, stats = simulate(cfg, batches, path, binary, physical=True)
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
        batches = schedule(cfg, LAYOUTS["layer_contiguous"](cfg, shape), read_access(path / "access.csv"))
        offsets = placement(cfg, batches)
        rounds = sum(die_rounds(cfg, [r for r in b.requests if r.op == "R"], 64, offsets)[0]
                     for b in batches if any(r.op == "R" for r in b.requests))
        counts[cache] = (stats["Issued_Flash_Read_CMD"] + stats["Issued_Flash_Multiplane_Read_CMD"], rounds,
                         sum(row["mqsim_ns"] for row in layers if row["step"] == 0))
    assert counts[False][0] == counts[False][1] == counts[True][0]
    assert counts[True][2] < counts[False][2]


def test_cache_read_rejects_suspension():
    with pytest.raises(ValueError, match="T-CACHE"):
        load("16x64", cache_read=True, suspend="PROGRAM")
