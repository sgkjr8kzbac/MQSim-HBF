"""T-PLANE: real die/plane hierarchy with multi-plane reads in MQSim."""
import csv
from collections import defaultdict

import pytest

from hbf.access import LAYOUTS, Shape, read_access
from hbf.analysis import locate
from hbf.cli import pipeline
from hbf.config import load
from hbf.controller import Batch, Request, schedule
from hbf.simulator import completions, simulate


def die_rounds(cfg, requests, group):
    """Sum over dies of the rounds each die needs (independent: group 1; lock-step: whole die)."""
    offsets = defaultdict(set)
    for req in requests:
        die, plane, offset = locate(cfg, req.lpa)
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
])
def test_plane_config_validation(overrides, message):
    with pytest.raises(ValueError, match=message):
        load("16x64", chips=1, **overrides)
