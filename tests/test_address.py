"""Static address mapping (hbf/device/address.py): scalar parity and MQSim I2 verification."""
import csv
import random

import numpy as np
import pytest

from hbf.backends.mqsim.run import simulate
from hbf.frontend.pages import Batch, Request
from hbf.backends.rounds.analysis import locate
from hbf.device.address import (block_and_page, block_ids, block_page, device_lpa, device_lpas,
                                host_lpa, host_lpas, place)
from hbf.device.config import load


@pytest.mark.parametrize("topology,extra", [
    ("ocp", {}),
    ("ocp-8ch", {}),
    ("64x16", {}),
    ("16x64", {}),
    ("256x4", {}),
    ("4x1", {"planes": 16, "plane_rule": "independent"}),
])
def test_vectorized_mapping_matches_scalar_config(topology, extra):
    cfg = load(topology, **extra)
    rng = random.Random(42)
    cap = min(cfg.capacity_sectors // cfg.spp, cfg.units * 300)
    lpas = np.array(sorted(rng.sample(range(cap), 256)), dtype=np.int64)
    dev = device_lpas(cfg, lpas)
    assert dev.tolist() == [cfg.device_lpa(int(x)) for x in lpas]
    assert dev.tolist() == [device_lpa(cfg, int(x)) for x in lpas]
    assert host_lpas(cfg, dev).tolist() == lpas.tolist()
    assert [host_lpa(cfg, int(d)) for d in dev] == lpas.tolist()

    placed = place(cfg, lpas)
    blks, pgs = block_and_page(cfg, placed)
    assert block_ids(cfg, placed).tolist() == blks.tolist()
    for i, x in enumerate(lpas.tolist()):
        ch, chip, die, plane = cfg.physical(x)
        h_die, h_plane, offset = locate(cfg, x)
        u, b, p = block_page(cfg, x)
        assert (int(placed.die[i]), int(placed.plane[i]), int(placed.offset[i])) == (h_die, h_plane, offset)
        assert int(placed.channel[i]) == ch and int(placed.unit[i]) == cfg.unit(x) == u
        assert int(placed.host[i]) == cfg.host_channel(x)
        assert int(blks[i]) == u + cfg.units * b and int(pgs[i]) == p


def test_static_mapping_matches_mqsim_i2_physical_reads(binary, tmp_path):
    """After a block-aligned prefill, random reads in MQSim I2 (physical.csv) match address.place."""
    cfg = load("ocp", channels=4, chips=2, planes=4, aggregate_gb_s=128,
               pages_per_block=8, blocks_per_plane=16)
    span = cfg.units * cfg.pages_per_block  # 1 full block per plane
    rng = random.Random(19)
    sample = sorted(rng.sample(range(span), 48))
    batches = [
        Batch(-1, -1, 0, 0, tuple(Request(lpa, "W") for lpa in range(span))),
        Batch(0, 0, 0, cfg.interval_ns, tuple(Request(lpa, "R") for lpa in sample)),
    ]
    out = tmp_path / "i2-addr"
    simulate(cfg, batches, out, binary, physical=True)

    placed = place(cfg, np.array(sample, dtype=np.int64))
    _, expected_pages = block_and_page(cfg, placed)
    expected = {
        sample[i]: (int(placed.channel[i]), int(placed.die[i]), int(placed.plane[i]), int(expected_pages[i]))
        for i in range(len(sample))
    }
    seen = {}
    for row in csv.reader((out / "physical.csv").open()):
        if row[1] == "issue" and row[2] in {"48", "50"}:
            ch, chip, die, plane, _block, page, dev = map(int, row[3:10])
            lpa = host_lpa(cfg, dev)
            hbf_die = (ch * cfg.chips + chip) * cfg.dies + die
            seen[lpa] = (ch, hbf_die, plane, page)
    assert seen == expected
