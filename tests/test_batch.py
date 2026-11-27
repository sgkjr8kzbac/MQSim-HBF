"""Request placement (hbf/batch.py): page tables, read parallelism per policy and write order legality."""
import json

import numpy as np
import pytest

from hbf.access import LAYOUTS, Shape, generate
from hbf.analysis import analyze_accesses
from hbf.batch import (BatchRequest, BatchSpec, PageTableLayout, check_write_order, generate_batch,
                       partition_geometry, place_batch)
from hbf.cli import main
from hbf.config import load

HBF_FLAGS = ["--topology", "ocp", "--read-plane-ns", "0", "--coalesce-ns", "1", "--cache-read", "--analysis-only"]


def batch(allocation, requests, **kwargs):
    return BatchSpec(tuple(BatchRequest(*r) for r in requests), allocation, **kwargs)


def test_check_write_order_counts():
    cfg = load("ocp", pages_per_block=4)
    row = lambda k: k * cfg.units  # page k of the block of bank 0
    assert check_write_order(cfg, [row(0), row(1), row(2)])["legal"]
    assert check_write_order(cfg, [row(0), row(2)])["skips"] == 1
    assert check_write_order(cfg, [row(0), row(1), row(1)])["overlaps"] == 1
    erased = check_write_order(cfg, [row(0), row(1), row(0)])
    assert erased["live_erases"] == 1 and not erased["legal"]
    assert check_write_order(cfg, [row(1)])["skips"] == 1  # a block starts at page 0
    assert check_write_order(cfg, [row(0), row(4), row(1), row(5)])["blocks"] == 2


def test_dedicated_superblocks_hold_one_request_each():
    # ppb 4: a superblock is 4,096 pages; a request of 2 layers x 8 heads x 2 tensors x 128 pages fills one.
    cfg = load("ocp", pages_per_block=4)
    spec = batch("dedicated", [(2048, 0)] * 3, intra="layer_contiguous")
    layout = PageTableLayout(cfg, Shape(2, 8, 128, 1), spec)
    table = layout.snapshot.table
    for r in range(3):
        assert sorted(table[r].ravel().tolist()) == list(range(4096 * r, 4096 * (r + 1)))
    rows, _ = analyze_accesses(cfg, layout, list(generate_batch(cfg, layout, "dense")))
    assert [row["pages"] for row in rows] == [3 * 2048] * 2 and [row["rounds"] for row in rows] == [6, 6]
    assert check_write_order(cfg, layout.snapshot.written())["legal"]


def test_partitions_confine_a_request():
    cfg = load("ocp")
    parts, banks, to_lpa = partition_geometry(cfg, "die")
    assert (parts, banks) == (64, 16)
    lpas = np.array([to_lpa(5, k) for k in range(32)])  # host channel 1, die 1
    placed_banks = {cfg.unit(int(lpa)) for lpa in lpas}
    assert len(placed_banks) == 16 and all(lpa % 16 == 1 for lpa in lpas)
    # One 8B-shaped request per channel: 64 banks per request instead of 1,024.
    spec = batch("packed", [(2048, 0)] * 2, partition="channel", intra="layer_contiguous")
    layout = PageTableLayout(cfg, Shape(1, 8, 128, 1), spec)
    (row,), _ = analyze_accesses(cfg, layout, list(generate_batch(cfg, layout, "dense")))
    assert row["pages"] == 2 * 2048 and row["rounds"] == 2048 // 64 and row["active_planes"] == 128


def test_layer_major_snapshot_equals_the_phase_a_layout():
    cfg = load("ocp")
    region = LAYOUTS["layer_contiguous"](cfg, Shape(2, 3 * 8, 128, 2048))
    spec = batch("layer_major", [(2048, 0)] * 3, intra="layer_contiguous")
    snapshot = place_batch(cfg, Shape(2, 8, 128, 1), spec, region.tensors, region.tokens_per_page)
    for r in range(3):
        for layer in range(2):
            for h in range(8):
                for t in range(2):
                    blocks = np.arange(128)
                    assert (snapshot.table[r, layer, h, t] == region.lpa_array(
                        layer, np.full(128, r * 8 + h), np.full(128, t), blocks)).all()
    # Written request by request, the regions of one block belong to different requests (request 0
    # holds rows 0, 3, 6, ... of every bank), so the per-block order breaks; written as one prefill in
    # address order (phase A's --prefill) the same pages are legal.
    assert not check_write_order(cfg, snapshot.written())["legal"]
    assert check_write_order(cfg, np.sort(snapshot.written()))["legal"]


def test_appends_break_reserved_and_pooled_layouts_only():
    cfg = load("ocp", pages_per_block=4)
    requests = [(1024, 256), (512, 64), (768, 0)]
    verdicts = {}
    for allocation in ("layer_major", "vllm_pool", "dedicated", "packed"):
        spec = batch(allocation, requests, intra="layer_contiguous")
        snapshot = place_batch(cfg, Shape(2, 4, 128, 1), spec, LAYOUTS["layer_contiguous"](
            cfg, Shape(1, 1, 128, 64)).tensors, (16, 16))
        verdicts[allocation] = check_write_order(cfg, snapshot.written())
        assert snapshot.kinds["decode"] == 2 * 4 * 2 * (256 // 16 + 64 // 16)
    assert not verdicts["layer_major"]["legal"] and not verdicts["vllm_pool"]["legal"]
    assert verdicts["dedicated"]["legal"] and verdicts["packed"]["legal"]


def test_flush_order_spreads_generated_pages():
    cfg = load("ocp")
    # A flush of one tensor is L x H = 256 pages: with layer order layer 0 always takes the first 8, at
    # frontiers that are multiples of 256, i.e. 4 x 8 banks; rotate moves it on every flush.
    for order, banks in (("layer", 32), ("rotate", 1024)):
        spec = batch("packed", [(2048, 2048)] * 2, flush_order=order, intra="layer_contiguous")
        layout = PageTableLayout(cfg, Shape(32, 8, 128, 1), spec)
        lpas = layout.snapshot.table[:, 0, :, :, 128:].ravel()  # layer 0's generated pages
        assert len({cfg.unit(int(lpa)) for lpa in lpas}) == banks, order


def test_vllm_pool_blocks_and_ids():
    cfg = load("ocp")
    fresh = PageTableLayout(cfg, Shape(1, 8, 128, 1), batch("vllm_pool", [(16 * 64, 0)]))
    (row,), _ = analyze_accesses(cfg, fresh, list(generate_batch(cfg, fresh, "dense")))
    assert row["pages"] == 1024 and row["rounds"] == 1  # 64 blocks x 8 heads x 2 tensors on 1,024 banks
    rounds = {}
    for order in ("fresh", "shuffled:3"):
        pool = PageTableLayout(cfg, Shape(1, 8, 128, 1), batch("vllm_pool", [(16 * 64, 0)] * 4, free_order=order))
        (row,), _ = analyze_accesses(cfg, pool, list(generate_batch(cfg, pool, "dense")))
        rounds[order] = row["rounds"]
    # Fresh ids fill the pool in order (4 rounds for 4 x 1,024 pages); per-request reads are what matter.
    assert rounds["fresh"] == 4 and rounds["shuffled:3"] == 4
    with pytest.raises(ValueError, match="block_tokens"):
        PageTableLayout(cfg, Shape(1, 8, 128, 1), batch("vllm_pool", [(64, 0)], block_tokens=24))


def test_generate_batch_respects_each_context():
    cfg = load("ocp")
    layout = PageTableLayout(cfg, Shape(1, 2, 128, 1), batch("packed", [(1000, 0), (100, 24)]))
    accesses = list(generate_batch(cfg, layout, "dense"))
    assert {(a.kv_head, a.token_end) for a in accesses} == {(0, 992), (1, 992), (2, 112), (3, 112)}
    tokens = list(generate_batch(cfg, layout, "uniform_tokens", fraction=.5, seed=2))
    assert all(a.token_end <= (992 if a.kv_head < 2 else 112) for a in tokens)


def test_placement_flags_in_the_pipeline(tmp_path):
    out = tmp_path / "p"
    main(HBF_FLAGS + ["--model", "llama-3.1-8b", "--context", "2048", "--batch", "3", "--contexts", "2048,1024,512",
                      "--generated", "64", "--alloc", "packed", "--check-writes", "--output", str(out)])
    meta = json.loads((out / "meta.json").read_text())
    assert meta["batch"]["allocation"] == "packed" and meta["batch"]["requests"] == [[2048, 64], [1024, 64], [512, 64]]
    assert meta["batch"]["write_check"]["legal"] and "写入顺序检查" in (out / "report.md").read_text()
    main(HBF_FLAGS + ["--layers", "2", "--context", "2048", "--batch", "3", "--check-writes",
                      "--output", str(tmp_path / "static")])
    static = json.loads((tmp_path / "static" / "meta.json").read_text())["batch"]["write_check"]
    assert not static["legal"] and static["kinds"]["decode"] == 0
    for index, extra in enumerate((["--alloc", "packed", "--mode", "uniform_pages"],
                                   ["--contexts", "1,2"], ["--alloc", "dedicated", "--partition", "die",
                                                           "--topology", "64x16"])):
        with pytest.raises(SystemExit):
            main(HBF_FLAGS + ["--layers", "1", "--context", "1024", "--batch", "3"] + extra
                 + ["--output", str(tmp_path / str(index))])
        assert not (tmp_path / str(index)).exists()


def test_default_batch_flags_keep_phase_a_outputs(tmp_path):
    for name, extra in (("plain", []), ("explicit", ["--alloc", "layer_major", "--generated", "0"])):
        main(HBF_FLAGS + ["--layers", "2", "--context", "2048", "--batch", "3", "--mode", "uniform_tokens",
                          "--timing", "--output", str(tmp_path / name)] + extra)
    for file in ("analysis_layers.csv", "analysis_steps.csv", "access.csv"):
        assert (tmp_path / "plain" / file).read_bytes() == (tmp_path / "explicit" / file).read_bytes()
