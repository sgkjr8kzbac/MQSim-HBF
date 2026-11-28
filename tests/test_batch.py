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
    assert check_write_order(cfg, [row(0), row(1), row(1)])["overwrites"] == 1
    erased = check_write_order(cfg, [row(0), row(1), row(0)])
    assert erased["live_erases"] == 1 and not erased["legal"]
    # A page-0 write starts a new erase epoch: the rewrite after it is in order, not an overwrite.
    again = check_write_order(cfg, [row(k) for k in (0, 1, 2, 3, 0, 1, 2, 3)])
    assert (again["live_erases"], again["overwrites"], again["skips"], again["backward"]) == (1, 0, 0, 0)
    back = check_write_order(cfg, [row(0), row(2), row(1), row(1)])
    assert (back["skips"], back["backward"], back["overwrites"]) == (1, 1, 1)
    assert check_write_order(cfg, [row(1)])["skips"] == 1  # a block starts at page 0
    assert check_write_order(cfg, [row(0), row(4), row(1), row(5)])["blocks"] == 2


def test_dedicated_superblocks_hold_one_request_each():
    # ppb 4: a superblock is 4,096 pages; a request of 2 layers x 8 heads x 2 tensors x 128 pages fills one.
    cfg = load("ocp", pages_per_block=4)
    spec = batch("dedicated", [(2048, 0)] * 3, intra="layer_contiguous", align="row")
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


def test_windowed_tensors_flush_after_admission():
    # kivi2: the body [0, -128] fills a 2-bit K page (128 tokens) only 128 tokens after the context does.
    cfg = load("ocp")
    from hbf.formats import parse_format
    tensors = parse_format("kivi2", 128).tensors
    for allocation in ("packed", "dedicated", "vllm_pool", "layer_major"):
        layout = PageTableLayout(cfg, Shape(1, 2, 128, 1), batch(allocation, [(4000, 128), (3000, 300)]),
                                 tensors=tensors)
        full = layout.snapshot.full_pages
        assert full[0, 0] == (4128 - 128) // 128 and full[1, 0] == (3300 - 128) // 128
        list(generate_batch(cfg, layout, "dense"))


def test_equal_contexts_set_the_stored_context(tmp_path):
    main(HBF_FLAGS + ["--layers", "1", "--batch", "2", "--contexts", "4096,4096", "--output", str(tmp_path / "e")])
    meta = json.loads((tmp_path / "e" / "meta.json").read_text())
    assert meta["shape"]["context"] == 4096 and meta["batch"] is None


def test_mixed_contexts_check_capacity_per_placed_table():
    cfg = load("ocp")
    spec = batch("packed", [(131072, 0)] + [(4096, 0)] * 31, intra="layer_contiguous")
    layout = PageTableLayout(cfg, Shape(32, 8, 128, 1), spec)  # a dense phase A layout of this would not fit
    assert layout.snapshot.table.dtype.name == "int32" and layout.span < cfg.capacity_sectors // cfg.spp


def test_request_bytes_use_each_context(tmp_path):
    main(HBF_FLAGS + ["--model", "llama-3.1-8b", "--batch", "2", "--contexts", "4096,1024", "--alloc", "packed",
                      "--output", str(tmp_path / "b")])
    system = json.loads((tmp_path / "b" / "meta.json").read_text())["system"]
    assert system["kv_bytes_per_request"] == (4096 + 1024) // 2 * 128 * 1024  # 128 KiB per token (8B BF16)
    main(HBF_FLAGS + ["--model", "llama-3.1-8b", "--batch", "2", "--context", "4096", "--format", "kivi2",
                      "--alloc", "packed", "--generated", "16", "--output", str(tmp_path / "k")])
    system = json.loads((tmp_path / "k" / "meta.json").read_text())["system"]
    assert system["hbm_tier_bytes_per_request"] == 2 * 128 * 256 * 32 * 8  # the residual of one request


def test_write_check_only_on_request(tmp_path):
    main(HBF_FLAGS + ["--layers", "1", "--context", "1024", "--batch", "2", "--generated", "16", "--alloc", "packed",
                      "--output", str(tmp_path / "n")])
    assert json.loads((tmp_path / "n" / "meta.json").read_text())["batch"]["write_check"] is None
    with pytest.raises(SystemExit):
        main(HBF_FLAGS + ["--layers", "1", "--context", "1024", "--batch", "2", "--alloc", "packed",
                          "--free-order", "shuffled:1", "--output", str(tmp_path / "x")])


def test_stagger_spreads_short_requests():
    # kivi2 at 2K: one request's layer is 96 HBF pages; with every prompt at bank 0 of a row all eight
    # requests' layer l sit on the same banks (5.3x the minimum rounds); staggering spreads them.
    from hbf.formats import parse_format
    cfg = load("ocp")
    rounds = {}
    for align in ("row", "stagger"):
        for allocation in ("packed", "dedicated"):
            layout = PageTableLayout(cfg, Shape(4, 8, 128, 1), batch(allocation, [(2048, 0)] * 8, align=align),
                                     tensors=parse_format("kivi2", 128).tensors)
            (row, *_), _ = analyze_accesses(cfg, layout, list(generate_batch(cfg, layout, "dense")))
            rounds[allocation, align] = row["rounds"] / row["min_rounds"]
            assert check_write_order(cfg, layout.snapshot.written())["legal"]
    assert rounds["packed", "row"] > 5 and rounds["packed", "stagger"] < 2
    assert rounds["dedicated", "row"] > 5 and rounds["dedicated", "stagger"] < 2
