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
    assert rounds["packed", "row"] > 5 and rounds["packed", "stagger"] <= 2
    assert rounds["dedicated", "row"] > 5 and rounds["dedicated", "stagger"] <= 2


def test_short_kivi_prompt_and_partial_blocks_in_segments():
    # H5: kivi2 prompt <= residual (0 full HBF pages at admission) and prompt where only some tensors
    # have full pages (e.g. prompt=256 -> K has 1 full page, K_qp has 0); HBM residual clamped when
    # context < residual; partial trailing prompt blocks not wasted as padding.
    from hbf.formats import parse_format
    cfg = load("ocp")
    tensors = parse_format("kivi2", 128).tensors
    for allocation in ("packed", "dedicated"):
        layout = PageTableLayout(cfg, Shape(2, 4, 128, 1),
                                 batch(allocation, [(64, 0), (64, 256), (256, 0), (512, 0)]),
                                 tensors=tensors)
        assert check_write_order(cfg, layout.snapshot.written())["legal"]
        accesses = list(generate_batch(cfg, layout, "dense"))
        # Request 0 has context 64 < residual 128: K_res/V_res must be clamped to [0, 64).
        req0_hbm = [a for a in accesses if a.kv_head < 4 and a.kind in ("K_res", "V_res")]
        assert req0_hbm and all((a.token_start, a.token_end) == (0, 64) for a in req0_hbm)
        rows, _ = analyze_accesses(cfg, layout, accesses, timing=True)
        assert len(rows) == 2
        # Partial trailing prompt block (kivi2 at prompt=2048 -> K_qp has 7 full pages, not 8) wastes
        # zero padding under layer_contiguous with align="row".
        exact = PageTableLayout(cfg, Shape(1, 2, 128, 1),
                                batch(allocation, [(2048, 0)], intra="layer_contiguous", align="row"),
                                tensors=tensors)
        assert exact.snapshot.kinds["padding"] == 0
        assert exact.snapshot.kinds["prefill"] == 2 * (15 + 7 + 15 + 7)


def test_partitioned_striped_intra_uses_partition_banks():
    # M2: channel (64 banks) and die (16 banks) partitions must stripe/align intra to partition banks,
    # not device-wide 1,024 banks (which caused gap=64 == 0 mod 64/16 and ~25% pure padding).
    cfg = load("ocp")
    for partition, banks in (("channel", 64), ("die", 16)):
        for allocation in ("packed", "dedicated"):
            layout = PageTableLayout(cfg, Shape(2, 8, 128, 1),
                                     batch(allocation, [(4096, 0)], partition=partition,
                                           intra="striped_contiguous", align="row"))
            assert check_write_order(cfg, layout.snapshot.written())["legal"]
            # Total prefill pages = 2 layers x 8 heads x 2 tensors x 256 blocks = 8,192.
            # Per-layer gap is banks // 16 (4 for channel, 1 for die), so padding is tiny (< 2%).
            prefill = layout.snapshot.kinds["prefill"]
            padding = layout.snapshot.kinds["padding"]
            assert prefill == 8192 and padding / prefill < 0.02
            # The 16 regions (8 heads x 2 tensors) of layer 0 start on distinct banks inside the partition.
            starts = layout.snapshot.table[0, 0, :, :, 0].ravel()
            assert len({cfg.unit(int(lpa)) for lpa in starts}) == min(16, banks)


def test_k7_hybrid_swa_class_frontiers_and_stranded_pages():
    # K7 acceptance test: hybrid LLLLLG (5 local with window=1024 on HBF + 1 global), 8 heads, BF16,
    # 1 request, prompt=32,768 + 4,096 generated -> dead local pages = 5 * 8 * 2 * (4096 // 16) = 20,480.
    from hbf.kvspec import parse_kvspec
    kv_spec = parse_kvspec("hybrid:pattern=LLLLLG,window=1024,local_tier=hbf", 6, 8, 32768 + 4096)
    shape = Shape(6, 8, 128, 1)

    # With ppb=8 (segment = 8,192 pages): single frontier mixes local and global pages in every segment
    # (all 20,480 dead pages stranded); class frontiers isolate short-lived local pages so at most one
    # open short-lived segment (4,096 < 8,192) is stranded.
    cfg8 = load("ocp", pages_per_block=8)
    for allocation in ("packed", "dedicated"):
        single = PageTableLayout(cfg8, shape, batch(allocation, [(32768, 4096)], intra="layer_contiguous",
                                                    frontiers="single", kv_spec=kv_spec))
        classed = PageTableLayout(cfg8, shape, batch(allocation, [(32768, 4096)], intra="layer_contiguous",
                                                     frontiers="class", kv_spec=kv_spec))
        s_chk = check_write_order(cfg8, single.snapshot)
        c_chk = check_write_order(cfg8, classed.snapshot)
        assert s_chk["legal"] and c_chk["legal"]
        assert single.snapshot.dead_pages == 20480 and single.snapshot.stranded_pages == 20480
        assert s_chk["dead_pages"] == 20480 and s_chk["stranded_pages"] == 20480
        assert classed.snapshot.dead_pages == 20480
        assert classed.snapshot.stranded_pages == 4096 < 8 * cfg8.units
        assert classed.snapshot.reclaimable_segments == 2
        assert c_chk["dead_pages"] == 20480 and c_chk["stranded_pages"] == 4096

    # Small ppb where 20,480 is an exact multiple of segment size -> 0 stranded under class frontiers.
    for ppb in (2, 4):
        cfg_small = load("ocp", pages_per_block=ppb)
        seg_size = ppb * cfg_small.units
        classed = PageTableLayout(cfg_small, shape, batch("packed", [(32768, 4096)], intra="layer_contiguous",
                                                          frontiers="class", kv_spec=kv_spec))
        assert classed.snapshot.dead_pages == 20480 and classed.snapshot.stranded_pages == 0
        assert classed.snapshot.reclaimable_segments == 20480 // seg_size
        assert check_write_order(cfg_small, classed.snapshot)["legal"]

    # Channel partition (64 banks) at realistic ppb=256 -> segment = 16,384 pages.
    cfg_ch = load("ocp", pages_per_block=256)
    ch_single = PageTableLayout(cfg_ch, shape, batch("packed", [(32768, 4096)], partition="channel",
                                                     intra="layer_contiguous", frontiers="single", kv_spec=kv_spec))
    ch_class = PageTableLayout(cfg_ch, shape, batch("packed", [(32768, 4096)], partition="channel",
                                                    intra="layer_contiguous", frontiers="class", kv_spec=kv_spec))
    assert ch_single.snapshot.dead_pages == 20480 and ch_single.snapshot.stranded_pages == 20480
    assert ch_class.snapshot.dead_pages == 20480 and ch_class.snapshot.stranded_pages == 4096 < 64 * 256
    assert ch_class.snapshot.reclaimable_segments == 1

    # Read path at snapshot (ctx=36,864): 5 local layers read 1,024 pages each, global layer reads 36,864.
    rows, _ = analyze_accesses(cfg8, classed, list(generate_batch(cfg8, classed, "dense")), timing=True)
    assert [row["pages"] for row in rows] == [1024, 1024, 1024, 1024, 1024, 36864]
    assert classed.request_bytes(cfg8)["kv_bytes_per_request"] == (5 * 1024 + 36864) * cfg8.page_bytes


def test_k7_shared_prefix_aliasing_and_reads():
    # K7 acceptance test: 4 requests sharing an 8,192-token prefix (1 layer, 8 heads, BF16):
    # 8 * 2 * (8192 / 16) = 8,192 prefix pages written once instead of 32,768, and read once per step
    # when coalesced.
    cfg = load("ocp")
    shape = Shape(1, 8, 128, 1)
    for allocation in ("packed", "dedicated", "layer_major", "vllm_pool"):
        spec = batch(allocation, [(8192, 0)] * 4, intra="layer_contiguous", align="row",
                     prefix_tokens=8192, prefix_groups=1, prefix_reads="coalesced")
        layout = PageTableLayout(cfg, shape, spec)
        assert layout.snapshot.prefix_pages == 8192
        assert layout.snapshot.kinds["prefill"] == 8192
        for r in range(1, 4):
            assert np.array_equal(layout.snapshot.table[r], layout.snapshot.table[0])
        (row,), (step_row,) = analyze_accesses(cfg, layout, list(generate_batch(cfg, layout, "dense")), timing=True)
        assert row["pages"] == 8192 and row["rounds"] == 8
        assert row["attended_tokens"] == 4 * 8 * 8192
        assert step_row["max_block_reads"] == 8
        if allocation in ("packed", "dedicated"):
            assert check_write_order(cfg, layout.snapshot)["legal"]

    # per_request mode: prefix still written ONCE (8,192), but read 4x per step (32,768 pages, 32 rounds)
    # on the same banks.
    per_req = PageTableLayout(cfg, shape, batch("packed", [(8192, 0)] * 4, intra="layer_contiguous", align="row",
                                                prefix_tokens=8192, prefix_groups=1, prefix_reads="per_request"))
    assert per_req.snapshot.prefix_pages == 8192 and per_req.snapshot.kinds["prefill"] == 8192
    assert check_write_order(cfg, per_req.snapshot)["legal"]
    (row_pr,), (step_pr,) = analyze_accesses(cfg, per_req, list(generate_batch(cfg, per_req, "dense")), timing=True)
    assert row_pr["pages"] == 32768 and row_pr["rounds"] == 32
    assert row_pr["attended_tokens"] == 4 * 8 * 8192
    assert step_pr["max_block_reads"] == 32
    # Decode works for virtual per-request LPAs as well.
    lpa_r2 = per_req.lpa(0, 2 * 8 + 3, 0, 5)
    assert per_req.decode(lpa_r2) == (0, 2 * 8 + 3, 0, 5)

    # 2 prefix groups across 4 requests -> 2 * 8,192 = 16,384 prefix pages written & coalesced-read.
    two_grp = PageTableLayout(cfg, shape, batch("packed", [(8192, 0)] * 4, intra="layer_contiguous", align="row",
                                                prefix_tokens=8192, prefix_groups=2))
    assert two_grp.snapshot.prefix_pages == 16384 and two_grp.snapshot.kinds["prefill"] == 16384
    assert np.array_equal(two_grp.snapshot.table[0], two_grp.snapshot.table[2])
    assert np.array_equal(two_grp.snapshot.table[1], two_grp.snapshot.table[3])
    assert not np.array_equal(two_grp.snapshot.table[0], two_grp.snapshot.table[1])
    (row_g2,), _ = analyze_accesses(cfg, two_grp, list(generate_batch(cfg, two_grp, "dense")))
    assert row_g2["pages"] == 16384

    # Prefix + suffix + decode + unequal prompts across allocations.
    for allocation in ("packed", "dedicated", "vllm_pool", "layer_major"):
        mixed = PageTableLayout(cfg, shape, batch(allocation, [(4096, 256), (10240, 128)],
                                                  intra="layer_contiguous", align="row",
                                                  prefix_tokens=8192, prefix_groups=1))
        # Group prefix is 8,192 (max shared across members): written once (8,192 pages).
        # r0 shares 4,096 (0 suffix), r1 shares 8,192 (2,048 suffix -> 2,048 pages).
        assert mixed.snapshot.prefix_pages == 8192
        assert mixed.snapshot.kinds["prefill"] == 8192 + 2048
        assert mixed.snapshot.kinds["decode"] == 8 * 2 * (256 // 16 + 128 // 16)
        # r0's generated blocks [256..271] must NOT collide with r1's prefix blocks [256..511].
        r0_gen = set(mixed.snapshot.table[0, 0, :, :, 256:272].ravel().tolist())
        r1_pref = set(mixed.snapshot.table[1, 0, :, :, 256:512].ravel().tolist())
        assert r0_gen.isdisjoint(r1_pref)
        if allocation in ("packed", "dedicated"):
            assert check_write_order(cfg, mixed.snapshot)["legal"]


def test_k7_sliding_age_ladder_and_grouped_presets():
    from hbf.access import TensorSpec
    from hbf.kvspec import parse_kvspec

    # Age-based re-quantization ladder without kv_spec: int4 (-4096, -128) is short-lived, int2 (0, -4096) long-lived.
    cfg = load("ocp", pages_per_block=2)
    ladder = (
        TensorSpec("K_int4", 4, 128, window=(-4096, -128)),
        TensorSpec("V_int4", 4, 128, window=(-4096, -128)),
        TensorSpec("K_int2", 2, 128, window=(0, -4096)),
        TensorSpec("V_int2", 2, 128, window=(0, -4096)),
    )
    shape = Shape(4, 8, 128, 1)
    # prompt=8192, generated=4096: int4 (tpp=64) loses 4096/64 = 64 blocks * 4 layers * 8 heads * 2 tensors = 4,096 pages.
    s_lad = PageTableLayout(cfg, shape, batch("packed", [(8192, 4096)], intra="layer_contiguous", frontiers="single"),
                            tensors=ladder)
    c_lad = PageTableLayout(cfg, shape, batch("packed", [(8192, 4096)], intra="layer_contiguous", frontiers="class"),
                            tensors=ladder)
    assert s_lad.snapshot.dead_pages == 4096 and c_lad.snapshot.dead_pages == 4096
    assert c_lad.snapshot.stranded_pages < s_lad.snapshot.stranded_pages
    assert c_lad.snapshot.stranded_pages <= 2 * cfg.units
    assert check_write_order(cfg, c_lad.snapshot)["legal"]

    # DuoAttention (head split): 4 retrieval heads (long-lived) + 4 streaming heads (sink + sliding recent on HBF).
    duo = parse_kvspec("duo:retrieval=0.5,sink=64,recent=256,streaming_tier=hbf", 4, 8, 4096 + 1024)
    duo_layout = PageTableLayout(cfg, shape, batch("packed", [(4096, 1024)], intra="layer_contiguous",
                                                   frontiers="class", kv_spec=duo))
    # Dead recent pages: 4 layers * 4 streaming heads * 2 tensors * (1024 / 16) = 2,048.
    assert duo_layout.snapshot.dead_pages == 2048
    assert duo_layout.snapshot.stranded_pages == 0
    assert check_write_order(cfg, duo_layout.snapshot)["legal"]
    rows, _ = analyze_accesses(cfg, duo_layout, list(generate_batch(cfg, duo_layout, "dense")))
    # Per layer: 4 retrieval heads * 2 * (5120/16) + 4 sink * 2 * 4 + 4 recent * 2 * 16 = 2560 + 32 + 128 = 2720 pages.
    assert [row["pages"] for row in rows] == [2720] * 4

    # CLA (cross-layer aliasing, share=2, reuse=hbf): 4 layers, 2 owners -> writes half, reads all 4.
    cla = parse_kvspec("cla:share=2,reuse=hbf", 4, 8, 2048)
    cla_layout = PageTableLayout(cfg, shape, batch("packed", [(2048, 0)], intra="layer_contiguous", kv_spec=cla))
    assert cla_layout.snapshot.kinds["prefill"] == 2 * 8 * 2 * 128
    assert np.array_equal(cla_layout.snapshot.table[0, 1], cla_layout.snapshot.table[0, 0])
    assert np.array_equal(cla_layout.snapshot.table[0, 3], cla_layout.snapshot.table[0, 2])
    rows, _ = analyze_accesses(cfg, cla_layout, list(generate_batch(cfg, cla_layout, "dense")))
    assert [row["pages"] for row in rows] == [8 * 2 * 128] * 4

    # SSM hybrid: 3 SSM layers (HBM state only, 0 HBF pages) + 1 attention layer.
    ssm = parse_kvspec("ssm:pattern=SSSA,state=1048576", 4, 8, 1024)
    ssm_layout = PageTableLayout(cfg, shape, batch("packed", [(1024, 0)], intra="layer_contiguous", kv_spec=ssm))
    rows, _ = analyze_accesses(cfg, ssm_layout, list(generate_batch(cfg, ssm_layout, "dense")), timing=True)
    assert [row["pages"] for row in rows] == [0, 0, 0, 8 * 2 * 64]
    assert [row["hbm_bytes"] for row in rows] == [1048576, 1048576, 1048576, 0]


def test_k7_batch_spec_defaults_and_validation():
    from hbf.kvspec import parse_kvspec
    assert batch("layer_major", [(2048, 0)] * 2).default
    assert not batch("layer_major", [(2048, 0)] * 2, prefix_tokens=512).default
    assert not batch("packed", [(2048, 0)], frontiers="class").default
    kv = parse_kvspec("hybrid:pattern=LG,window=256,local_tier=hbf", 2, 8, 2048)
    assert not batch("layer_major", [(2048, 0)], kv_spec=kv).default

    for kwargs in (
        {"allocation": "packed", "frontiers": "invalid"},
        {"allocation": "layer_major", "frontiers": "class"},
        {"allocation": "vllm_pool", "frontiers": "class"},
        {"allocation": "packed", "prefix_tokens": -1},
        {"allocation": "packed", "prefix_groups": 0},
        {"allocation": "packed", "prefix_reads": "bad"},
    ):
        with pytest.raises(ValueError):
            BatchSpec((BatchRequest(1024, 0),), **kwargs)
