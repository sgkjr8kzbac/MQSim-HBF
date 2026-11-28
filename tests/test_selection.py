"""Selection realism and compacted storage density (phase 3, T-K5).

Acceptance checks from docs/design/phase3-kv-terms.md section 3 (K5) on ocp (1,024 banks, 4 KiB pages).
"""
import csv
import json
from fractions import Fraction

import numpy as np
import pytest

from hbf.frontend.kv.layouts import LAYOUTS
from hbf.frontend.kv.tensors import Shape, TensorSpec, parse_density, parse_tensors, tensors_meta
from hbf.frontend.trace.synthetic import generate
from hbf.frontend.trace.tokens import Access
from hbf.backends.rounds.analysis import analyze_accesses
from hbf.cli import main
from hbf.device.config import load
from hbf.frontend.kv.formats import parse_format
from hbf.frontend.pages import iter_layer_pages, schedule
from tests.reference import analyze


def test_density_validation_and_meta():
    assert parse_density(1) == 1 and type(parse_density(1)) is int
    assert parse_density(Fraction(1, 1)) == 1 and type(parse_density(Fraction(1, 1))) is int
    assert parse_density("1/4") == Fraction(1, 4)
    assert parse_density(0.25) == Fraction(1, 4)
    assert parse_density(0.1) == Fraction(1, 10)
    assert parse_density(1 / 3) == Fraction(1, 3)
    for bad in (0, -0.5, 1.5, "0", "3/2", "1/0", "abc", True, False, float("nan"), float("inf"),
                Fraction(1, 10_000_000), "1/10000001"):
        with pytest.raises(ValueError, match="density"):
            parse_density(bad)
    cfg = load("ocp")
    plain = LAYOUTS["layer_contiguous"](cfg, Shape(1, 1, 128, 64))
    assert "density" not in tensors_meta(plain)[0]
    compacted = LAYOUTS["layer_contiguous"](
        cfg, Shape(1, 1, 128, 64),
        tensors=(TensorSpec("K", 16, 128, density="1/4"), TensorSpec("V", 16, 128, density=Fraction(1, 4))))
    meta = tensors_meta(compacted)
    assert meta[0]["density"] == "1/4" and meta[1]["density"] == "1/4"
    assert parse_tensors(json.dumps(meta)) == compacted.all_tensors
    with pytest.raises(ValueError, match="density"):
        LAYOUTS["vllm_interleaved"](cfg, Shape(1, 1, 128, 1024), tensors=(
            TensorSpec("K", 16, 128, density=Fraction(1, 4)), TensorSpec("V", 16, 128)))


def test_density_quarter_at_32k_bf16():
    cfg = load("ocp")
    tensors = (TensorSpec("K", 16, 128, density=Fraction(1, 4)),
               TensorSpec("V", 16, 128, density=Fraction(1, 4)))
    assert tensors[0].blocks(32768, cfg.page_bytes) == 512
    layout = LAYOUTS["layer_contiguous"](cfg, Shape(1, 8, 128, 32768), tensors=tensors)
    assert layout.block_counts == (512, 512)
    (pages,) = iter_layer_pages(cfg, layout, generate(cfg, layout, "dense"))
    assert len(pages) == 8 * 2 * 512  # 512 pages per (head, tensor)
    assert int(pages.useful.sum()) == 8 * 2 * 8192 * 256
    assert pages.tensor_tokens == {"K": 8 * 8192, "V": 8 * 8192}
    assert pages.attended_tokens == 8 * 8192
    assert pages.attended_elements == 8 * 8192 * 256
    assert pages.hbf_elements == 8 * 8192 * 256
    # Float 1/3 at ctx 32,768 must not overflow int64: 10,923 compacted tokens -> ceil(10,923 / 16) = 683 pages.
    third = (TensorSpec("K", 16, 128, density=1 / 3), TensorSpec("V", 16, 128, density=1 / 3))
    lay_third = LAYOUTS["layer_contiguous"](cfg, Shape(1, 1, 128, 32768), tensors=third)
    (p_third,) = iter_layer_pages(cfg, lay_third, generate(cfg, lay_third, "dense"))
    assert lay_third.block_counts == (683, 683) and len(p_third) == 2 * 683
    assert p_third.tensor_tokens == {"K": 10923, "V": 10923}


def test_overlapping_cached_and_hbf_reads_do_not_double_count():
    cfg = load("ocp")
    # Density 1: partial overlap [0, 16) R + [8, 24) H -> 16 HBF tokens + 8 HBM-cached tokens = 24 total.
    plain = LAYOUTS["layer_contiguous"](cfg, Shape(1, 1, 128, 64))
    (p1,) = iter_layer_pages(cfg, plain, [Access(0, 0, 0, "K", "R", 0, 16, 16),
                                          Access(0, 0, 0, "K", "H", 8, 24, 16)])
    assert p1.tensor_tokens["K"] == 24 and p1.hbf_elements == 16 * 128 and p1.hbm_bytes == 8 * 256
    assert p1.attended_elements == 24 * 128
    # Density 1/4: [0, 2) R and [2, 4) H alias to the same compacted token [0, 1) -> 1 HBF token, 0 HBM bytes.
    compacted = LAYOUTS["layer_contiguous"](
        cfg, Shape(1, 1, 128, 64),
        tensors=(TensorSpec("K", 16, 128, density=Fraction(1, 4)), TensorSpec("V", 16, 128, density=Fraction(1, 4))))
    (p2,) = iter_layer_pages(cfg, compacted, [Access(0, 0, 0, "K", "R", 0, 2, 16),
                                              Access(0, 0, 0, "K", "H", 2, 4, 16)])
    assert p2.tensor_tokens["K"] == 1 and p2.hbf_elements == 128 and p2.hbm_bytes == 0
    assert p2.attended_elements == 128


def test_density_non_divisible_and_mixed_tiers():
    cfg = load("ocp")
    # Small/non-divisible windows must never produce 0 blocks or out-of-range blocks.
    for context, rho in ((1, "1/4"), (5, "1/4"), (65, "1/4"), (100, "2/3")):
        tensors = (TensorSpec("K", 16, 128, density=rho), TensorSpec("V", 16, 128, density=rho))
        layout = LAYOUTS["layer_contiguous"](cfg, Shape(1, 2, 128, context), tensors=tensors)
        for mode in ("dense", "uniform_tokens", "uniform_pages"):
            (pages,) = iter_layer_pages(cfg, layout, generate(cfg, layout, mode, fraction=0.5, seed=2))
            assert len(pages) > 0
    # KIVI2 with compacted HBF body (density 1/4) and full HBM residual (density 1).
    base = parse_format("kivi2", 128).tensors
    compacted = tuple(TensorSpec(s.name, s.bits, s.dims, s.window, s.tier, s.role, s.dequant_ops,
                                 density=Fraction(1, 4) if s.tier == "hbf" else 1) for s in base)
    layout = LAYOUTS["layer_contiguous"](cfg, Shape(1, 1, 128, 1024), tensors=compacted)
    # Body window [0, 896): compacted tokens = 224. K/V (128/page) -> 2 pages; K_qp/V_qp (256/page) -> 1 page.
    assert layout.block_counts == (2, 1, 2, 1)
    (pages,) = iter_layer_pages(cfg, layout, generate(cfg, layout, "dense"))
    assert len(pages) == 6
    assert pages.tensor_tokens["K"] == 224 and pages.tensor_tokens["K_res"] == 128
    assert pages.attended_tokens == 224 + 128
    assert pages.hbm_bytes == 2 * 128 * 256
    assert pages.hbf_elements == 2 * 224 * 128
    assert pages.attended_elements == 2 * (224 + 128) * 128
    assert pages.dequant_ops == 2 * 224 * 128 * 3


def test_schedule_rejects_density_less_than_one():
    cfg = load("ocp")
    tensors = (TensorSpec("K", 16, 128, density=Fraction(1, 4)), TensorSpec("V", 16, 128))
    layout = LAYOUTS["layer_contiguous"](cfg, Shape(1, 2, 128, 1024), tensors=tensors)
    accesses = list(generate(cfg, layout, "dense"))
    with pytest.raises(ValueError, match="density"):
        schedule(cfg, layout, accesses)


@pytest.mark.parametrize("mode", ["uniform_tokens", "uniform_pages"])
def test_select_static_keeps_exact_page_set(mode):
    cfg = load("ocp")
    layout = LAYOUTS["layer_contiguous"](cfg, Shape(2, 4, 128, 4096))
    static = list(iter_layer_pages(cfg, layout, generate(cfg, layout, mode, steps=3, fraction=0.1, seed=7,
                                                         select_static=True)))
    by_step = {(p.step, p.layer): p for p in static}
    for layer in range(2):
        for step in (1, 2):
            assert np.array_equal(by_step[step, layer].lpa, by_step[0, layer].lpa)
            assert np.array_equal(by_step[step, layer].useful, by_step[0, layer].useful)
            assert np.array_equal(by_step[step, layer].burst, by_step[0, layer].burst)
            assert np.array_equal(by_step[step, layer].runs, by_step[0, layer].runs)
    dynamic = list(iter_layer_pages(cfg, layout, generate(cfg, layout, mode, steps=2, fraction=0.1, seed=7)))
    assert not np.array_equal(dynamic[0].lpa, dynamic[2].lpa)


@pytest.mark.parametrize("mode,select_static", [("uniform_tokens", False), ("uniform_tokens", True),
                                                ("uniform_pages", False), ("uniform_pages", True)])
def test_temporal_reuse_moves_repeated_blocks_to_hbm(mode, select_static):
    cfg = load("ocp")
    layout = LAYOUTS["layer_contiguous"](cfg, Shape(1, 8, 128, 8192))
    base = list(iter_layer_pages(cfg, layout, generate(cfg, layout, mode, steps=2, fraction=0.1, seed=5,
                                                       select_static=select_static)))
    reused = list(iter_layer_pages(cfg, layout, generate(cfg, layout, mode, steps=2, fraction=0.1, seed=5,
                                                         select_static=select_static, reuse=0.8)))
    p0, p1 = reused
    # Step 0 is unaffected by reuse (cold cache).
    assert np.array_equal(p0.lpa, base[0].lpa) and p0.hbm_bytes == 0
    # Step 1 reads ~20% as many HBF pages as step 0, and moves the ~80% reused bytes to HBM.
    assert len(p1) / len(p0) == pytest.approx(0.2, abs=0.02)
    assert p1.hbm_bytes == int(p0.useful.sum()) - int(p1.useful.sum())
    assert p1.attended_elements == p0.attended_elements
    assert p1.hbf_elements / p0.hbf_elements == pytest.approx(0.2, abs=0.02)
    if mode == "uniform_tokens":
        assert p1.attended_tokens == p0.attended_tokens


def test_reuse_full_and_engine_parity():
    cfg = load("ocp")
    layout = LAYOUTS["striped_contiguous"](cfg, Shape(1, 4, 128, 4096))
    # reuse=1.0: step 1 has 0 HBF pages and serves all bytes from HBM.
    p0, p1 = list(iter_layer_pages(cfg, layout, generate(cfg, layout, "uniform_tokens", steps=2, fraction=0.1,
                                                         seed=3, reuse=1.0)))
    assert len(p1) == 0 and p1.hbm_bytes == int(p0.useful.sum()) and p1.hbf_elements == 0
    assert p1.attended_tokens == p0.attended_tokens
    # Reference and numpy engines agree when reuse > 0 (op="H" skipped by both for HBF pages).
    accesses = list(generate(cfg, layout, "uniform_tokens", steps=2, fraction=0.1, seed=3, reuse=0.8))
    assert analyze_accesses(cfg, layout, accesses, (1, 16), timing=False) == analyze(
        cfg, schedule(cfg, layout, accesses), (1, 16))


@pytest.mark.parametrize("mode", ["uniform_tokens", "uniform_pages"])
def test_query_groups_gqa_union(mode):
    cfg = load("ocp")
    context = 32768
    layout = LAYOUTS["layer_contiguous"](cfg, Shape(1, 8, 128, context))
    (pages,) = iter_layer_pages(cfg, layout, generate(cfg, layout, mode, fraction=0.05, seed=3, query_groups=4))
    expected_share = 1 - 0.95 ** 4  # 0.18549375
    dense_pages = 8 * 2 * (context // 16)
    assert len(pages) / dense_pages == pytest.approx(expected_share, abs=0.001)
    if mode == "uniform_tokens":
        assert pages.attended_tokens / (8 * context) == pytest.approx(expected_share, abs=0.001)


def test_eviction_live_pages_random_vs_clustered():
    cfg = load("ocp")
    context, heads = 32768, 8
    # BF16 random token eviction (keep 25%, select_tokens=1): 1 - 0.75^16 = 0.98998 of dense pages.
    bf16 = LAYOUTS["layer_contiguous"](cfg, Shape(1, heads, 128, context))
    dense_bf16 = heads * 2 * (context // 16)
    (scattered,) = iter_layer_pages(cfg, bf16, generate(cfg, bf16, "uniform_tokens", fraction=0.25,
                                                        select_tokens=1, seed=11))
    assert len(scattered) / dense_bf16 == pytest.approx(1 - 0.75 ** 16, abs=0.005)
    # BF16 aligned 16-token clusters: exactly 0.25 of dense pages.
    (clustered,) = iter_layer_pages(cfg, bf16, generate(cfg, bf16, "uniform_tokens", fraction=0.25,
                                                        select_tokens=16, seed=11))
    assert len(clustered) / dense_bf16 == pytest.approx(0.25, abs=1e-9)
    # KIVI-2 random token eviction (tpp=128 on K/V, 256 on K_qp/V_qp): ~1.0 of dense HBF pages.
    kivi = LAYOUTS["layer_contiguous"](cfg, Shape(1, heads, 128, context), tensors=parse_format("kivi2", 128).tensors)
    (dense_kivi,) = iter_layer_pages(cfg, kivi, generate(cfg, kivi, "dense"))
    (scattered_kivi,) = iter_layer_pages(cfg, kivi, generate(cfg, kivi, "uniform_tokens", fraction=0.25,
                                                             select_tokens=1, seed=11))
    assert len(scattered_kivi) / len(dense_kivi) == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("mode", ["uniform_tokens", "uniform_pages", "sink_recent"])
def test_full_tensors_reads_specified_tensor_whole(mode):
    cfg = load("ocp")
    context, heads = 4096, 4
    layout = LAYOUTS["layer_contiguous"](cfg, Shape(1, heads, 128, context))
    (pages,) = iter_layer_pages(cfg, layout, generate(cfg, layout, mode, fraction=0.1, seed=4, full_tensors=("K",)))
    assert pages.tensor_tokens["K"] == heads * context
    assert pages.tensor_tokens["V"] < heads * context // 2
    with pytest.raises(ValueError, match="full_tensors"):
        list(generate(cfg, layout, mode, full_tensors=("missing",)))


def test_generator_parameter_validation():
    cfg = load("ocp")
    layout = LAYOUTS["layer_contiguous"](cfg, Shape(1, 1, 128, 256))
    for kwargs in ({"select_static": 1}, {"reuse": -0.1}, {"reuse": 1.1}, {"reuse": True},
                   {"query_groups": 0}, {"query_groups": 1.5}):
        with pytest.raises(ValueError, match="Invalid generator parameters"):
            list(generate(cfg, layout, "uniform_tokens", **kwargs))
    for mode in ("dense", "sink_recent"):
        for kwargs in ({"select_static": True}, {"reuse": 0.5}, {"query_groups": 4}):
            with pytest.raises(ValueError, match="uniform_tokens, uniform_pages and channel_select"):
                list(generate(cfg, layout, mode, **kwargs))


HBF_FLAGS = ["run", "--topology", "ocp", "--read-plane-ns", "0", "--coalesce-ns", "1", "--cache-read", "--analysis-only"]


def test_cli_round_trip_and_default_meta_unchanged(tmp_path):
    # Default run leaves new fields out of meta.json.
    plain = tmp_path / "plain"
    main(HBF_FLAGS + ["--layers", "1", "--heads", "2", "--context", "1024", "--output", str(plain)])
    plain_meta = json.loads((plain / "meta.json").read_text())
    assert set(plain_meta["tensors"][0]) == {"name", "bits", "dims", "window", "tier", "role", "dequant_ops"}
    for key in ("select_static", "reuse", "query_groups"):
        assert key not in plain_meta["generator"]

    # --density 1/4 at ctx 32,768 -> 512 pages per (head, tensor).
    dense_q = tmp_path / "density"
    main(HBF_FLAGS + ["--layers", "1", "--heads", "8", "--context", "32768", "--density", "1/4",
                      "--output", str(dense_q)])
    (row,) = csv.DictReader((dense_q / "analysis_layers.csv").open())
    assert int(row["pages"]) == 8 * 2 * 512
    assert json.loads((dense_q / "meta.json").read_text())["tensors"][0]["density"] == "1/4"

    # --select-static + --reuse 0.8 + --query-groups 4, plus access.csv round-trip.
    sel = tmp_path / "sel"
    main(HBF_FLAGS + ["--mode", "uniform_tokens", "--layers", "1", "--heads", "8", "--context", "8192",
                      "--steps", "2", "--fraction", "0.05", "--select-static", "--reuse", "0.8",
                      "--query-groups", "4", "--output", str(sel)])
    steps = list(csv.DictReader((sel / "analysis_steps.csv").open()))
    assert len(steps) == 2
    assert int(steps[1]["pages"]) / int(steps[0]["pages"]) == pytest.approx(0.2, abs=0.02)
    assert int(steps[1]["hbm_bytes"]) > 0
    sel_meta = json.loads((sel / "meta.json").read_text())
    assert sel_meta["generator"]["select_static"] is True
    assert sel_meta["generator"]["reuse"] == 0.8
    assert sel_meta["generator"]["query_groups"] == 4
    # Re-importing access.csv reproduces analysis_steps.csv identically.
    reimp = tmp_path / "reimp"
    main(HBF_FLAGS + ["--layers", "1", "--heads", "8", "--context", "8192",
                      "--access-csv", str(sel / "access.csv"), "--output", str(reimp)])
    assert (reimp / "analysis_steps.csv").read_text() == (sel / "analysis_steps.csv").read_text()

    # --query-groups auto with --model resolves to q_heads // kv_heads (32 // 8 = 4 for llama-3.1-8b).
    auto_g = tmp_path / "auto_g"
    main(HBF_FLAGS + ["--model", "llama-3.1-8b", "--mode", "uniform_tokens", "--context", "1024",
                      "--fraction", "0.05", "--select-static", "--overlap", "prefetch:1",
                      "--query-groups", "auto", "--output", str(auto_g)])
    auto_meta = json.loads((auto_g / "meta.json").read_text())
    assert auto_meta["generator"]["query_groups"] == 4
    assert auto_meta["system"]["selection"] == "static"

    # Invalid CLI combinations fail cleanly without creating an output directory.
    write_csv = tmp_path / "writes.csv"
    write_csv.write_text("step,layer,kv_head,kind,op,token_start,token_end,bits\n-1,0,0,K,W,0,16,16\n")
    for idx, extra in enumerate((
        ["--density", "1/4", "--engine", "python"],
        ["--density", "0"],
        ["--reuse", "1.5"],
        ["--reuse", "nan"],
        ["--mode", "dense", "--reuse", "0.5"],
        ["--mode", "sink_recent", "--query-groups", "4"],
        ["--mode", "dense", "--select-static"],
        ["--query-groups", "0"],
        ["--query-groups", "auto"],
        ["--density", "1/4", "--check-writes"],
        ["--density", "1/4", "--mode", "write_then_read"],
        ["--density", "1/4", "--access-csv", str(write_csv)],
        ["--density", "1/4", "--kv-spec", "cla:share=2"],
        ["--mode", "uniform_tokens", "--select-static", "--kv-spec", "cla:share=2"],
    )):
        out = tmp_path / f"bad-{idx}"
        with pytest.raises(SystemExit):
            main(HBF_FLAGS + ["--layers", "2", "--context", "1024"] + extra + ["--output", str(out)])
        assert not out.exists()


# ---------------------------------------------------------------------------------------------
# T-K6 partial-dimension reads: loki, sparq[:dual], full_tensors, channel_select.

def test_loki_channel_select_at_32k():
    cfg = load("ocp")
    context = 32768
    fmt = parse_format("loki:rank=32", 128)
    layout = LAYOUTS["layer_contiguous"](cfg, Shape(1, 1, 128, context), tensors=fmt.tensors)
    # Works both with explicit full_tensors and with auto-detected K_lo.
    acc_auto = list(generate(cfg, layout, "channel_select", fraction=0.1, select_tokens=16, seed=3))
    acc_explicit = list(generate(cfg, layout, "channel_select", fraction=0.1, select_tokens=16, seed=3,
                                 full_tensors=fmt.full_tensors))
    assert acc_auto == acc_explicit
    (pages,) = iter_layer_pages(cfg, layout, acc_auto)
    k_lo_pages = int((pages.tensor == layout.index["K_lo"]).sum())
    k_hi_pages = int((pages.tensor == layout.index["K_hi"]).sum())
    v_pages = int((pages.tensor == layout.index["V"]).sum())
    # K_lo is read in full (512 pages vs 2,048 for full K); K_hi and V are read sparsely (~10%).
    assert k_lo_pages == 512
    selected_blocks = -(-int(context // 16) * 1 // 10)  # ceil(2048 * 0.1) = 205
    selected_tokens = 205 * 16
    assert v_pages == 205 and 205 <= k_hi_pages < 450
    assert pages.tensor_tokens == {"K_lo": context, "K_hi": selected_tokens, "V": selected_tokens}
    # No double-counting of K elements across scoring (K_lo) and exact (K_hi + V) phases.
    expected_elems = context * 32 + selected_tokens * 96 + selected_tokens * 128
    assert pages.attended_elements == expected_elems and pages.hbf_elements == expected_elems


def test_sparq_groups_touched_and_scoring_pages():
    cfg = load("ocp")
    # K6: sparq:channels=32,groups=8 -> mean groups touched = 8 * (1 - C(112,32)/C(128,32)) = 7.94 +- 0.1.
    fmt8 = parse_format("sparq:channels=32,groups=8", 128)
    layout8 = LAYOUTS["layer_contiguous"](cfg, Shape(4, 8, 128, 256), tensors=fmt8.tensors)
    acc8 = list(generate(cfg, layout8, "channel_select", steps=20, fraction=0.1, seed=7,
                         channels=fmt8.channel_select_channels, channel_groups=fmt8.channel_groups))
    touched_per_draw = {}
    for a in acc8:
        if a.kind.startswith("K_g") and a.token_start == 0 and a.token_end == 256:
            touched_per_draw.setdefault((a.step, a.layer, a.kv_head), set()).add(a.kind)
    assert len(touched_per_draw) == 20 * 4 * 8
    mean_touched = sum(len(s) for s in touched_per_draw.values()) / len(touched_per_draw)
    assert mean_touched == pytest.approx(7.94, abs=0.1)

    # K6: sparq:channels=32,groups=128 at ctx=32,768 -> exactly 32 * 16 = 512 scoring pages per head.
    context = 32768
    fmt128_dual = parse_format("sparq:channels=32,groups=128,dual=1", 128)
    lay_dual = LAYOUTS["layer_contiguous"](cfg, Shape(1, 1, 128, context), tensors=fmt128_dual.tensors)
    (p_dual,) = iter_layer_pages(cfg, lay_dual, generate(cfg, lay_dual, "channel_select", fraction=0.1,
                                                         select_tokens=16, seed=5, channels=32))
    k_g_indices = {lay_dual.index[name] for name in fmt128_dual.channel_groups}
    scoring_pages = int(np.isin(p_dual.tensor, list(k_g_indices)).sum())
    k_pages = int((p_dual.tensor == lay_dual.index["K"]).sum())
    v_pages = int((p_dual.tensor == lay_dual.index["V"]).sum())
    assert scoring_pages == 32 * 16 == 512
    assert k_pages == 205 and v_pages == 205 and len(p_dual) == 512 + 205 + 205
    # dual=1 counts exact K and V once (no double-counting with aux K_g*).
    assert p_dual.attended_elements == 205 * 16 * 256 and p_dual.hbf_elements == 205 * 16 * 256

    # Without dual=1, untouched 96 K_g* tensors (tpp=2048) touch nearly all their 16 pages each in Phase 2.
    fmt128 = parse_format("sparq:channels=32,groups=128", 128)
    lay_nodual = LAYOUTS["layer_contiguous"](cfg, Shape(1, 1, 128, context), tensors=fmt128.tensors)
    (p_nodual,) = iter_layer_pages(cfg, lay_nodual, generate(cfg, lay_nodual, "channel_select", fraction=0.1,
                                                             select_tokens=16, seed=5, channels=32))
    assert len(p_nodual) > len(p_dual) + 1000
    expected_nodual = context * 32 + 205 * 16 * 96 + 205 * 16 * 128
    assert p_nodual.attended_elements == expected_nodual and p_nodual.hbf_elements == expected_nodual


@pytest.mark.parametrize("fmt_text", ["loki:rank=32", "sparq:channels=32,groups=8", "sparq:channels=32,groups=8,dual=1"])
def test_channel_select_realism_options_and_parity(fmt_text):
    cfg = load("ocp")
    fmt = parse_format(fmt_text, 128)
    layout = LAYOUTS["striped_contiguous"](cfg, Shape(2, 2, 128, 2048), tensors=fmt.tensors)
    extra = {"channels": fmt.channel_select_channels} if fmt.channel_groups else {}

    # select_static keeps identical page sets across steps.
    static = list(iter_layer_pages(cfg, layout, generate(cfg, layout, "channel_select", steps=2, fraction=0.1,
                                                         seed=4, select_static=True, **extra)))
    by_key = {(p.step, p.layer): p for p in static}
    for layer in range(2):
        assert np.array_equal(by_key[1, layer].lpa, by_key[0, layer].lpa)

    # reuse > 0 moves repeated Phase 2 blocks to HBM on step 1; engines agree bit-for-bit.
    acc = list(generate(cfg, layout, "channel_select", steps=2, fraction=0.1, seed=4, reuse=0.5,
                        query_groups=2, **extra))
    pages = list(iter_layer_pages(cfg, layout, acc))
    assert pages[0].hbm_bytes == 0 and pages[2].hbm_bytes > 0
    assert analyze_accesses(cfg, layout, acc, (1, 16), timing=False) == analyze(
        cfg, schedule(cfg, layout, acc), (1, 16))


def test_channel_select_validation():
    cfg = load("ocp")
    bf16 = LAYOUTS["layer_contiguous"](cfg, Shape(1, 1, 128, 256))
    with pytest.raises(ValueError, match="channel_select requires"):
        list(generate(cfg, bf16, "channel_select"))
    with pytest.raises(ValueError, match="channel_select only"):
        list(generate(cfg, bf16, "uniform_tokens", channels=32))
    with pytest.raises(ValueError, match="channel_select only"):
        list(generate(cfg, bf16, "dense", channel_groups=("K",)))
    sparq = LAYOUTS["layer_contiguous"](cfg, Shape(1, 1, 128, 256), tensors=parse_format("sparq", 128).tensors)
    for bad_ch in (None, 0, 129, 32.5):
        with pytest.raises(ValueError, match="channels"):
            list(generate(cfg, sparq, "channel_select", channels=bad_ch))
    with pytest.raises(ValueError, match="both channel_groups and full_tensors"):
        list(generate(cfg, sparq, "channel_select", channels=32, full_tensors=("K_g0",)))


def test_channel_select_cli_end_to_end(tmp_path):
    # Loki end-to-end at ctx=32,768.
    loki_out = tmp_path / "loki"
    main(HBF_FLAGS + ["--format", "loki:rank=32", "--mode", "channel_select",
                      "--layers", "1", "--heads", "1", "--context", "32768", "--fraction", "0.1",
                      "--output", str(loki_out)])
    loki_meta = json.loads((loki_out / "meta.json").read_text())
    assert loki_meta["format"]["name"] == "loki"
    assert loki_meta["generator"]["full_tensors"] == ["K_lo"]
    (loki_row,) = csv.DictReader((loki_out / "analysis_layers.csv").open())
    assert int(loki_row["pages"]) < 1200  # 512 K_lo + sparse K_hi & V (well below 4,096 dense)

    # SparQ end-to-end + access.csv round-trip.
    sparq_out = tmp_path / "sparq"
    main(HBF_FLAGS + ["--format", "sparq:channels=32,groups=8", "--mode", "channel_select",
                      "--layers", "1", "--heads", "2", "--context", "4096", "--fraction", "0.1",
                      "--output", str(sparq_out)])
    sparq_meta = json.loads((sparq_out / "meta.json").read_text())
    assert sparq_meta["generator"]["channels"] == 32
    assert len(sparq_meta["generator"]["channel_groups"]) == 8
    reimp = tmp_path / "sparq-reimp"
    main(HBF_FLAGS + ["--format", "sparq:channels=32,groups=8",
                      "--layers", "1", "--heads", "2", "--context", "4096",
                      "--access-csv", str(sparq_out / "access.csv"), "--output", str(reimp)])
    assert (reimp / "analysis_steps.csv").read_text() == (sparq_out / "analysis_steps.csv").read_text()

    # SparQ dual=1 with --model, --select-static, --reuse, --query-groups auto: 1.5x stored KV bytes.
    bf16_sys = tmp_path / "bf16-sys"
    dual_sys = tmp_path / "dual-sys"
    main(HBF_FLAGS + ["--model", "llama-3.1-8b", "--context", "1024", "--output", str(bf16_sys)])
    main(HBF_FLAGS + ["--model", "llama-3.1-8b", "--format", "sparq:channels=32,groups=8,dual=1",
                      "--mode", "channel_select", "--context", "1024", "--steps", "2", "--fraction", "0.1",
                      "--select-static", "--reuse", "0.5", "--query-groups", "auto", "--output", str(dual_sys)])
    bf16_kv = json.loads((bf16_sys / "meta.json").read_text())["system"]["kv_bytes_per_request"]
    dual_meta = json.loads((dual_sys / "meta.json").read_text())
    assert dual_meta["system"]["kv_bytes_per_request"] == bf16_kv * 3 // 2
    assert dual_meta["generator"]["query_groups"] == 4
    assert dual_meta["system"]["selection"] == "static"
    assert len(list(csv.DictReader((dual_sys / "system_steps.csv").open()))) == 2

    # Invalid CLI combinations fail cleanly without creating output directory.
    for idx, extra in enumerate((
        ["--mode", "channel_select"],
        ["--format", "loki:rank=32", "--mode", "channel_select", "--kv-spec", "cla:share=2"],
        ["--format", "loki:rank=32", "--mode", "channel_select", "--alloc", "packed"],
    )):
        out = tmp_path / f"k6-bad-{idx}"
        with pytest.raises(SystemExit):
            main(HBF_FLAGS + ["--layers", "2", "--context", "1024"] + extra + ["--output", str(out)])
        assert not out.exists()
