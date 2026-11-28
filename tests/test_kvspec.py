"""KV layer and head groups (hbf/kvspec.py, T-K2): hand-computed values of design section 3, K2."""
import csv
import json

import pytest

from hbf.frontend.kv.layouts import LAYOUTS
from hbf.frontend.kv.tensors import Shape, default_tensors
from hbf.frontend.trace.synthetic import generate
from hbf.frontend.trace.tokens import Access
from hbf.backends.rounds.analysis import analyze_accesses
from hbf.cli import main
from hbf.device.config import load
from hbf.kvspec import GroupedLayout, KVGroup, KVSpec, from_model, generate_grouped, parse_kvspec
from hbf.backends.rounds.pages import iter_layer_pages

HBF_FLAGS = ["run", "--analysis-only", "--topology", "ocp", "--read-plane-ns", "0", "--coalesce-ns", "1", "--cache-read"]


def grouped(text, shape, mode="dense", intra="contiguous", requests=1, **generation):
    cfg = load("ocp")
    spec = parse_kvspec(text, shape.layers, shape.kv_heads // requests, shape.context)
    layout = GroupedLayout(cfg, shape, spec, default_tensors(shape), requests=requests, intra=intra)
    accesses = list(generate_grouped(cfg, layout, mode, **generation))
    rows, steps = analyze_accesses(cfg, layout, accesses, timing=True)
    return layout, rows, steps


def test_hybrid_local_and_global_layers():
    layout, rows, steps = grouped("hybrid:pattern=LLLLLG,window=1024,local_tier=hbf", Shape(6, 8, 128, 32768))
    assert [row["pages"] for row in rows] == [1024] * 5 + [32768]
    assert [row["rounds"] for row in rows] == [1] * 5 + [32] and steps[0]["pages"] == 37_888
    assert layout.request_bytes(load("ocp"))["kv_bytes_per_request"] == 134_217_728 + 20_971_520
    layout, rows, _ = grouped("hybrid:local_tier=hbm", Shape(6, 8, 128, 32768))
    assert [row["pages"] for row in rows] == [0] * 5 + [32768] and len(rows) == 6
    assert [row["hbm_bytes"] for row in rows] == [4_194_304] * 5 + [0]
    assert layout.request_bytes(load("ocp"))["hbm_tier_bytes_per_request"] == 20_971_520


def test_cross_layer_sharing():
    layout, rows, steps = grouped("cla:share=2,reuse=hbf", Shape(4, 8, 128, 4096))
    assert [row["pages"] for row in rows] == [4096] * 4 and steps[0]["pages"] == 16_384 and layout.pages == 8192
    for t in range(2):
        assert all(layout.lpa(1, head, t, 5) == layout.lpa(0, head, t, 5) for head in range(8))
    cfg = load("ocp")
    plain = LAYOUTS["layer_contiguous"](cfg, Shape(2, 8, 128, 4096))
    _, (unshared,) = analyze_accesses(cfg, plain, list(generate(cfg, plain, "dense")), timing=True)
    assert steps[0]["max_block_reads"] == 2 * unshared["max_block_reads"]
    layout, rows, steps = grouped("cla:share=2,reuse=hbm", Shape(4, 8, 128, 4096))
    assert steps[0]["pages"] == 8192 and [row["hbm_bytes"] for row in rows] == [0, 16_777_216, 0, 16_777_216]
    # 2 owner layers stored (33,554,432 B); only 1 owner layer live in HBM staging at a time (16,777,216 B).
    assert layout.request_bytes(cfg) == {"kv_bytes_per_request": 33_554_432,
                                         "hbm_tier_bytes_per_request": 16_777_216}


def test_duo_heads():
    layout, rows, _ = grouped("duo:retrieval=0.5,sink=64,recent=256", Shape(1, 8, 128, 32768))
    (row,) = rows
    assert row["pages"] == 16_384 and row["hbm_bytes"] == 655_360 and row["attended_tokens"] == 132_352
    with pytest.raises(ValueError, match="Invalid logical access"):  # head 0 is a retrieval head
        Access(0, 0, 0, "recent_K", "R", 32512, 32768, 16).validate(layout)
    batched, rows2, _ = grouped("duo:retrieval=0.5", Shape(1, 16, 128, 32768), requests=2)
    assert rows2[0]["pages"] == 2 * 16_384 and batched.read_map["retrieval_K"][2] == (0, 1, 2, 3, 8, 9, 10, 11)


def test_pyramid_budgets_and_ssm_layers():
    _, rows, _ = grouped("pyramid:first=8192,last=2048", Shape(4, 8, 128, 8192))
    assert [row["pages"] for row in rows] == [8192, 6144, 4096, 2048]
    _, rows, _ = grouped("ssm:pattern=SSSA,state=1000000", Shape(4, 8, 128, 4096))
    assert [row["pages"] for row in rows] == [0, 0, 0, 4096] and [row["hbm_bytes"] for row in rows[:3]] == [10 ** 6] * 3
    layout, rows, _ = grouped("yoco:window=1024", Shape(4, 8, 128, 4096))
    assert [row["pages"] for row in rows] == [0, 0, 4096, 4096] and layout.pages == 4096


@pytest.mark.parametrize("name,intra", [("layer_contiguous", "contiguous"), ("striped_contiguous", "striped")])
def test_one_group_reproduces_the_plain_layout(name, intra):
    cfg = load("ocp")
    shape = Shape(3, 4, 128, 3000)
    plain = LAYOUTS[name](cfg, shape)
    spec = KVSpec("one", {}, (KVGroup("all", (0, 1, 2)),))
    layout = GroupedLayout(cfg, shape, spec, default_tensors(shape), intra=intra)
    assert layout.span == plain.span and layout.pages == plain.pages
    for layer in range(3):
        for t in range(2):
            for head in range(4):
                assert layout.lpa(layer, head, t, 7) == plain.lpa(layer, head, t, 7)
    for mode in ("dense", "uniform_tokens", "sink_recent"):
        a = analyze_accesses(cfg, plain, list(generate(cfg, plain, mode, fraction=.05, seed=4)), timing=True)
        b = analyze_accesses(cfg, layout, list(generate_grouped(cfg, layout, mode, fraction=.05, seed=4)), timing=True)
        assert a == b


def test_parameters_are_checked():
    for text in ("nope", "hybrid:pattern=LX", "cla:share=1", "duo:retrieval=2", "hybrid:local_tier=ssd",
                 "cla:reuse=x", "pyramid:first=0", "hybrid:x=1"):
        with pytest.raises(ValueError):
            parse_kvspec(text, 4, 8, 4096)
    with pytest.raises(ValueError, match="alias"):
        KVGroup("g", (0, 1), alias=((1, 2),))


def test_cli_kv_spec_with_the_system_model(tmp_path):
    model = {"name": "tiny6", "layers": 6, "hidden": 1024, "q_heads": 32, "kv_heads": 8, "head_dim": 128,
             "intermediate": 2048, "vocab": 1000}
    out = tmp_path / "h"
    main(HBF_FLAGS + ["--model", json.dumps(model), "--context", "4096", "--kv-spec", "hybrid:window=512",
                      "--output", str(out)])
    rows = list(csv.DictReader((out / "system_layers.csv").open()))
    assert len(rows) == 6
    meta = json.loads((out / "meta.json").read_text())
    assert meta["kv_spec"]["name"] == "hybrid" and [t["name"] for t in meta["tensors"]][:2] == ["local_K", "local_V"]
    local = 5 * 8 * 2 * 512 * 256
    assert meta["system"]["kv_bytes_per_request"] == 8 * 2 * 4096 * 256 + local
    assert meta["system"]["hbm_tier_bytes_per_request"] == local
    assert "局部层" in (out / "report.md").read_text()
    for extra in (["--layout", "vllm_interleaved"], ["--mode", "uniform_pages"]):
        with pytest.raises(SystemExit):
            main(HBF_FLAGS + ["--kv-spec", "cla", "--layers", "2", "--context", "1024", *extra,
                              "--output", str(tmp_path / "x")])


def test_short_context_clamps_group_windows():
    # H1: context < window must clamp instead of failing TensorSpec.tokens().
    for tier in ("hbf", "hbm"):
        layout, rows, _ = grouped(f"hybrid:pattern=LLLLLG,window=1024,local_tier={tier}", Shape(6, 8, 128, 512))
        assert len(rows) == 6
        assert [row["attended_tokens"] for row in rows] == [8 * 512] * 6
        if tier == "hbf":
            assert [row["pages"] for row in rows] == [512] * 6
        else:
            assert [row["pages"] for row in rows] == [0] * 5 + [512]
            assert [row["hbm_bytes"] for row in rows[:5]] == [8 * 2 * 512 * 256] * 5


def test_overlapping_alias_peak_live_owners_and_yoco_hbm_reuse():
    # H2: peak live owners for interleaved aliases ((2, 0), (3, 1)) is 2; for yoco it is 1;
    # kv_bytes_per_request excludes staging copies.
    cfg = load("ocp")
    shape = Shape(4, 8, 128, 4096)
    spec = KVSpec("interleaved", {}, (KVGroup("kv", (0, 1, 2, 3), alias=((2, 0), (3, 1)), reuse="hbm"),))
    layout = GroupedLayout(cfg, shape, spec, default_tensors(shape))
    assert layout.copy_owners == {"kv_K_copy": 2, "kv_V_copy": 2}
    layer_bytes = 8 * 2 * 4096 * 256
    assert layout.request_bytes(cfg) == {"kv_bytes_per_request": 2 * layer_bytes,
                                         "hbm_tier_bytes_per_request": 2 * layer_bytes}
    yoco, _, _ = grouped("yoco:window=1024,local_tier=hbm,reuse=hbm", shape)
    local_bytes = 2 * 8 * 2 * 1024 * 256
    assert yoco.copy_owners == {"global_K_copy": 1, "global_V_copy": 1}
    assert yoco.request_bytes(cfg) == {"kv_bytes_per_request": layer_bytes + local_bytes,
                                       "hbm_tier_bytes_per_request": local_bytes + layer_bytes}


def test_striped_grouped_layout_avoids_bank_collisions():
    # H4: 1,024-token local layers (16 entries x 64 blocks) and 3,072-token layers (16 x 192)
    # must not collide onto half or a quarter of the banks under intra="striped".
    _, rows, _ = grouped("hybrid:pattern=LLLLLG,window=1024,local_tier=hbf", Shape(6, 8, 128, 32768), intra="striped")
    assert [row["rounds"] for row in rows] == [1] * 5 + [32]
    assert [row["balance_loss"] for row in rows] == [pytest.approx(1.0)] * 6
    _, rows3k, _ = grouped("hybrid:pattern=LLLLLG,window=3072,local_tier=hbf", Shape(6, 8, 128, 32768), intra="striped")
    assert [row["rounds"] for row in rows3k] == [3] * 5 + [32]
    assert [row["balance_loss"] for row in rows3k] == [pytest.approx(1.0)] * 6


def test_duo_overlapping_sink_and_recent_does_not_double_count():
    # M1: when sink + recent >= context, streaming heads must count each token once.
    cfg = load("ocp")
    for context in (320, 100):
        layout, (row,), _ = grouped("duo:retrieval=0.5,sink=64,recent=256", Shape(1, 8, 128, context))
        assert row["attended_tokens"] == 8 * context
        assert row["hbm_bytes"] == 4 * 2 * context * 256
        assert layout.request_bytes(cfg)["hbm_tier_bytes_per_request"] == 4 * 2 * context * 256
        layout_hbf, (row_hbf,), _ = grouped("duo:retrieval=0.5,sink=64,recent=256,streaming_tier=hbf",
                                            Shape(1, 8, 128, context))
        assert row_hbf["attended_tokens"] == 8 * context
        assert row_hbf["pages"] == 8 * 2 * -(-context // 16)


def test_cli_pads_missing_sparse_layers_under_timing(tmp_path):
    # H3: sparse uniform_tokens on hybrid with local_tier=hbf can miss a local layer completely;
    # --timing and --model must pad zero rows per plane group so CSVs and system.evaluate succeed.
    model = {"name": "tiny6", "layers": 6, "hidden": 1024, "q_heads": 32, "kv_heads": 8, "head_dim": 128,
             "intermediate": 2048, "vocab": 1000}
    out = tmp_path / "sparse"
    main(HBF_FLAGS + ["--model", json.dumps(model), "--context", "32768", "--kv-spec",
                      "hybrid:pattern=LLLLLG,window=128,local_tier=hbf", "--mode", "uniform_tokens",
                      "--fraction", "0.001", "--seed", "1", "--plane-group", "1", "--plane-group", "16",
                      "--mocs", "256", "--output", str(out)])
    layers = list(csv.DictReader((out / "analysis_layers.csv").open()))
    assert len(layers) == 12  # 2 plane groups x 6 layers
    g1 = [row for row in layers if row["group"] == "1"]
    assert [int(row["layer"]) for row in g1] == list(range(6))
    zeros = [row for row in g1 if row["pages"] == "0"]
    assert zeros  # at least one local layer had no sampled tokens inside its 128-token window
    for row in zeros:
        assert (row["rounds"], row["est_ns"], row["bound_ns"], row["bottleneck"], row["commands"]) == (
            "0", "0", "0", "none", "0")
    steps = list(csv.DictReader((out / "analysis_steps.csv").open()))
    assert [row["layers"] for row in steps] == ["6", "6"]
    sys_layers = list(csv.DictReader((out / "system_layers.csv").open()))
    assert len(sys_layers) == 6


@pytest.mark.parametrize("spec_text", ["hybrid:pattern=LLLLLG,window=1024,local_tier=hbf",
                                       "cla:share=2,reuse=hbf", "duo:retrieval=0.5,sink=64,recent=256",
                                       "pyramid:first=4096,last=1024", "yoco:window=512,local_tier=hbf"])
@pytest.mark.parametrize("intra", ["contiguous", "striped"])
def test_grouped_layout_engine_parity(spec_text, intra):
    # L6: numpy and pure-Python reference engines must agree row-by-row on GroupedLayout.
    from hbf.backends.rounds.analysis import Protocol
    from hbf.backends.mqsim.schedule import schedule
    from tests.reference import analyze
    cfg = load("ocp")
    shape = Shape(6, 4, 128, 4096)
    spec = parse_kvspec(spec_text, shape.layers, shape.kv_heads, shape.context)
    layout = GroupedLayout(cfg, shape, spec, default_tensors(shape), intra=intra)
    protocol = Protocol(256, read_unit="wanted")
    for mode in ("dense", "uniform_tokens", "sink_recent"):
        accesses = list(generate_grouped(cfg, layout, mode, fraction=.1, seed=2))
        expected = analyze(cfg, schedule(cfg, layout, accesses), (1, 16), protocol=protocol)
        actual = analyze_accesses(cfg, layout, accesses, (1, 16), protocol=protocol, timing=False)
        assert actual == expected


# ---------------------------------------------------------------------------------------------
# T-K3 from_model: exact KV bytes/token and GroupedLayout integration for MLA, hybrid, SSM.

@pytest.mark.parametrize("name,expected_bpt", [
    ("deepseek-v2-lite", 27 * 1152),              # 31,104 B (27 MLA layers x 576 BF16 dims)
    ("qwen3-next-80b-a3b", 12 * 2 * 2 * 256 * 2), # 24,576 B (12 full-attn layers, 2 KV heads, d=256)
    ("gpt-oss-120b", 18 * 8 * 2 * 64 * 2),        # 36,864 B (18 global layers, 8 KV heads, d=64)
    ("gpt-oss-20b", 12 * 8 * 2 * 64 * 2),         # 24,576 B (12 global layers, 8 KV heads, d=64)
    ("gemma-3-27b", 10 * 16 * 2 * 128 * 2),       # 81,920 B (10 global layers, 16 KV heads, d=128)
    ("llama-3.1-8b", 32 * 8 * 2 * 128 * 2),       # 131,072 B (32 uniform layers)
])
def test_from_model_kv_bytes_per_token(name, expected_bpt):
    from hbf.models import load_model
    model = load_model(name)
    spec = from_model(model)
    assert spec.bytes_per_token(model.kv_heads) == expected_bpt
    assert from_model(name).bytes_per_token(model.kv_heads) == expected_bpt


def test_from_model_mla_formats_and_layout():
    from hbf.models import load_model
    cfg = load("ocp")
    ds = load_model("deepseek-v2-lite")
    bf16 = from_model(ds)
    fp8 = from_model(ds, "fp8")
    split = from_model(ds, "mla:split=1")
    idx = from_model(ds, "mla:indexer=1")
    assert bf16.bytes_per_token(ds.kv_heads) == 31_104
    assert fp8.bytes_per_token(ds.kv_heads) == 27 * 656
    assert split.bytes_per_token(ds.kv_heads) == 31_104
    assert idx.bytes_per_token(ds.kv_heads) == 27 * (1152 + 132)
    shape = Shape(ds.layers, ds.kv_heads, ds.head_dim, 3072)
    layout = GroupedLayout(cfg, shape, bf16, default_tensors(shape))
    rows, _ = analyze_accesses(cfg, layout, list(generate_grouped(cfg, layout, "dense")), timing=True)
    assert len(rows) == 27 and layout.tokens_per_page == (3,)
    assert all(row["pages"] == 1024 and row["attended_elements"] == 3072 * 1088 for row in rows)
    with pytest.raises(ValueError, match="MLA model"):
        from_model(ds, "kivi2")


def test_from_model_hybrid_default_hbm_and_hbf_variant():
    from hbf.models import load_model
    cfg = load("ocp")
    gpt = load_model("gpt-oss-120b")
    spec_hbm = from_model(gpt)  # default local_tier="hbm"
    spec_hbf = from_model(gpt, local_tier="hbf")
    assert spec_hbm.params["local_tier"] == "hbm" and spec_hbf.params["local_tier"] == "hbf"
    assert spec_hbm.bytes_per_token(gpt.kv_heads) == spec_hbf.bytes_per_token(gpt.kv_heads) == 36_864
    shape = Shape(gpt.layers, gpt.kv_heads, gpt.head_dim, 4096)
    lay_hbm = GroupedLayout(cfg, shape, spec_hbm, default_tensors(shape))
    rows_hbm, _ = analyze_accesses(cfg, lay_hbm, list(generate_grouped(cfg, lay_hbm, "dense")), timing=True)
    assert [row["pages"] for row in rows_hbm[:4]] == [0, 8 * 2 * 128, 0, 8 * 2 * 128]
    assert [row["hbm_bytes"] for row in rows_hbm[:4]] == [8 * 2 * 128 * 128, 0, 8 * 2 * 128 * 128, 0]
    lay_hbf = GroupedLayout(cfg, shape, spec_hbf, default_tensors(shape))
    rows_hbf, _ = analyze_accesses(cfg, lay_hbf, list(generate_grouped(cfg, lay_hbf, "dense")), timing=True)
    assert [row["pages"] for row in rows_hbf[:4]] == [8 * 2 * 4, 8 * 2 * 128, 8 * 2 * 4, 8 * 2 * 128]
    assert all(row["hbm_bytes"] == 0 for row in rows_hbf)
    # Gemma-3-27B: 52 local (W=1,024) + 10 global layers.
    gemma = load_model("gemma-3-27b")
    g_spec = from_model(gemma)
    assert len(g_spec.groups[0].layers) == 52 and len(g_spec.groups[1].layers) == 10
    assert g_spec.groups[0].tier == "hbm" and g_spec.groups[0].window == (-1024, None)
    with pytest.raises(ValueError, match="local_tier"):
        from_model(gpt, local_tier="ssd")


def test_from_model_ssm_emits_state_and_attention_groups():
    from hbf.models import load_model
    cfg = load("ocp")
    qn = load_model("qwen3-next-80b-a3b")
    spec = from_model(qn)
    assert spec.name == "ssm" and spec.bytes_per_token(qn.kv_heads) == 24_576
    shape = Shape(qn.layers, qn.kv_heads, qn.head_dim, 2048)
    layout = GroupedLayout(cfg, shape, spec, default_tensors(shape))
    rows, _ = analyze_accesses(cfg, layout, list(generate_grouped(cfg, layout, "dense")), timing=True)
    assert len(rows) == 48
    # Pattern SSSA: layers 0,1,2 are SSM (0 HBF pages, 2 MiB HBM state), layer 3 is full attention (2*2*256 pages).
    assert [row["pages"] for row in rows[:4]] == [0, 0, 0, 2 * 2 * 256]
    assert [row["hbm_bytes"] for row in rows[:4]] == [2_097_152, 2_097_152, 2_097_152, 0]
    assert [row["attended_elements"] for row in rows[:4]] == [0, 0, 0, 2 * 2048 * 512]


def test_cli_mla_format_and_hybrid_tensors_integration(tmp_path):
    from fractions import Fraction
    from hbf.frontend.kv.tensors import TensorSpec
    from hbf.kvspec import resolve_format
    from hbf.models import load_model

    # MLA --model with --format fp8 / bf16 resolves through resolve_format; incompatible formats raise.
    ds = load_model("deepseek-v2-lite")
    assert resolve_format(ds, "fp8").name == "mla_fp8"
    assert resolve_format(ds, "bf16").name == "mla"
    for fmt, expected_tpp, expected_name in (("fp8", 6, "mla_fp8"), ("bf16", 3, "mla")):
        out = tmp_path / f"ds-{fmt}"
        main(HBF_FLAGS + ["--model", "deepseek-v2-lite", "--format", fmt, "--context", "1024",
                          "--output", str(out)])
        meta = json.loads((out / "meta.json").read_text())
        assert meta["format"]["name"] == expected_name
        (first, *_) = csv.DictReader((out / "analysis_layers.csv").open())
        assert int(first["pages"]) == -(-1024 // expected_tpp)
    bad_mla = tmp_path / "ds-kivi2"
    with pytest.raises(SystemExit):
        main(HBF_FLAGS + ["--model", "deepseek-v2-lite", "--format", "kivi2", "--context", "1024",
                          "--output", str(bad_mla)])
    assert not bad_mla.exists()

    # Hybrid --model with --tensors applies custom tensors instead of silently using BF16.
    fp8_tensors = (TensorSpec("K", 8, 128), TensorSpec("V", 8, 128))
    gemma_spec = from_model("gemma-3-27b", tensors=fp8_tensors)
    assert all(s.bits == 8 for g in gemma_spec.groups for s in g.tensors)
    out_gemma = tmp_path / "gemma-fp8"
    main(HBF_FLAGS + ["--model", "gemma-3-27b", "--context", "2048",
                      "--tensors", '[{"name":"K","bits":8,"dims":128},{"name":"V","bits":8,"dims":128}]',
                      "--output", str(out_gemma)])
    g_meta = json.loads((out_gemma / "meta.json").read_text())
    assert all(t["bits"] == 8 for t in g_meta["tensors"])
    g_rows = list(csv.DictReader((out_gemma / "analysis_layers.csv").open()))
    # Global layer 5: 16 KV heads x 2 tensors x ceil(2048 / 32) = 2,048 FP8 pages (half of BF16's 4,096).
    assert int(g_rows[5]["pages"]) == 16 * 2 * (2048 // 32)

    # GroupedLayout rejects density < 1; generate_grouped rejects unsupported kwargs.
    cfg = load("ocp")
    shape = Shape(2, 4, 128, 1024)
    spec = parse_kvspec("cla:share=2", 2, 4, 1024)
    with pytest.raises(ValueError, match="density"):
        GroupedLayout(cfg, shape, spec, (TensorSpec("K", 16, 128, density=Fraction(1, 4)), TensorSpec("V", 16, 128)))
    layout = GroupedLayout(cfg, shape, spec, default_tensors(shape))
    for bad_kw in ({"reuse": 0.8}, {"select_static": True}, {"query_groups": 4}, {"full_tensors": ("K",)}):
        with pytest.raises(TypeError):
            list(generate_grouped(cfg, layout, "uniform_tokens", **bad_kw))


# ---------------------------------------------------------------------------------------------
# T-K4: Token flow model, eviction policies, age-based ladders, and KVGeometry parity.

def test_k4_sliding_window_token_flow_and_geometry():
    """Design section 3 K4: window (-4096, -128), prompt 32,768 -> admission writes 3,968 tokens;
    after +1,000 generated, decode writes 1,000 and 1,000 die; stored stays 3,968."""
    import numpy as np
    from hbf.frontend.kv.tensors import TensorSpec
    from hbf.kvspec import flow

    tensor = TensorSpec("kv", 8, 4096, window=(-4096, -128))  # 1 token per 4 KiB page
    f = flow(tensor, prompt=32768, generated=1000)
    assert f.admission_tokens == 3968
    assert f.decode_tokens == 1000
    assert f.died_tokens == 1000
    assert f.rewritten_tokens == 0
    assert f.stored_tokens == 3968
    # Same check through KVSpec + KVGeometry (both 1 tok/page and BF16 16 tok/page).
    spec = KVSpec("win", {}, (KVGroup("g", (0,), heads=(0, 1), tensors=(tensor,)),))
    geo = spec.geometry(layers=1, heads=1)
    assert int(geo.stored_pages([32768])[0]) == 3968
    assert int(geo.stored_pages([33768])[0]) == 3968
    assert int(geo.written_pages([0], [32768])[0]) == 3968
    assert int(geo.written_pages([32768], [33768])[0]) == 1000
    assert int(geo.dead_pages([0], [32768])[0]) == 0
    assert int(geo.dead_pages([32768], [33768])[0]) == 1000
    # Additivity over sub-steps matches single-step decode writes and deaths.
    steps = np.arange(32768, 33768, 100)
    assert int(geo.written_pages(steps, steps + 100).sum()) == 1000
    assert int(geo.dead_pages(steps, steps + 100).sum()) == 1000


def test_k4_kivi4_to_kivi2_age_based_ladder():
    """Design section 3 K4: kivi4 (-4096, -128) -> kivi2 (0, -4096) writes 160 + 96 = 256 B per
    decode token per head against 96 B for direct kivi2 (2.67x), and the kivi4 pages die."""
    from hbf.frontend.kv.formats import parse_format
    from hbf.kvspec import flow

    k4 = parse_format("kivi4", 128)
    k2 = parse_format("kivi2", 128)
    k4_hbf = tuple(s for s in k4.tensors if s.tier == "hbf")
    k2_hbf = tuple(s for s in k2.tensors if s.tier == "hbf")
    res_hbm = tuple(s for s in k4.tensors if s.tier == "hbm")
    assert sum(s.token_bytes for s in k4_hbf) == 160
    assert sum(s.token_bytes for s in k2_hbf) == 96

    ladder = KVSpec("ladder", {}, (
        KVGroup("res", (0,), heads=(0, 1), tensors=res_hbm, window=(-128, None), tier="hbm"),
        KVGroup("k4", (0,), heads=(0, 1), tensors=k4_hbf, window=(-4096, -128)),
        KVGroup("k2", (0,), heads=(0, 1), tensors=k2_hbf, window=(0, -4096)),
    ))
    direct = KVSpec("k2", {}, (
        KVGroup("res", (0,), heads=(0, 1), tensors=res_hbm, window=(-128, None), tier="hbm"),
        KVGroup("k2", (0,), heads=(0, 1), tensors=k2_hbf, window=(0, -128)),
    ))
    fl = flow(ladder, prompt=32768, generated=1000)
    fd = flow(direct, prompt=32768, generated=1000)
    assert fl.decode_bytes_per_token == pytest.approx(256.0)
    assert fd.decode_bytes_per_token == pytest.approx(96.0)
    assert fl.decode_bytes_per_token / fd.decode_bytes_per_token == pytest.approx(256 / 96)  # 2.67x
    assert fl.died_bytes == pytest.approx(1000 * 160)
    assert fd.died_bytes == 0.0
    # KVGeometry on the ladder writes the matching decode page bytes (multiple of LCM(tpp)=256 tokens).
    l_geo = ladder.geometry(layers=1, heads=1)
    d_geo = direct.geometry(layers=1, heads=1)
    assert int(l_geo.written_pages([32768], [32768 + 1024])[0]) * 4096 == 1024 * 256
    assert int(d_geo.written_pages([32768], [32768 + 1024])[0]) * 4096 == 1024 * 96
    assert int(l_geo.dead_pages([32768], [32768 + 1024])[0]) * 4096 == 1024 * 160


def test_k4_eviction_policies_and_h2o_live_page_fraction():
    """Design section 3 K4: H2O live-page fraction 1 - 0.75^16 = 0.98998; compaction threshold
    theta drives rewrite traffic; prefill/flush eviction compact before writing."""
    from hbf.frontend.kv.tensors import TensorSpec
    from hbf.kvspec import flow, live_page_fraction, parse_evict

    # H2O keep=25%, BF16 tpp=16, cluster=1 -> 1 - 0.75^16 = 0.989977... (~0.98998).
    lp = live_page_fraction(0.25, 16, cluster=1)
    assert lp == pytest.approx(1.0 - 0.75 ** 16, rel=1e-12)
    assert round(lp, 5) == 0.98998
    assert live_page_fraction(0.25, 16, cluster=16) == pytest.approx(0.25)
    assert live_page_fraction(0.25, 16, cluster=32) == pytest.approx(0.25)

    bf16_k = TensorSpec("K", 16, 128)  # 256 B/tok -> tpp=16
    none_f = flow(bf16_k, prompt=4096, generated=1024, evict="none")
    pre_f = flow(bf16_k, prompt=4096, generated=1024, evict="prefill:keep=0.25")
    flush_f = flow(bf16_k, prompt=4096, generated=1024, evict="flush:keep=0.25")
    h2o_0 = flow(bf16_k, prompt=4096, generated=1024, evict="decode:keep=0.25,cluster=1,compact=0")
    h2o_50 = flow(bf16_k, prompt=4096, generated=1024, evict="decode:keep=0.25,c=1,theta=0.5")
    h2o_25 = flow(bf16_k, prompt=4096, generated=1024, evict="decode:keep=0.25,c=1,theta=0.25")

    assert pre_f.admission_tokens == pytest.approx(1024) and pre_f.decode_tokens == 1024
    assert pre_f.live_page_frac == 1.0 and pre_f.rewritten_tokens == 0
    assert flush_f.admission_tokens == pytest.approx(1024) and flush_f.decode_tokens == pytest.approx(256)
    assert flush_f.live_page_frac == 1.0 and flush_f.rewritten_tokens == 0
    # Uncompacted H2O: all 5,120 tokens written, 75% die, 0 rewritten, 98.998% of pages stay live.
    assert h2o_0.admission_tokens == 4096 and h2o_0.decode_tokens == 1024
    assert h2o_0.died_tokens == pytest.approx(5120 * 0.75) and h2o_0.stored_tokens == pytest.approx(5120 * 0.25)
    assert h2o_0.rewritten_tokens == 0.0 and h2o_0.live_page_frac == pytest.approx(1.0 - 0.75 ** 16)
    # Compacted H2O: lower theta (eager compaction at 25% dead) causes 3x more rewrites than theta=0.5.
    assert h2o_50.rewritten_tokens == pytest.approx(5120 * 0.75 * 1.0)
    assert h2o_25.rewritten_tokens == pytest.approx(5120 * 0.75 * 3.0)
    assert h2o_50.live_page_frac == pytest.approx(0.25) and h2o_25.live_page_frac == pytest.approx(0.25)
    assert h2o_25.total_written_bytes > h2o_50.total_written_bytes > none_f.total_written_bytes
    # Validation errors on bad evict specs.
    for bad in ("bogus", "decode:keep=1.5", "decode:compact=1.0", "none:keep=0.5"):
        with pytest.raises(ValueError):
            parse_evict(bad)


@pytest.mark.parametrize("spec_text,shape", [
    ("hybrid:pattern=LLLLLG,window=1024,local_tier=hbf", Shape(6, 8, 128, 4096)),
    ("hybrid:pattern=LLLLLG,window=1024,local_tier=hbm", Shape(6, 8, 128, 4096)),
    ("cla:share=2,reuse=hbf", Shape(4, 8, 128, 4096)),
    ("cla:share=2,reuse=hbm", Shape(4, 8, 128, 4096)),
    ("duo:retrieval=0.5,sink=64,recent=256", Shape(2, 8, 128, 4096)),
    ("pyramid:first=4096,last=1024", Shape(4, 8, 128, 4096)),
    ("ssm:pattern=SSSA,state=1000000", Shape(4, 8, 128, 4096)),
    ("yoco:window=1024,local_tier=hbm,reuse=hbf", Shape(4, 8, 128, 4096)),
])
def test_k4_kvgeometry_matches_grouped_layout_per_layer(spec_text, shape):
    """KVGeometry layer_row_pages, layer_demand, stored_pages and hbm_bytes match GroupedLayout."""
    cfg = load("ocp")
    spec = parse_kvspec(spec_text, shape.layers, shape.kv_heads, shape.context)
    tensors = default_tensors(shape)
    layout = GroupedLayout(cfg, shape, spec, tensors)
    rows, _ = analyze_accesses(cfg, layout, list(generate_grouped(cfg, layout, "dense")), timing=True)
    geo = spec.geometry(shape.layers, shape.kv_heads, tensors)

    assert int(geo.stored_pages([shape.context])[0]) == layout.pages
    assert int(geo.hbm_bytes([shape.context])[0]) == layout.request_bytes(cfg)["hbm_tier_bytes_per_request"]
    assert geo.layer_row_pages([shape.context]).tolist() == [row["pages"] for row in rows]
    demands = geo.layer_demand([shape.context])
    for row, d in zip(rows, demands):
        for key in ("burst_bytes", "hbm_bytes", "attended_tokens", "attended_elements", "hbf_elements", "dequant_ops"):
            assert d[key] == pytest.approx(row[key]), (spec_text, row["layer"], key)
    # Cross-layer sharing read multiplier: s for reuse=hbf, 1 for reuse=hbm or unshared.
    if spec_text == "cla:share=2,reuse=hbf":
        assert geo.read_multiplier == 2.0
    elif spec_text == "yoco:window=1024,local_tier=hbm,reuse=hbf":
        assert geo.read_multiplier == 2.0  # 4 layers: layers 2 and 3 share layer 2
    else:
        assert geo.read_multiplier == 1.0
