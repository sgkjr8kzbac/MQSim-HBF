"""Tensor windows and tiers, named KV formats, and the system model wired into the pipeline."""
import csv
import json

import pytest

from hbf.frontend.kv.layouts import LAYOUTS
from hbf.frontend.kv.tensors import Shape, TensorSpec, parse_tensors
from hbf.frontend.trace.synthetic import generate
from hbf.frontend.trace.tokens import Access
from hbf.backends.rounds.analysis import analyze_accesses
from hbf.cli import main
from hbf.device.config import load
from hbf.backends.mqsim.schedule import schedule
from hbf.frontend.kv.formats import parse_format
from hbf.backends.rounds.pages import iter_layer_pages
from tests.reference import analyze


def kivi_layout(context=1000):
    cfg = load("ocp")
    return cfg, LAYOUTS["layer_contiguous"](cfg, Shape(1, 1, 128, context), tensors=parse_format("kivi2", 128).tensors)


def test_kivi_body_on_hbf_and_residual_in_hbm():
    cfg, layout = kivi_layout()
    # Body [0, 872): K and V 2-bit (128 tokens/page, 7 pages), K_qp and V_qp 16 B/token (256/page, 4 pages).
    assert layout.windows["K"] == (0, 872) and layout.windows["K_res"] == (872, 1000)
    assert layout.block_counts == (7, 4, 7, 4) and [spec.name for spec in layout.hbm_tensors] == ["K_res", "V_res"]
    (pages,) = iter_layer_pages(cfg, layout, generate(cfg, layout, "dense"))
    assert len(pages) == 22 and int(pages.useful.sum()) == 872 * 96 == 83_712
    assert pages.hbm_bytes == 2 * 128 * 256 == 65_536 and pages.attended_tokens == 1000
    assert pages.dequant_ops == 2 * 872 * 128 * 3
    with pytest.raises(ValueError, match="Invalid logical access"):
        Access(0, 0, 0, "K", "R", 900, 901, 2).validate(layout)
    # MQSim's schedule skips the HBM tier and agrees page by page.
    requests = [r for batch in schedule(cfg, layout, generate(cfg, layout, "dense")) for r in batch.requests]
    assert [r.lpa for r in requests] == pages.lpa.tolist()
    assert [r.useful_bytes for r in requests] == pages.useful.tolist()


@pytest.mark.parametrize("name,hbf_bytes", [("bf16", 512), ("fp8", 256), ("int4", 136), ("int2:group=64", 80),
                                            ("kivi4", 160), ("kivi2", 96), ("think", 384), ("think:ratio=0.25", 448),
                                            ("kivi2:think=0.5", 72), ("quest", 544), ("loki", 512),
                                            ("loki:rank=16", 512), ("sparq", 512), ("sparq:groups=128", 512),
                                            ("sparq:dual=1", 768), ("sparq:groups=128,dual=1", 768)])
def test_preset_bytes_per_token(name, hbf_bytes):
    fmt = parse_format(name, 128)
    assert sum(spec.token_bytes for spec in fmt.tensors if spec.tier == "hbf") == hbf_bytes
    json.dumps([vars(spec) for spec in fmt.tensors])


def test_format_parameters_are_checked():
    for text in ("int4:group=3", "kivi2:residual=-1", "think:ratio=1", "bf16:x=1", "nope", "kivi2:group=a"):
        with pytest.raises(ValueError):
            parse_format(text, 128)
    assert parse_format("kivi2:think=0.5", 128).attn_dims == 64 + 128


def test_quest_summaries_are_read_whole_and_not_attended():
    cfg = load("ocp")
    layout = LAYOUTS["layer_contiguous"](cfg, Shape(1, 2, 128, 1024), tensors=parse_format("quest", 128).tensors)
    accesses = list(generate(cfg, layout, "uniform_tokens", fraction=.05, seed=2))
    summaries = [a for a in accesses if a.kind == "K_sum"]
    assert [(a.kv_head, a.token_start, a.token_end) for a in summaries] == [(0, 0, 1024), (1, 0, 1024)]
    (pages,) = iter_layer_pages(cfg, layout, accesses)
    assert pages.attended_tokens == 2 * 4 * 16  # 4 of 64 blocks per head; the summaries are metadata
    assert pages.tensor_tokens["K_sum"] == 2 * 1024


def test_window_specs_parse_and_validate():
    (spec,) = parse_tensors('[{"name": "K", "bits": 2, "dims": 128, "window": [0, -128], "dequant_ops": 3}]')
    assert spec.window == (0, -128) and spec.tokens(1000) == (0, 872)
    for bad in ('{"window": [5]}', '{"window": [0, -2000]}', '{"tier": "ssd"}', '{"role": "x"}',
                '{"dequant_ops": -1}'):
        with pytest.raises(ValueError):
            spec = TensorSpec("K", 16, 128, **json.loads(bad))
            spec.tokens(1000)
    cfg = load("ocp")
    with pytest.raises(ValueError, match="HBF"):
        LAYOUTS["layer_contiguous"](cfg, Shape(1, 1, 128, 64), tensors=(TensorSpec("K", 16, 128, tier="hbm"),))
    with pytest.raises(ValueError, match="window"):
        LAYOUTS["vllm_interleaved"](cfg, Shape(1, 1, 128, 1024), tensors=(
            TensorSpec("K", 16, 128, (0, 512)), TensorSpec("V", 16, 128)))


@pytest.mark.parametrize("mode", ["dense", "uniform_tokens", "sink_recent", "uniform_pages"])
def test_engines_agree_on_windowed_formats(mode):
    cfg = load("ocp")
    for name in ("kivi2", "kivi4:residual=0", "quest"):
        layout = LAYOUTS["striped_contiguous"](cfg, Shape(2, 4, 128, 3000), tensors=parse_format(name, 128).tensors)
        accesses = list(generate(cfg, layout, mode, steps=2, fraction=.07, seed=3, count=40))
        assert analyze_accesses(cfg, layout, accesses, (1, 16), timing=False) == analyze(
            cfg, schedule(cfg, layout, accesses), (1, 16))


def test_layer_with_only_hbm_reads_gives_a_zero_row():
    cfg, layout = kivi_layout()
    accesses = [Access(0, 0, 0, "K_res", "R", 872, 1000, 16)]
    (row,), (step,) = analyze_accesses(cfg, layout, accesses, timing=True)
    assert row["pages"] == row["rounds"] == row["est_ns"] == row["bound_ns"] == 0
    assert row["hbm_bytes"] == 128 * 256 and row["bottleneck"] == "none" and step["utilization"] is None


def test_system_model_in_the_pipeline(tmp_path):
    out = tmp_path / "s"
    main(["--topology", "ocp", "--layout", "striped_contiguous", "--model", "llama-3.1-8b", "--gpu", "h100-sxm",
          "--context", "4096", "--batch", "4", "--stacks", "2", "--format", "kivi2", "--read-plane-ns", "0",
          "--coalesce-ns", "1", "--cache-read", "--analysis-only", "--output", str(out)])
    meta = json.loads((out / "meta.json").read_text())
    assert meta["shape"] == {"layers": 32, "kv_heads": 16, "head_dim": 128, "context": 4096}
    system = meta["system"]
    assert (system["batch"], system["stacks"], system["analysis_batch"], system["selection"]) == (4, 2, 2, "static")
    per_head = (2 * 31 + 2 * 16) * 4096 + 2 * 128 * 256  # K, V 31 pages and K_qp, V_qp 16 pages; residual in HBM
    assert system["kv_bytes_per_request"] == per_head * 32 * 8
    assert system["hbm_tier_bytes_per_request"] == 2 * 128 * 256 * 32 * 8
    (step,) = csv.DictReader((out / "system_steps.csv").open())
    assert float(step["tpot_hbf_est_ns"]) > float(step["tpot_hbf_bound_ns"]) > 0
    layers = list(csv.DictReader((out / "analysis_layers.csv").open()))
    assert len(layers) == 32 and layers[0]["attn_dims"] == "256"
    assert "系统级评估" in (out / "report.md").read_text()


def test_system_flags_are_checked(tmp_path):
    base = ["--topology", "ocp", "--model", "llama-3.1-8b", "--analysis-only", "--context", "1024"]
    for index, extra in enumerate((["--layers", "2"], ["--overlap", "prefetch:1", "--mode", "uniform_tokens"],
                                   ["--format", "kivi2", "--tensors", "[]"], ["--batch", "0"], ["--plane-group", "16"])):
        with pytest.raises(SystemExit):
            main(base + extra + ["--output", str(tmp_path / str(index))])
        assert not (tmp_path / str(index)).exists()


HBF_FLAGS = ["--topology", "ocp", "--read-plane-ns", "0", "--coalesce-ns", "1", "--cache-read", "--analysis-only"]


def test_batched_trace_must_cover_every_request(tmp_path):
    main(HBF_FLAGS + ["--model", "llama-3.1-8b", "--context", "1024", "--output", str(tmp_path / "one")])
    with pytest.raises(SystemExit):
        main(HBF_FLAGS + ["--model", "llama-3.1-8b", "--context", "1024", "--batch", "8",
                          "--access-csv", str(tmp_path / "one" / "access.csv"), "--output", str(tmp_path / "eight")])
    assert not (tmp_path / "eight").exists()
    main(HBF_FLAGS + ["--model", "llama-3.1-8b", "--context", "1024", "--batch", "2", "--output", str(tmp_path / "two")])
    main(HBF_FLAGS + ["--model", "llama-3.1-8b", "--context", "1024", "--batch", "2",
                      "--access-csv", str(tmp_path / "two" / "access.csv"), "--output", str(tmp_path / "again")])
    assert (tmp_path / "again" / "analysis_steps.csv").read_text() == (tmp_path / "two" / "analysis_steps.csv").read_text()


def test_reevaluation_infers_the_analysed_batch(tmp_path):
    from hbf.system import main as system_main
    run = tmp_path / "r"
    main(HBF_FLAGS + ["--layers", "32", "--context", "1024", "--batch", "4", "--output", str(run)])
    result = system_main([str(run), "--model", "llama-3.1-8b", "--gpu", "h100-sxm", "--batch", "4",
                          "--output", str(tmp_path / "ok")])
    assert result.steps[0]["tpot_hbf_bound_ns"] > 0
    with pytest.raises(SystemExit):  # 4 requests were analysed, not 8
        system_main([str(run), "--model", "llama-3.1-8b", "--gpu", "h100-sxm", "--batch", "8",
                     "--output", str(tmp_path / "bad")])


def test_step_without_hbf_pages_still_reports(tmp_path):
    trace = tmp_path / "trace.csv"
    trace.write_text("step,layer,kv_head,kind,op,token_start,token_end,bits\n0,0,0,K_res,R,128,256,16\n"
                     "0,0,0,V_res,R,128,256,16\n")
    main(HBF_FLAGS + ["--format", "kivi2", "--layers", "1", "--heads", "1", "--context", "256",
                      "--access-csv", str(trace), "--output", str(tmp_path / "hbm")])
    report = (tmp_path / "hbm" / "report.md").read_text()
    assert "| 1 | 0 | 1 | 0 | 0 |" in report and "—" in report
    main(["--topology", "ocp", "--analysis-only", "--format", "kivi2", "--layers", "1", "--heads", "1",
          "--context", "256", "--access-csv", str(trace), "--output", str(tmp_path / "plain")])
    assert (tmp_path / "plain" / "analysis_layers.csv").is_file()  # timing is always computed (zero row kept)


def test_attended_elements_skip_scales_and_metadata():
    cfg = load("ocp")
    layout = LAYOUTS["layer_contiguous"](cfg, Shape(1, 8, 128, 32768), tensors=parse_format("kivi2", 128).tensors)
    (pages,) = iter_layer_pages(cfg, layout, generate(cfg, layout, "uniform_pages", fraction=.1, seed=4))
    tokens = pages.tensor_tokens
    assert pages.attended_elements == 128 * (tokens["K"] + tokens["V"] + tokens["K_res"] + tokens["V_res"])
    assert tokens["K_qp"] > 0 and parse_format("kivi2", 128).tensors[1].role == "aux"


def test_format_attention_dims_without_model(tmp_path):
    main(HBF_FLAGS + ["--format", "think", "--context", "1024", "--output", str(tmp_path / "t")])
    (row, *_) = csv.DictReader((tmp_path / "t" / "analysis_layers.csv").open())
    assert row["attn_dims"] == "192"
    assert json.loads((tmp_path / "t" / "meta.json").read_text())["format"]["name"] == "think"


def test_unvalidated_settings_are_flagged_and_rejected_early(tmp_path):
    main(HBF_FLAGS + ["--read-ns", "2000", "--context", "1024", "--output", str(tmp_path / "r")])
    assert "未经验证" in (tmp_path / "r" / "report.md").read_text()
    for index, extra in enumerate((["--placement", "mqsim"], ["--gpu", "h100-sxm"],
                                   ["--overlap", "ideal"])):
        with pytest.raises(SystemExit):
            main(HBF_FLAGS + extra + ["--context", "1024", "--output", str(tmp_path / str(index))])
        assert not (tmp_path / str(index)).exists()


# ---------------------------------------------------------------------------------------------
# T-K1 page formats (docs/design/phase3-kv-terms.md section 3, K1): hand-computed values on ocp.

def dense_pages(name, context, mode="dense", heads=1, q_heads=None, **generation):
    cfg = load("ocp")
    fmt = parse_format(name, 128, q_heads)
    layout = LAYOUTS["layer_contiguous"](cfg, Shape(1, heads, 128, context), tensors=fmt.tensors)
    accesses = list(generate(cfg, layout, mode, **generation))
    (pages,) = iter_layer_pages(cfg, layout, accesses)
    (row,), _ = analyze_accesses(cfg, layout, accesses)
    return fmt, layout, pages, row


def test_mla_latent_rows():
    fmt, layout, pages, row = dense_pages("mla", 32768)
    assert layout.tokens_per_page == (3,) and fmt.attn_dims == 1088
    assert len(pages) == 10_923 and row["min_rounds"] == 11  # ceil(32,768 / 3), ceil(10,923 / 1,024)
    assert pages.attended_elements == 32768 * 1088 and pages.hbf_elements == 32768 * 576
    _, layout, pages, _ = dense_pages("mla:split=1", 32768)
    assert layout.tokens_per_page == (4, 32) and len(pages) == 8192 + 1024  # 9,216 pages, 15.6% fewer
    assert pages.attended_elements == 32768 * 1088
    _, layout, pages, _ = dense_pages("mla_fp8", 32768)
    assert layout.tokens_per_page == (6,) and len(pages) == 5462  # 656 B rows
    assert pages.dequant_ops == pytest.approx(32768 * 512)  # only the FP8 latent is dequantized


def test_dsa_indexer_is_read_whole_and_attended_by_its_heads():
    with pytest.raises(ValueError, match="q_heads"):
        parse_format("mla:indexer=1", 128)
    context, fraction = 131072, 2048 / 131072
    fmt, layout, pages, _ = dense_pages("mla:indexer=1", context, "uniform_tokens", q_heads=128,
                                        fraction=fraction, select_tokens=1, seed=5)
    index = layout.index["K_idx"]
    assert layout.tokens_per_page[index] == 31 and fmt.tensors[1].attended_dims == 64 * 128 // 128
    indexer = int((pages.tensor == index).sum())
    assert indexer == -(-context // 31) == 4229
    data = len(pages) - indexer  # 43,691 latent pages, each touched with probability 1 - (1 - f)^3
    assert data == pytest.approx(43_691 * (1 - (1 - fraction) ** 3), rel=.03)
    assert indexer / len(pages) > .6
    assert pages.attended_elements == pages.tensor_tokens["C"] * 1088 + context * 64
    _, layout, pages, _ = dense_pages("mla:indexer_hbm=1", 1024, q_heads=16)
    assert pages.hbm_bytes == 1024 * 132 and pages.attended_elements == 1024 * (1088 + 512)


def test_fused_quantization_rows():
    _, layout, pages, _ = dense_pages("kivi2:residual=0", 4096)
    assert len(pages) == 96 and layout.block_counts == (32, 16, 32, 16)
    fmt, layout, fused, _ = dense_pages("kivi2:residual=0,fused=1", 4096)
    assert layout.tokens_per_page == (42,) and len(fused) == 98  # one 96 B row per token
    assert fused.dequant_ops == pages.dequant_ops and fused.hbf_elements == pages.hbf_elements == 4096 * 256
    assert fused.attended_elements == pages.attended_elements
    _, layout, padded, _ = dense_pages("kivi2:residual=0,fused=1,pow2=1", 4096)
    assert layout.tokens_per_page == (32,) and len(padded) == 128
    for text in ("kivi2:pow2=1", "int4:fused=2", "bf16:fused=1"):
        with pytest.raises(ValueError):
            parse_format(text, 128)
    assert parse_format("kivi2:fused=1", 128).tensors[1].name == "K_res"  # the HBM residual stays apart


@pytest.mark.parametrize("name,per_block", [("kivi2:residual=0", 4.0), ("kivi2:residual=0,fused=1,pow2=1", 1.0),
                                            ("kivi2:residual=0,fused=1", 28 / 21)])
def test_fused_rows_cut_pages_per_selected_block(name, per_block):
    # Every 16-token block of a 336-token context (LCM of 16 and 42) selected alone, one per step.
    cfg = load("ocp")
    layout = LAYOUTS["layer_contiguous"](cfg, Shape(1, 1, 128, 336), tensors=parse_format(name, 128).tensors)
    accesses = [Access(block, 0, 0, spec.name, "R", 16 * block, 16 * block + 16, spec.bits)
                for block in range(21) for spec in layout.tensors]
    pages = list(iter_layer_pages(cfg, layout, accesses))
    assert sum(len(p) for p in pages) / 21 == pytest.approx(per_block)


def test_attention_flops_use_attn_dims():
    from hbf.models import ModelSpec, load_gpu
    from hbf.system import SystemOptions, evaluate
    fmt, layout, pages, _ = dense_pages("mla", 3)
    model = ModelSpec("mla-tiny", layers=1, hidden=256, q_heads=16, kv_heads=1, head_dim=128, intermediate=512,
                      vocab=1000)
    row = {"step": 0, "layer": 0, "est_ns": 0, "bound_ns": 0, "burst_bytes": 0, "hbm_bytes": 0,
           "attended_tokens": pages.attended_tokens, "dequant_ops": 0, "attended_elements": pages.attended_elements}
    result = evaluate([row], model, load_gpu("b200"), SystemOptions(),
                      {"batch": 1, "analysis_batch": 1, "selection": "static"})
    assert result.layers[0]["attn_flops"] == 2 * 16 * 3 * 1088 == 104_448


def test_new_tensor_fields_are_optional_in_meta():
    from hbf.frontend.kv.tensors import tensors_meta
    cfg = load("ocp")
    plain = LAYOUTS["layer_contiguous"](cfg, Shape(1, 1, 128, 64))
    assert set(tensors_meta(plain)[0]) == {"name", "bits", "dims", "window", "tier", "role", "dequant_ops"}
    mla = LAYOUTS["layer_contiguous"](cfg, Shape(1, 1, 128, 64), tensors=parse_format("mla", 128).tensors)
    assert tensors_meta(mla)[0]["attn_dims"] == 1088 and "logical_dims" not in tensors_meta(mla)[0]
    with pytest.raises(ValueError):
        TensorSpec("K", 16, 128, attn_dims=-1)
    with pytest.raises(ValueError):
        TensorSpec("K", 16, 128, logical_dims=0)
    assert parse_tensors('[{"name": "C", "bits": 16, "dims": 576, "attn_dims": 1088}]')[0].attended_dims == 1088


def test_mla_rejects_both_indexers_and_invalid_q_heads():
    # M3: indexer and indexer_hbm cannot both be set; q_heads must be a positive int.
    with pytest.raises(ValueError, match="mutually exclusive"):
        parse_format("mla:indexer=1,indexer_hbm=1,q_heads=16", 128)
    for text in ("mla:indexer=1,q_heads=-1", "mla:indexer=1,q_heads=1.5", "mla:q_heads=-2", "mla:q_heads=1.5"):
        with pytest.raises(ValueError, match="q_heads"):
            parse_format(text, 128, q_heads=16)
    with pytest.raises(ValueError, match="q_heads"):
        parse_format("mla:indexer=1", 128, q_heads=1.5)


@pytest.mark.parametrize("name,q_heads", [("mla_fp8", None), ("mla:indexer=1", 128),
                                          ("kivi2:fused=1", None), ("kivi2:fused=1,pow2=1", None)])
@pytest.mark.parametrize("mode", ["dense", "uniform_tokens", "sink_recent"])
def test_engines_agree_on_non_64b_aligned_formats(name, q_heads, mode):
    # L6: numpy and reference engines agree on non-64B-aligned rows (656 B, 132 B, 96 B).
    from hbf.backends.rounds.analysis import Protocol
    cfg = load("ocp")
    fmt = parse_format(name, 128, q_heads=q_heads)
    layout = LAYOUTS["striped_contiguous"](cfg, Shape(2, 2, 128, 2048), tensors=fmt.tensors)
    accesses = list(generate(cfg, layout, mode, steps=1, fraction=.07, seed=3, count=40))
    for protocol in (None, Protocol(256, read_unit="wanted"), Protocol(256, read_burst=True, read_unit="wanted")):
        expected = analyze(cfg, schedule(cfg, layout, accesses), (1, 16), protocol=protocol)
        actual = analyze_accesses(cfg, layout, accesses, (1, 16), protocol=protocol, timing=False)
        assert actual == expected


# ---------------------------------------------------------------------------------------------
# T-K6 partial-dimension formats (docs/design/phase3-kv-terms.md section 3, K6).

def test_loki_format_and_pages_at_32k():
    fmt, layout, pages, _ = dense_pages("loki:rank=32", 32768)
    assert fmt.attn_dims == 256 and fmt.full_tensors == ("K_lo",)
    assert [spec.name for spec in fmt.tensors] == ["K_lo", "K_hi", "V"]
    # K_lo: 32 dims x 2 B = 64 B -> 64 tokens/page -> 512 pages per head (vs 2,048 for full K).
    assert layout.tokens_per_page == (64, 21, 16)
    assert layout.block_counts == (512, 1561, 2048)
    assert pages.attended_elements == 32768 * 256 and pages.hbf_elements == 32768 * 256
    for bad in ("loki:rank=0", "loki:rank=-4", "loki:rank=128", "loki:rank=200", "loki:rank=16.5", "loki:x=1"):
        with pytest.raises(ValueError):
            parse_format(bad, 128)


def test_sparq_format_dual_and_validation():
    fmt, layout, pages, _ = dense_pages("sparq:channels=32,groups=8", 32768)
    assert fmt.attn_dims == 256 and fmt.channel_select_channels == 32
    assert fmt.channel_groups == tuple(f"K_g{i}" for i in range(8))
    assert layout.block_counts == (256,) * 8 + (2048,)
    assert pages.attended_elements == 32768 * 256 and pages.hbf_elements == 32768 * 256

    fmt128, layout128, _, _ = dense_pages("sparq:channels=32,groups=128", 32768)
    assert len(fmt128.channel_groups) == 128
    # 1 channel x 2 B = 2 B/token -> 2,048 tokens/page -> 16 pages per group at ctx 32,768.
    assert layout128.tokens_per_page[:128] == (2048,) * 128
    assert layout128.block_counts[:128] == (16,) * 128

    # dual=1 adds token-major K alongside channel-grouped K_g* (aux), raising total HBF bytes/token by 1.5x
    # without double-counting attended_elements or hbf_elements.
    bf16_bytes = sum(spec.token_bytes for spec in parse_format("bf16", 128).tensors)
    dual_fmt, dual_layout, dual_pages, _ = dense_pages("sparq:channels=32,groups=8,dual=1", 4096)
    dual_bytes = sum(spec.token_bytes for spec in dual_fmt.tensors if spec.tier == "hbf")
    assert dual_bytes == bf16_bytes * 3 // 2 == 768
    assert all(spec.role == "aux" for spec in dual_fmt.tensors[:8])
    assert [spec.name for spec in dual_fmt.tensors[-2:]] == ["K", "V"]
    assert dual_pages.attended_elements == 4096 * 256 and dual_pages.hbf_elements == 4096 * 256

    for bad in ("sparq:groups=0", "sparq:groups=3", "sparq:groups=256", "sparq:groups=8.5",
                "sparq:channels=0", "sparq:channels=129", "sparq:channels=32.5",
                "sparq:dual=2", "sparq:dual=-1", "sparq:dual=0.5", "sparq:x=1"):
        with pytest.raises(ValueError):
            parse_format(bad, 128)
