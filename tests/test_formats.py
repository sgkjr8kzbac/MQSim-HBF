"""Tensor windows and tiers, named KV formats, and the system model wired into the pipeline."""
import csv
import json

import pytest

from hbf.access import LAYOUTS, Access, Shape, TensorSpec, generate, parse_tensors
from hbf.analysis import analyze, analyze_accesses
from hbf.cli import main
from hbf.config import load
from hbf.controller import schedule
from hbf.formats import parse_format
from hbf.pages import iter_layer_pages


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
                                            ("kivi2:think=0.5", 72), ("quest", 544)])
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
        assert analyze_accesses(cfg, layout, accesses, (1, 16), engine="numpy") == analyze(
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
