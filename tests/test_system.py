"""System model (hbf/system.py): hand-computed compute, overlap, scaling and capacity terms."""
import csv
import json

import numpy as np
import pytest

from hbf.models import GPUS, MODELS, GpuSpec, ModelSpec, experts_touched, load_gpu, load_model, presets
from hbf.system import SystemOptions, evaluate, linear_ns, main, parse_overlap, read_layers, timeline

LLAMA = load_model("llama-3.1-8b")
META = {"batch": 1, "stacks": 1, "selection": "static", "kv_format": "bf16"}
# Llama-3.1-8B at B = 1 on 3.35e12 B/s, memory_eff 1: (pre + post) x 2 B / BW = 436,207,616 / 3,350 ns.
LINEAR_NS = 436_207_616 / 3350  # 130,211.2287 ns per layer
LM_HEAD_NS = 128256 * 4096 * 2 / 3350  # 313,633.78 ns per step


def gpu(**overrides):
    """H100-like rates with efficiencies 1 so that times are bytes / 3.35e12 and FLOPs / 989e12."""
    return GpuSpec(**({"name": "test", "tensor_flops": 989e12, "vector_ops": 67e12, "hbm_bytes_s": 3.35e12,
                       "hbm_capacity": 80 * 10 ** 9, "compute_eff": 1.0, "memory_eff": 1.0} | overrides))


def rows(layers=32, steps=(0,), **fields):
    base = {"est_ns": 1_000_000, "bound_ns": 800_000, "burst_bytes": 0, "hbm_bytes": 0, "attended_tokens": 0,
            "dequant_ops": 0}
    return [{"step": step, "layer": layer} | base | fields for step in steps for layer in range(layers)]


def test_presets_parse_with_sources():
    for name in presets(MODELS):
        model = load_model(name)
        assert model.name == name and "https://" in model.source
    for name in presets(GPUS):
        spec = load_gpu(name)
        assert spec.name == name and "https://" in spec.source and "Assumptions" in spec.source
        assert (spec.compute_eff, spec.memory_eff) == (.5, .8)
    assert load_gpu("h100-sxm").bandwidth == pytest.approx(3.35e12 * .8)
    assert load_gpu('{"preset": "b200", "memory_eff": 1.0}').bandwidth == 8e12
    assert load_model(json.dumps({"preset": "qwen3-32b"})).q_heads * 128 == 8192
    with pytest.raises(ValueError, match="presets are"):
        load_model("no-such-model")
    with pytest.raises(ValueError, match="Unknown GpuSpec fields"):
        load_gpu('{"preset": "h200", "hbm_bw": 1}')
    with pytest.raises(ValueError):
        ModelSpec("bad", 32, 4096, 30, 8, 128, 14336, 128256)  # q_heads not a multiple of kv_heads
    with pytest.raises(ValueError):
        gpu(memory_eff=1.2)


def test_llama_parameters():
    assert (LLAMA.pre_params, LLAMA.post_params, LLAMA.layer_params) == (25_165_824, 192_937_984, 218_103_808)
    assert LLAMA.weights_bytes == (32 * 218_103_808 + 2 * 128256 * 4096) * 2 == 16_059_990_016


def test_linear_time_at_batch_one():
    result = evaluate(rows(), LLAMA, gpu(), SystemOptions(), META)
    context = result.context
    assert context["pre_ns"] + context["post_ns"] == pytest.approx(LINEAR_NS, rel=1e-12)
    assert context["pre_ns"] + context["post_ns"] == pytest.approx(130_211.2287, abs=1e-3)
    assert context["lm_head_ns"] == pytest.approx(LM_HEAD_NS, rel=1e-12)
    # No attention demand: serial stream gives T = sum(pre + h + post); bound uses bound_ns.
    step = result.steps[0]
    assert step["tpot_hbf_est_ns"] == pytest.approx(32 * (LINEAR_NS + 1e6) + LM_HEAD_NS, rel=1e-12)
    assert step["tpot_hbf_bound_ns"] == pytest.approx(32 * (LINEAR_NS + 8e5) + LM_HEAD_NS, rel=1e-12)
    assert step["tpot_hbm_ns"] == step["gpu_ns"] == pytest.approx(32 * LINEAR_NS + LM_HEAD_NS, rel=1e-12)
    assert step["tokens_per_s_hbf"] == pytest.approx(1e9 / step["tpot_hbf_est_ns"])
    assert step["hbf_wait_share"] == pytest.approx(32e6 / (32 * (LINEAR_NS + 1e6)))


def test_attention_flops():
    # 8,192 tokens per KV head over 8 KV heads, q/kv = 4, head_dim 128: 4 x 4 x 128 x 65,536.
    result = evaluate(rows(attended_tokens=8 * 8192), LLAMA, gpu(), SystemOptions(), META)
    assert {row["attn_flops"] for row in result.layers} == {134_217_728}
    assert result.layers[0]["attn_compute_ns"] == pytest.approx(134_217_728 / 989e3)
    # ThinK-pruned K (d_k = 64): attn_dims = 64 + 128 per row.
    pruned = evaluate(rows(attended_tokens=8 * 8192, attn_dims=192), LLAMA, gpu(), SystemOptions(), META)
    assert pruned.layers[0]["attn_flops"] == 2 * 4 * 192 * 65536 == 100_663_296


def test_overlap_recurrence_hand_computed():
    h, pre, a, post = [3, 1, 2], [0] * 3, [2] * 3, [0] * 3
    # serial stream: max(3,2) + max(1,2) + max(2,2) = 7; wait = 1.
    assert timeline(h, pre, a, post, "serial") == (7, 1, [1, 0, 0])
    # serial staged: (3+2) + (1+2) + (2+2) = 12; wait = sum h.
    assert timeline(h, pre, a, post, "serial", staged=True) == (12, 6, [3, 1, 2])
    # prefetch:1: r_end = 3, 4, 7 (layer 2 waits for c_end(0) = 5); c_end = 5, 7, 9.
    assert timeline(h, pre, a, post, "prefetch:1") == (9, 3, [3, 0, 0])
    # prefetch:0 reads layer l only after layer l-1 is computed: same as serial staged here.
    assert timeline(h, pre, a, post, "prefetch:0") == (12, 6, [3, 1, 2])
    # ideal: max(sum h, sum g) = max(6, 6).
    assert timeline(h, pre, a, post, "ideal") == (6, 0, None)
    assert timeline([5, 5, 5], pre, a, post, "ideal") == (15, 9, None)


def test_prefetch_through_evaluate():
    data = rows(steps=(0, 1), attended_tokens=65536, burst_bytes=4 * 2 ** 20, est_ns=300_000)
    result = evaluate(data, LLAMA, gpu(), SystemOptions("prefetch:1"), META)
    layers = [row for row in result.layers if row["step"] == 0]
    total, wait, _ = timeline([300_000] * 32, [row["pre_ns"] for row in layers],
                              [row["attn_staged_ns"] for row in layers], [row["post_ns"] for row in layers],
                              "prefetch:1")
    step = result.steps[0]
    assert step["tpot_hbf_est_ns"] == pytest.approx(total + LM_HEAD_NS)
    assert step["hbf_wait_share"] == pytest.approx(wait / total)
    assert step["gpu_ns"] == pytest.approx(sum(r["pre_ns"] + r["attn_staged_ns"] + r["post_ns"] for r in layers)
                                           + LM_HEAD_NS)


def test_prefetch_rejected_for_dynamic_selection():
    for selection in ("dynamic", None):
        with pytest.raises(ValueError, match="static"):
            evaluate(rows(), LLAMA, gpu(), SystemOptions("prefetch:1"), META | {"selection": selection})
    assert evaluate(rows(), LLAMA, gpu(), SystemOptions("ideal"), META | {"selection": "dynamic"}).steps


def test_options_validation():
    assert parse_overlap("prefetch:2") == ("prefetch", 2)
    for text in ("prefetch", "prefetch:x", "overlap"):
        with pytest.raises(ValueError):
            SystemOptions(text)
    with pytest.raises(ValueError):
        SystemOptions(kv_consume="lazy")
    with pytest.raises(NotImplementedError):
        SystemOptions(weights="hbf")


def test_est_none_gives_no_est_tpot():
    data = rows(steps=(0, 1))
    data[3]["est_ns"] = None  # step 0, layer 3
    first, second = evaluate(data, LLAMA, gpu(), SystemOptions(), META).steps
    assert first["tpot_hbf_est_ns"] is None and first["tokens_per_s_hbf"] is None
    assert first["hbf_over_hbm"] is None and first["hbf_wait_share"] is None and first["hbf_read_est_ns"] is None
    assert first["tpot_hbf_bound_ns"] == second["tpot_hbf_bound_ns"] is not None
    assert second["tpot_hbf_est_ns"] is not None


def test_missing_duplicate_or_extra_layer_raises():
    data = rows()
    del data[5]
    with pytest.raises(ValueError, match=r"layers \[5\]"):
        evaluate(data, LLAMA, gpu(), SystemOptions(), META)
    with pytest.raises(ValueError, match="Duplicate"):
        evaluate(rows() + rows()[:1], LLAMA, gpu(), SystemOptions(), META)
    with pytest.raises(ValueError, match="32 layers"):
        evaluate(rows(layers=33), LLAMA, gpu(), SystemOptions(), META)
    with pytest.raises(ValueError, match="bound_ns"):
        evaluate(rows(bound_ns=None), LLAMA, gpu(), SystemOptions(), META)


def test_hbm_fits_boundary():
    weights = LLAMA.weights_bytes
    meta = META | {"batch": 4, "stacks": 2, "kv_bytes_per_request": 1_000_000}
    fits = evaluate(rows(), LLAMA, gpu(hbm_capacity=weights + 4_000_000), SystemOptions(), meta).steps[0]
    assert fits["hbm_fits"] is True and fits["hbm_fits_hbf"] is True
    over = evaluate(rows(), LLAMA, gpu(hbm_capacity=weights + 3_999_999), SystemOptions(), meta).steps[0]
    assert over["hbm_fits"] is False
    assert evaluate(rows(), LLAMA, gpu(), SystemOptions(), META).steps[0]["hbm_fits"] is None
    # HBF plan: weights + B x HBM-tier bytes + (d+1) x max layer burst x B/b = 4 x 500 + 2 x 1,000 x 2.
    data = rows()
    data[7]["burst_bytes"] = 1000
    meta |= {"hbm_tier_bytes_per_request": 500}
    for capacity, expected in ((weights + 6000, True), (weights + 5999, False)):
        step = evaluate(data, LLAMA, gpu(hbm_capacity=capacity), SystemOptions("prefetch:1"), meta).steps[0]
        assert step["hbm_fits_hbf"] is expected
    step = evaluate(data, LLAMA, gpu(hbm_capacity=weights + 2000), SystemOptions(), meta).steps[0]
    assert step["hbm_fits_hbf"] is True  # serial: no prefetch buffer


def test_batch_scaling_with_two_stacks():
    # B = 4 over 2 stacks: rows cover b = 2 requests, demand scales by 2, HBF time does not.
    data = rows(est_ns=5000, bound_ns=4000, attended_tokens=1000, hbm_bytes=3350, burst_bytes=3_350_000,
                dequant_ops=67_000)
    result = evaluate(data, LLAMA, gpu(), SystemOptions(), META | {"batch": 4, "stacks": 2})
    row = result.layers[0]
    assert (row["hbf_est_ns"], row["hbf_bound_ns"]) == (5000, 4000)
    assert row["attn_flops"] == 2 * 4 * 256 * 1000 * 2 == 4_096_000
    assert row["dequant_ns"] == pytest.approx(2.0)  # 67,000 x 2 / 67e12 s
    assert row["attn_compute_ns"] == pytest.approx(4_096_000 / 989e3 + 2)
    assert row["attn_ns"] == pytest.approx(row["attn_compute_ns"])  # HBM tier: 3,350 x 2 B = 2 ns
    assert row["attn_staged_ns"] == pytest.approx(2002)  # (3,350 + 3,350,000) x 2 B / 3.35e12 B/s
    linear = max(2 * 4 * LLAMA.layer_params / 989e3, LINEAR_NS)
    assert row["pre_ns"] + row["post_ns"] == pytest.approx(linear)
    step = result.steps[0]
    assert step["tpot_hbm_ns"] == pytest.approx(32 * (linear + 2002) + LM_HEAD_NS)
    assert step["tokens_per_s_hbm"] == pytest.approx(4e9 / step["tpot_hbm_ns"])
    # B mod N != 0: B = 3 over 2 stacks also has b = 2, so demand scales by 1.5.
    odd = evaluate(data, LLAMA, gpu(), SystemOptions(), META | {"batch": 3, "stacks": 2})
    assert odd.layers[0]["attn_flops"] == pytest.approx(4_096_000 * .75)
    assert any("B mod N" in line for line in odd.assumptions)
    with pytest.raises(ValueError, match="analysis_batch"):
        evaluate(data, LLAMA, gpu(), SystemOptions(), META | {"batch": 4, "stacks": 1, "analysis_batch": 2})


def test_numpy_inputs_give_python_outputs():
    data = [{key: np.int64(value) if isinstance(value, int) else value for key, value in row.items()}
            for row in rows(attended_tokens=4096, burst_bytes=8192)]
    result = evaluate(data, LLAMA, gpu(), SystemOptions("prefetch:1"), META | {"kv_bytes_per_request": 10})
    for row in result.layers + result.steps:
        assert all(type(value) in (int, float, bool, type(None)) for value in row.values()), row


def write_run(directory, layers=32, steps=(0, 1)):
    """Synthetic analysis run: group 1 and group 2 rows, step 1 lacks est_ns on one layer."""
    directory.mkdir()
    columns = ["group", "step", "layer", "pages", "rounds", "est_ns", "bound_ns", "burst_bytes", "hbm_bytes",
               "attended_tokens", "dequant_ops"]
    with (directory / "analysis_layers.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for group in (1, 2):
            for step in steps:
                for layer in range(layers):
                    est = "" if (step, layer) == (1, 4) else 250_000 + 1000 * layer
                    writer.writerow([group, step, layer, 4096, 4, est if group == 1 else "x", 200_000,
                                     16 * 2 ** 20, 0, 2 * 8 * 8192, 0.0])
    (directory / "meta.json").write_text(json.dumps({
        "shape": {"layers": layers, "kv_heads": 16, "head_dim": 128, "context": 8192},
        "system": {"model": "llama-3.1-8b", "gpu": "h100-sxm", "batch": 4, "analysis_batch": 2, "stacks": 2,
                   "selection": "static", "kv_format": "bf16", "kv_bytes_per_request": 32 * 2 * 8 * 8192 * 256,
                   "hbm_tier_bytes_per_request": 0}}))
    return directory


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def test_reevaluator_end_to_end(tmp_path):
    run = write_run(tmp_path / "run")
    loaded = read_layers(run / "analysis_layers.csv")
    assert len(loaded) == 64 and loaded[32 + 4]["est_ns"] is None and type(loaded[0]["dequant_ops"]) is float
    result = main([str(run), "--overlap", "prefetch:1"])
    expected = evaluate(loaded, LLAMA, load_gpu("h100-sxm"), SystemOptions("prefetch:1"),
                        json.loads((run / "meta.json").read_text())["system"])
    assert result.steps == expected.steps
    steps = read_csv(run / "system_steps.csv")
    assert [row["step"] for row in steps] == ["0", "1"]
    assert float(steps[0]["tpot_hbf_est_ns"]) == pytest.approx(expected.steps[0]["tpot_hbf_est_ns"])
    assert steps[1]["tpot_hbf_est_ns"] == "" and steps[1]["tpot_hbf_bound_ns"] != ""
    assert steps[0]["hbm_fits"] == "True"
    assert len(read_csv(run / "system_layers.csv")) == 64
    text = "".join(path.read_text() for path in run.glob("system_*"))
    assert "np." not in text and "numpy" not in text
    report = (run / "system_report.md").read_text()
    assert report.startswith("# 系统级评估") and "## 假设" in report and "prefetch:1" in report
    assert all(line in report for line in result.assumptions)
    # Flags override meta: another GPU, staged consumption and B = 8 over 4 stacks (still b = 2).
    other = main([str(run), "--gpu", "b200", "--batch", "8", "--stacks", "4", "--kv-consume", "staged",
                  "--output", str(tmp_path / "b200")])
    assert other.context["gpu"].name == "b200" and other.context["scale"] == 4
    assert (tmp_path / "b200" / "system_steps.csv").is_file()
    for argv in (["--batch", "8"],  # ceil(8 / 2) != analysis batch 2
                 ["--model", "llama-3.1-70b"],  # 80 layers, not the analysed 32
                 ["--overlap", "prefetch"]):
        with pytest.raises(SystemExit):
            main([str(run), *argv, "--output", str(tmp_path / "rejected")])


# ---------------------------------------------------------------------------------------------
# T-K3 model presets (MLA, hybrid attention, SSM hybrid, MoE): design section 3, K3.

@pytest.mark.parametrize("name,pub_total,pub_active,active_rel", [
    ("deepseek-v2-lite", 15.7e9, 2.4e9, 0.025),   # exact 15.706B / 2.451B (published 2.4B has 2 sig figs)
    ("gemma-3-27b", 27.016e9, 27.016e9, 0.01),    # text-only: 25.600B non-emb + 1.416B tied emb = 27.016B
    ("gpt-oss-20b", 20.91e9, 3.61e9, 0.01),       # Model Card Table 1: 20.91B / 3.61B
    ("gpt-oss-120b", 116.83e9, 5.13e9, 0.01),     # Model Card Table 1: 116.83B / 5.13B
    ("qwen3-next-80b-a3b", 79.7e9, 3.56e9, 0.01), # 79.67B total (within 1% of both 79.7B and 80B), 3.25B layers + 0.31B LM head
])
def test_k3_preset_parameter_counts(name, pub_total, pub_active, active_rel):
    model = load_model(name)
    assert model.total_params == pytest.approx(pub_total, rel=0.01)
    assert model.active_params == pytest.approx(pub_active, rel=active_rel)
    assert model.weights_bytes == model.total_params * model.weight_bits // 8


def test_k3_preset_specific_parameter_checks():
    gemma = load_model("gemma-3-27b")
    assert gemma.total_params - gemma.embedding_params == pytest.approx(25.6e9, rel=0.01)
    qwen_next = load_model("qwen3-next-80b-a3b")
    assert qwen_next.total_params == pytest.approx(80.0e9, rel=0.01)
    assert qwen_next.total_params - qwen_next.embedding_params == pytest.approx(79.0e9, rel=0.01)
    assert qwen_next.active_params - qwen_next.vocab * qwen_next.hidden == pytest.approx(3.25e9, rel=0.01)


def test_experts_touched_formula():
    assert experts_touched(64, 6, 1) == pytest.approx(6.0, rel=1e-12)
    values = [experts_touched(64, 6, b) for b in (1, 2, 4, 8, 16, 32, 64, 256)]
    assert values == sorted(values) and len(set(values)) == len(values)
    assert values[-1] > 63.99 and values[-1] < 64.0
    ds = load_model("deepseek-v2-lite")
    assert ds.experts_touched(1) == pytest.approx(6.0, rel=1e-12)
    assert ds.experts_touched(8) == pytest.approx(64 * (1 - (58 / 64) ** 8), rel=1e-12)
    assert LLAMA.experts_touched(8) == 0.0


def test_moe_linear_split_and_per_layer_timing():
    ds = load_model("deepseek-v2-lite")
    g = gpu()
    # Layer 0 is dense; layers 1..26 are MoE (64 routed + 2 shared, top-6).
    assert not ds.is_moe_layer(0) and ds.is_moe_layer(1)
    # At B=1, memory-bound on active experts (6 + 2 = 8); at B=32, touched experts (~59.6 + 2) drive weight bytes.
    t_b1 = linear_ns(ds.layer_post_active_params(1), 1, ds, g, ds.layer_post_touched_params(1, 1))
    t_b32 = linear_ns(ds.layer_post_active_params(1), 32, ds, g, ds.layer_post_touched_params(1, 32))
    assert t_b1 == pytest.approx(ds.layer_post_active_params(1) * 2 / g.bandwidth * 1e9, rel=1e-12)
    assert t_b32 == pytest.approx(ds.layer_post_touched_params(1, 32) * 2 / g.bandwidth * 1e9, rel=1e-12)
    assert t_b32 > 5 * t_b1  # touched experts grow sharply with B even though FLOP-active experts stay 8
    # At huge B, FLOP-active params dominate compute while touched experts saturate at E=64.
    t_huge = linear_ns(ds.layer_post_active_params(1), 100_000, ds, g, ds.layer_post_touched_params(1, 100_000))
    assert t_huge == pytest.approx(2 * 100_000 * ds.layer_post_active_params(1) / g.flops * 1e9, rel=1e-12)
    # Evaluate: layer 0 (dense) and layer 1 (MoE) get distinct post_ns.
    result = evaluate(rows(layers=ds.layers), ds, g, SystemOptions(), META | {"batch": 16, "stacks": 16})
    assert result.layers[0]["post_ns"] != result.layers[1]["post_ns"]
    assert result.layers[1]["post_ns"] == pytest.approx(result.layers[26]["post_ns"])
    assert any("MoE" in line for line in result.assumptions)
    assert any("MLA" in line for line in result.assumptions)


def test_mla_absorbed_decode_flops_and_fallback():
    ds = load_model("deepseek-v2-lite")
    assert ds.default_attn_dims == 2 * 512 + 64 == 1088
    # With attended_elements (from K1 mla format) and without (fallback via default_attn_dims).
    with_el = evaluate(rows(layers=ds.layers, attended_tokens=1024, attended_elements=1024 * 1088),
                       ds, gpu(), SystemOptions(), META)
    fallback = evaluate(rows(layers=ds.layers, attended_tokens=1024), ds, gpu(), SystemOptions(), META)
    expected = 2 * 16 * 1024 * 1088
    assert with_el.layers[0]["attn_flops"] == expected
    assert fallback.layers[0]["attn_flops"] == expected


def test_k3_end_to_end_evaluate_all_new_presets():
    from hbf.frontend.kv.tensors import Shape, default_tensors
    from hbf.backends.rounds.analysis import analyze_accesses
    from hbf.device.config import load
    from hbf.kvspec import GroupedLayout, from_model, generate_grouped

    cfg = load("ocp")
    b200 = load_gpu("b200")
    for name in ("deepseek-v2-lite", "gemma-3-27b", "gpt-oss-20b", "gpt-oss-120b", "qwen3-next-80b-a3b"):
        model = load_model(name)
        spec = from_model(model)
        shape = Shape(model.layers, model.kv_heads, model.head_dim, 2048)
        layout = GroupedLayout(cfg, shape, spec, default_tensors(shape))
        layer_rows, _ = analyze_accesses(cfg, layout, list(generate_grouped(cfg, layout, "dense")), timing=True)
        meta = layout.request_bytes(cfg) | {"batch": 8, "stacks": 8, "analysis_batch": 1,
                                            "selection": "static", "kv_format": spec.name}
        serial = evaluate(layer_rows, model, b200, SystemOptions("serial"), meta)
        prefetch = evaluate(layer_rows, model, b200, SystemOptions("prefetch:1"), meta)
        assert len(serial.layers) == model.layers
        assert serial.steps[0]["tpot_hbf_est_ns"] > 0 and serial.steps[0]["tpot_hbm_ns"] > 0
        assert prefetch.steps[0]["tpot_hbf_est_ns"] > 0
    # SSM layers have distinct pre_ns from full-attention layers on Qwen3-Next.
    qn = load_model("qwen3-next-80b-a3b")
    qn_res = evaluate(rows(layers=qn.layers), qn, gpu(), SystemOptions(), META)
    assert qn_res.layers[0]["pre_ns"] != qn_res.layers[3]["pre_ns"]  # layer 0 is S, layer 3 is A


def test_k3_model_spec_validation():
    base = dict(name="m", layers=4, hidden=256, q_heads=8, kv_heads=2, head_dim=64, intermediate=512, vocab=1000)
    for bad in (
        {"attention": "mha"},
        {"attention": "mla", "kv_lora_rank": 64, "rope_dim": 16},  # kv_heads != 1
        {"attention": "mla", "kv_heads": 1, "kv_lora_rank": 0, "rope_dim": 16},
        {"kv_lora_rank": 64},  # GQA with MLA field
        {"layer_pattern": "LX", "window": 128},
        {"layer_pattern": "LG"},  # missing window
        {"layer_pattern": "SSSA"},  # missing ssm_state_bytes
        {"window": 128},  # window without layer_pattern
        {"experts": 8, "top_k": 0, "expert_intermediate": 256},
        {"experts": 8, "top_k": 9, "expert_intermediate": 256},
        {"experts": 8, "top_k": 2, "expert_intermediate": 256, "dense_layers": 4},
        {"top_k": 2},  # MoE field on dense model
    ):
        with pytest.raises(ValueError):
            ModelSpec(**(base | bad))


def test_existing_presets_unchanged():
    # Regression guard: Llama and Qwen presets keep exact parameter counts and single-step TPOT.
    g = load_gpu("b200")
    for name in ("llama-3.1-8b", "llama-3.1-70b", "qwen2.5-7b", "qwen3-8b", "qwen3-32b"):
        m = load_model(name)
        assert m.total_params == m.layers * m.layer_params + m.embedding_params
        assert m.weights_bytes == (m.layers * m.layer_params + m.embedding_params) * m.weight_bits // 8
        res = evaluate(rows(layers=m.layers, est_ns=100_000, bound_ns=80_000, attended_tokens=4096,
                            burst_bytes=4096 * 256), m, g, SystemOptions(), META)
        expected_pre = max(2 * m.pre_params / g.flops, m.pre_params * m.weight_bytes / g.bandwidth) * 1e9
        expected_post = max(2 * m.post_params / g.flops, m.post_params * m.weight_bytes / g.bandwidth) * 1e9
        assert res.context["pre_ns"] == expected_pre and res.context["post_ns"] == expected_post
        assert not any("MoE" in line or "MLA" in line for line in res.assumptions)
