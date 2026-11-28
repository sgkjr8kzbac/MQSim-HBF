"""KV layer and head groups (hbf/kvspec.py, T-K2): hand-computed values of design section 3, K2."""
import csv
import json

import pytest

from hbf.access import LAYOUTS, Access, Shape, default_tensors, generate
from hbf.analysis import analyze_accesses
from hbf.cli import main
from hbf.config import load
from hbf.kvspec import GroupedLayout, KVGroup, KVSpec, generate_grouped, parse_kvspec
from hbf.pages import iter_layer_pages

HBF_FLAGS = ["--analysis-only", "--topology", "ocp", "--read-plane-ns", "0", "--coalesce-ns", "1", "--cache-read"]


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
    assert layout.request_bytes(cfg) == {"kv_bytes_per_request": 2 * 33_554_432,
                                         "hbm_tier_bytes_per_request": 33_554_432}


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
    for extra in (["--layout", "vllm_interleaved"], ["--mode", "uniform_pages"], ["--alloc", "packed"]):
        with pytest.raises(SystemExit):
            main(HBF_FLAGS + ["--kv-spec", "cla", "--layers", "2", "--context", "1024", *extra,
                              "--output", str(tmp_path / "x")])
