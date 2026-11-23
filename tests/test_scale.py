"""T9 and T10 are opt-in when using the fast pytest selection."""
from statistics import mean

import pytest

from hbf.access import Shape
from hbf.cli import pipeline
from hbf.config import load
from hbf.controller import Batch, Request
from hbf.simulator import dump, simulate


@pytest.mark.scale
@pytest.mark.parametrize("topology", ["16x64", "64x16", "256x4"])
@pytest.mark.parametrize("layout", ["layer_contiguous", "vllm_interleaved"])
def test_t9_dense_matrix(binary, evidence, topology, layout):
    cfg = load(topology)
    shape = Shape(32, 8, 128, 32768)
    directory = evidence / f"t9-{topology}-{layout}"
    batches, layers, steps, stats = pipeline(cfg, shape, layout, directory, binary)
    summary = {
        "topology": topology, "layout": layout,
        "mqsim_ns_per_layer": mean(row["mqsim_ns"] for row in layers),
        "min_layer_ns": min(row["mqsim_ns"] for row in layers),
        "max_layer_ns": max(row["mqsim_ns"] for row in layers),
        "mq_est_split_ns_per_layer": mean(row["mq_est_ns"] for row in layers),
        "mq_est_unsplit_ns_per_layer": mean(row["mq_est_unsplit_ns"] for row in layers),
        "ideal_split_ns_per_layer": mean(row["ideal_ns"] for row in layers),
        "ideal_unsplit_ns_per_layer": mean(row["ideal_unsplit_ns"] for row in layers),
        "max_unit_pages": max(row["max_unit_pages"] for row in layers),
        "mqsim_step_ns": steps[0]["mqsim_ns"], "ideal_step_ns": steps[0]["ideal_unsplit_ns"],
        **{key: stats[key] for key in ("wall_seconds", "peak_rss_kib")},
    }
    dump(directory / "summary.json", summary)
    assert len(layers) == 32 and len(batches) == 128 and len(steps) == 1
    assert all(row["read_pages"] == 32768 for row in layers)
    assert steps[0]["read_pages"] == 1048576
    assert stats["gc_enqueued"] == stats["Issued_Flash_Erase_CMD"] == 0
    if layout == "layer_contiguous":
        assert all(row["active_units"] == 1024 and row["max_unit_pages"] == 32 for row in layers)
        assert all(row["ideal_unsplit_ns"] == 131072 for row in layers)
    else:
        assert all(row["active_units"] == 32 and row["max_unit_pages"] == 1024 for row in layers)
        assert all(row["ideal_unsplit_ns"] == 4194304 for row in layers)


@pytest.mark.scale
@pytest.mark.parametrize("size", [1024, 4096, 8192, 16384, 32768])
def test_t10_performance(binary, evidence, size):
    cfg = load(batch_pages=size)
    directory = evidence / f"t10-{size}"
    _, _, _, stats = simulate(cfg, [Batch(0, 0, 0, 0, tuple(Request(i) for i in range(size)))],
                              directory, binary)
    dump(directory / "summary.json", {"pages": size, "wall_seconds": stats["wall_seconds"],
        "peak_rss_kib": stats["peak_rss_kib"], "seconds_per_million": stats["wall_seconds"] * 1e6 / size})
    assert stats["wall_seconds"] > 0 and stats["peak_rss_kib"] > 0


@pytest.mark.scale
def test_t10_64k_split_comparison(binary, evidence):
    cfg = load()
    pages = tuple(Request(i) for i in range(65536))
    unsplit = [Batch(0, 0, 0, 0, pages)]
    split = [Batch(0, 0, i // cfg.batch_pages, i // cfg.batch_pages * cfg.interval_ns,
                   pages[i:i + cfg.batch_pages])
             for i in range(0, len(pages), cfg.batch_pages)]
    result = {}
    for label, batches in (("64k-open-burst", unsplit), ("8x8k", split)):
        _, layers, _, stats = simulate(cfg, batches, evidence / f"t10-{label}", binary)
        result[label] = {"wall_seconds": stats["wall_seconds"], "peak_rss_kib": stats["peak_rss_kib"],
                         "reconstructed_service_ns": layers[0]["mqsim_ns"]}
    dump(evidence / "t10-split-summary.json", result)
    assert all(value["wall_seconds"] > 0 for value in result.values())
