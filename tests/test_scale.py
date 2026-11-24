"""T9 and T10 are opt-in when using the fast pytest selection."""
from statistics import mean

import pytest

from hbf.access import LAYOUTS, Shape, generate
from hbf.cli import pipeline
from hbf.config import load
from hbf.controller import Batch, Request, schedule
from hbf.simulator import dump, simulate


def run_dense_case(binary, evidence, topology, layout, expected_mq_est_ns):
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
    assert all(row["read_pages"] == 8192 for row in batches)
    assert all(row["read_pages"] == 32768 for row in layers)
    assert steps[0]["read_pages"] == 1048576
    assert stats["gc_enqueued"] == stats["Issued_Flash_Erase_CMD"] == 0
    # Independent checks: each estimator-1 term is a serial resource MQSim must respect, and
    # the ideal bound drops the command and transfer overheads, so ideal <= estimate <= MQSim.
    for row in (*batches, *layers, *steps):
        assert row["ideal_ns"] <= row["mq_est_ns"] <= row["mqsim_ns"]
    if layout == "layer_contiguous":
        assert all(row["active_units"] == 1024 and row["max_unit_pages"] == 32 for row in layers)
        ideal_layer_ns = 131_072
    else:
        assert all(row["active_units"] == 32 and row["max_unit_pages"] == 1024 for row in layers)
        ideal_layer_ns = 4_194_304
    assert all(row["mq_est_ns"] == expected_mq_est_ns // 4 for row in batches)
    assert all(row["ideal_ns"] == ideal_layer_ns // 4 for row in batches)
    expected_layer = {
        "mq_est_ns": expected_mq_est_ns,
        "mq_est_unsplit_ns": expected_mq_est_ns - 3,  # One host constant vs. four.
        "ideal_ns": ideal_layer_ns,
        "ideal_unsplit_ns": ideal_layer_ns,
    }
    for key, expected_ns in expected_layer.items():
        assert all(row[key] == expected_ns for row in layers)
    for key in ("mqsim_ns", *expected_layer):
        assert steps[0][key] == sum(row[key] for row in layers)
    return summary


@pytest.mark.scale
@pytest.mark.parametrize(
    ("topology", "contiguous_baseline", "interleaved_baseline"),
    [
        ("16x64", (738_448, 724_996), (4_578_760, 4_577_284)),
        ("64x16", (288_080, 279_556), (4_776_136, 4_773_892)),
        ("256x4", (273_088, 173_764), (6_608_960, 5_560_324)),
    ],
    ids=["16x64", "64x16", "256x4"],
)
def test_t9_dense_matrix(binary, evidence, topology, contiguous_baseline, interleaved_baseline):
    contiguous_ns, contiguous_est_ns = contiguous_baseline
    interleaved_ns, interleaved_est_ns = interleaved_baseline
    contiguous = run_dense_case(binary, evidence, topology, "layer_contiguous", contiguous_est_ns)
    interleaved = run_dense_case(binary, evidence, topology, "vllm_interleaved", interleaved_est_ns)
    # Regression baseline only: these values were measured on the pinned build, not derived.
    for summary, expected_ns in ((contiguous, contiguous_ns), (interleaved, interleaved_ns)):
        assert summary["min_layer_ns"] == pytest.approx(expected_ns, rel=.05)
        assert summary["max_layer_ns"] == pytest.approx(expected_ns, rel=.05)
    measured_ratio = interleaved["mqsim_ns_per_layer"] / contiguous["mqsim_ns_per_layer"]
    assert measured_ratio == pytest.approx(interleaved_ns / contiguous_ns, rel=.10)


@pytest.mark.scale
@pytest.mark.parametrize("size", [1024, 4096, 8192, 16384, 32768])
def test_t10_performance(binary, evidence, size):
    cfg = load()
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


# Pre-review reference values for one unsplit 32,768-page layer (context summary, section 4.4).
PRE_REVIEW_UNSPLIT_NS = {("16x64", "layer_contiguous"): 728_000, ("256x4", "layer_contiguous"): 272_000}


@pytest.mark.scale
@pytest.mark.parametrize("layout", ["layer_contiguous", "vllm_interleaved"])
@pytest.mark.parametrize("topology", ["16x64", "64x16", "256x4"])
def test_t9_unsplit_layer(binary, evidence, topology, layout):
    """Submit layer 0 as one open-loop burst to quantify the cost of 8K sub-batching."""
    cfg = load(topology)
    pages = LAYOUTS[layout](cfg, Shape(32, 8, 128, 32768))
    split = schedule(cfg, pages, (a for a in generate(cfg, pages, "dense") if a.layer == 0))
    requests = tuple(req for batch in split for req in batch.requests)
    assert len(split) == 4 and len(requests) == 32768
    directory = evidence / f"t9-unsplit-{topology}-{layout}"
    rows, _, _, stats = simulate(cfg, [Batch(0, 0, 0, 0, requests)], directory, binary, timeout=1800)
    split_rows, _, _, _ = simulate(cfg, split, evidence / f"t9-split-{topology}-{layout}", binary)
    unsplit_ns = rows[0]["mqsim_ns"]
    split_ns = sum(row["mqsim_ns"] for row in split_rows)
    dump(directory / "summary.json", {
        "topology": topology, "layout": layout, "unsplit_ns": unsplit_ns, "split_ns": split_ns,
        "split_over_unsplit": split_ns / unsplit_ns, "mq_est_unsplit_ns": rows[0]["mq_est_ns"],
        "ideal_unsplit_ns": rows[0]["ideal_ns"], "wall_seconds": stats["wall_seconds"],
        "peak_rss_kib": stats["peak_rss_kib"],
    })
    assert rows[0]["ideal_ns"] <= rows[0]["mq_est_ns"] <= unsplit_ns
    reference = PRE_REVIEW_UNSPLIT_NS.get((topology, layout))
    if reference is not None:
        assert unsplit_ns == pytest.approx(reference, rel=.05)
