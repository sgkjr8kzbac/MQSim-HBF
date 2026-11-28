"""Sweep driver (hbf/sweep.py): grid expansion, infeasible points, summaries and parallel runs."""
import csv
import io
import json
from collections import Counter

import pytest

from hbf.sweep import expand, load_grid, main, sweep, to_argv

TINY = {"name": "tiny", "layers": 2, "hidden": 256, "q_heads": 4, "kv_heads": 2, "head_dim": 64,
        "intermediate": 512, "vocab": 1000}
BASE = {"topology": "ocp", "model": TINY, "gpu": "h100-sxm", "context": 2048, "steps": 2}


def read_csv(path, group="1"):
    with path.open(newline="") as stream:
        return [row for row in csv.DictReader(stream) if row.get("group", group) == group]


def number(text):
    return None if text == "" else float(text)


def test_product_zip_seeds_expansion():
    grid = load_grid({"base": {"batch": 8, "context": 1024},
                      "vary": {"batch": [1, 2], "format": ["bf16", "kivi2"]},
                      "zip": [{"topology": "ocp-8ch", "aggregate_gb_s": 384, "ucie_grade": 1},
                              {"topology": "ocp", "aggregate_gb_s": 1536, "ucie_grade": 2}],
                      "seeds": [11, 12, 13]})
    runs = expand(grid)
    assert len(runs) == 2 * 2 * 2 * 3
    params, varied = runs[0]
    assert varied == {"batch": 1, "format": "bf16", "topology": "ocp-8ch", "aggregate_gb_s": 384,
                      "ucie_grade": 1, "seed": 11}
    assert params == {"batch": 1, "context": 1024, "format": "bf16", "topology": "ocp-8ch",
                      "aggregate_gb_s": 384, "ucie_grade": 1, "seed": 11}
    # Seeds vary fastest, then zip entries, then the product (last vary key fastest).
    assert [runs[i][1]["seed"] for i in range(3)] == [11, 12, 13]
    assert runs[3][1]["aggregate_gb_s"] == 1536
    assert runs[6][1]["format"] == "kivi2" and runs[12][1]["batch"] == 2
    assert len({json.dumps(varied, sort_keys=True) for _, varied in runs}) == 24
    assert expand(load_grid({"base": {"context": 1024}})) == [({"context": 1024}, {})]
    for bad in ({"vary": {"batch": 1}}, {"vary": {"no_such_flag": [1]}}, {"base": {"output": "x"}},
                {"vary": {"batch": [1]}, "zip": [{"batch": 2}]}, {"vary": {"seed": [1]}, "seeds": [2]},
                {"seeds": [1.5]}, {"bases": {}}):
        with pytest.raises(ValueError):
            load_grid(bad)


def test_to_argv():
    argv = to_argv({"prefill": True, "physical": False, "access_csv": None, "plane_group": [1, 4],
                    "model": {"name": "m"}, "context": 2048, "fraction": .5, "format": "kivi2:residual=64"})
    assert argv == ["--prefill", "--plane-group", "1", "--plane-group", "4", "--model", '{"name": "m"}',
                    "--context", "2048", "--fraction", "0.5", "--format", "kivi2:residual=64"]


def test_invalid_points_are_infeasible_rows(tmp_path):
    grid = {"base": {"layers": 1, "heads": 2, "head_dim": 64, "context": 1024},
            "vary": {"layout": ["layer_contiguous", "no_such_layout"]},
            "zip": [{"format": "bf16"}, {"format": "fp3"}, {"context": 0},
                    {"topology": "1x4", "blocks_per_plane": 1, "pages_per_block": 4}]}
    rows = sweep(grid, tmp_path / "sweep", progress=io.StringIO())
    status = {(row["layout"], row["format"], row["context"], row["topology"]): row["status"] for row in rows}
    assert status["layer_contiguous", "bf16", None, None] == "ok"
    assert status["no_such_layout", "bf16", None, None].startswith("infeasible: argument --layout: invalid choice")
    assert status["layer_contiguous", "fp3", None, None].startswith("infeasible: Unknown KV format")
    assert status["layer_contiguous", None, 0, None] == ("infeasible: Model dimensions and context must be "
                                                         "positive integers")
    assert status["layer_contiguous", None, None, "1x4"] == "infeasible: LPA 127 exceeds logical capacity"
    assert rows[0]["run"] == "0-layer_contiguous-bf16"
    summary = read_csv(tmp_path / "sweep" / "summary.csv")
    assert len(summary) == 8 and sum(row["status"] == "ok" for row in summary) == 1
    assert all(row["est_ns"] == "" for row in summary if row["status"] != "ok")


def test_summary_equals_run_csv_values(tmp_path):
    grid = {"base": BASE | {"mode": "uniform_pages", "fraction": .5},
            "vary": {"batch": [1, 2], "format": ["bf16", "kivi2"]}, "seeds": [3]}
    rows = sweep(grid, tmp_path / "sweep", progress=io.StringIO())
    assert [row["status"] for row in rows] == ["ok"] * 4
    summary = read_csv(tmp_path / "sweep" / "summary.csv")
    assert [row["format"] for row in summary] == ["bf16", "kivi2", "bf16", "kivi2"]
    assert {row["seed"] for row in summary} == {"3"}
    for row in summary:
        run = tmp_path / "sweep" / "runs" / row["run"]
        steps = read_csv(run / "analysis_steps.csv")
        assert len(steps) == 2
        for key in ("pages", "rounds", "min_rounds", "channel_rounds", "est_ns", "est_batches_ns", "bound_ns",
                    "t_balanced_ns", "t_imbalance_ns", "t_protocol_ns", "t_channel_ns", "t_overhead_ns",
                    "useful_bytes", "burst_bytes", "hbm_bytes", "attended_tokens", "attended_elements", "dequant_ops",
                    "utilization"):
            assert number(row[key]) == pytest.approx(sum(float(step[key]) for step in steps) / 2), key
        assert row["protocol_rounds"] == ""  # no protocol flags
        assert int(row["max_block_reads"]) == max(int(step["max_block_reads"]) for step in steps)
        layers = read_csv(run / "analysis_layers.csv")
        assert row["bottleneck"] == Counter(layer["bottleneck"] for layer in layers).most_common(1)[0][0]
        system = read_csv(run / "system_steps.csv")
        for key in ("tpot_hbf_est_ns", "tpot_hbf_bound_ns", "tpot_hbm_ns", "tokens_per_s_hbf", "tokens_per_s_hbm",
                    "hbf_over_hbm", "hbf_wait_share"):
            assert number(row[key]) == pytest.approx(sum(float(step[key]) for step in system) / 2), key
        for key in ("hbm_fits", "hbm_fits_hbf"):
            assert row[key] == str(all(step[key] == "True" for step in system))
        assert number(row["effective_gb_s"]) == pytest.approx(
            number(row["hbf_elements"]) * 2 / number(row["est_ns"]))
        meta = json.loads((run / "meta.json").read_text())
        assert "--analysis-only" in meta["argv"] and meta["system"]["batch"] == int(row["batch"])
    # A larger batch attends more tokens on the same stack.
    assert number(summary[2]["attended_tokens"]) > number(summary[0]["attended_tokens"])


def test_jobs_two_matches_jobs_one(tmp_path):
    grid = {"base": BASE, "vary": {"batch": [1, 2], "format": ["bf16", "kivi2"]},
            "zip": [{"overlap": "serial"}, {"overlap": "prefetch:1"}]}
    sweep(grid, tmp_path / "one", jobs=1, progress=io.StringIO())
    sweep(grid, tmp_path / "two", jobs=2, progress=io.StringIO())

    def table(name):
        rows = read_csv(tmp_path / name / "summary.csv")
        return sorted(({key: value for key, value in row.items() if key != "run"} for row in rows),
                      key=lambda row: int(row["index"]))

    one = table("one")
    assert one == table("two") and len(one) == 8
    assert all(row["status"] == "ok" for row in one)


def test_no_keep_and_skip_access_csv(tmp_path, capsys):
    grid = tmp_path / "grid.json"
    grid.write_text(json.dumps({"base": BASE, "vary": {"batch": [1, 2]}}))
    summary = main([str(grid), "--output", str(tmp_path / "gone"), "--no-keep"])
    assert summary == tmp_path / "gone" / "summary.csv"
    assert not (tmp_path / "gone" / "runs").exists()
    assert [row["status"] for row in read_csv(summary)] == ["ok", "ok"]
    out, err = capsys.readouterr()
    assert out.strip() == str(summary.resolve()) and "[2/2]" in err
    main([str(grid), "--output", str(tmp_path / "kept"), "--skip-access-csv"])
    runs = sorted((tmp_path / "kept" / "runs").iterdir())
    assert len(runs) == 2 and all((run / "analysis_steps.csv").is_file() for run in runs)
    assert not any((run / "access.csv").exists() for run in runs)
    for argv in (["--output", str(tmp_path / "kept")],  # already holds a sweep
                 ["--output", str(tmp_path / "jobs"), "--jobs", "4"]):
        with pytest.raises(SystemExit):
            main([str(grid), *argv])


def test_plane_groups_without_one_are_infeasible(tmp_path):
    from hbf.sweep import sweep
    grid = {"base": {"topology": "ocp", "layers": 1, "context": 1024}, "vary": {"plane_group": [[16], [1, 16]]}}
    rows = sweep(grid, tmp_path / "g", progress=open("/dev/null", "w"))
    assert rows[0]["status"].startswith("infeasible") and "plane group 1" in rows[0]["status"]
    assert rows[1]["status"] == "ok" and rows[1]["rounds"] is not None


def test_effective_bandwidth_leaves_out_the_hbm_tier(tmp_path):
    from hbf.sweep import sweep
    grid = {"base": {"topology": "ocp", "read_plane_ns": 0, "coalesce_ns": 1, "cache_read": True, "layers": 1,
                     "context": 2048, "format": "kivi2"}}
    (row,) = sweep(grid, tmp_path / "k", progress=open("/dev/null", "w"))
    assert row["hbf_elements"] == 8 * (1920 * 128 * 2)  # body tokens of K and V; the residual is in HBM
    assert row["attended_elements"] == 8 * 2048 * 256
    assert row["effective_gb_s"] == pytest.approx(row["hbf_elements"] * 2 / row["est_ns"])


ENDURANCE = {"tool": "endurance",
             "base": {"analytic": True, "no_calibrate": True, "model": "llama-3.1-8b", "pe_limit": "1e4,1e5"},
             "vary": {"workload": ["chat", "long32k"], "rd_threshold": [1e5, 1e6]}}


def test_endurance_tool_summary(tmp_path):
    rows = sweep(ENDURANCE, tmp_path / "sweep")
    assert [row["status"] for row in rows] == ["ok"] * 4
    table = read_csv(tmp_path / "sweep" / "summary.csv")
    assert [(row["workload"], float(row["rd_threshold"])) for row in table] == [
        ("chat", 1e5), ("chat", 1e6), ("long32k", 1e5), ("long32k", 1e6)]
    for row in table:  # the summary equals the run's own endurance_summary.csv
        run = read_csv(tmp_path / "sweep" / "runs" / row["run"] / "endurance_summary.csv")
        assert len(run) == 1
        for key in ("avg_pec_per_day", "kv_write_gb_s", "days_to_limit_avg@1e4", "days_to_limit_avg@1e5"):
            assert number(row[key]) == pytest.approx(number(run[0][key]))
        assert number(row["days_to_limit_avg@1e4"]) * 10 == pytest.approx(number(row["days_to_limit_avg@1e5"]))
    meta = json.loads((tmp_path / "sweep" / "runs" / table[0]["run"] / "meta.json").read_text())
    assert meta["tool"] == "endurance" and meta["argv"][0] == "hbf.endurance"


def test_endurance_tool_checks(tmp_path):
    with pytest.raises(ValueError, match="hbf.endurance long options"):
        load_grid({"tool": "endurance", "base": {"layout": "striped_contiguous"}})
    with pytest.raises(ValueError, match="set by the sweep"):
        load_grid({"tool": "endurance", "base": {"output": "x"}})
    with pytest.raises(ValueError, match="Unknown tool"):
        load_grid({"tool": "mqsim"})
    assert load_grid({"base": {"batch": 2}})["tool"] == "analysis"
    grid = {"tool": "endurance", "base": {"analytic": True, "no_calibrate": True},
            "vary": {"workload": ["chat,long32k", "chat"]}}
    rows = sweep(grid, tmp_path / "sweep")
    assert rows[0]["status"].startswith("infeasible: the point gives 2 rows")
    assert rows[1]["status"] == "ok"
    assert not (tmp_path / "sweep" / "runs" / rows[0]["run"]).exists()


def test_endurance_sweep_jobs_two_matches_jobs_one(tmp_path):
    # L6/M7: parallel endurance sweep with shared calibration directory matches jobs=1.
    one = sweep(ENDURANCE, tmp_path / "one", jobs=1, progress=io.StringIO())
    two = sweep(ENDURANCE, tmp_path / "two", jobs=2, progress=io.StringIO())
    assert [row["status"] for row in two] == ["ok"] * 4
    assert one == two


def test_workload_length_validation_and_describe():
    # M5 & L5: parse_length rejects lo > hi, min > max, median <= 0; describe omits None history.
    from hbf.workload import Workload, parse_length
    for bad in ("uniform:lo=100,hi=50", "lognormal:median=0,sigma=0.8",
                "lognormal:median=100,sigma=0.8,min=50,max=20",
                "lognormal:median=100,sigma=0.8,max=0", "uniform:lo=10,hi=20,min=30,max=10"):
        with pytest.raises(ValueError):
            parse_length(bad)
    session = Workload(name="s", calls=4, history=None, think_s=1.0)
    desc = session.describe()
    assert "None" not in desc and "会话 4 次调用" in desc
    with_hist = Workload(name="s", calls=4, history="fixed:512", think_s=1.0)
    assert "初始历史 fixed:512" in with_hist.describe()
