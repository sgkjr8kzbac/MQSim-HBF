"""R4 acceptance test: copy hbf/ to a temp directory, delete hbf/ext/, and run core smoke tests."""
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HBF_SRC = ROOT / "hbf"


def _copy_core_without_ext(tmp_path):
    """Copy hbf/ into tmp_path/pkg/hbf, delete ext/, and symlink mqsim/ and .build/ for cache reuse."""
    pkg_root = tmp_path / "pkg"
    dst = pkg_root / "hbf"
    def _ignore(dirpath, names):
        skip = [n for n in names if n == "__pycache__" or n.endswith(".pyc")]
        if Path(dirpath) == HBF_SRC and "mqsim" in names:
            skip.append("mqsim")
        return skip

    shutil.copytree(HBF_SRC, dst, ignore=_ignore)
    (dst / "mqsim").symlink_to(HBF_SRC / "mqsim")
    if (ROOT / ".build").exists():
        (pkg_root / ".build").symlink_to(ROOT / ".build")
    ext_dir = dst / "ext"
    assert ext_dir.is_dir()
    shutil.rmtree(ext_dir)
    assert not ext_dir.exists()
    return pkg_root


def _run_core(pkg_root, args):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(pkg_root)
    env.pop("HBF_NO_EXT", None)
    return subprocess.run(
        [sys.executable, *args],
        cwd=str(pkg_root),
        env=env,
        capture_output=True,
        text=True,
    )


def test_core_runs_without_ext_directory(tmp_path):
    """Deleting hbf/ext/ must leave MQSim, rounds backend, and lifetime extrapolation working."""
    pkg_root = _copy_core_without_ext(tmp_path)

    # 0. Verify we imported the copied package (no ext/) and registry skips missing hbf.ext.
    probe = _run_core(pkg_root, ["-c", (
        "from pathlib import Path; import hbf, hbf.registry as r; "
        f"assert Path(hbf.__file__).resolve().is_relative_to(Path({str(pkg_root)!r}).resolve()), hbf.__file__; "
        "loaded = r.load_extensions(); "
        "assert loaded == [], loaded; "
        "assert 'bf16' in r.REGISTRY.formats and 'kivi2' in r.REGISTRY.formats; "
        "assert 'mla' not in r.REGISTRY.formats and 'tpot' not in r.REGISTRY.clocks; "
        "assert 'read_time' in r.REGISTRY.clocks"
    )])
    assert probe.returncode == 0, probe.stderr

    # 1. Small-topology MQSim smoke run (reuses ROOT/.build cache).
    out_mq = tmp_path / "mqsim-core"
    res_mq = _run_core(pkg_root, [
        "-m", "hbf", "run",
        "--topology", "4x1", "--planes", "4", "--plane-rule", "independent",
        "--read-plane-ns", "0", "--coalesce-ns", "1", "--cache-read",
        "--layers", "1", "--heads", "2", "--context", "256", "--blocks-per-plane", "32",
        "--output", str(out_mq),
    ])
    assert res_mq.returncode == 0, res_mq.stderr
    assert (out_mq / "layers.csv").is_file()
    assert (out_mq / "analysis_layers.csv").is_file()
    assert (out_mq / "addr.npz").is_file()
    assert (out_mq / "config.json").is_file()

    # 2. Rounds estimation with core model preset + core format (no ext/system).
    out_rd = tmp_path / "rounds-core"
    res_rd = _run_core(pkg_root, [
        "-m", "hbf", "run", "--analysis-only",
        "--model", "llama-3.1-8b", "--format", "kivi2", "--layout", "striped_contiguous",
        "--context", "4096", "--batch", "4", "--output", str(out_rd),
    ])
    assert res_rd.returncode == 0, res_rd.stderr
    assert (out_rd / "analysis_layers.csv").is_file()
    assert (out_rd / "analysis_steps.csv").is_file()
    assert not (out_rd / "system_steps.csv").exists()
    rows = list(csv.DictReader((out_rd / "analysis_layers.csv").open()))
    assert len(rows) == 32 and int(rows[0]["est_ns"]) > 0

    # 3. Lifetime extrapolation using core ReadTimeClock (no ext/system).
    out_lt = tmp_path / "lifetime-core"
    res_lt = _run_core(pkg_root, [
        "-m", "hbf", "lifetime", "--analytic", "--no-calibrate",
        "--model", "llama-3.1-8b", "--format", "bf16,kivi2",
        "--workload", "long32k", "--rd-threshold", "1e6", "--output", str(out_lt),
    ])
    assert res_lt.returncode == 0, res_lt.stderr
    lt_rows = list(csv.DictReader((out_lt / "endurance_summary.csv").open()))
    assert len(lt_rows) == 2 and all(float(r["avg_pec_per_day"]) > 0 for r in lt_rows)
    lt_cfg = json.loads((out_lt / "config.json").read_text())
    assert lt_cfg["ext"]["system"] is None

    # 4. Requesting any extension feature fails cleanly and names the owning extension.
    cases = [
        (["-m", "hbf", "run", "--analysis-only", "--format", "mla", "--heads", "1", "--context", "256"],
         "hbf.ext.kv_terms"),
        (["-m", "hbf", "run", "--analysis-only", "--format", "loki", "--context", "256"],
         "hbf.ext.kv_terms"),
        (["-m", "hbf", "run", "--analysis-only", "--mode", "channel_select", "--context", "256"],
         "hbf.ext.kv_terms"),
        (["-m", "hbf", "run", "--analysis-only", "--kv-spec", "hybrid:window=128", "--layers", "6", "--context", "512"],
         "hbf.ext.kv_terms"),
        (["-m", "hbf", "run", "--analysis-only", "--density", "1/4", "--context", "256"],
         "hbf.ext.kv_terms"),
        (["-m", "hbf", "run", "--analysis-only", "--alloc", "packed", "--frontiers", "class", "--batch", "2", "--context", "512"],
         "hbf.ext.kv_terms"),
        (["-m", "hbf", "run", "--analysis-only", "--model", "llama-3.1-8b", "--gpu", "b200", "--context", "256"],
         "hbf.ext.system"),
        (["-m", "hbf", "system", str(out_rd)],
         "hbf.ext.system"),
        (["-m", "hbf", "lifetime", "--analytic", "--no-calibrate", "--gpu", "b200"],
         "hbf.ext.system"),
    ]
    for argv, expected_ext in cases:
        out_bad = tmp_path / f"bad-{abs(hash(tuple(argv)))}"
        res = _run_core(pkg_root, argv + (["--output", str(out_bad)] if argv[2] != "system" else []))
        assert res.returncode != 0, f"Expected failure without ext for {argv}"
        assert expected_ext in res.stderr, f"Expected {expected_ext!r} in stderr for {argv}, got:\n{res.stderr}"
