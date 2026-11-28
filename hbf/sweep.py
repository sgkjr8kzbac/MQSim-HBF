"""Parameter sweeps of the analysis-only pipeline or the endurance model: ``python -m hbf.sweep GRID.json --output DIR``.

GRID.json::

    {"tool": "analysis",
     "base": {"topology": "ocp", "model": "llama-3.1-8b", "context": 32768},
     "vary": {"batch": [1, 4], "format": ["bf16", "kivi2"]},
     "zip": [{"aggregate_gb_s": 384, "ucie_grade": 1}, {"aggregate_gb_s": 1536, "ucie_grade": 2}],
     "seeds": [11, 12, 13]}

Keys are ``hbf.cli`` long options with dashes replaced by underscores: ``true`` turns on a flag,
``false`` or ``null`` leaves it out, a list of scalars repeats the option (``plane_group``), and a
JSON object (or a list of objects) is passed as JSON text (``model``, ``gpu``, ``tensors``). The runs
are the Cartesian product of ``vary``, times each ``zip`` entry, times each seed; later parts
override ``base``. Every run is parsed and validated by ``hbf.cli`` itself and runs with
``--analysis-only --timing`` in DIR/runs/<index>-<slug>; MQSim never runs. DIR/summary.csv has one
row per run (see ``summarize_run``); a run whose arguments or layout are rejected (ValueError) is
recorded as ``infeasible: <message>``.

``"tool": "endurance"`` sweeps ``hbf.endurance`` instead (keys are its long options, e.g.
``analytic``, ``workload``, ``rd_threshold``). Each point must be one combination of model, format,
workload, pages per block and X (vary them in the grid rather than as comma lists); a comma list of
``pe_limit`` is fine and adds one ``days_to_limit_*`` column per limit. The runs share the calibration
cache DIR/calibration unless the grid sets ``calibration_dir``. The summary has ``hbf.endurance.METRICS``
plus the ``days_to_limit_*`` columns.
"""
import argparse
import csv
import itertools
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
import re
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .backends.rounds.analysis import TIMING_KEYS
from .cli import build_parser, configure, pipeline
from .util import write_table

MAX_JOBS = 3
GRID_KEYS = ("tool", "base", "vary", "zip", "seeds")
T_KEYS = tuple(key for key in TIMING_KEYS if key.startswith("t_"))
STEP_KEYS = ("pages", "rounds", "min_rounds", "channel_rounds", "protocol_rounds", "est_ns", "est_batches_ns",
             "bound_ns") + T_KEYS + ("useful_bytes", "burst_bytes", "hbm_bytes", "attended_tokens",
                                     "attended_elements", "hbf_elements", "dequant_ops", "utilization")
SYSTEM_KEYS = ("tpot_hbf_est_ns", "tpot_hbf_bound_ns", "tpot_hbm_ns", "tokens_per_s_hbf", "tokens_per_s_hbm",
               "hbf_over_hbm", "hbf_wait_share")
FIT_KEYS = ("hbm_fits", "hbm_fits_hbf")
WRITE_KEYS = ("write_legal", "write_skips", "write_backward", "write_overwrites", "write_live_erases", "decode_pages",
              "padding_pages")
METRICS = (STEP_KEYS + ("max_block_reads", "bottleneck") + SYSTEM_KEYS + FIT_KEYS + ("effective_gb_s",)
           + WRITE_KEYS)


def _analysis_check(args):
    if args.plane_groups and 1 not in args.plane_groups:
        raise ValueError("the summary reads plane group 1; include 1 in plane_group")


def _analysis_execute(directory, run):
    pipeline(directory=directory, binary=None, **run)
    return summarize_run(directory, run["shape"].head_dim)


def _endurance():
    from .lifetime import cli
    return cli


def _endurance_check(args):
    pass


def _endurance_points(run):
    """Rows one endurance run writes; the sweep needs exactly one per point."""
    count = 1
    for key in ("models", "formats", "workloads", "pages_per_block", "x_values"):
        count *= len(run.get(key) or [None])
    return count


def _endurance_execute(directory, run):
    endurance = _endurance()
    count = _endurance_points(run)
    if count != 1:
        raise ValueError(f"the point gives {count} rows; vary model, format, workload, pages_per_block and "
                         "rd_threshold in the grid instead of comma lists")
    endurance.run(directory, **run)
    row = endurance.summarize_run(directory)
    with (Path(directory) / "endurance_summary.csv").open(newline="") as stream:
        first = next(csv.DictReader(stream), None) or {}
    row |= {key: _parse(value) for key, value in first.items() if key.startswith("days_to_limit_")}
    return row


@dataclass(frozen=True)
class Tool:
    """One program the sweep can drive: its parser, configuration, run and summary columns."""
    module: str
    parser: Callable
    configure: Callable
    check: Callable  # extra validation of the parsed arguments
    execute: Callable  # (directory, configuration) -> summary row
    metrics: Callable  # () -> summary columns
    flags: tuple  # appended to every run's arguments
    reserved: tuple  # options set by the sweep itself


TOOLS = {
    "analysis": Tool("hbf.cli", build_parser, configure, _analysis_check, _analysis_execute, lambda: METRICS,
                     ("--analysis-only",), ("output", "help", "analysis_only")),
    "endurance": Tool("hbf.lifetime.cli", lambda: _endurance().build_parser(), lambda args: _endurance().configure(args),
                      _endurance_check, _endurance_execute, lambda: _endurance().METRICS, (), ("output", "help")),
}


def tool_of(grid):
    name = grid.get("tool") or "analysis"
    if name not in TOOLS:
        raise ValueError(f"Unknown tool {name!r}: choose {', '.join(TOOLS)}")
    return TOOLS[name]


def load_grid(text):
    """Grid from a dict, JSON text or a JSON file, with option names checked against the tool's parser."""
    if isinstance(text, dict):
        grid = text
    else:
        try:
            grid = json.loads(text)
        except json.JSONDecodeError:
            with open(text) as stream:
                grid = json.load(stream)
    if not isinstance(grid, dict) or set(grid) - set(GRID_KEYS):
        raise ValueError(f"A grid is a JSON object with keys {', '.join(GRID_KEYS)}")
    base, vary, zipped, seeds = (grid.get("base") or {}, grid.get("vary") or {}, grid.get("zip") or [],
                                 grid.get("seeds"))
    if not isinstance(base, dict) or not isinstance(vary, dict):
        raise ValueError("base and vary must be JSON objects")
    if not all(isinstance(values, list) and values for values in vary.values()):
        raise ValueError("Every vary entry must be a non-empty list")
    if not isinstance(zipped, list) or not all(isinstance(entry, dict) and entry for entry in zipped):
        raise ValueError("zip must be a list of non-empty objects")
    if seeds is not None and (not isinstance(seeds, list) or not seeds
                              or not all(type(seed) is int for seed in seeds)):
        raise ValueError("seeds must be a non-empty list of integers")
    zip_keys = {key for entry in zipped for key in entry}
    if set(vary) & zip_keys:
        raise ValueError(f"Keys in both vary and zip: {', '.join(sorted(set(vary) & zip_keys))}")
    if seeds is not None and "seed" in set(vary) | zip_keys:
        raise ValueError("Give the seed in seeds or in vary/zip, not both")
    tool = tool_of(grid)
    options = tool.parser()._option_string_actions
    for key in set(base) | set(vary) | zip_keys:
        if key in tool.reserved:
            raise ValueError(f"{key} is set by the sweep")
        if not isinstance(key, str) or flag(key) not in options:
            raise ValueError(f"Unknown option {key!r}: use {tool.module} long options with _ for -")
    return {"tool": grid.get("tool") or "analysis", "base": base, "vary": vary, "zip": zipped, "seeds": seeds}


def flag(key):
    return "--" + key.replace("_", "-")


def expand(grid):
    """Runs as (params, varied) pairs: product of vary, times each zip entry, times each seed."""
    seeds = [{"seed": seed} for seed in grid["seeds"]] if grid["seeds"] is not None else [{}]
    runs = []
    for values in itertools.product(*grid["vary"].values()):
        for entry in grid["zip"] or [{}]:
            for seed in seeds:
                varied = dict(zip(grid["vary"], values)) | entry | seed
                runs.append((grid["base"] | varied, varied))
    return runs


def to_argv(params):
    """hbf.cli arguments for a dict of option values."""
    argv = []
    for key, value in params.items():
        if value is None or value is False:
            continue
        if value is True:
            argv.append(flag(key))
        elif isinstance(value, dict) or (isinstance(value, list) and any(isinstance(v, (dict, list)) for v in value)):
            argv += [flag(key), json.dumps(value)]
        elif isinstance(value, list):
            for item in value:
                argv += [flag(key), str(item)]
        else:
            argv += [flag(key), str(value)]
    return argv


def cell(value):
    """Summary cell of a varied value: scalars as they are, anything else as JSON."""
    return value if value is None or isinstance(value, (str, int, float)) else json.dumps(value)


def slug(varied, limit=48):
    text = "-".join(str(cell(value)) for value in varied.values())
    return re.sub(r"[^A-Za-z0-9._=+]+", "-", text).strip("-")[:limit] or "base"


def _raise(message):
    raise ValueError(message)


def _status(kind, error):
    return " ".join(f"{kind}: {error}".split())


def _group1(path):
    """Group 1 rows of an analysis CSV (all rows when it has no group column), fields parsed."""
    if not path.is_file():
        return []
    with path.open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row.get("group", "1") == "1"]
    return [{key: _parse(value) for key, value in row.items()} for row in rows]


def _parse(text):
    if text is None or text == "":
        return None
    if text in ("True", "False"):
        return text == "True"
    for kind in (int, float):
        try:
            return kind(text)
        except ValueError:
            pass
    return text


def _mean(values):
    """Mean over steps; None when there are no steps or any step lacks the value; exact ints stay ints."""
    if not values or any(value is None for value in values):
        return None
    total = sum(values)
    if all(type(value) is int for value in values) and total % len(values) == 0:
        return total // len(values)
    return total / len(values)


def summarize_run(directory, head_dim):
    """Summary columns of one finished run directory.

    Values from analysis_steps.csv (plane group 1) and system_steps.csv are means over the run's
    decode steps, except ``max_block_reads`` (maximum over steps) and ``hbm_fits``/``hbm_fits_hbf``
    (true only if every step fits). A mean is empty when any step lacks the value, e.g. an
    estimate that does not exist for the topology. ``bottleneck`` is the most common per-layer
    bottleneck of group 1 over all steps. ``effective_gb_s`` = hbf_elements x 2 B (the K and V
    elements attention uses that come from HBF, as BF16-equivalent logical bytes, of one stack's
    requests) / est_ns, i.e. bytes per ns; HBM-tier elements (a KIVI residual window) are left out
    because est_ns is the HBF read time. Runs without the column fall back to attended_elements,
    then attended_tokens x 2 x head_dim.
    """
    steps = _group1(directory / "analysis_steps.csv")
    row = {key: _mean([step.get(key) for step in steps]) for key in STEP_KEYS}
    blocks = [step.get("max_block_reads") for step in steps]
    row["max_block_reads"] = None if not blocks or None in blocks else max(blocks)
    counts = Counter(layer["bottleneck"] for layer in _group1(directory / "analysis_layers.csv")
                     if layer.get("bottleneck") is not None)
    row["bottleneck"] = counts.most_common(1)[0][0] if counts else None
    system = _group1(directory / "system_steps.csv")
    row |= {key: _mean([step.get(key) for step in system]) for key in SYSTEM_KEYS}
    for key in FIT_KEYS:
        fits = [step.get(key) for step in system]
        row[key] = None if not fits or None in fits else all(fits)
    est, tokens = row["est_ns"], row["attended_tokens"]
    elements = row.get("hbf_elements")
    if elements is None:
        elements = row.get("attended_elements")
    if elements is None and tokens is not None:
        elements = tokens * 2 * head_dim
    row["effective_gb_s"] = None if not est or elements is None else elements * 2 / est
    meta = directory / "meta.json"
    check = ((json.loads(meta.read_text()).get("batch") or {}).get("write_check") if meta.exists() else None)
    if check:  # --check-writes: the write order verdict of hbf.batch.check_write_order
        row |= {"write_legal": check["legal"], "write_skips": check["skips"], "write_backward": check["backward"],
                "write_overwrites": check["overwrites"], "write_live_erases": check["live_erases"],
                "decode_pages": check["kinds"]["decode"],
                "padding_pages": check["kinds"]["padding"]}
    return row


def run_one(task, tool_name="analysis"):
    """Parse, validate and run one point (analysis-only pipeline or endurance model); returns (row, seconds)."""
    started = time.monotonic()
    index, name, params, varied, output, keep, skip_access = task
    tool = TOOLS[tool_name]
    directory = output / "runs" / name
    row = {"index": index, "run": name} | {key: cell(value) for key, value in varied.items() if key != "seed"}
    row |= {"seed": params.get("seed"), "status": "ok"} | dict.fromkeys(tool.metrics())
    if tool_name == "endurance" and params.get("calibration_dir") is None:
        params = params | {"calibration_dir": str(output / "calibration")}  # one cache for the whole sweep
    argv = to_argv(params) + list(tool.flags) + ["--output", str(directory)]
    parser = tool.parser()
    parser.error = _raise  # argument errors become ValueError instead of exiting
    try:
        args = parser.parse_args(argv)
        row["seed"] = args.seed
        tool.check(args)
        run = tool.configure(args)
    except (OSError, TypeError, ValueError) as error:  # as hbf.cli.main
        row["status"] = _status("infeasible", error)
        return row, time.monotonic() - started
    saved, sys.argv = sys.argv, [tool.module, *argv]  # meta.json records the run's own arguments
    try:
        if tool_name == "analysis" and (skip_access or not keep):
            run = run | {"skip_trace_files": True}
        row |= tool.execute(directory, run)
    except ValueError as error:
        row["status"] = _status("infeasible", error)
    except Exception as error:  # keep the sweep going; reported at the end
        row["status"] = _status("error", f"{type(error).__name__}: {error}")
    finally:
        sys.argv = saved
    if directory.is_dir():
        if not keep:
            shutil.rmtree(directory)
        elif skip_access:
            for fname in ("access.csv", "addr.npz", "layers.npz", "addr.json"):
                (directory / fname).unlink(missing_ok=True)
    return row, time.monotonic() - started


def sweep(grid, output, jobs=1, keep=True, skip_access=False, progress=None):
    """Run every point of the grid into ``output`` and write output/summary.csv; returns the rows."""
    if type(jobs) is not int or not 1 <= jobs <= MAX_JOBS:
        raise ValueError(f"jobs must be 1 to {MAX_JOBS}")
    grid = load_grid(grid)
    tool = tool_of(grid)
    runs = expand(grid)
    output = Path(output)
    if (output / "runs").exists() or (output / "summary.csv").exists():
        raise ValueError(f"{output} already holds a sweep; choose a new --output")
    (output / "runs").mkdir(parents=True)
    (output / "grid.json").write_text(json.dumps(grid, indent=2, ensure_ascii=False) + "\n")
    width = len(str(len(runs) - 1))
    tasks = [(index, f"{index:0{width}d}-{slug(varied)}", params, varied, output, keep, skip_access)
             for index, (params, varied) in enumerate(runs)]
    rows, progress = [], progress or sys.stderr

    def record(result):
        row, seconds = result
        rows.append(row)
        print(f"[{len(rows)}/{len(tasks)}] {row['run']}: {row['status'][:120]} ({seconds:.1f} s)", file=progress,
              flush=True)

    if jobs == 1:
        for task in tasks:
            record(run_one(task, grid["tool"]))
    else:
        # A killed worker (e.g. out of memory) breaks the pool; the unfinished runs are then recorded
        # as errors instead of waiting forever.
        with ProcessPoolExecutor(jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(run_one, task, grid["tool"]): task for task in tasks}
            for future in as_completed(futures):
                try:
                    record(future.result())
                except BrokenProcessPool as error:
                    index, name, _, varied, *_ = futures[future]
                    row = {"index": index, "run": name, "status": _status("error", f"worker died: {error}")}
                    record((row | {key: cell(value) for key, value in varied.items() if key != "seed"}, 0.0))
    rows.sort(key=lambda row: row["index"])
    varied_keys = list(dict.fromkeys(key for _, varied in runs for key in varied if key != "seed"))
    metrics = tuple(tool.metrics())
    extras = sorted({key for row in rows for key in row if key.startswith("days_to_limit_")} - set(metrics))
    fields = ["index", "run", *varied_keys, "seed", "status", *metrics, *extras]
    rows = [{key: row.get(key) for key in fields} for row in rows]
    write_table(output / "summary.csv", rows)
    if not keep and not any((output / "runs").iterdir()):
        (output / "runs").rmdir()
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m hbf.sweep",
                                     description="Run a grid of analysis-only pipelines (or endurance-model runs, "
                                                 "\"tool\": \"endurance\") and summarize them.")
    parser.add_argument("grid", help="GRID.json file or JSON text")
    parser.add_argument("--output", type=Path, required=True, help="new sweep directory")
    parser.add_argument("--jobs", type=int, default=1, help=f"worker processes (1-{MAX_JOBS}, default 1)")
    parser.add_argument("--no-keep", action="store_true", help="delete each run directory after summarizing it")
    parser.add_argument("--skip-access-csv", action="store_true",
                        help="delete each run's access.csv after the run (saves disk, not time)")
    args = parser.parse_args(argv)
    try:
        rows = sweep(args.grid, args.output, args.jobs, not args.no_keep, args.skip_access_csv)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    summary = args.output / "summary.csv"
    print(summary.resolve())
    errors = sum(row["status"].startswith("error") for row in rows)
    if errors:
        print(f"{errors} runs raised errors; see the status column", file=sys.stderr)
        raise SystemExit(1)
    return summary


if __name__ == "__main__":
    main()
