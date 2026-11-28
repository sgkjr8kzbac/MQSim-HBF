# 只在 tag baseline-pre-refactor 上使用（R2 后不再维护）。
"""E5.6: how much decode flushes, refresh and erases stretch TPOT (hbf.interference), as a table.

usage: python evidence/scripts/interference_scan.py OUTPUT_DIR [--jobs N]

1. Analysis-only system runs (ocp, HBF command model, B200, dense, striped) for each base point.
2. Every run re-timed under each write scenario x tPROG x tBERS x interference mode.
Writes OUTPUT_DIR/interference-summary.csv (one row per combination, step 0) and keeps the runs.

Write scenarios (per stack):
* decode: step-driven flushes only, B x L x H x sum_t 1 / tokens_per_page pages per step;
* refresh_1e6 and refresh_1e5: decode plus a time-driven refresh write equal to dense read bandwidth x
  pages per block / X (every block refreshed after X reads; design section 3), with one erase per
  rewritten block;
* prefill: decode plus a time-driven prefill write of 2.66 GB/s (8B, B = 8, 32K arrivals, design 3).
These are scenario loads to bound the effect; the endurance model supplies workload-derived loads.
"""
import argparse
import csv
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hbf.cli import build_parser, configure, pipeline  # noqa: E402
from hbf.interference import retime  # noqa: E402

BASES = [  # (name, flags)
    ("8b-b8-32k-bf16", ["--model", "llama-3.1-8b", "--batch", "8", "--context", "32768"]),
    ("8b-b32-32k-bf16", ["--model", "llama-3.1-8b", "--batch", "32", "--context", "32768"]),
    ("8b-b8-32k-kivi2", ["--model", "llama-3.1-8b", "--batch", "8", "--context", "32768", "--format", "kivi2"]),
    ("70b-b8-32k-bf16", ["--model", "llama-3.1-70b", "--batch", "8", "--context", "32768"]),
]
COMMON = ["--analysis-only", "--topology", "ocp", "--read-plane-ns", "0", "--coalesce-ns", "1", "--cache-read",
          "--layout", "striped_contiguous", "--gpu", "b200"]
T_PROG = (75_000, 100_000, 200_000, 350_000)  # ns: Kouchi 2020 / SPLASH / 2x / pSLC on TLC (docs/research/params.md)
T_BERS = (1_000_000, 3_500_000, 10_000_000)  # ns: SPLASH / planar SLC / 3D TLC datasheet
MODES = (("fifo", None), ("gaps", None), ("suspend", 5_000), ("suspend", 20_000), ("random", None))
PAGES_PER_BLOCK = 256
BLOCK_BYTES = PAGES_PER_BLOCK * 4096


def run_base(output, name, flags):
    directory = output / "runs" / name
    if not (directory / "system_steps.csv").exists():
        run = configure(build_parser().parse_args(COMMON + flags + ["--output", str(directory)]))
        pipeline(directory=directory, binary=None, **run)
    with (directory / "analysis_steps.csv").open() as stream:
        step = next(row for row in csv.DictReader(stream) if row["group"] == "1")
    with (directory / "system_steps.csv").open() as stream:
        system = next(csv.DictReader(stream))
    return directory, run_meta(directory), step, system


def run_meta(directory):
    import json
    meta = json.loads((directory / "meta.json").read_text())
    tensors = [t for t in meta["tensors"] if t["tier"] == "hbf"]
    tokens_per_page = [4096 * 8 // (t["bits"] * t["dims"]) for t in tensors]
    shape = meta["shape"]
    return {"decode_pages": shape["layers"] * shape["kv_heads"] * sum(1 / tpp for tpp in tokens_per_page)}


def scenarios(meta, step, system):
    read_bytes_s = int(step["pages"]) * 4096 / (float(system["tpot_hbf_est_ns"]) / 1e9)
    decode = {"programs_per_step": meta["decode_pages"], "erases_per_step": meta["decode_pages"] / PAGES_PER_BLOCK}
    out = {"decode": decode | {"write_gb_s": 0.0, "erases_per_s": 0.0}}
    for label, threshold in (("refresh_1e6", 1e6), ("refresh_1e5", 1e5)):
        refresh = read_bytes_s * PAGES_PER_BLOCK / threshold
        out[label] = decode | {"write_gb_s": refresh / 1e9, "erases_per_s": refresh / BLOCK_BYTES}
    out["prefill"] = decode | {"write_gb_s": 2.66, "erases_per_s": 2.66e9 / BLOCK_BYTES}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = []
    for name, flags in BASES:
        directory, meta, step, system = run_base(args.output, name, flags)
        print(f"{name}: TPOT {float(system['tpot_hbf_est_ns']) / 1e6:.2f} ms", file=sys.stderr, flush=True)
        for (label, load), t_prog, t_bers, (mode, suspend) in itertools.product(
                scenarios(meta, step, system).items(), T_PROG, T_BERS, MODES):
            (result, *_) = retime(directory, mode=mode, t_prog=t_prog, t_bers=t_bers, suspend_ns=suspend, **load)
            rows.append({"base": name, "scenario": label, "mode": mode, "suspend_ns": suspend, "t_prog_ns": t_prog,
                         "t_bers_ns": t_bers, "write_gb_s": round(load["write_gb_s"], 4),
                         "erases_per_s": round(load["erases_per_s"], 2),
                         "programs_per_step": round(load["programs_per_step"], 2),
                         "tpot_ms": result["tpot_ns"] / 1e6, "tpot_m5_ms": result["tpot_m5_ns"] / 1e6,
                         "extra_pct": 100 * (result["tpot_m5_ns"] / result["tpot_ns"] - 1),
                         "max_step_extra_ms": result["max_step_extra_ns"] / 1e6})
    path = args.output / "interference-summary.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(path.resolve())


if __name__ == "__main__":
    main()
