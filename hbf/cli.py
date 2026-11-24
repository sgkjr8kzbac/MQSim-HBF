"""Generate or import logical accesses, run MQSim, and report the results."""
import argparse
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .access import LAYOUTS, MODES, Shape, generate, parse_tensors, read_access, tensors_meta, write_access
from .config import MAX_BATCH_PAGES, load
from .controller import schedule
from .simulator import build, dump, simulate


def pipeline(cfg, shape, layout_name, directory, binary, *, mode="dense", generation=None,
             input_csv=None, physical=False, tensors=None):
    directory.mkdir(parents=True, exist_ok=False)
    generation = {
        "steps": 1, "seed": 1, "fraction": .1, "count": 4, "unit": 0,
        "hot_fraction": .5, "hot_units": 1,
    } | (generation or {})
    layout = LAYOUTS[layout_name](cfg, shape, tensors=tensors)
    access_path = directory / "access.csv"
    if input_csv is None:
        write_access(access_path, generate(cfg, layout, mode, **generation))
    else:
        write_access(access_path, read_access(input_csv))
    dump(directory / "meta.json", {"shape": asdict(shape), "layout": layout_name,
                                   "tensors": tensors_meta(layout),
                                   "generator": {"mode": mode, **generation} if input_csv is None else None,
                                   "input_csv": str(input_csv) if input_csv else None,
                                   "argv": sys.argv})
    batches = schedule(cfg, layout, read_access(access_path))
    result = simulate(cfg, batches, directory, binary, physical=physical)
    _, _, steps, stats = result
    report = ["# 本次运行", "", f"布局 `{layout_name}`，子批 {len(batches)} 个。",
              f"MQSim 用时 {stats['wall_seconds']:.3f} s，峰值 RSS {stats['peak_rss_kib']} KiB。", "",
              "| Step | 读页 | MQSim ns | MQSim 估算 ns | 理想 HBF ns |",
              "|---:|---:|---:|---:|---:|"]
    for row in steps:
        report.append(f"| {row['step']} | {row['read_pages']} | {row['mqsim_ns']} | "
                      f"{row['mq_est_ns']} | {row['ideal_ns']:.0f} |")
    (directory / "report.md").write_text("\n".join(report) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", choices=("16x64", "64x16", "256x4"), default="64x16")
    parser.add_argument("--layout", choices=LAYOUTS, default="layer_contiguous")
    parser.add_argument("--mode", choices=MODES, default="dense")
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--fraction", type=float, default=.1)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--unit", type=int, default=0)
    parser.add_argument("--hot-fraction", type=float, default=.5)
    parser.add_argument("--hot-units", type=int, default=1)
    parser.add_argument("--batch-pages", type=int, default=MAX_BATCH_PAGES,
                        help=f"maximum pages per sub-batch (1–{MAX_BATCH_PAGES})")
    parser.add_argument("--physical", action="store_true")
    parser.add_argument("--access-csv", type=Path)
    parser.add_argument("--tensors", help="JSON list or file of stored tensor formats, "
                        'e.g. \'[{"name":"K","bits":2,"dims":128},{"name":"V","bits":16,"dims":128}]\'; '
                        "default BF16 K and V with head_dim channels")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    directory = args.output or Path("runs") / f"{now}-{uuid4().hex[:8]}"
    generation = {key: getattr(args, key) for key in (
        "steps", "seed", "fraction", "count", "unit", "hot_fraction", "hot_units"
    )}
    try:
        cfg = load(args.topology, batch_pages=args.batch_pages)
        tensors = parse_tensors(args.tensors)
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    shape = Shape(args.layers, args.heads, args.head_dim, args.context)
    pipeline(cfg, shape, args.layout, directory, build(), mode=args.mode, generation=generation,
             input_csv=args.access_csv, physical=args.physical, tensors=tensors)
    print(directory.resolve())


if __name__ == "__main__":
    main()
