"""Generate or import logical accesses, run MQSim, and report the results."""
import argparse
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from itertools import chain
from pathlib import Path
from uuid import uuid4

from .access import (LAYOUTS, MODES, Shape, generate, parse_tensors, prefill, read_access, tensors_meta,
                     write_access)
from .analysis import analyze, check_group
from .config import MAX_BATCH_PAGES, load
from .controller import schedule
from .simulator import build, dump, simulate, write_table


def analysis_report(layer_rows, step_rows):
    report = ["## 机制分析", "",
              "| 并行组 | Step | 层数 | 读页 | 轮数 | 最少轮数 | plane 利用率 | 均衡损失 | 感测放大 | 读放大 | 传输放大 |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    amplification = {}
    for row in layer_rows:
        key = row["group"], row["step"]
        page, useful, burst = amplification.get(key, (0, 0, 0))
        amplification[key] = page + row["page_bytes"], useful + row["useful_bytes"], burst + row["burst_bytes"]
    for row in step_rows:
        page, useful, burst = amplification[row["group"], row["step"]]
        report.append(
            f"| {row['group']} | {row['step']} | {row['layers']} | {row['pages']} | {row['rounds']} | "
            f"{row['min_rounds']} | {row['utilization']:.3f} | {row['balance_loss']:.2f} | "
            f"{row['sense_amplification']:.2f} | {page / useful:.2f} | {burst / useful:.2f} |")
    return report


def reject_write(access):
    if access.op == "W":
        raise ValueError("--prefill writes the whole layout; the accesses must not contain writes")
    return access


def pipeline(cfg, shape, layout_name, directory, binary, *, mode="dense", generation=None,
             input_csv=None, physical=False, tensors=None, plane_groups=(1,), prefill_layout=False):
    """Schedule the accesses, write the mechanism analysis, and run MQSim unless binary is None."""
    directory.mkdir(parents=True, exist_ok=False)
    generation = {
        "steps": 1, "seed": 1, "fraction": .1, "count": 4, "unit": 0,
        "hot_fraction": .5, "hot_units": 1,
    } | (generation or {})
    layout = LAYOUTS[layout_name](cfg, shape, tensors=tensors)
    access_path = directory / "access.csv"
    accesses = generate(cfg, layout, mode, **generation) if input_csv is None else read_access(input_csv)
    if prefill_layout:
        accesses = chain(prefill(layout), (reject_write(access) for access in accesses))
    write_access(access_path, accesses)
    dump(directory / "meta.json", {"shape": asdict(shape), "layout": layout_name,
                                   "tensors": tensors_meta(layout),
                                   "generator": {"mode": mode, **generation} if input_csv is None else None,
                                   "input_csv": str(input_csv) if input_csv else None,
                                   "prefill": prefill_layout,
                                   "argv": sys.argv})
    batches = schedule(cfg, layout, read_access(access_path))
    layer_rows, step_rows = analyze(cfg, batches, plane_groups)
    if layer_rows:
        write_table(directory / "analysis_layers.csv", layer_rows)
        write_table(directory / "analysis_steps.csv", step_rows)
    report = ["# 本次运行", "", f"布局 `{layout_name}`，拓扑 {cfg.channels}×{cfg.chips}，"
              f"子批 {len(batches)} 个。", ""]
    if layer_rows:
        report += analysis_report(layer_rows, step_rows) + [""]
    result = None
    if binary is not None:
        result = simulate(cfg, batches, directory, binary, physical=physical)
        _, _, steps, stats = result
        report += ["## MQSim", "",
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
    parser.add_argument("--topology", default="64x16",
                        help="16x64, 64x16, 256x4 or any <channels>x<chips per channel>")
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
    parser.add_argument("--plane-group", type=int, action="append", dest="plane_groups",
                        help="planes per lock-step group for the mechanism analysis; repeat to compare "
                             "(1 = independent planes, the default)")
    parser.add_argument("--prefill", action="store_true",
                        help="write the whole KV layout before the reads so that each page's physical "
                             "offset follows its LPA (needed for lock-step comparisons)")
    parser.add_argument("--plane-rule", choices=("independent", "lockstep"),
                        help="T-PLANE multi-plane rule; without it MQSim keeps its upstream same-page rule")
    parser.add_argument("--planes", type=int, default=1,
                        help="planes per MQSim die (with --topology Cx1 this gives the real die/plane hierarchy)")
    parser.add_argument("--dbsy-ns", type=int, help="t_DBSY between planes of a multi-plane command (T-PLANE)")
    parser.add_argument("--read-plane-ns", type=int,
                        help="T-CMD: a read command for n planes costs the single-plane command plus (n-1) "
                             "times this step, instead of ONFI's 6*t_WC + t_DBSY per extra plane")
    parser.add_argument("--cache-read", action="store_true",
                        help="T-CACHE: two buffers per plane, so a die senses the next read while the "
                             "previous pages are transferred")
    parser.add_argument("--coalesce-ns", type=int,
                        help="T-CMD: hold newly received transactions this long before dispatching, so that "
                             "a batch leaves as full multi-plane commands")
    parser.add_argument("--analysis-only", action="store_true",
                        help="write the mechanism analysis without building or running MQSim")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    directory = args.output or Path("runs") / f"{now}-{uuid4().hex[:8]}"
    generation = {key: getattr(args, key) for key in (
        "steps", "seed", "fraction", "count", "unit", "hot_fraction", "hot_units"
    )}
    try:
        cfg = load(args.topology, batch_pages=args.batch_pages, planes=args.planes,
                   plane_rule=args.plane_rule, dbsy_ns=args.dbsy_ns, coalesce_ns=args.coalesce_ns,
                   read_plane_ns=args.read_plane_ns, cache_read=args.cache_read)
        tensors = parse_tensors(args.tensors)
        for group in args.plane_groups or (1,):
            check_group(cfg, group)
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    shape = Shape(args.layers, args.heads, args.head_dim, args.context)
    binary = None if args.analysis_only else build()
    pipeline(cfg, shape, args.layout, directory, binary, mode=args.mode, generation=generation,
             input_csv=args.access_csv, physical=args.physical, tensors=tensors,
             plane_groups=tuple(args.plane_groups or (1,)), prefill_layout=args.prefill)
    print(directory.resolve())


if __name__ == "__main__":
    main()
