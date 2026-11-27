"""Compare the analysis read-time estimate with MQSim, layer by layer, on kept MQSim run directories.

usage: python evidence/scripts/time_calibration.py RUN_DIR... > evidence/time-calibration.csv

Each run directory holds meta.json (the command line), access.csv and layers.csv (MQSim's time per
layer). The estimate is recomputed with the current code from the same accesses and settings, using
the plane organization MQSim ran: one plane per group for the independent rule, the whole die for
lockstep. Chip-as-plane runs have no estimate and are skipped.
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hbf.access import LAYOUTS, read_access  # noqa: E402
from hbf.analysis import analyze_accesses, plane_geometry  # noqa: E402
from hbf.cli import build_parser, configure  # noqa: E402
from hbf.timing import hbf_command_model  # noqa: E402

writer = csv.writer(sys.stdout)
writer.writerow(["run", "plane_rule", "cache_read", "coalesce_ns", "aggregate_gb_s", "step", "layer", "pages",
                 "rounds", "est_batches_ns", "mqsim_ns", "error", "hbf_command_model"])
for directory in map(Path, sys.argv[1:]):
    argv = json.loads((directory / "meta.json").read_text())["argv"][1:]
    if "--output" in argv:
        index = argv.index("--output")
        del argv[index:index + 2]
    run = configure(build_parser().parse_args(argv))
    cfg = run["cfg"]
    if cfg.dies == cfg.planes == 1:
        continue
    group = plane_geometry(cfg)[1] if cfg.plane_rule == "lockstep" else 1
    layout = LAYOUTS[run["layout_name"]](cfg, run["shape"], tensors=run["tensors"])
    rows, _ = analyze_accesses(cfg, layout, list(read_access(directory / "access.csv")), (group,),
                               protocol=run["protocol"], timing=True)
    mqsim = {(int(r["step"]), int(r["layer"])): int(r["mqsim_ns"]) for r in csv.DictReader((directory / "layers.csv").open())
             if int(r["step"]) >= 0}
    for row in rows:
        measured = mqsim[row["step"], row["layer"]]
        writer.writerow([directory.name, cfg.plane_rule, cfg.cache_read, cfg.coalesce_ns, cfg.aggregate_gb_s,
                         row["step"], row["layer"], row["pages"], row["rounds"], row["est_batches_ns"], measured,
                         f"{row['est_batches_ns'] / measured - 1:.4f}", hbf_command_model(cfg)])
