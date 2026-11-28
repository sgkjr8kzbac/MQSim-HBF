# 只在 tag baseline-pre-refactor 上使用（R2 后不再维护）。
import csv, sys
from statistics import mean
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from hbf.access import LAYOUTS, Shape, TensorSpec, generate
from hbf.analysis import analyze
from hbf.config import load
from hbf.controller import schedule
cfg = load("ocp")
shape = Shape(32, 8, 128, 32768)
formats = {"bf16": None, "2bit": (TensorSpec("K", 2, 128), TensorSpec("V", 2, 128)),
           "kivi(K2,V16)": (TensorSpec("K", 2, 128), TensorSpec("V", 16, 128))}
seeds = (11, 12, 13, 14, 15)
rows = []
for fmt, tensors in formats.items():
    layouts = {name: LAYOUTS[name](cfg, shape, tensors=tensors) for name in ("layer_contiguous", "striped_contiguous", "striped_staggered")}
    for mode in ("uniform_pages", "uniform_tokens"):
        for fraction in (.001, .003, .01, .05):
            for name, layout in layouts.items():
                results = {}
                for seed in seeds:
                    batches = schedule(cfg, layout, generate(cfg, layout, mode, fraction=fraction, seed=seed))
                    for st in analyze(cfg, batches, (1, 16))[1]:
                        results.setdefault(st["group"], []).append((st["pages"], st["rounds"]))
                for group, values in results.items():
                    rounds = [r for _, r in values]
                    row = dict(format=fmt, mode=mode, fraction=fraction, layout=name, group=group,
                               seeds=len(seeds), pages_mean=round(mean(p for p, _ in values)),
                               rounds_mean=round(mean(rounds), 1), rounds_min=min(rounds), rounds_max=max(rounds))
                    rows.append(row); print(*row.values(), flush=True)
with open(sys.argv[1], "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
