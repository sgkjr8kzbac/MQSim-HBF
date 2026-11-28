# 只在 tag baseline-pre-refactor 上使用（R2 后不再维护）。
import csv, sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from hbf.access import LAYOUTS, Shape, TensorSpec, generate
from hbf.analysis import Protocol, analyze
from hbf.config import load
from hbf.controller import schedule
cfg = load("ocp")
cfg_8ch = load("ocp-8ch")
shape = Shape(32, 8, 128, 32768)
rows = []
cases = [("bf16", None, "uniform_pages", dict(fraction=.05, seed=11)),
         ("bf16", None, "sink_recent", dict(count=4, fraction=1/64)),
         ("2bit", (TensorSpec("K", 2, 128), TensorSpec("V", 2, 128)), "uniform_pages", dict(fraction=.05, seed=11))]
protocols = [("none", cfg, None), ("fcfs256", cfg, Protocol(256)), ("fcfs1024", cfg, Protocol(1024)),
             ("fcfs256 wanted", cfg, Protocol(256, False, "wanted")), ("burst256", cfg, Protocol(256, True)),
             ("trigger256", cfg, Protocol(256, host_order="trigger")),
             ("trigger256 g1 (ocp-8ch)", cfg_8ch, Protocol(256, host_order="trigger", ucie_grade=1)),
             ("trigger256 g2 (ocp-8ch)", cfg_8ch, Protocol(256, host_order="trigger", ucie_grade=2)),
             ("trigger256 g2", cfg, Protocol(256, host_order="trigger", ucie_grade=2)),
             ("trigger256 g3", cfg, Protocol(256, host_order="trigger", ucie_grade=3))]
for fmt, tensors, mode, kw in cases:
    for layout_name in ("layer_contiguous", "striped_contiguous", "striped_staggered"):
        cached = {}
        for name, target, protocol in protocols:
            if target not in cached:
                layout = LAYOUTS[layout_name](target, shape, tensors=tensors)
                cached[target] = schedule(target, layout, generate(target, layout, mode, **kw))
            _, (step,) = analyze(target, cached[target], (1,), "static", protocol)
            r = step["protocol_rounds"] if protocol else step["rounds"]
            rows.append(dict(format=fmt, mode=mode, layout=layout_name, protocol=name, pages=step["pages"],
                             commands=step.get("commands", ""), rounds=r, credit_rounds=step.get("credit_rounds", ""),
                             utilization=round(step["pages"] / (r * step["planes"]), 4)))
            print(fmt, mode[:7], layout_name[:7], name, step["pages"], step.get("commands", ""), r, rows[-1]["utilization"], flush=True)
with open(sys.argv[1], "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
