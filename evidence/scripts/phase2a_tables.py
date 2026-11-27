"""Markdown tables of the phase A sweeps (evidence/phase2a/*.csv) used in REPORT.

usage: python evidence/scripts/phase2a_tables.py [evidence/phase2a]
"""
import collections
import csv
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parents[1] / "phase2a")


def load(name):
    return list(csv.DictReader((root / f"{name}.csv").open()))


def ms(value):
    return f"{float(value) / 1e6:.2f}"


def device():
    print("### item1-device: single stack, B200, dense, striped (ms per step)\n")
    print("| model | context | B | tR ns | core-die GB/s | HBF est | HBF bound | TPOT HBF | TPOT HBM | HBF/HBM | GPU waits | KV fits in HBM |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|")
    for r in load("item1-device"):
        if r["status"] != "ok":
            continue
        print(f"| {r['model']} | {r['context']} | {r['batch']} | {r['read_ns']} | {r['aggregate_gb_s']} | {ms(r['est_ns'])} | "
              f"{ms(r['bound_ns'])} | {ms(r['tpot_hbf_est_ns'])} | {ms(r['tpot_hbm_ns'])} | {float(r['hbf_over_hbm']):.2f} | "
              f"{float(r['hbf_wait_share']):.2f} | {r['hbm_fits']} |")


def system():
    print("\n### item1-system: 8 stacks per GPU, B200, dense, striped (ms per step)\n")
    print("| model | context | B | overlap | HBF est | TPOT HBF | TPOT HBM | HBF/HBM | GPU waits | KV fits in HBM |")
    print("|---|---:|---:|---|---:|---:|---:|---:|---:|:---:|")
    for r in load("item1-system"):
        print(f"| {r['model']} | {r['context']} | {r['batch']} | {r['overlap']} | {ms(r['est_ns'])} | {ms(r['tpot_hbf_est_ns'])} | "
              f"{ms(r['tpot_hbm_ns'])} | {float(r['hbf_over_hbm']):.2f} | {float(r['hbf_wait_share']):.2f} | {r['hbm_fits']} |")


def layout_protocol():
    print("\n### item1-layout-protocol: llama-3.1-8b, 32K, B=4, one stack, mean of 3 seeds (ms per step)\n")
    print("| mode | layout | MOCS | host order | UCIe grade | HBF est | TPOT HBF | utilization | bottleneck |")
    print("|---|---|---:|---|---:|---:|---:|---:|---|")
    groups = collections.defaultdict(list)
    for r in load("item1-layout-protocol"):
        groups[r["mode"], r["layout"], r.get("mocs", ""), r.get("host_order", ""), r.get("ucie_grade", "")].append(r)
    for key, rows in groups.items():
        mean = lambda col: statistics.mean(float(row[col]) for row in rows)
        common = collections.Counter(row["bottleneck"] for row in rows).most_common(1)[0][0]
        print(f"| {' | '.join(k or '—' for k in key)} | {mean('est_ns') / 1e6:.2f} | {mean('tpot_hbf_est_ns') / 1e6:.2f} | "
              f"{mean('utilization'):.3f} | {common} |")


def compression():
    rows = load("item2-compression")
    groups = collections.defaultdict(list)

    def selection(r):
        if r["mode"] == "uniform_tokens":
            return f"uniform_tokens {float(r['fraction']) * 100:g}% / {r['select_tokens']}"
        return "sink_recent 5%" if r["mode"] == "sink_recent" else "dense"
    for r in rows:
        groups[r["context"], r["batch"], r["layout"], selection(r), r["format"]].append(r)
    formats = list(dict.fromkeys(r["format"] for r in rows))
    selections = list(dict.fromkeys(selection(r) for r in rows))
    for context, batch, layout in sorted({key[:3] for key in groups}):
        print(f"\n### item2: {layout}, context {context}, B={batch}: speedup of HBF read time (est) and TPOT over bf16\n")
        print("| selection | bf16 HBF est ms | " + " | ".join(formats[1:]) + " |")
        print("|---|---:|" + "---:|" * (len(formats) - 1))
        for sel in selections:
            base = groups[context, batch, layout, sel, "bf16"]
            mean = lambda rs, col: statistics.mean(float(x[col]) for x in rs)
            cells = []
            for fmt in formats[1:]:
                rs = groups[context, batch, layout, sel, fmt]
                cells.append(f"{mean(base, 'est_ns') / mean(rs, 'est_ns'):.2f} / "
                             f"{mean(base, 'tpot_hbf_est_ns') / mean(rs, 'tpot_hbf_est_ns'):.2f}")
            print(f"| {sel} | {mean(base, 'est_ns') / 1e6:.2f} | " + " | ".join(cells) + " |")


device()
system()
layout_protocol()
compression()
