"""Write, erase and refresh interference with decode reads (module M5), analytic.

A program (tPROG) or erase (tBERS) blocks reads of the unit it runs on. The OCP HBF spec v0.7.0
has no program or erase suspend ("suspend" and "resume" do not occur in it); its only rule is
section 11.5: when refresh is interleaved with work, "refresh commands and read commands are not
issued to the same die at the same time". The default blocking unit is therefore the die
(16 banks, section 4.6 Figure 4), which is also what MQSim does on ``ocp`` (a chip is a die).

Inputs are per decode step (averages, so counts may be fractional):

* ``programs`` - pages programmed; ``erases`` - blocks erased;
* ``wave_banks`` - banks one write wave covers (s): 1,024 for row-striped writes, 64 or 16 under
  channel or die partitions, 1 for a single-bank replay;
* ``gaps_ns`` - HBF-idle time before each layer's read (g_l), or ``gap_ns`` and ``layers``;
* ``step_ns`` - the step time (TPOT), needed by the ``random`` and ``correlated`` modes;
* ``burst_programs`` - pages that land in one step without being spread (an unspread refresh),
  used only for ``max_step_extra_ns``.

Every layer of a striped layout reads every die, so the busiest unit sets the delay. With waves
spread evenly, a unit sees K = (programs / s) x (units per wave) / units operations per step:
K = programs / 1,024 for any s >= 16 under die blocking, and programs / 64 for single-bank waves.

Modes (extra read time per step):

* ``fifo`` - writes run when issued, reads queue behind them: W = K tPROG + M tBERS.
* ``gaps`` (default) - the host defers non-preemptive operations to HBF-idle gaps (what section
  11.5 asks for). A gap g holds c = floor(g / t) operations for free; the first extra operation in
  a gap costs (c + 1) t - g, further ones t. Extra operations go to the cheapest gaps first.
  Erases are placed first, programs into what is left.
* ``suspend`` - product specific, not in OCP v0.7.0: reads preempt and operations progress only
  in gaps. Extra = N_s t_s + max(0, W - G), N_s = min(L, ceil(W / g)) layers that start with a
  suspended operation (for under one operation per step, the expected count of one operation).
* ``random`` - uncoordinated operations, independent per unit with busy fraction u: the expected
  largest residual over D units, T - T / (u (D + 1)) (1 - (1 - u)^(D + 1)), per layer.
* ``correlated`` - striped waves keep all units busy together: u T / 2 per layer.

``python -m hbf.interference RUN_DIR --write-gb-s X --erases-per-s Y --mode gaps`` re-times a
finished phase A run (system_layers.csv) and writes interference_steps.csv with TPOT'.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

MODES = ("fifo", "gaps", "suspend", "random", "correlated")
BLOCK_UNITS = ("die", "bank")
from ..device.config import PARAMETERS


def blocking_units(dies=64, banks_per_die=16, block_unit="die"):
    """(units that block reads, banks per unit): dies (default) or single banks (optimistic ablation)."""
    if block_unit not in BLOCK_UNITS:
        raise ValueError(f"block_unit must be one of {', '.join(BLOCK_UNITS)}")
    if type(dies) is not int or type(banks_per_die) is not int or dies < 1 or banks_per_die < 1:
        raise ValueError("dies and banks_per_die must be positive integers")
    return (dies, banks_per_die) if block_unit == "die" else (dies * banks_per_die, 1)


def unit_ops(count, wave_banks, units, banks_per_unit):
    """Operations per blocking unit for ``count`` page programs (or block erases) in waves of ``wave_banks``."""
    if count <= 0:
        return 0.0
    if wave_banks < 1:
        raise ValueError("wave_banks must be at least 1")
    per_wave = min(units, max(1, math.ceil(wave_banks / banks_per_unit)))
    return count / wave_banks * per_wave / units


def _gaps(step):
    if step.get("gaps_ns") is not None:
        gaps = np.asarray(step["gaps_ns"], dtype=float)
    else:
        layers = step.get("layers")
        if type(layers) is not int or layers < 1 or step.get("gap_ns") is None:
            raise ValueError("Give gaps_ns (one per layer) or gap_ns and layers")
        gaps = np.full(layers, float(step["gap_ns"]))
    if not len(gaps) or (gaps < 0).any() or not np.isfinite(gaps).all():
        raise ValueError("Gaps must be finite and non-negative, one per layer")
    return gaps


def pack(gaps, count, t_op):
    """Place ``count`` operations of ``t_op`` into idle gaps (continuous counts).

    Returns (extra read time per layer, idle time left per layer). Free operations take the
    largest gaps first; extra ones overflow the gaps with the cheapest overflow first, then cost
    t_op each, spread evenly over the layers.
    """
    gaps = np.asarray(gaps, dtype=float)
    left, extra = gaps.copy(), np.zeros(len(gaps))
    if count <= 0 or t_op <= 0:
        return extra, left
    free = np.floor(gaps / t_op)
    placed = min(count, float(free.sum()))
    remaining = placed
    for layer in np.argsort(-gaps, kind="stable"):
        if remaining <= 0:
            break
        take = min(free[layer], remaining)
        left[layer] -= take * t_op
        remaining -= take
    over = count - placed
    if over <= 0:
        return extra, left
    cost = t_op - left  # every free slot is used, so left holds g mod t_op
    order = np.argsort(cost, kind="stable")
    first = min(over, len(gaps))
    whole = int(first)
    extra[order[:whole]] += cost[order[:whole]]
    left[order[:whole]] = 0
    if first > whole:
        part = first - whole
        extra[order[whole]] += part * cost[order[whole]]
        left[order[whole]] *= 1 - part
    rest = over - first
    if rest > 0:
        extra += rest * t_op / len(gaps)
        left[:] = 0
    return extra, left


def residual_max(u, t_op, units):
    """Expected largest residual busy time over ``units`` independent units, each busy with probability u."""
    u = min(max(float(u), 0.0), 1.0)
    if u <= 0 or t_op <= 0:
        return 0.0
    n = units + 1
    return t_op - t_op / (u * n) * (1 - (1 - u) ** n)


def _extra(mode, gaps, k, m, t_prog, t_bers, suspend_ns, step_ns, units):
    """(extra per layer array, overflow ns) for k program and m erase operations per unit and step."""
    layers, idle = len(gaps), float(gaps.sum())
    busy = k * t_prog + m * t_bers
    if mode == "fifo":
        return np.full(layers, busy / layers), 0.0
    if mode == "gaps":
        extra_e, left = pack(gaps, m, t_bers)
        extra_p, _ = pack(left, k, t_prog)
        return extra_e + extra_p, 0.0
    if mode == "suspend":
        if suspend_ns is None or suspend_ns < 0:
            raise ValueError("suspend mode needs suspend_ns (product specific; OCP v0.7.0 has no suspend)")
        extra = np.zeros(layers)
        if busy <= 0:
            return extra, 0.0
        mean_gap = idle / layers
        ops = k + m
        if mean_gap <= 0:
            suspended = min(1.0, ops) * layers
        elif ops >= 1:
            suspended = float(min(layers, math.ceil(busy / mean_gap - 1e-9)))
        else:
            suspended = ops * min(layers, math.ceil(busy / ops / mean_gap - 1e-9))
        order = np.argsort(-gaps, kind="stable")
        whole = int(suspended)
        extra[order[:whole]] += suspend_ns
        if suspended > whole:
            extra[order[whole]] += (suspended - whole) * suspend_ns
        overflow = max(0.0, busy - idle)
        return extra + overflow / layers, overflow
    if mode in ("random", "correlated"):
        if step_ns is None or step_ns <= 0:
            raise ValueError(f"{mode} mode needs step_ns (the step time) for the busy fraction")
        u_p, u_e = k * t_prog / step_ns, m * t_bers / step_ns
        if mode == "random":
            delay = residual_max(u_p, t_prog, units) + residual_max(u_e, t_bers, units)
        else:
            delay = min(u_p, 1.0) * t_prog / 2 + min(u_e, 1.0) * t_bers / 2
        return np.full(layers, delay), 0.0
    raise ValueError(f"mode must be one of {', '.join(MODES)}")


def interference(step_inputs, mode, t_prog, t_bers, suspend_ns=None, dies=64, banks_per_die=16, block_unit="die"):
    """Extra read time per decode step caused by programs and erases (see the module docstring).

    Returns a dict: ``extra_step_ns``, ``extra_layer_ns`` (list, one per layer), ``max_step_extra_ns``
    (a step that carries whole operations plus any unspread burst; an approximate p99), and the
    inputs it derived: ``program_ops`` and ``erase_ops`` per unit and step, ``busy_ns`` (W),
    ``idle_ns`` (G), ``overflow_ns`` (suspend: work that does not fit in idle time).
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}")
    if t_prog < 0 or t_bers < 0:
        raise ValueError("t_prog and t_bers must be non-negative")
    step = dict(step_inputs)
    units, per_unit = blocking_units(dies, banks_per_die, block_unit)
    gaps = _gaps(step)
    wave = step.get("wave_banks", dies * banks_per_die)
    programs, erases = float(step.get("programs", 0) or 0), float(step.get("erases", 0) or 0)
    burst = float(step.get("burst_programs", 0) or 0)
    if programs < 0 or erases < 0 or burst < 0:
        raise ValueError("programs, erases and burst_programs must be non-negative")
    k = unit_ops(programs, wave, units, per_unit)
    m = unit_ops(erases, wave, units, per_unit)
    step_ns = step.get("step_ns")
    extra, overflow = _extra(mode, gaps, k, m, t_prog, t_bers, suspend_ns, step_ns, units)
    k_max = (math.ceil(k - 1e-12) if k > 0 else 0) + unit_ops(burst, wave, units, per_unit)
    m_max = math.ceil(m - 1e-12) if m > 0 else 0
    if mode in ("random", "correlated"):
        worst = t_bers if m > 0 else t_prog if k > 0 or burst > 0 else 0.0
    else:
        worst = float(_extra(mode, gaps, k_max, m_max, t_prog, t_bers, suspend_ns, step_ns, units)[0].sum())
    total = float(extra.sum())
    return {"extra_step_ns": total, "extra_layer_ns": [float(value) for value in extra],
            "max_step_extra_ns": max(worst, total), "program_ops": k, "erase_ops": m,
            "busy_ns": k * t_prog + m * t_bers, "idle_ns": float(gaps.sum()), "overflow_ns": overflow}


def hbf_gaps(h, pre, attn, post, overlap="serial", staged=False, tail_ns=0.0):
    """HBF-idle time before each layer's read in one step, from the system model's timeline.

    ``tail_ns`` is time after the last layer (lm_head, layer overheads); it is idle for HBF and
    is added to layer 0's gap (the wrap-around to the next step). The gaps sum to TPOT - sum(h).
    """
    from ..util import parse_overlap
    mode, depth = parse_overlap(overlap)
    layers = len(h)
    if not layers:
        return []
    if mode == "ideal":
        work = sum(pre[l] + attn[l] + post[l] for l in range(layers))
        idle = max(0.0, work - sum(h)) + tail_ns
        return [idle / layers] * layers
    if mode == "serial":
        # Layer l: pre, then the read (with attention when streaming), then post.
        gaps = []
        for l in range(layers):
            before = l - 1 if l else layers - 1
            slack = attn[before] if staged else max(0.0, attn[before] - h[before])
            gaps.append(post[before] + slack + pre[l] + (tail_ns if l == 0 else 0.0))
        return gaps
    r_end, c_end, gaps = 0.0, [], []
    for l in range(layers):
        before = c_end[l - 1] if l else 0.0
        start = max(r_end, c_end[l - 1 - depth] if l - 1 - depth >= 0 else 0.0)
        gaps.append(start - r_end)
        r_end = start + h[l]
        c_end.append(max(before + pre[l], r_end) + attn[l] + post[l])
    gaps[0] += c_end[-1] - r_end + tail_ns
    return gaps


def _rows(path):
    with open(path, newline="") as stream:
        return list(csv.DictReader(stream))


def _float(text):
    return None if text in (None, "") else float(text)


def retime(run_dir, *, mode="gaps", write_gb_s=0.0, erases_per_s=0.0, programs_per_step=0.0,
           erases_per_step=0.0, t_prog=None, t_bers=None, suspend_ns=None, wave_banks=1024, page_bytes=4096,
           dies=64, banks_per_die=16, block_unit="die", iterations=500):
    """interference_steps rows for a finished phase A run with the system model.

    Time-driven loads (``write_gb_s``, ``erases_per_s``) scale with TPOT', so TPOT' is a fixed point
    TPOT' = TPOT + extra(load x TPOT'); step-driven loads (``*_per_step``) do not. A load that
    the idle time cannot absorb has no fixed point and gets an infinite TPOT'.
    """
    run_dir = Path(run_dir)
    params = json.loads(PARAMETERS.read_text())
    t_prog = params["program_ns"] if t_prog is None else t_prog
    t_bers = params["erase_ns"] if t_bers is None else t_bers
    meta = json.loads((run_dir / "meta.json").read_text())
    system = meta.get("system") or {}
    overlap = system.get("overlap") or "serial"
    staged = overlap.startswith("prefetch") or system.get("kv_consume") == "staged"
    layers_by_step = {}
    for row in _rows(run_dir / "system_layers.csv"):
        layers_by_step.setdefault(int(row["step"]), []).append(row)
    out = []
    for step_row in _rows(run_dir / "system_steps.csv"):
        step = int(step_row["step"])
        rows = sorted(layers_by_step[step], key=lambda row: int(row["layer"]))
        kind = "est" if step_row.get("tpot_hbf_est_ns") not in (None, "") else "bound"
        tpot = float(step_row[f"tpot_hbf_{kind}_ns"])
        h = [_float(row[f"hbf_{kind}_ns"]) for row in rows]
        pre, post = [float(row["pre_ns"]) for row in rows], [float(row["post_ns"]) for row in rows]
        attn = [float(row["attn_staged_ns" if staged else "attn_ns"]) for row in rows]
        gaps = hbf_gaps(h, pre, attn, post, overlap, staged)
        gaps[0] += max(0.0, tpot - sum(h) - sum(gaps))  # lm_head and layer overheads
        current, result, converged = tpot, None, False
        for _ in range(iterations):
            load = {"programs": programs_per_step + write_gb_s * 1e9 / page_bytes * current / 1e9,
                    "erases": erases_per_step + erases_per_s * current / 1e9,
                    "wave_banks": wave_banks, "gaps_ns": gaps, "step_ns": current}
            result = interference(load, mode, t_prog, t_bers, suspend_ns, dies, banks_per_die, block_unit)
            following = tpot + result["extra_step_ns"]
            if abs(following - current) <= 1e-6 * tpot:
                current, converged = following, True
                break
            if following > 1e3 * tpot:
                break
            current = following
        out.append({"step": step, "mode": mode, "tpot_ns": tpot,
                    "tpot_m5_ns": current if converged else math.inf,
                    "extra_step_ns": result["extra_step_ns"] if converged else math.inf,
                    "max_step_extra_ns": result["max_step_extra_ns"] if converged else math.inf,
                    "programs_per_step": load["programs"],
                    "erases_per_step": load["erases"], "busy_ns": result["busy_ns"], "idle_ns": result["idle_ns"],
                    "overflow_ns": result["overflow_ns"], "t_prog_ns": t_prog, "t_bers_ns": t_bers,
                    "suspend_ns": suspend_ns, "wave_banks": wave_banks, "block_unit": block_unit})
    return out


def report_lines(rows, mode, write_gb_s, erases_per_s):
    lines = ["# 写入与擦除对读的干扰（M5）", "",
             f"模式 `{mode}`，写入 {write_gb_s:g} GB/s，擦除 {erases_per_s:g} 块/s。"
             "OCP v0.7.0 没有编程或擦除挂起（suspend）；第 11.5 节只要求刷新命令与读命令不同时发往同一个 die，"
             "因此默认阻塞单位是 die（16 个 bank）。", "",
             "| Step | TPOT ms | TPOT′ ms | 每步额外 µs | 单步最大额外 µs | 每步编程页 | 每步擦除块 |",
             "|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['step']} | {row['tpot_ns'] / 1e6:.3f} | {row['tpot_m5_ns'] / 1e6:.3f} | "
                     f"{row['extra_step_ns'] / 1e3:.1f} | {row['max_step_extra_ns'] / 1e3:.1f} | "
                     f"{row['programs_per_step']:.1f} | {row['erases_per_step']:.3f} |")
    lines += ["", "假设：", "",
              "- 编程、擦除按波次进行；条带化写入一个波次覆盖全部 1,024 个 bank，每个 die 阻塞一次 tPROG（或 tBERS）。",
              "- 每层读都覆盖所有 die，所以最忙的 die 决定延迟。",
              "- 时间驱动的负载（GB/s、块/s）随 TPOT′ 增长，取不动点；不动点不存在时 TPOT′ 记为无穷大。",
              "- tPROG、tBERS 默认值取自 hbf/configs/parameters.json（SPLASH 表 1：100 µs、1 ms）。"]
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m hbf.interference",
                                     description="Re-time a finished phase A run with write/erase interference (M5).")
    parser.add_argument("run_dir", type=Path, help="run directory with system_layers.csv and system_steps.csv")
    parser.add_argument("--mode", choices=MODES, default="gaps")
    parser.add_argument("--write-gb-s", type=float, default=0.0, help="programmed bytes per second (1e9 B/s)")
    parser.add_argument("--erases-per-s", type=float, default=0.0, help="block erases per second")
    parser.add_argument("--programs-per-step", type=float, default=0.0, help="step-driven pages per step")
    parser.add_argument("--erases-per-step", type=float, default=0.0, help="step-driven block erases per step")
    parser.add_argument("--suspend-ns", type=float, help="suspend latency (suspend mode; product specific)")
    parser.add_argument("--program-ns", type=float, help="tPROG (default from hbf/configs/parameters.json)")
    parser.add_argument("--erase-ns", type=float, help="tBERS (default from hbf/configs/parameters.json)")
    parser.add_argument("--wave-banks", type=int, default=1024, help="banks per write wave (default 1024)")
    parser.add_argument("--block-unit", choices=BLOCK_UNITS, default="die")
    parser.add_argument("--output", type=Path, help="output directory (default: RUN_DIR)")
    args = parser.parse_args(argv)
    try:
        rows = retime(args.run_dir, mode=args.mode, write_gb_s=args.write_gb_s, erases_per_s=args.erases_per_s,
                      programs_per_step=args.programs_per_step, erases_per_step=args.erases_per_step,
                      t_prog=args.program_ns, t_bers=args.erase_ns, suspend_ns=args.suspend_ns,
                      wave_banks=args.wave_banks, block_unit=args.block_unit)
    except (OSError, KeyError, ValueError) as error:
        parser.error(str(error))
    output = args.output or args.run_dir
    output.mkdir(parents=True, exist_ok=True)
    with (output / "interference_steps.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "interference_report.md").write_text(
        "\n".join(report_lines(rows, args.mode, args.write_gb_s, args.erases_per_s)) + "\n")
    print((output / "interference_steps.csv").resolve())
    return rows


if __name__ == "__main__":
    main()
