"""R5 write-interference comparison: MQSim (T-HBFW1..4, fifo & read_first) vs interference.py."""
import csv
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hbf.backends.mqsim.build import build
from hbf.backends.mqsim.parse import completions
from hbf.backends.mqsim.run import simulate
from hbf.device.config import load
from hbf.frontend.pages import Batch, Request
from hbf.lifetime.interference import interference

OUT_CSV = ROOT / "evidence" / "r5-write-interference.csv"


def _read_metrics(comp_path, read_batches):
    """Return (per-layer durations, last read completion timestamp) for read_batches."""
    by_arrival = {}
    for c in completions(comp_path):
        if c.op == "R":
            cur = by_arrival.get(c.arrival_ns, 0)
            if c.complete_ns > cur:
                by_arrival[c.arrival_ns] = c.complete_ns
    durs = [by_arrival[b.arrival_ns] - b.arrival_ns for b in read_batches]
    last_done = max(by_arrival[b.arrival_ns] for b in read_batches)
    return durs, last_done


def run_point(binary, tmp_root, spec):
    """Run read-only, with-writes (fifo), and with-writes (read_first) on MQSim and compare to interference.py."""
    name = spec["name"]
    cfg = load(
        spec["topology"],
        channels=spec["channels"], chips=spec["chips"], dies=1, planes=spec["planes"],
        blocks_per_plane=spec["blocks"], pages_per_block=spec["ppb"],
        aggregate_gb_s=spec["gb_s"],
        program_ns=spec["t_prog"], erase_ns=spec["t_bers"],
        coalesce_ns=spec.get("coalesce_ns", 5_000), read_plane_ns=spec.get("read_plane_ns", 0),
        cache_read=spec.get("cache_read", True),
    )
    units = cfg.units
    ppb = cfg.pages_per_block
    layers = spec["layers"]
    gap_ns = spec["gap_ns"]
    wave_banks = spec["wave_banks"]
    prog_waves = spec["prog_waves"]
    erase_waves = spec["erase_waves"]

    # Prefill step -1: write block 0 (pages 0..layers-1) for reads, and if erase_waves > 0 also
    # write page 0 of target erase block(s) (block 1..) so rewriting page 0 triggers T-HBFW2 auto-erase.
    t = 0
    prefill = []
    for p in range(layers):
        reqs = tuple(Request(u + units * p, "W") for u in range(units))
        prefill.append(Batch(-1, -1, len(prefill), t, reqs))
        t += 10_000_000
    for e in range(erase_waves):
        blk = 1 + e
        reqs = tuple(Request(u + units * (blk * ppb), "W") for u in range(wave_banks))
        prefill.append(Batch(-1, -1, len(prefill), t, reqs))
        t += 10_000_000

    # Measure solo read duration of one layer.
    probe_dir = tmp_root / f"{name}-probe"
    probe_batches = prefill + [Batch(0, 0, 0, t, tuple(Request(u, "R") for u in range(units)))]
    simulate(cfg, probe_batches, probe_dir, binary, hbf_write=True)
    layer_read_solo_ns = _read_metrics(probe_dir / "completions.csv", [probe_batches[-1]])[0][0]

    # Build step-0 trace: L layers; each layer l has an HBF-idle gap gap_ns before its read.
    # For multi-layer traces where a write overflows a single layer's gap, we space layers by
    # max(gap_ns, layer_pad_ns) when independent_layers=True so each layer's stall is measured
    # cleanly without artificial open-loop timestamp pile-up, while writes STILL start exactly
    # `gap_ns` before that layer's read!
    step_start = t + 20_000_000
    ro_reads = []
    ww_events = []

    write_slots = []
    next_prog_page = 0
    for e in range(erase_waves):
        blk = 1 + e
        write_slots.append(("E", tuple(Request(u + units * (blk * ppb), "W") for u in range(wave_banks))))
    for _ in range(prog_waves - erase_waves):
        blk = 1 + erase_waves + (next_prog_page // ppb)
        pg = next_prog_page % ppb
        next_prog_page += 1
        write_slots.append(("W", tuple(Request(u + units * (blk * ppb + pg), "W") for u in range(wave_banks))))

    by_layer_writes = {l: [] for l in range(layers)}
    for idx, item in enumerate(write_slots):
        target_l = 0 if spec.get("burst_in_layer0") else (idx % layers)
        by_layer_writes[target_l].append(item)

    # Space layer windows generously enough that one layer's overflow never bleeds into the next
    # layer's read timestamp, while keeping the write-to-read lead time equal to exactly `gap_ns`
    # (or 200 ns before read when gap_ns == 0 so the write arrives immediately ahead of the read).
    window_stride = max(gap_ns + layer_read_solo_ns + 50_000, 5_000_000)
    for l in range(layers):
        read_arrival = step_start + l * window_stride + max(gap_ns, 50_000)
        if by_layer_writes[l]:
            lead = gap_ns if gap_ns > 0 else 200
            w_t = read_arrival - lead
            for k_idx, (_, reqs) in enumerate(by_layer_writes[l]):
                # Space consecutive burst waves by 100 ns (within coalesce window? No: outside coalesce_ns
                # if we want distinct waves, or distinct page offsets on the same block so T-PLANE lockstep
                # naturally issues them as separate multi-plane program commands!).
                ww_events.append(Batch(0, l, k_idx, w_t + k_idx * 10, reqs))
        r_batch = Batch(0, l, 100, read_arrival, tuple(Request(u + units * l, "R") for u in range(units)))
        ro_reads.append(r_batch)
        ww_events.append(r_batch)

    ww_events.sort(key=lambda b: b.arrival_ns)

    # 1. Read-only run (exact same read trace, no step-0 writes)
    ro_dir = tmp_root / f"{name}-ro"
    simulate(cfg, prefill + ro_reads, ro_dir, binary, hbf_write=True)
    ro_durs, _ = _read_metrics(ro_dir / "completions.csv", ro_reads)

    # 2. With-writes (fifo)
    fifo_dir = tmp_root / f"{name}-fifo"
    _, _, _, fifo_stats = simulate(cfg, prefill + ww_events, fifo_dir, binary, hbf_write=True, write_order="fifo")
    fifo_durs, _ = _read_metrics(fifo_dir / "completions.csv", ro_reads)

    # 3. With-writes (read_first)
    rf_dir = tmp_root / f"{name}-rf"
    _, _, _, rf_stats = simulate(cfg, prefill + ww_events, rf_dir, binary, hbf_write=True, write_order="read_first")
    rf_durs, _ = _read_metrics(rf_dir / "completions.csv", ro_reads)

    mqsim_ro_ns = sum(ro_durs)
    mqsim_fifo_extra = sum(a - b for a, b in zip(fifo_durs, ro_durs))
    mqsim_rf_extra = sum(a - b for a, b in zip(rf_durs, ro_durs))

    dies = cfg.channels * cfg.chips * cfg.dies
    banks_per_die = cfg.planes
    step_in = {
        "programs": prog_waves * wave_banks,
        "erases": erase_waves * wave_banks,
        "wave_banks": wave_banks,
        "gaps_ns": [gap_ns] * layers,
    }
    an_fifo = interference(step_in, "fifo", cfg.program_ns, cfg.erase_ns, dies=dies, banks_per_die=banks_per_die)
    an_gaps = interference(step_in, "gaps", cfg.program_ns, cfg.erase_ns, dies=dies, banks_per_die=banks_per_die)

    return {
        "point": name,
        "dies": dies,
        "planes": banks_per_die,
        "layers": layers,
        "coalesce_ns": cfg.coalesce_ns,
        "t_prog_ns": cfg.program_ns,
        "t_bers_ns": cfg.erase_ns,
        "gap_ns": gap_ns,
        "wave_banks": wave_banks,
        "prog_waves": prog_waves,
        "erase_waves": erase_waves,
        "ro_step_ns": mqsim_ro_ns,
        "mqsim_fifo_extra_ns": mqsim_fifo_extra,
        "analytic_fifo_ns": round(an_fifo["extra_step_ns"]),
        "mqsim_read_first_extra_ns": mqsim_rf_extra,
        "analytic_gaps_ns": round(an_gaps["extra_step_ns"]),
        "analytic_gaps_max_ns": round(an_gaps["max_step_extra_ns"]),
        "prog_cmds_fifo": fifo_stats["program_commands"] - len(prefill) * (dies if wave_banks == units else 1),
        "erases_issued": fifo_stats["erase_commands"],
    }


POINTS = [
    # 1. Program fits inside idle gap (gap 200 us > tPROG 100 us)
    {"name": "P1_prog_fits_gap", "topology": "ocp", "channels": 2, "chips": 2, "planes": 16,
     "blocks": 8, "ppb": 16, "gb_s": 100, "layers": 4, "t_prog": 100_000, "t_bers": 1_000_000,
     "gap_ns": 200_000, "wave_banks": 64, "prog_waves": 1, "erase_waves": 0},
    # 2. Program overflows tight idle gap (gap 40 us < tPROG 100 us -> +60 us overflow)
    {"name": "P2_prog_overflows_gap", "topology": "ocp", "channels": 2, "chips": 2, "planes": 16,
     "blocks": 8, "ppb": 16, "gb_s": 100, "layers": 4, "t_prog": 100_000, "t_bers": 1_000_000,
     "gap_ns": 40_000, "wave_banks": 64, "prog_waves": 1, "erase_waves": 0},
    # 3. Two back-to-back program waves in layer 0, 10 us gap (wave 1 already running: fifo waits for 2, read_first preempts 2nd)
    {"name": "P3_prog_burst_tight", "topology": "ocp", "channels": 2, "chips": 2, "planes": 16,
     "blocks": 8, "ppb": 16, "gb_s": 100, "layers": 4, "t_prog": 100_000, "t_bers": 1_000_000,
     "gap_ns": 10_000, "wave_banks": 64, "prog_waves": 2, "erase_waves": 0, "burst_in_layer0": True},
    # 4. Calibrated tPROG=350 us, 2 spread waves across 4 layers with 100 us gap (2 x 250 us = 500 us)
    {"name": "P4_prog_350us_spread", "topology": "ocp", "channels": 2, "chips": 2, "planes": 16,
     "blocks": 8, "ppb": 16, "gb_s": 100, "layers": 4, "t_prog": 350_000, "t_bers": 3_500_000,
     "gap_ns": 100_000, "wave_banks": 64, "prog_waves": 2, "erase_waves": 0},
    # 5. Erase + program fits inside large gap (1.2 ms > 1.0 ms + 0.1 ms)
    {"name": "P5_erase_fits_gap", "topology": "ocp", "channels": 2, "chips": 2, "planes": 16,
     "blocks": 8, "ppb": 16, "gb_s": 100, "layers": 4, "t_prog": 100_000, "t_bers": 1_000_000,
     "gap_ns": 1_200_000, "wave_banks": 64, "prog_waves": 1, "erase_waves": 1},
    # 6. Erase + program overflows gap (400 us < 1 ms erase -> fifo +700 us, read_first +600 us)
    {"name": "P6_erase_overflows_gap", "topology": "ocp", "channels": 2, "chips": 2, "planes": 16,
     "blocks": 8, "ppb": 16, "gb_s": 100, "layers": 4, "t_prog": 100_000, "t_bers": 1_000_000,
     "gap_ns": 400_000, "wave_banks": 64, "prog_waves": 1, "erase_waves": 1},
    # 7. Partial-die wave (1 of 4 dies = 16 banks, 10 us gap): worst-die stall (~100 us) vs mean unit_ops (25 us)
    {"name": "P7_partial_1die_wave", "topology": "ocp", "channels": 2, "chips": 2, "planes": 16,
     "blocks": 8, "ppb": 16, "gb_s": 100, "layers": 4, "t_prog": 100_000, "t_bers": 1_000_000,
     "gap_ns": 10_000, "wave_banks": 16, "prog_waves": 1, "erase_waves": 0},
    # 8. Full-scale 64-die (1,024-bank) OCP stack: 2 program waves, 50 us gap, 4 layers (2 x 50 us = 100 us)
    {"name": "P8_full_ocp_64die", "topology": "ocp", "channels": 32, "chips": 2, "planes": 16,
     "blocks": 8, "ppb": 16, "gb_s": 1600, "layers": 4, "t_prog": 100_000, "t_bers": 1_000_000,
     "gap_ns": 50_000, "wave_banks": 1024, "prog_waves": 2, "erase_waves": 0},
    # 9. Diagnostic: P2 with default ocp coalesce_ns=1 ns (shows write-DMA multi-plane command splitting)
    {"name": "P9_p2_coalesce_1ns", "topology": "ocp", "channels": 2, "chips": 2, "planes": 16,
     "blocks": 8, "ppb": 16, "gb_s": 100, "layers": 4, "t_prog": 100_000, "t_bers": 1_000_000,
     "gap_ns": 40_000, "wave_banks": 64, "prog_waves": 1, "erase_waves": 0, "coalesce_ns": 1},
]


def main():
    binary = build()
    rows = []
    with tempfile.TemporaryDirectory(prefix="hbf-r5-interf-") as tmp:
        tmp_root = Path(tmp)
        for spec in POINTS:
            row = run_point(binary, tmp_root, spec)
            rows.append(row)
            print(f"{row['point']:24s}  MQSim(fifo)={row['mqsim_fifo_extra_ns']:>9d}  "
                  f"an(fifo)={row['analytic_fifo_ns']:>9d}  "
                  f"MQSim(rf)={row['mqsim_read_first_extra_ns']:>9d}  "
                  f"an(gaps)={row['analytic_gaps_ns']:>9d}")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
