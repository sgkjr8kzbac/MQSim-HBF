"""Write/erase interference (hbf/interference.py, M5): hand-computed values of design-B section 7."""
import csv
import json
import math

import pytest

from hbf.interference import hbf_gaps, interference, main, pack, residual_max, retime, unit_ops
from hbf.system import timeline

T_PROG, T_BERS = 100_000, 1_000_000
G = 7_864.32 + 60_293.12  # pre + post of Llama-3.1-8B at B = 8 on B200 (design-B section 5.4)
ROW = 1024  # pages of one row-striped write wave


def load(programs=0.0, erases=0.0, **extra):
    return {"programs": programs, "erases": erases, "gap_ns": G, "layers": 32} | extra


def test_17_fifo_one_wave():
    result = interference(load(ROW), "fifo", T_PROG, T_BERS)
    assert result["extra_step_ns"] == pytest.approx(100_000)
    assert result["extra_layer_ns"] == pytest.approx([100_000 / 32] * 32)
    assert result["program_ops"] == 1


def test_18_gaps_one_operation_per_layer():
    result = interference(load(32 * ROW), "gaps", T_PROG, T_BERS)
    assert result["extra_layer_ns"] == pytest.approx([31_842.56] * 32)
    # Design-B section 5.4: decode flushes alone (256 pages per step = 0.25 waves).
    assert interference(load(256), "gaps", T_PROG, T_BERS)["extra_step_ns"] == pytest.approx(0.25 * 31_842.56)
    assert interference(load(256), "fifo", T_PROG, T_BERS)["extra_step_ns"] == pytest.approx(25_000)
    # Refresh at X = 1e5: 21.5 waves per step.
    assert interference(load(21.5 * ROW), "gaps", T_PROG, T_BERS)["extra_step_ns"] == pytest.approx(
        21.5 * 31_842.56)
    assert interference(load(21.5 * ROW), "fifo", T_PROG, T_BERS)["extra_step_ns"] == pytest.approx(2_150_000)


def test_19_suspend():
    idle = 32 * G
    assert idle == pytest.approx(2_181_038.08)
    under = interference(load(21.5 * ROW), "suspend", T_PROG, T_BERS, suspend_ns=5_000)
    assert under["extra_step_ns"] == pytest.approx(160_000)
    assert under["overflow_ns"] == 0
    over = interference(load(30 * ROW), "suspend", T_PROG, T_BERS, suspend_ns=5_000)
    assert over["extra_step_ns"] == pytest.approx(160_000 + 3_000_000 - idle)
    assert over["extra_step_ns"] - 160_000 == pytest.approx(818_962, abs=1)
    with pytest.raises(ValueError, match="suspend"):
        interference(load(ROW), "suspend", T_PROG, T_BERS)


def test_20_random_and_correlated():
    # D = 64 dies, u = 0.01, T = 100 us; the design rounded the result to 26.2055 us.
    expected = 100 - 100 / (0.01 * 65) * (1 - 0.99 ** 65)
    assert residual_max(0.01, 100, 64) == pytest.approx(expected) == pytest.approx(26.2062, abs=1e-3)
    step = 1e8  # 100 ms step: one program wave per step keeps each die busy u = 0.001 of the time
    result = interference(load(ROW, step_ns=step / 10), "random", T_PROG, T_BERS)
    assert result["extra_layer_ns"][0] == pytest.approx(residual_max(0.01, T_PROG, 64))
    correlated = interference(load(ROW, step_ns=step / 10), "correlated", T_PROG, T_BERS)
    assert correlated["extra_layer_ns"][0] == pytest.approx(0.01 * T_PROG / 2) == pytest.approx(500)
    with pytest.raises(ValueError, match="step_ns"):
        interference(load(ROW), "random", T_PROG, T_BERS)


def test_wave_width_and_blocking_unit():
    # Waves of 64 or 16 banks (channel or die partitions) load each die like a full row ...
    assert unit_ops(ROW, 1024, 64, 16) == unit_ops(ROW, 64, 64, 16) == unit_ops(ROW, 16, 64, 16) == 1
    # ... but single-bank waves (per-bank replay) block a whole die for one bank's page.
    assert unit_ops(ROW, 1, 64, 16) == 16
    assert unit_ops(ROW, 1, 1024, 1) == 1  # bank-level blocking (optimistic ablation)
    die = interference(load(64, wave_banks=1), "fifo", T_PROG, T_BERS)
    bank = interference(load(64, wave_banks=1), "fifo", T_PROG, T_BERS, block_unit="bank")
    assert die["extra_step_ns"] == pytest.approx(16 * bank["extra_step_ns"]) == pytest.approx(T_PROG)


def test_erases_and_spikes():
    # One superblock erase (1,024 blocks) every 10 steps: 0.1 erase waves per step.
    result = interference(load(erases=102.4), "gaps", T_PROG, T_BERS)
    assert result["extra_step_ns"] == pytest.approx(0.1 * (T_BERS - G))
    assert result["max_step_extra_ns"] == pytest.approx(T_BERS - G)
    # An unspread superblock refresh (256 rows) in one step under gaps: 256 waves into 32 gaps.
    spike = interference(load(256, burst_programs=256 * ROW), "gaps", T_PROG, T_BERS)
    assert spike["max_step_extra_ns"] == pytest.approx(32 * (T_PROG - G) + (257 - 32) * T_PROG)
    # Fifo erases cost tBERS each.
    assert interference(load(erases=ROW), "fifo", T_PROG, T_BERS)["extra_step_ns"] == pytest.approx(T_BERS)


def test_pack_uses_large_gaps_first():
    gaps = [250, 50, 150]
    extra, left = pack(gaps, 2, 100)  # two free slots in the 250 gap
    assert list(extra) == [0, 0, 0] and list(left) == [50, 50, 150]
    extra, left = pack(gaps, 4, 100)  # 3 free (2 + 0 + 1); the 4th overflows the cheapest gap (cost 50)
    assert sum(extra) == pytest.approx(50)
    extra, _ = pack(gaps, 7, 100)  # 3 free, 3 overflow (50 + 50 + 50), 1 more at 100
    assert sum(extra) == pytest.approx(250)


@pytest.mark.parametrize("overlap,staged", [("serial", False), ("serial", True), ("prefetch:1", True),
                                            ("ideal", False)])
def test_gaps_sum_to_idle_time(overlap, staged):
    h, pre, attn, post = [300, 100, 200], [20, 20, 20], [50, 400, 10], [30, 30, 30]
    total, _, _ = timeline(h, pre, attn, post, overlap, staged)
    gaps = hbf_gaps(h, pre, attn, post, overlap, staged, tail_ns=77)
    assert len(gaps) == 3 and min(gaps) >= 0
    assert sum(gaps) == pytest.approx(total + 77 - sum(h))


def write_run(directory, h=200_000.0, pre=7_864.32, post=60_293.12, layers=32, lm_head=164_000.0):
    directory.mkdir()
    (directory / "meta.json").write_text(json.dumps({"system": {"overlap": "serial", "kv_consume": "stream"}}))
    rows = [{"step": 0, "layer": l, "hbf_est_ns": h, "hbf_bound_ns": h, "pre_ns": pre, "attn_ns": 1_000.0,
             "attn_staged_ns": 2_000.0, "post_ns": post} for l in range(layers)]
    with (directory / "system_layers.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tpot = layers * (pre + h + post) + lm_head
    with (directory / "system_steps.csv").open("w", newline="") as stream:
        stream.write(f"step,tpot_hbf_est_ns,tpot_hbf_bound_ns\n0,{tpot},{tpot}\n")
    return tpot


def test_retime_fixed_point(tmp_path):
    tpot = write_run(tmp_path / "run")
    # Step-driven: 0.25 waves per step under fifo.
    row = retime(tmp_path / "run", mode="fifo", programs_per_step=256)[0]
    assert row["tpot_m5_ns"] == pytest.approx(tpot + 25_000)
    # Time-driven: 1 GB/s of programs; under fifo TPOT' = TPOT + (B/s x TPOT' / 4096 / 1024) x tPROG.
    rate = 1e9 / 4096 / 1024 * T_PROG / 1e9  # extra ns per ns of TPOT'
    row = retime(tmp_path / "run", mode="fifo", write_gb_s=1.0)[0]
    assert row["tpot_m5_ns"] == pytest.approx(tpot / (1 - rate), rel=1e-5)
    # A load beyond what the device can absorb has no fixed point.
    assert math.isinf(retime(tmp_path / "run", mode="fifo", write_gb_s=50.0)[0]["tpot_m5_ns"])


def test_cli_writes_steps(tmp_path):
    write_run(tmp_path / "run")
    rows = main([str(tmp_path / "run"), "--write-gb-s", "0.5", "--erases-per-s", "100", "--mode", "suspend",
                 "--suspend-ns", "5000", "--output", str(tmp_path / "out")])
    assert (tmp_path / "out" / "interference_steps.csv").is_file()
    assert "写入与擦除" in (tmp_path / "out" / "interference_report.md").read_text()
    assert rows[0]["tpot_m5_ns"] > rows[0]["tpot_ns"]


def test_suspend_zero_gap_scales_with_fractional_ops():
    # M4: with mean_gap == 0 and ops < 1, suspended layers scale by ops (continuous with gap -> 0+).
    zero = interference(load(0.1 * ROW, gap_ns=0.0), "suspend", T_PROG, T_BERS, suspend_ns=5_000)
    tiny = interference(load(0.1 * ROW, gap_ns=1e-6), "suspend", T_PROG, T_BERS, suspend_ns=5_000)
    assert zero["extra_step_ns"] == pytest.approx(0.1 * 32 * 5_000 + 0.1 * T_PROG)
    assert zero["extra_step_ns"] == pytest.approx(tiny["extra_step_ns"], rel=1e-6)


def test_retime_high_duty_convergence_and_inf_max_step(tmp_path):
    # M4: high-duty finite fixed point (rate = 0.95 -> TPOT' = 20 x TPOT) converges; non-converged
    # load reports inf for max_step_extra_ns as well as tpot_m5_ns and extra_step_ns.
    tpot = write_run(tmp_path / "run")
    write_gb_s = 0.95 / (1e9 / 4096 / 1024 * T_PROG / 1e9)
    high = retime(tmp_path / "run", mode="fifo", write_gb_s=write_gb_s)[0]
    assert high["tpot_m5_ns"] == pytest.approx(tpot / 0.05, rel=1e-4)
    diverged = retime(tmp_path / "run", mode="fifo", write_gb_s=50.0)[0]
    assert math.isinf(diverged["tpot_m5_ns"]) and math.isinf(diverged["extra_step_ns"])
    assert math.isinf(diverged["max_step_extra_ns"])
