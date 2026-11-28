"""Run MQSim with trace validation, GC-budget guard, completion joins, and artifacts."""
import os
import signal
import subprocess
import threading
import time
from collections import defaultdict
from contextlib import suppress
from dataclasses import asdict

from ...device.config import device_xml, workload_xml
from ...util import dump, write_table
from .build import sha256, version
from .parse import aggregate, completions, flash_stats
from .schedule import metrics, unwritten_reads

GNU_TIME = "/usr/bin/time"


def run_timed(command, directory, env, timeout):
    if not os.access(GNU_TIME, os.X_OK):
        raise RuntimeError(f"GNU time is required at {GNU_TIME} to record peak RSS "
                           "(install the 'time' package)")
    timed_command = [GNU_TIME, "-f", "%M", "-o", "rss_kib.txt", *command]
    process = None
    terminated = False

    def stop_group():
        if process is not None and process.returncode is None:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)

    def on_sigterm(signum, frame):
        nonlocal terminated
        terminated = True
        stop_group()

    previous = None
    if threading.current_thread() is threading.main_thread():
        previous = signal.getsignal(signal.SIGTERM)
        if previous not in (None, signal.SIG_IGN):
            signal.signal(signal.SIGTERM, on_sigterm)
        else:
            previous = None
    try:
        with (directory / "stdout.log").open("w") as log, subprocess.Popen(
            timed_command, cwd=directory, env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=log, start_new_session=True,
        ) as child:
            process = child
            if terminated:
                stop_group()
            try:
                returncode = child.wait(timeout=timeout)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                stop_group()
                child.wait()
                raise
    finally:
        if previous is not None:
            signal.signal(signal.SIGTERM, previous)
            if terminated:
                signal.raise_signal(signal.SIGTERM)
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, timed_command)


def check_gc_budget(cfg, batches):
    """Reject runs whose fresh page allocations could reach MQSim's GC threshold."""
    seen, used = set(), [0] * cfg.units
    for batch in batches:
        for req in batch.requests:
            if req.op == "W" or req.lpa not in seen:
                used[cfg.unit(req.lpa)] += 1
                seen.add(req.lpa)
    worst = max(range(cfg.units), key=used.__getitem__)
    if used[worst] > cfg.gc_page_budget:
        raise ValueError(f"Unit {worst} would allocate {used[worst]} fresh pages, above the "
                         f"GC-free budget of {cfg.gc_page_budget} pages per plane")
    return used[worst]


def simulate(cfg, batches, directory, binary, *, physical=False, suspension=False, onfi=False,
             overlap=False, timeout=1200, allow_unwritten=False, hbf_write=False,
             hbfw1=None, hbfw2=None, hbfw3=None, hbfw4=None, write_order=None):
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "trace.txt").exists():
        raise ValueError(f"Run directory already contains a trace: {directory}")
    binary = binary.resolve()
    use_w1 = hbf_write if hbfw1 is None else bool(hbfw1)
    use_w2 = hbf_write if hbfw2 is None else bool(hbfw2)
    use_w3 = hbf_write if hbfw3 is None else bool(hbfw3)
    use_w4 = hbf_write if hbfw4 is None else bool(hbfw4)
    if write_order not in {None, "read_first", "fifo"}:
        raise ValueError(f"write_order must be None, 'read_first', or 'fifo', got {write_order!r}")
    allow_overlap = overlap or hbf_write
    if suspension and (cfg.dies != 1 or cfg.planes != 1 or cfg.suspend != "PROGRAM"):
        raise ValueError("T-SUSP probe requires PROGRAM with one die and plane per chip")
    if not (directory / "meta.json").exists():
        dump(directory / "meta.json", {"source": "page-level device probe", "batch_count": len(batches)})
    if cfg.onfi is not None and not onfi:
        raise ValueError("ONFI XML overrides require T-ONFI")
    device_xml(cfg, directory / "device.xml")
    workload_xml(cfg, directory / "workload.xml")
    if not batches or not all(batch.requests for batch in batches):
        raise ValueError("At least one nonempty batch is required")
    if not use_w3:
        check_gc_budget(cfg, batches)
    unwritten = unwritten_reads(cfg, batches)
    if unwritten and cfg.planes > 1 and not allow_unwritten:
        raise ValueError(f"{unwritten} reads of pages that were never written; the OCP spec answers them "
                         "with error 0x7 and MQSim would place them at first-read offsets. Use --prefill "
                         "(device-level probes that only time commands may pass allow_unwritten=True)")
    expected = {}
    reads = writes = 0
    with (directory / "trace.txt").open("w") as stream:
        previous = -1
        for idx, batch in enumerate(batches):
            if batch.arrival_ns < 0 or batch.arrival_ns <= previous:
                raise ValueError("Batch timestamps must strictly increase")
            previous = batch.arrival_ns
            for req in batch.requests:
                lba = cfg.lba(req.lpa)
                key = batch.arrival_ns, lba
                if key in expected or req.op not in {"R", "W"}:
                    raise ValueError(f"Invalid or duplicate request {key}")
                expected[key] = idx, req.op
                reads += req.op == "R"
                writes += req.op == "W"
                stream.write(f"{batch.arrival_ns} 0 {lba} {cfg.spp} {int(req.op == 'R')}\n")

    flags = {"I1": "completions.csv", "I2": "physical.csv" if physical else None,
             "I3": "1", "I4": "1", "T_SUSP": "1" if suspension else None,
             "T_ONFI": "1" if onfi else None, "T_PLANE": cfg.plane_rule,
             "T_COALESCE_NS": None if cfg.coalesce_ns is None else str(cfg.coalesce_ns),
             "T_READ_PLANE_NS": None if cfg.read_plane_ns is None else str(cfg.read_plane_ns),
             "T_CACHE": "1" if cfg.cache_read else None,
             "T_HBFW1": "1" if use_w1 else None,
             "T_HBFW2": "1" if use_w2 else None,
             "T_HBFW3": "1" if use_w3 else None,
             "T_HBFW4": "block_counts.csv" if use_w4 else None,
             "WRITE_ORDER": write_order}
    env = {key: value for key, value in os.environ.items() if not key.startswith("HBF_")}
    env.update({f"HBF_{key}": value for key, value in flags.items() if value is not None})
    command = [str(binary), "-i", "device.xml", "-w", "workload.xml"]
    dump(directory / "parameters.json", asdict(cfg))
    dump(directory / "version.json", {
        **version([name.replace("_", "-") for name, val in flags.items() if val is not None]),
        "binary_sha256": sha256(binary), "mqsim_command": command, "mqsim_env": flags,
    })
    start = time.perf_counter()
    run_timed(command, directory, env, timeout)
    elapsed = time.perf_counter() - start
    stats = flash_stats(directory / "workload_scenario_1.xml")
    stats.update(wall_seconds=elapsed, peak_rss_kib=int((directory / "rss_kib.txt").read_text()),
                 unwritten_reads=unwritten)
    dump(directory / "statistics.json", stats)
    multi = cfg.planes > 1
    read_cmds = stats["Issued_Flash_Read_CMD"] + stats["Issued_Flash_Multiplane_Read_CMD"]
    program_cmds = stats["Issued_Flash_Program_CMD"] + stats["Issued_Flash_Multiplane_Program_CMD"]
    erase_cmds = stats.get("Issued_Flash_Erase_CMD", 0) + stats.get("Issued_Flash_Multiplane_Erase_CMD", 0)
    if (stats["request_count"] != len(expected)
            or not (multi or stats["Issued_Flash_Read_CMD"] == reads and stats["Issued_Flash_Program_CMD"] == writes)
            or multi and not (stats["Issued_Flash_Read_CMD"] + 2 * stats["Issued_Flash_Multiplane_Read_CMD"]
                              <= reads <= stats["Issued_Flash_Read_CMD"]
                              + cfg.planes * stats["Issued_Flash_Multiplane_Read_CMD"])
            or multi and not (stats["Issued_Flash_Program_CMD"] + 2 * stats["Issued_Flash_Multiplane_Program_CMD"]
                              <= writes <= stats["Issued_Flash_Program_CMD"]
                              + cfg.planes * stats["Issued_Flash_Multiplane_Program_CMD"])):
        raise ValueError(f"MQSim statistics do not match trace request/read/write counts: {stats}")
    stats["read_commands"], stats["program_commands"], stats["erase_commands"] = read_cmds, program_cmds, erase_cmds
    if use_w2:
        if stats.get("Total_GC_Executions", 0) or stats.get("gc_read_write_enqueued", 0):
            raise ValueError(f"Garbage collection ran during the experiment: {stats}")
    elif stats["Issued_Flash_Erase_CMD"] or stats["gc_enqueued"]:
        raise ValueError(f"Garbage collection ran during the experiment: {stats}")

    last = [0] * len(batches)
    for done in completions(directory / "completions.csv"):
        idx, op = expected.pop((done.arrival_ns, done.start_lba))
        if not done.arrival_ns <= done.enqueue_ns <= done.complete_ns or done.lba_count != cfg.spp or done.op != op:
            raise ValueError(f"Invalid completion: {done}")
        last[idx] = max(done.complete_ns, last[idx])
    if expected:
        raise ValueError(f"Missing {len(expected)} completions")
    if not allow_overlap and any(end >= batch.arrival_ns for end, batch in zip(last, batches[1:])):
        raise ValueError("Overlapping sub-batches: detected completion at or after the next arrival")

    rows, groups, by_step = [], defaultdict(list), defaultdict(list)
    for batch, done in zip(batches, last):
        timing = done - batch.arrival_ns
        row = {"step": batch.step, "layer": batch.layer, "subbatch": batch.subbatch,
               "arrival_ns": batch.arrival_ns, "complete_ns": done, "mqsim_ns": timing,
               **metrics(cfg, batch.requests)}
        row["mqsim_over_est"] = timing / row["mq_est_ns"] if row["mq_est_ns"] else None
        rows.append(row)
        groups[batch.step, batch.layer].append((batch, row))
        by_step[batch.step].append((batch, row))
    layers = [aggregate(cfg, group, {"step": step, "layer": layer}) for (step, layer), group in groups.items()]
    steps = [aggregate(cfg, group, {"step": step}) for step, group in by_step.items()]
    write_table(directory / "batches.csv", rows)
    write_table(directory / "layers.csv", layers)
    write_table(directory / "steps.csv", steps)
    return rows, layers, steps, stats
