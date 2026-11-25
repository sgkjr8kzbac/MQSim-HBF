"""Pinned build, trace validation, simulation, completion joins and artifacts."""
import csv
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import threading
import time
from collections import defaultdict
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

from .config import PACKAGE, ROOT, device_xml, workload_xml
from .controller import metrics, unwritten_reads

PIN = (PACKAGE / "mqsim.commit").read_text().strip()
MQSIM_SOURCE = PACKAGE / "mqsim"
MQSIM_MANIFEST = PACKAGE / "mqsim.sha256"


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def write_table(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def patch_series():
    """Patches in the order of patches/series; every patch file must be listed exactly once."""
    directory = PACKAGE / "patches"
    names = [line.strip() for line in (directory / "series").read_text().splitlines()
             if line.strip() and not line.startswith("#")]
    if sorted(names) != sorted(p.name for p in directory.glob("*.patch")) or len(set(names)) != len(names):
        raise ValueError("patches/series must list every patch file exactly once")
    return [directory / name for name in names]


def build(cache=None):
    cache = Path(cache or Path.cwd() / ".build").resolve()
    patches = patch_series()
    manifest = MQSIM_MANIFEST.read_bytes()
    source_files = []
    for line in manifest.decode().splitlines():
        expected, relative_path = line.split("  ", 1)
        upstream_file = MQSIM_SOURCE / relative_path
        if not upstream_file.is_file() or sha256(upstream_file) != expected:
            raise ValueError(f"Pinned MQSim source integrity check failed: {relative_path}")
        source_files.append(relative_path)
    build_inputs = manifest + (PIN + "".join(map(sha256, patches))).encode()
    digest = hashlib.sha256(build_inputs).hexdigest()[:16]
    source = cache / digest
    binary = source / "MQSim"
    ready = source / ".built"
    if binary.is_file() and ready.is_file():
        return binary
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / "build.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if binary.is_file() and ready.is_file():
            return binary
        if source.exists():
            shutil.rmtree(source)
        source.mkdir()
        for relative_path in source_files:
            target = source / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(MQSIM_SOURCE / relative_path, target)
        with (source / "build.log").open("w") as log:
            subprocess.run(["git", "apply", *map(str, patches)], cwd=source, check=True, stdout=log, stderr=log,
                           env={**os.environ, "GIT_CEILING_DIRECTORIES": str(cache)})
            subprocess.run(["make", "-j", str(min(8, os.cpu_count() or 1))], cwd=source,
                           check=True, stdout=log, stderr=log)
        ready.touch()
    return binary


@dataclass(frozen=True)
class Completion:
    arrival_ns: int
    enqueue_ns: int
    complete_ns: int
    start_lba: int
    lba_count: int
    op: str


def completions(path):
    with path.open(newline="") as stream:
        for arrival, enqueue, done, lba, size, op in csv.reader(stream):
            yield Completion(int(arrival), int(enqueue), int(done), int(lba), int(size), op)


def flash_stats(path):
    tree = ET.parse(path)
    ftl = tree.find(".//SSDDevice.FTL")
    if ftl is None:
        raise ValueError("MQSim output is missing FTL statistics")
    stats = {key: int(val) for key, val in ftl.attrib.items() if key.startswith("Issued_Flash")}
    stats["request_count"] = int(tree.findtext(".//Request_Count"))
    stats["gc_enqueued"] = sum(
        int(el.get("No_Of_Transactions_Enqueued"))
        for el in tree.iter() if ".GC_" in el.tag and el.get("No_Of_Transactions_Enqueued") is not None
    )
    return stats


def version(patches):
    rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                            cwd=ROOT, capture_output=True, text=True)
    return {"mqsim_commit": PIN, "repository_commit": rev.stdout.strip() or None,
            "working_tree_modified": bool(status.stdout.strip()), "patches_enabled": patches,
            "patch_sha256": {p.name: sha256(p) for p in sorted((PACKAGE / "patches").glob("*.patch"))}}


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

    # Python installs signal handlers only on the main thread. Keep the caller's
    # disposition and restore it before delivering SIGTERM after cleanup.
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
    """Reject runs whose fresh page allocations could reach MQSim's GC threshold.

    Every write and every first read of an LPA (online_create_entry_for_reads) consumes one
    free physical page on its plane. Upstream MQSim livelocks instead of finishing once the
    free block pool falls below the GC threshold with only fully valid blocks.
    """
    seen, used = set(), [0] * cfg.units
    for batch in batches:
        for req in batch.requests:
            if req.op == "W" or req.lpa not in seen:
                used[req.lpa % cfg.units] += 1
                seen.add(req.lpa)
    worst = max(range(cfg.units), key=used.__getitem__)
    if used[worst] > cfg.gc_page_budget:
        raise ValueError(f"Unit {worst} would allocate {used[worst]} fresh pages, above the "
                         f"GC-free budget of {cfg.gc_page_budget} pages per plane")
    return used[worst]


def simulate(cfg, batches, directory, binary, *, physical=False, suspension=False, onfi=False,
             overlap=False, timeout=1200, allow_unwritten=False):
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "trace.txt").exists():
        raise ValueError(f"Run directory already contains a trace: {directory}")
    binary = binary.resolve()
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
    check_gc_budget(cfg, batches)
    unwritten = unwritten_reads(batches)
    if unwritten and cfg.planes > 1 and not allow_unwritten:
        # One plane per chip makes the offset irrelevant; with multi-plane commands MQSim's first-read
        # allocation would put the pages at offsets the OCP spec does not allow (Table 12, error 0x7).
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
             "T_CACHE": "1" if cfg.cache_read else None}
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
    # A multi-plane command serves 2..planes pages; MQSim counts commands, not pages, for it.
    multi = cfg.planes > 1
    read_cmds = stats["Issued_Flash_Read_CMD"] + stats["Issued_Flash_Multiplane_Read_CMD"]
    program_cmds = stats["Issued_Flash_Program_CMD"] + stats["Issued_Flash_Multiplane_Program_CMD"]
    if (stats["request_count"] != len(expected)
            or not (multi or stats["Issued_Flash_Read_CMD"] == reads and stats["Issued_Flash_Program_CMD"] == writes)
            or multi and not (stats["Issued_Flash_Read_CMD"] + 2 * stats["Issued_Flash_Multiplane_Read_CMD"]
                              <= reads <= stats["Issued_Flash_Read_CMD"]
                              + cfg.planes * stats["Issued_Flash_Multiplane_Read_CMD"])
            or multi and not (stats["Issued_Flash_Program_CMD"] + 2 * stats["Issued_Flash_Multiplane_Program_CMD"]
                              <= writes <= stats["Issued_Flash_Program_CMD"]
                              + cfg.planes * stats["Issued_Flash_Multiplane_Program_CMD"])):
        raise ValueError(f"MQSim statistics do not match trace request/read/write counts: {stats}")
    stats["read_commands"], stats["program_commands"] = read_cmds, program_cmds
    if stats["Issued_Flash_Erase_CMD"] or stats["gc_enqueued"]:
        raise ValueError(f"Garbage collection ran during the experiment: {stats}")

    last = [0] * len(batches)
    for done in completions(directory / "completions.csv"):
        idx, op = expected.pop((done.arrival_ns, done.start_lba))
        if not done.arrival_ns <= done.enqueue_ns <= done.complete_ns or done.lba_count != cfg.spp or done.op != op:
            raise ValueError(f"Invalid completion: {done}")
        last[idx] = max(done.complete_ns, last[idx])
    if expected:
        raise ValueError(f"Missing {len(expected)} completions")
    if not overlap and any(end >= batch.arrival_ns for end, batch in zip(last, batches[1:])):
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


def aggregate(cfg, group, keys):
    result = {**keys, "mqsim_ns": sum(row["mqsim_ns"] for _, row in group),
              **metrics(cfg, (req for batch, _ in group for req in batch.requests))}
    result["mq_est_unsplit_ns"] = result["mq_est_ns"]
    result["ideal_unsplit_ns"] = result["ideal_ns"]
    result["mq_est_ns"] = sum(row["mq_est_ns"] for _, row in group)
    result["ideal_ns"] = sum(row["ideal_ns"] for _, row in group)
    result["estimate_gap_ns"] = result["mq_est_ns"] - result["ideal_ns"]
    result["mqsim_over_est"] = result["mqsim_ns"] / result["mq_est_ns"] if result["mq_est_ns"] else None
    if "layer" not in keys:
        # Each layer runs serially in the reconstructed step. Keep unsplit bounds additive.
        layers = defaultdict(list)
        for batch, row in group:
            layers[batch.layer].append((batch, row))
        per_layer = [metrics(cfg, (req for batch, _ in items for req in batch.requests)) for items in layers.values()]
        result["mq_est_unsplit_ns"] = sum(layer["mq_est_ns"] for layer in per_layer)
        result["ideal_unsplit_ns"] = sum(layer["ideal_ns"] for layer in per_layer)
    return result
