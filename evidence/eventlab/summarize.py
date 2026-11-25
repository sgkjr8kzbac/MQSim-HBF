"""Summarize HBF-EventLab statistics JSON files of the read probes as CSV on stdout.

Reads start at 2 ms (after the prefill). For each run: read time (last event minus 2 ms), read
senses, cache hits are in the .out log, and sense waves: senses are grouped into a wave while
they are issued less than half a sense time after the wave's first sense.
"""
import csv
import json
import sys
from pathlib import Path

READ_AT = 2_000_000_000  # ps
SENSE = 4_096_000  # ps

writer = csv.writer(sys.stdout)
writer.writerow(["run", "read_us", "senses", "waves", "wave_sizes", "cache_hits"])
for path in map(Path, sys.argv[1:]):
    data = json.loads(path.read_text())
    senses = sorted(op["issued_at"] for op in data["bank_operations"]
                    if op["kind"] == "sense" and op["issued_at"] >= READ_AT)
    waves = []
    for issued in senses:
        if not waves or issued >= waves[-1][0] + SENSE // 2:
            waves.append([issued, 0])
        waves[-1][1] += 1
    hits = ""
    log = path.with_suffix(".out")
    if log.exists():
        for line in log.read_text().splitlines():
            if line.startswith("DEVICE "):
                hits = dict(item.split("=", 1) for item in line.split()[1:] if "=" in item).get("cache_hits", "")
    writer.writerow([path.stem, f"{(data['at'] - READ_AT) / 1e6:.2f}", len(senses), len(waves),
                     "+".join(str(size) for _, size in waves), hits])
