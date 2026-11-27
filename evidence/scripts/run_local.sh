#!/usr/bin/env bash
# Run the test and experiment tasks of docs/local-runs.md on your own machine and collect the summaries.
# usage: evidence/scripts/run_local.sh TASK [TASK ...]
#   tasks: tests | phase2a | mqsim | placement | interference | endurance | all
# Raw runs go to runs/local/<task>/ (not committed); summaries are copied into evidence/ so that they can
# be committed and pushed. Needs: python3 with numpy and pytest; g++, make and GNU time for MQSim tasks.
set -euo pipefail
cd "$(dirname "$0")/../.."
root=runs/local
mkdir -p "$root" evidence/pytest evidence/phase2a evidence/phase2b
stamp=$(date -u +%Y%m%dT%H%M%S)
jobs=${JOBS:-3}

task_tests() {  # full suite, about 15 min, runs MQSim one process at a time
    python3 -m pytest -q -p no:cacheprovider 2>&1 | tail -40 | tee "evidence/pytest/local-$stamp.log"
}

task_phase2a() {  # the four phase A sweeps, about 15 min with 3 jobs, under 1 GB per job
    for grid in item1-layout-protocol item1-device item1-system item2-compression; do
        rm -rf "$root/$grid"
        python3 -m hbf.sweep "evidence/grids/$grid.json" --output "$root/$grid" --jobs "$jobs" --skip-access-csv \
            2> "$root/$grid.log" > /dev/null
        cp "$root/$grid/summary.csv" "evidence/phase2a/$grid.csv"
    done
    python3 evidence/scripts/phase2a_tables.py > "$root/phase2a-tables.md"
}

task_mqsim() {  # 15 MQSim confirmation points, about 5 min, 4.2 GB peak, strictly one at a time
    rm -rf "$root/mqsim"
    evidence/scripts/mqsim_confirm.sh "$root/mqsim"
    python3 evidence/scripts/time_calibration.py "$root"/mqsim/*/ > evidence/phase2a/mqsim-confirm.csv
}

task_placement() {  # E5.1/E5.2: 96 runs; the largest need about 5 GB each, so at most 2 jobs on 16 GB
    rm -rf "$root/placement"
    python3 -m hbf.sweep evidence/grids/phase2b-placement.json --output "$root/placement" --jobs "${PLACEMENT_JOBS:-2}" \
        --no-keep 2> "$root/placement.log" > /dev/null
    cp "$root/placement/summary.csv" evidence/phase2b/placement.csv
}

task_interference() {  # E5.6: 4 system runs and 960 re-timings, a few minutes
    rm -rf "$root/interference"
    python3 evidence/scripts/interference_scan.py "$root/interference"
    cp "$root/interference/interference-summary.csv" evidence/phase2b/interference.csv
}

task_endurance() {  # E3.1/E4.1 analytic curves, about 1-2 min (small calibration runs, cached)
    rm -rf "$root/endurance-bf16" "$root/endurance-formats"
    python3 -m hbf.endurance --analytic --model llama-3.1-8b,llama-3.1-70b --format bf16 \
        --workload chat,long32k,agentic,reasoning --concurrency 8 --rd-threshold 1e5,1e6,1e7 \
        --pe-limit 1e4,1e5,1e6 --pages-per-block 192,256,512,1024 --output "$root/endurance-bf16" \
        --calibration-dir "$root/calibration" > /dev/null
    cp "$root/endurance-bf16/endurance_summary.csv" evidence/phase2b/endurance-analytic.csv
    for fmt in bf16 fp8 int4 kivi4 kivi2 kivi2:think=0.5; do  # E4.1: compression, 8B, long32k and agentic
        python3 -m hbf.endurance --analytic --model llama-3.1-8b --format "$fmt" --workload long32k,agentic \
            --concurrency 8 --rd-threshold 1e5,1e6,1e7 --pe-limit 1e4,1e5,1e6 \
            --output "$root/endurance-formats/${fmt//[:=.]/-}" --calibration-dir "$root/calibration" > /dev/null
    done
    python3 - "$root/endurance-formats" evidence/phase2b/endurance-formats.csv <<'PY'
import csv, sys
from pathlib import Path
rows = [row for path in sorted(Path(sys.argv[1]).glob("*/endurance_summary.csv")) for row in csv.DictReader(path.open())]
with open(sys.argv[2], "w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
PY
}

[ $# -gt 0 ] || { sed -n '2,6p' "$0"; exit 2; }
for task in "$@"; do
    if [ "$task" = all ]; then
        set -- tests phase2a mqsim placement interference endurance
        for each in "$@"; do echo "== $each" >&2; "task_$each" || echo "$each failed" >&2; done
        exit 0
    fi
    declare -F "task_$task" > /dev/null || { echo "unknown task $task" >&2; exit 2; }
    echo "== $task" >&2
    "task_$task"
done
