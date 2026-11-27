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

task_endurance() {  # E3.1/E4.1 analytic curves; needs hbf/endurance.py (phase 2B part 1)
    if [ ! -f evidence/grids/phase2b-endurance.json ]; then
        echo "endurance: evidence/grids/phase2b-endurance.json is not there yet (endurance model in progress)" >&2
        return 1
    fi
    rm -rf "$root/endurance"
    python3 -m hbf.endurance --grid evidence/grids/phase2b-endurance.json --output "$root/endurance"
    cp "$root/endurance/endurance_summary.csv" evidence/phase2b/endurance-analytic.csv
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
