#!/usr/bin/env bash
# Rerun the HBF-EventLab probes behind tests/test_analysis.py and summarize them.
# usage: evidence/eventlab/run.sh /path/to/hbf_eventlab_cli OUTPUT_DIR
# Every probe: ocp.ini geometry (16 host channels x 4 dies x 16 banks, sense 4.096 us, grade 3),
# whole-page (burst4096) prefill writes, reads issued at 2 ms. Statistics JSON files are large
# (up to ~30 MB each), so they go to OUTPUT_DIR, not into the repository.
set -euo pipefail
cli=${1:?path to hbf_eventlab_cli}
out=${2:?output directory}
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$out"
out=$(cd "$out" && pwd)
run() {  # name workload mocs
    "$cli" "$here/workloads/$2.txt" --config "$here/ocp.ini" --burst --mocs "$3" \
        --stats-json "$out/$1.json" > "$out/$1.out"
}
run pages128-m256 pages128 256
run pages128-m1024 pages128 1024
run pages128-trigger-m256 pages128-trigger 256
run bank8x4-m256 bank8x4 256
run bank8x4-m1024 bank8x4 1024
run bank8x4-trigger-m256 bank8x4-trigger 256
run banks8-m1 banks8 1
run banks8-m256 banks8 256
run bank0x8-m1 bank0x8 1
run bank0x8-m256 bank0x8 256
run wanted7-m256 wanted7 256
run wanted13-m256 wanted13 256
run bank2x4-m1 bank2x4 1
run bank2x4-m256 bank2x4 256
python3 -I "$here/summarize.py" "$out"/*.json
