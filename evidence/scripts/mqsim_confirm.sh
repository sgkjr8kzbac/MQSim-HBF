#!/usr/bin/env bash
# MQSim confirmations of the phase A read-time estimate on new formats, batch emulation and tR.
# usage: evidence/scripts/mqsim_confirm.sh OUTPUT_DIR   (runs one 1,024-plane MQSim at a time)
# Uses the default ocp topology (HBF command model + automatic prefill).
set -euo pipefail
out=${1:?output directory}
mkdir -p "$out"
run() {  # name, flags...
    local name=$1; shift
    [ -e "$out/$name" ] && return
    python3 -m hbf check "$@" --output "$out/$name" > /dev/null
}
run dense-128k-1024 --layers 2 --context 131072
run dense-128k-384 --layers 2 --context 131072 --aggregate-gb-s 384
run dense-128k-3072 --layers 2 --context 131072 --aggregate-gb-s 3072
run dense-32k-b8 --layers 1 --heads 64 --context 32768
run tokens1-128k-striped --layers 2 --context 131072 --mode uniform_tokens --fraction 0.01 --layout striped_contiguous
run tokens5-128k-striped --layers 2 --context 131072 --mode uniform_tokens --fraction 0.05 --layout striped_contiguous
run tokens5-128k-layer --layers 2 --context 131072 --mode uniform_tokens --fraction 0.05
run kivi2-dense-32k --layers 2 --context 32768 --format kivi2
run kivi2-tokens1-16 --layers 2 --context 131072 --format kivi2 --mode uniform_tokens --fraction 0.01 --select-tokens 16
run kivi2-tokens1-128 --layers 2 --context 131072 --format kivi2 --mode uniform_tokens --fraction 0.01 --select-tokens 128
run int4-dense-32k --layers 2 --context 32768 --format int4
run quest-tokens5-32k --layers 2 --context 32768 --format quest --mode uniform_tokens --fraction 0.05
run tr2048-dense-8k --layers 2 --context 8192 --read-ns 2048
run tr8192-tokens5-32k --layers 2 --context 32768 --mode uniform_tokens --fraction 0.05 --read-ns 8192
run staggered-sink --layers 2 --context 32768 --mode sink_recent --fraction 0.05 --layout striped_staggered
