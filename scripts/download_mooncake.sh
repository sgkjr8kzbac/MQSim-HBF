#!/usr/bin/env bash
# Download public Mooncake FAST'25 request-level traces (Apache License 2.0)
# into .build/traces/mooncake/ (never committed to git).
#
# Usage:
#   scripts/download_mooncake.sh [conversation|toolagent|all] [--dest DIR]
#
# Source repository:
#   https://github.com/kvcache-ai/Mooncake (FAST25-release/traces/, Apache-2.0)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BASE_URL="https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces"

DEST="${REPO_ROOT}/.build/traces/mooncake"
TARGET="conversation"

while [[ $# -gt 0 ]]; do
    case "$1" in
        conversation|toolagent|all)
            TARGET="$1"
            shift
            ;;
        --dest)
            [[ $# -ge 2 ]] || { echo "Error: --dest requires a directory" >&2; exit 2; }
            DEST="$2"
            shift 2
            ;;
        -h|--help)
            sed -n '2,9p' "$0"
            exit 0
            ;;
        *)
            echo "Error: unknown argument '$1' (expected conversation|toolagent|all or --dest DIR)" >&2
            exit 2
            ;;
    esac
done

mkdir -p "${DEST}"

# Verified SHA-256 checksums for upstream Mooncake FAST'25 traces.
declare -A SHA256_MAP=(
    ["conversation_trace.jsonl"]="b8cbb061a85206d729d91cdc2981f43c9e0d99209dce588d3af5f7934408b9df"
)

sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    else
        echo "Error: neither sha256sum nor shasum found" >&2
        exit 1
    fi
}

fetch_one() {
    local filename="$1"
    local expected="${SHA256_MAP[$filename]:-}"
    local out="${DEST}/${filename}"
    local url="${BASE_URL}/${filename}"

    if [[ -f "${out}" ]]; then
        local actual
        actual="$(sha256_of "${out}")"
        if [[ -n "${expected}" ]]; then
            if [[ "${actual}" == "${expected}" ]]; then
                echo "[ok] ${out} already present and SHA-256 verified (${actual})"
                return 0
            else
                echo "[warn] ${out} checksum mismatch (got ${actual}, expected ${expected}); re-downloading" >&2
            fi
        else
            echo "[ok] ${out} already present (SHA-256: ${actual})"
            return 0
        fi
    fi

    local tmp="${out}.tmp.$$"
    trap 'rm -f "${tmp}"' RETURN
    echo "[download] ${url} -> ${out}"
    if command -v curl >/dev/null 2>&1; then
        curl -fL --retry 3 -o "${tmp}" "${url}"
    elif command -v wget >/dev/null 2>&1; then
        wget -q -O "${tmp}" "${url}"
    else
        echo "Error: neither curl nor wget found" >&2
        exit 1
    fi

    local got
    got="$(sha256_of "${tmp}")"
    if [[ -n "${expected}" && "${got}" != "${expected}" ]]; then
        echo "Error: SHA-256 mismatch for ${filename}: got ${got}, expected ${expected}" >&2
        rm -f "${tmp}"
        exit 1
    fi

    mv "${tmp}" "${out}"
    trap - RETURN
    if [[ -n "${expected}" ]]; then
        echo "[ok] downloaded and SHA-256 verified: ${out} (${got})"
    else
        echo "[ok] downloaded: ${out} (SHA-256: ${got})"
    fi
}

case "${TARGET}" in
    conversation)
        fetch_one "conversation_trace.jsonl"
        ;;
    toolagent)
        fetch_one "toolagent_trace.jsonl"
        ;;
    all)
        fetch_one "conversation_trace.jsonl"
        fetch_one "toolagent_trace.jsonl"
        ;;
esac
