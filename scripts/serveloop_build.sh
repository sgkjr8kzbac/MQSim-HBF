#!/usr/bin/env bash
# Fetch pinned ServeLoop commit into .build/serveloop/<short-commit> (not tracked in git).
# Modeled on scripts/hbfsim_build.sh.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMMIT_FILE="$ROOT/hbf/serveloop.commit"
REPO_URL="${SERVELOOP_REPO_URL:-https://github.com/FlashLabAE/ServeLoop.git}"
CACHE_ROOT="${SERVELOOP_CACHE_DIR:-$ROOT/.build/serveloop}"

if [[ ! -f "$COMMIT_FILE" ]]; then
    echo "error: missing $COMMIT_FILE" >&2
    exit 1
fi
COMMIT="$(tr -d '[:space:]' < "$COMMIT_FILE")"
SHORT_COMMIT="${COMMIT:0:12}"

MIRROR_DIR="$CACHE_ROOT/src"
WORK_DIR="$CACHE_ROOT/$SHORT_COMMIT"
STAMP_FILE="$WORK_DIR/.checked_out.sha256"

if [[ -f "$STAMP_FILE" && "$(cat "$STAMP_FILE")" == "$COMMIT" \
      && -f "$WORK_DIR/serveloop/__init__.py" ]]; then
    echo "$WORK_DIR"
    exit 0
fi

mkdir -p "$CACHE_ROOT"

if [[ ! -d "$MIRROR_DIR/.git" ]]; then
    git clone "$REPO_URL" "$MIRROR_DIR"
fi
if ! git -C "$MIRROR_DIR" cat-file -e "${COMMIT}^{commit}" 2>/dev/null; then
    git -C "$MIRROR_DIR" fetch origin "$COMMIT"
fi

rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"
git -C "$MIRROR_DIR" archive "$COMMIT" | tar -x -C "$WORK_DIR"

printf '%s\n' "$COMMIT" > "$STAMP_FILE"
echo "$WORK_DIR"
