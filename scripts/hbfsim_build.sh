#!/usr/bin/env bash
# Fetch pinned HBFSim commit, apply patches from hbf/patches/hbfsim/series,
# and build hbfsim + hbfsim_reference in Release mode under .build/hbfsim/<short-commit>.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMMIT_FILE="$ROOT/hbf/hbfsim.commit"
PATCH_DIR="$ROOT/hbf/patches/hbfsim"
SERIES_FILE="$PATCH_DIR/series"
REPO_URL="${HBFSIM_REPO_URL:-https://github.com/FlashLabAE/HBFSim.git}"
CACHE_ROOT="${HBFSIM_CACHE_DIR:-$ROOT/.build/hbfsim}"

if [[ ! -f "$COMMIT_FILE" ]]; then
    echo "error: missing $COMMIT_FILE" >&2
    exit 1
fi
COMMIT="$(tr -d '[:space:]' < "$COMMIT_FILE")"
SHORT_COMMIT="${COMMIT:0:12}"

MIRROR_DIR="$CACHE_ROOT/src"
WORK_DIR="$CACHE_ROOT/$SHORT_COMMIT"
BUILD_DIR="$WORK_DIR/build"
STAMP_FILE="$BUILD_DIR/.built.sha256"

# Collect patches in series order and compute a build input fingerprint.
PATCHES=()
if [[ -f "$SERIES_FILE" ]]; then
    while IFS= read -r line || [[ -n "$line" ]]; do
        line="${line%%#*}"
        line="$(printf '%s' "$line" | xargs || true)"
        [[ -z "$line" ]] && continue
        PATCHES+=("$PATCH_DIR/$line")
    done < "$SERIES_FILE"
fi

FINGERPRINT="$(
    {
        printf '%s\n' "$COMMIT"
        for p in "${PATCHES[@]}"; do
            if [[ ! -f "$p" ]]; then
                echo "error: missing patch $p" >&2
                exit 1
            fi
            sha256sum "$p"
        done
    } | sha256sum | awk '{print $1}'
)"

if [[ -f "$STAMP_FILE" && "$(cat "$STAMP_FILE")" == "$FINGERPRINT" \
      && -x "$BUILD_DIR/hbfsim" && -x "$BUILD_DIR/hbfsim-reference" ]]; then
    echo "$BUILD_DIR"
    exit 0
fi

mkdir -p "$CACHE_ROOT"

# Ensure mirror checkout exists and has the pinned commit.
if [[ ! -d "$MIRROR_DIR/.git" ]]; then
    git clone "$REPO_URL" "$MIRROR_DIR"
fi
if ! git -C "$MIRROR_DIR" cat-file -e "${COMMIT}^{commit}" 2>/dev/null; then
    git -C "$MIRROR_DIR" fetch origin "$COMMIT"
fi

# Prepare clean working tree at the exact pinned commit.
rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"
git -C "$MIRROR_DIR" archive "$COMMIT" | tar -x -C "$WORK_DIR"

# Apply patches cleanly (with GIT_CEILING_DIRECTORIES so parent repo doesn't interfere).
if [[ "${#PATCHES[@]}" -gt 0 ]]; then
    GIT_CEILING_DIRECTORIES="$CACHE_ROOT" git -C "$WORK_DIR" apply "${PATCHES[@]}"
fi

GENERATOR_ARGS=()
if command -v ninja >/dev/null 2>&1; then
    GENERATOR_ARGS=(-G Ninja)
fi

cmake -S "$WORK_DIR" -B "$BUILD_DIR" \
    "${GENERATOR_ARGS[@]}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=OFF

JOBS="$(nproc 2>/dev/null || echo 4)"
cmake --build "$BUILD_DIR" --target hbfsim hbfsim_reference -j "$JOBS"

printf '%s\n' "$FINGERPRINT" > "$STAMP_FILE"
echo "$BUILD_DIR"
