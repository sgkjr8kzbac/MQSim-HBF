"""Pinned HBFSim commit check, patch series application, and build cache."""
import fcntl
import hashlib
import os
import subprocess
from pathlib import Path

from ...device.config import PACKAGE, PATCHES_DIR, ROOT

HBFSIM_COMMIT = PACKAGE / "hbfsim.commit"
HBFSIM_PATCHES_DIR = PATCHES_DIR / "hbfsim"
HBFSIM_BUILD_SCRIPT = ROOT / "scripts" / "hbfsim_build.sh"
PIN = HBFSIM_COMMIT.read_text().strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def patch_series() -> list[Path]:
    """Patches in the order of hbf/patches/hbfsim/series; every *.patch must be listed once."""
    series_file = HBFSIM_PATCHES_DIR / "series"
    if not series_file.is_file():
        if list(HBFSIM_PATCHES_DIR.glob("*.patch")):
            raise ValueError("hbf/patches/hbfsim/series is missing")
        return []
    names = [
        line.split("#", 1)[0].strip()
        for line in series_file.read_text().splitlines()
        if line.split("#", 1)[0].strip()
    ]
    disk = sorted(p.name for p in HBFSIM_PATCHES_DIR.glob("*.patch"))
    if sorted(names) != disk or len(set(names)) != len(names):
        raise ValueError("hbf/patches/hbfsim/series must list every patch file exactly once")
    return [HBFSIM_PATCHES_DIR / name for name in names]


def build(cache=None, target="hbfsim") -> Path:
    """Locate or build the pinned HBFSim binary (``hbfsim`` or ``hbfsim-reference``)."""
    if target not in {"hbfsim", "hbfsim-reference"}:
        raise ValueError(f"Unknown HBFSim target {target!r}; expected 'hbfsim' or 'hbfsim-reference'")
    patches = patch_series()
    cache_root = Path(cache or (ROOT / ".build" / "hbfsim")).resolve()
    short_commit = PIN[:12]
    build_dir = cache_root / short_commit / "build"
    binary = build_dir / target

    # Compute expected fingerprint matching scripts/hbfsim_build.sh.
    lines = [f"{PIN}\n"]
    for patch in patches:
        lines.append(f"{sha256(patch)}  {patch}\n")
    expected_fp = hashlib.sha256("".join(lines).encode()).hexdigest()
    stamp = build_dir / ".built.sha256"
    if binary.is_file() and stamp.is_file() and stamp.read_text().strip() == expected_fp:
        return binary

    cache_root.mkdir(parents=True, exist_ok=True)
    with (cache_root / "build.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if binary.is_file() and stamp.is_file() and stamp.read_text().strip() == expected_fp:
            return binary
        env = {**os.environ, "HBFSIM_CACHE_DIR": str(cache_root)}
        subprocess.run(
            ["bash", str(HBFSIM_BUILD_SCRIPT)],
            cwd=str(ROOT),
            env=env,
            check=True,
        )
    if not binary.is_file():
        raise FileNotFoundError(f"HBFSim build did not produce {binary}")
    return binary


def version(patches=None) -> dict:
    """Return HBFSim commit, patch SHA-256 map, and repository provenance for ``meta.json``."""
    rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    all_patches = patch_series()
    return {
        "hbfsim_commit": PIN,
        "repository_commit": rev.stdout.strip() or None,
        "working_tree_modified": bool(status.stdout.strip()),
        "patches_enabled": list(patches) if patches is not None else [p.name for p in all_patches],
        "patch_sha256": {p.name: sha256(p) for p in all_patches},
    }
