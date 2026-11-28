"""Pinned MQSim source integrity check, patch series application, and build cache."""
import fcntl
import hashlib
import os
import shutil
import subprocess
from pathlib import Path

from ...device.config import MQSIM_COMMIT, MQSIM_MANIFEST, MQSIM_SOURCE, PATCHES_DIR, ROOT

PIN = MQSIM_COMMIT.read_text().strip()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def patch_series():
    """Patches in the order of patches/series; every patch file must be listed exactly once."""
    names = [line.strip() for line in (PATCHES_DIR / "series").read_text().splitlines()
             if line.strip() and not line.startswith("#")]
    if sorted(names) != sorted(p.name for p in PATCHES_DIR.glob("*.patch")) or len(set(names)) != len(names):
        raise ValueError("patches/series must list every patch file exactly once")
    return [PATCHES_DIR / name for name in names]


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


def version(patches):
    rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                            cwd=ROOT, capture_output=True, text=True)
    return {"mqsim_commit": PIN, "repository_commit": rev.stdout.strip() or None,
            "working_tree_modified": bool(status.stdout.strip()), "patches_enabled": patches,
            "patch_sha256": {p.name: sha256(p) for p in sorted(PATCHES_DIR.glob("*.patch"))}}
