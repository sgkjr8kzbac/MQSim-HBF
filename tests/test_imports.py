"""Verify every module under hbf/ imports cleanly in isolation with no circular imports."""
import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HBF_DIR = ROOT / "hbf"


def _hbf_modules():
    modules = []
    for path in sorted(HBF_DIR.rglob("*.py")):
        rel = path.relative_to(ROOT)
        if "mqsim" in rel.parts or "__pycache__" in rel.parts:
            continue
        if path.name == "__init__.py":
            mod = ".".join(rel.parent.parts)
        elif path.name == "__main__.py":
            continue
        else:
            mod = ".".join(rel.with_suffix("").parts)
        modules.append(mod)
    return modules


def test_every_hbf_module_imports_in_isolation():
    modules = _hbf_modules()
    assert len(modules) >= 20
    failures = []
    for mod in modules:
        proc = subprocess.run(
            [sys.executable, "-c", f"import {mod}"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            failures.append(f"{mod}:\n{proc.stderr.strip()}")
    assert not failures, "Module import failures:\n" + "\n\n".join(failures)


def test_lifetime_does_not_import_hbf_cli():
    """R3.5b: hbf/lifetime/ must not import hbf.cli, and clock calibration must not read system_steps.csv."""
    assert "system_steps.csv" not in (HBF_DIR / "lifetime" / "clock.py").read_text()
    for path in sorted((HBF_DIR / "lifetime").rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module == "cli" and node.level >= 2:
                    raise AssertionError(f"{path} imports from hbf.cli")
                if node.module == "hbf.cli":
                    raise AssertionError(f"{path} imports from hbf.cli")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name != "hbf.cli", f"{path} imports hbf.cli"


def _imported_targets(path):
    """Yield every (module, level) imported inside ``path``."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            yield (node.module or ""), node.level
        elif isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, 0


def test_frontend_does_not_import_backends():
    """Frontend produces traces/pages only and must never import from hbf.backends."""
    for path in sorted((HBF_DIR / "frontend").rglob("*.py")):
        for module, level in _imported_targets(path):
            parts = module.split(".") if module else []
            if "backends" in parts:
                raise AssertionError(f"{path} imports backend module {'.' * level}{module}")


def test_backends_do_not_cross_import():
    """Sibling backends (rounds and mqsim) must remain independent of each other."""
    for backend, forbidden in (("rounds", "mqsim"), ("mqsim", "rounds")):
        for path in sorted((HBF_DIR / "backends" / backend).rglob("*.py")):
            for module, level in _imported_targets(path):
                parts = module.split(".") if module else []
                if forbidden in parts:
                    raise AssertionError(f"{path} cross-imports {'.' * level}{module}")


def test_core_never_imports_ext():
    """R4: core modules outside hbf/ext/ must never statically import hbf.ext."""
    for path in sorted(HBF_DIR.rglob("*.py")):
        rel = path.relative_to(HBF_DIR)
        if rel.parts[0] in ("ext", "mqsim") or "__pycache__" in rel.parts:
            continue
        for module, level in _imported_targets(path):
            parts = module.split(".") if module else []
            if "ext" in parts:
                raise AssertionError(f"Core module {path} imports extension {'.' * level}{module}")


