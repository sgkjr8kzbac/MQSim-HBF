"""Execute every Quick Start command in README.md and verify TOML/JSON configs + subcommands."""
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"


def _quickstart_commands():
    text = README.read_text()
    match = re.search(r"^## 快速开始\s*$(.*?)^---", text, re.MULTILINE | re.DOTALL)
    assert match, "README.md is missing '## 快速开始' section"
    section = match.group(1)
    commands = []
    for block in re.findall(r"```bash\n(.*?)```", section, re.DOTALL):
        for line in block.strip().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("python3 -m hbf "):
                commands.append(line)
    return commands


def test_readme_quickstart_commands_execute(tmp_path):
    """R3.6: every python3 -m hbf command in README Quick Start runs cleanly end-to-end."""
    commands = _quickstart_commands()
    assert len(commands) >= 5, f"Expected at least 5 Quick Start commands, found {len(commands)}"
    for idx, cmd in enumerate(commands):
        tokens = shlex.split(cmd)
        assert tokens[:3] == ["python3", "-m", "hbf"]
        out = tmp_path / f"qs-{idx}"
        argv = [sys.executable, *tokens[1:], "--output", str(out)]
        proc = subprocess.run(argv, cwd=str(ROOT), capture_output=True, text=True)
        assert proc.returncode == 0, f"Quick Start command #{idx} failed:\n{cmd}\nSTDERR:\n{proc.stderr}"
        assert (out / "config.json").is_file(), f"Missing config.json for {cmd}"
        assert (out / "meta.json").is_file(), f"Missing meta.json for {cmd}"
        if tokens[3] == "check":
            check = json.loads((out / "check.json").read_text())
            assert check["steps"] and check["steps"][0]["mqsim_ns"] > 0


def test_toml_and_json_configs_with_set_overrides_and_subcommands(tmp_path):
    """R3.6: TOML/JSON config files, --set overrides, config.json output, and unified subcommands."""
    from hbf.cli import main

    toml_cfg = tmp_path / "run.toml"
    toml_cfg.write_text(
        '[device]\ntopology = "ocp"\n\n'
        '[kv]\nmodel = "llama-3.1-8b"\nlayout = "striped_contiguous"\ncontext = 1024\nformat = "bf16"\n\n'
        '[trace]\nmode = "dense"\n\n'
        '[placement]\nbatch = 2\n\n'
        '[backend]\nname = "rounds"\n\n'
        '[ext.system]\ngpu = "b200"\noverlap = "serial"\n'
    )
    out_run = tmp_path / "run-out"
    main(["run", str(toml_cfg), "--set", "kv.format=kivi2", "--set", "placement.batch=4",
          "--output", str(out_run)])
    cfg = json.loads((out_run / "config.json").read_text())
    assert cfg["kv"]["format"] == "kivi2"
    assert cfg["placement"]["batch"] == 4
    assert cfg["backend"]["name"] == "rounds"
    assert cfg["ext"]["system"]["gpu"] == "b200"

    # Subcommands: system and interference on the finished run directory.
    out_sys = tmp_path / "sys-out"
    main(["system", str(out_run), "--overlap", "prefetch:1", "--output", str(out_sys)])
    assert (out_sys / "system_steps.csv").is_file()

    out_int = tmp_path / "int-out"
    main(["interference", str(out_run), "--write-gb-s", "0.5", "--erases-per-s", "10", "--output", str(out_int)])
    assert (out_int / "interference_steps.csv").is_file()

    # JSON config + --set override for lifetime subcommand.
    json_cfg = tmp_path / "life.json"
    json_cfg.write_text(json.dumps({
        "kv": {"model": "llama-3.1-8b", "format": "bf16"},
        "lifetime": {"analytic": True, "no_calibrate": True, "workload": "chat", "rd_threshold": [1e6]},
    }))
    out_life = tmp_path / "life-out"
    main(["lifetime", str(json_cfg), "--set", "kv.format=fp8", "--output", str(out_life)])
    life_cfg = json.loads((out_life / "config.json").read_text())
    assert life_cfg["kv"]["formats"] == ["fp8"]
    assert (out_life / "endurance_summary.csv").is_file()

    # Subcommand: sweep via python -m hbf sweep.
    grid = tmp_path / "grid.json"
    grid.write_text(json.dumps({
        "base": {"device.topology": "ocp", "kv.layers": 1, "kv.heads": 2, "kv.context": 512},
        "vary": {"kv.format": ["bf16", "fp8"]},
    }))
    out_sweep = tmp_path / "sweep-out"
    main(["sweep", str(grid), "--output", str(out_sweep)])
    assert (out_sweep / "summary.csv").is_file()

    # Invalid subcommand exits cleanly.
    with pytest.raises(SystemExit):
        main(["unknown-cmd"])
