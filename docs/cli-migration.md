# CLI Migration Guide (Old Entrypoints → Unified `python -m hbf` Subcommands)

Starting in R3.6, all user-facing entrypoints are unified under `python3 -m hbf <subcommand>` (or the installed `hbf` console script). Every run writes its resolved structured configuration to `config.json` alongside `meta.json`.

## Old Entrypoint → New Subcommand Mapping

| Old Entrypoint | New Subcommand | Notes |
|---|---|---|
| `python3 -m hbf.cli [FLAGS...]` | `python3 -m hbf run [CONFIG] [--set section.key=val ...] [FLAGS...]` | Default topology is now `ocp` (with HBF command model enabled and automatic prefill for single-request MQSim runs). `--timing`, `--engine`, `--placement`, and `--dbsy-ns` were removed in R2. |
| `python3 -m hbf.endurance [FLAGS...]` | `python3 -m hbf lifetime [CONFIG] [--set section.key=val ...] [FLAGS...]` | `hbf/endurance.py` was split into `hbf/lifetime/`. Also callable directly as `python3 -m hbf.lifetime.cli`. |
| `python3 -m hbf.sweep GRID.json --output DIR [FLAGS...]` | `python3 -m hbf sweep GRID.json --output DIR [FLAGS...]` | Grids now accept dotted config keys (`device.topology`, `kv.format`, `placement.batch`, `ext.system.gpu`, etc.) in `base`, `vary`, and `zip`. |
| `python3 -m hbf.system RUN_DIR [FLAGS...]` | `python3 -m hbf system RUN_DIR [FLAGS...]` | Re-evaluates decode TPOT on an existing run directory. Direct `python3 -m hbf.system` remains available. |
| `python3 -m hbf.interference RUN_DIR [FLAGS...]` | `python3 -m hbf interference RUN_DIR [FLAGS...]` | Moved to `hbf/lifetime/interference.py`. Direct `python3 -m hbf.lifetime.interference` also works. |
| *(new)* | `python3 -m hbf check [CONFIG] [--set ...] [FLAGS...]` | Runs both the rounds backend and MQSim on the same address trace (`addr.npz`) and writes `check.json` with relative timing errors. |

## Config File Format (TOML / JSON)

A single run or lifetime experiment can be described by a TOML or JSON file partitioned into sections:

```toml
[device]
topology = "ocp"
read_ns = 4096

[kv]
model = "llama-3.1-8b"
layout = "striped_contiguous"
context = 32768
format = "bf16"

[trace]
mode = "dense"

[placement]
batch = 8

[backend]
name = "rounds"

[ext.system]
gpu = "b200"
overlap = "serial"
```

Override any field on the command line with repeatable `--set section.key=value`:

```bash
python3 -m hbf run run.toml --set kv.format=kivi2 --set placement.batch=16 --output runs/kivi2-b16
```
