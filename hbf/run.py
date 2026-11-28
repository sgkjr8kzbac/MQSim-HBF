"""Unified configuration loading (TOML/JSON), ``--set`` overrides, and run/check helpers."""
import csv
import json
import tomllib
from pathlib import Path

SECTIONS = ("device", "trace", "kv", "placement", "backend", "lifetime", "ext")

# Mapping from dotted config key -> hbf run option key (underscore form of long CLI flag).
RUN_KEY_MAP = {
    # [device]
    "device.topology": "topology",
    "device.aggregate_gb_s": "aggregate_gb_s",
    "device.planes": "planes",
    "device.plane_rule": "plane_rule",
    "device.read_plane_ns": "read_plane_ns",
    "device.coalesce_ns": "coalesce_ns",
    "device.cache_read": "cache_read",
    "device.read_ns": "read_ns",
    "device.program_ns": "program_ns",
    "device.erase_ns": "erase_ns",
    "device.pages_per_block": "pages_per_block",
    "device.blocks_per_plane": "blocks_per_plane",
    "device.batch_pages": "batch_pages",
    # [trace]
    "trace.mode": "mode",
    "trace.steps": "steps",
    "trace.seed": "seed",
    "trace.fraction": "fraction",
    "trace.count": "count",
    "trace.unit": "unit",
    "trace.hot_fraction": "hot_fraction",
    "trace.hot_units": "hot_units",
    "trace.select_tokens": "select_tokens",
    "trace.access_csv": "access_csv",
    "trace.physical": "physical",
    "trace.prefill": "prefill",
    "trace.plane_group": "plane_group",
    "trace.plane_groups": "plane_group",
    "trace.mocs": "mocs",
    "trace.read_burst": "read_burst",
    "trace.read_unit": "read_unit",
    "trace.host_order": "host_order",
    "trace.ucie_grade": "ucie_grade",
    "trace.group_sense": "group_sense",
    # [kv]
    "kv.model": "model",
    "kv.layers": "layers",
    "kv.heads": "heads",
    "kv.head_dim": "head_dim",
    "kv.context": "context",
    "kv.format": "format",
    "kv.tensors": "tensors",
    "kv.layout": "layout",
    "kv.density": "density",
    # [placement]
    "placement.batch": "batch",
    "placement.stacks": "stacks",
    "placement.alloc": "alloc",
    "placement.partition": "partition",
    "placement.contexts": "contexts",
    "placement.generated": "generated",
    "placement.flush_order": "flush_order",
    "placement.align": "align",
    "placement.block_tokens": "block_tokens",
    "placement.free_order": "free_order",
    "placement.check_writes": "check_writes",
    # [backend]
    "backend.analysis_only": "analysis_only",
    # [ext.system]
    "ext.system.gpu": "gpu",
    "ext.system.overlap": "overlap",
    "ext.system.kv_consume": "kv_consume",
    "ext.system.selection": "selection",
    # [ext.kv_terms]
    "ext.kv_terms.kv_spec": "kv_spec",
    "ext.kv_terms.select_static": "select_static",
    "ext.kv_terms.reuse": "reuse",
    "ext.kv_terms.query_groups": "query_groups",
    "ext.kv_terms.full_tensors": "full_tensors",
    "ext.kv_terms.frontiers": "frontiers",
    "ext.kv_terms.prefix_tokens": "prefix_tokens",
    "ext.kv_terms.prefix_groups": "prefix_groups",
    "ext.kv_terms.prefix_reads": "prefix_reads",
}

# Mapping from dotted config key -> hbf lifetime option key.
LIFETIME_KEY_MAP = {
    # [device]
    "device.pages_per_block": "pages_per_block",
    "device.capacity_gib": "capacity_gib",
    "device.stacks": "stacks",
    "device.read_ns": "read_ns",
    "device.aggregate_gb_s": "aggregate_gb_s",
    "device.program_ns": "program_ns",
    "device.erase_ns": "erase_ns",
    # [kv]
    "kv.model": "model",
    "kv.format": "format",
    # [trace]
    "trace.mode": "mode",
    "trace.fraction": "fraction",
    "trace.select_tokens": "select_tokens",
    "trace.count": "count",
    "trace.seed": "seed",
    # [placement]
    "placement.alloc": "alloc",
    "placement.partition": "partition",
    "placement.align": "align",
    "placement.block_tokens": "block_tokens",
    "placement.free_order": "free_order",
    "placement.stacks": "stacks",
    # [lifetime]
    "lifetime.analytic": "analytic",
    "lifetime.workload": "workload",
    "lifetime.concurrency": "concurrency",
    "lifetime.rate": "rate",
    "lifetime.max_batch": "max_batch",
    "lifetime.input_len": "input_len",
    "lifetime.output_len": "output_len",
    "lifetime.history": "history",
    "lifetime.calls": "calls",
    "lifetime.think_s": "think_s",
    "lifetime.prefix_groups": "prefix_groups",
    "lifetime.prefix_tokens": "prefix_tokens",
    "lifetime.prefix_share": "prefix_share",
    "lifetime.zipf": "zipf",
    "lifetime.retain_s": "retain_s",
    "lifetime.rd_threshold": "rd_threshold",
    "lifetime.rd_hot": "rd_hot",
    "lifetime.read_share": "read_share",
    "lifetime.pe_limit": "pe_limit",
    "lifetime.retention_h": "retention_h",
    "lifetime.prefix_reads": "prefix_reads",
    "lifetime.m5": "m5",
    "lifetime.suspend_ns": "suspend_ns",
    "lifetime.block_unit": "block_unit",
    "lifetime.no_calibrate": "no_calibrate",
    "lifetime.calibration_dir": "calibration_dir",
    "lifetime.calib_context": "calib_context",
    "lifetime.hours": "hours",
    "lifetime.warmup_h": "warmup_h",
    "lifetime.wear": "wear",
    "lifetime.refresh": "refresh",
    "lifetime.refresh_idle": "refresh_idle",
    "lifetime.reclaim": "reclaim",
    "lifetime.victim": "victim",
    "lifetime.prefill": "prefill",
    "lifetime.pre_open": "pre_open",
    "lifetime.separate_prefix": "separate_prefix",
    "lifetime.vllm_reuse": "vllm_reuse",
    "lifetime.low_watermark": "low_watermark",
    "lifetime.report_s": "report_s",
    "lifetime.quantum_s": "quantum_s",
    "lifetime.rd_slack": "rd_slack",
    "lifetime.slots": "slots",
    "lifetime.no_stagger": "no_stagger",
    # [ext.system]
    "ext.system.gpu": "gpu",
    "ext.system.overlap": "overlap",
    "ext.system.kv_consume": "kv_consume",
    # [ext.kv_terms]
    "ext.kv_terms.kv_spec": "kv_spec",
}


def load_config(source):
    """Load a run config from a TOML/JSON file path or a dict; validate top-level sections."""
    if source is None:
        return {}
    if isinstance(source, dict):
        data = json.loads(json.dumps(source))
    else:
        path = Path(source)
        text = path.read_text()
        if path.suffix.lower() == ".toml":
            data = tomllib.loads(text)
        elif path.suffix.lower() == ".json":
            data = json.loads(text)
        else:
            try:
                data = tomllib.loads(text)
            except tomllib.TOMLDecodeError:
                data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Configuration must be a mapping of sections")
    unknown = set(data) - set(SECTIONS)
    if unknown:
        raise ValueError(f"Unknown config section(s) {sorted(unknown)}; expected subset of {list(SECTIONS)}")
    return data


def parse_value(text):
    """Parse a ``--set`` value string into bool/None/int/float/list/dict, or keep as string."""
    low = text.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low in ("null", "none"):
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def apply_overrides(config, items):
    """Apply ``--set section.key=value`` overrides onto a deep copy of ``config``."""
    out = json.loads(json.dumps(config))
    for item in items or ():
        if "=" not in item:
            raise ValueError(f"Invalid --set override {item!r}: expected section.key=value")
        key, raw = item.split("=", 1)
        parts = [p.strip() for p in key.split(".") if p.strip()]
        if len(parts) < 2 or parts[0] not in SECTIONS:
            raise ValueError(f"Invalid --set key {key!r}: must start with one of {list(SECTIONS)}")
        cur = out
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
            if not isinstance(cur, dict):
                raise ValueError(f"Cannot set nested key {key!r}: {part!r} is not a section")
        cur[parts[-1]] = parse_value(raw)
    return out


def flatten_config(config, prefix=""):
    """Flatten a nested section dict into dotted ``section.key`` pairs."""
    flat = {}
    for key, value in (config or {}).items():
        dotted = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict) and (not prefix or prefix == "ext"):
            flat.update(flatten_config(value, dotted))
        else:
            flat[dotted] = value
    return flat


def _option_argv(flag_name, value):
    flag = "--" + flag_name.replace("_", "-")
    if value is None or value is False:
        return []
    if value is True:
        return [flag]
    if isinstance(value, dict) or (isinstance(value, list) and any(isinstance(v, (dict, list)) for v in value)):
        return [flag, json.dumps(value)]
    if isinstance(value, list):
        out = []
        for item in value:
            out += [flag, str(item)]
        return out
    return [flag, str(value)]


def resolve_grid_key(key, tool_name="analysis"):
    """Map a dotted config key (or flat option name) to the tool's CLI option key."""
    if "." not in key:
        return key
    table = RUN_KEY_MAP if tool_name == "analysis" else LIFETIME_KEY_MAP
    if key not in table:
        raise ValueError(f"Unknown config key {key!r} for {tool_name}")
    return table[key]


def config_to_run_argv(config):
    """Translate a structured config dict into CLI argv tokens for ``hbf run``."""
    argv = []
    for dotted, value in flatten_config(config).items():
        if dotted == "backend.name":
            if value == "rounds":
                argv.append("--analysis-only")
            elif value not in ("mqsim", None):
                raise ValueError(f"Unknown backend.name {value!r}: choose 'rounds' or 'mqsim'")
            continue
        if dotted.startswith("lifetime."):
            continue
        if dotted not in RUN_KEY_MAP:
            raise ValueError(f"Unknown config key {dotted!r} for run")
        argv += _option_argv(RUN_KEY_MAP[dotted], value)
    return argv


def config_to_lifetime_argv(config):
    """Translate a structured config dict into CLI argv tokens for ``hbf lifetime``."""
    argv = []
    for dotted, value in flatten_config(config).items():
        if dotted.startswith("backend."):
            continue
        if dotted not in LIFETIME_KEY_MAP:
            raise ValueError(f"Unknown config key {dotted!r} for lifetime")
        opt = LIFETIME_KEY_MAP[dotted]
        if isinstance(value, list):
            argv += ["--" + opt.replace("_", "-"), ",".join(str(v) for v in value)]
        else:
            argv += _option_argv(opt, value)
    return argv


def resolved_run_config(cfg, shape, layout_name, mode, generation, input_csv, physical, tensors,
                        plane_groups, prefill_layout, protocol, system, kv_format, requests,
                        check_writes, kv_spec, backend_name):
    """Build the structured resolved config dict written to ``config.json`` for every run."""
    from dataclasses import asdict
    from .frontend.kv.placement import BatchSpec
    return {
        "device": {
            "topology": f"ocp-{cfg.host_channels}ch" if cfg.address_map == "ocp" and cfg.host_channels != 16
                        else ("ocp" if cfg.address_map == "ocp" else f"{cfg.channels}x{cfg.chips}"),
            "channels": cfg.channels,
            "chips": cfg.chips,
            "dies": cfg.dies,
            "planes": cfg.planes,
            "address_map": cfg.address_map,
            "plane_rule": cfg.plane_rule,
            "read_plane_ns": cfg.read_plane_ns,
            "coalesce_ns": cfg.coalesce_ns,
            "cache_read": cfg.cache_read,
            "aggregate_gb_s": cfg.aggregate_gb_s,
            "read_ns": cfg.read_ns,
            "program_ns": cfg.program_ns,
            "erase_ns": cfg.erase_ns,
            "pages_per_block": cfg.pages_per_block,
            "blocks_per_plane": cfg.blocks_per_plane,
            "batch_pages": cfg.batch_pages,
        },
        "trace": {
            "mode": mode if input_csv is None else None,
            "access_csv": str(input_csv) if input_csv else None,
            "physical": physical,
            "prefill": bool(prefill_layout),
            "plane_groups": list(plane_groups),
            "protocol": None if protocol is None else asdict(protocol),
            "generation": {k: v for k, v in (generation or {}).items()
                           if k not in ("select_static", "reuse", "query_groups", "full_tensors",
                                        "channel_groups", "channels")},
        },
        "kv": {
            "model": system["model"].name if system else None,
            "layers": shape.layers,
            "heads": system["heads"] if system else (
                requests[1] if isinstance(requests, tuple) else (
                    shape.kv_heads // max(1, len(requests.requests)) if isinstance(requests, BatchSpec)
                    else shape.kv_heads
                )
            ),
            "head_dim": shape.head_dim,
            "context": shape.context,
            "layout": layout_name,
            "format": kv_format.name if kv_format else None,
        },
        "placement": {
            "batch": system["batch"] if system else (
                requests[0] if isinstance(requests, tuple) else (
                    len(requests.requests) if isinstance(requests, BatchSpec) else 1
                )
            ),
            "stacks": system["stacks"] if system else 1,
            "alloc": requests.allocation if isinstance(requests, BatchSpec) else "layer_major",
            "partition": requests.partition if isinstance(requests, BatchSpec) else "none",
            "align": requests.align if isinstance(requests, BatchSpec) else "stagger",
            "flush_order": requests.flush_order if isinstance(requests, BatchSpec) else "rotate",
            "check_writes": check_writes,
        },
        "backend": {
            "name": backend_name,
        },
        "ext": {
            "system": None if system is None else {
                "gpu": system["gpu"].name,
                "overlap": system["options"].overlap,
                "kv_consume": system["options"].kv_consume,
                "selection": system["selection"],
            },
            "kv_terms": {
                "kv_spec": kv_spec.name if kv_spec is not None else None,
                "select_static": bool((generation or {}).get("select_static", False)),
                "reuse": (generation or {}).get("reuse", 0.0),
                "query_groups": (generation or {}).get("query_groups", 1),
                "full_tensors": (generation or {}).get("full_tensors"),
                "frontiers": requests.frontiers if isinstance(requests, BatchSpec) else "single",
                "prefix_tokens": requests.prefix_tokens if isinstance(requests, BatchSpec) else 0,
                "prefix_groups": requests.prefix_groups if isinstance(requests, BatchSpec) else 1,
                "prefix_reads": requests.prefix_reads if isinstance(requests, BatchSpec) else "coalesced",
            },
        },
    }


def resolved_lifetime_config(config):
    """Build the structured resolved config dict written to ``config.json`` for lifetime runs."""
    return {
        "device": {
            "pages_per_block": config["pages_per_block"],
            "capacity_gib": config["capacity_gib"],
            "stacks": config["stacks"],
            "read_ns": config["read_ns"],
            "program_ns": config["t_prog"],
            "erase_ns": config["t_bers"],
            "device_flags": config["device_flags"],
        },
        "kv": {
            "models": [m.name for m in config["models"]],
            "formats": config["formats"],
        },
        "trace": {
            "mode": config["mode"],
            "generation": config["generation"],
        },
        "placement": {
            "alloc": config["policy"],
            "partition": config["partition"],
            "align": config["align"],
            "block_tokens": config["block_tokens"],
            "free_order": config["free_order"],
        },
        "lifetime": {
            "analytic": config["analytic"],
            "workloads": [w.name for w in config["workloads"]],
            "x_values": config["x_values"],
            "pe_limits": config["pe_limits"],
            "hot": config["hot"],
            "read_share": config["read_share"],
            "retention_s": config["retention_s"],
            "prefix_reads": config["prefix_reads"],
            "m5": config["m5"],
            "suspend_ns": config["suspend_ns"],
            "block_unit": config["block_unit"],
            "calibrate": config["calibrate"],
            "hours": config["hours"],
            "warmup_h": config["warmup_h"],
            "wear": config["wear"],
            "refresh": config["refresh"],
            "refresh_idle": config["refresh_idle"],
            "reclaim": config["reclaim"],
            "victim": config["victim"],
            "prefill": config["prefill"],
            "pre_open": config["pre_open"],
            "separate_prefix": config["separate_prefix"],
            "vllm_reuse": config["vllm_reuse"],
            "low_watermark": config["low_watermark"],
            "report_s": config["report_s"],
            "quantum_s": config["quantum_s"],
            "rd_slack": config["rd_slack"],
            "slots": config["slots"],
            "stagger": config["stagger"],
        },
        "ext": {
            "system": {
                "gpu": config["gpu_name"],
                "overlap": config["options"].overlap,
                "kv_consume": config["options"].kv_consume,
            },
            "kv_terms": {
                "kv_spec": config["kv_spec"],
            },
        },
    }


def split_config_and_overrides(tokens):
    """Extract optional leading CONFIG path and any ``--set`` overrides without touching other flags."""
    tokens = list(tokens)
    config_path = None
    if tokens and not tokens[0].startswith("-"):
        config_path = tokens.pop(0)
    overrides, rest = [], []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok == "--set":
            if i + 1 >= len(tokens):
                raise ValueError("--set requires section.key=value")
            overrides.append(tokens[i + 1])
            i += 2
        elif tok.startswith("--set="):
            overrides.append(tok[6:])
            i += 1
        else:
            rest.append(tok)
            i += 1
    return config_path, overrides, rest


def summarize_check(directory):
    """Compare rounds estimate (analysis_steps.csv) against MQSim (steps.csv) in ``directory``."""
    directory = Path(directory)
    with (directory / "analysis_steps.csv").open(newline="") as stream:
        est_by_step = {
            int(row["step"]): row for row in csv.DictReader(stream) if row.get("group", "1") == "1"
        }
    with (directory / "steps.csv").open(newline="") as stream:
        mq_rows = [row for row in csv.DictReader(stream) if int(row["step"]) >= 0]
    comparisons = []
    for mq in mq_rows:
        step = int(mq["step"])
        mq_ns = float(mq["mqsim_ns"])
        est_row = est_by_step.get(step, {})
        est_batches = float(est_row["est_batches_ns"]) if est_row.get("est_batches_ns") not in (None, "") else None
        est_ns = float(est_row["est_ns"]) if est_row.get("est_ns") not in (None, "") else None
        bound_ns = float(est_row["bound_ns"]) if est_row.get("bound_ns") not in (None, "") else None
        rel_err = (est_batches - mq_ns) / mq_ns if (est_batches is not None and mq_ns > 0) else None
        comparisons.append({
            "step": step,
            "read_pages": int(mq["read_pages"]),
            "mqsim_ns": mq_ns,
            "est_batches_ns": est_batches,
            "est_ns": est_ns,
            "bound_ns": bound_ns,
            "rel_error": rel_err,
        })
    summary = {"directory": str(directory.resolve()), "steps": comparisons}
    (directory / "check.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
