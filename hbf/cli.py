"""Generate or import logical accesses, run MQSim, and report the results."""
import argparse
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .access import LAYOUTS, MODES, Shape, generate, parse_tensors, read_access, tensors_meta, write_access
from .analysis import ENGINES, GROUP_SENSES, HOST_ORDERS, READ_UNITS, Protocol, analyze_accesses, check_group
from .config import MAX_BATCH_PAGES, load
from .controller import schedule_run, unwritten_reads
from .simulator import build, dump, simulate, write_table


def fixed(value, digits):
    """A number with ``digits`` decimals, or a dash when there is none (a step without HBF pages)."""
    return "—" if value is None else f"{value:.{digits}f}"


def protocol_report(step_rows, protocol):
    report = ["## 协议约束（每主机通道命令额度）", "",
              f"MOCS = {protocol.mocs or '不限'}，读突发：{'支持' if protocol.read_burst else '不支持（每条读 64 B）'}，"
              f"读取单位：{'整页' if protocol.read_unit == 'page' else '只读需要的 64 B 段'}，"
              f"主机发送顺序：{'整页命令一起排队（fcfs）' if protocol.host_order == 'fcfs' else '先发一条触发感测（trigger）'}，"
              f"链路档位：{protocol.ucie_grade or '不计'}，"
              f"联动组感测：{'只感测命令已到的页（arrived）' if protocol.group_sense == 'arrived' else '一次感测整组（whole）'}。", "",
              "| 并行组 | Step | 读命令 | 只受额度限制的轮数 | 只受链路限制的轮数 | 全部约束下的轮数 | 利用率 |",
              "|---:|---:|---:|---:|---:|---:|---:|"]
    for row in step_rows:
        report.append(f"| {row['group']} | {row['step']} | {row['commands']} | {row['credit_rounds']} | "
                      f"{row['link_rounds']} | {row['protocol_rounds']} | {fixed(row['protocol_utilization'], 3)} |")
    return report


def timing_report(cfg, step_rows, plane_groups=(1,), protocol=None):
    from .timing import HBF_COMMAND_MODEL, hbf_command_model
    report = ["## 读时间估计", "",
              f"tR = {cfg.read_ns} ns，每页传输 {cfg.transfer_ns} ns。理想下界：有效轮数 × tR、最忙核心通道的传输、"
              "最忙主机链路三者取大，不计命令开销，感测与传输完全重叠。估算：按核心通道逐轮累加命令、感测、R/B 和"
              "传输（含 cache read 的重叠），整层作为一次突发；按子批相加的估算与 MQSim 的运行方式对应。"]
    if cfg.dies == cfg.planes == 1:
        report.append("chip 当 plane 的拓扑每页一条命令，没有估算，只给理想下界。")
    elif not hbf_command_model(cfg):
        report.append(f"注意：估算只在 HBF 命令模型（`{HBF_COMMAND_MODEL}`）、tR = 4,096 ns 下与 MQSim 校准过，"
                      "当前配置下未经验证。")
    if cfg.dies != cfg.planes or cfg.planes > 1:
        per_die = cfg.planes
        if any(1 < group < per_die for group in plane_groups):
            report.append("联动组小于一个 die 时的估算没有 MQSim 对照（MQSim 的 lockstep 规则对应整个 die）。")
        if protocol is not None:
            report.append("协议约束（额度、链路）只按轮数拉长估算，没有 MQSim 对照（MQSim 没有 MOCS），"
                          "轮数只与 HBF-EventLab 的波次核对过。")
    report += ["", "| 并行组 | Step | 理想下界 ns | 估算 ns | 按子批估算 ns | 均衡部分 ns | 不均衡 ns | 协议 ns | 核心通道 ns | "
               "开销 ns | 单块最多读次数 |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in step_rows:
        cells = [row[key] for key in ("group", "step", "bound_ns", "est_ns", "est_batches_ns", "t_balanced_ns",
                                      "t_imbalance_ns", "t_protocol_ns", "t_channel_ns", "t_overhead_ns",
                                      "max_block_reads")]
        report.append("| " + " | ".join("—" if cell is None else str(cell) for cell in cells) + " |")
    return report


def topology_name(cfg):
    return "ocp" if cfg.address_map == "ocp" else f"{cfg.channels}x{cfg.chips}"


def analysis_report(layer_rows, step_rows):
    report = ["## 机制分析", "",
              "| 并行组 | Step | 层数 | 读页 | 轮数 | 子批轮数 | 最少轮数 | 核心通道下界轮数 | plane 利用率 | 均衡损失 | 感测放大 | 读放大 | 传输放大 |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    amplification = {}
    for row in layer_rows:
        key = row["group"], row["step"]
        page, useful, burst = amplification.get(key, (0, 0, 0))
        amplification[key] = page + row["page_bytes"], useful + row["useful_bytes"], burst + row["burst_bytes"]
    for row in step_rows:
        page, useful, burst = amplification[row["group"], row["step"]]
        report.append(
            f"| {row['group']} | {row['step']} | {row['layers']} | {row['pages']} | {row['rounds']} | "
            f"{row['batch_rounds']} | "
            f"{row['min_rounds']} | {row['channel_rounds']} | {fixed(row['utilization'], 3)} | "
            f"{fixed(row['balance_loss'], 2)} | {fixed(row['sense_amplification'], 2)} | "
            f"{fixed(page / useful if useful else None, 2)} | {fixed(burst / useful if useful else None, 2)} |")
    return report


def request_bytes(cfg, layout, heads):
    """Stored KV of one request (all tiers; HBF tensors page-granular) and its HBM-tier part."""
    per_head = sum(count * cfg.page_bytes for count in layout.block_counts)
    hbm = sum((layout.windows[spec.name][1] - layout.windows[spec.name][0]) * spec.token_bytes
              for spec in layout.hbm_tensors)
    scale = layout.shape.layers * heads
    return {"kv_bytes_per_request": (per_head + hbm) * scale, "hbm_tier_bytes_per_request": hbm * scale}


def subbatch_count(layer_rows, plane_groups):
    """Sub-batches of a run without a schedule (no writes): the read sub-batches of every layer."""
    return sum(row["subbatches"] for row in layer_rows if row["group"] == plane_groups[0])


def pipeline(cfg, shape, layout_name, directory, binary, *, mode="dense", generation=None,
             input_csv=None, physical=False, tensors=None, plane_groups=(1,), prefill_layout=False,
             placement="static", protocol=None, engine="auto", timing=False, system=None, kv_format=None,
             requests=None):
    """Schedule the accesses, write the mechanism analysis, and run MQSim unless binary is None.

    Without MQSim and with static placement the analysis needs no request schedule, so the numpy
    engine works on the accesses directly; the rows are identical either way. ``requests`` is
    (requests per stack, KV heads per request) of a ``--batch`` emulation: an imported trace must
    use every request's heads.
    """
    generation = {
        "steps": 1, "seed": 1, "fraction": .1, "count": 4, "unit": 0,
        "hot_fraction": .5, "hot_units": 1, "select_tokens": 16,
    } | (generation or {})
    layout = LAYOUTS[layout_name](cfg, shape, tensors=tensors)
    accesses = list(generate(cfg, layout, mode, **generation) if input_csv is None else read_access(input_csv))
    if input_csv is not None and requests is not None and requests[0] > 1:
        seen = {access.kv_head // requests[1] for access in accesses}
        if len(seen) < requests[0]:
            raise ValueError(f"The access trace reads {len(seen)} of the {requests[0]} requests that --batch puts on "
                             f"a stack (KV heads r x {requests[1]} + h); a batched trace must cover them all")
    directory.mkdir(parents=True, exist_ok=False)  # after the inputs are known to be valid
    access_path = directory / "access.csv"
    write_access(access_path, accesses)
    system_meta = None
    if system is not None:
        system_meta = request_bytes(cfg, layout, system["heads"]) | {
            "model": system["model"].name, "gpu": system["gpu"].name, "batch": system["batch"],
            "stacks": system["stacks"], "analysis_batch": system["analysis_batch"], "selection": system["selection"],
            "kv_format": kv_format.name if kv_format else "tensors" if tensors else "bf16",
            "overlap": system["options"].overlap, "kv_consume": system["options"].kv_consume}
    fmt = kv_format
    dump(directory / "meta.json", {"shape": asdict(shape), "layout": layout_name,
                                   "system": system_meta,
                                   "format": None if fmt is None else {"name": fmt.name, "params": fmt.params,
                                                                       "attn_dims": fmt.attn_dims},
                                   "tensors": tensors_meta(layout),
                                   "generator": {"mode": mode, **generation} if input_csv is None else None,
                                   "input_csv": str(input_csv) if input_csv else None,
                                   "prefill": prefill_layout and
                                   f"every LPA in [{layout.base}, {layout.base + layout.span}) in order",
                                   "placement": placement,
                                   "protocol": None if protocol is None else asdict(protocol),
                                   "argv": sys.argv})
    if prefill_layout and any(access.op == "W" for access in accesses):
        raise ValueError("--prefill writes the whole layout; the accesses must not contain writes")
    needs_schedule = (binary is not None or placement == "mqsim" or engine == "python" or prefill_layout
                      or any(access.op == "W" for access in accesses))
    batches = schedule_run(cfg, layout, accesses, prefill_layout) if needs_schedule else []
    layer_rows, step_rows = analyze_accesses(cfg, layout, accesses, plane_groups, placement, protocol, engine,
                                             batches=batches if needs_schedule else None, timing=timing)
    if timing:
        attn_dims = fmt.attn_dims if fmt else 2 * shape.head_dim
        for row in layer_rows:
            row["attn_dims"] = attn_dims
    if layer_rows:
        write_table(directory / "analysis_layers.csv", layer_rows)
        write_table(directory / "analysis_steps.csv", step_rows)
    unwritten = unwritten_reads(batches) if needs_schedule else 0
    hierarchy = (f"{cfg.channels} 通道 × {cfg.chips} chip × {cfg.dies} die × {cfg.planes} plane"
                 f"（地址映射 {cfg.address_map}）")
    switches = [f"核心通道合计 {cfg.aggregate_gb_s} GB/s", f"plane 规则 {cfg.plane_rule or '上游'}"]
    switches += [f"{name} {value}" for name, value in (
        ("T-CMD 每 plane 步长 ns", cfg.read_plane_ns), ("合并窗口 ns", cfg.coalesce_ns), ("t_DBSY ns", cfg.dbsy_ns))
        if value is not None]
    if cfg.cache_read:
        switches.append("T-CACHE 开")
    report = ["# 本次运行", "", f"布局 `{layout_name}`，拓扑 `{topology_name(cfg)}`：{hierarchy}；"
              f"{'，'.join(switches)}。子批 {len(batches) if needs_schedule else subbatch_count(layer_rows, plane_groups)} 个。"
              f"机制分析的页位置：{placement}。", ""]
    if unwritten and (binary is not None or placement == "mqsim"):
        report += [f"有 {unwritten} 次读的页在本次运行中没有写过。OCP 规范对这种读返回错误 0x7（表 12）；"
                   "机制分析按静态位置计算时假定数据已在 HBF 中。需要 MQSim 按规范放置时请加 `--prefill`。", ""]
    if layer_rows:
        report += analysis_report(layer_rows, step_rows) + [""]
        if protocol is not None:
            report += protocol_report(step_rows, protocol) + [""]
        if timing:
            report += timing_report(cfg, step_rows, plane_groups, protocol) + [""]
    if system is not None:
        from .system import evaluate, report_lines, write_table as write_system_table
        rows = [row for row in layer_rows if row["group"] == 1] if 1 in plane_groups else []
        if not rows:
            raise ValueError("The system model needs plane group 1 (independent planes) in --plane-group")
        outcome = evaluate(rows, system["model"], system["gpu"], system["options"], system_meta)
        write_system_table(directory / "system_layers.csv", outcome.layers)
        write_system_table(directory / "system_steps.csv", outcome.steps)
        report += report_lines(outcome) + [""]
        if fmt and fmt.notes:
            report += ["KV 格式假设：", ""] + [f"- {note}" for note in fmt.notes] + [""]
    result = None
    if binary is not None:
        result = simulate(cfg, batches, directory, binary, physical=physical)
        _, _, steps, stats = result
        report += ["## MQSim", "",
                   f"MQSim 用时 {stats['wall_seconds']:.3f} s，峰值 RSS {stats['peak_rss_kib']} KiB。", "",
                   "| Step | 读页 | MQSim ns | MQSim 估算 ns | 理想 HBF ns |",
                   "|---:|---:|---:|---:|---:|"]
        for row in steps:
            if row["step"] == -1:
                report.append(f"| −1（prefill 写，不计入读） | 0 | {row['mqsim_ns']} | — | — |")
                continue
            report.append(f"| {row['step']} | {row['read_pages']} | {row['mqsim_ns']} | "
                          f"{'—' if row['mq_est_ns'] is None else row['mq_est_ns']} | {row['ideal_ns']:.0f} |")
    (directory / "report.md").write_text("\n".join(report) + "\n")
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", default="64x16",
                        help="ocp (the spec's Figure 4 example), 16x64, 64x16, 256x4 or any "
                             "<channels>x<chips per channel>")
    parser.add_argument("--layout", choices=LAYOUTS, default="layer_contiguous")
    parser.add_argument("--mode", choices=MODES, default="dense")
    parser.add_argument("--layers", type=int, help="default 1, or the --model's layers")
    parser.add_argument("--heads", type=int, help="KV heads per request (default 8, or the --model's)")
    parser.add_argument("--head-dim", type=int, help="default 128, or the --model's")
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--fraction", type=float, default=.1)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--unit", type=int, default=0)
    parser.add_argument("--hot-fraction", type=float, default=.5)
    parser.add_argument("--hot-units", type=int, default=1)
    parser.add_argument("--select-tokens", type=int, default=16,
                        help="uniform_tokens: tokens per selected block (Quest's page; default 16)")
    parser.add_argument("--batch-pages", type=int, default=MAX_BATCH_PAGES,
                        help=f"maximum pages per sub-batch (1–{MAX_BATCH_PAGES})")
    parser.add_argument("--physical", action="store_true")
    parser.add_argument("--access-csv", type=Path)
    parser.add_argument("--format", dest="kv_format",
                        help="named KV storage format NAME[:key=value,...]: bf16, fp8, int4, int2, kivi4, kivi2 "
                             "(group, residual, think), think (ratio), quest (page); see hbf/formats.py")
    parser.add_argument("--tensors", help="JSON list or file of stored tensor formats, "
                        'e.g. \'[{"name":"K","bits":2,"dims":128},{"name":"V","bits":16,"dims":128}]\'; '
                        "default BF16 K and V with head_dim channels")
    parser.add_argument("--aggregate-gb-s", type=int,
                        help="total NAND channel bandwidth in GB/s, split evenly over the channels "
                             "(default 1024, from SPLASH; OCP grades are 384, 1536 and 3072)")
    parser.add_argument("--plane-group", type=int, action="append", dest="plane_groups",
                        help="planes per lock-step group for the mechanism analysis; repeat to compare "
                             "(1 = independent planes, the default)")
    parser.add_argument("--prefill", action="store_true",
                        help="write every address of the layout (padding included) in order before the "
                             "reads, as the OCP spec requires; needed for multi-plane MQSim runs")
    parser.add_argument("--plane-rule", choices=("independent", "lockstep"),
                        help="T-PLANE multi-plane rule; without it MQSim keeps its upstream same-page rule")
    parser.add_argument("--planes", type=int,
                        help="planes per MQSim die (with --topology Cx1 this gives the real die/plane hierarchy)")
    parser.add_argument("--dbsy-ns", type=int, help="t_DBSY between planes of a multi-plane command (T-PLANE)")
    parser.add_argument("--read-plane-ns", type=int,
                        help="T-CMD: a read command for n planes costs the single-plane command plus (n-1) "
                             "times this step, instead of ONFI's 6*t_WC + t_DBSY per extra plane")
    parser.add_argument("--cache-read", action="store_true",
                        help="T-CACHE: a die senses the next read while at most one earlier batch waits "
                             "for transfer (buffers are accounted per die; the OCP spec has two per bank)")
    parser.add_argument("--coalesce-ns", type=int,
                        help="T-CMD: hold newly received transactions this long before dispatching, so that "
                             "a batch leaves as full multi-plane commands")
    parser.add_argument("--placement", choices=("static", "mqsim"), default="static",
                        help="page offsets for the mechanism analysis: static follows the OCP spec "
                             "(address decides bank and page); mqsim follows MQSim's allocation order")
    parser.add_argument("--mocs", type=int,
                        help="analysis: outstanding commands per host channel (OCP MOCS: 1 or 256-16384, power "
                             "of two); enables the protocol-limited rounds. Any protocol flag enables them; "
                             "without --mocs there is no credit limit")
    parser.add_argument("--read-burst", action="store_true",
                        help="analysis: the product supports burst reads (one command per contiguous run)")
    parser.add_argument("--read-unit", choices=READ_UNITS, default="page",
                        help="analysis: read whole pages or only the 64 B units with wanted tokens")
    parser.add_argument("--host-order", choices=HOST_ORDERS, default="fcfs",
                        help="analysis: fcfs queues a page's commands together (they hold credits during the "
                             "sense); trigger sends one 64 B read first and the rest as cache hits")
    parser.add_argument("--group-sense", choices=GROUP_SENSES, default="arrived",
                        help="analysis, lock-step groups: arrived senses only pages whose commands arrived; "
                             "whole senses the group's offset at the first command (an assumption)")
    parser.add_argument("--ucie-grade", type=int, choices=(1, 2, 3),
                        help="analysis: bound rounds by the host link (48/94/188 GB/s per channel)")
    parser.add_argument("--engine", choices=ENGINES, default="auto",
                        help="mechanism analysis engine: numpy (vectorised) or python (reference); auto picks numpy "
                             "for static placement. Both give identical results")
    parser.add_argument("--analysis-only", action="store_true",
                        help="write the mechanism analysis without building or running MQSim")
    parser.add_argument("--output", type=Path)
    device = parser.add_argument_group("device parameters (defaults from hbf/configs/parameters.json; the OCP "
                                       "spec leaves them product specific)")
    device.add_argument("--read-ns", type=int, help="page sense time tR (default 4096, from SPLASH)")
    device.add_argument("--program-ns", type=int, help="page program time (default 100000)")
    device.add_argument("--erase-ns", type=int, help="block erase time (default 1000000)")
    device.add_argument("--pages-per-block", type=int, help="pages per erase block (default 256)")
    device.add_argument("--blocks-per-plane", type=int, help="blocks per plane (default 512)")
    parser.add_argument("--timing", action="store_true",
                        help="analysis: add read-time estimates (ideal bound and MQSim-calibrated estimate), "
                             "their breakdown, demand columns and per-block read counts")
    system = parser.add_argument_group("system model (decode TPOT; needs --model, implies --timing)")
    system.add_argument("--model", help="model preset (hbf/configs/models.json) or JSON; sets --layers, --heads "
                                        "and --head-dim unless they are given")
    system.add_argument("--gpu", help="GPU preset (hbf/configs/gpus.json) or JSON (default b200)")
    system.add_argument("--batch", type=int, default=1,
                        help="decode batch B: the analysed storage holds ceil(B / stacks) requests per stack, "
                             "placed layer-major as extra KV heads (an emulation of a batch)")
    system.add_argument("--stacks", type=int, default=1, help="HBF stacks sharing the batch (default 1)")
    system.add_argument("--overlap", default="serial", help="serial, prefetch:<d> or ideal")
    system.add_argument("--kv-consume", choices=("stream", "staged"), default="stream",
                        help="serial/ideal: attention consumes HBF pages as they arrive, or after staging in HBM")
    system.add_argument("--selection", choices=("auto", "static", "dynamic"), default="auto",
                        help="whether the pages a layer reads are known before its query (static: dense, "
                             "sink_recent) or not (dynamic: uniform_* selections, imported traces); prefetch "
                             "needs static")
    return parser


def configure(args):
    """Pipeline arguments from parsed command-line arguments (raises ValueError on bad input)."""
    given = {"aggregate_gb_s": args.aggregate_gb_s, "planes": args.planes, "plane_rule": args.plane_rule,
             "dbsy_ns": args.dbsy_ns, "coalesce_ns": args.coalesce_ns, "read_plane_ns": args.read_plane_ns,
             "cache_read": args.cache_read or None, "read_ns": args.read_ns, "program_ns": args.program_ns,
             "erase_ns": args.erase_ns, "pages_per_block": args.pages_per_block,
             "blocks_per_plane": args.blocks_per_plane}
    # Only flags given on the command line override the topology's own settings (e.g. ocp).
    cfg = load(args.topology, batch_pages=args.batch_pages,
               **{key: value for key, value in given.items() if value is not None})
    model = None
    if args.model is not None:
        from .model import load_model
        model = load_model(args.model)
    def given(value, from_model, default):
        return value if value is not None else from_model if model else default
    layers = given(args.layers, model and model.layers, 1)
    heads = given(args.heads, model and model.kv_heads, 8)
    head_dim = given(args.head_dim, model and model.head_dim, 128)
    if args.kv_format is not None and args.tensors is not None:
        raise ValueError("--format and --tensors are mutually exclusive")
    kv_format = None
    if args.kv_format is not None:
        from .formats import parse_format
        kv_format = parse_format(args.kv_format, head_dim)
    tensors = kv_format.tensors if kv_format else parse_tensors(args.tensors)
    if type(args.batch) is not int or args.batch < 1 or type(args.stacks) is not int or args.stacks < 1:
        raise ValueError("--batch and --stacks must be positive")
    per_stack = -(-args.batch // args.stacks)
    for group in args.plane_groups or (1,):
        check_group(cfg, group)
    protocol = None
    if (args.mocs is not None or args.read_burst or args.read_unit != "page" or args.host_order != "fcfs"
            or args.ucie_grade is not None or args.group_sense != "arrived"):
        protocol = Protocol(args.mocs, args.read_burst, args.read_unit,
                            args.host_order, args.ucie_grade, args.group_sense)
    shape = Shape(layers, heads * per_stack, head_dim, args.context)
    generation = {key: getattr(args, key) for key in (
        "steps", "seed", "fraction", "count", "unit", "hot_fraction", "hot_units", "select_tokens"
    )}
    if args.placement != "static" and (args.timing or model is not None):
        raise ValueError("--timing and --model need static placement (the numpy engine)")
    if model is None and (args.gpu is not None or args.overlap != "serial" or args.kv_consume != "stream"
                          or args.selection != "auto"):
        raise ValueError("--gpu, --overlap, --kv-consume and --selection need --model")
    system = None
    if model is not None:
        from .model import load_gpu
        from .system import SystemOptions
        if (layers, heads, head_dim) != (model.layers, model.kv_heads, model.head_dim):
            raise ValueError(f"--model {model.name} has {model.layers} layers, {model.kv_heads} KV heads and "
                             f"head_dim {model.head_dim}; do not override them")
        selection = args.selection
        if selection == "auto":
            selection = "static" if args.access_csv is None and args.mode in ("dense", "sink_recent") else "dynamic"
        options = SystemOptions(overlap=args.overlap, kv_consume=args.kv_consume)
        if options.mode == "prefetch" and selection != "static":
            raise ValueError("Prefetching the next layer's KV needs a static selection (dense or sink_recent, "
                             "or --selection static)")
        if 1 not in (args.plane_groups or (1,)):
            raise ValueError("The system model needs plane group 1 (independent planes) in --plane-group")
        system = {"model": model, "gpu": load_gpu(args.gpu or "b200"), "options": options, "batch": args.batch,
                  "stacks": args.stacks, "analysis_batch": per_stack, "selection": selection,
                  "kv_format": kv_format, "heads": heads}
    if (args.timing or system) and args.engine == "python":
        raise ValueError("--timing needs the numpy engine")
    return {"cfg": cfg, "shape": shape, "layout_name": args.layout, "mode": args.mode, "generation": generation,
            "input_csv": args.access_csv, "physical": args.physical, "tensors": tensors,
            "plane_groups": tuple(args.plane_groups or (1,)), "prefill_layout": args.prefill,
            "placement": args.placement, "protocol": protocol, "engine": args.engine,
            "timing": args.timing or system is not None, "system": system, "kv_format": kv_format,
            "requests": (per_stack, heads)}


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    directory = args.output or Path("runs") / f"{now}-{uuid4().hex[:8]}"
    try:
        run = configure(args)
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    binary = None if args.analysis_only else build()
    try:
        pipeline(directory=directory, binary=binary, **run)
    except ValueError as error:
        parser.error(str(error))
    print(directory.resolve())


if __name__ == "__main__":
    main()
