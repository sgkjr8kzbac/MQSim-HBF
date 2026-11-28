"""Markdown reports and modelling-assumption lists for lifetime runs."""
import math

from .geometry import CAUSES, sci

def _fmt(value, digits=2):
    if value is None:
        return "—"
    if isinstance(value, float) and math.isinf(value):
        return "∞"
    if abs(value) >= 1e5 or (0 < abs(value) < 10 ** -digits):
        mantissa, exponent = f"{value:.2e}".split("e")
        return f"{mantissa}e{int(exponent)}"
    return f"{value:,.{digits}f}"


def report_lines(rows, config, calibrations=()):
    """Chinese report section 寿命评估 with every assumption and its source tag."""
    limits = config["pe_limits"]
    policy, partition = config["policy"], config["partition"]
    names = {"packed": "紧凑追加（packed）", "dedicated": "每请求独占段（dedicated）",
             "layer_major": "静态批次波（layer_major）", "hbm_decode": "生成的 KV 留在 HBM（hbm_decode）",
             "vllm_pool": "vLLM 空闲块池（vllm_pool，规范下不合法）"}
    kv_spec_note = f"；KV 规格 `{config['kv_spec']}`" if config.get("kv_spec") else ""
    lines = ["# 寿命评估（解析稳态）", "",
             f"分配方式：{names[policy]}，分区 `{partition}`；读取方式 `{config['mode']}`{kv_spec_note}；"
             f"每个 stack 容量 {config['capacity_gib']:g} GiB，{config['stacks']} 个 stack 分担请求，"
             f"tPROG {config['t_prog'] / 1e3:g} µs，tBERS {config['t_bers'] / 1e3:g} µs，"
             f"保持刷新间隔 {config['retention_s'] / 3600:g} h（0 表示关闭），持续读偏斜 h = {config['hot']:g}。", ""]
    for w in config["workloads"]:
        lines.append(f"- 负载 {w.describe()}。来源：{w.source}")
    lines += ["", "下表每行是一个 stack 的稳态（写入速率、P/E 均按每个 stack 计）。P/E 每天按理想磨损均衡（AVGPEC）。"
              "“到 N 的天数”= N_PE / 每天 P/E。阈值步数 = X / (ppb × 读比例 × h)。写放大 = 编程页 / 主机 KV 页；"
              "擦除放大 = 擦除容量 / 编程页（dedicated 与 layer_major 大于 1）。等量 X：读扰刷新写入等于 KV 写入时的 X"
              "（连续近似）。批是每个 stack 平均同时解码的请求数。", "",
              "| 模型 | 格式 | 负载 | ppb | X | 批 | TPOT ms | TPOT′ ms | KV 写 GB/s | 读扰刷新 GB/s | 保持刷新 GB/s "
              "| 重放 GB/s | 编程 GB/s | 刷新/KV | 写放大 | 擦除放大 | P/E 每天 | "
              + " | ".join(f"到 {sci(limit)} 天" for limit in limits) + " | 阈值步数 | 刷新/请求 | 等量 X |",
              "|" + "---|" * 3 + "---:|" * (18 + len(limits))]
    for row in rows:
        cells = [row["model"], row["format"], row["workload"], str(row["pages_per_block"]), sci(row["x"]),
                 str(row["mean_batch"]), _fmt(row["mean_tpot_ms"]), _fmt(row["tpot_m5_ms"]),
                 _fmt(row["kv_write_gb_s"], 3), _fmt(row["refresh_rd_gb_s"], 3), _fmt(row["refresh_ret_gb_s"], 3),
                 _fmt(row["replay_gb_s"], 3), _fmt(row["program_gb_s"], 3), _fmt(row["refresh_to_kv"]),
                 _fmt(row["wa"]), _fmt(row["erase_amp"]), _fmt(row["avg_pec_per_day"], 1)]
        cells += [_fmt(row[f"days_to_limit_avg@{sci(limit)}"], 1) for limit in limits]
        cells += [_fmt(row["threshold_steps"], 1), _fmt(row["refreshes_per_request"]), _fmt(row["break_even_x"], 0)]
        lines.append("| " + " | ".join(cells) + " |")
    first = rows[0] if rows else None
    if first:
        lines += ["", "其他列（见 endurance_summary.csv）：每秒请求与 token、写入按原因拆分、阵列编程带宽占比"
                  f"（例如第一行 {first['program_bw_share']:.4f}）、HBM 暂存峰值 {first['staging_peak_mib']:.1f} MiB、"
                  f"逐页涓流写时每主机通道同时未完成的 4 KiB 单元 {first['pending_units_per_channel']:.0f} 个"
                  f"（每个要开 {first['unit_open_s']:.2f} s；规范第 5.4.1 节第 7 条举例的产品上限是每通道 64 或 128 个）。"]
    lines += _calibration_lines(calibrations)
    lines += ["", "## 结论要点", "",
              "- 稠密读取时，一个装满的块每步被读 ppb 次，所以读扰阈值 = X / ppb 步，与 TPOT 和压缩格式无关"
              "（X = 1e5 / 1e6 / 1e7、ppb = 256 时为 390.6 / 3,906 / 39,063 步）。只有活得比这更久的数据（长输出、"
              "会话历史、共享前缀）才需要读扰刷新；稀疏选择把阈值步数放大 1 / 读比例 倍。",
              "- 固定负载下，压缩按压缩比减少 KV 写入和 P/E；在 HBF 带宽受限的最大吞吐下，每秒感测页数基本不变，"
              "压缩省下的是每 token 的 P/E（pe_per_mtok），而不是每秒的写入字节。",
              "- 阵列编程带宽 = 1,024 × 4 KiB / tPROG（tPROG = 100 µs 时 41.9 GB/s），KV 写入远低于它；"
              "真正的代价是编程和擦除占用 die 造成的读延迟（TPOT′，见 hbf/interference.py）。",
              "", "## 假设", ""]
    lines += [f"- {item}" for item in assumptions(config)]
    return lines


def _calibration_lines(calibrations):
    if not calibrations:
        return []
    lines = ["", "## 时钟校准", "",
             "时钟按 phase A 的每层估算 est = layer_ns + round_ns × 轮数（对 MQSim 校准过的估算），"
             "两点（稠密 layer-major，批大小使每层约 32 与 128 轮）拟合 layer_ns、round_ns（截距为负时固定为 4,000 ns 只拟合斜率）；策略快照（`--alloc`）给出均衡系数 κ"
             "（实测轮数 / 理想轮数）和稀疏读取相对稠密的需求比例。", "",
             "| 模型 | 格式 | 负载 | 拟合批 | layer_ns | round_ns | 快照批 | κ | 读比例 | 单步热度 | 快照 phase A TPOT ms |",
             "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for c in calibrations:
        tpot = c["snapshot"]["tpot_ns"]
        lines.append(f"| {c['model']} | {c['format']} | {c['workload']} | {c['batches']} | {c['layer_ns']:.0f} | "
                     f"{c['round_ns']:.1f} | {c['snapshot']['batch']} | {c['balance']:.4f} | {c['read_share']:.4f} | "
                     f"{c['hot_step']:.2f} | "
                     f"{_fmt(tpot / 1e6 if tpot else None, 3)} |")
    return lines


SIM_NAMES = {"packed": "紧凑追加（packed）", "dedicated": "每请求独占段（dedicated）",
             "layer_major": "静态批次波（layer_major）", "hbm_decode": "生成的 KV 留在 HBM（hbm_decode）",
             "vllm_pool": "vLLM 空闲块池（vllm_pool，规范下不合法）"}


def sim_report_lines(rows, config, calibrations=()):
    """Chinese report section 寿命评估 of an event simulation, with every assumption and its source tag."""
    limits = config["pe_limits"]
    extras = []
    if config.get("kv_spec"):
        extras.append(f"KV 规格 `{config['kv_spec']}`")
    if config.get("victim", "greedy") != "greedy":
        extras.append(f"回收选段 `{config['victim']}`")
    if config.get("prefill", "disaggregated") != "disaggregated":
        extras.append(f"prefill `{config['prefill']}`")
    if config.get("pre_open"):
        extras.append("预开段 `--pre-open`")
    if config.get("separate_prefix"):
        extras.append("前缀独立前沿 `--separate-prefix`")
    if config["policy"] == "vllm_pool":
        extras.append(f"vLLM 重用 `{config.get('vllm_reuse', 'fifo')}`、空闲顺序 `{config.get('free_order', 'fresh')}`")
    extra_note = ("；" + "，".join(extras)) if extras else ""
    lines = ["# 寿命评估（事件模拟）", "",
             f"分配方式：{SIM_NAMES[config['policy']]}，分区 `{config['partition']}`；读取方式 `{config['mode']}`；"
             f"每个 stack 容量 {config['capacity_gib']:g} GiB，{config['stacks']} 个 stack 分担请求（模拟其中一个）；"
             f"tPROG {config['t_prog'] / 1e3:g} µs，tBERS {config['t_bers'] / 1e3:g} µs；"
             f"保持刷新间隔 {config['retention_s'] / 3600:g} h（0 表示关闭），持续读偏斜 h = {config['hot']:g}；"
             f"模拟 {config['hours']:g} h，前 {config['warmup_h']:g} h 预热不计入速率；"
             f"磨损分配 `{config['wear']}`，刷新 `{config['refresh']}`，空闲条目刷新时 `{config['refresh_idle']}`，"
             f"回收 `{config['reclaim']}`，写入干扰 `{config['m5']}`（以 {config['block_unit']} 为阻塞单位）"
             f"{extra_note}。", ""]
    for w in config["workloads"]:
        lines.append(f"- 负载 {w.describe()}。来源：{w.source}")
    lines += ["", "下表每行是一次模拟（一个 stack，预热之后的速率）。P/E 每天取各段 PEC 的增量：平均即 AVGPEC，"
              "最大即 MAXPEC（第 5.9.2.17–18 节）；“到 N 的天数”= N_PE / 每天 P/E。写放大 = 编程页 / 主机 KV 页；"
              "擦除放大 = 擦除的页 / 主机 KV 页。TPOT′ 含写入干扰（M5，滞后一个量子反馈到步时）。"
              "“解析”两列是同一设置下 `--analytic` 的值，用来对照。", "",
              "| 模型 | 格式 | 负载 | ppb | X | 批 | TPOT ms | TPOT′ ms | 最大单步拖慢 ms | KV 写 GB/s | 读扰刷新 GB/s "
              "| 保持刷新 GB/s | 搬移 GB/s | 重放 GB/s | 编程 GB/s | 刷新/KV | 写放大 | 擦除放大 | P/E 每天（平均） "
              "| P/E 每天（最大） | " + " | ".join(f"到 {sci(limit)} 天（最大）" for limit in limits)
              + " | 驱逐 GB | 解析 P/E 每天 | 解析 刷新/KV |",
              "|" + "---|" * 3 + "---:|" * (20 + len(limits))]
    for row in rows:
        cells = [row["model"], row["format"], row["workload"], str(row["pages_per_block"]), sci(row["x"]),
                 _fmt(row["mean_batch"]), _fmt(row["mean_tpot_ms"]), _fmt(row["tpot_m5_ms"]),
                 _fmt(row["max_step_extra_ms"]), _fmt(row["kv_write_gb_s"], 3), _fmt(row["refresh_rd_gb_s"], 3),
                 _fmt(row["refresh_ret_gb_s"], 3), _fmt(row["compaction_gb_s"], 3), _fmt(row["replay_gb_s"], 3),
                 _fmt(row["program_gb_s"], 3), _fmt(row["refresh_to_kv"]), _fmt(row["wa"]), _fmt(row["erase_wa"]),
                 _fmt(row["avg_pec_per_day"], 1), _fmt(row["max_pec_per_day"], 1)]
        cells += [_fmt(row[f"days_to_limit_max@{sci(limit)}"], 1) for limit in limits]
        cells += [_fmt(row["evicted_gb"], 1), _fmt(row["analytic_avg_pec_per_day"], 1),
                  _fmt(row["analytic_refresh_to_kv"])]
        lines.append("| " + " | ".join(cells) + " |")
    first = rows[0] if rows else None
    if first:
        lines += ["", f"每次模拟 {first['segments']} 个段（每段 {first['segment_pages']} 页），回收低水位 "
                  f"{first['low_watermark']} 段；第一行用了 {first['quanta']} 个量子、{first['wall_s']:.1f} s。"
                  "时间序列（每 --report-s 一行：运行与排队请求、TPOT、各原因累计写入与擦除、空闲段、有效页、平均与最大 PEC、"
                  "最大读次数 / X）在 endurance_timeseries.csv，每段的 PEC、各原因擦除次数和有效页在 endurance_segments.csv。"
                  f"HBM 暂存峰值 {first['staging_peak_mib']:.1f} MiB。"]
    lines += ["", "## 与解析模型的差别", "",
              "- 解析模型把生成的页当作从第一步起就在、每页按请求的全部解码步数计读次数；模拟按页的实际写入时间计，"
              "所以长输出（reasoning）的读扰刷新低于解析值。",
              "- 模拟按块计读：被搬移的数据落在正在填充的前沿段里，块中页少、每步读次数少，到 X 的时间更长；"
              "数据在到 X 前死去时它积累的读次数作废。整段由一个条目占满时两者一致（见 tests/test_endurance.py）。",
              "- 模拟里写入干扰会拖慢步时、降低请求速率；批大小和上下文随时间波动，TPOT 随批增大，"
              "所以按请求平均的步时高于平均批下的 TPOT（会话负载约低 10–15 % 的请求速率）。",
              "- 只在空闲段之间做磨损均衡：容量被空闲缓存占满时只有少数空闲段在轮转，最大 PEC 可以远高于平均值。"]
    lines += _calibration_lines(calibrations)
    lines += ["", "## 假设", ""]
    lines += [f"- {item}" for item in assumptions(config)]
    return lines


def assumptions(config):
    """Assumption list of the endurance model, each with its source tag (params.md tags in brackets).

    The closed form's list is unchanged; for an event simulation (``analytic`` false) the items that
    describe the closed form's shortcuts are replaced and the simulation's own items are appended.
    """
    items = _analytic_assumptions()
    if config.get("analytic", True):
        return items
    replace_items = {
        "一个请求的所有页": "读扰按块计：一个段（分区内所有 bank 的同一块号）里的条目每步各读一次自己的页，块读次数 = "
                    "Σ 条目页数 × 读比例 × h / 段的 bank 数；块读次数达到 X 时刷新整段。命中共享前缀或会话历史时，"
                    "缓存的每页在 prefill 读一次（不乘读比例和 h，与解析模型一致）。",
        "每天 P/E = 编程页数": "每天 P/E 取预热后各段 PEC 的增量除以天数：平均即理想磨损均衡下的 AVGPEC，"
                         "最大即 MAXPEC（第 5.9.2.17–18 节）；wear ideal 时最大按平均报告。",
        "packed：每个新条目": "packed：每个新条目（请求、前缀或会话）的 prompt 从前沿的下一行起点开始（--align row，填充按实际计）；"
                         "解码页先在 HBM 暂存，每个量子结束时按分区攒成整行写入；请求结束时剩下的整页一起写入"
                         "（与解析模型的每请求页数一致），只有每 (层, head, 张量) 最后不满的一页留在 HBM 丢弃。",
        "闭环：每个 stack": "闭环：每个 stack 保持 round(concurrency / stacks) 个请求（会话负载为会话数）；"
                       "请求在准确的解码步完成并立即补上，prefill 分离，KV 写入完成后开始解码。启动时前 B 个请求的输出依次缩短"
                       "（第 i 个解码 out − round(i·out/B) 个 token），会话从第 ⌊i·calls/B⌋ 次调用开始，使完成时间错开；"
                       "预热时间不计入速率（假设）。TPOT 每个量子按当时运行请求的上下文计算。",
        "写入干扰（M5）": "写入干扰（M5）：默认 gaps 模式（主机把编程、擦除放进 HBF 空闲间隙，符合第 11.5 节“刷新与读不同时发往同一 die”），"
                     "以 die 为阻塞单位；OCP v0.7.0 没有挂起（suspend）。每个量子按该量子的编程页数和块擦除数（每步平均）计算一次，"
                     "加到下一个量子的步时上（滞后一个量子），所以会降低请求速率；最大单步拖慢取预热后各量子的最大值。",
        "速率按每个 stack 计": "速率按每个 stack 计；N 个 stack 时只模拟一个 stack（并发 round(C/N) 或到达率 λ/N），GPU 计算批为 b × N。",
        "vllm_pool：": None,
    }
    out = []
    for item in items:
        prefix = next((key for key in replace_items if item.startswith(key)), None)
        if prefix is None:
            out.append(item)
        elif replace_items[prefix] is not None:
            out.append(replace_items[prefix])
    out += [
        "段级状态（设计 B 第 4.1 节）：段 = 分区内所有 bank 的同一块号（ppb 256 时超级块 1 GiB、通道块 64 MiB、die 块 16 MiB）；"
        "按行对齐放置时一个条目在段内每个块里的页数相同，所以按段计数是精确的（假设：主机按行写入）。",
        f"量子：每个量子至多 {config.get('quantum_s', 1.0):g} s，在最早的预计读扰阈值或保持到期处截断，但不短于最快的段阈值步数的 "
        f"{config.get('rd_slack', .01):g}（刷新最多晚这么多的 X，计数偏差不超过这个比例）；量子内步时不变，"
        "请求在准确的步完成和到达。暂存在 HBM 的页不计 HBF 读次数，量子内新写入的页从下一个量子开始计读。",
        "读扰与保持刷新：relocate（日志式分配的默认）把段内有效页写到前沿（dedicated 写到条目自己的新段），旧段在下次写第 0 行时才擦除；"
        "replay（layer_major 默认，第 5.7 节）原地重写第 0 行到最后写入的行，包括无效页和填充页（不得跳跃），需要一段的 HBM 副本。"
        "--refresh-idle evict 时段内的空闲缓存条目直接丢弃（以后重算）。",
        "回收：某分区空闲段少于低水位（默认 2 段 + 一个最大请求的段数）时，选需要搬移的页最少的段（贪心；相同时选数据最老的；"
        "`--victim cost_benefit` 按 (1−u)·age/(1+u) 选段），"
        "先驱逐其中的空闲条目（整条丢弃，记为重算 token），再把活跃页搬到前沿（compaction）；--reclaim compact 时空闲条目也搬移。"
        "准入时为请求预留全部预计页数（prefill + 解码）和低水位减一段；做不到时先强制回收，仍放不下就排队。",
        "磨损分配：none 取编号最小的空闲段，dynamic 取 PEC 最小的空闲段，ideal 同 dynamic，"
        "remap 按区域循环轮转（第 11.4.1 节区域重映射，无额外写入）。只在空闲段之间均衡，"
        "不搬移冷数据（没有静态磨损均衡），所以容量被空闲缓存占满时最大 PEC 会远高于平均。"
        "新条目放在有效页最少的分区，相同时选磨损最少的分区（wear none 时选编号最小的）。",
        "hbm_decode 的会话：上一次调用生成的 KV 不写入 HBF，由下一次调用的 prompt 一起写入（偏保守）。"
        "被驱逐的会话历史在下一次调用时整段重算并重写。",
        "vllm_pool：磨损按 packed 计，同时由 VllmTracker 跟踪每层块池的块号分配与释放，统计写序违规"
        "（skip/overlap/live_erase）、时间平均滞留比例（stranded_frac）与合法化所需重放写入量（legalize_gb）；"
        "该策略在规范下不合法（legal = False）。",
        "可选开关：`--prefill colocated` 在每次准入时串行计入 input / prefill_tok_s 的 prefill 时间；"
        "`--pre-open` 在空闲时预先打开段（非重放擦除不计入 M5 干扰，P/E 照计）；"
        "`--separate-prefix` 把共享前缀写到独立前沿以实现冷热分离。",
    ]
    return out


def _analytic_assumptions():
    items = [
        "写入顺序：NAND 块内地址连续、不得跳跃，写页 0（WL-0、STR-0）时自动擦除该块（OCP v0.7.0 第 5.4.1 节第 4、5、9 条）。"
        "第 4 条的“主机通道分配的全局地址范围内不得跳跃”按每个 NAND 块理解（与第 9 条和第 11.5.2.6 节一致）；"
        "若按每个主机通道字面理解，每个主机通道只能有一个追加日志，`dedicated` 不再合法。",
        "每次写页 0 都计一次 P/E，包括从未写过的块（规范字面、偏保守，假设）；块的读计数在擦除时清零（假设）。",
        "没有垃圾回收（第 11.4 节）；空间只能由主机回收（丢弃后重算、搬移或整块重写）。区域重映射（第 11.4.1 节）只改映射。",
        "读扰：主机在一个块被读 X 次后刷新它（第 11.5 节，产品相关；基片按块计读次数，第 4.4 节）。X 扫 1e5–1e7："
        "3D TLC 实测 5.5e4–5.2e5（STRAW 2025 [W]，作为未知 SLC 值的保守替代），经典 SLC 约 1e6（Cai 等 DSN 2015 [W]），"
        "1e7 是猜测 [G]（params.md 第 2、8 节）。",
        "稠密读取每步把每个 KV 页读一次，所以装满数据的块每步读 ρ = ppb × 读比例 × h 次；h（持续偏斜）未知，默认 1 [G]；"
        "稀疏模式的读比例取自校准快照（phase A 同批次的读页数 / 稠密读页数）。",
        "一个请求的所有页按该请求的解码步数 s 计刷新次数 ⌊s·ρ/X⌋（偏保守：生成的页按一开始就在计）；"
        "会话第 k 次调用写入的数据在之后每次调用中被读（每次 out 步，外加一次 prefill 读）。",
        "保持：数据寿命超过 T_ret 时每 T_ret 重写一次；T_ret = 24 h（第 9 节表 33 上电保持 24 h；第 11.5 节建议 24–48 h）。",
        "N_PE 由产品决定（第 9 节表 33 注 2，寄存器 MAXPEC/AVGPEC）；扫 1e4（pSLC 低端 [S]）、1e5（SPLASH、FlashAccel 的 SLC 假设 [P]）、"
        "1e6（FlashAccel 放宽保持后的 10× 假设 [P]）。",
        "每天 P/E = 编程页数 × 擦除放大 × 86,400 / 容量页数，即理想磨损均衡下的平均 PEC（AVGPEC，第 5.9.2.18 节）；"
        "实际最大 PEC 更高。",
        "容量取 --capacity-gib（默认 512 GiB，规范表 3 [P]），与每块页数无关。每块页数：第 11.5.1.8 节提示 1/2/4 MiB 块，"
        "即 4 KiB 页时 256/512/1024 页 [P]；SPLASH、FlashAccel 用 192 [P]。",
        "tPROG = 100 µs、tBERS = 1 ms 取自 SPLASH 表 1 [P]（hbf/configs/parameters.json）；规范未给出。",
        "packed：每个新条目（请求或会话）的 prompt 从一行（1,024 页，分区时 64/16 页）的起点开始，平均补 (行宽 − 1)/2 页填充（假设）；"
        "解码页先在 HBM 暂存，攒满一行再写；擦除放大为 1。",
        "dedicated：每个条目独占整段（超级块 1 GiB，分区时 64/16 MiB），擦除放大 = ⌈P/P_seg⌉·P_seg/P。",
        "layer_major：静态批次波，每波按 phase A 布局（H × B 个 head、波内最长 prompt）整区重写（含保留尾部的填充），"
        "生成的 KV 留在 HBM；读扰或保持到期时整段原地重放（第 5.7 节），需要一段的 HBM 副本。",
        "hbm_decode：生成的 KV 从不写入 HBF（SK hynix H³ 式，[S]）；HBM 暂存随输出长度增长。",
        "vllm_pool：磨损按 packed 计，但空闲块重用会写入仍有有效数据的块中间，违反第 5.4.1 节，规范下不合法。",
        "时钟：每层估算 est = layer_ns + round_ns × 轮数，轮数 = ⌈κ × 每层读页 / 1,024⌉（分区时取最忙分区）；"
        "F1 常数（4,000 ns + 4,406 ns × 轮数）只在 HBF 命令模型、tR = 4,096 ns 下与 MQSim 校准过；"
        "其余计算项、重叠方式和 HBM 基线沿用系统级评估（hbf/system.py）的假设。",
        "闭环：每个 stack 保持 concurrency / stacks 个请求（会话负载为会话数）在解码；prefill 分离（默认），"
        "KV 写入完成后才开始解码，不占解码时间。TPOT 取平均解码上下文下的值。",
        "共享前缀：per_request 时每个共享的请求每步各读一次前缀（序列级），once 时每步只读一次（级联注意力）；"
        "phase A 的按 (step, layer) 去重对应 once。",
        "写入干扰（M5）：默认 gaps 模式（主机把编程、擦除放进 HBF 空闲间隙，符合第 11.5 节“刷新与读不同时发往同一 die”），"
        "以 die 为阻塞单位；OCP v0.7.0 没有挂起（suspend）。只算一遍，不反馈到请求速率。",
        "逐页涓流写需要每个运行请求每 (层, head, HBF 张量) 一个未完成 4 KiB 单元，每个单元要开 tpp × TPOT；"
        "第 5.4.1 节第 6、7 条要求单元在主机设定的时限内写完，产品上限举例为每通道 64 或 128 个，所以按行攒满再写。",
        "权重放在 HBM。若权重放在 HBF（规范第 13 节的用例），权重每 token 都被读，权重块每 X/ppb 步就要刷新一次，不在本模型范围内。",
        "速率按每个 stack 计；N 个 stack 时每个 stack 承担 1/N 的请求，GPU 计算批为 b × N。",
    ]
    return items
