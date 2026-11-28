"""System-level decode model: GPU compute, HBF KV reads, overlap, TPOT and an HBM-resident baseline.

``evaluate`` turns the per-(step, layer) HBF analysis of one stack's share of the batch (group 1
rows) into per-layer and per-step times. ``python -m hbf.system RUN_DIR`` re-evaluates a finished
analysis run with another model, GPU, batch, stack count or overlap mode.
"""
import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

from .model import load_gpu, load_model

NS = 1e9
KV_CONSUME = ("stream", "staged")
SELECTIONS = ("static", "dynamic")
WEIGHTS = ("hbm", "hbf")
ROW_KEYS = ("step", "layer", "est_ns", "bound_ns", "burst_bytes", "hbm_bytes", "attended_tokens", "dequant_ops",
            "attended_elements")
REQUIRED_ROW_KEYS = ("step", "layer", "bound_ns", "burst_bytes", "attended_tokens")


def parse_overlap(text):
    """'serial', 'ideal' or 'prefetch:<d>' (reads run up to d layers ahead) -> (mode, d or None)."""
    if text in ("serial", "ideal"):
        return text, None
    mode, _, depth = str(text).partition(":")
    if mode == "prefetch" and depth.isdigit():
        return mode, int(depth)
    raise ValueError(f"Invalid overlap {text!r}: use serial, ideal or prefetch:<d>")


@dataclass(frozen=True)
class SystemOptions:
    overlap: str = "serial"  # serial | prefetch:<d> | ideal
    kv_consume: str = "stream"  # serial and ideal: attention streams HBF pages or reads them staged in HBM
    weights: str = "hbm"

    def __post_init__(self):
        parse_overlap(self.overlap)
        if self.kv_consume not in KV_CONSUME:
            raise ValueError(f"kv_consume must be one of {', '.join(KV_CONSUME)}")
        if self.weights not in WEIGHTS:
            raise ValueError(f"weights must be one of {', '.join(WEIGHTS)}")
        if self.weights == "hbf":
            raise NotImplementedError("Weights on HBF (OCP spec section 13) are not implemented; use weights=hbm")

    @property
    def mode(self):
        return parse_overlap(self.overlap)[0]

    @property
    def depth(self):
        return parse_overlap(self.overlap)[1]

    @property
    def staged(self):
        """Prefetched data always lands in HBM first."""
        return self.mode == "prefetch" or self.kv_consume == "staged"


@dataclass
class SystemResult:
    layers: list  # system_layers rows
    steps: list  # system_steps rows
    assumptions: list  # report lines (Chinese)
    context: dict  # model, gpu, options, normalized meta and per-step constants for the report


def _plain(value):
    """Python scalar from a numpy scalar; other values unchanged."""
    return value.item() if hasattr(value, "item") and not isinstance(value, (str, bytes)) else value


def _number(value, name, optional=False):
    value = _plain(value)
    if value is None and optional:
        return None
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative number, got {value!r}")
    return value


def _count(value, name, optional=False, positive=False):
    value = _plain(value)
    if value is None and optional:
        return None
    if type(value) is not int or value < (1 if positive else 0):
        raise ValueError(f"{name} must be a {'positive' if positive else 'non-negative'} integer, got {value!r}")
    return value


def system_meta(meta):
    """Validate the run metadata: batch B, stacks N, analysis batch b = ceil(B / N), selection, KV sizes."""
    batch = _count(meta.get("batch"), "batch", positive=True)
    stacks = _count(1 if meta.get("stacks") is None else meta["stacks"], "stacks", positive=True)
    expected = -(-batch // stacks)
    analysis_batch = meta.get("analysis_batch")
    analysis_batch = _count(expected if analysis_batch is None else analysis_batch, "analysis_batch", positive=True)
    if analysis_batch != expected:
        raise ValueError(f"The HBF rows cover analysis_batch = {analysis_batch} requests per stack, but "
                         f"ceil(batch / stacks) = ceil({batch} / {stacks}) = {expected}; choose batch and "
                         "stacks that match the analysis")
    selection = meta.get("selection")
    if selection not in (None,) + SELECTIONS:
        raise ValueError(f"selection must be one of {', '.join(SELECTIONS)}")
    kv_format = meta.get("kv_format")
    if kv_format is not None and not isinstance(kv_format, str):
        raise ValueError("kv_format must be a string")
    return {
        "batch": batch, "stacks": stacks, "analysis_batch": analysis_batch, "selection": selection,
        "kv_format": kv_format,
        "kv_bytes_per_request": _count(meta.get("kv_bytes_per_request"), "kv_bytes_per_request", optional=True),
        "hbm_tier_bytes_per_request": _count(meta.get("hbm_tier_bytes_per_request") or 0,
                                             "hbm_tier_bytes_per_request"),
    }


def _steps(layer_rows, layers):
    """{step: [row of layer 0, ..., row of layer L-1]} with validated numbers; every layer is required."""
    steps = {}
    for raw in layer_rows:
        missing = [key for key in REQUIRED_ROW_KEYS if raw.get(key) is None]
        if missing:
            raise ValueError(f"Layer row {raw} lacks {', '.join(missing)}")
        step, layer = _count(raw["step"], "step"), _count(raw["layer"], "layer")
        if layer >= layers:
            raise ValueError(f"Step {step} has layer {layer}, but the model has {layers} layers")
        if layer in steps.setdefault(step, {}):
            raise ValueError(f"Duplicate row for step {step}, layer {layer}")
        steps[step][layer] = {
            "est_ns": _number(raw.get("est_ns"), "est_ns", optional=True),
            "bound_ns": _number(raw["bound_ns"], "bound_ns"),
            "burst_bytes": _number(raw["burst_bytes"], "burst_bytes"),
            "hbm_bytes": _number(raw.get("hbm_bytes") or 0, "hbm_bytes"),
            "attended_tokens": _number(raw["attended_tokens"], "attended_tokens"),
            "dequant_ops": _number(raw.get("dequant_ops") or 0, "dequant_ops"),
            "attn_dims": _count(raw.get("attn_dims"), "attn_dims", optional=True, positive=True),
            "attended_elements": _number(raw.get("attended_elements"), "attended_elements", optional=True),
        }
    if not steps:
        raise ValueError("No layer rows to evaluate")
    for step, rows in steps.items():
        missing = sorted(set(range(layers)) - set(rows))
        if missing:
            raise ValueError(f"Step {step} has no rows for layers {missing[:8]}{' ...' if len(missing) > 8 else ''}; "
                             "the analysis must emit a row (zeros) for every layer")
    return {step: [steps[step][layer] for layer in range(layers)] for step in sorted(steps)}


def linear_ns(params, batch, model, gpu, byte_params=None):
    """Roofline time of a weight-stationary matrix product over the batch: max(compute, weight bytes).

    For MoE layers ``params`` is the active parameter count (per-token FLOPs) and ``byte_params``
    is the expected parameter count touched across the batch (weight bytes read from HBM).
    """
    bytes_ = params if byte_params is None else byte_params
    return max(2 * batch * params / gpu.flops, bytes_ * model.weight_bytes / gpu.bandwidth) * NS


def lm_head_ns(model, gpu, batch):
    return linear_ns(model.vocab * model.hidden, batch, model, gpu)


def timeline(h, pre, attn, post, overlap="serial", staged=False):
    """Layer time T and HBF wait of one step from per-layer lists (ns).

    ``attn`` is the attention term the mode uses (a_l when streaming, a_l_staged when staged or
    prefetched). Returns (T, wait, per-layer waits or None for ideal); T = sum(pre + attn + post) + wait.
    """
    mode, depth = parse_overlap(overlap)
    layers = range(len(h))
    if mode == "serial" and not staged:
        waits = [max(0, h[l] - attn[l]) for l in layers]
        return sum(pre[l] + max(h[l], attn[l]) + post[l] for l in layers), sum(waits), waits
    if mode == "serial":
        return sum(pre[l] + h[l] + attn[l] + post[l] for l in layers), sum(h), list(h)
    if mode == "ideal":
        work = sum(pre[l] + attn[l] + post[l] for l in layers)
        return max(sum(h), work), max(0, sum(h) - work), None
    # prefetch:d: r_start(l) = max(r_end(l-1), c_end(l-1-d)); r_end(l) = r_start(l) + h_l;
    # c_end(l) = max(c_end(l-1) + pre_l, r_end(l)) + attn_l + post_l. Indexes below 0 read 0.
    r_end, c_end, waits = 0, [], []
    for l in layers:
        before = c_end[l - 1] if l else 0
        r_end = max(r_end, c_end[l - 1 - depth] if l - 1 - depth >= 0 else 0) + h[l]
        waits.append(max(0, r_end - (before + pre[l])))
        c_end.append(max(before + pre[l], r_end) + attn[l] + post[l])
    return (c_end[-1] if c_end else 0), sum(waits), waits


def evaluate(layer_rows, model, gpu, options, meta):
    """Per-layer and per-step decode times for compute batch B (see SystemResult).

    ``layer_rows`` are the group 1 analysis rows of one stack (analysis batch b = ceil(B / N)); demand
    terms scale by B / b, HBF times do not (stacks read in parallel). ``meta`` holds batch, stacks,
    analysis_batch, selection, kv_format, kv_bytes_per_request and hbm_tier_bytes_per_request.
    """
    meta = system_meta(meta)
    if options.mode == "prefetch" and meta["selection"] != "static":
        raise ValueError("Prefetching the next layer's KV needs a static (query-independent) selection; "
                         f"selection is {meta['selection'] or 'not given'}")
    batch, scale = meta["batch"], meta["batch"] / meta["analysis_batch"]
    flops, vector, bandwidth = gpu.flops, gpu.vector, gpu.bandwidth
    pre = [linear_ns(model.layer_pre_params(layer), batch, model, gpu) for layer in range(model.layers)]
    post = [linear_ns(model.layer_post_active_params(layer), batch, model, gpu,
                      model.layer_post_touched_params(layer, batch)) for layer in range(model.layers)]
    head = lm_head_ns(model, gpu, batch)
    extra = head + model.layers * gpu.layer_overhead_ns
    group = model.q_heads // model.kv_heads
    weights = model.weights_bytes
    layer_out, step_out = [], []
    for step, rows in _steps(layer_rows, model.layers).items():
        terms = []  # attention per layer: a_l reads only the HBM tier, a_l_staged also the HBF bytes
        for row in rows:
            elements = row["attended_elements"]
            if elements is None:
                elements = (row["attn_dims"] or model.default_attn_dims) * row["attended_tokens"]
            attn_flops = 2 * group * elements * scale
            dequant = row["dequant_ops"] * scale / vector * NS
            compute = attn_flops / flops * NS + dequant
            terms.append({
                "attn_compute_ns": compute, "dequant_ns": dequant,
                "attn_ns": max(compute, row["hbm_bytes"] * scale / bandwidth * NS),
                "attn_staged_ns": max(compute, (row["hbm_bytes"] + row["burst_bytes"]) * scale / bandwidth * NS),
                "attn_flops": attn_flops,
            })
        attn = [term["attn_staged_ns" if options.staged else "attn_ns"] for term in terms]
        work = sum(p + a + q for p, a, q in zip(pre, attn, post))
        tpot, waits, wait_share = {}, {}, {}
        for kind in ("est", "bound"):
            h = [row[f"{kind}_ns"] for row in rows]
            if any(value is None for value in h):
                tpot[kind], waits[kind], wait_share[kind] = None, [None] * len(rows), None
                continue
            total, wait, per_layer = timeline(h, pre, attn, post, options.overlap, options.staged)
            tpot[kind] = total + extra
            waits[kind] = per_layer or [None] * len(rows)
            wait_share[kind] = wait / total if total else 0.0
        # HBM baseline: the same compute with all KV bytes read from HBM (= a_l_staged).
        hbm_tpot = sum(p + term["attn_staged_ns"] + q for p, term, q in zip(pre, terms, post)) + extra
        for layer, (row, term) in enumerate(zip(rows, terms)):
            layer_out.append({
                "step": step, "layer": layer, "hbf_est_ns": row["est_ns"], "hbf_bound_ns": row["bound_ns"],
                "pre_ns": pre[layer], "attn_compute_ns": term["attn_compute_ns"], "dequant_ns": term["dequant_ns"],
                "attn_ns": term["attn_ns"], "attn_staged_ns": term["attn_staged_ns"], "post_ns": post[layer],
                "hbm_layer_ns": pre[layer] + term["attn_staged_ns"] + post[layer], "attn_flops": term["attn_flops"],
                "wait_est_ns": waits["est"][layer], "wait_bound_ns": waits["bound"][layer],
            })
        buffer = ((options.depth + 1) * max(row["burst_bytes"] for row in rows) * scale
                  if options.mode == "prefetch" else 0)
        kv = meta["kv_bytes_per_request"]
        step_out.append({
            "step": step, "tpot_hbf_est_ns": tpot["est"], "tpot_hbf_bound_ns": tpot["bound"],
            "tpot_hbm_ns": hbm_tpot,
            "tokens_per_s_hbf": None if tpot["est"] is None else batch / tpot["est"] * NS,
            "tokens_per_s_hbf_bound": batch / tpot["bound"] * NS,
            "tokens_per_s_hbm": batch / hbm_tpot * NS,
            "hbf_over_hbm": None if tpot["est"] is None else tpot["est"] / hbm_tpot,
            "hbf_over_hbm_bound": tpot["bound"] / hbm_tpot,
            "gpu_ns": work + extra,
            "hbf_read_est_ns": None if tpot["est"] is None else sum(row["est_ns"] for row in rows),
            "hbf_read_bound_ns": sum(row["bound_ns"] for row in rows),
            "hbf_wait_share": wait_share["est"], "hbf_wait_share_bound": wait_share["bound"],
            "hbm_fits": None if kv is None else weights + batch * kv <= gpu.hbm_capacity,
            "hbm_fits_hbf": weights + batch * meta["hbm_tier_bytes_per_request"] + buffer <= gpu.hbm_capacity,
        })
    pre_avg = pre[0] if len(set(pre)) == 1 else sum(pre) / len(pre)
    post_avg = post[0] if len(set(post)) == 1 else sum(post) / len(post)
    context = {"model": model, "gpu": gpu, "options": options, "meta": meta, "scale": scale,
               "pre_ns": pre_avg, "post_ns": post_avg, "pre_layer_ns": pre, "post_layer_ns": post,
               "lm_head_ns": head, "weights_bytes": weights,
               "ridge_batch": model.weight_bytes * flops / (2 * bandwidth)}
    return SystemResult(layer_out, step_out, assumptions(context), context)


def assumptions(context):
    """Every modelling assumption behind the result, as Chinese report lines."""
    model, gpu, options, meta = context["model"], context["gpu"], context["options"], context["meta"]
    batch, stacks, b = meta["batch"], meta["stacks"], meta["analysis_batch"]
    lines = [
        "线性层（QKV 投影、输出投影、MLP）和 lm_head 按 roofline 计时：max(2·B·P / F, P·w / BW)，"
        "F = tensor_flops × compute_eff，BW = hbm_bytes_s × memory_eff，w = weight_bits / 8。"
        "每个序列每步只有一个 query token（不考虑投机解码）；lm_head 每步一次。",
        f"compute_eff = {gpu.compute_eff:g} 是 prefill 的 MFU 数字（假设），用于 decode；"
        f"按此值线性层在 B < {context['ridge_batch']:.0f} 时受 HBM 带宽限制（B* = w·F / (2·BW)）。",
        f"memory_eff = {gpu.memory_eff:g} 是 decode 能达到的 HBM 带宽比例（假设）。",
        f"tensor_flops 取 BF16 稠密峰值；weight_bits = {model.weight_bits} 只决定权重字节，不改变算力。",
        "权重全部放在 HBM（weights = hbm）。norm、bias、RoPE、残差加法、embedding 查表和采样不计时间；"
        "norm 与 bias 权重不计容量。",
        "注意力 FLOPs = 2 × (q_heads / kv_heads) × E。E 是被注意的 K、V 元素数：各数据张量（不含量化参数等"
        "辅助张量和选择元数据，含 HBM 层）读到的 token 数 × 维数，按 (请求, KV head) 汇总；K、V 读同一批 token 时"
        "E = A × (d_k + d_v)。没有 attended_elements 列时用 attn_dims × A。softmax、exp、缩放等逐元素运算忽略。",
        "按 query head 选择 token 时，A 按 KV head 汇总，因此是上界。选择打分（如 Quest 页摘要、top-k）的 FLOPs 不计。",
        f"反量化时间 = dequant_ops / (vector_ops × compute_eff)，vector_ops = {gpu.vector_ops / 1e12:g} TFLOP/s "
        "取该 GPU 非 tensor FP32 峰值（假设），与注意力矩阵乘串行相加。",
        f"HBF 时间 h 是一个 stack 上 b = {b} 个请求的分析结果；{stacks} 个 stack 并行读，由同一块 GPU 驱动。"
        f"计算项和 HBM 项（attended_tokens、dequant_ops、hbm_bytes、burst_bytes）按 B/b = {context['scale']:g} 放大，"
        "h 不放大。",
        "est TPOT 用 est_ns（整层设备视角，命令模型随配置），bound TPOT 用 bound_ns（下界）。"
        "某一步任一层 est_ns 为空时该步 est TPOT 为空，不回退到 bound。",
        f"层内 pre、注意力、post 依次执行；每层另加 layer_overhead_ns = {gpu.layer_overhead_ns:g} ns（kernel 启动等，假设）。",
        "HBM 基线：KV 以同一存储格式（64 B 粒度的 burst 字节加 HBM 层字节）全部从 HBM 读取，"
        "层时间 = pre + max(KV 字节 / BW, 注意力计算) + post。",
        "容量：hbm_fits = 权重 + B × 每请求 KV 页占用 ≤ HBM 容量（KV 全放 HBM）；"
        "hbm_fits_hbf = 权重 + B × 每请求 HBM 层字节 + 预取缓冲 ≤ HBM 容量。"
        "HBM 容量按标称 GB 的十进制字节，比驱动报告的可用容量约小 7%，偏保守；激活、CUDA 上下文和内存碎片不计。",
        f"效率不对称：HBM 按 memory_eff = {gpu.memory_eff:g} 折算；HBF 的 est 已含命令、R/B 和传输开销（按 MQSim 的命令"
        "模型校准），bound 是不含任何开销的下界。公平的对比是 est 对 HBM；bound 对 HBM 偏向 HBF。"
        "未给 --ucie-grade 时不计主机链路。",
        "decode 追加写（新 token 的 KV 写入 HBF）以及 HBF 编程、擦除对读的干扰不计。",
    ]
    if model.attention == "mla":
        lines.append(f"MLA 吸收式解码：kv_lora_rank = {model.kv_lora_rank}、rope_dim = {model.rope_dim}，"
                     f"每 token 每查询头注意 {model.default_attn_dims} 个元素；W_UK 计入 pre，W_UV 计入 post。")
    if model.experts:
        touched = model.experts_touched(batch)
        lines.append(f"MoE（{model.experts} 路由专家 + {model.shared_experts} 共享专家，每 token 激活 {model.top_k} 个）："
                     f"计算按每 token 激活专家计，权重读取按批内期望触达专家数 E(1−(1−k/E)^B) = {touched:.2f} "
                     "加共享专家计；采用 DP-attention 设定（单卡持有所属请求的 KV，不计专家并行通信）。")
    if batch % stacks:
        lines.append(f"B mod N = {batch % stacks} ≠ 0：B/b 是平均近似，而 h 是最忙的 stack（b 个请求）的时间。")
    if options.mode == "serial" and not options.staged:
        lines.append("serial + stream：注意力边读边算（HBF 页到达即消费），层时间 = pre + max(h, a) + post，"
                     "a 只计 HBM 层（如 KIVI 残差窗口）的字节；等待占比 = Σ max(0, h − a) / T。")
    elif options.mode == "serial":
        lines.append("serial + staged：HBF 读完一层写入 HBM 后再计算，层时间 = pre + h + a_staged + post，"
                     "a_staged 计入 HBM 层与 HBF 层全部字节；写入 HBM 的开销不计；等待占比 = Σ h / T。")
    elif options.mode == "prefetch":
        d = options.depth
        lines.append(f"prefetch:{d}：第 l 层的 HBF 读最早在第 l−{d + 1} 层计算结束后开始（缓冲 {d + 1} 层），"
                     "也不早于第 l−1 层读完；预取数据先进入 HBM，注意力项用 a_staged，kv_consume 不起作用；"
                     "写入 HBM 的开销不计。查询相关的选择（selection = dynamic）不能预取。"
                     "等待占比 = Σ max(0, r_end(l) − (c_end(l−1) + pre)) / T。")
        lines.append(f"预取缓冲 = {d + 1} 层 × 该步最大层 burst 字节 × B/b，计入 hbm_fits_hbf。")
    else:
        lines.append(f"ideal：HBF 读与计算完全重叠，T = max(Σ h, Σ (pre + a + post))，a 按 kv_consume = "
                     f"{options.kv_consume} 取值，是上界；等待占比 = max(0, Σ h − Σ (pre + a + post)) / T。")
    if options.mode != "prefetch":
        lines.append("serial 与 ideal 不计预取缓冲。")
    lines.append("等待占比的分母 T 是各层时间之和，不含 lm_head 与 layer_overhead。")
    return lines


def _ms(value):
    return "—" if value is None else f"{value / 1e6:.3f}"


def _fits(value):
    return "—" if value is None else ("是" if value else "否")


def report_lines(result):
    """Chinese markdown section for system_report.md (or the run's report.md)."""
    context = result.context
    model, gpu, options, meta = context["model"], context["gpu"], context["options"], context["meta"]
    consume = "`staged`（prefetch 固定）" if options.mode == "prefetch" else f"`{options.kv_consume}`"
    lines = [
        "# 系统级评估", "",
        f"模型 `{model.name}`：{model.layers} 层，hidden {model.hidden}，Q/KV head {model.q_heads}/{model.kv_heads}，"
        f"head_dim {model.head_dim}，权重 {context['weights_bytes'] / 1e9:.2f} GB。"
        f"GPU `{gpu.name}`：BF16 {gpu.tensor_flops / 1e12:g} TFLOP/s × {gpu.compute_eff:g}，"
        f"HBM {gpu.hbm_bytes_s / 1e12:g} TB/s × {gpu.memory_eff:g}，容量 {gpu.hbm_capacity / 1e9:g} GB。", "",
        f"计算批 B = {meta['batch']}，HBF stack {meta['stacks']} 个，每个 stack 的分析批 b = {meta['analysis_batch']}；"
        f"重叠方式 `{options.overlap}`，KV 消费 {consume}，选择 `{meta['selection'] or '未给出'}`，"
        f"KV 格式 `{meta['kv_format'] or '未给出'}`。"
        f"每层 pre {context['pre_ns'] / 1e3:.1f} µs、post {context['post_ns'] / 1e3:.1f} µs，"
        f"lm_head {context['lm_head_ns'] / 1e3:.1f} µs。", "",
        "## 每步结果", "",
        "TPOT 单位 ms。tok/s 与 HBF/HBM 用 est（为空时见 bound 列）；等待占比是计算等待 HBF 的时间占各层时间之和的比例。", "",
        "| Step | TPOT HBF est | TPOT HBF bound | TPOT HBM | HBF tok/s | HBM tok/s | HBF/HBM | HBF/HBM（bound） "
        "| GPU 计算 | 等待占比 | KV 全放 HBM 放得下 | HBF 方案 HBM 放得下 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for row in result.steps:
        lines.append(
            f"| {row['step']} | {_ms(row['tpot_hbf_est_ns'])} | {_ms(row['tpot_hbf_bound_ns'])} | "
            f"{_ms(row['tpot_hbm_ns'])} | "
            f"{'—' if row['tokens_per_s_hbf'] is None else format(row['tokens_per_s_hbf'], '.0f')} | "
            f"{row['tokens_per_s_hbm']:.0f} | "
            f"{'—' if row['hbf_over_hbm'] is None else format(row['hbf_over_hbm'], '.2f')} | "
            f"{row['hbf_over_hbm_bound']:.2f} | {_ms(row['gpu_ns'])} | "
            f"{'—' if row['hbf_wait_share'] is None else format(row['hbf_wait_share'], '.3f')} | "
            f"{_fits(row['hbm_fits'])} | {_fits(row['hbm_fits_hbf'])} |")
    lines += ["", "## 假设", ""] + [f"- {line}" for line in result.assumptions]
    return lines


def write_table(path, rows):
    """CSV of plain Python values (numpy scalars converted; None written empty)."""
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({key: _plain(value) for key, value in row.items()} for row in rows)


def write_outputs(result, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    write_table(directory / "system_layers.csv", result.layers)
    write_table(directory / "system_steps.csv", result.steps)
    (directory / "system_report.md").write_text("\n".join(report_lines(result)) + "\n")


def _parse_field(text):
    if text is None or text == "":
        return None
    try:
        return int(text)
    except ValueError:
        return float(text)


def read_layers(path):
    """Group 1 rows of analysis_layers.csv as evaluate() input; empty fields parse as None."""
    with open(path, newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row.get("group") == "1"]
    if not rows:
        raise ValueError(f"{path} has no group 1 rows")
    keys = ROW_KEYS + ("attn_dims",)
    return [{key: _parse_field(row[key]) for key in keys if key in row} for row in rows]


def check_shape(shape, model, analysis_batch):
    """The analysed storage shape must be the model's, with kv_heads = model KV heads x analysis batch."""
    if not shape:
        return
    if (shape.get("layers") != model.layers or shape.get("head_dim") != model.head_dim
            or shape.get("kv_heads") != model.kv_heads * analysis_batch):
        raise ValueError(f"meta.json shape {shape} does not match model {model.name} "
                         f"({model.layers} layers, {model.kv_heads} KV heads, head_dim {model.head_dim}) "
                         f"with analysis batch {analysis_batch}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m hbf.system",
                                     description="Re-evaluate the system model of a finished analysis run.")
    parser.add_argument("run_dir", type=Path, help="run directory with analysis_layers.csv and meta.json")
    parser.add_argument("--model", help="model preset, JSON text or JSON file (default: meta.json)")
    parser.add_argument("--gpu", help="GPU preset, JSON text or JSON file (default: meta.json)")
    parser.add_argument("--batch", type=int, help="compute batch B (default: meta.json)")
    parser.add_argument("--stacks", type=int, help="HBF stacks N (default: meta.json)")
    parser.add_argument("--overlap", help="serial, prefetch:<d> or ideal (default: meta.json or serial)")
    parser.add_argument("--kv-consume", choices=KV_CONSUME, help="default: meta.json or stream")
    parser.add_argument("--selection", choices=SELECTIONS,
                        help="static or dynamic page selection (default: meta.json); prefetch needs static")
    parser.add_argument("--output", type=Path, help="output directory (default: RUN_DIR)")
    args = parser.parse_args(argv)
    meta = json.loads((args.run_dir / "meta.json").read_text())
    system = dict(meta.get("system") or {})
    for key in ("model", "gpu", "batch", "stacks", "selection"):
        if getattr(args, key) is not None:
            system[key] = getattr(args, key)
    for key in ("model", "gpu"):
        if system.get(key) is None:
            parser.error(f"--{key} is required: meta.json has no system.{key}")
    if system.get("batch") is None:
        parser.error("--batch is required: meta.json has no system.batch")
    try:
        model, gpu = load_model(system["model"]), load_gpu(system["gpu"])
        shape = meta.get("shape")
        if system.get("analysis_batch") is None and shape:
            # A run without system metadata analysed shape.kv_heads / model KV heads requests per stack.
            if shape.get("kv_heads", 0) % model.kv_heads:
                raise ValueError(f"meta.json shape has {shape.get('kv_heads')} KV heads, not a multiple of "
                                 f"{model.name}'s {model.kv_heads}")
            system["analysis_batch"] = shape["kv_heads"] // model.kv_heads
        options = SystemOptions(overlap=args.overlap or system.get("overlap") or "serial",
                                kv_consume=args.kv_consume or system.get("kv_consume") or "stream")
        check_shape(meta.get("shape"), model, system_meta(system)["analysis_batch"])
        result = evaluate(read_layers(args.run_dir / "analysis_layers.csv"), model, gpu, options, system)
    except (ValueError, NotImplementedError) as error:
        parser.error(str(error))
    write_outputs(result, args.output or args.run_dir)
    return result


if __name__ == "__main__":
    main()
