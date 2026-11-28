"""Extension KV storage formats registered by ``hbf.ext.kv_terms``: mla, mla_fp8, loki, sparq."""
from ...frontend.kv.formats import FP8_DEQUANT_OPS, KVFormat
from ...frontend.kv.tensors import TensorSpec

MLA_DEFAULTS = {
    "lora": 512, "rope": 64, "split": 0, "indexer": 0, "indexer_hbm": 0,
    "index_heads": 64, "index_dims": 128, "q_heads": 0,
}


def _flag(params, key):
    value = params[key]
    if type(value) is not int or value not in (0, 1):
        raise ValueError(f"{key} must be 0 or 1")
    return bool(value)


def mla_format(name, params, q_heads=None):
    """MLA's per-token latent (DeepSeek-V2) as one or two tensors, optionally with a DSA indexer."""
    lora, rope, heads, dims = (params[key] for key in ("lora", "rope", "index_heads", "index_dims"))
    if any(type(value) is not int or value <= 0 for value in (lora, rope, heads, dims)):
        raise ValueError("lora, rope, index_heads and index_dims must be positive integers")
    if type(params["q_heads"]) is not int or params["q_heads"] < 0:
        raise ValueError("q_heads must be a positive integer")
    split, indexer, indexer_hbm = (_flag(params, key) for key in ("split", "indexer", "indexer_hbm"))
    if indexer and indexer_hbm:
        raise ValueError("indexer and indexer_hbm are mutually exclusive")
    fp8 = name == "mla_fp8"
    if fp8 and lora % 128:
        raise ValueError("mla_fp8 needs lora to be a multiple of 128 (one FP32 scale per 128 elements)")
    notes = [f"MLA：每层一个潜向量，lora {lora} + rope {rope} 维；吸收式解码每个查询头每 token 计算 "
             f"QK {lora + rope} + PV {lora} = {2 * lora + rope} 个元素"]
    if fp8:
        latent = lora + 4 * lora // 128
        notes.append(f"FP8 潜向量：{lora} B + {4 * lora // 128} B scale，rope {2 * rope} B 保持 BF16；反量化只计潜向量")
        if split:
            tensors = [TensorSpec("C", 8, latent, attn_dims=2 * lora, logical_dims=lora, dequant_ops=FP8_DEQUANT_OPS),
                       TensorSpec("R", 16, rope, attn_dims=rope)]
        else:
            tensors = [TensorSpec("C", 8, latent + 2 * rope, attn_dims=2 * lora + rope, logical_dims=lora + rope,
                                  dequant_ops=FP8_DEQUANT_OPS * lora / (lora + rope))]
    elif split:
        tensors = [TensorSpec("C", 16, lora, attn_dims=2 * lora), TensorSpec("R", 16, rope, attn_dims=rope)]
    else:
        tensors = [TensorSpec("C", 16, lora + rope, attn_dims=2 * lora + rope)]
    if split:
        notes.append("潜向量与 rope 分成两个张量，各自整页存放；稀疏读时一个 token 要读两页")
    if indexer or indexer_hbm:
        q_heads = params["q_heads"] or q_heads
        if type(q_heads) is not int or q_heads <= 0:
            raise ValueError("The DSA indexer needs a positive integer q_heads (from --model or the q_heads parameter)")
        if heads * dims % q_heads:
            raise ValueError(f"index_heads x index_dims ({heads * dims}) must be a multiple of q_heads ({q_heads})")
        tensors.append(TensorSpec("K_idx", 8, dims + 4, tier="hbm" if indexer_hbm else "hbf", role="meta",
                                  attn_dims=heads * dims // q_heads, logical_dims=dims))
        notes.append(f"DSA 索引器：每 token {dims} 维 FP8 键 + 4 B scale，每步整体读取（{'HBM' if indexer_hbm else 'HBF'}）；"
                     f"打分计算按 {heads} 个索引头 × {dims} 维计入（假设 FP8 与 BF16 同速）")
    return tuple(tensors), 2 * lora + rope, notes


def build_mla(name, params, head_dim, q_heads=None):
    tensors, attn_dims, notes = mla_format(name, params, q_heads)
    return KVFormat(name, params, tensors, attn_dims, tuple(notes))


def build_loki(name, params, head_dim, q_heads=None):
    d = head_dim
    rank = params["rank"]
    if type(rank) is not int or not 0 < rank < d:
        raise ValueError(f"rank must be an integer in (0, {d})")
    tensors = (TensorSpec("K_lo", 16, rank), TensorSpec("K_hi", 16, d - rank), TensorSpec("V", 16, d))
    notes = (f"Loki：K 按静态 PCA 前缀拆成 K_lo（{rank} 维，每步全上下文读取打分）与 "
             f"K_hi（{d - rank} 维，只读选中 token）；PCA 投影不计入",)
    return KVFormat(name, params, tensors, 2 * d, notes, extra={"full_tensors": ("K_lo",)})


def build_sparq(name, params, head_dim, q_heads=None):
    d = head_dim
    channels, groups = params["channels"], params["groups"]
    if type(groups) is not int or not 1 <= groups <= d or d % groups:
        raise ValueError(f"groups must be a positive divisor of head_dim ({d})")
    if type(channels) is not int or not 1 <= channels <= d:
        raise ValueError(f"channels must be an integer in [1, {d}]")
    dual = _flag(params, "dual")
    group_dims = d // groups
    group_names = tuple(f"K_g{i}" for i in range(groups))
    if dual:
        tensors = tuple(TensorSpec(gname, 16, group_dims, role="aux") for gname in group_names) + (
            TensorSpec("K", 16, d), TensorSpec("V", 16, d))
        notes = (f"SparQ：K 按通道拆成 {groups} 个分组张量 K_g0..K_g{groups - 1}（每组 {group_dims} 维）供全上下文打分，"
                 f"每步按 |q| 选 {channels}/{d} 通道；dual=1 另存一份 token-major K 供精确阶段稀疏读"
                 "（总 KV 字节 1.5 倍，K 写两份）；打分计算不重复计入",)
    else:
        tensors = tuple(TensorSpec(gname, 16, group_dims) for gname in group_names) + (TensorSpec("V", 16, d),)
        notes = (f"SparQ：K 按通道拆成 {groups} 个分组张量 K_g0..K_g{groups - 1}（每组 {group_dims} 维），"
                 f"每步按 |q| 选 {channels}/{d} 通道全上下文打分，精确阶段读剩余分组与 V 的选中 token",)
    return KVFormat(name, params, tensors, 2 * d, notes,
                    extra={"channel_groups": group_names, "channel_select_channels": channels})


def register_formats(registry):
    registry.register_format("mla", build_mla, defaults=MLA_DEFAULTS)
    registry.register_format("mla_fp8", build_mla, defaults=MLA_DEFAULTS)
    registry.register_format("loki", build_loki, defaults={"rank": 32})
    registry.register_format("sparq", build_sparq, defaults={"channels": 32, "groups": 8, "dual": 0})
