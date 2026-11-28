"""Named KV storage formats (``--format NAME[:key=value,...]``) expanded into tensor specs.

Every preset describes what one KV head stores per token; the numbers that are not fixed by the
cited method are assumptions and are listed in ``KVFormat.notes``.

| preset | tensors (bits x dims, window, tier) | source |
|---|---|---|
| ``bf16`` | K, V 16 x d | default |
| ``fp8`` | K, V 8 x d (per-tensor scale, not stored) | vLLM FP8 KV cache |
| ``int4``/``int2`` (``group``) | K, V b x d; K_qp, V_qp 16 x 2d/g (scale and zero per g channels) | group quantization |
| ``kivi4``/``kivi2`` (``group``, ``residual``) | K, V b x d and K_qp, V_qp 16 x 2d/G on [0, -R]; K_res, V_res 16 x d on [-R, end] in HBM | KIVI (ICML 2024): G = 32, R = 128 |
| ``think`` (``ratio``) | K 16 x (1-r)d, V 16 x d | ThinK (ICLR 2025) |
| ``quest`` (``page``) | K, V 16 x d; K_sum 16 x 2d/p (min and max keys per p-token page), role meta | Quest (ICML 2024): p = 16 |
| ``mla`` (``lora``, ``rope``, ``split``, ``indexer``, ``indexer_hbm``, ``index_heads``, ``index_dims``, ``q_heads``) | C 16 x (lora + rope), attended as 2 lora + rope; ``split=1``: C 16 x lora and R 16 x rope; ``indexer=1``: K_idx 8 x (index_dims + 4) (FP8 keys and an FP32 scale), role meta | DeepSeek-V2 (MLA): lora 512, rope 64; DeepSeek-V3.2 (DSA): 64 index heads of 128 dims |
| ``mla_fp8`` (same) | C 8 x (lora + 4 lora/128 + 2 rope) bytes: FP8 latent, FP32 scale per 128, BF16 rope | FlashMLA FP8 KV cache, 656 B per token |

One KV "head" per layer holds MLA's latent, so MLA runs use ``--heads 1`` (or an MLA model preset);
``q_heads`` (from ``--model`` or the parameter) sets the indexer's attention FLOPs per query head.
``fused=1`` on the int and KIVI presets packs K, V and both scale tensors into one row per token
(tensor KV, ``bits=8``, ``dims`` = bytes), so a selected block reads one page set instead of four;
``pow2=1`` pads the row to a power of two so that power-of-two selection blocks align to page
boundaries instead of straddling two pages.

``think=r`` also applies to the int and KIVI presets (K keeps (1-r)d channels). KIVI's per-channel
K scales are one row per G tokens; they are amortized as per-token rows of 2d/G elements, as are
Quest's per-page summaries. Dequantization costs ``dequant_ops`` GPU vector operations per stored
element read: 1 for FP8 and 3 (unpack, scale, shift) for integer formats, both assumptions.
"""
from dataclasses import dataclass, field

from .access import TensorSpec

INT_DEQUANT_OPS = 3  # assumption: unpack, multiply by scale, add zero point
FP8_DEQUANT_OPS = 1  # assumption: one conversion per element
PRESETS = ("bf16", "fp8", "int4", "int2", "kivi4", "kivi2", "think", "quest", "mla", "mla_fp8")
MLA = {"lora": 512, "rope": 64, "split": 0, "indexer": 0, "indexer_hbm": 0, "index_heads": 64, "index_dims": 128,
       "q_heads": 0}
DEFAULTS = {"int4": {"group": 128}, "int2": {"group": 128}, "kivi4": {"group": 32, "residual": 128},
            "kivi2": {"group": 32, "residual": 128}, "think": {"ratio": .5}, "quest": {"page": 16},
            "mla": MLA, "mla_fp8": MLA}
INT = ("int4", "int2", "kivi4", "kivi2")


@dataclass(frozen=True)
class KVFormat:
    name: str
    params: dict
    tensors: tuple
    attn_dims: int  # d_k + d_v: elements per token one query head multiplies in QK^T and PV
    notes: tuple = field(default_factory=tuple)


def parse_params(text):
    params = {}
    for item in filter(None, text.split(",")):
        key, sep, value = item.partition("=")
        if not sep or not key.isidentifier():
            raise ValueError(f"Invalid format parameter {item!r}: use key=value")
        try:
            params[key] = int(value)
        except ValueError:
            try:
                params[key] = float(value)
            except ValueError:
                raise ValueError(f"Format parameter {key} needs a number, not {value!r}") from None
    return params


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
    if fp8:  # FP8 latent + FP32 scale per 128 elements + BF16 rope, as one byte row
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


def fuse(tensors, k_dims, d, pow2):
    """K, K_qp, V and V_qp packed into one byte row per token (tensor KV); the HBM residual stays apart."""
    body = [spec for spec in tensors if spec.tier == "hbf"]
    row = sum(spec.token_bytes for spec in body)
    size = 1 << (row - 1).bit_length() if pow2 else row
    fused = TensorSpec("KV", 8, size, body[0].window, attn_dims=k_dims + d, logical_dims=k_dims + d,
                       dequant_ops=INT_DEQUANT_OPS)
    note = (f"量化参数与数据同页：K、V 与两组 scale/zero 合成一行 {row} B"
            + (f"，补齐到 {size} B（2 的幂，2 的幂选择块对齐页边界）" if size != row else "（行不整除页时页尾留空）"))
    return (fused,) + tuple(spec for spec in tensors if spec.tier != "hbf"), note


def parse_format(text, head_dim, q_heads=None):
    """``kivi2:group=32,residual=128`` -> KVFormat for ``head_dim`` channels per head.

    ``q_heads`` (query heads per stored KV head, i.e. the model's q_heads / kv_heads; all query
    heads for MLA) only matters for the MLA indexer, whose FLOPs the system model multiplies by it.
    """
    name, _, rest = text.partition(":")
    if name not in PRESETS:
        raise ValueError(f"Unknown KV format {name!r}; presets: {', '.join(PRESETS)}")
    given = parse_params(rest)
    allowed = set(DEFAULTS.get(name, {})) | ({"think", "fused", "pow2"} if name in INT else set())
    if set(given) - allowed:
        raise ValueError(f"Format {name} takes {sorted(allowed) or 'no parameters'}, not {sorted(set(given) - allowed)}")
    params = DEFAULTS.get(name, {}) | given
    if name in ("mla", "mla_fp8"):
        tensors, attn_dims, notes = mla_format(name, params, q_heads)
        return KVFormat(name, params, tensors, attn_dims, tuple(notes))
    d = head_dim
    ratio = params.get("ratio", params.get("think", 0))
    if not 0 <= ratio < 1:
        raise ValueError("A pruning ratio must be in [0, 1)")
    k_dims = round(d * (1 - ratio))
    notes = []
    if name == "bf16":
        tensors = (TensorSpec("K", 16, d), TensorSpec("V", 16, d))
    elif name == "fp8":
        tensors = (TensorSpec("K", 8, d, dequant_ops=FP8_DEQUANT_OPS), TensorSpec("V", 8, d, dequant_ops=FP8_DEQUANT_OPS))
        notes.append("FP8：每个张量一个静态 scale，不占存储；反量化每元素 1 次运算（假设）")
    elif name in ("int4", "int2", "kivi4", "kivi2"):
        bits = int(name[-1])
        group = params["group"]
        if type(group) is not int or group <= 0 or (2 * k_dims) % group or (2 * d) % group:
            raise ValueError(f"group must divide 2 x the channels ({2 * k_dims} and {2 * d})")
        window = None
        tensors = []
        if name.startswith("kivi"):
            residual = params["residual"]
            if type(residual) is not int or residual < 0:
                raise ValueError("residual must be a nonnegative number of tokens")
            window = (0, -residual) if residual else None
            notes.append(f"KIVI：K 按通道、V 按 token 分组量化（G = {group}），最近 R = {residual} 个 token "
                         "保持 16 bit、放在 HBM（每步都变，HBF 不能原地改写）；R 取固定值，不随上下文取模")
        else:
            notes.append(f"按通道分组量化（每 {group} 个通道一对 scale、zero）")
        notes.append("量化参数（16 bit scale 与 zero）按每 token 一行均摊存放（假设）；反量化每元素 3 次运算（假设）")
        tensors += [TensorSpec("K", bits, k_dims, window, dequant_ops=INT_DEQUANT_OPS),
                    TensorSpec("K_qp", 16, 2 * k_dims // group, window, role="aux"),
                    TensorSpec("V", bits, d, window, dequant_ops=INT_DEQUANT_OPS),
                    TensorSpec("V_qp", 16, 2 * d // group, window, role="aux")]
        if window is not None:
            tensors += [TensorSpec("K_res", 16, k_dims, (-params["residual"], None), "hbm"),
                        TensorSpec("V_res", 16, d, (-params["residual"], None), "hbm")]
        tensors = tuple(tensors)
        fused, pow2 = params.get("fused", 0), params.get("pow2", 0)
        if fused not in (0, 1) or pow2 not in (0, 1) or (pow2 and not fused):
            raise ValueError("fused and pow2 take 0 or 1, and pow2 needs fused=1")
        if fused:
            tensors, note = fuse(tensors, k_dims, d, pow2)
            notes.append(note)
    elif name == "think":
        tensors = (TensorSpec("K", 16, k_dims), TensorSpec("V", 16, d))
        notes.append(f"ThinK：K 只保留 {k_dims}/{d} 个通道")
    else:  # quest
        page = params["page"]
        if type(page) is not int or page <= 0 or (2 * d) % page:
            raise ValueError("page must divide 2 x head_dim")
        tensors = (TensorSpec("K", 16, d), TensorSpec("V", 16, d), TensorSpec("K_sum", 16, 2 * d // page, role="meta"))
        notes.append(f"Quest：每 {page} 个 token 的页摘要（逐元素 min、max 键）每步整体读取，按每 token 一行均摊；"
                     "选择打分的计算未计入")
    if ratio and name != "think":
        notes.append(f"ThinK 裁剪：K 保留 {k_dims}/{d} 个通道")
    return KVFormat(name, params, tensors, k_dims + d, tuple(notes))
