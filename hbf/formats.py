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

``think=r`` also applies to the int and KIVI presets (K keeps (1-r)d channels). KIVI's per-channel
K scales are one row per G tokens; they are amortized as per-token rows of 2d/G elements, as are
Quest's per-page summaries. Dequantization costs ``dequant_ops`` GPU vector operations per stored
element read: 1 for FP8 and 3 (unpack, scale, shift) for integer formats, both assumptions.
"""
from dataclasses import dataclass, field

from .access import TensorSpec

INT_DEQUANT_OPS = 3  # assumption: unpack, multiply by scale, add zero point
FP8_DEQUANT_OPS = 1  # assumption: one conversion per element
PRESETS = ("bf16", "fp8", "int4", "int2", "kivi4", "kivi2", "think", "quest")
DEFAULTS = {"int4": {"group": 128}, "int2": {"group": 128}, "kivi4": {"group": 32, "residual": 128},
            "kivi2": {"group": 32, "residual": 128}, "think": {"ratio": .5}, "quest": {"page": 16}}


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


def parse_format(text, head_dim):
    """``kivi2:group=32,residual=128`` -> KVFormat for ``head_dim`` channels per head."""
    name, _, rest = text.partition(":")
    if name not in PRESETS:
        raise ValueError(f"Unknown KV format {name!r}; presets: {', '.join(PRESETS)}")
    given = parse_params(rest)
    allowed = set(DEFAULTS.get(name, {})) | ({"think"} if name in ("int4", "int2", "kivi4", "kivi2") else set())
    if set(given) - allowed:
        raise ValueError(f"Format {name} takes {sorted(allowed) or 'no parameters'}, not {sorted(set(given) - allowed)}")
    params = DEFAULTS.get(name, {}) | given
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
                    TensorSpec("K_qp", 16, 2 * k_dims // group, window),
                    TensorSpec("V", bits, d, window, dequant_ops=INT_DEQUANT_OPS),
                    TensorSpec("V_qp", 16, 2 * d // group, window)]
        if window is not None:
            tensors += [TensorSpec("K_res", 16, k_dims, (-params["residual"], None), "hbm"),
                        TensorSpec("V_res", 16, d, (-params["residual"], None), "hbm")]
        tensors = tuple(tensors)
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
