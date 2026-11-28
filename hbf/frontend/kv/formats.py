"""Core KV storage formats (``--format NAME[:key=value,...]``) expanded into tensor specs.

Core built-in presets registered into ``hbf.registry``:
- ``bf16``: K, V 16 x d
- ``fp8``: K, V 8 x d (static per-tensor scale)
- ``int4`` / ``int2`` (``group``, ``think``, ``fused``, ``pow2``): group quantization
- ``kivi4`` / ``kivi2`` (``group``, ``residual``, ``think``, ``fused``, ``pow2``): KIVI (ICML 2024)
- ``think`` (``ratio``): ThinK (ICLR 2025)
- ``quest`` (``page``): Quest (ICML 2024)

Optional formats (``mla``, ``mla_fp8``, ``loki``, ``sparq``) register into the exact same
``hbf.registry.REGISTRY.formats`` table from ``hbf.ext.kv_terms``.
"""
from dataclasses import dataclass, field

from .tensors import TensorSpec

INT_DEQUANT_OPS = 3  # assumption: unpack, multiply by scale, add zero point
FP8_DEQUANT_OPS = 1  # assumption: one conversion per element
CORE_PRESETS = ("bf16", "fp8", "int4", "int2", "kivi4", "kivi2", "think", "quest")
CORE_DEFAULTS = {
    "int4": {"group": 128},
    "int2": {"group": 128},
    "kivi4": {"group": 32, "residual": 128},
    "kivi2": {"group": 32, "residual": 128},
    "think": {"ratio": .5},
    "quest": {"page": 16},
}
INT = ("int4", "int2", "kivi4", "kivi2")


@dataclass(frozen=True)
class KVFormat:
    name: str
    params: dict
    tensors: tuple
    attn_dims: int  # d_k + d_v: elements per token one query head multiplies in QK^T and PV
    notes: tuple = field(default_factory=tuple)
    extra: dict = field(default_factory=dict)

    @property
    def full_tensors(self):
        return tuple(self.extra.get("full_tensors", ()))

    @property
    def channel_groups(self):
        return tuple(self.extra.get("channel_groups", ()))

    @property
    def channel_select_channels(self):
        return self.extra.get("channel_select_channels")


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


def _build_core_format(name, params, head_dim, q_heads=None):
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


def register_builtins(registry):
    for name in CORE_PRESETS:
        extra = ("think", "fused", "pow2") if name in INT else ()
        registry.register_format(name, _build_core_format, defaults=CORE_DEFAULTS.get(name), extra_params=extra)


def parse_format(text, head_dim, q_heads=None):
    """``kivi2:group=32,residual=128`` -> KVFormat for ``head_dim`` channels per head.

    Looks up ``name`` in ``hbf.registry.REGISTRY.formats`` (covering both core built-in formats
    and any loaded ``hbf.ext.*`` formats).
    """
    from ...registry import REGISTRY
    name, _, rest = text.partition(":")
    builder, (defaults, extra_allowed) = REGISTRY.get_format(name)
    given = parse_params(rest)
    allowed = set(defaults) | set(extra_allowed)
    if set(given) - allowed:
        raise ValueError(f"Format {name} takes {sorted(allowed) or 'no parameters'}, not {sorted(set(given) - allowed)}")
    params = defaults | given
    return builder(name, params, head_dim, q_heads)
