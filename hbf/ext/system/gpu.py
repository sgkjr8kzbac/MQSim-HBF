"""GPU roofline specifications, preset loader, and GPU prefill rate (hbf.ext.system)."""
import math
from dataclasses import dataclass
from pathlib import Path

from ...models import _load, presets

GPUS = Path(__file__).resolve().parent / "gpus.json"


def _positive_number(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


@dataclass(frozen=True)
class GpuSpec:
    """Roofline parameters of one GPU. Rates are peak values; the efficiencies derate them."""
    name: str
    tensor_flops: float  # dense BF16 tensor-core FLOP/s
    vector_ops: float  # non-tensor op/s used for dequantization
    hbm_bytes_s: float
    hbm_capacity: int  # bytes
    compute_eff: float = .5
    memory_eff: float = .8
    layer_overhead_ns: float = 0
    source: str = ""

    def __post_init__(self):
        rates = (self.tensor_flops, self.vector_ops, self.hbm_bytes_s, self.hbm_capacity)
        if (not isinstance(self.name, str) or not self.name or not all(map(_positive_number, rates))
                or not all(_positive_number(e) and e <= 1 for e in (self.compute_eff, self.memory_eff))
                or type(self.layer_overhead_ns) not in (int, float) or not math.isfinite(self.layer_overhead_ns)
                or self.layer_overhead_ns < 0 or not isinstance(self.source, str)):
            raise ValueError(f"Invalid GPU spec {self}: need a name, positive finite rates and capacity, "
                             "efficiencies in (0, 1] and a non-negative layer overhead")

    @property
    def flops(self):
        """Effective tensor FLOP/s."""
        return self.tensor_flops * self.compute_eff

    @property
    def vector(self):
        """Effective vector op/s (dequantization)."""
        return self.vector_ops * self.compute_eff

    @property
    def bandwidth(self):
        """Effective HBM bytes/s."""
        return self.hbm_bytes_s * self.memory_eff


def load_gpu(name_or_json):
    return _load(GpuSpec, name_or_json, GPUS)


def prefill_tok_s(model, gpu, tokens):
    """Analytic prefill rate: FLOP/token = 2P + attn_flops(S) at the GPU's compute_eff (params.md section 6)."""
    s = max(tokens, 1)
    if not model.experts and not model.layer_pattern and model.attention == "gqa":
        params = model.layers * model.layer_params + model.embedding_params
        return gpu.flops / (2 * params + 2 * model.layers * model.q_heads * model.head_dim * s)
    params = model.active_params if model.experts else model.total_params
    attn_per_tok = 0
    for layer in range(model.layers):
        kind = model.layer_kind(layer)
        if kind == "S":
            continue
        span = min(s, model.window) if kind == "L" else s
        attn_per_tok += model.q_heads * model.default_attn_dims * span
    return gpu.flops / (2 * params + attn_per_tok)
