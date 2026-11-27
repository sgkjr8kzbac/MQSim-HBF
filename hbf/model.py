"""Transformer and GPU specifications for the decode compute model (hbf/system.py)."""
import json
import math
from dataclasses import dataclass, fields
from pathlib import Path

CONFIGS = Path(__file__).resolve().parent / "configs"
MODELS = CONFIGS / "models.json"
GPUS = CONFIGS / "gpus.json"


def _positive_int(value):
    return type(value) is int and value > 0


def _positive_number(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


@dataclass(frozen=True)
class ModelSpec:
    """Dense decoder-only transformer with grouped-query attention.

    Weight counts cover the attention projections and the MLP; norms, biases and Q/K norms are
    ignored (well under 0.1 % of the weights of the presets).
    """
    name: str
    layers: int
    hidden: int
    q_heads: int
    kv_heads: int
    head_dim: int
    intermediate: int
    vocab: int
    weight_bits: int = 16
    gated_mlp: bool = True
    tied_embeddings: bool = False
    source: str = ""

    def __post_init__(self):
        dims = (self.layers, self.hidden, self.q_heads, self.kv_heads, self.head_dim, self.intermediate,
                self.vocab, self.weight_bits)
        if (not isinstance(self.name, str) or not self.name or not all(map(_positive_int, dims))
                or self.weight_bits > 64 or self.q_heads % self.kv_heads
                or type(self.gated_mlp) is not bool or type(self.tied_embeddings) is not bool
                or not isinstance(self.source, str)):
            raise ValueError(f"Invalid model spec {self}: need a name, positive integer dimensions, "
                             "weight_bits <= 64, q_heads a multiple of kv_heads and boolean flags")

    @property
    def pre_params(self):
        """Q, K and V projections of one layer."""
        return self.hidden * (self.q_heads + 2 * self.kv_heads) * self.head_dim

    @property
    def post_params(self):
        """Output projection and MLP of one layer."""
        mlp = (3 if self.gated_mlp else 2) * self.hidden * self.intermediate
        return self.q_heads * self.head_dim * self.hidden + mlp

    @property
    def layer_params(self):
        return self.pre_params + self.post_params

    @property
    def embedding_params(self):
        """Input embedding plus LM head (one matrix when tied)."""
        return self.vocab * self.hidden * (1 if self.tied_embeddings else 2)

    @property
    def weight_bytes(self):
        """Bytes per weight (may be fractional below 8 bits)."""
        return self.weight_bits / 8

    @property
    def weights_bytes(self):
        """HBM footprint of all weights."""
        return (self.layers * self.layer_params + self.embedding_params) * self.weight_bits // 8


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


def presets(path):
    with open(path) as stream:
        return json.load(stream)


def _load(cls, value, path):
    """Build cls from an instance, a preset name, JSON text or a JSON file.

    A JSON object may name a ``"preset"`` and override some of its fields, e.g.
    ``{"preset": "h100-sxm", "memory_eff": 1.0}``.
    """
    if isinstance(value, cls):
        return value
    table = presets(path)
    if isinstance(value, str):
        if value in table:
            return cls(name=value, **table[value])
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            if not Path(value).is_file():
                raise ValueError(f"Unknown {cls.__name__} {value!r}: presets are {', '.join(table)}, "
                                 "or give JSON text or a JSON file") from None
            with open(value) as stream:
                value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{cls.__name__} must be a preset name or a JSON object")
    value = dict(value)
    preset = value.pop("preset", None)
    if preset is not None:
        if preset not in table:
            raise ValueError(f"Unknown {cls.__name__} preset {preset!r}: presets are {', '.join(table)}")
        value = {"name": preset, **table[preset]} | value
    unknown = set(value) - {field.name for field in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {', '.join(sorted(unknown))}")
    return cls(**value)


def load_model(name_or_json):
    return _load(ModelSpec, name_or_json, MODELS)


def load_gpu(name_or_json):
    return _load(GpuSpec, name_or_json, GPUS)
