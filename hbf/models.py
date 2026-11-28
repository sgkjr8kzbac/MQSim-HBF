"""Transformer and GPU specifications for the decode compute model (hbf/system.py)."""
import json
import math
from dataclasses import dataclass, fields
from pathlib import Path

from .device.config import CONFIG_DIR as CONFIGS, MODELS_JSON as MODELS


def _positive_int(value):
    return type(value) is int and value > 0


def _nonneg_int(value):
    return type(value) is int and value >= 0


def _positive_number(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


ATTENTIONS = ("gqa", "mla")


def experts_touched(experts, top_k, batch):
    """Expected distinct routed experts touched by a decode batch of size ``batch``: E(1 - (1 - k/E)^B)."""
    if not experts:
        return 0.0
    if not _positive_int(experts) or not _positive_int(top_k) or top_k > experts or not _positive_int(batch):
        raise ValueError(f"Invalid MoE routing parameters: experts={experts}, top_k={top_k}, batch={batch}")
    return experts * (1.0 - (1.0 - top_k / experts) ** batch)


@dataclass(frozen=True)
class ModelSpec:
    """Decoder-only transformer specification (GQA, MLA, hybrid/SSM layers, dense MLP or MoE).

    Weight counts cover the attention/SSM projections, MLP/MoE weights (including MoE routers) and
    embeddings; norms, biases, short conv kernels and Q/K norms are ignored (well under 0.2 % of
    the weights of the presets).
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
    attention: str = "gqa"
    kv_lora_rank: int = 0
    rope_dim: int = 0
    q_lora_rank: int = 0
    v_head_dim: int = 0
    gated_attention: bool = False
    layer_pattern: str = ""
    window: int = 0
    ssm_state_bytes: int = 0
    ssm_pre_params: int = 0
    ssm_post_params: int = 0
    experts: int = 0
    top_k: int = 0
    shared_experts: int = 0
    expert_intermediate: int = 0
    dense_layers: int = 0

    def __post_init__(self):
        dims = (self.layers, self.hidden, self.q_heads, self.kv_heads, self.head_dim, self.intermediate,
                self.vocab, self.weight_bits)
        nonneg = (self.kv_lora_rank, self.rope_dim, self.q_lora_rank, self.v_head_dim, self.window,
                  self.ssm_state_bytes, self.ssm_pre_params, self.ssm_post_params, self.experts,
                  self.top_k, self.shared_experts, self.expert_intermediate, self.dense_layers)
        if (not isinstance(self.name, str) or not self.name or not all(map(_positive_int, dims))
                or not all(map(_nonneg_int, nonneg)) or self.weight_bits > 64 or self.q_heads % self.kv_heads
                or type(self.gated_mlp) is not bool or type(self.tied_embeddings) is not bool
                or type(self.gated_attention) is not bool or not isinstance(self.source, str)
                or self.attention not in ATTENTIONS or not isinstance(self.layer_pattern, str)):
            raise ValueError(f"Invalid model spec {self}: need a name, positive integer dimensions, "
                             "weight_bits <= 64, q_heads a multiple of kv_heads and valid flags")
        if self.attention == "mla":
            if self.kv_heads != 1 or not self.kv_lora_rank or not self.rope_dim:
                raise ValueError(f"MLA model {self.name} requires kv_heads=1 and positive kv_lora_rank and rope_dim")
        elif any((self.kv_lora_rank, self.rope_dim, self.q_lora_rank, self.v_head_dim)):
            raise ValueError(f"GQA model {self.name} cannot set MLA dimensions")
        if self.layer_pattern:
            letters = set(self.layer_pattern)
            if not (letters <= {"L", "G"} or letters <= {"S", "A"}):
                raise ValueError(f"layer_pattern {self.layer_pattern!r} must use L/G (hybrid) or S/A (SSM)")
            if "L" in letters and not self.window:
                raise ValueError(f"Hybrid model {self.name} with L layers requires positive window")
            if "S" in letters and not self.ssm_state_bytes:
                raise ValueError(f"SSM model {self.name} with S layers requires positive ssm_state_bytes")
            if "L" not in letters and self.window:
                raise ValueError(f"Model {self.name} sets window without L layers in layer_pattern")
            if "S" not in letters and any((self.ssm_state_bytes, self.ssm_pre_params, self.ssm_post_params)):
                raise ValueError(f"Model {self.name} sets SSM fields without S layers in layer_pattern")
        elif any((self.window, self.ssm_state_bytes, self.ssm_pre_params, self.ssm_post_params)):
            raise ValueError(f"Model {self.name} sets hybrid/SSM fields without layer_pattern")
        if self.experts:
            if not 1 <= self.top_k <= self.experts or not self.expert_intermediate or self.dense_layers >= self.layers:
                raise ValueError(f"MoE model {self.name} requires 1 <= top_k <= experts, positive "
                                 "expert_intermediate and dense_layers < layers")
        elif any((self.top_k, self.shared_experts, self.expert_intermediate, self.dense_layers)):
            raise ValueError(f"Dense model {self.name} cannot set MoE parameters")

    def layer_kind(self, layer):
        """Layer type letter from ``layer_pattern`` ('G' when uniform full attention)."""
        if not 0 <= layer < self.layers:
            raise ValueError(f"Layer {layer} outside [0, {self.layers})")
        return self.layer_pattern[layer % len(self.layer_pattern)] if self.layer_pattern else "G"

    def is_moe_layer(self, layer):
        return bool(self.experts and layer >= self.dense_layers)

    @property
    def default_attn_dims(self):
        """Elements one query head multiplies per attended token when the row does not specify them."""
        if self.attention == "mla":
            return 2 * self.kv_lora_rank + self.rope_dim
        return 2 * self.head_dim

    def experts_touched(self, batch):
        """Expected distinct routed experts touched by a decode batch of size ``batch``."""
        return experts_touched(self.experts, self.top_k, batch)

    def layer_pre_params(self, layer):
        """Pre-attention linear projection parameters of ``layer`` (QKV, absorbed MLA, or SSM input)."""
        if self.layer_kind(layer) == "S" and self.ssm_pre_params:
            return self.ssm_pre_params
        if self.attention == "mla":
            q_dim = self.q_heads * (self.head_dim + self.rope_dim)
            q_proj = (self.hidden * self.q_lora_rank + self.q_lora_rank * q_dim
                      if self.q_lora_rank else self.hidden * q_dim)
            kv_down = self.hidden * (self.kv_lora_rank + self.rope_dim)
            k_up = self.q_heads * self.head_dim * self.kv_lora_rank  # absorbed W_UK
            return q_proj + kv_down + k_up
        q_mult = 2 if self.gated_attention else 1
        return self.hidden * (q_mult * self.q_heads + 2 * self.kv_heads) * self.head_dim

    def _layer_o_params(self, layer):
        """Output projection parameters of ``layer`` (O, absorbed MLA W_UV + O, or SSM output)."""
        if self.layer_kind(layer) == "S" and self.ssm_post_params:
            return self.ssm_post_params
        if self.attention == "mla":
            v_dim = self.v_head_dim or self.head_dim
            return self.q_heads * self.kv_lora_rank * v_dim + self.q_heads * v_dim * self.hidden
        return self.q_heads * self.head_dim * self.hidden

    def _layer_mlp_params(self, layer, experts_count):
        factor = 3 if self.gated_mlp else 2
        if not self.is_moe_layer(layer):
            return factor * self.hidden * self.intermediate
        return self.hidden * self.experts + experts_count * factor * self.hidden * self.expert_intermediate

    def layer_post_params(self, layer):
        """Total stored parameters of output projection and MLP/MoE in ``layer``."""
        return self._layer_o_params(layer) + self._layer_mlp_params(layer, self.experts + self.shared_experts)

    def layer_post_active_params(self, layer):
        """Active (per-token FLOP) parameters of output projection and MLP/MoE in ``layer``."""
        return self._layer_o_params(layer) + self._layer_mlp_params(layer, self.top_k + self.shared_experts)

    def layer_post_touched_params(self, layer, batch):
        """Expected parameters read from HBM in output projection and MLP/MoE of ``layer`` at batch ``batch``."""
        if not self.is_moe_layer(layer):
            return self.layer_post_params(layer)
        return self._layer_o_params(layer) + self._layer_mlp_params(
            layer, self.experts_touched(batch) + self.shared_experts)

    @property
    def pre_params(self):
        """Pre-attention projections of layer 0 (identical across layers for uniform models)."""
        return self.layer_pre_params(0)

    @property
    def post_params(self):
        """Output projection and MLP/MoE of layer 0 (identical across layers for uniform dense models)."""
        return self.layer_post_params(0)

    @property
    def layer_params(self):
        return self.pre_params + self.post_params

    @property
    def embedding_params(self):
        """Input embedding plus LM head (one matrix when tied)."""
        return self.vocab * self.hidden * (1 if self.tied_embeddings else 2)

    @property
    def total_params(self):
        """Total stored parameters across all layers plus embeddings."""
        return (sum(self.layer_pre_params(l) + self.layer_post_params(l) for l in range(self.layers))
                + self.embedding_params)

    @property
    def active_params(self):
        """Parameters active per decoded token (all layers' active projections + LM head)."""
        return (sum(self.layer_pre_params(l) + self.layer_post_active_params(l) for l in range(self.layers))
                + self.vocab * self.hidden)

    @property
    def weight_bytes(self):
        """Bytes per weight (may be fractional below 8 bits)."""
        return self.weight_bits / 8

    @property
    def weights_bytes(self):
        """HBM footprint of all weights."""
        return self.total_params * self.weight_bits // 8


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
