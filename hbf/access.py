"""Logical access schema, replaceable layouts and seeded workloads."""
import csv
from dataclasses import dataclass, fields
from math import ceil
from random import Random

from .config import Config


@dataclass(frozen=True)
class Shape:
    layers: int
    kv_heads: int
    head_dim: int
    context: int

    def __post_init__(self):
        if any(type(n) is not int or n <= 0 for n in vars(self).values()):
            raise ValueError("Model dimensions and context must be positive integers")


@dataclass(frozen=True)
class Access:
    step: int
    layer: int
    kv_head: int
    kind: str
    op: str
    token_start: int
    token_end: int
    bits: int = 16

    def validate(self, shape):
        numbers = (self.step, self.layer, self.kv_head, self.token_start, self.token_end, self.bits)
        if any(type(n) is not int for n in numbers) or not (
            self.step >= -1 and 0 <= self.layer < shape.layers and
            0 <= self.kv_head < shape.kv_heads and self.kind in {"K", "V"} and
            self.op in {"R", "W"} and 0 <= self.token_start < self.token_end <= shape.context
            and self.bits == 16
        ):
            raise ValueError(f"Invalid logical access: {self}")


def write_access(path, accesses):
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([f.name for f in fields(Access)])
        writer.writerows(vars(access).values() for access in accesses)


def read_access(path):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        names = [field.name for field in fields(Access)]
        if reader.fieldnames != names:
            raise ValueError("Invalid logical access CSV header")
        for row in reader:
            if None in row or None in row.values():
                raise ValueError("Invalid logical access CSV row")
            yield Access(**{key: val if key in {"op", "kind"} else int(val) for key, val in row.items()})


class Layout:
    def __init__(self, cfg: Config, shape: Shape, base=0):
        self.shape, self.base = shape, base
        self.token_bytes = shape.head_dim * 2
        self.tokens = cfg.page_bytes // self.token_bytes
        if base < 0 or self.tokens == 0:
            raise ValueError("A BF16 token must fit in one page and base must be nonnegative")
        self.blocks = (shape.context + self.tokens - 1) // self.tokens
        cfg.lba(base + self.blocks * shape.layers * shape.kv_heads * 2 - 1)

    def lpa(self, layer, head, kind, block):
        raise NotImplementedError

    def decode(self, lpa):
        raise NotImplementedError

    def page_access(self, lpa, step=0, op="R"):
        layer, head, kind, block = self.decode(lpa)
        lo = block * self.tokens
        return Access(step, layer, head, "KV"[kind], op, lo, min(lo + self.tokens, self.shape.context))


class LayerContiguous(Layout):
    def lpa(self, layer, head, kind, block):
        return self.base + ((layer * 2 + kind) * self.shape.kv_heads + head) * self.blocks + block

    def decode(self, lpa):
        group, block = divmod(lpa - self.base, self.blocks)
        group, head = divmod(group, self.shape.kv_heads)
        layer, kind = divmod(group, 2)
        return layer, head, kind, block


class VllmInterleaved(Layout):
    def lpa(self, layer, head, kind, block):
        return self.base + (((block * self.shape.layers + layer) * self.shape.kv_heads + head) * 2 + kind)

    def decode(self, lpa):
        group, kind = divmod(lpa - self.base, 2)
        group, head = divmod(group, self.shape.kv_heads)
        block, layer = divmod(group, self.shape.layers)
        return layer, head, kind, block


LAYOUTS = {"layer_contiguous": LayerContiguous, "vllm_interleaved": VllmInterleaved}
MODES = (
    "dense", "uniform_pages", "same_unit", "distinct_units", "same_channel_distinct_units",
    "hot_unit", "write_then_read",
)


def generate(cfg, layout, mode, *, steps=1, seed=1, fraction=.1, count=4, unit=0,
             hot_fraction=.5, hot_units=1):
    shape = layout.shape
    if (mode not in MODES or steps <= 0 or not 0 < fraction <= 1 or count <= 0
            or not 0 <= hot_fraction <= 1 or not 1 <= hot_units <= cfg.units
            or not 0 <= unit < cfg.units):
        raise ValueError("Invalid generator parameters")
    rng = Random(seed)
    if mode in {"dense", "uniform_pages"}:
        for step in range(steps):
            for layer in range(shape.layers):
                for head in range(shape.kv_heads):
                    for kind in "KV":
                        if mode == "dense":
                            yield Access(step, layer, head, kind, "R", 0, shape.context)
                        else:
                            blocks = rng.sample(range(layout.blocks), ceil(layout.blocks * fraction))
                            for block in sorted(blocks):
                                pa = layout.lpa(layer, head, "KV".index(kind), block)
                                yield layout.page_access(pa, step)
        return

    if mode == "same_unit":
        targets = [unit] * count
    elif mode == "distinct_units":
        if count > cfg.units:
            raise ValueError("Too many pages for distinct units")
        targets = list(range(count))
    elif mode == "same_channel_distinct_units":
        if count > cfg.units // cfg.channels:
            raise ValueError("Too many pages for one channel")
        targets = [unit % cfg.channels + n * cfg.channels for n in range(count)]
    else:
        targets = [n % cfg.units for n in range(count)]
        rng.shuffle(targets)
        if mode == "hot_unit":
            hot = [(unit + n) % cfg.units for n in range(hot_units)]
            for n in range(round(count * hot_fraction)):
                targets[n] = rng.choice(hot)

    used = {}
    addresses = []
    for target in targets:
        k = used.get(target, 0)
        used[target] = k + 1
        pa = cfg.unit_lpa(target, k, layout.base)
        access = layout.page_access(pa)
        access.validate(shape)
        addresses.append(pa)
    if mode == "write_then_read":
        for pa in addresses:
            yield layout.page_access(pa, -1, "W")
    for step in range(steps):
        for pa in addresses:
            yield layout.page_access(pa, step)
