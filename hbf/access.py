"""Logical access schema, tensor formats, replaceable layouts and seeded workloads."""
import csv
import json
from bisect import bisect_right
from dataclasses import asdict, dataclass, fields
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
class TensorSpec:
    """Stored format of one per-head KV tensor: one row of ``dims`` elements of ``bits`` per token.

    The defaults are BF16 K and V. Compressed methods declare their own tensors, e.g. 2-bit
    quantized K plus a 16-bit scale tensor (KIVI), K with pruned channels (ThinK) or a low-rank
    latent. Pages hold tokens of a single tensor. Per-block metadata such as Quest's page
    summaries has one row per block rather than per token and is not expressible yet.
    """
    name: str
    bits: int
    dims: int

    def __post_init__(self):
        if (not isinstance(self.name, str) or not self.name.isidentifier()
                or any(type(n) is not int or n <= 0 for n in (self.bits, self.dims))
                or self.bits > 64 or self.dims * self.bits % 8):
            raise ValueError(f"Invalid tensor spec {self}: need an identifier name, "
                             "positive bits <= 64 and whole bytes per token")

    @property
    def token_bytes(self):
        return self.dims * self.bits // 8


def default_tensors(shape):
    return TensorSpec("K", 16, shape.head_dim), TensorSpec("V", 16, shape.head_dim)


def parse_tensors(text):
    """Parse ``[{"name": "K", "bits": 2, "dims": 128}, ...]`` from JSON text or a file path."""
    if text is None:
        return None
    try:
        items = json.loads(text)
    except json.JSONDecodeError:
        with open(text) as stream:
            items = json.load(stream)
    if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
        raise ValueError("Tensor specs must be a JSON list of objects")
    return tuple(TensorSpec(**item) for item in items)


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

    def validate(self, layout):
        """``kind`` names a tensor of the layout and ``bits`` must match its stored format."""
        shape = layout.shape
        numbers = (self.step, self.layer, self.kv_head, self.token_start, self.token_end, self.bits)
        spec = layout.specs.get(self.kind)
        if any(type(n) is not int for n in numbers) or spec is None or not (
            self.step >= -1 and 0 <= self.layer < shape.layers and
            0 <= self.kv_head < shape.kv_heads and
            self.op in {"R", "W"} and 0 <= self.token_start < self.token_end <= shape.context
            and self.bits == spec.bits
        ):
            raise ValueError(f"Invalid logical access: {self}")


def write_access(path, accesses):
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([f.name for f in fields(Access)])
        writer.writerows(vars(access).values() for access in accesses)


def tensors_meta(layout):
    return [asdict(spec) for spec in layout.tensors]


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
    """Maps (layer, tensor index, head, block) to LPAs. A block is one page of one tensor."""

    def __init__(self, cfg: Config, shape: Shape, base=0, tensors=None):
        self.shape, self.base = shape, base
        self.tensors = tuple(tensors) if tensors is not None else default_tensors(shape)
        names = [spec.name for spec in self.tensors]
        if not self.tensors or len(set(names)) != len(names):
            raise ValueError("Tensor names must be unique and nonempty")
        self.specs = {spec.name: spec for spec in self.tensors}
        self.index = {name: t for t, name in enumerate(names)}
        self.tokens_per_page = tuple(cfg.page_bytes // spec.token_bytes for spec in self.tensors)
        if base < 0 or 0 in self.tokens_per_page:
            raise ValueError("A token row must fit in one page and base must be nonnegative")
        self.block_counts = tuple(-(-shape.context // tokens) for tokens in self.tokens_per_page)
        self.pages = shape.layers * shape.kv_heads * sum(self.block_counts)
        self.span = self.pages  # LPAs [base, base + span) belong to the layout, padding included
        cfg.lba(base + self.pages - 1)

    def tensor_index(self, name):
        return self.index[name]

    def tokens(self, t):
        return self.tokens_per_page[t]

    def blocks(self, t):
        return self.block_counts[t]

    def token_bytes(self, t):
        return self.tensors[t].token_bytes

    def lpa(self, layer, head, t, block):
        raise NotImplementedError

    def decode(self, lpa):
        raise NotImplementedError

    def page_access(self, lpa, step=0, op="R"):
        layer, head, t, block = self.decode(lpa)
        lo = block * self.tokens(t)
        spec = self.tensors[t]
        return Access(step, layer, head, spec.name, op, lo, min(lo + self.tokens(t), self.shape.context),
                      spec.bits)


class LayerContiguous(Layout):
    """Regions ordered (layer, tensor, head), each holding that tensor's blocks contiguously.

    With the default K/V specs this is ((layer*2 + kind)*H + head)*blocks + block.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        heads = self.shape.kv_heads
        self.offsets, offset = [], 0
        for count in self.block_counts:
            self.offsets.append(offset)
            offset += heads * count
        self.layer_pages = offset

    def lpa(self, layer, head, t, block):
        if not 0 <= block < self.block_counts[t]:
            raise ValueError("Block outside the tensor's reserved region")
        return (self.base + layer * self.layer_pages + self.offsets[t]
                + head * self.block_counts[t] + block)

    def decode(self, lpa):
        layer, rest = divmod(lpa - self.base, self.layer_pages)
        t = max(i for i, offset in enumerate(self.offsets) if offset <= rest)
        head, block = divmod(rest - self.offsets[t], self.block_counts[t])
        return layer, head, t, block


class VllmInterleaved(Layout):
    """vLLM-style block table: one block index shared by all tensors, layers and heads."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if len(set(self.tokens_per_page)) != 1:
            raise ValueError("vllm_interleaved requires the same tokens per page for every tensor")

    def lpa(self, layer, head, t, block):
        if not 0 <= block < self.block_counts[t]:
            raise ValueError("Block outside the block table")
        count = len(self.tensors)
        return self.base + (((block * self.shape.layers + layer) * self.shape.kv_heads + head) * count + t)

    def decode(self, lpa):
        group, t = divmod(lpa - self.base, len(self.tensors))
        group, head = divmod(group, self.shape.kv_heads)
        block, layer = divmod(group, self.shape.layers)
        return layer, head, t, block


class StripedContiguous(Layout):
    """``layer_contiguous`` regions started at a chosen plane phase (SPLASH-style striping).

    Pages land on plane ``lpa mod units`` (CWDP), so a region's first block decides where all its
    blocks go. With 32K BF16 tokens a head holds 2,048 blocks, a multiple of 1,024 planes, and
    ``layer_contiguous`` puts block b of every head and tensor on the same plane. Here region
    (layer, t, head) starts on plane ``t*P/T + head*P/(T*H)`` (mod P): heads P/(2H) apart and V half
    a device away from K for the default two tensors, as in SPLASH. Regions keep their order and
    are padded by less than P pages each to reach their phase. The OCP spec forbids skipping
    addresses inside a block (section 5.4.1), so a prefill writes the padding pages too.
    """

    def __init__(self, cfg, shape, base=0, tensors=None):
        Layout.__init__(self, cfg, shape, base, tensors)
        units, count, heads = cfg.units, len(self.tensors), shape.kv_heads
        self.starts, self.regions, cursor = {}, [], 0
        for layer in range(shape.layers):
            for t in range(count):
                for head in range(heads):
                    phase = (t * units // count + head * units // (count * heads)) % units
                    start = cursor + (phase - (base + cursor)) % units
                    self.starts[layer, t, head] = start
                    self.regions.append((start, layer, t, head))
                    cursor = start + self.block_counts[t]
        self.span = cursor
        cfg.lba(base + self.span - 1)

    def lpa(self, layer, head, t, block):
        if not 0 <= block < self.block_counts[t]:
            raise ValueError("Block outside the tensor's reserved region")
        return self.base + self.starts[layer, t, head] + block

    def decode(self, lpa):
        offset = lpa - self.base
        index = bisect_right(self.regions, (offset, float("inf"))) - 1
        if index < 0 or not 0 <= offset - self.regions[index][0] < self.block_counts[self.regions[index][2]]:
            raise ValueError(f"LPA {lpa} is padding, not a page of the layout")
        start, layer, t, head = self.regions[index]
        return layer, head, t, offset - start


LAYOUTS = {"layer_contiguous": LayerContiguous, "vllm_interleaved": VllmInterleaved,
           "striped_contiguous": StripedContiguous}


MODES = (
    "dense", "uniform_pages", "sink_recent", "same_unit", "distinct_units", "same_channel_distinct_units",
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
    if mode == "sink_recent":
        # StreamingLLM-style: the first ``count`` tokens plus the last ``fraction`` of the context.
        recent = max(1, round(shape.context * fraction))
        for step in range(steps):
            for layer in range(shape.layers):
                for head in range(shape.kv_heads):
                    for spec in layout.tensors:
                        sink = min(count, shape.context - recent)
                        if sink > 0:
                            yield Access(step, layer, head, spec.name, "R", 0, sink, spec.bits)
                        yield Access(step, layer, head, spec.name, "R", shape.context - recent, shape.context,
                                     spec.bits)
        return
    if mode in {"dense", "uniform_pages"}:
        for step in range(steps):
            for layer in range(shape.layers):
                for head in range(shape.kv_heads):
                    for t, spec in enumerate(layout.tensors):
                        if mode == "dense":
                            yield Access(step, layer, head, spec.name, "R", 0, shape.context, spec.bits)
                        else:
                            blocks = layout.blocks(t)
                            for block in sorted(rng.sample(range(blocks), ceil(blocks * fraction))):
                                yield layout.page_access(layout.lpa(layer, head, t, block), step)
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
        access.validate(layout)
        addresses.append(pa)
    if mode == "write_then_read":
        for pa in addresses:
            yield layout.page_access(pa, -1, "W")
    for step in range(steps):
        for pa in addresses:
            yield layout.page_access(pa, step)
