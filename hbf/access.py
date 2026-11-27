"""Logical access schema, tensor formats, replaceable layouts and seeded workloads."""
import csv
import json
from bisect import bisect_right
from dataclasses import asdict, dataclass, fields
from math import ceil
from random import Random

from .config import Config


def numpy():
    """numpy, imported on first use so the reference (pure-Python) path does not need it."""
    import numpy as np
    return np


@dataclass(frozen=True)
class Shape:
    layers: int
    kv_heads: int
    head_dim: int
    context: int

    def __post_init__(self):
        if any(type(n) is not int or n <= 0 for n in vars(self).values()):
            raise ValueError("Model dimensions and context must be positive integers")


TIERS = ("hbf", "hbm")
ROLES = ("data", "aux", "meta")


@dataclass(frozen=True)
class TensorSpec:
    """Stored format of one per-head KV tensor: one row of ``dims`` elements of ``bits`` per token.

    The defaults are BF16 K and V. Compressed methods declare their own tensors, e.g. 2-bit
    quantized K plus a 16-bit scale tensor (KIVI), K with pruned channels (ThinK) or a low-rank
    latent. Pages hold tokens of a single tensor. Optional fields:

    * ``window`` - the tokens [lo, hi) the tensor holds; negative values count from the end of
      the context and None means the context's start or end. KIVI keeps the last R tokens in full
      precision: the quantized tensors take ``[0, -R]`` and the full-precision ones ``[-R, None]``.
    * ``tier`` - ``hbf`` (stored in HBF pages) or ``hbm`` (read from HBM, e.g. a residual window
      that changes every step and cannot be rewritten in place on HBF, section 5.4.1).
    * ``role`` - ``data`` (attended K or V elements), ``aux`` (per-token data that is read with the
      selected tokens but not attended, e.g. quantization scales) or ``meta`` (selection metadata
      such as Quest's page summaries, read in full every step by every generator mode and not
      attended). Per-block metadata is amortized as a per-token row (e.g. min and max keys of
      16-token blocks: 2 x 128 / 16 = 16 elements per token).
    * ``dequant_ops`` - GPU vector operations per stored element read (an assumption of the format).
    """
    name: str
    bits: int
    dims: int
    window: tuple | None = None
    tier: str = "hbf"
    role: str = "data"
    dequant_ops: float = 0

    def __post_init__(self):
        if (not isinstance(self.name, str) or not self.name.isidentifier()
                or any(type(n) is not int or n <= 0 for n in (self.bits, self.dims))
                or self.bits > 64 or self.dims * self.bits % 8):
            raise ValueError(f"Invalid tensor spec {self}: need an identifier name, "
                             "positive bits <= 64 and whole bytes per token")
        if self.window is not None:
            if (not isinstance(self.window, (list, tuple)) or len(self.window) != 2
                    or any(end is not None and type(end) is not int for end in self.window)):
                raise ValueError(f"Invalid window {self.window!r} of tensor {self.name}: need [lo, hi]")
            object.__setattr__(self, "window", tuple(self.window))
        if self.tier not in TIERS or self.role not in ROLES:
            raise ValueError(f"Tensor {self.name}: tier must be one of {TIERS} and role one of {ROLES}")
        if isinstance(self.dequant_ops, bool) or not isinstance(self.dequant_ops, (int, float)) or self.dequant_ops < 0:
            raise ValueError(f"Tensor {self.name}: dequant_ops must be a nonnegative number")

    @property
    def token_bytes(self):
        return self.dims * self.bits // 8

    def tokens(self, context):
        """The window as absolute token indices [lo, hi) of a ``context``-token sequence."""
        lo, hi = self.window or (None, None)
        lo = 0 if lo is None else lo + context if lo < 0 else lo
        hi = context if hi is None else hi + context if hi < 0 else hi
        if not 0 <= lo < hi <= context:
            raise ValueError(f"Window {self.window} of tensor {self.name} is empty or outside {context} tokens")
        return lo, hi


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
            self.op in {"R", "W"} and layout.windows[self.kind][0] <= self.token_start < self.token_end
            <= layout.windows[self.kind][1] and self.bits == spec.bits
        ):
            raise ValueError(f"Invalid logical access: {self}")


def write_access(path, accesses):
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([f.name for f in fields(Access)])
        writer.writerows(vars(access).values() for access in accesses)


def tensors_meta(layout):
    return [asdict(spec) for spec in layout.all_tensors]


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
    """Maps (layer, tensor index, head, block) to LPAs. A block is one page of one tensor.

    ``tensors`` are the HBF-tier tensors, the ones with pages; ``all_tensors`` adds the HBM tier.
    A tensor's block b holds the tokens ``window_lo + b * tokens_per_page`` onwards.
    """

    def __init__(self, cfg: Config, shape: Shape, base=0, tensors=None):
        self.shape, self.base = shape, base
        self.all_tensors = tuple(tensors) if tensors is not None else default_tensors(shape)
        names = [spec.name for spec in self.all_tensors]
        if not self.all_tensors or len(set(names)) != len(names):
            raise ValueError("Tensor names must be unique and nonempty")
        self.tensors = tuple(spec for spec in self.all_tensors if spec.tier == "hbf")
        self.hbm_tensors = tuple(spec for spec in self.all_tensors if spec.tier == "hbm")
        if not self.tensors:
            raise ValueError("At least one tensor must be stored on HBF")
        self.specs = {spec.name: spec for spec in self.all_tensors}
        self.windows = {spec.name: spec.tokens(shape.context) for spec in self.all_tensors}
        self.index = {spec.name: t for t, spec in enumerate(self.tensors)}
        self.tokens_per_page = tuple(cfg.page_bytes // spec.token_bytes for spec in self.tensors)
        if base < 0 or 0 in self.tokens_per_page:
            raise ValueError("A token row must fit in one page and base must be nonnegative")
        self.window_starts = tuple(self.windows[spec.name][0] for spec in self.tensors)
        self.block_counts = tuple(-(-(self.windows[spec.name][1] - self.windows[spec.name][0]) // tokens)
                                  for spec, tokens in zip(self.tensors, self.tokens_per_page))
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
        spec = self.tensors[t]
        lo = self.window_starts[t] + block * self.tokens(t)
        return Access(step, layer, head, spec.name, op, lo, min(lo + self.tokens(t), self.windows[spec.name][1]),
                      spec.bits)

    def token_blocks(self, t, lo, hi):
        """Blocks of HBF tensor ``t`` that hold tokens [lo, hi), and the first token of block 0."""
        start, tokens = self.window_starts[t], self.tokens(t)
        return range((lo - start) // tokens, (hi - 1 - start) // tokens + 1), start


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

    def lpa_array(self, layer, head, t, block):
        """``lpa`` for numpy arrays of heads, tensor indices and blocks (blocks assumed in range)."""
        np = numpy()
        counts = np.asarray(self.block_counts, dtype=np.int64)[t]
        return (self.base + layer * self.layer_pages + np.asarray(self.offsets, dtype=np.int64)[t]
                + head * counts + block)

    def decode(self, lpa):
        layer, rest = divmod(lpa - self.base, self.layer_pages)
        t = max(i for i, offset in enumerate(self.offsets) if offset <= rest)
        head, block = divmod(rest - self.offsets[t], self.block_counts[t])
        return layer, head, t, block


class VllmInterleaved(Layout):
    """vLLM-style block table: one block index shared by all tensors, layers and heads."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if len(set(self.tokens_per_page)) != 1 or len({self.windows[spec.name] for spec in self.tensors}) != 1:
            raise ValueError("vllm_interleaved requires the same tokens per page and window for every tensor")

    def lpa(self, layer, head, t, block):
        if not 0 <= block < self.block_counts[t]:
            raise ValueError("Block outside the block table")
        count = len(self.tensors)
        return self.base + (((block * self.shape.layers + layer) * self.shape.kv_heads + head) * count + t)

    def lpa_array(self, layer, head, t, block):
        return self.base + (((block * self.shape.layers + layer) * self.shape.kv_heads + head) * len(self.tensors) + t)

    def decode(self, lpa):
        group, t = divmod(lpa - self.base, len(self.tensors))
        group, head = divmod(group, self.shape.kv_heads)
        block, layer = divmod(group, self.shape.layers)
        return layer, head, t, block


class StripedContiguous(Layout):
    """``layer_contiguous`` regions separated by fixed gaps of padding (SPLASH-style striping).

    Pages land on plane ``lpa mod units`` (CWDP, a permutation of it under the ocp map), so a
    region's first block decides where all its blocks go. With 32K BF16 tokens a head holds 2,048
    blocks, a multiple of 1,024 planes, and ``layer_contiguous`` puts block b of every head and
    tensor on the same plane. Here every layer starts on phase 0 and each region after the first
    is preceded by a gap of ``P/(T*H)`` pages. For regions that are a multiple of P long this gives
    SPLASH's phases ``t*P/T + head*P/(T*H)``: heads P/(2H) apart and V half a device away from K
    for the default two tensors. For shorter or unequal regions (2-bit or KIVI tensors) the gap
    keeps neighbours apart without padding each region up to a fixed phase, so a layer costs
    less than P extra pages. The OCP spec forbids skipping addresses inside a block (section
    5.4.1), so a prefill writes the padding pages too.

    ``stagger`` makes the gap odd, so that consecutive regions also start on different host
    channels (``lpa mod 16``); with an even gap of 64 or 512 pages the block b of every region of a
    layer shares one host channel, which matters under host credit limits (``--mocs``).
    """
    stagger = False

    def __init__(self, cfg, shape, base=0, tensors=None):
        Layout.__init__(self, cfg, shape, base, tensors)
        units, count, heads = cfg.units, len(self.tensors), shape.kv_heads
        gap = max(units // (count * heads), 1)
        if self.stagger:
            gap |= 1
        self.gap = gap
        self.starts, self.regions, cursor = {}, [], 0
        for layer in range(shape.layers):
            cursor += -(base + cursor) % units  # each layer starts on phase 0
            for t in range(count):
                for head in range(heads):
                    start = cursor + (gap if (t, head) != (0, 0) else 0)
                    self.starts[layer, t, head] = start
                    self.regions.append((start, layer, t, head))
                    cursor = start + self.block_counts[t]
        self.span = cursor
        self._start_table = None
        cfg.lba(base + self.span - 1)

    def lpa(self, layer, head, t, block):
        if not 0 <= block < self.block_counts[t]:
            raise ValueError("Block outside the tensor's reserved region")
        return self.base + self.starts[layer, t, head] + block

    def lpa_array(self, layer, head, t, block):
        np = numpy()
        if self._start_table is None:
            table = np.zeros((self.shape.layers, len(self.tensors), self.shape.kv_heads), dtype=np.int64)
            for (row_layer, row_t, row_head), start in self.starts.items():
                table[row_layer, row_t, row_head] = start
            self._start_table = table
        return self.base + self._start_table[layer][t, head] + block

    def decode(self, lpa):
        offset = lpa - self.base
        index = bisect_right(self.regions, (offset, float("inf"))) - 1
        if index < 0 or not 0 <= offset - self.regions[index][0] < self.block_counts[self.regions[index][2]]:
            raise ValueError(f"LPA {lpa} is padding, not a page of the layout")
        start, layer, t, head = self.regions[index]
        return layer, head, t, offset - start


class StripedStaggered(StripedContiguous):
    """``striped_contiguous`` with an odd gap, so neighbouring regions start on different host channels."""
    stagger = True


LAYOUTS = {"layer_contiguous": LayerContiguous, "vllm_interleaved": VllmInterleaved,
           "striped_contiguous": StripedContiguous, "striped_staggered": StripedStaggered}


MODES = (
    "dense", "uniform_pages", "uniform_tokens", "sink_recent", "same_unit", "distinct_units", "same_channel_distinct_units",
    "hot_unit", "write_then_read",
)


def share(count, fraction):
    """ceil(count * fraction), ignoring float noise (100 * 0.07 is 7.000000000000001)."""
    return ceil(round(count * fraction, 9))


def generate(cfg, layout, mode, *, steps=1, seed=1, fraction=.1, count=4, unit=0,
             hot_fraction=.5, hot_units=1, select_tokens=16):
    """Seeded accesses of one workload mode.

    Every mode reads ``meta`` tensors (selection metadata) in full every (step, layer, head);
    selection modes also read HBM-tier tensors (hot windows) in full and clip the selected tokens
    to each tensor's window; with the default tensors this changes nothing. ``uniform_pages``
    samples each tensor's pages independently, so K and V (and scales) cover different tokens.

    ``uniform_pages`` picks ``fraction`` of each tensor's storage pages, independently per tensor, so
    the selection unit grows with compression (a 2-bit page holds 128 tokens, a BF16 page 16) and K
    and V pages of one head cover different tokens. ``uniform_tokens`` picks ``fraction`` of the
    blocks of ``select_tokens`` tokens per (step, layer, KV head), as Quest-style methods do, and
    reads every tensor of the head for the same tokens. Quest selects per query head; under GQA a
    KV head serves the union of its query heads' blocks, which this mode does not model (use a
    higher fraction).
    """
    shape = layout.shape
    if (mode not in MODES or steps <= 0 or not 0 < fraction <= 1 or count <= 0
            or not 0 <= hot_fraction <= 1 or not 1 <= hot_units <= cfg.units
            or not 0 <= unit < cfg.units or type(select_tokens) is not int or select_tokens <= 0):
        raise ValueError("Invalid generator parameters")
    rng = Random(seed)
    windows = layout.windows
    always = [spec for spec in layout.all_tensors if spec.role == "meta" or spec.tier == "hbm"]

    def whole(step, layer, head, spec):
        return Access(step, layer, head, spec.name, "R", *windows[spec.name], spec.bits)

    def clipped(step, layer, head, spec, lo, hi):
        lo, hi = max(lo, windows[spec.name][0]), min(hi, windows[spec.name][1])
        return [Access(step, layer, head, spec.name, "R", lo, hi, spec.bits)] if lo < hi else []

    if mode == "uniform_tokens":
        blocks = -(-shape.context // select_tokens)
        selective = [spec for spec in layout.tensors if spec.role != "meta"]
        for step in range(steps):
            for layer in range(shape.layers):
                for head in range(shape.kv_heads):
                    for block in sorted(rng.sample(range(blocks), share(blocks, fraction))):
                        lo, hi = block * select_tokens, min((block + 1) * select_tokens, shape.context)
                        for spec in selective:
                            yield from clipped(step, layer, head, spec, lo, hi)
                    for spec in always:
                        yield whole(step, layer, head, spec)
        return
    if mode == "sink_recent":
        # StreamingLLM-style: the first ``count`` tokens plus the last ``fraction`` of the context.
        recent = max(1, round(shape.context * fraction))
        for step in range(steps):
            for layer in range(shape.layers):
                for head in range(shape.kv_heads):
                    for spec in layout.all_tensors:
                        if spec in always:
                            yield whole(step, layer, head, spec)
                            continue
                        sink = min(count, shape.context - recent)
                        if sink > 0:
                            yield from clipped(step, layer, head, spec, 0, sink)
                        yield from clipped(step, layer, head, spec, shape.context - recent, shape.context)
        return
    if mode in {"dense", "uniform_pages"}:
        for step in range(steps):
            for layer in range(shape.layers):
                for head in range(shape.kv_heads):
                    if mode == "dense":
                        for spec in layout.all_tensors:
                            yield whole(step, layer, head, spec)
                        continue
                    for t, spec in enumerate(layout.tensors):
                        if spec.role == "meta":
                            yield whole(step, layer, head, spec)
                            continue
                        blocks = layout.blocks(t)
                        for block in sorted(rng.sample(range(blocks), share(blocks, fraction))):
                            yield layout.page_access(layout.lpa(layer, head, t, block), step)
                    for spec in layout.hbm_tensors:
                        yield whole(step, layer, head, spec)
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
        while True:  # the unit's next page that belongs to the layout (striped layouts have padding)
            pa = cfg.unit_lpa(target, k, layout.base)
            if pa >= layout.base + layout.span:
                raise ValueError(f"{mode}: unit {target} needs its page {k}, beyond the layout's {layout.span} "
                                 "pages; use fewer pages per unit or a larger shape")
            k += 1
            try:
                access = layout.page_access(pa)
                break
            except ValueError:  # a padding page
                continue
        used[target] = k
        access.validate(layout)
        addresses.append(pa)
    if mode == "write_then_read":
        for pa in addresses:
            yield layout.page_access(pa, -1, "W")
    for step in range(steps):
        for pa in addresses:
            yield layout.page_access(pa, step)
