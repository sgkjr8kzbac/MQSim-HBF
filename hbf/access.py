"""Logical access schema, tensor formats, replaceable layouts and seeded workloads."""
import csv
import json
from bisect import bisect_right
from dataclasses import asdict, dataclass, fields
from fractions import Fraction
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


MAX_DENSITY_DENOM = 1_000_000


def parse_density(value):
    """Parse a storage density in (0, 1] from an int, float, Fraction or string (e.g. ``1/4``)."""
    if isinstance(value, bool):
        raise ValueError(f"Invalid density {value!r}: must be in (0, 1]")
    try:
        if isinstance(value, Fraction):
            rho = value
        elif isinstance(value, int):
            rho = Fraction(value)
        elif isinstance(value, float):
            rho = Fraction(str(value)).limit_denominator(MAX_DENSITY_DENOM)
        elif isinstance(value, str):
            rho = Fraction(value.strip())
        else:
            raise ValueError
    except (ValueError, ZeroDivisionError):
        raise ValueError(f"Invalid density {value!r}: must be a number or fraction in (0, 1]") from None
    if not 0 < rho <= 1:
        raise ValueError(f"Invalid density {value!r}: must be in (0, 1]")
    if rho.denominator > MAX_DENSITY_DENOM:
        raise ValueError(f"Invalid density {value!r}: denominator must be <= {MAX_DENSITY_DENOM}")
    return 1 if rho == 1 else rho


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
    * ``dequant_ops`` - GPU vector operations per stored element read (an assumption of the format);
      the elements counted are ``logical_dims`` per token.
    * ``attn_dims`` - elements one query head multiplies per stored token (QK^T plus PV). Unset means
      ``dims`` for data tensors and 0 for aux and meta tensors. MLA's single 576-dim latent is attended
      as 1,088 elements (QK over 576, PV over 512); a DSA indexer is a meta tensor with attn_dims > 0.
    * ``logical_dims`` - BF16-equivalent elements stored per token (unset: ``dims``). It feeds
      ``hbf_elements`` and ``dequant_ops`` when one row packs several fields, e.g. a fused row of
      2-bit K and V plus their scales (``bits=8``, ``dims`` = bytes, ``logical_dims`` = d_k + d_v).
    * ``density`` - share of the window's tokens stored compacted in token order, in (0, 1]
      (default 1). Stored as ``1`` when full and a ``Fraction`` when < 1; numpy engine only.
    """
    name: str
    bits: int
    dims: int
    window: tuple | None = None
    tier: str = "hbf"
    role: str = "data"
    dequant_ops: float = 0
    attn_dims: int | None = None
    logical_dims: int | None = None
    density: Fraction | int = 1

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
        if self.attn_dims is not None and (type(self.attn_dims) is not int or self.attn_dims < 0):
            raise ValueError(f"Tensor {self.name}: attn_dims must be a nonnegative integer")
        if self.logical_dims is not None and (type(self.logical_dims) is not int or self.logical_dims <= 0):
            raise ValueError(f"Tensor {self.name}: logical_dims must be a positive integer")
        object.__setattr__(self, "density", parse_density(self.density))

    @property
    def token_bytes(self):
        return self.dims * self.bits // 8

    @property
    def attended_dims(self):
        """Elements one query head multiplies per stored token (``attn_dims`` or the role default)."""
        if self.attn_dims is not None:
            return self.attn_dims
        return self.dims if self.role == "data" else 0

    @property
    def logical(self):
        """BF16-equivalent elements stored per token."""
        return self.dims if self.logical_dims is None else self.logical_dims

    def tokens(self, context):
        """The window as absolute token indices [lo, hi) of a ``context``-token sequence."""
        lo, hi = self.window or (None, None)
        lo = 0 if lo is None else lo + context if lo < 0 else lo
        hi = context if hi is None else hi + context if hi < 0 else hi
        if not 0 <= lo < hi <= context:
            raise ValueError(f"Window {self.window} of tensor {self.name} is empty or outside {context} tokens")
        return lo, hi

    def stored_tokens(self, context):
        """Compacted tokens stored in this tensor's window (all window tokens when density == 1)."""
        lo, hi = self.tokens(context)
        width = hi - lo
        if self.density == 1:
            return width
        return ((width - 1) * self.density.numerator) // self.density.denominator + 1

    def blocks(self, context, page_bytes):
        """Pages (blocks) per head required to store this tensor's compacted window."""
        tpp = page_bytes // self.token_bytes
        if tpp <= 0:
            raise ValueError(f"Token row of {self.name} ({self.token_bytes} B) exceeds page size {page_bytes} B")
        return -(-self.stored_tokens(context) // tpp)


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
            self.op in {"R", "W", "H"} and layout.windows[self.kind][0] <= self.token_start < self.token_end
            <= layout.windows[self.kind][1] and self.bits == spec.bits
        ) or (hasattr(layout, "reads") and not layout.reads(self.layer, self.kind, self.kv_head)):
            raise ValueError(f"Invalid logical access: {self}")


def write_access(path, accesses):
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([f.name for f in fields(Access)])
        writer.writerows(vars(access).values() for access in accesses)


OPTIONAL_FIELDS = ("attn_dims", "logical_dims")  # left out of meta.json while unset, so older runs compare equal


def _tensor_meta(spec):
    item = {}
    for key, value in asdict(spec).items():
        if key in OPTIONAL_FIELDS and value is None:
            continue
        if key == "density":
            if value == 1:
                continue
            value = str(value)
        item[key] = value
    return item


def tensors_meta(layout):
    return [_tensor_meta(spec) for spec in layout.all_tensors]


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
        self.block_counts = tuple(spec.blocks(shape.context, cfg.page_bytes) for spec in self.tensors)
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
        tpp = self.tokens(t)
        origin = self.window_starts[t]
        end = self.windows[spec.name][1]
        if spec.density == 1:
            lo = origin + block * tpp
            hi = min(lo + tpp, end)
        else:
            num, den = spec.density.numerator, spec.density.denominator
            lo = origin + -(-(block * tpp * den) // num)
            hi = min(origin + -(-((block + 1) * tpp * den) // num), end)
        return Access(step, layer, head, spec.name, op, lo, hi, spec.bits)

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
        if (len(set(self.tokens_per_page)) != 1
                or len({(self.windows[spec.name], spec.density) for spec in self.tensors}) != 1):
            raise ValueError("vllm_interleaved requires the same tokens per page, window and density for every tensor")

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
    "hot_unit", "write_then_read", "channel_select",
)


def share(count, fraction):
    """ceil(count * fraction), ignoring float noise (100 * 0.07 is 7.000000000000001)."""
    return ceil(round(count * fraction, 9))


def _sample_excluding(rng, total, excluded, count):
    """Sample ``count`` distinct integers uniformly from ``range(total) \\ excluded``."""
    if count == 0:
        return []
    avail = total - len(excluded)
    if count >= avail:
        return [b for b in range(total) if b not in excluded]
    if len(excluded) * 2 <= total:
        picked = set()
        while len(picked) < count:
            need = count - len(picked)
            for b in rng.sample(range(total), min(total, need * 2)):
                if b not in excluded and b not in picked:
                    picked.add(b)
                    if len(picked) == count:
                        break
        return list(picked)
    pool = [b for b in range(total) if b not in excluded]
    return rng.sample(pool, count)


def _select_blocks(rng, blocks, pick_n, step, key, prev, select_static, reuse):
    """Return ``(ordered_blocks, reused_set)`` for one (step, selection key).

    On the default path (``select_static=False, reuse=0.0``) this performs a single
    ``rng.sample(range(blocks), pick_n)`` with zero extra draws.
    """
    if not select_static and not reuse:
        return sorted(rng.sample(range(blocks), pick_n)), ()
    if step == 0:
        picked = sorted(rng.sample(range(blocks), pick_n))
        prev[key] = picked
        return picked, ()
    prev_list = prev[key]
    if select_static:
        if not reuse:
            return prev_list, ()
        n_reused = min(len(prev_list), round(len(prev_list) * reuse))
        reused = set(rng.sample(prev_list, n_reused)) if n_reused else set()
        return prev_list, reused
    # Dynamic selection with temporal reuse: fraction ``reuse`` carries over from the prior step
    # and is served from the HBM cache, while the rest are freshly selected from HBF.
    n_reused = min(len(prev_list), round(pick_n * reuse))
    reused = set(rng.sample(prev_list, n_reused)) if n_reused else set()
    n_miss = pick_n - n_reused
    prev_set = set(prev_list)
    avail = blocks - len(prev_set)
    if n_miss <= avail:
        fresh = _sample_excluding(rng, blocks, prev_set, n_miss)
    else:
        fresh = [b for b in range(blocks) if b not in prev_set]
        fresh.extend(rng.sample([b for b in prev_list if b not in reused], n_miss - avail))
    picked = sorted(reused | set(fresh))
    prev[key] = picked
    return picked, reused


def _detect_channel_groups(layout):
    """Detect contiguous SparQ-style channel-group tensors ``K_g0, K_g1, ...`` in ``layout.tensors``."""
    found = []
    for spec in layout.tensors:
        if spec.name.startswith("K_g") and spec.name[3:].isdigit():
            found.append((int(spec.name[3:]), spec.name))
    if not found:
        return ()
    found.sort()
    if [idx for idx, _ in found] != list(range(len(found))):
        return ()
    return tuple(name for _, name in found)


def generate(cfg, layout, mode, *, steps=1, seed=1, fraction=.1, count=4, unit=0,
             hot_fraction=.5, hot_units=1, select_tokens=16,
             select_static=False, reuse=0.0, query_groups=1, full_tensors=(),
             channels=None, channel_groups=()):
    """Seeded accesses of one workload mode.

    Every mode reads ``meta`` tensors (selection metadata) in full every (step, layer, head);
    selection modes also read HBM-tier tensors (hot windows) and any tensors named in
    ``full_tensors`` in full and clip the selected tokens to each tensor's window; with the
    default tensors and options this changes nothing. ``uniform_pages`` samples each tensor's
    pages independently, so K and V (and scales) cover different tokens.

    ``uniform_pages`` picks ``fraction`` of each tensor's storage pages, independently per tensor, so
    the selection unit grows with compression (a 2-bit page holds 128 tokens, a BF16 page 16) and K
    and V pages of one head cover different tokens. ``uniform_tokens`` picks ``fraction`` of the
    blocks of ``select_tokens`` tokens per (step, layer, KV head), as Quest-style methods do, and
    reads every tensor of the head for the same tokens.

    ``channel_select`` models two-phase partial-dimension scoring followed by sparse exact attention:
    * Static prefix (Loki): reads ``full_tensors`` (auto-detecting ``K_lo`` when unset) over the
      whole context, then reads the remaining HBF tensors (e.g. ``K_hi``, ``V``) on the selected
      ``select_tokens``-token blocks.
    * Dynamic channel groups (SparQ): per (step, layer, head), draws ``channels`` distinct channels
      uniformly without replacement across ``channel_groups`` (auto-detecting ``K_g0..K_g{g-1}``
      when unset) and reads the touched group tensors over the whole context; the exact phase reads
      the remaining data tensors (token-major ``K`` and ``V`` when ``dual=1``, or untouched ``K_g*``
      and ``V`` when ``dual=0``) on the selected blocks.

    Selection realism options (all off by default, preserving exact RNG draw order):
    * ``select_static`` - keep the step-0 selection across all steps (eviction state).
    * ``reuse`` - fraction in [0, 1] of blocks carried over from the previous step on step >= 1
      and served from an HBM cache (emitted with ``op="H"``) instead of HBF.
    * ``query_groups`` - GQA union across ``g`` query heads sharing one KV head, giving effective
      selection fraction ``1 - (1 - fraction) ** g``.
    * ``full_tensors`` - tensor names read over their whole window even in sparse selection modes.
    """
    shape = layout.shape
    if (mode not in MODES or steps <= 0 or not 0 < fraction <= 1 or count <= 0
            or not 0 <= hot_fraction <= 1 or not 1 <= hot_units <= cfg.units
            or not 0 <= unit < cfg.units or type(select_tokens) is not int or select_tokens <= 0
            or type(select_static) is not bool
            or isinstance(reuse, bool) or not isinstance(reuse, (int, float)) or not 0 <= reuse <= 1
            or type(query_groups) is not int or query_groups < 1):
        raise ValueError("Invalid generator parameters")
    if (select_static or reuse or query_groups != 1) and mode not in ("uniform_tokens", "uniform_pages", "channel_select"):
        raise ValueError("select_static, reuse and query_groups apply to uniform_tokens, uniform_pages and channel_select only")
    if (channels is not None or channel_groups) and mode != "channel_select":
        raise ValueError("channels and channel_groups apply to channel_select only")
    full = frozenset(full_tensors)
    if any(not isinstance(name, str) or name not in layout.specs for name in full):
        raise ValueError(f"full_tensors must name tensors of the layout, not {sorted(full - set(layout.specs))}")
    rng = Random(seed)
    windows = layout.windows
    always = [spec for spec in layout.all_tensors if spec.role == "meta" or spec.tier == "hbm" or spec.name in full]
    eff_fraction = fraction if query_groups == 1 else 1.0 - (1.0 - fraction) ** query_groups

    def whole(step, layer, head, spec):
        return Access(step, layer, head, spec.name, "R", *windows[spec.name], spec.bits)

    def clipped(step, layer, head, spec, lo, hi, op="R"):
        lo, hi = max(lo, windows[spec.name][0]), min(hi, windows[spec.name][1])
        return [Access(step, layer, head, spec.name, op, lo, hi, spec.bits)] if lo < hi else []

    if mode == "channel_select":
        groups = tuple(channel_groups) if channel_groups else _detect_channel_groups(layout)
        if any(not isinstance(name, str) or name not in layout.specs for name in groups):
            raise ValueError(f"channel_groups must name tensors of the layout, not "
                             f"{sorted(set(groups) - set(layout.specs))}")
        if len(set(groups)) != len(groups):
            raise ValueError("channel_groups must not contain duplicate tensor names")
        groups_set = frozenset(groups)
        if groups_set & full:
            raise ValueError(f"Tensors cannot be in both channel_groups and full_tensors: "
                             f"{sorted(groups_set & full)}")
        if groups:
            if any(layout.specs[name].tier != "hbf" for name in groups):
                raise ValueError("channel_groups tensors must be on the hbf tier")
            total_dims = sum(layout.specs[name].dims for name in groups)
            if type(channels) is not int or not 1 <= channels <= total_dims:
                raise ValueError(f"channel_select with channel_groups needs integer channels in [1, {total_dims}]")
            channel_to_group = [i for i, name in enumerate(groups) for _ in range(layout.specs[name].dims)]
        elif channels is not None:
            raise ValueError("channels requires channel_groups (e.g. sparq format)")
        if not groups and not full and "K_lo" in layout.specs:
            full = frozenset(("K_lo",))
            always = [spec for spec in layout.all_tensors
                      if spec.role == "meta" or spec.tier == "hbm" or spec.name in full]
        if not groups and not full:
            raise ValueError("channel_select requires full_tensors (e.g. loki K_lo) or "
                             "channel_groups (e.g. sparq K_g*)")
        blocks = -(-shape.context // select_tokens)
        pick_n = share(blocks, eff_fraction)
        base_selective = [spec for spec in layout.tensors
                          if spec.role != "meta" and spec.name not in full and spec.name not in groups_set]
        prev = {}
        prev_groups = {}
        for step in range(steps):
            for layer in range(shape.layers):
                for head in range(shape.kv_heads):
                    if groups:
                        if select_static and step > 0:
                            touched_idx = prev_groups[layer, head]
                        else:
                            picked_ch = rng.sample(range(total_dims), channels)
                            touched_idx = sorted({channel_to_group[c] for c in picked_ch})
                            if select_static:
                                prev_groups[layer, head] = touched_idx
                        touched_set = set(touched_idx)
                        for idx in touched_idx:
                            yield whole(step, layer, head, layout.specs[groups[idx]])
                        untouched_data = [layout.specs[groups[i]] for i in range(len(groups))
                                          if i not in touched_set and layout.specs[groups[i]].role == "data"]
                        exact_tensors = untouched_data + base_selective
                    else:
                        exact_tensors = base_selective
                    picked, reused = _select_blocks(rng, blocks, pick_n, step, (layer, head), prev,
                                                    select_static, reuse)
                    for block in picked:
                        lo, hi = block * select_tokens, min((block + 1) * select_tokens, shape.context)
                        op = "H" if block in reused else "R"
                        for spec in exact_tensors:
                            yield from clipped(step, layer, head, spec, lo, hi, op)
                    for spec in always:
                        yield whole(step, layer, head, spec)
        return

    if mode == "uniform_tokens":
        blocks = -(-shape.context // select_tokens)
        pick_n = share(blocks, eff_fraction)
        selective = [spec for spec in layout.tensors if spec.role != "meta" and spec.name not in full]
        prev = {}
        for step in range(steps):
            for layer in range(shape.layers):
                for head in range(shape.kv_heads):
                    picked, reused = _select_blocks(rng, blocks, pick_n, step, (layer, head), prev,
                                                    select_static, reuse)
                    for block in picked:
                        lo, hi = block * select_tokens, min((block + 1) * select_tokens, shape.context)
                        op = "H" if block in reused else "R"
                        for spec in selective:
                            yield from clipped(step, layer, head, spec, lo, hi, op)
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
        prev = {}
        for step in range(steps):
            for layer in range(shape.layers):
                for head in range(shape.kv_heads):
                    if mode == "dense":
                        for spec in layout.all_tensors:
                            yield whole(step, layer, head, spec)
                        continue
                    for t, spec in enumerate(layout.tensors):
                        if spec.role == "meta" or spec.name in full:
                            yield whole(step, layer, head, spec)
                            continue
                        blocks = layout.blocks(t)
                        pick_n = share(blocks, eff_fraction)
                        picked, reused = _select_blocks(rng, blocks, pick_n, step, (layer, head, t), prev,
                                                        select_static, reuse)
                        for block in picked:
                            op = "H" if block in reused else "R"
                            yield layout.page_access(layout.lpa(layer, head, t, block), step, op)
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
