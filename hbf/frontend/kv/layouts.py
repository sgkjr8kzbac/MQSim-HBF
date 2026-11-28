"""KV cache layouts: map (layer, head, tensor, block) to host LPAs."""
from bisect import bisect_right

from ...device.config import Config
from ..trace.tokens import Access
from .tensors import Shape, default_tensors


def numpy():
    """numpy, imported on first use so the reference (pure-Python) path does not need it."""
    import numpy as np
    return np


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
    """``layer_contiguous`` regions separated by fixed gaps of padding (SPLASH-style striping)."""
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
