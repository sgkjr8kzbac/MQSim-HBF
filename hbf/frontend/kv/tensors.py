"""Model shape, stored KV tensor specifications, and density parsing."""
import json
from dataclasses import asdict, dataclass
from fractions import Fraction


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
