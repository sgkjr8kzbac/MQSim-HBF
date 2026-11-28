"""KV specs with layer and head groups (phase 3, T-K2): ``--kv-spec PRESET[:key=value,...]``.

A ``KVSpec`` is a list of ``KVGroup``s. Each group names the layers and the per-request heads that
store (or read) its tensors, plus an optional token window, tier, per-layer budget, cross-layer
aliasing and a fixed SSM state. ``GroupedLayout`` compiles the groups into flat tensors named
``<group>_<tensor>`` and a start table [layer, tensor, head] (-1 where a tensor is absent), so the
analysis engine (``hbf.pages``) runs unchanged; ``generate_grouped`` emits the accesses.

| preset | groups | methods |
|---|---|---|
| ``hybrid`` (``pattern``, ``window``, ``local_tier``) | local layers (L) keep the last W tokens, global layers (G) all | Gemma 3 (LLLLLG, W 1,024), GPT-OSS (LG, W 128) |
| ``ssm`` (``pattern``, ``state``) | attention layers (A) store KV; SSM layers (S) read a fixed state from HBM | Qwen3-Next (SSSA), Jamba |
| ``cla`` (``share``, ``reuse``) | layer l reads the KV of layer l - l mod s | Cross-Layer Attention (CLA-2) |
| ``yoco`` (``window``, ``local_tier``, ``reuse``) | first half local (sliding window), second half reads one shared KV | YOCO |
| ``duo`` (``retrieval``, ``sink``, ``recent``, ``streaming_tier``) | retrieval heads keep all tokens, streaming heads sink + recent | DuoAttention, RazorAttention |
| ``pyramid`` (``first``, ``last``) | per-layer budgets falling linearly from the first to the last layer | PyramidKV (prefill-time eviction, compacted) |

Defaults that are assumptions: local and streaming KV live in HBM (``*_tier=hbm``; on HBF they are
written once and die after W tokens, see design section 1); the SSM state is 2 MiB per layer and
request. A budget is modelled as a compacted window [0, budget), so pages and rounds are right but
the kept tokens' positions are not.
"""
from bisect import bisect_right
from dataclasses import dataclass, field, replace
from random import Random

from ...frontend.kv.layouts import Layout
from ...frontend.kv.tensors import TensorSpec
from ...frontend.trace.synthetic import share
from ...frontend.trace.tokens import Access

PRESETS = ("hybrid", "ssm", "cla", "yoco", "duo", "pyramid")
DEFAULTS = {
    "hybrid": {"pattern": "LLLLLG", "window": 1024, "local_tier": "hbm"},
    "ssm": {"pattern": "SSSA", "state": 2 * 2 ** 20},
    "cla": {"share": 2, "reuse": "hbf"},
    "yoco": {"window": 1024, "local_tier": "hbm", "reuse": "hbf"},
    "duo": {"retrieval": 0.5, "sink": 64, "recent": 256, "streaming_tier": "hbm"},
    "pyramid": {"first": None, "last": None},
}
GROUP_MODES = ("dense", "uniform_tokens", "sink_recent")
REUSE = ("hbf", "hbm")


@dataclass(frozen=True)
class KVGroup:
    """Layers and per-request heads [lo, hi) that read this group's tensors.

    ``alias`` lists (consumer, owner) layers: a consumer stores nothing and reads the owner's KV,
    from the owner's HBF pages (``reuse="hbf"``) or from a copy staged in HBM (``reuse="hbm"``).
    ``budget`` keeps that many tokens (window [0, budget)); ``window`` and ``tier`` apply on top of
    every tensor's own. ``tensors`` None means the run's format.
    """
    name: str
    layers: tuple
    heads: tuple | None = None
    tensors: tuple | None = None
    window: tuple | None = None
    tier: str | None = None
    alias: tuple = ()
    reuse: str = "hbf"
    budget: int | None = None

    def __post_init__(self):
        if not self.name.isidentifier() or not self.layers or self.reuse not in REUSE:
            raise ValueError(f"Invalid KV group {self.name!r}")
        if self.tier not in (None, "hbf", "hbm"):
            raise ValueError(f"Group {self.name}: tier must be hbf or hbm")
        consumers = {c for c, _ in self.alias}
        if any(c not in self.layers or o not in self.layers or o in consumers or c == o for c, o in self.alias):
            raise ValueError(f"Group {self.name}: an alias maps a consumer layer to an owner layer of the group")


def _scales_with_context(window):
    """True if ``window`` grows linearly with context length (unbounded or [0, ctx - R))."""
    lo, hi = window or (None, None)
    return (lo is None or lo >= 0) and (hi is None or hi < 0)


@dataclass(frozen=True)
class KVSpec:
    name: str
    params: dict
    groups: tuple
    notes: tuple = field(default_factory=tuple)

    def bytes_per_token(self, heads, tensors=None, *, tier=None, include_windowed=False):
        """Stored KV bytes per token across owner layers for context-scaling groups/tensors.

        Fixed-size tensors (SSM state ``(0, 1)``, sliding-window local layers ``(-W, None)``,
        bounded budgets, KIVI HBM residuals) are excluded unless ``include_windowed=True``.
        """
        total = 0
        for group in self.groups:
            if not include_windowed and (group.budget is not None or not _scales_with_context(group.window)):
                continue
            group_tensors = group.tensors if group.tensors is not None else tensors
            if group_tensors is None:
                raise ValueError(f"Group {group.name} has no tensors; pass tensors or build the spec with from_model")
            owners = len(group.layers) - len(group.alias)
            g_heads = (group.heads[1] - group.heads[0]) if group.heads is not None else heads
            for spec in group_tensors:
                effective_tier = group.tier or spec.tier
                if tier is not None and effective_tier != tier:
                    continue
                if not include_windowed and not _scales_with_context(spec.window):
                    continue
                total += owners * g_heads * spec.token_bytes
        return total

    def geometry(self, layers, heads, tensors=None, page_bytes=4096):
        """Return a ``KVGeometry`` (duck type of ``endurance.Geometry``) for this KV spec."""
        return KVGeometry(self, layers, heads, tensors=tensors, page_bytes=page_bytes)


def parse_params(text):
    params = {}
    for item in filter(None, text.split(",")):
        key, sep, value = item.partition("=")
        if not sep or not key.isidentifier():
            raise ValueError(f"Invalid KV spec parameter {item!r}: use key=value")
        for kind in (int, float):
            try:
                params[key] = kind(value)
                break
            except ValueError:
                continue
        else:
            params[key] = value
    return params


def _tier(value, key):
    if value not in ("hbf", "hbm"):
        raise ValueError(f"{key} must be hbf or hbm")
    return None if value == "hbf" else "hbm"


def _count(value, key, low=1):
    if type(value) is not int or value < low:
        raise ValueError(f"{key} must be an integer >= {low}")
    return value


def _pattern(pattern, letters, layers):
    if not isinstance(pattern, str) or not pattern or set(pattern) - set(letters):
        raise ValueError(f"pattern must be a string of {', '.join(letters)}")
    return [pattern[layer % len(pattern)] for layer in range(layers)]


def parse_kvspec(text, layers, heads, context):
    """``hybrid:pattern=LLLLLG,window=1024`` -> KVSpec for ``layers`` layers and ``heads`` heads per request."""
    name, _, rest = text.partition(":")
    if name not in PRESETS:
        raise ValueError(f"Unknown KV spec {name!r}; presets: {', '.join(PRESETS)}")
    given = parse_params(rest)
    if set(given) - set(DEFAULTS[name]):
        raise ValueError(f"KV spec {name} takes {sorted(DEFAULTS[name])}, not {sorted(set(given) - set(DEFAULTS[name]))}")
    p = DEFAULTS[name] | given
    every = tuple(range(layers))
    notes = []
    if name in ("hybrid", "yoco"):
        window, tier = _count(p["window"], "window"), _tier(p["local_tier"], "local_tier")
        if name == "hybrid":
            kinds = _pattern(p["pattern"], "LG", layers)
            local = tuple(l for l in every if kinds[l] == "L")
            full = tuple(l for l in every if kinds[l] == "G")
            alias = ()
        else:
            if layers < 2 or p["reuse"] not in REUSE:
                raise ValueError("yoco needs at least 2 layers and reuse hbf or hbm")
            local, full = every[:layers // 2], every[layers // 2:]
            alias = tuple((l, full[0]) for l in full[1:])
        groups = [KVGroup("local", local, window=(-window, None), tier=tier)] if local else []
        if full:
            groups.append(KVGroup("global", full, alias=alias, reuse=p.get("reuse", "hbf")))
        notes.append(f"局部层（{len(local)} 层）只保留最近 {window} 个 token，放在 {'HBM' if tier else 'HBF'}"
                     + ("；后半的层共用第一层全局层的 KV" if name == "yoco" else ""))
    elif name == "ssm":
        kinds = _pattern(p["pattern"], "SA", layers)
        state = _count(p["state"], "state")
        attention = tuple(l for l in every if kinds[l] == "A")
        recurrent = tuple(l for l in every if kinds[l] == "S")
        groups = [KVGroup("attn", attention)] if attention else []
        if recurrent:
            groups.append(KVGroup("ssm", recurrent, heads=(0, 1),
                                  tensors=(TensorSpec("state", 8, state, (0, 1), "hbm", "aux"),)))
        notes.append(f"SSM/线性注意力层（{len(recurrent)} 层）每个请求每层读 {state} B 固定状态（HBM），不存 KV")
    elif name == "cla":
        group = _count(p["share"], "share", 2)
        if p["reuse"] not in REUSE:
            raise ValueError("reuse must be hbf or hbm")
        alias = tuple((l, l - l % group) for l in every if l % group)
        groups = [KVGroup("kv", every, alias=alias, reuse=p["reuse"])]
        notes.append(f"跨层共享：每 {group} 层共用一份 KV，"
                     + ("其余层重读所属层的 HBF 页" if p["reuse"] == "hbf" else "其余层读暂存在 HBM 的副本"))
    elif name == "duo":
        retrieval = p["retrieval"]
        if isinstance(retrieval, bool) or not isinstance(retrieval, (int, float)) or not 0 <= retrieval <= 1:
            raise ValueError("retrieval must be in [0, 1]")
        sink, recent = _count(p["sink"], "sink", 0), _count(p["recent"], "recent")
        tier = _tier(p["streaming_tier"], "streaming_tier")
        keep = round(retrieval * heads)
        groups = [KVGroup("retrieval", every, heads=(0, keep))] if keep else []
        if keep < heads:
            if sink and sink + recent >= context:
                groups.append(KVGroup("streaming", every, heads=(keep, heads), tier=tier))
            else:
                if sink:
                    groups.append(KVGroup("sink", every, heads=(keep, heads), window=(0, sink), tier=tier))
                groups.append(KVGroup("recent", every, heads=(keep, heads), window=(-recent, None), tier=tier))
        notes.append(f"检索头 {keep}/{heads} 保留全部 token；流式头只保留前 {sink} 个和最近 {recent} 个，"
                     f"放在 {'HBM' if tier else 'HBF'}")
    else:  # pyramid
        first = context if p["first"] is None else _count(p["first"], "first")
        last = max(1, context // 4) if p["last"] is None else _count(p["last"], "last")
        budgets = [round(first + (last - first) * l / max(layers - 1, 1)) for l in every]
        groups = []
        for budget in dict.fromkeys(budgets):
            groups.append(KVGroup(f"b{budget}", tuple(l for l in every if budgets[l] == budget),
                                  budget=min(budget, context)))
        notes.append(f"PyramidKV：每层保留的 token 从第一层的 {first} 线性降到最后一层的 {last}（prefill 时淘汰、压紧存放）")
    return KVSpec(name, p, tuple(groups), tuple(notes))


def resolve_format(model, fmt):
    """Resolve ``fmt`` (None, format string or KVFormat) for ``model``."""
    from ...frontend.kv.formats import KVFormat, parse_format, parse_params as parse_fmt_params
    if isinstance(fmt, KVFormat):
        return fmt
    q_heads = model.q_heads // model.kv_heads
    if model.attention == "mla":
        if fmt is None or fmt == "bf16":
            name, given = "mla", {}
        elif fmt == "fp8":
            name, given = "mla_fp8", {}
        elif isinstance(fmt, str):
            name, _, rest = fmt.partition(":")
            if name not in ("mla", "mla_fp8"):
                raise ValueError(f"MLA model {model.name} requires bf16, fp8, mla or mla_fp8 format, not {fmt!r}")
            given = parse_fmt_params(rest)
        else:
            raise ValueError(f"Invalid KV format {fmt!r}")
        params = {"lora": model.kv_lora_rank, "rope": model.rope_dim} | given
        text = f"{name}:" + ",".join(f"{key}={value}" for key, value in params.items())
        return parse_format(text, model.head_dim, q_heads)
    if fmt is None:
        fmt = "bf16"
    if not isinstance(fmt, str):
        raise ValueError(f"Invalid KV format {fmt!r}")
    return parse_format(fmt, model.head_dim, q_heads)


_resolve_format = resolve_format


def from_model(model, fmt="bf16", local_tier="hbm", *, tensors=None):
    """Build a self-contained ``KVSpec`` from a ``ModelSpec`` (or preset name) and KV format.

    Sliding-window local KV defaults to HBM (``local_tier="hbm"``; pass ``"hbf"`` as an
    experiment variant). MLA models default to the latent format matching ``kv_lora_rank`` and
    ``rope_dim``; SSM hybrid models emit fixed HBM state groups on ``S`` layers. When ``tensors``
    is provided, attention groups use those tensor specs directly instead of resolving ``fmt``.
    """
    if isinstance(model, str):
        from ...models import load_model
        model = load_model(model)
    tier = _tier(local_tier, "local_tier")
    if tensors is not None:
        group_tensors = tuple(tensors)
        if not group_tensors:
            raise ValueError("tensors must be nonempty")
        fmt_name = "tensors"
        notes = []
    else:
        kv_fmt = resolve_format(model, fmt)
        group_tensors = kv_fmt.tensors
        fmt_name = kv_fmt.name
        notes = list(kv_fmt.notes)
    every = tuple(range(model.layers))
    if not model.layer_pattern:
        name = "mla" if model.attention == "mla" else "model"
        params = {"model": model.name, "format": fmt_name}
        groups = (KVGroup("kv", every, tensors=group_tensors),)
        return KVSpec(name, params, groups, tuple(notes))
    letters = set(model.layer_pattern)
    if letters <= {"L", "G"}:
        kinds = _pattern(model.layer_pattern, "LG", model.layers)
        local = tuple(l for l in every if kinds[l] == "L")
        full = tuple(l for l in every if kinds[l] == "G")
        groups = []
        if local:
            groups.append(KVGroup("local", local, tensors=group_tensors,
                                  window=(-model.window, None), tier=tier))
        if full:
            groups.append(KVGroup("global", full, tensors=group_tensors))
        notes.append(f"局部层（{len(local)} 层）只保留最近 {model.window} 个 token，放在 {'HBM' if tier else 'HBF'}")
        return KVSpec("hybrid", {"pattern": model.layer_pattern, "window": model.window, "local_tier": local_tier},
                      tuple(groups), tuple(notes))
    if letters <= {"S", "A"}:
        kinds = _pattern(model.layer_pattern, "SA", model.layers)
        attention = tuple(l for l in every if kinds[l] == "A")
        recurrent = tuple(l for l in every if kinds[l] == "S")
        groups = []
        if attention:
            groups.append(KVGroup("attn", attention, tensors=group_tensors))
        if recurrent:
            groups.append(KVGroup("ssm", recurrent, heads=(0, 1),
                                  tensors=(TensorSpec("state", 8, model.ssm_state_bytes, (0, 1), "hbm", "aux"),)))
        notes.append(f"SSM/线性注意力层（{len(recurrent)} 层）每个请求每层读 {model.ssm_state_bytes} B 固定状态（HBM），不存 KV")
        return KVSpec("ssm", {"pattern": model.layer_pattern, "state": model.ssm_state_bytes},
                      tuple(groups), tuple(notes))
    raise ValueError(f"Unsupported layer_pattern {model.layer_pattern!r} on model {model.name}")


def _absolute(window, context):
    lo, hi = window or (None, None)
    lo = 0 if lo is None else max(0, lo + context) if lo < 0 else min(lo, context)
    hi = context if hi is None else max(0, hi + context) if hi < 0 else min(hi, context)
    return lo, hi


EMPTY = "empty"


def combine(own, group, context):
    """A tensor's window inside a group window (clamped to [0, context)); EMPTY if nothing is left."""
    if group is None:
        return own
    if own is None:
        lo, hi = _absolute(group, context)
        return (lo, hi) if lo < hi else EMPTY
    (a, b), (c, d) = _absolute(own, context), _absolute(group, context)
    lo, hi = max(a, c), min(b, d)
    return (lo, hi) if lo < hi else EMPTY


def _peak_live_owners(alias):
    """Peak number of simultaneously live aliased owners during layer-by-layer execution."""
    spans = {}
    for consumer, owner in alias:
        lo, hi = spans.get(owner, (owner, owner))
        spans[owner] = (min(lo, consumer), max(hi, consumer))
    return max((sum(lo <= point <= hi for lo, hi in spans.values()) for point, _ in spans.values()), default=0)


def _striped_gap(units, counts):
    """Inter-region gap for striped placement of one layer without bank collisions."""
    if sum(counts) <= units:
        return 0
    base = max(units // len(counts), 1)
    target = min(len(counts), units)

    def distinct(gap):
        pos, seen = 0, set()
        for n, count in enumerate(counts):
            pos += gap if n else 0
            seen.add(pos % units)
            pos += count
        return len(seen)

    if distinct(base) == target:
        return base
    for candidate in (0, base + 1, max(base - 1, 1)):
        if distinct(candidate) == target:
            return candidate
    return max((0, base, base + 1), key=distinct)


class GroupedLayout(Layout):
    """Flat tensors ``<group>_<tensor>`` with per-layer presence, per-request head ranges and aliases.

    Regions are ordered (layer, tensor, head) over the stored (owner) entries; ``intra="striped"``
    starts every layer on phase 0 and puts a gap of ``units // regions of the layer`` before each
    region but the first, as ``striped_contiguous`` does. Aliased consumer entries point at the
    owner's starts (``reuse="hbf"``) or read an HBM-tier copy tensor ``<group>_<tensor>_copy``.
    ``requests`` replicates the per-request heads (the ``--batch`` emulation).
    """

    def __init__(self, cfg, shape, spec, tensors, requests=1, intra="contiguous", base=0):
        np = __import__("numpy")
        if intra not in ("contiguous", "striped"):
            raise ValueError("intra must be contiguous or striped")
        if shape.kv_heads % requests:
            raise ValueError("KV heads must be a multiple of the requests")
        self.shape, self.base, self.spec, self.requests = shape, base, spec, requests
        per_request = shape.kv_heads // requests
        flat, reads = [], {}  # reads[name] = (layers read, stored layers, heads, owner of each layer)
        self.copy_owners = {}  # HBM copy tensor -> number of owner layers it copies
        for group in spec.groups:
            if max(group.layers) >= shape.layers:
                raise ValueError(f"Group {group.name} names layer {max(group.layers)} of a {shape.layers}-layer model")
            lo, hi = group.heads or (0, per_request)
            if not 0 <= lo < hi <= per_request:
                raise ValueError(f"Group {group.name}: heads [{lo}, {hi}) outside the {per_request} heads per request")
            heads = tuple(r * per_request + h for r in range(requests) for h in range(lo, hi))
            owner = dict(group.alias)
            stored = tuple(l for l in group.layers if l not in owner)
            window = (0, group.budget) if group.budget is not None else group.window
            for base_spec in group.tensors or tensors:
                own = combine(base_spec.window, window, shape.context)
                if own == EMPTY:
                    continue  # empty inside the group window (e.g. a KIVI residual under a small budget)
                spec_ = replace(base_spec, name=f"{group.name}_{base_spec.name}", window=own,
                                tier=group.tier or base_spec.tier)
                if spec_.tier == "hbm" or group.reuse == "hbf" or not owner:
                    flat.append(spec_)
                    reads[spec_.name] = (group.layers, stored, heads, owner)
                    continue
                flat.append(spec_)  # owners read their own pages, consumers an HBM copy
                reads[spec_.name] = (stored, stored, heads, {})
                copy = replace(spec_, name=f"{spec_.name}_copy", tier="hbm")
                flat.append(copy)
                reads[copy.name] = (tuple(owner), (), heads, {})
                self.copy_owners[copy.name] = _peak_live_owners(group.alias)
        names = [s.name for s in flat]
        if not flat or len(set(names)) != len(names):
            raise ValueError("A KV spec needs uniquely named tensors")
        if any(s.density != 1 for s in flat):
            raise ValueError("GroupedLayout does not yet support tensor density < 1")
        self.all_tensors = tuple(flat)
        self.tensors = tuple(s for s in flat if s.tier == "hbf")
        self.hbm_tensors = tuple(s for s in flat if s.tier == "hbm")
        if not self.tensors:
            raise ValueError("At least one tensor must be stored on HBF")
        self.specs = {s.name: s for s in flat}
        self.windows = {s.name: s.tokens(shape.context) for s in flat}
        self.index = {s.name: t for t, s in enumerate(self.tensors)}
        self.tokens_per_page = tuple(cfg.page_bytes // s.token_bytes for s in self.tensors)
        if base < 0 or 0 in self.tokens_per_page:
            raise ValueError("A token row must fit in one page and base must be nonnegative")
        self.window_starts = tuple(self.windows[s.name][0] for s in self.tensors)
        self.block_counts = tuple(-(-(self.windows[s.name][1] - self.windows[s.name][0]) // tokens)
                                  for s, tokens in zip(self.tensors, self.tokens_per_page))
        self.read_map = reads
        self.layer_reads = [[(s, reads[s.name][2]) for s in flat if layer in reads[s.name][0]]
                            for layer in range(shape.layers)]
        self._present = {(layer, s.name): frozenset(reads[s.name][2])
                         for s in flat for layer in reads[s.name][0]}
        table = np.full((shape.layers, len(self.tensors), shape.kv_heads), -1, dtype=np.int64)
        self.regions, cursor, pages = [], 0, 0
        for layer in range(shape.layers):
            entries = [(t, head) for t, s in enumerate(self.tensors) if layer in reads[s.name][1]
                       for head in reads[s.name][2]]
            if not entries:
                continue
            gap = _striped_gap(cfg.units, [self.block_counts[t] for t, _ in entries]) if intra == "striped" else 0
            if intra == "striped":
                cursor += -(base + cursor) % cfg.units
            for n, (t, head) in enumerate(entries):
                start = cursor + (gap if n else 0)
                table[layer, t, head] = start
                self.regions.append((start, layer, t, head))
                cursor = start + self.block_counts[t]
                pages += self.block_counts[t]
        for t, s in enumerate(self.tensors):
            for consumer, owner in reads[s.name][3].items():
                table[consumer, t] = table[owner, t]
        self._start_table = table
        self.pages, self.span = pages, cursor
        cfg.lba(base + self.span - 1)

    def reads(self, layer, name, head):
        return head in self._present.get((layer, name), ())

    def lpa(self, layer, head, t, block):
        start = int(self._start_table[layer, t, head])
        if start < 0 or not 0 <= block < self.block_counts[t]:
            raise ValueError("Block outside the tensor's reserved region")
        return self.base + start + block

    def lpa_array(self, layer, head, t, block):
        return self.base + self._start_table[layer][t, head] + block

    def decode(self, lpa):
        offset = lpa - self.base
        index = bisect_right(self.regions, (offset, float("inf"))) - 1
        if index < 0 or not 0 <= offset - self.regions[index][0] < self.block_counts[self.regions[index][2]]:
            raise ValueError(f"LPA {lpa} is padding, not a page of the layout")
        start, layer, t, head = self.regions[index]
        return layer, head, t, offset - start

    def request_bytes(self, cfg):
        """Stored KV per request: owner HBF pages plus permanent HBM-tier tensors (all-HBM baseline),
        and HBM footprint under HBF (permanent HBM-tier tensors plus peak live staging copies for
        ``reuse="hbm"``)."""
        hbf = self.pages * cfg.page_bytes
        hbm_stored = hbm_copies = 0
        for s in self.hbm_tensors:
            _, stored, heads, _ = self.read_map[s.name]
            lo, hi = self.windows[s.name]
            size = len(heads) * (hi - lo) * s.token_bytes
            if s.name in self.copy_owners:
                hbm_copies += self.copy_owners[s.name] * size
            else:
                hbm_stored += len(stored) * size
        return {"kv_bytes_per_request": round((hbf + hbm_stored) / self.requests),
                "hbm_tier_bytes_per_request": round((hbm_stored + hbm_copies) / self.requests)}


def generate_grouped(cfg, layout, mode, *, steps=1, seed=1, fraction=.1, count=4, select_tokens=16):
    """Accesses of a grouped layout: ``dense``, ``uniform_tokens`` or ``sink_recent`` (as ``access.generate``).

    Selection modes clip each head's selected tokens to every tensor's window and read meta and
    HBM-tier tensors whole, so a local window under ``uniform_tokens`` is read sparsely too.
    """
    if mode not in GROUP_MODES:
        raise ValueError(f"--kv-spec supports the modes {', '.join(GROUP_MODES)}")
    shape = layout.shape
    if steps <= 0 or not 0 < fraction <= 1 or count <= 0 or type(select_tokens) is not int or select_tokens <= 0:
        raise ValueError("Invalid generator parameters")
    rng, windows = Random(seed), layout.windows

    def whole(step, layer, head, spec):
        return Access(step, layer, head, spec.name, "R", *windows[spec.name], spec.bits)

    def clipped(step, layer, head, spec, lo, hi):
        lo, hi = max(lo, windows[spec.name][0]), min(hi, windows[spec.name][1])
        return [Access(step, layer, head, spec.name, "R", lo, hi, spec.bits)] if lo < hi else []

    for step in range(steps):
        for layer in range(shape.layers):
            by_head = {}
            for spec, heads in layout.layer_reads[layer]:
                for head in heads:
                    by_head.setdefault(head, []).append(spec)
            for head in sorted(by_head):
                specs = by_head[head]
                always = [s for s in specs if s.role == "meta" or s.tier == "hbm"]
                if mode == "dense":
                    for spec in specs:
                        yield whole(step, layer, head, spec)
                    continue
                if mode == "uniform_tokens":
                    blocks = -(-shape.context // select_tokens)
                    picked = sorted(rng.sample(range(blocks), share(blocks, fraction)))
                    for block in picked:
                        lo, hi = block * select_tokens, min((block + 1) * select_tokens, shape.context)
                        for spec in specs:
                            if spec not in always:
                                yield from clipped(step, layer, head, spec, lo, hi)
                else:  # sink_recent
                    recent = max(1, round(shape.context * fraction))
                    sink = min(count, shape.context - recent)
                    for spec in specs:
                        if spec in always:
                            continue
                        if sink > 0:
                            yield from clipped(step, layer, head, spec, 0, sink)
                        yield from clipped(step, layer, head, spec, shape.context - recent, shape.context)
                for spec in always:
                    yield whole(step, layer, head, spec)


def spec_meta(spec):
    return {"name": spec.name, "params": spec.params,
            "groups": [{"name": g.name, "layers": list(g.layers), "heads": g.heads, "window": g.window,
                        "tier": g.tier, "alias": [list(pair) for pair in g.alias], "reuse": g.reuse,
                        "budget": g.budget, "tensors": None if g.tensors is None else [s.name for s in g.tensors]}
                       for g in spec.groups]}


# --------------------------------------------------------------------------------------------
# Token flow and KVGeometry (phase 3, T-K4)

EVICT_POLICIES = ("none", "prefill", "flush", "decode")


def live_page_fraction(keep, tpp, cluster=1):
    """Share of pages that keep at least one live token under random token eviction: 1 - (1 - keep)^(tpp / cluster).

    Clusters larger than a page span whole pages, so each page stays live with probability ``keep``.
    """
    if not 0.0 <= keep <= 1.0:
        raise ValueError(f"keep must be in [0, 1], got {keep}")
    if tpp < 1 or cluster < 1:
        raise ValueError(f"tpp and cluster must be >= 1, got tpp={tpp}, cluster={cluster}")
    return 1.0 - (1.0 - keep) ** max(1.0, tpp / cluster)


@dataclass(frozen=True)
class EvictPolicy:
    """Token eviction policy for the write-side flow model: none, prefill, flush, or decode."""
    kind: str = "none"
    keep: float = 1.0
    cluster: int = 1
    compact: float = 0.0

    def __post_init__(self):
        if self.kind not in EVICT_POLICIES:
            raise ValueError(f"evict kind must be one of {', '.join(EVICT_POLICIES)}, got {self.kind!r}")
        if not 0.0 <= self.keep <= 1.0:
            raise ValueError(f"evict keep must be in [0, 1], got {self.keep}")
        if type(self.cluster) is not int or self.cluster < 1:
            raise ValueError(f"evict cluster must be an integer >= 1, got {self.cluster}")
        if not 0.0 <= self.compact < 1.0:
            raise ValueError(f"evict compact threshold must be in [0, 1), got {self.compact}")


def parse_evict(evict="none"):
    """Parse ``none``, ``prefill[:keep=s]``, ``flush[:keep=s]``, or ``decode[:keep=s,cluster=c,compact=theta]``."""
    if isinstance(evict, EvictPolicy):
        return evict
    if not isinstance(evict, str) or not evict:
        raise ValueError(f"Invalid eviction policy {evict!r}")
    kind, _, rest = evict.partition(":")
    if kind not in EVICT_POLICIES:
        raise ValueError(f"Unknown eviction policy {kind!r}; choices: {', '.join(EVICT_POLICIES)}")
    given = parse_params(rest)
    if "c" in given:
        given["cluster"] = given.pop("c")
    if "theta" in given:
        given["compact"] = given.pop("theta")
    allowed = {"none": set(), "prefill": {"keep"}, "flush": {"keep"}, "decode": {"keep", "cluster", "compact"}}[kind]
    if set(given) - allowed:
        raise ValueError(f"Evict policy {kind} takes {sorted(allowed) or 'no parameters'}, not "
                         f"{sorted(set(given) - allowed)}")
    defaults = {"none": {}, "prefill": {"keep": 1.0}, "flush": {"keep": 1.0},
                "decode": {"keep": 1.0, "cluster": 1, "compact": 0.0}}[kind]
    params = defaults | given
    return EvictPolicy(kind, float(params.get("keep", 1.0)), int(params.get("cluster", 1)),
                       float(params.get("compact", 0.0)))


def _raw_bounds(window, ctx):
    """Unclamped intersection helpers: returns (lo, hi) each clipped to [0, max(ctx, 0)]."""
    np = __import__("numpy")
    ctx = np.asarray(ctx, dtype=np.int64)
    upper = np.maximum(ctx, 0)
    if window is None:
        return np.zeros_like(upper), upper
    lo_s, hi_s = window
    lo = np.zeros_like(ctx) if lo_s is None else (ctx + lo_s if lo_s < 0 else np.full_like(ctx, lo_s))
    hi = ctx.copy() if hi_s is None else (ctx + hi_s if hi_s < 0 else np.full_like(ctx, hi_s))
    return np.clip(lo, 0, upper), np.clip(hi, 0, upper)


def _combined_bounds(spec_window, group_window, ctx):
    """Dynamic intersection of a tensor window and a group window at context array ``ctx``."""
    np = __import__("numpy")
    lo_a, hi_a = _raw_bounds(spec_window, ctx)
    if group_window is None:
        return np.minimum(lo_a, hi_a), hi_a
    lo_b, hi_b = _raw_bounds(group_window, ctx)
    raw_hi = np.minimum(hi_a, hi_b)
    lo = np.maximum(lo_a, lo_b)
    return np.minimum(lo, raw_hi), raw_hi


def _is_sliding(spec_window, group_window):
    """True when either window's lower bound slides forward with context (negative offset)."""
    return any(w is not None and w[0] is not None and w[0] < 0 for w in (spec_window, group_window))


def _union_interval_lengths(intervals):
    """Total union length across a list of ``(lo, hi)`` int64 arrays (elementwise over requests)."""
    np = __import__("numpy")
    if not intervals:
        return None
    if len(intervals) == 1:
        return intervals[0][1] - intervals[0][0]
    n = len(intervals[0][0])
    total = np.zeros(n, dtype=np.int64)
    for idx in range(n):
        spans = sorted((int(lo[idx]), int(hi[idx])) for lo, hi in intervals if hi[idx] > lo[idx])
        if not spans:
            continue
        cur_lo, cur_hi = spans[0]
        acc = 0
        for lo, hi in spans[1:]:
            if lo <= cur_hi:
                cur_hi = max(cur_hi, hi)
            else:
                acc += cur_hi - cur_lo
                cur_lo, cur_hi = lo, hi
        total[idx] = acc + (cur_hi - cur_lo)
    return total


@dataclass(frozen=True)
class TensorFlow:
    """Write-side token and byte flow of one HBF tensor entry over a request."""
    name: str
    owners: int
    heads: int
    token_bytes: int
    tpp: int
    admission_tokens: float
    decode_tokens: float
    died_tokens: float
    rewritten_tokens: float
    stored_tokens: float
    live_page_frac: float
    admission_bytes: float
    decode_bytes: float
    died_bytes: float
    rewritten_bytes: float
    stored_bytes: float


@dataclass(frozen=True)
class TokenFlow:
    """Aggregate write-side token and byte flow across all HBF tensors of a spec."""
    prompt: int
    generated: int
    evict: EvictPolicy
    tensors: tuple
    admission_tokens: float
    decode_tokens: float
    died_tokens: float
    rewritten_tokens: float
    stored_tokens: float
    admission_bytes: float
    decode_bytes: float
    died_bytes: float
    rewritten_bytes: float
    stored_bytes: float
    live_page_frac: float

    @property
    def decode_bytes_per_token(self):
        return self.decode_bytes / self.generated if self.generated > 0 else 0.0

    @property
    def total_written_bytes(self):
        return self.admission_bytes + self.decode_bytes + self.rewritten_bytes

    @property
    def by_tensor(self):
        return {entry.name: entry for entry in self.tensors}


def _flow_entries(spec, heads, tensors):
    """Normalize a ``KVSpec``, single ``TensorSpec``, or sequence of ``TensorSpec``s into flow entries."""
    if isinstance(spec, TensorSpec):
        return [(spec.name, spec, None, spec.tier, 1, heads or 1)]
    if isinstance(spec, (tuple, list)) and spec and all(isinstance(s, TensorSpec) for s in spec):
        return [(s.name, s, None, s.tier, 1, heads or 1) for s in spec]
    if not isinstance(spec, KVSpec):
        raise TypeError(f"flow() expects a KVSpec, TensorSpec, or sequence of TensorSpecs, got {type(spec).__name__}")
    entries = []
    for group in spec.groups:
        group_tensors = group.tensors if group.tensors is not None else tensors
        if group_tensors is None:
            raise ValueError(f"Group {group.name} has no tensors; pass tensors= or build the spec with from_model")
        owners = len(group.layers) - len(group.alias)
        g_heads = (group.heads[1] - group.heads[0]) if group.heads is not None else (heads or 1)
        g_win = (0, group.budget) if group.budget is not None else group.window
        for base_spec in group_tensors:
            tier = group.tier or base_spec.tier
            entries.append((f"{group.name}_{base_spec.name}", base_spec, g_win, tier, owners, g_heads))
    return entries


def flow(spec, prompt, generated, evict="none", *, heads=None, tensors=None, page_bytes=4096):
    """Write-side token and byte flow (admission, decode, died, rewritten, stored) on HBF.

    Supports sliding windows, age-based re-quantization ladders (multiple windowed groups/tensors),
    and eviction policies ``none``, ``prefill``, ``flush``, and ``decode(keep, cluster, compact)``.
    """
    if prompt < 0 or generated < 0:
        raise ValueError("prompt and generated must be non-negative")
    policy = parse_evict(evict)
    end = prompt + generated
    out = []
    for name, base_spec, g_win, tier, owners, g_heads in _flow_entries(spec, heads, tensors):
        if tier != "hbf" or owners <= 0 or g_heads <= 0:
            continue
        tpp = page_bytes // base_spec.token_bytes
        if tpp < 1:
            raise ValueError(f"Tensor {name} row ({base_spec.token_bytes} B) exceeds page size ({page_bytes} B)")
        lo_p_a, hi_p_a = _combined_bounds(base_spec.window, g_win, [prompt])
        lo_e_a, hi_e_a = _combined_bounds(base_spec.window, g_win, [end])
        lo_p, hi_p = int(lo_p_a[0]), int(hi_p_a[0])
        lo_e, hi_e = int(lo_e_a[0]), int(hi_e_a[0])
        sliding = _is_sliding(base_spec.window, g_win)
        base_admit = float(hi_p - lo_p)
        if sliding:
            base_decode = float(max(0, hi_e - hi_p))
            base_died = float(max(0, lo_e - lo_p))
        else:
            base_decode = float(max(0, (hi_e - lo_e) - (hi_p - lo_p)))
            base_died = 0.0
        rewritten = 0.0
        lp_frac = 1.0
        if policy.kind == "none":
            admit, dec, died = base_admit, base_decode, base_died
            stored = float(hi_e - lo_e)
        elif policy.kind == "prefill":
            s = policy.keep
            admit = base_admit * s
            dec = base_decode
            if sliding:
                prompt_died = max(0, min(lo_e, hi_p) - lo_p) * s
                decode_died = float(max(0, lo_e - max(lo_p, hi_p)))
                died = prompt_died + decode_died
            else:
                died = 0.0
            stored = admit + dec - died
        elif policy.kind == "flush":
            s = policy.keep
            admit, dec, died = base_admit * s, base_decode * s, base_died * s
            stored = float(hi_e - lo_e) * s
        else:  # decode eviction (H2O, TOVA)
            s, theta = policy.keep, policy.compact
            admit, dec = base_admit, base_decode
            surviving = float(hi_e - lo_e)
            evicted = surviving * (1.0 - s)
            died = base_died + evicted
            stored = surviving * s
            if theta > 0.0 and (1.0 - s) >= theta:
                rewritten = evicted * (1.0 - theta) / theta
                lp_frac = s
            else:
                lp_frac = live_page_fraction(s, tpp, policy.cluster)
        scale = owners * g_heads * base_spec.token_bytes
        out.append(TensorFlow(
            name=name, owners=owners, heads=g_heads, token_bytes=base_spec.token_bytes, tpp=tpp,
            admission_tokens=admit, decode_tokens=dec, died_tokens=died, rewritten_tokens=rewritten,
            stored_tokens=stored, live_page_frac=lp_frac,
            admission_bytes=admit * scale, decode_bytes=dec * scale, died_bytes=died * scale,
            rewritten_bytes=rewritten * scale, stored_bytes=stored * scale,
        ))
    if not out:
        raise ValueError("No HBF tensors in spec")
    admit_tok = sum(e.owners * e.heads * e.admission_tokens for e in out)
    dec_tok = sum(e.owners * e.heads * e.decode_tokens for e in out)
    died_tok = sum(e.owners * e.heads * e.died_tokens for e in out)
    rew_tok = sum(e.owners * e.heads * e.rewritten_tokens for e in out)
    stored_tok = sum(e.owners * e.heads * e.stored_tokens for e in out)
    admit_b = sum(e.admission_bytes for e in out)
    dec_b = sum(e.decode_bytes for e in out)
    died_b = sum(e.died_bytes for e in out)
    rew_b = sum(e.rewritten_bytes for e in out)
    stored_b = sum(e.stored_bytes for e in out)
    weight_sum = sum(e.owners * e.heads * (e.admission_tokens + e.decode_tokens) * e.token_bytes for e in out)
    if weight_sum > 0:
        lp_avg = sum(e.live_page_frac * e.owners * e.heads * (e.admission_tokens + e.decode_tokens) * e.token_bytes
                     for e in out) / weight_sum
    else:
        lp_avg = sum(e.live_page_frac for e in out) / len(out)
    return TokenFlow(
        prompt=prompt, generated=generated, evict=policy, tensors=tuple(out),
        admission_tokens=admit_tok, decode_tokens=dec_tok, died_tokens=died_tok,
        rewritten_tokens=rew_tok, stored_tokens=stored_tok,
        admission_bytes=admit_b, decode_bytes=dec_b, died_bytes=died_b,
        rewritten_bytes=rew_b, stored_bytes=stored_b, live_page_frac=lp_avg,
    )


@dataclass(frozen=True)
class _GeoEntry:
    name: str
    spec: TensorSpec
    group_window: tuple | None
    tier: str
    layers: tuple
    stored_layers: tuple
    consumer_layers: frozenset
    head_range: tuple
    heads: int
    reuse: str
    peak_copies: int
    tpp: int
    sliding: bool


class KVGeometry:
    """Duck type of ``endurance.Geometry`` for a ``KVSpec`` with layer/head groups and aliases.

    Owner layers store and write pages; consumer layers add HBF read pressure (``reuse="hbf"``,
    raising ``read_multiplier`` by the sharing factor ``s``) or HBM staging reads/footprint
    (``reuse="hbm"``). Exposes ``written_pages``, ``dead_pages``, ``layer_row_pages``,
    ``per_request_layer_row_pages``, and ``layer_demand`` for ``hbf.endurance``.
    """

    def __init__(self, spec, layers, heads, tensors=None, page_bytes=4096):
        self.spec = spec
        self.layers, self.heads, self.page_bytes = layers, heads, page_bytes
        self._default_tensors = tuple(tensors) if tensors is not None else None
        entries = []
        read_mult = 1.0
        for group in spec.groups:
            if max(group.layers) >= layers:
                raise ValueError(f"Group {group.name} names layer {max(group.layers)} of a {layers}-layer model")
            head_range = group.heads or (0, heads)
            lo_h, hi_h = head_range
            if not 0 <= lo_h < hi_h <= heads:
                raise ValueError(f"Group {group.name}: heads [{lo_h}, {hi_h}) outside {heads} heads")
            g_heads = hi_h - lo_h
            group_tensors = group.tensors if group.tensors is not None else self._default_tensors
            if group_tensors is None:
                raise ValueError(f"Group {group.name} has no tensors; pass tensors= or build the spec with from_model")
            owner_map = dict(group.alias)
            stored_layers = tuple(l for l in group.layers if l not in owner_map)
            consumer_layers = frozenset(owner_map)
            group_window = (0, group.budget) if group.budget is not None else group.window
            peak_copies = _peak_live_owners(group.alias) if (group.reuse == "hbm" and consumer_layers) else 0
            for base_spec in group_tensors:
                tier = group.tier or base_spec.tier
                tpp = page_bytes // base_spec.token_bytes if tier == "hbf" else 0
                if tier == "hbf" and tpp < 1:
                    raise ValueError(f"Tensor {group.name}_{base_spec.name} row exceeds {page_bytes} B page")
                if tier == "hbf" and group.reuse == "hbf" and owner_map and stored_layers:
                    max_r = max(1 + sum(1 for _, ow in group.alias if ow == o) for o in stored_layers)
                    read_mult = max(read_mult, float(max_r))
                entries.append(_GeoEntry(
                    name=f"{group.name}_{base_spec.name}", spec=base_spec, group_window=group_window,
                    tier=tier, layers=group.layers, stored_layers=stored_layers,
                    consumer_layers=consumer_layers, head_range=head_range, heads=g_heads,
                    reuse=group.reuse, peak_copies=peak_copies, tpp=tpp,
                    sliding=_is_sliding(base_spec.window, group_window),
                ))
        self._entries = tuple(entries)
        self._hbf_entries = tuple(e for e in entries if e.tier == "hbf")
        self._hbm_entries = tuple(e for e in entries if e.tier == "hbm")
        if not self._hbf_entries:
            raise ValueError("At least one tensor must be stored on HBF")
        self.tensors = tuple(e.spec for e in entries)
        self.hbf = tuple(e.spec for e in self._hbf_entries)
        self.hbm = tuple(e.spec for e in self._hbm_entries)
        self.tpp = tuple(e.tpp for e in self._hbf_entries)
        self.data = tuple(e.spec for e in entries if e.spec.role == "data")
        self.read_multiplier = read_mult
        self.is_uniform = (
            len(spec.groups) == 1
            and tuple(spec.groups[0].layers) == tuple(range(layers))
            and (spec.groups[0].heads is None or spec.groups[0].heads == (0, heads))
            and not spec.groups[0].alias
        )
        self._layer_entries = tuple(tuple(e for e in entries if l in e.layers) for l in range(layers))

    @property
    def token_bytes(self):
        bpt = self.spec.bytes_per_token(self.heads, self._default_tensors, include_windowed=False)
        if bpt > 0:
            return bpt
        return self.spec.bytes_per_token(self.heads, self._default_tensors, include_windowed=True)

    @property
    def hbf_token_bytes(self):
        bpt = self.spec.bytes_per_token(self.heads, self._default_tensors, tier="hbf", include_windowed=False)
        if bpt > 0:
            return bpt
        return self.spec.bytes_per_token(self.heads, self._default_tensors, tier="hbf", include_windowed=True)

    def stored_pages(self, ctx):
        """Full HBF pages stored across owner layers for each context length in ``ctx``."""
        np = __import__("numpy")
        ctx = np.asarray(ctx, dtype=np.int64)
        total = np.zeros_like(ctx)
        for e in self._hbf_entries:
            lo, hi = _combined_bounds(e.spec.window, e.group_window, ctx)
            total += len(e.stored_layers) * e.heads * ((hi - lo) // e.tpp)
        return total

    def written_pages(self, c0, c1):
        """Full HBF pages written across owner layers as context grows from ``c0`` to ``c1`` (``c1 >= c0``)."""
        np = __import__("numpy")
        c0 = np.asarray(c0, dtype=np.int64)
        c1 = np.asarray(c1, dtype=np.int64)
        total = np.zeros(np.broadcast_shapes(c0.shape, c1.shape), dtype=np.int64)
        for e in self._hbf_entries:
            lo0, hi0 = _combined_bounds(e.spec.window, e.group_window, c0)
            lo1, hi1 = _combined_bounds(e.spec.window, e.group_window, c1)
            if e.sliding:
                admit = (hi1 - lo1) // e.tpp
                grow = np.maximum(0, hi1 // e.tpp - hi0 // e.tpp)
                pages = np.where(c0 <= 0, admit, grow)
            else:
                pages = np.maximum(0, (hi1 - lo1) // e.tpp - (hi0 - lo0) // e.tpp)
            total += len(e.stored_layers) * e.heads * pages
        return total

    def dead_pages(self, c0, c1):
        """Full HBF pages that die (slide out of window) across owner layers as context grows from ``c0`` to ``c1``."""
        np = __import__("numpy")
        c0 = np.asarray(c0, dtype=np.int64)
        c1 = np.asarray(c1, dtype=np.int64)
        total = np.zeros(np.broadcast_shapes(c0.shape, c1.shape), dtype=np.int64)
        for e in self._hbf_entries:
            if not e.sliding:
                continue
            lo0, _ = _combined_bounds(e.spec.window, e.group_window, c0)
            lo1, _ = _combined_bounds(e.spec.window, e.group_window, c1)
            died = np.where(c0 <= 0, 0, np.maximum(0, lo1 // e.tpp - lo0 // e.tpp))
            total += len(e.stored_layers) * e.heads * died
        return total

    def per_request_layer_row_pages(self, ctx, role_share=None):
        """HBF pages sensed per (layer, request) in one step (ceil), shape ``(layers, len(ctx))``."""
        np = __import__("numpy")
        ctx = np.asarray(ctx, dtype=np.int64)
        out = np.zeros((self.layers, len(ctx)), dtype=float)
        for e in self._hbf_entries:
            read_layers = e.layers if e.reuse == "hbf" else e.stored_layers
            if not read_layers:
                continue
            lo, hi = _combined_bounds(e.spec.window, e.group_window, ctx)
            share_ = 1.0 if role_share is None else role_share.get(e.spec.role, 1.0)
            pages = (-(-(hi - lo) // e.tpp)) * (share_ * e.heads)
            for l in read_layers:
                out[l] += pages
        return out

    def layer_row_pages(self, ctx, role_share=None):
        """HBF pages sensed per layer in one step (ceil), summed over requests, shape ``(layers,)``."""
        return self.per_request_layer_row_pages(ctx, role_share).sum(axis=1)

    def per_request_row_pages(self, ctx, role_share=None):
        """Mean HBF pages sensed per layer for each request in ``ctx``."""
        return self.per_request_layer_row_pages(ctx, role_share).mean(axis=0)

    def row_pages(self, ctx, role_share=None):
        """Mean HBF pages sensed per layer, summed over requests."""
        return float(self.per_request_row_pages(ctx, role_share).sum())

    def hbm_bytes(self, ctx):
        """HBM-tier KV bytes per request (permanent HBM tensors plus peak live ``reuse="hbm"`` copies)."""
        np = __import__("numpy")
        ctx = np.asarray(ctx, dtype=np.int64)
        total = np.zeros_like(ctx)
        for e in self._entries:
            if e.tier == "hbm":
                lo, hi = _combined_bounds(e.spec.window, e.group_window, ctx)
                total += len(e.stored_layers) * e.heads * (hi - lo) * e.spec.token_bytes
            elif e.peak_copies:
                lo, hi = _combined_bounds(e.spec.window, e.group_window, ctx)
                total += e.peak_copies * e.heads * (hi - lo) * e.spec.token_bytes
        return total

    def partial_pages(self):
        """Pages of HBM staging per running request: one partial page per (owner layer, head, HBF tensor)."""
        return sum(len(e.stored_layers) * e.heads for e in self._hbf_entries)

    def layer_demand(self, ctx, hbm_tokens=None):
        """Per-layer demand list of one dense step over requests with ``ctx`` tokens (phase A columns)."""
        np = __import__("numpy")
        ctx = np.asarray(ctx, dtype=np.int64)
        extra = np.zeros_like(ctx) if hbm_tokens is None else np.asarray(hbm_tokens, dtype=np.int64)
        extra_sum = float(np.sum(extra))
        # Cache bounds per (spec_window, group_window) so repeated layers don't recompute arrays.
        bounds_cache = {}

        def bounds_of(e):
            key = (e.spec.window, e.group_window)
            hit = bounds_cache.get(key)
            if hit is None:
                hit = bounds_cache[key] = _combined_bounds(e.spec.window, e.group_window, ctx)
            return hit

        out = []
        for l in range(self.layers):
            burst = hbm = elements = hbf_elements = dequant = 0.0
            by_heads = {}
            for e in self._layer_entries[l]:
                lo, hi = bounds_of(e)
                tokens = hi - lo
                tok_sum = float(np.sum(tokens))
                on_hbf = e.tier == "hbf" and not (l in e.consumer_layers and e.reuse == "hbm")
                if on_hbf:
                    full_burst = -(-(e.tpp * e.spec.token_bytes) // 64) * 64
                    full = tokens // e.tpp
                    rem = tokens - full * e.tpp
                    burst += e.heads * float(np.sum(full * full_burst + -(-(rem * e.spec.token_bytes) // 64) * 64))
                    if e.spec.role == "data":
                        hbf_elements += e.heads * tok_sum * e.spec.logical
                    if extra_sum:
                        hbm += e.heads * extra_sum * e.spec.token_bytes
                else:
                    hbm += e.heads * tok_sum * e.spec.token_bytes
                if e.spec.attended_dims:
                    elements += e.heads * tok_sum * e.spec.attended_dims
                if e.spec.dequant_ops:
                    dequant += e.heads * tok_sum * e.spec.logical * e.spec.dequant_ops
                if e.spec.role == "data":
                    by_heads.setdefault(e.head_range, []).append((lo, hi))
            attended = 0.0
            for (lo_h, hi_h), intervals in by_heads.items():
                union = _union_interval_lengths(intervals)
                if union is not None:
                    attended += (hi_h - lo_h) * float(np.sum(union))
            out.append({"burst_bytes": burst, "hbm_bytes": hbm, "attended_tokens": attended,
                        "attended_elements": elements, "hbf_elements": hbf_elements, "dequant_ops": dequant})
        return out

    def demand(self, ctx, hbm_tokens=None):
        """Mean per-layer demand across all layers of one dense step."""
        per_layer = self.layer_demand(ctx, hbm_tokens)
        keys = ("burst_bytes", "hbm_bytes", "attended_tokens", "attended_elements", "hbf_elements", "dequant_ops")
        return {k: sum(row[k] for row in per_layer) / self.layers for k in keys}

    def region_span(self, slots, prompt, pages_per_block=256):
        """Pages a ``layer_major`` wave writes under ``GroupedLayout`` (striped)."""
        from ...device.config import load
        from ...frontend.kv.tensors import Shape
        cfg = load("ocp", pages_per_block=pages_per_block, blocks_per_plane=max(1, 131072 // pages_per_block))
        head_dim = max((s.dims for s in self.tensors), default=128)
        lay = GroupedLayout(cfg, Shape(self.layers, self.heads * slots, head_dim, int(prompt)),
                            self.spec, self._default_tensors or (), requests=slots, intra="striped")
        return float(lay.span)
