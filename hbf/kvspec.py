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

from .access import Access, Layout, TensorSpec, share

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
    from .formats import KVFormat, parse_format, parse_params as parse_fmt_params
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
        from .model import load_model
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
