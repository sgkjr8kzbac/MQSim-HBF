"""Seeded synthetic access generators (dense, sparse selection, stage-0 probes, registered modes)."""
from math import ceil
from random import Random

from .tokens import Access

CORE_MODES = (
    "dense", "uniform_pages", "uniform_tokens", "sink_recent", "same_unit", "distinct_units",
    "same_channel_distinct_units", "hot_unit", "write_then_read",
)
MODES = CORE_MODES


def share(count, fraction):
    """ceil(count * fraction), ignoring float noise (100 * 0.07 is 7.000000000000001)."""
    return ceil(round(count * fraction, 9))


def _core_select_blocks(rng, blocks, pick_n):
    return sorted(rng.sample(range(blocks), pick_n)), ()


def generate(cfg, layout, mode, *, steps=1, seed=1, fraction=.1, count=4, unit=0,
             hot_fraction=.5, hot_units=1, select_tokens=16,
             select_static=False, reuse=0.0, query_groups=1, full_tensors=(),
             channels=None, channel_groups=()):
    """Seeded accesses of one workload mode (built-in or extension-registered)."""
    from ...registry import REGISTRY, ensure_loaded
    ensure_loaded()
    if mode not in REGISTRY.modes:
        if mode in ("channel_select",):
            raise REGISTRY.missing_error("mode", mode)
        raise ValueError("Invalid generator parameters")
    shape = layout.shape
    if (steps <= 0 or not 0 < fraction <= 1 or count <= 0
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

    has_k5 = bool(select_static or reuse or query_groups != 1 or full)
    k5 = REGISTRY.get_hook("selection_k5", required=has_k5) if has_k5 else None
    rng = Random(seed)
    eff_fraction = fraction if query_groups == 1 else 1.0 - (1.0 - fraction) ** query_groups

    ext_handler = REGISTRY.modes.get(mode)
    if ext_handler is not None:
        yield from ext_handler(
            cfg, layout, rng=rng, steps=steps, eff_fraction=eff_fraction,
            select_tokens=select_tokens, select_static=select_static, reuse=reuse,
            full=full, channels=channels, channel_groups=channel_groups,
        )
        return

    windows = layout.windows
    always = [spec for spec in layout.all_tensors if spec.role == "meta" or spec.tier == "hbm" or spec.name in full]

    def pick_blocks(blocks, pick_n, step, key, prev):
        if k5 is not None:
            return k5.select_blocks(rng, blocks, pick_n, step, key, prev, select_static, reuse)
        return _core_select_blocks(rng, blocks, pick_n)

    def whole(step, layer, head, spec):
        return Access(step, layer, head, spec.name, "R", *windows[spec.name], spec.bits)

    def clipped(step, layer, head, spec, lo, hi, op="R"):
        lo, hi = max(lo, windows[spec.name][0]), min(hi, windows[spec.name][1])
        return [Access(step, layer, head, spec.name, op, lo, hi, spec.bits)] if lo < hi else []

    if mode == "uniform_tokens":
        blocks = -(-shape.context // select_tokens)
        pick_n = share(blocks, eff_fraction)
        selective = [spec for spec in layout.tensors if spec.role != "meta" and spec.name not in full]
        prev = {}
        for step in range(steps):
            for layer in range(shape.layers):
                for head in range(shape.kv_heads):
                    picked, reused = pick_blocks(blocks, pick_n, step, (layer, head), prev)
                    for block in picked:
                        lo, hi = block * select_tokens, min((block + 1) * select_tokens, shape.context)
                        op = "H" if block in reused else "R"
                        for spec in selective:
                            yield from clipped(step, layer, head, spec, lo, hi, op)
                    for spec in always:
                        yield whole(step, layer, head, spec)
        return
    if mode == "sink_recent":
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
                        picked, reused = pick_blocks(blocks, pick_n, step, (layer, head, t), prev)
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
