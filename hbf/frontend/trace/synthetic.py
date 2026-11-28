"""Seeded synthetic access generators (dense, sparse selection, stage-0 probes)."""
from math import ceil
from random import Random

from .tokens import Access

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
    """Return ``(ordered_blocks, reused_set)`` for one (step, selection key)."""
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
    """Seeded accesses of one workload mode."""
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
