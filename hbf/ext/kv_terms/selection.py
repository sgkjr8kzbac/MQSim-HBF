"""K5 selection realism (density, select_static, reuse, query_groups, full_tensors) and K6 channel_select."""
import numpy as np

from ...frontend.trace.synthetic import share
from ...frontend.trace.tokens import Access


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


def select_blocks(rng, blocks, pick_n, step, key, prev, select_static, reuse):
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


def generate_channel_select(cfg, layout, *, rng, steps, eff_fraction, select_tokens,
                            select_static, reuse, full, channels, channel_groups):
    """K6 channel_select mode (Loki / SparQ partial-dimension scoring + exact-phase reads)."""
    shape = layout.shape
    windows = layout.windows
    always = [spec for spec in layout.all_tensors if spec.role == "meta" or spec.tier == "hbm" or spec.name in full]

    def whole(step, layer, head, spec):
        return Access(step, layer, head, spec.name, "R", *windows[spec.name], spec.bits)

    def clipped(step, layer, head, spec, lo, hi, op="R"):
        lo, hi = max(lo, windows[spec.name][0]), min(hi, windows[spec.name][1])
        return [Access(step, layer, head, spec.name, op, lo, hi, spec.bits)] if lo < hi else []

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
                picked, reused = select_blocks(rng, blocks, pick_n, step, (layer, head), prev,
                                               select_static, reuse)
                for block in picked:
                    lo, hi = block * select_tokens, min((block + 1) * select_tokens, shape.context)
                    op = "H" if block in reused else "R"
                    for spec in exact_tensors:
                        yield from clipped(step, layer, head, spec, lo, hi, op)
                for spec in always:
                    yield whole(step, layer, head, spec)


def map_density_coords(layout, specs, g, lo, hi):
    """Map logical token ranges [lo, hi) into compacted coordinates for ``density < 1``."""
    w_lo = np.asarray([layout.windows[spec.name][0] for spec in specs], dtype=np.int64)[g]
    num = np.asarray([spec.density.numerator for spec in specs], dtype=np.int64)[g]
    den = np.asarray([spec.density.denominator for spec in specs], dtype=np.int64)[g]
    c_lo = ((lo - w_lo) * num) // den
    c_hi = ((hi - 1 - w_lo) * num) // den + 1
    return w_lo + c_lo, w_lo + c_hi, c_lo, c_hi


def enrich_generation(kv_format, mode, generation):
    """Auto-populate format-driven K6 generation defaults (Loki full_tensors, SparQ channel_groups)."""
    if kv_format is None:
        return
    if mode in ("uniform_tokens", "uniform_pages", "sink_recent", "channel_select") and kv_format.full_tensors:
        generation.setdefault("full_tensors", list(kv_format.full_tensors))
    if mode == "channel_select" and kv_format.channel_groups:
        generation.setdefault("channel_groups", list(kv_format.channel_groups))
        if kv_format.channel_select_channels is not None:
            generation.setdefault("channels", kv_format.channel_select_channels)


class _SelectionHook:
    select_blocks = staticmethod(select_blocks)
    map_density_coords = staticmethod(map_density_coords)
    enrich_generation = staticmethod(enrich_generation)


def register_selection(registry):
    registry.register_mode("channel_select", generate_channel_select)
    registry.register_hook("selection_k5", _SelectionHook())
