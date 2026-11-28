"""Built-in extension ``hbf.ext.kv_terms``: MLA/Loki/SparQ formats, kvspec groups, K4/K5/K6/K7 hooks."""
from .formats import mla_format, register_formats
from .kvspec import (GROUP_MODES, GroupedLayout, KVGeometry, KVGroup, KVSpec, EvictPolicy,
                     flow, from_model, generate_grouped, layer_head_specs, live_page_fraction,
                     parse_evict, parse_kvspec, resolve_format, spec_meta)
from .placement import register_placement
from .selection import generate_channel_select, register_selection, select_blocks


class _KVSpecHook:
    GROUP_MODES = GROUP_MODES
    KVSpec = KVSpec
    GroupedLayout = GroupedLayout
    parse_kvspec = staticmethod(parse_kvspec)
    from_model = staticmethod(from_model)
    resolve_format = staticmethod(resolve_format)
    spec_meta = staticmethod(spec_meta)
    generate_grouped = staticmethod(generate_grouped)

    @staticmethod
    def geometry_for(model, kv_format, page_bytes=4096, *, kv_spec=None, context=32768, local_tier="hbm"):
        fmt = resolve_format(model, kv_format or "bf16")
        if kv_spec is None:
            spec = from_model(model, fmt, local_tier=local_tier)
        elif isinstance(kv_spec, KVSpec):
            spec = kv_spec
        else:
            spec = parse_kvspec(kv_spec, model.layers, model.kv_heads, context)
        return spec.geometry(model.layers, model.kv_heads, fmt.tensors, page_bytes), fmt


class _GroupedAccessHook:
    supported_modes = GROUP_MODES

    @staticmethod
    def is_grouped(layout):
        return isinstance(layout, GroupedLayout)

    layer_head_specs = staticmethod(layer_head_specs)


def register(registry):
    register_formats(registry)
    register_selection(registry)
    register_placement(registry)
    registry.register_hook("kvspec", _KVSpecHook())
    registry.register_hook("grouped_access", _GroupedAccessHook())
