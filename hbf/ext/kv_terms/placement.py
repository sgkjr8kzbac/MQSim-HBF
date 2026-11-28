"""K7 placement extension hooks: grouped kv_spec compilation, class frontiers, and shared prefixes."""
from dataclasses import replace

from ...frontend.kv.placement import _TensorPlacement, _compose_window, _is_short_lived, _peak_live_owners


class _PlacementK7Hook:
    @staticmethod
    def validate_spec(spec):
        """Extra validation for K7 BatchSpec fields (already checked structurally in BatchSpec)."""
        return True

    @staticmethod
    def compile_kv_spec(shape, base_tensors, kv_spec):
        """Compile a ``KVSpec`` into flat tensors and per-tensor placement metadata."""
        flat, hbf_meta, hbm_meta, copy_owners = [], [], {}, {}
        for group in kv_spec.groups:
            if max(group.layers) >= shape.layers:
                raise ValueError(f"Group {group.name} names layer {max(group.layers)} of a {shape.layers}-layer model")
            h_lo, h_hi = group.heads or (0, shape.kv_heads)
            if not 0 <= h_lo < h_hi <= shape.kv_heads:
                raise ValueError(f"Group {group.name}: heads [{h_lo}, {h_hi}) outside the {shape.kv_heads} heads per request")
            heads = tuple(range(h_lo, h_hi))
            owner = dict(group.alias)
            stored = tuple(l for l in group.layers if l not in owner)
            group_window = (0, group.budget) if group.budget is not None else group.window
            for base_spec in group.tensors or base_tensors:
                tier = group.tier or base_spec.tier
                composed = _compose_window(base_spec.window, group_window)
                spec_ = replace(base_spec, name=f"{group.name}_{base_spec.name}", window=composed, tier=tier)
                if tier == "hbm":
                    flat.append(spec_)
                    hbm_meta[spec_.name] = (group.layers, stored, heads, base_spec.window, group_window)
                    continue
                short = _is_short_lived(base_spec.window, group_window)
                if group.reuse == "hbf" or not owner:
                    flat.append(spec_)
                    hbf_meta.append(_TensorPlacement(spec_, group.layers, stored, heads, owner,
                                                     base_spec.window, group_window, short))
                else:
                    flat.append(spec_)
                    hbf_meta.append(_TensorPlacement(spec_, stored, stored, heads, {},
                                                     base_spec.window, group_window, short))
                    copy = replace(spec_, name=f"{spec_.name}_copy", tier="hbm")
                    flat.append(copy)
                    hbm_meta[copy.name] = (tuple(owner), (), heads, base_spec.window, group_window)
                    copy_owners[copy.name] = _peak_live_owners(group.alias)
        names = [s.name for s in flat]
        if not flat or len(set(names)) != len(names):
            raise ValueError("A KV spec needs uniquely named tensors")
        all_tensors = tuple(flat)
        hbf = tuple(m.spec for m in hbf_meta)
        hbm = tuple(s for s in all_tensors if s.tier == "hbm")
        return all_tensors, hbf, hbm, tuple(hbf_meta), hbm_meta, copy_owners


def register_placement(registry):
    registry.register_hook("placement_k7", _PlacementK7Hook())
