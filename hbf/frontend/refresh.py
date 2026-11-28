"""Frontend read-disturb and retention refresh insertion on an AddressTrace.

Tracks per-block read counts and elapsed time across ``AddressTrace`` chunks and inserts
OCP HBF v0.7.0 §5.4.1 / §11.5 legal refresh operations:
- ``relocate=False`` (in-place replay): read live pages (``op='R', cls='refresh'``), release
  old pages (``op='D', cls='refresh'``), rewrite sequentially from page 0 (``op='W', cls='relocate'``),
  which triggers ``T-HBFW2`` auto-erase on page 0.
- ``relocate=True`` (relocate to fresh block): read live pages (``op='R', cls='refresh'``),
  write sequentially to a clean destination block on the same unit (``op='W', cls='relocate'``),
  release old pages (``op='D', cls='refresh'``), and remap subsequent accesses to the moved LPAs.

The resulting ``AddressTrace`` is validated by ``addrtrace.validate(cfg, trace)``.
"""
import numpy as np

from ..device.address import block_page
from .addrtrace import AddressTrace, LayerChunk, validate


def insert_refreshes(cfg, trace: AddressTrace, *, rd_threshold=None, retention_ns=None,
                     step_ns=None, relocate=False, spare_block_start=None):
    """Return ``(refreshed_trace, summary)`` after inserting triggered read-disturb/retention refreshes."""
    if rd_threshold is not None and rd_threshold < 1:
        raise ValueError("rd_threshold must be >= 1")
    if retention_ns is not None and (retention_ns <= 0 or step_ns is None or step_ns <= 0):
        raise ValueError("retention_ns requires positive retention_ns and step_ns")

    ppb = cfg.pages_per_block
    cap = cfg.capacity_sectors // cfg.spp
    resident = tuple((int(lo), int(hi)) for lo, hi in trace.meta.get("resident", ()))

    def _in_resident(lpa):
        for lo, hi in resident:
            if lo <= lpa < hi:
                return True
        return False

    # Per-block state: (unit, block) -> {"live": set[int], "reads": int, "born_ns": int}
    blocks = {}
    used_blocks_per_unit = {}

    def _mark_used(unit, block):
        used_blocks_per_unit.setdefault(unit, set()).add(block)

    # Seed resident blocks so retention/RD tracking sees every resident page accurately.
    for lo, hi in resident:
        for lpa in range(lo, hi):
            u, b, p = block_page(cfg, lpa)
            _mark_used(u, b)
            st = blocks.setdefault((u, b), {"live": set(), "reads": 0, "born_ns": 0})
            st["live"].add(p)

    # Also scan max block used across the entire trace so spare blocks never collide with future writes.
    max_block_per_unit = {}
    for chunk in trace.chunks:
        for lpa in chunk.lpa.tolist():
            u, b, _ = block_page(cfg, lpa)
            if b > max_block_per_unit.get(u, -1):
                max_block_per_unit[u] = b
    for (u, b) in blocks:
        if b > max_block_per_unit.get(u, -1):
            max_block_per_unit[u] = b

    next_spare = {}

    def _alloc_spare(unit):
        if unit not in next_spare:
            base = (spare_block_start if spare_block_start is not None
                    else max_block_per_unit.get(unit, -1) + 1)
            next_spare[unit] = base
        b = next_spare[unit]
        while b in used_blocks_per_unit.get(unit, ()):
            b += 1
        # Verify the whole destination block fits in logical capacity.
        if cfg.unit_lpa(unit, b * ppb + ppb - 1) >= cap:
            raise ValueError(f"No spare block available on unit {unit} for refresh relocation")
        next_spare[unit] = b + 1
        _mark_used(unit, b)
        return b

    remap = {}  # old_lpa -> current_lpa (only used when relocate=True)
    new_chunks = []
    summary = {"refresh_events": 0, "rd_refreshes": 0, "retention_refreshes": 0,
               "refresh_reads": 0, "relocate_writes": 0, "releases": 0}
    limit = cfg.batch_pages

    for chunk in trace.chunks:
        now_ns = max(0, (chunk.step + 1) * step_ns) if step_ns is not None and chunk.step >= 0 else 0
        # Apply any active LPA remappings to incoming reads/releases.
        lpas = chunk.lpa.copy()
        ops = chunk.op.copy()
        if remap and len(lpas):
            for i in range(len(lpas)):
                if ops[i] in ("R", "D") and int(lpas[i]) in remap:
                    lpas[i] = remap[int(lpas[i])]

        # Update per-block state from this chunk's records.
        for op, lpa in zip(ops.tolist(), lpas.tolist()):
            u, b, p = block_page(cfg, lpa)
            _mark_used(u, b)
            st = blocks.setdefault((u, b), {"live": set(), "reads": 0, "born_ns": now_ns})
            if op == "R":
                st["reads"] += 1
            elif op == "D":
                st["live"].discard(p)
            elif op == "W":
                if p == 0:
                    st["live"] = {0}
                    st["reads"] = 0
                    st["born_ns"] = now_ns
                else:
                    st["live"].add(p)

        # Check whether any live block triggers read-disturb or retention refresh after this chunk.
        triggered = []
        if chunk.step >= 0:
            for key in sorted(blocks):
                st = blocks[key]
                if not st["live"]:
                    continue
                rd_hit = rd_threshold is not None and st["reads"] >= rd_threshold
                ret_hit = (retention_ns is not None and step_ns is not None
                           and (now_ns - st["born_ns"]) >= retention_ns)
                if rd_hit or ret_hit:
                    triggered.append((key, rd_hit, ret_hit))

        if not triggered:
            new_chunks.append(LayerChunk(
                step=chunk.step, layer=chunk.layer, subbatch=chunk.subbatch.copy(),
                request=chunk.request.copy(), op=ops, lpa=lpas,
                useful_bytes=chunk.useful_bytes.copy(), burst_bytes=chunk.burst_bytes.copy(),
                burst_runs=chunk.burst_runs.copy(), cls=chunk.cls.copy(),
                hbm_bytes=chunk.hbm_bytes, attended_tokens=chunk.attended_tokens,
                attended_elements=chunk.attended_elements, hbf_elements=chunk.hbf_elements,
                dequant_ops=chunk.dequant_ops, tensor_tokens=dict(chunk.tensor_tokens),
            ))
            continue

        # Append refresh operations as additional subbatches at the end of this (step, layer) chunk.
        next_sb = (int(chunk.subbatch.max()) + 1) if len(chunk.subbatch) else 0
        extra_sb, extra_op, extra_lpa, extra_cls = [], [], [], []

        def _append_ops(op_code, cls_name, lpa_list):
            nonlocal next_sb
            if not lpa_list:
                return
            arr = np.asarray(lpa_list, dtype=np.int64)
            sbs = next_sb + np.arange(len(arr), dtype=np.int64) // limit
            next_sb = int(sbs[-1]) + 1
            extra_sb.append(sbs)
            extra_op.append(np.full(len(arr), op_code, dtype="U1"))
            extra_lpa.append(arr)
            extra_cls.append(np.full(len(arr), cls_name, dtype="U8"))

        for (u, b), rd_hit, ret_hit in triggered:
            st = blocks[u, b]
            live_pages = sorted(st["live"])
            src_lpas = [cfg.unit_lpa(u, b * ppb + p) for p in live_pages]
            summary["refresh_events"] += 1
            summary["rd_refreshes"] += int(rd_hit)
            summary["retention_refreshes"] += int(ret_hit)
            summary["refresh_reads"] += len(src_lpas)
            summary["relocate_writes"] += len(src_lpas)
            summary["releases"] += len(src_lpas)

            if relocate:
                dst_b = _alloc_spare(u)
                dst_lpas = [cfg.unit_lpa(u, dst_b * ppb + idx) for idx in range(len(live_pages))]
                _append_ops("R", "refresh", src_lpas)
                _append_ops("W", "relocate", dst_lpas)
                _append_ops("D", "refresh", src_lpas)
                for old_lpa, new_lpa in zip(src_lpas, dst_lpas):
                    remap[old_lpa] = new_lpa
                    # Update any prior alias pointing to old_lpa so chained relocations stay valid.
                    for k, v in list(remap.items()):
                        if v == old_lpa:
                            remap[k] = new_lpa
                st["live"].clear()
                st["reads"] = 0
                blocks[u, dst_b] = {"live": set(range(len(live_pages))), "reads": 0, "born_ns": now_ns}
            else:
                # In-place replay: read live pages -> release block -> write pages 0..k-1 (auto-erase on page 0).
                dst_lpas = [cfg.unit_lpa(u, b * ppb + idx) for idx in range(len(live_pages))]
                _append_ops("R", "refresh", src_lpas)
                _append_ops("D", "refresh", src_lpas)
                _append_ops("W", "relocate", dst_lpas)
                for old_lpa, new_lpa in zip(src_lpas, dst_lpas):
                    if old_lpa != new_lpa:
                        remap[old_lpa] = new_lpa
                st["live"] = set(range(len(live_pages)))
                st["reads"] = 0
                st["born_ns"] = now_ns

        ex_sb = np.concatenate(extra_sb)
        ex_op = np.concatenate(extra_op)
        ex_lpa = np.concatenate(extra_lpa)
        ex_cls = np.concatenate(extra_cls)
        n_ex = len(ex_lpa)
        ex_req = np.zeros(n_ex, dtype=np.int64)
        ex_u = np.full(n_ex, cfg.page_bytes, dtype=np.int64)
        ex_b = np.full(n_ex, cfg.page_bytes, dtype=np.int64)
        ex_r = np.ones(n_ex, dtype=np.int64)

        if len(lpas):
            merged_sb = np.concatenate([chunk.subbatch, ex_sb])
            merged_req = np.concatenate([chunk.request, ex_req])
            merged_op = np.concatenate([ops, ex_op])
            merged_lpa = np.concatenate([lpas, ex_lpa])
            merged_u = np.concatenate([chunk.useful_bytes, ex_u])
            merged_b = np.concatenate([chunk.burst_bytes, ex_b])
            merged_r = np.concatenate([chunk.burst_runs, ex_r])
            merged_cls = np.concatenate([chunk.cls, ex_cls])
        else:
            merged_sb, merged_req, merged_op, merged_lpa = ex_sb, ex_req, ex_op, ex_lpa
            merged_u, merged_b, merged_r, merged_cls = ex_u, ex_b, ex_r, ex_cls

        new_chunks.append(LayerChunk(
            step=chunk.step, layer=chunk.layer, subbatch=merged_sb, request=merged_req,
            op=merged_op, lpa=merged_lpa, useful_bytes=merged_u, burst_bytes=merged_b,
            burst_runs=merged_r, cls=merged_cls,
            hbm_bytes=chunk.hbm_bytes, attended_tokens=chunk.attended_tokens,
            attended_elements=chunk.attended_elements, hbf_elements=chunk.hbf_elements,
            dequant_ops=chunk.dequant_ops, tensor_tokens=dict(chunk.tensor_tokens),
        ))

    out = AddressTrace(meta=dict(trace.meta), chunks=new_chunks)
    validate(cfg, out)
    return out, summary
