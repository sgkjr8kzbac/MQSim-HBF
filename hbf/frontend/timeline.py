"""Page-level request-stream timeline frontend (R6a).

Expands a serving ``Workload`` stream (closed-loop, Poisson, or scripted; single-turn or
multi-call agentic sessions; optional shared prefixes) into a step-by-step LPA-level
``AddressTrace`` suitable for both the rounds backend and MQSim (``T-HBFW1``..``T-HBFW4``).

Policy semantics intentionally mirror ``hbf.lifetime.sim`` (segment-level event simulator):
- Segment ``s`` covers LPAs ``[s * P, (s + 1) * P)`` where ``P = cfg.units * cfg.pages_per_block``,
  i.e. block ``s`` across every bank ``0 .. cfg.units - 1`` in ascending page order ``0 .. ppb - 1``.
- Prefill writes (``cls='prefill'``), decode flushes (``cls='flush'``), completions/expiries
  (``op='D', cls='kv'``), read-disturb & retention refreshes (``R`` ``cls='refresh'``,
  ``W`` ``cls='relocate'``, ``D`` ``cls='refresh'``), and greedy segment reclaim all follow
  OCP HBF v0.7.0 §5.4.1 sequential in-block write order and pass ``addrtrace.validate()``.
"""
import heapq
import math
from dataclasses import dataclass

import numpy as np

from ..device.address import block_page
from .addrtrace import AddressTrace, LayerChunk, make_meta, validate
from .trace.requests import Stream


@dataclass
class _Entry:
    eid: int
    kind: str  # "private", "prefix", "session"
    state: str = "active"  # "active", "idle", "dead"
    readers: int = 0
    ttl: float = math.inf
    stored_tokens: int = 0
    # Per-segment live LPAs owned by this entry: seg_id -> list[int]
    seg_lpas: dict = None
    # Oldest write timestamp per segment for retention tracking: seg_id -> float
    seg_born: dict = None
    # Dedicated segments owned exclusively by this entry (policy == "dedicated")
    owned_segs: list = None
    dseg: int = -1
    key: tuple | None = None

    def __post_init__(self):
        if self.seg_lpas is None:
            self.seg_lpas = {}
        if self.seg_born is None:
            self.seg_born = {}
        if self.owned_segs is None:
            self.owned_segs = []

    def all_lpas(self):
        out = []
        for s in sorted(self.seg_lpas):
            out.extend(self.seg_lpas[s])
        return out

    def total_pages(self):
        return sum(len(v) for v in self.seg_lpas.values())


@dataclass
class _Slot:
    req: object
    entry: _Entry
    prefix: _Entry | None
    ctx: int
    end: int
    staged: int = 0


def build_timeline(cfg, geometry, workload, *, steps, step_s, policy="packed",
                   x=None, retention_s=0.0, refresh="relocate", wear="dynamic",
                   align="row", prefix_reads="per_request", low_watermark=None,
                   reclaim="evict", refresh_idle="relocate", stagger=True,
                   slots=None, max_segments=None, validate_trace=True):
    """Expand ``workload`` over ``steps`` decode steps into ``(AddressTrace, stats)``.

    ``geometry`` is any object with ``layers``, ``stored_pages(np.ndarray)`` (e.g.
    ``hbf.lifetime.geometry.Geometry``). ``step_s`` is the physical duration (seconds) of one
    decode step used to advance request arrivals, session think times, and retention timers.
    """
    if policy not in ("packed", "dedicated"):
        raise ValueError(f"timeline supports policy 'packed' or 'dedicated', got {policy!r}")
    if refresh not in ("relocate", "replay"):
        raise ValueError(f"refresh must be 'relocate' or 'replay', got {refresh!r}")
    if wear not in ("dynamic", "none"):
        raise ValueError(f"wear must be 'dynamic' or 'none', got {wear!r}")
    if align not in ("row", "none"):
        raise ValueError(f"align must be 'row' or 'none', got {align!r}")
    if steps < 1 or step_s <= 0:
        raise ValueError("steps must be >= 1 and step_s > 0")

    row = cfg.units
    ppb = cfg.pages_per_block
    P = row * ppb
    cap_pages = cfg.capacity_sectors // cfg.spp
    S = cap_pages // P
    if max_segments is not None:
        S = min(S, int(max_segments))
    if S < 2:
        raise ValueError(f"Need at least 2 segments of {P} pages (capacity={cap_pages} pages)")
    layers = int(geometry.layers)
    cap_slots = slots or (workload.concurrency if workload.arrival == "closed" else workload.max_batch)
    low_wm = low_watermark if low_watermark is not None else min(2, max(1, S // 2))

    page_cache = {}

    def stored_pages(tokens):
        tokens = int(tokens)
        hit = page_cache.get(tokens)
        if hit is None:
            hit = page_cache[tokens] = int(geometry.stored_pages(np.array([tokens], dtype=np.int64))[0])
        return hit

    # Segment state
    seg_state = ["free"] * S  # "free", "open", "full"
    seg_fill = [0] * S  # next offset in [0, P]
    seg_live = [0] * S  # number of currently live pages in segment
    seg_pec = [0] * S  # opens (for wear-leveling heap key)
    seg_ever_written = [False] * S  # True once written at least once (for auto-erase counting)
    seg_owner = [-1] * S  # entry id for dedicated, -1 for shared
    seg_busy = [False] * S
    seg_padding = [[] for _ in range(S)]  # written padding LPAs awaiting release on segment free
    # Per-block read counts within each segment (length row per segment)
    seg_block_reads = [[0] * row for _ in range(S)]
    seg_max_reads = [0] * S
    # Entries occupying each segment: seg -> set[eid]
    seg_entries = [set() for _ in range(S)]

    def _heap_key(s):
        return (seg_pec[s], s) if wear == "dynamic" else (0, s)

    free_heap = [_heap_key(s) for s in range(S)]
    heapq.heapify(free_heap)
    free_set = set(range(S))
    frontier = -1

    entries = {}
    next_eid = 0
    prefix_map = {}
    session_map = {}
    protected = set()
    active_slots = []
    queue = []
    stream = Stream(workload, stagger=stagger)

    # Stats tracking (exact Python counts matching both T-HBFW4 and Simulator semantics)
    programs_by_cause = {"prefill": 0, "decode": 0, "refresh_rd": 0, "refresh_ret": 0,
                         "compaction": 0, "replay": 0, "padding": 0}
    seg_opens = 0
    seg_erases_actual = 0  # actual page-0 rewrites of previously-written segments
    completed = 0
    admitted = 0
    refresh_events_rd = 0
    refresh_events_ret = 0
    compaction_events = 0

    # Per-step operation buffers: list of (op, cls, lpa) tuples
    cur_ops = []

    def _emit(op, cls_name, lpas):
        for lpa in lpas:
            cur_ops.append((op, cls_name, int(lpa)))

    def _release_segment_if_empty(s):
        nonlocal frontier
        if s in free_set or seg_busy[s] or seg_state[s] == "open" or seg_owner[s] != -1 or seg_live[s] > 0:
            return
        if seg_padding[s]:
            _emit("D", "kv", seg_padding[s])
            seg_padding[s].clear()
        seg_state[s] = "free"
        seg_fill[s] = 0
        seg_block_reads[s] = [0] * row
        seg_max_reads[s] = 0
        free_set.add(s)
        heapq.heappush(free_heap, _heap_key(s))

    def _close_segment(s):
        nonlocal frontier
        seg_state[s] = "full"
        if frontier == s:
            frontier = -1
        o = seg_owner[s]
        if o >= 0 and o in entries and entries[o].dseg == s:
            entries[o].dseg = -1
        _release_segment_if_empty(s)

    def _take_free():
        while free_heap:
            _, s = heapq.heappop(free_heap)
            if s in free_set:
                free_set.remove(s)
                return s
        return -1

    def _open_segment(cause, owner=-1):
        nonlocal frontier, seg_opens, seg_erases_actual
        s = _take_free()
        if s < 0:
            _reclaim(force=True)
            if owner < 0 and frontier >= 0:
                return frontier
            s = _take_free()
        if s < 0:
            raise RuntimeError(f"No free segment available ({S} segments of {P} pages); increase blocks_per_plane")
        seg_pec[s] += 1
        seg_opens += 1
        if seg_ever_written[s]:
            seg_erases_actual += 1
        seg_state[s] = "open"
        seg_fill[s] = 0
        seg_owner[s] = owner
        seg_block_reads[s] = [0] * row
        seg_max_reads[s] = 0
        if owner >= 0:
            entries[owner].owned_segs.append(s)
        else:
            frontier = s
        return s

    def _write_pages(entry, n, cause, now_s, do_align=False):
        """Append ``n`` pages for ``entry`` and emit ``op='W'`` records."""
        nonlocal frontier
        if n <= 0:
            return
        cls_map = {
            "prefill": "prefill",
            "decode": "flush",
            "refresh_rd": "relocate",
            "refresh_ret": "relocate",
            "compaction": "relocate",
            "replay": "relocate",
        }
        cls_name = cls_map[cause]
        rem = int(n)
        while rem > 0:
            if policy == "dedicated":
                s = entry.dseg
                if s < 0:
                    s = _open_segment(cause, owner=entry.eid)
                    entry.dseg = s
            else:
                s = frontier
                if s >= 0 and do_align and align == "row":
                    pad = (-seg_fill[s]) % row
                    if pad > 0:
                        pad_lpas = [s * P + seg_fill[s] + i for i in range(pad)]
                        seg_fill[s] += pad
                        seg_padding[s].extend(pad_lpas)
                        programs_by_cause["padding"] += pad
                        _emit("W", "prefill", pad_lpas)
                        seg_ever_written[s] = True
                        if seg_fill[s] >= P:
                            _close_segment(s)
                            s = -1
                if s < 0:
                    s = _open_segment(cause)
                do_align = False
            k = min(rem, P - seg_fill[s])
            lpas = [s * P + seg_fill[s] + i for i in range(k)]
            seg_fill[s] += k
            seg_live[s] += k
            seg_ever_written[s] = True
            seg_entries[s].add(entry.eid)
            entry.seg_lpas.setdefault(s, []).extend(lpas)
            if s not in entry.seg_born:
                entry.seg_born[s] = now_s
            programs_by_cause[cause] += k
            _emit("W", cls_name, lpas)
            if seg_fill[s] >= P:
                _close_segment(s)
            rem -= k

    def _drop_entry_from_seg(entry, s, cls_name="kv"):
        lpas = entry.seg_lpas.pop(s, [])
        entry.seg_born.pop(s, None)
        seg_entries[s].discard(entry.eid)
        if lpas:
            _emit("D", cls_name, lpas)
            seg_live[s] -= len(lpas)
            if seg_live[s] <= 0:
                seg_live[s] = 0
                _release_segment_if_empty(s)
        return lpas

    def _kill_entry(entry):
        for s in list(entry.seg_lpas):
            _drop_entry_from_seg(entry, s, cls_name="kv")
        for s in list(entry.owned_segs):
            if seg_owner[s] == entry.eid:
                seg_owner[s] = -1
                if seg_state[s] == "open":
                    seg_state[s] = "full"
                _release_segment_if_empty(s)
        entry.owned_segs.clear()
        entry.dseg = -1
        if entry.key is not None:
            ktype, kval = entry.key
            table = prefix_map if ktype == "prefix" else session_map
            if table.get(kval) is entry:
                del table[kval]
        entry.state = "dead"
        entries.pop(entry.eid, None)

    def _relocate_segment(s, cause, now_s):
        seg_busy[s] = True
        o = seg_owner[s]
        if o >= 0 and o in entries:
            if s in entries[o].owned_segs:
                entries[o].owned_segs.remove(s)
            seg_owner[s] = -1
            if entries[o].dseg == s:
                entries[o].dseg = -1
        if seg_state[s] == "open":
            _close_segment(s)
        evict = (reclaim == "evict") if cause == "compaction" else (refresh_idle == "evict")
        moves = []
        for eid in sorted(list(seg_entries[s])):
            ent = entries.get(eid)
            if ent is None:
                continue
            if evict and ent.state == "idle" and eid not in protected:
                _kill_entry(ent)
                continue
            lpas = ent.seg_lpas.get(s, [])
            if lpas:
                _emit("R", "refresh", lpas)
                moves.append((ent, len(lpas)))
        # Drop old pages from segment s after reading them, then write relocated copies.
        for ent, _ in moves:
            _drop_entry_from_seg(ent, s, cls_name="refresh")
        # Release any remaining padding on segment s so it becomes cleanly empty.
        if seg_padding[s]:
            _emit("D", "refresh", seg_padding[s])
            seg_padding[s].clear()
        moved_ids = {ent.eid for ent, _ in moves}
        protected.update(moved_ids)
        try:
            for ent, cnt in moves:
                _write_pages(ent, cnt, cause, now_s, do_align=False)
        finally:
            protected.difference_update(moved_ids)
        seg_busy[s] = False
        seg_block_reads[s] = [0] * row
        seg_max_reads[s] = 0
        _release_segment_if_empty(s)

    def _replay_segment(s, now_s):
        """In-place replay of segment s: read live, release all, rewrite 0..fill-1 (OCP section 5.7)."""
        nonlocal seg_opens, seg_erases_actual
        old_fill = seg_fill[s]
        live_entries = []
        for eid in sorted(list(seg_entries[s])):
            ent = entries.get(eid)
            if ent is not None and ent.seg_lpas.get(s):
                live_entries.append((ent, list(ent.seg_lpas[s])))
        # 1. Read all live pages of segment s.
        for _, lpas in live_entries:
            _emit("R", "refresh", lpas)
        # 2. Release all live pages and padding on segment s so every block has 0 live pages.
        for ent, lpas in live_entries:
            _emit("D", "refresh", lpas)
            ent.seg_lpas[s] = []
        if seg_padding[s]:
            _emit("D", "refresh", seg_padding[s])
            seg_padding[s].clear()
        # 3. Rewrite sequentially starting at offset 0 (page 0 triggers T-HBFW2 auto-erase on every bank).
        #    Per OCP section 5.7, in-place replay rewrites the full written span (0..old_fill-1),
        #    so any dead slots in the segment are rewritten as padding and released when the segment frees.
        seg_pec[s] += 1
        seg_opens += 1
        if seg_ever_written[s]:
            seg_erases_actual += 1
        seg_fill[s] = 0
        seg_live[s] = 0
        for ent, lpas in live_entries:
            k = len(lpas)
            new_lpas = [s * P + seg_fill[s] + i for i in range(k)]
            seg_fill[s] += k
            seg_live[s] += k
            ent.seg_lpas[s] = new_lpas
            ent.seg_born[s] = now_s
            programs_by_cause["replay"] += k
            _emit("W", "relocate", new_lpas)
        if seg_fill[s] < old_fill:
            dead_tail = old_fill - seg_fill[s]
            pad_lpas = [s * P + seg_fill[s] + i for i in range(dead_tail)]
            seg_fill[s] = old_fill
            seg_padding[s].extend(pad_lpas)
            programs_by_cause["replay"] += dead_tail
            _emit("W", "relocate", pad_lpas)
        seg_ever_written[s] = True
        seg_block_reads[s] = [0] * row
        seg_max_reads[s] = 0

    def _reclaim(force=False):
        nonlocal compaction_events
        for _ in range(S):
            if not force and len(free_set) >= low_wm:
                break
            if force and len(free_set) >= 1:
                break
            # Greedy victim: full unowned (or idle-owned) non-busy segment with fewest live pages.
            cands = []
            for s in range(S):
                if s in free_set or seg_busy[s] or seg_live[s] <= 0:
                    continue
                owner = seg_owner[s]
                if seg_state[s] == "full" and owner == -1:
                    cands.append((seg_live[s], s))
                elif owner >= 0 and owner in entries and entries[owner].state == "idle" and owner not in protected:
                    cands.append((0, s))
            if not cands:
                break
            cands.sort()
            victim = cands[0][1]
            owner = seg_owner[victim]
            if owner >= 0 and owner in entries:
                _kill_entry(entries[owner])
            else:
                compaction_events += 1
                _relocate_segment(victim, "compaction", cur_t)

    def _record_reads(lpas):
        for lpa in lpas:
            u, b, _ = block_page(cfg, lpa)
            if 0 <= b < S:
                val = seg_block_reads[b][u] + 1
                seg_block_reads[b][u] = val
                if val > seg_max_reads[b]:
                    seg_max_reads[b] = val

    def _new_entry(kind, key=None):
        nonlocal next_eid
        eid = next_eid
        next_eid += 1
        ent = _Entry(eid=eid, kind=kind, key=key)
        entries[eid] = ent
        return ent

    def _admit_one(req, now_s):
        nonlocal admitted
        pre_tokens = (req.prefix_tokens or workload.prefix_tokens) if req.prefix >= 0 else 0
        start = pre_tokens + req.history + req.input
        end = start + req.output
        sess = session_map.get(req.session) if req.session >= 0 else None
        pre = prefix_map.get(req.prefix) if req.prefix >= 0 else None
        have = sess.stored_tokens if sess is not None else pre_tokens
        own = max(0, stored_pages(start) - stored_pages(have)) if have > 0 else stored_pages(start)
        pre_need = stored_pages(pre_tokens) if (req.prefix >= 0 and pre is None) else 0
        keep = [e.eid for e in (sess, pre) if e is not None]
        protected.update(keep)
        try:
            # Ensure free space before admitting.
            free_pages = len(free_set) * P + (P - seg_fill[frontier] if frontier >= 0 else 0)
            if free_pages < own + pre_need + row:
                _reclaim(force=True)
            if req.prefix >= 0:
                if pre is None:
                    pre = _new_entry("prefix", key=("prefix", req.prefix))
                    prefix_map[req.prefix] = pre
                    _write_pages(pre, pre_need, "prefill", now_s, do_align=True)
                    pre.stored_tokens = pre_tokens
                else:
                    # Prefill reads existing prefix pages once.
                    lpas = pre.all_lpas()
                    _emit("R", "kv", lpas)
                    _record_reads(lpas)
                pre.state = "active"
                pre.ttl = math.inf
                pre.readers += 1
            if sess is not None:
                ent = sess
                lpas = ent.all_lpas()
                _emit("R", "kv", lpas)
                _record_reads(lpas)
            else:
                ent = _new_entry("session" if req.session >= 0 else "private",
                                 key=("session", req.session) if req.session >= 0 else None)
                if req.session >= 0:
                    session_map[req.session] = ent
            ent.state = "active"
            ent.readers = 1
            ent.ttl = math.inf
            ent.stored_tokens = start
            _write_pages(ent, own, "prefill", now_s, do_align=(sess is None))
        finally:
            protected.difference_update(keep)
        active_slots.append(_Slot(req=req, entry=ent, prefix=pre, ctx=start, end=end, staged=0))
        admitted += 1

    def _admit_pending(now_s):
        queue.extend(stream.arrived(now_s))
        if workload.arrival == "closed" and not workload.sessions:
            while len(active_slots) + len(queue) < cap_slots:
                queue.append(stream.closed_new(now_s))
        while queue and len(active_slots) < cap_slots:
            req = queue.pop(0)
            _admit_one(req, now_s)

    def _flush_decode(now_s):
        if policy == "dedicated":
            for sl in active_slots:
                whole = (sl.staged // row) * row
                if whole > 0:
                    sl.staged -= whole
                    _write_pages(sl.entry, whole, "decode", now_s)
            return
        total = sum(sl.staged for sl in active_slots)
        whole = (total // row) * row
        if whole <= 0:
            return
        rem = whole
        for sl in active_slots:
            if rem <= 0:
                break
            take = min(sl.staged, rem)
            if take > 0:
                sl.staged -= take
                _write_pages(sl.entry, take, "decode", now_s)
                rem -= take

    def _complete_slot(sl, now_s):
        nonlocal completed
        if sl.staged > 0:
            _write_pages(sl.entry, sl.staged, "decode", now_s)
            sl.staged = 0
        sl.entry.stored_tokens = sl.end
        completed += 1
        if sl.prefix is not None:
            sl.prefix.readers -= 1
            if sl.prefix.readers <= 0:
                if workload.retain_s > 0:
                    sl.prefix.state = "idle"
                    sl.prefix.ttl = now_s + workload.retain_s
                else:
                    _kill_entry(sl.prefix)
        if sl.req.session >= 0 and not sl.req.last:
            sl.entry.state = "idle"
            sl.entry.readers = 0
            sl.entry.ttl = math.inf
        elif sl.req.retain_s > 0:
            sl.entry.state = "idle"
            sl.entry.readers = 0
            sl.entry.ttl = now_s + sl.req.retain_s
        else:
            _kill_entry(sl.entry)
        stream.call_done(sl.req, now_s)

    def _housekeeping(now_s):
        nonlocal refresh_events_rd, refresh_events_ret
        # 1. Idle TTL expiry
        for ent in list(entries.values()):
            if ent.state == "idle" and ent.ttl <= now_s + 1e-12:
                _kill_entry(ent)
        # 2. Read-disturb refresh
        if x is not None:
            for s in range(S):
                if seg_max_reads[s] >= x:
                    if seg_live[s] > 0:
                        refresh_events_rd += 1
                        if refresh == "replay":
                            _replay_segment(s, now_s)
                        else:
                            _relocate_segment(s, "refresh_rd", now_s)
                    elif seg_state[s] == "open" and seg_owner[s] == -1:
                        _close_segment(s)
        # 3. Retention refresh
        if retention_s > 0:
            for s in range(S):
                if seg_live[s] <= 0:
                    continue
                oldest = min((entries[eid].seg_born[s] for eid in seg_entries[s]
                              if eid in entries and s in entries[eid].seg_born), default=math.inf)
                if oldest <= now_s - retention_s + 1e-9:
                    refresh_events_ret += 1
                    if refresh == "replay":
                        _replay_segment(s, now_s)
                    else:
                        _relocate_segment(s, "refresh_ret", now_s)
        # 4. Background reclaim
        if len(free_set) < low_wm:
            _reclaim(force=False)

    def _ops_to_chunk(step_idx, layer_idx, op_groups):
        """Pack a list of (op, cls, lpa) subbatch-groups into a single ``LayerChunk``."""
        limit = cfg.batch_pages
        sb_parts, op_parts, lpa_parts, cls_parts = [], [], [], []
        next_sb = 0
        for group in op_groups:
            if not group:
                continue
            # Keep distinct (op, cls) transitions or duplicate LPAs in separate subbatches so
            # MQSim never sees duplicate (arrival_ns, lba) in a single subbatch!
            seen = set()
            buf_op, buf_cls, buf_lpa = [], [], []

            def _flush_buf():
                nonlocal next_sb
                if not buf_lpa:
                    return
                m = len(buf_lpa)
                sb_parts.append(np.full(m, next_sb, dtype=np.int64))
                op_parts.append(np.array(buf_op, dtype="U1"))
                lpa_parts.append(np.array(buf_lpa, dtype=np.int64))
                cls_parts.append(np.array(buf_cls, dtype="U8"))
                next_sb += 1
                seen.clear()
                buf_op.clear()
                buf_cls.clear()
                buf_lpa.clear()

            for op_c, cls_c, lpa_c in group:
                # Start a new subbatch on op change, capacity limit, or duplicate LPA (e.g. R+D+W).
                if buf_lpa and (op_c != buf_op[-1] or lpa_c in seen or len(buf_lpa) >= limit):
                    _flush_buf()
                buf_op.append(op_c)
                buf_cls.append(cls_c)
                buf_lpa.append(lpa_c)
                seen.add(lpa_c)
            _flush_buf()

        if not lpa_parts:
            return LayerChunk(step=step_idx, layer=layer_idx)
        sb = np.concatenate(sb_parts)
        op = np.concatenate(op_parts)
        lpa = np.concatenate(lpa_parts)
        cls_arr = np.concatenate(cls_parts)
        n = len(lpa)
        return LayerChunk(
            step=step_idx, layer=layer_idx, subbatch=sb,
            request=np.zeros(n, dtype=np.int64), op=op, lpa=lpa,
            useful_bytes=np.full(n, cfg.page_bytes, dtype=np.int64),
            burst_bytes=np.full(n, cfg.page_bytes, dtype=np.int64),
            burst_runs=np.ones(n, dtype=np.int64), cls=cls_arr,
        )

    chunks = []
    cur_t = 0.0
    _admit_pending(cur_t)
    if cur_ops:
        chunks.append(_ops_to_chunk(-1, -1, [cur_ops]))
        cur_ops = []

    for step_idx in range(steps):
        # Advance idle clock if no active requests (e.g. agentic think time), then run this decode step.
        if not active_slots:
            nxt = stream.next_time()
            cur_t = max(cur_t + step_s, nxt if nxt < math.inf else cur_t + step_s)
            _housekeeping(cur_t)
            _admit_pending(cur_t)
            if not active_slots:
                if cur_ops:
                    chunks.append(_ops_to_chunk(step_idx, 0, [cur_ops]))
                    cur_ops = []
                else:
                    chunks.append(LayerChunk(step=step_idx, layer=0))
                for lyr in range(1, layers):
                    chunks.append(LayerChunk(step=step_idx, layer=lyr))
                continue

        # Pre-step ops (e.g. admissions that happened right after idle jump) go before reads on layer 0.
        pre_ops = cur_ops
        cur_ops = []

        # Collect active LPAs for this decode step, partitioned evenly across layers (idx % layers).
        per_layer_reads = [[] for _ in range(layers)]
        seen_prefix = set()
        for sl in active_slots:
            for lpa in sl.entry.all_lpas():
                per_layer_reads[lpa % layers].append(("R", "kv", lpa))
            if sl.prefix is not None:
                if prefix_reads == "per_request" or sl.prefix.eid not in seen_prefix:
                    seen_prefix.add(sl.prefix.eid)
                    for lpa in sl.prefix.all_lpas():
                        per_layer_reads[lpa % layers].append(("R", "kv", lpa))

        for lyr in range(layers):
            _record_reads([lpa for _, _, lpa in per_layer_reads[lyr]])

        # Advance active slots by 1 decode step.
        cur_t += step_s
        finished = []
        for sl in active_slots:
            new_ctx = sl.ctx + 1
            delta = stored_pages(new_ctx) - stored_pages(sl.ctx)
            if delta > 0:
                sl.staged += delta
            sl.ctx = new_ctx
            if sl.ctx >= sl.end:
                finished.append(sl)

        # Complete finished slots, flush staged decode pages, run housekeeping, admit pending.
        for sl in finished:
            active_slots.remove(sl)
            _complete_slot(sl, cur_t)
        _flush_decode(cur_t)
        _housekeeping(cur_t)
        _admit_pending(cur_t)
        post_ops = cur_ops
        cur_ops = []

        # Emit per-layer chunks for this step; attach pre-step ops to layer 0 and post-step ops to last layer.
        for lyr in range(layers):
            groups = []
            if lyr == 0 and pre_ops:
                groups.append(pre_ops)
            groups.append(per_layer_reads[lyr])
            if lyr == layers - 1 and post_ops:
                groups.append(post_ops)
            chunks.append(_ops_to_chunk(step_idx, lyr, groups))

    meta = make_meta(
        cfg,
        layout_meta={"name": f"timeline:{policy}", "base": 0, "span": S * P,
                     "layers": layers, "segments": S, "segment_pages": P},
        config_meta={"workload": workload.name, "policy": policy, "steps": steps,
                     "step_s": step_s, "x": x, "retention_s": retention_s, "refresh": refresh},
        resident=(),
    )
    trace = AddressTrace(meta=meta, chunks=chunks)
    if validate_trace:
        validate(cfg, trace)

    host_programs = programs_by_cause["prefill"] + programs_by_cause["decode"]
    refresh_programs = (programs_by_cause["refresh_rd"] + programs_by_cause["refresh_ret"]
                        + programs_by_cause["replay"])
    stats = {
        "steps": steps,
        "duration_s": cur_t,
        "segments": S,
        "segment_pages": P,
        "total_blocks": S * row,
        "admitted": admitted,
        "completed": completed,
        "programs_by_cause": programs_by_cause,
        "host_programs": host_programs,
        "refresh_programs": refresh_programs,
        "total_programs": sum(programs_by_cause.values()),
        "refresh_to_kv": refresh_programs / host_programs if host_programs else 0.0,
        "segment_opens": seg_opens,
        "segment_erases_actual": seg_erases_actual,
        "refresh_events_rd": refresh_events_rd,
        "refresh_events_ret": refresh_events_ret,
        "compaction_events": compaction_events,
    }
    return trace, stats
