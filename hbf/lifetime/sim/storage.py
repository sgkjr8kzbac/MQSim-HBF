"""Segment, entry, row, read-counter, and vLLM storage helpers for Simulator."""
import heapq
import math

import numpy as np

from ..geometry import CAUSES
from .options import ACTIVE, CAUSE, EPS, FULL, IDLE, OPEN, PREFIX, PRIVATE, SESSION, CapacityError


class _StorageMixin:
    def _key(self, s):
        if self.opt.wear == "none":
            return (0, s)
        if self.opt.wear == "remap":
            return (int(self.remap_epoch[s]), s)
        return (int(self.pec[s]), s)

    def pages(self, tokens):
        """Full HBF pages of ``tokens`` tokens of one request (cached)."""
        tokens = int(tokens)
        hit = self.page_cache.get(tokens)
        if hit is None:
            hit = self.page_cache[tokens] = float(self.geo.stored_pages(np.array([tokens]))[0])
        return hit

    def _written(self, c0, c1):
        c0_i, c1_i = int(c0), int(c1)
        if hasattr(self.geo, "written_pages"):
            return float(self.geo.written_pages(np.array([c0_i]), np.array([c1_i]))[0])
        return max(0.0, self.pages(c1_i) - self.pages(c0_i))

    def _dead(self, c0, c1):
        if not hasattr(self.geo, "dead_pages"):
            return 0.0
        return float(self.geo.dead_pages(np.array([int(c0)]), np.array([int(c1)]))[0])

    def _retire_dead(self, e, n):
        """Expire ``n`` oldest live pages of entry ``e`` (sliding window deaths). Returns unretired remainder."""
        if n <= EPS or not self.e_rows.get(e):
            return max(0.0, float(n))
        self._entry_reads(e)
        rows = sorted(self.e_rows[e].values(), key=lambda i: (self.c_t[i], i))
        rem = float(n)
        retired = 0.0
        for i in rows:
            if rem <= EPS:
                break
            avail = float(self.c_n[i])
            if avail <= rem + EPS:
                _, dropped = self._drop(i)
                rem -= dropped
                retired += dropped
            else:
                s = int(self.c_s[i])
                p = self.part_of[s]
                self.c_n[i] -= rem
                self.e_pages[e] -= rem
                self.live[s] -= rem
                self.part_live[p] -= rem
                self.drops += 1
                self.part_ver[p] += 1
                if self.live[s] < EPS:
                    self.live[s] = 0.0
                    self._release(s)
                retired += rem
                rem = 0.0
        self.n["dead_pages"] += retired
        return max(0.0, rem)

    def _max_request_pages(self):
        w = self.workload

        def top(length):
            if length is None:
                return 0.0
            return float(length.get("n")) if length.kind == "fixed" else float(length._grid()[-1])
        if w.arrival == "script":
            tokens = max(
                int(item["input"]) + int(item["output"])
                + (int(item.get("prefix_tokens", w.prefix_tokens)) if int(item.get("prefix", -1)) >= 0 else 0)
                for item in w.script
            )
            return self.pages(int(tokens))
        if w.sessions:
            tokens = top(w.history) + w.calls * (top(w.input) + top(w.output))
        else:
            tokens = top(w.input) + top(w.output)
        return self.pages(int(tokens + w.prefix_tokens))

    def _grow_entries(self, new):
        fields = (("e_pages", float, 0), ("e_readers", float, 0), ("e_state", np.int8, 0),
                  ("e_ttl", float, math.inf), ("e_kind", np.int8, 0), ("e_part", np.int64, 0),
                  ("e_dseg", np.int64, -1), ("e_tokens", float, 0), ("e_steps", float, 0),
                  ("e_stored", np.int64, 0))  # e_stored: context position up to which the entry is on HBF
        old = self.e_cap
        for name, kind, fill in fields:
            grown = np.full(new, fill, dtype=kind)
            if old:
                grown[:old] = getattr(self, name)
            setattr(self, name, grown)
        self.free_entries.extend(range(new - 1, old - 1, -1))
        self.e_cap = new

    def _new_entry(self, kind, part, tokens):
        if not self.free_entries:
            self._grow_entries(2 * self.e_cap)
        e = self.free_entries.pop()
        self.e_pages[e] = self.e_readers[e] = self.e_steps[e] = 0.0
        self.e_state[e], self.e_ttl[e], self.e_kind[e] = ACTIVE, math.inf, kind
        self.e_part[e], self.e_dseg[e], self.e_tokens[e], self.e_stored[e] = part, -1, tokens, 0
        self.e_rows[e], self.e_owned[e] = {}, []
        return e

    def _program(self, cause, n):
        self.programs[CAUSE[cause]] += n

    # ---- occupancy rows --------------------------------------------------------------------
    def _row(self, e, s, n):
        rows = self.e_rows[e]
        i = rows.get(s)
        if i is not None:  # keeps the older write time (conservative for retention)
            self.c_n[i] += n
            return
        if self.c_count == len(self.c_e):
            if self.dead_rows > self.c_count // 2:
                self._compact_rows()
            else:
                for name in ("c_e", "c_s", "c_n", "c_t", "c_v"):
                    array = getattr(self, name)
                    grown = np.zeros(2 * len(array), dtype=array.dtype)
                    grown[:len(array)] = array
                    setattr(self, name, grown)
        i = self.c_count
        self.c_e[i], self.c_s[i], self.c_n[i], self.c_t[i], self.c_v[i] = e, s, n, self.t, True
        self.c_count += 1
        rows[s] = i

    def _compact_rows(self):
        self._flush_reads()
        keep = np.flatnonzero(self.c_v[:self.c_count])
        for name in ("c_e", "c_s", "c_n", "c_t", "c_v"):
            array = getattr(self, name)
            array[:len(keep)] = array[keep]
        self.c_v[len(keep):] = False
        self.c_count, self.dead_rows = len(keep), 0
        for rows in self.e_rows.values():
            rows.clear()
        for i, (e, s) in enumerate(zip(self.c_e[:len(keep)].tolist(), self.c_s[:len(keep)].tolist())):
            self.e_rows[e][s] = i

    def _drop(self, i):
        """Invalidate occupancy row i; returns (entry, pages)."""
        e, s, n = int(self.c_e[i]), int(self.c_s[i]), float(self.c_n[i])
        self.c_v[i] = False
        self.dead_rows += 1
        self.drops += 1
        p = self.part_of[s]
        self.part_ver[p] += 1
        del self.e_rows[e][s]
        self.e_pages[e] -= n
        self.live[s] -= n
        self.part_live[p] -= n
        if self.live[s] < EPS:
            self.live[s] = 0.0
            self._release(s)
        return e, n

    # ---- segments --------------------------------------------------------------------------
    def _release(self, s):
        """Mark segment s free when it is closed, unowned and holds no live data."""
        if (self.is_free[s] or self.busy[s] or self.state[s] == OPEN or self.owner[s] != -1
                or self.live[s] > EPS):
            return
        self.is_free[s] = True
        p = self.part_of[s]
        heapq.heappush(self.heaps[p], self._key(s))
        self.free_count[p] += 1
        self.drops += 1
        self.part_ver[p] += 1

    def _take(self, part):
        heap = self.heaps[part]
        while heap:
            s = heapq.heappop(heap)[1]
            if self.is_free[s]:
                self.is_free[s] = False
                self.remap_epoch[s] += 1
                self.free_count[part] -= 1
                self.free_min = min(self.free_min, int(self.free_count.sum()))
                return s
        return -1

    def _erase(self, s, cause):
        """Row-0 write of segment s: every block of it is auto-erased (spec section 5.4.1 item 5)."""
        c = CAUSE[cause]
        self.pec[s] += 1
        self.part_pec[self.part_of[s]] += 1
        self.erases[s, c] += 1
        self.seg_erases[c] += 1
        self.block_erases += self.row
        if not self.opt.pre_open or cause == "replay":
            self.m5_erases += self.row
        self.reads[s] = 0.0

    def _open(self, part, cause, owner=-1, prefix=False):
        """Take a free segment of ``part`` and write its row 0 (an erase), reclaiming first when none is free."""
        front = self.prefix_frontier if (prefix and self.opt.separate_prefix) else self.frontier
        s = self._take(part)
        if s < 0 and not self.guard:
            self._reclaim(part, force=True, target=lambda: self.free_count[part] >= 1 or (owner < 0 and front[part] >= 0))
            if owner < 0 and front[part] >= 0:  # compaction opened a frontier: continue there
                return front[part]
            s = self._take(part)
        if s < 0:
            raise CapacityError(f"No free segment in partition {part} at t = {self.t:.1f} s "
                                f"({self.S} segments of {self.P} pages); raise --capacity-gib or --low-watermark")
        self._erase(s, cause)
        self.state[s], self.owner[s], self.fill[s] = OPEN, owner, 0.0
        if owner >= 0:
            self.e_owned[owner].append(s)
        else:
            front[part] = s
        return s

    def _close(self, s):
        self.state[s] = FULL
        p = self.part_of[s]
        if self.frontier[p] == s:
            self.frontier[p] = -1
        if self.prefix_frontier[p] == s:
            self.prefix_frontier[p] = -1
        o = self.owner[s]
        if o >= 0 and self.e_dseg[o] == s:
            self.e_dseg[o] = -1
        self.part_ver[p] += 1
        self._release(s)

    # ---- writes ----------------------------------------------------------------------------
    def _write(self, e, n, cause, align=False):
        """Append n pages of entry e: the partition's frontier, or the entry's own segment (dedicated)."""
        part = int(self.e_part[e])
        dedicated = self.opt.policy == "dedicated"
        use_prefix = self.opt.separate_prefix and self.e_kind[e] == PREFIX
        front = self.prefix_frontier if use_prefix else self.frontier
        while n > EPS:
            if dedicated:
                s = int(self.e_dseg[e])
                if s < 0:
                    s = self._open(part, cause, owner=e)
                    self.e_dseg[e] = s
            else:
                s = front[part]
                if s >= 0 and align and self.opt.align == "row":
                    pad = math.ceil(self.fill[s] / self.row - 1e-9) * self.row - self.fill[s]
                    if pad > EPS:
                        self.fill[s] += pad
                        self._program("padding", pad)
                        if self.fill[s] >= self.P - EPS:
                            self._close(s)
                            s = -1
                if s < 0:
                    s = self._open(part, cause, prefix=use_prefix)
                align = False
            k = min(n, self.P - self.fill[s])
            self.fill[s] += k
            self.live[s] += k
            self.part_live[self.part_of[s]] += k
            self.e_pages[e] += k
            self._row(e, s, k)
            self._program(cause, k)
            n -= k
            if self.fill[s] >= self.P - EPS:
                self._close(s)

    def _read_once(self, e):
        """Prefill attention over a cached entry: every page is read once (no share, no skew)."""
        for s, i in self.e_rows[e].items():
            self.reads[s] += self.c_n[i] * self.read_mult / self.row

    # ---- block reads -----------------------------------------------------------------------
    def _flush_reads(self):
        """Apply the read steps accrued since the last flush to the segments holding each entry."""
        if not self.pending_reads:
            return
        n = self.c_count
        weight = self.c_n[:n] * self.e_steps[self.c_e[:n]]
        weight[~self.c_v[:n]] = 0.0
        self.reads += np.bincount(self.c_s[:n], weight, self.S) * self.read_scale
        self.e_steps[:] = 0.0
        self.pending_reads = False

    def _entry_reads(self, e):
        k = self.e_steps[e]
        if k:
            for s, i in self.e_rows[e].items():
                self.reads[s] += self.c_n[i] * k * self.read_scale
            self.e_steps[e] = 0.0

    def _rates(self):
        """Block reads per decode step per segment for the current running set."""
        on = np.flatnonzero(self.sl_on)
        rate = np.zeros(self.e_cap)
        np.add.at(rate, self.sl_entry[on], 1.0)
        pre = self.sl_prefix[on]
        pre = pre[pre >= 0]
        if len(pre):
            if self.opt.prefix_reads == "per_request":
                np.add.at(rate, pre, 1.0)
            else:
                rate[np.unique(pre)] = 1.0
        n = self.c_count
        weight = self.c_n[:n] * rate[self.c_e[:n]]
        weight[~self.c_v[:n]] = 0.0
        return np.bincount(self.c_s[:n], weight, self.S) * self.read_scale

    # ---- admission and completion ------------------------------------------------------------
    def _partition(self):
        """Partition of a new entry: least live pages; ties to the least worn (wear levelling) or lowest index."""
        if self.parts == 1:
            return 0
        low = np.flatnonzero(self.part_live <= self.part_live.min() + EPS)
        if self.opt.wear == "none":
            return int(low[0])
        return int(low[np.argmin(self.part_pec[low])])

    def _outstanding(self, part):
        """Pages that running requests of ``part`` will still write (remaining decode plus staged)."""
        if not self.writes_decode:
            return 0.0
        on = np.flatnonzero(self.sl_on & (self.sl_part == part))
        if not len(on):
            return 0.0
        geo = self.geo
        if hasattr(geo, "written_pages"):
            rem = geo.written_pages(self.sl_ctx[on], self.sl_end[on]).sum()
        else:
            rem = (geo.stored_pages(self.sl_end[on]) - geo.stored_pages(self.sl_ctx[on])).sum()
        return float(rem + self.sl_staged[on].sum())

    def _frontier_rem(self, part, prefix=False):
        if self.opt.policy == "dedicated":
            return 0.0
        front = self.prefix_frontier if (prefix and self.opt.separate_prefix) else self.frontier
        s = front[part]
        return (self.P - self.fill[s]) if s >= 0 else 0.0

    def _room(self, part, prefix=False):
        return float(self.free_count[part] * self.P) + self._frontier_rem(part, prefix)

    def _segs_needed(self, part, main_pages, prefix_pages=0.0):
        """Free segments of ``part`` needed to write ``main_pages`` and ``prefix_pages``."""
        if self.opt.policy == "dedicated":
            s_main = math.ceil(main_pages / self.P - 1e-9) if main_pages > EPS else 0
            s_pre = math.ceil(prefix_pages / self.P - 1e-9) if prefix_pages > EPS else 0
            return s_main + s_pre
        if self.opt.separate_prefix:
            rem_main = self._frontier_rem(part, prefix=False)
            rem_pre = self._frontier_rem(part, prefix=True)
            s_main = max(0, math.ceil((main_pages - rem_main) / self.P - 1e-9)) if main_pages > EPS else 0
            s_pre = max(0, math.ceil((prefix_pages - rem_pre) / self.P - 1e-9)) if prefix_pages > EPS else 0
            return s_main + s_pre
        total = main_pages + prefix_pages
        rem = self._frontier_rem(part, prefix=False)
        return max(0, math.ceil((total - rem) / self.P - 1e-9)) if total > EPS else 0

    def _space(self, part, need, prefix_need=0.0):
        """Room for ``need`` (and ``prefix_need``) pages plus outstanding writes and reserve, reclaiming when short."""
        want = need + self._outstanding(part)
        reserve = self.low_wm - 1
        if self.free_count[part] >= self._segs_needed(part, want, prefix_need) + reserve:
            return True
        self._reclaim(part, force=True,
                      target=lambda: self.free_count[part] >= self._segs_needed(part, want, prefix_need) + reserve)
        return self.free_count[part] >= self._segs_needed(part, want, prefix_need)

    def _vllm_tokens(self, tokens):
        return max(0, int(tokens) - self._vllm_residual)

    def _vllm_outstanding(self):
        """vLLM block ids that currently running requests will still allocate during decode."""
        if self.vllm is None or not self.writes_decode:
            return 0
        on = np.flatnonzero(self.sl_on)
        if not len(on):
            return 0
        bt, res = self.block_tokens, self._vllm_residual
        end_ids = np.maximum(0, self.sl_end[on] - res) // bt
        ctx_ids = np.maximum(0, self.sl_ctx[on] - res) // bt
        return int((end_ids - ctx_ids).sum())

    def _vllm_ensure(self, need_ids):
        """Evict oldest idle entries when the vLLM free queue lacks ``need_ids`` ids."""
        if self.vllm is None or need_ids <= len(self.vllm.queue):
            return True
        idle = [e for e in np.flatnonzero(self.e_state == IDLE).tolist()
                if e not in self.protected and e in self.e_ids]
        idle.sort(key=lambda e: min((self.c_t[i] for i in self.e_rows[e].values()), default=math.inf))
        for e in idle:
            self._kill(e, evicted=True)
            if need_ids <= len(self.vllm.queue):
                return True
        return need_ids <= len(self.vllm.queue)

    def _vllm_write(self, e, lo, hi):
        """vllm_pool: allocate and write the ids of full vLLM blocks in HBF tokens [lo, hi) of entry e."""
        if self.vllm is None:
            return
        bt = self.block_tokens
        count = self._vllm_tokens(hi) // bt - self._vllm_tokens(lo) // bt
        if count > 0:
            self._vllm_ensure(count)
            self.e_ids.setdefault(e, []).extend(self.vllm.allocate(count))
            self.vllm_dirty = True

    def _update_stranded(self, dt):
        """Integrate time-averaged stranded fraction over ``dt`` seconds (vllm_pool only)."""
        if self.vllm is not None and dt > 0:
            if self.vllm_dirty:
                self.vllm_stranded = self.vllm.stranded()
                self.vllm_dirty = False
            self.acc["stranded_s"] += self.vllm_stranded * dt

