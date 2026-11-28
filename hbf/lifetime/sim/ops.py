"""Request admission/completion, waves, refresh, relocation, and reclaim for Simulator."""
import math

import numpy as np

from ..geometry import prefill_tok_s, region_pages
from .options import ACTIVE, CAUSE, EPS, FULL, IDLE, OPEN, PREFIX, PRIVATE, SESSION, CapacityError


class _OpsMixin:
    def _admit(self, req):
        """Start a request: prefix and session entries, prefill writes. False when space is short."""
        w = self.workload
        pre_tokens = w.prefix_tokens if req.prefix >= 0 else 0
        start = pre_tokens + req.history + req.input
        end = start + req.output
        sess = self.session_entry.get(req.session, -1) if req.session >= 0 else -1
        pre = self.prefix_entry.get(req.prefix, -1) if req.prefix >= 0 else -1
        new_prefix = req.prefix >= 0 and pre < 0
        # A session's history is on HBF up to e_stored (under hbm_decode the last call's output is not:
        # it is written with this call's prompt); an evicted history is recomputed and rewritten.
        have = int(self.e_stored[sess]) if sess >= 0 else pre_tokens
        own = min(self._written(have, start), self.pages(start)) if have > 0 else self._written(0, start)
        pre_need = self.pages(pre_tokens) if new_prefix else 0.0
        need = own + (self._written(start, end) if self.writes_decode else 0.0)
        if not self.opt.separate_prefix:
            need += pre_need
            pre_need = 0.0
        vllm_need = 0
        if self.vllm is not None:
            bt = self.block_tokens
            if new_prefix:
                vllm_need += self._vllm_tokens(pre_tokens) // bt
            v_hi = self._vllm_tokens(end if self.writes_decode else start)
            vllm_need += max(0, v_hi // bt - self._vllm_tokens(have) // bt)
            vllm_need += self._vllm_outstanding()
        part = int(self.e_part[sess]) if sess >= 0 else self._partition()
        keep = [e for e in (sess, pre) if e >= 0]
        self.protected.update(keep)  # a forced reclaim must not evict what this request reuses
        try:
            if not self._space(part, need, pre_need) or not self._vllm_ensure(vllm_need):
                return False
        finally:
            self.protected.difference_update(keep)
        if req.prefix >= 0:
            if pre >= 0:
                self._read_once(pre)
            else:
                pre = self._new_entry(PREFIX, part, pre_tokens)
                self.prefix_entry[req.prefix], self.e_key[pre] = pre, ("prefix", req.prefix)
                self._write(pre, self.pages(pre_tokens), "prefill", align=True)
                self._vllm_write(pre, 0, pre_tokens)
            self.e_state[pre], self.e_ttl[pre] = ACTIVE, math.inf
            self.e_readers[pre] += 1
        if sess >= 0:
            e = sess
            self._read_once(e)
            if have > 0:
                dead_sess = self._dead(have, start)
                if dead_sess > EPS:
                    self._retire_dead(e, dead_sess)
        else:
            e = self._new_entry(SESSION if req.session >= 0 else PRIVATE, part, 0)
            if req.session >= 0:
                self.session_entry[req.session], self.e_key[e] = e, ("session", req.session)
                self.n["recompute_tokens"] += req.history  # an evicted history is recomputed
        self.e_state[e], self.e_readers[e], self.e_ttl[e] = ACTIVE, 1.0, math.inf
        self.e_tokens[e], self.e_stored[e] = start - pre_tokens, start
        self._write(e, own, "prefill", align=sess < 0)
        self._vllm_write(e, have, start)
        if self.opt.prefill == "colocated":
            prefill_toks = req.input + (pre_tokens if new_prefix else 0) + (req.history if sess < 0 else 0)
            dt = self.opt.stacks * prefill_toks / prefill_tok_s(self.model, self.gpu, start)
            self.prefill_dt += dt
            self.acc["prefill_s"] += dt
        slot = self.free_slots.pop()
        self.sl_on[slot], self.sl_req[slot] = True, req
        self.sl_entry[slot], self.sl_prefix[slot], self.sl_part[slot] = e, pre, part
        self.sl_ctx[slot], self.sl_end[slot], self.sl_gen[slot], self.sl_staged[slot] = start, end, 0, 0.0
        self.sl_dead[slot] = 0.0
        self.n_running += 1
        self.n["admitted"] += 1
        return True

    def _complete(self, slot):
        req, e, pre = self.sl_req[slot], int(self.sl_entry[slot]), int(self.sl_prefix[slot])
        self._entry_reads(e)
        if self.sl_staged[slot] > EPS:  # the request's last staged full pages (a partial row)
            self._write(e, float(self.sl_staged[slot]), "decode")
        if self.sl_dead[slot] > EPS:
            self._retire_dead(e, float(self.sl_dead[slot]))
            self.sl_dead[slot] = 0.0
        if self.writes_decode:
            self.e_stored[e] = self.sl_end[slot]
            self.e_tokens[e] = self.sl_end[slot] - (self.workload.prefix_tokens if pre >= 0 else 0)
        self.sl_on[slot], self.sl_req[slot], self.sl_staged[slot], self.sl_dead[slot] = False, None, 0.0, 0.0
        self.free_slots.append(slot)
        self.n_running -= 1
        self.n["completed"] += 1
        w = self.workload
        if pre >= 0:
            self.e_readers[pre] -= 1
            if self.e_readers[pre] <= 0:
                if w.retain_s > 0:
                    self._idle(pre, self.t + w.retain_s)
                else:
                    self._kill(pre)
        if req.session >= 0 and not req.last:
            self._idle(e, math.inf)
        elif req.retain_s > 0:
            self._idle(e, self.t + req.retain_s)
        else:
            self._kill(e)
        self.stream.call_done(req, self.t)

    def _idle(self, e, ttl):
        self.e_state[e], self.e_readers[e], self.e_ttl[e] = IDLE, 0.0, ttl
        self.part_ver[self.e_part[e]] += 1

    def _kill(self, e, evicted=False):
        self._entry_reads(e)
        if evicted:
            self.n["evicted_pages"] += self.e_pages[e]
            self.n["recompute_tokens"] += self.e_tokens[e]
        for i in list(self.e_rows[e].values()):
            self._drop(i)
        for s in self.e_owned.pop(e, ()):
            if self.owner[s] == e:
                self.owner[s] = -1
                if self.state[s] == OPEN:
                    self.state[s] = FULL
                self._release(s)
        if self.vllm is not None and e in self.e_ids:
            self.vllm.free(self.e_ids.pop(e))
            self.vllm_dirty = True
        key = self.e_key.pop(e, None)
        if key is not None:
            table = self.prefix_entry if key[0] == "prefix" else self.session_entry
            if table.get(key[1]) == e:
                del table[key[1]]
        del self.e_rows[e]
        self.e_state[e], self.e_readers[e], self.e_pages[e], self.e_dseg[e] = 0, 0.0, 0.0, -1
        self.e_ttl[e], self.e_steps[e] = math.inf, 0.0
        self.free_entries.append(e)

    def _admit_pending(self):
        self.queue.extend(self.stream.arrived(self.t))
        w = self.workload
        if self.opt.policy == "layer_major":
            if not self.n_running:
                if self.region:
                    self._end_wave()
                self._start_wave()
            return
        if w.arrival == "closed" and not w.sessions:
            while self.n_running + len(self.queue) < self.cap:
                self.queue.append(self.stream.closed_new(self.t))
        if self.blocked == self.drops:  # nothing died or moved since the last failed admission
            return
        while self.queue and self.free_slots:
            if not self._admit(self.queue[0]):
                self.n["stalls"] += 1
                self.blocked = self.drops
                return
            self.queue.pop(0)
        self.blocked = None

    # ---- layer-major waves -------------------------------------------------------------------
    def _region_span(self, prompt):
        """Pages a wave writes: phase A's striped layout of H x slots heads at the longest prompt."""
        if prompt not in self.region_cache:
            if hasattr(self.geo, "region_span"):
                span = self.geo.region_span(self.cap, prompt, self.dev.pages_per_block)
            elif self.model is not None:
                span = region_pages(self.model, self.geo.tensors, self.cap, prompt, self.dev.pages_per_block)
            else:  # no model (tests): every per-head stream padded to whole pages
                per = sum(int(-(-(hi - lo)[0] // tpp)) for spec, tpp in zip(self.geo.hbf, self.geo.tpp)
                          for lo, hi in [self.geo.window(spec, np.array([prompt]))])
                span = float(self.cap * self.geo.layers * self.geo.heads * per)
            self.region_cache[prompt] = span
        return self.region_cache[prompt]

    def _start_wave(self):
        w = self.workload
        reqs = self.queue[:self.cap]
        del self.queue[:len(reqs)]
        while len(reqs) < self.cap and w.arrival == "closed" and not w.sessions:
            reqs.append(self.stream.closed_new(self.t))
        if not reqs:
            return
        prompts = [(w.prefix_tokens if r.prefix >= 0 else 0) + r.history + r.input for r in reqs]
        span = self._region_span(max(prompts))
        need = math.ceil(span / self.P - 1e-9)
        if self.free_count[0] < need:
            self._reclaim(0, force=True, target=lambda: self.free_count[0] >= need)
        if self.free_count[0] < need:
            raise CapacityError(f"A wave region of {span:.0f} pages needs {need} segments, "
                                f"{self.free_count[0]} are free")
        self.region = [self._take(0) for _ in range(need)]
        fills, left = [], span
        for s in self.region:
            self._erase(s, "wave")
            self.state[s], self.owner[s] = FULL, -2
            self.fill[s] = min(float(self.P), left)
            fills.append(self.fill[s])
            left -= self.fill[s]
        real = 0.0
        for req, prompt in zip(reqs, prompts):
            pages = self.pages(prompt)
            real += pages
            e = self._new_entry(PRIVATE, 0, prompt)
            for s, f in zip(self.region, fills):
                k = pages * f / span
                self.live[s] += k
                self.part_live[0] += k
                self.e_pages[e] += k
                self._row(e, s, k)
            self.e_readers[e] = 1.0
            slot = self.free_slots.pop()
            self.sl_on[slot], self.sl_req[slot], self.sl_entry[slot], self.sl_prefix[slot] = True, req, e, -1
            self.sl_ctx[slot], self.sl_end[slot], self.sl_gen[slot] = prompt, prompt + req.output, 0
            self.sl_part[slot], self.sl_staged[slot], self.sl_dead[slot] = 0, 0.0, 0.0
            self.n_running += 1
            self.n["admitted"] += 1
            if self.opt.prefill == "colocated":
                dt = self.opt.stacks * prompt / prefill_tok_s(self.model, self.gpu, prompt)
                self.prefill_dt += dt
                self.acc["prefill_s"] += dt
        self._program("prefill", real)
        self._program("padding", span - real)
        self.n["waves"] += 1

    def _end_wave(self):
        for s in self.region:
            self.owner[s] = -1
            self._release(s)
        self.region = []

    # ---- refresh and reclaim -----------------------------------------------------------------
    def _refresh(self, s, cause):
        self.n[cause] += 1
        if self.refresh == "replay":
            self._replay(s)
        else:
            self._relocate(s, cause)

    def _replay(self, s):
        """Rewrite rows 0 .. fill - 1 of segment s in place, dead pages included (spec section 5.7)."""
        self._flush_reads()
        self._erase(s, "replay")
        self._program("replay", self.fill[s])
        n = self.c_count
        self.c_t[:n][self.c_v[:n] & (self.c_s[:n] == s)] = self.t

    def _relocate(self, s, cause):
        """Move the live pages of segment s (refresh or compaction); idle entries may be evicted instead.

        The pages go to the partition's frontier, or to the owner's own segment (dedicated). Segment s
        is free afterwards; its erase is counted at its next row-0 write.
        """
        self._flush_reads()
        self.busy[s] = True
        o = int(self.owner[s])
        if o >= 0:
            self.e_owned[o].remove(s)
            self.owner[s] = -1
            if self.e_dseg[o] == s:
                self.e_dseg[o] = -1
        if self.state[s] == OPEN:
            self._close(s)
        evict = self.opt.reclaim == "evict" if cause == "compaction" else self.opt.refresh_idle == "evict"
        n = self.c_count
        moves = []
        for i in np.flatnonzero(self.c_v[:n] & (self.c_s[:n] == s)).tolist():
            if not self.c_v[i]:
                continue
            e = int(self.c_e[i])
            if evict and self.e_state[e] == IDLE and e not in self.protected:
                self._kill(e, evicted=True)
                continue
            moves.append(self._drop(i))
        moved = {e for e, _ in moves}
        self.protected.update(moved)
        try:
            for e, pages in moves:
                self._write(e, pages, cause)
        finally:
            self.protected.difference_update(moved)
        self.busy[s] = False
        self.reads[s] = 0.0
        self._release(s)

    def _victim_state(self):
        """Per-segment copy counts and idle-owner mask for victim selection (shared across partitions)."""
        n = self.c_count
        e = self.c_e[:n]
        v = self.c_v[:n]
        if self.opt.reclaim == "compact":
            keep = (self.e_state[e] > 0) & v
        else:
            keep = (self.e_state[e] == ACTIVE) & v
            if self.protected:
                keep |= np.isin(e, list(self.protected)) & v
        if self.opt.separate_prefix:
            is_pre = self.e_kind[e] == PREFIX
            copy_main = np.bincount(self.c_s[:n][keep & ~is_pre], self.c_n[:n][keep & ~is_pre], self.S)
            copy_pre = np.bincount(self.c_s[:n][keep & is_pre], self.c_n[:n][keep & is_pre], self.S)
            copy = copy_main + copy_pre
        else:
            copy_main = copy_pre = None
            copy = np.bincount(self.c_s[:n][keep], self.c_n[:n][keep], self.S)
        owner_idle = np.zeros(self.S, dtype=bool)
        owned = self.owner >= 0
        if owned.any():
            owner_idle[owned] = (self.e_state[self.owner[owned]] == IDLE)
            if self.protected:
                owner_idle[owned] &= ~np.isin(self.owner[owned], list(self.protected))
            copy[owner_idle] = 0.0
            if self.opt.separate_prefix:
                copy_main[owner_idle] = 0.0
                copy_pre[owner_idle] = 0.0
        return [copy, owner_idle, None, copy_main, copy_pre]

    def _oldest_times(self, state):
        if state[2] is None:
            n = self.c_count
            v = self.c_v[:n]
            oldest = np.full(self.S, math.inf)
            np.minimum.at(oldest, self.c_s[:n][v], self.c_t[:n][v])
            state[2] = oldest
        return state[2]

    def _victim(self, part, state=None):
        """Reclaim victim of ``part``: fewest pages to copy (greedy) or highest (1 - u) * age / (1 + u)."""
        if state is None:
            state = self._victim_state()
        copy, owner_idle, _, copy_main, copy_pre = state
        sl = slice(part, self.S, self.parts)
        cand = (~self.busy[sl] & ~self.is_free[sl] & (self.live[sl] > EPS)
                & (((self.state[sl] == FULL) & (self.owner[sl] == -1)) | owner_idle[sl]))
        local = np.flatnonzero(cand)
        if not len(local):
            return -1
        index = part + local * self.parts
        if self.opt.separate_prefix:
            rem_main = self._frontier_rem(part, prefix=False)
            rem_pre = self._frontier_rem(part, prefix=True)
            cm, cp = copy_main[index], copy_pre[index]
            segs_main = np.where(cm > EPS, np.maximum(0, np.ceil((cm - rem_main) / self.P - 1e-9)), 0)
            segs_pre = np.where(cp > EPS, np.maximum(0, np.ceil((cp - rem_pre) / self.P - 1e-9)), 0)
            fits = (copy[index] < self.P - EPS) & (segs_main + segs_pre <= self.free_count[part])
            index = index[fits]
            if not len(index):
                return -1
        else:
            room = self._room(part)
            if self.opt.victim == "cost_benefit":
                index = index[(copy[index] < self.P - EPS) & (copy[index] <= room + EPS)]
                if not len(index):
                    return -1
            else:
                c_idx = copy[index]
                c_min = float(c_idx.min())
                if c_min >= self.P - EPS or c_min > room + EPS:
                    return -1
                least = index[c_idx <= c_min + EPS]
                if len(least) > 1:  # ties: the segment with the oldest data (least recently written first)
                    oldest = self._oldest_times(state)
                    least = least[np.argsort(oldest[least], kind="stable")]
                return int(least[0])
        if self.opt.victim == "cost_benefit":
            oldest = self._oldest_times(state)
            u = copy[index] / self.P
            age = np.maximum(self.t - oldest[index], 1e-6)
            score = (1.0 - u) * age / (1.0 + u)
            best_score = float(score.max())
            top = index[score >= best_score - 1e-12 * max(1.0, best_score)]
            if len(top) > 1:
                top = top[np.argsort(copy[top], kind="stable")]
            return int(top[0])
        c_idx = copy[index]
        c_min = float(c_idx.min())
        least = index[c_idx <= c_min + EPS]
        if len(least) > 1:
            oldest = self._oldest_times(state)
            least = least[np.argsort(oldest[least], kind="stable")]
        return int(least[0])

    def _reclaim(self, part, force=False, target=None, state=None):
        """Free segments of ``part``: while fewer than the low watermark, or (forced) until ``target`` holds.
        Returns True if any segment was reclaimed."""
        if self.guard:
            return False
        if not force and not self.protected and self.reclaim_blocked[part] == self.part_ver[part]:
            return False
        self.guard = True
        changed = False
        try:
            for _ in range(self.S):
                if force:
                    if target is not None and (target() if callable(target) else self._room(part) >= target):
                        break
                elif self.free_count[part] >= self.low_wm:
                    break
                victim = self._victim(part, state)
                if victim < 0:
                    if not self.protected:
                        self.reclaim_blocked[part] = self.part_ver[part]
                    break
                changed = True
                state = None  # partition state changed; recompute if another victim is needed
                self.n["reclaims"] += 1
                owner = int(self.owner[victim])
                if owner >= 0:  # dedicated: an idle owner is evicted with all its segments
                    self._kill(owner, evicted=True)
                else:
                    self.n["compaction"] += 1
                    self._relocate(victim, "compaction")
        finally:
            self.guard = False
        return changed

