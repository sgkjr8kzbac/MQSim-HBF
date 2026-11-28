"""Segment-level event simulation engine (M4b)."""
import heapq
import math
import time

import numpy as np

from ...frontend.trace.requests import Stream
from ..geometry import CAUSES, DAY, GIB, Device, region_pages, sci, threshold_steps
from ..interference import hbf_gaps, interference
from .ops import _OpsMixin
from .options import (ACTIVE, CAUSE, EPS, FULL, IDLE, OPEN, PREFIX, PRIVATE, SESSION,
                      CapacityError, SimOptions)
from .storage import _StorageMixin
from .vllm import VllmTracker


class Simulator(_StorageMixin, _OpsMixin):
    """Segment-level event simulation of one HBF stack (design-B section 4); ``simulate`` runs it.

    A segment is one block index across the banks of a partition (a superblock of 1,024 banks, a
    channel block of 64 or a die block of 16). Placement is row-aligned, so an entry with n pages
    in a segment puts n / banks pages in each of its blocks and segment-level counts are exact.

    State (struct of arrays): segments (fill, live, pec, block reads since erase x hot, state,
    owner: a dedicated entry or -2 for a wave region, erases per cause); entries (a request's own
    KV, a shared prefix or a session history: pages, readers, active or idle, idle deadline);
    occupancy rows (entry, segment, pages, oldest write time) in COO arrays; running requests in
    slot arrays (context, end context, staged decode pages, entry, prefix entry, partition).

    One quantum: step time T from the Clock plus the previous quantum's M5 extra; n = quantum / T
    steps, cut at the earliest predicted read-threshold or retention crossing (not below rd_slack
    of the fastest segment's threshold in steps); requests complete and arrive at their exact step
    inside it. Then block reads are applied, staged decode pages are written in whole rows, and
    idle expiry, read-disturb refresh, retention refresh and reclaim run (design-B section 4.2).
    """

    def __init__(self, geometry, clock, workload, device=None, options=None, model=None, gpu=None):
        self.geo, self.clock, self.workload = geometry, clock, workload
        self.dev = d = device or Device()
        self.opt = o = options or SimOptions()
        self.model = model if model is not None else getattr(clock, "model", None)
        self.gpu = gpu if gpu is not None else getattr(clock, "gpu", None)
        if o.policy in ("layer_major", "vllm_pool") and d.partition != "none":
            raise ValueError(f"{o.policy} uses the whole device (no partition)")
        if o.prefill == "colocated" and o.prefill_rate is None and (self.model is None or self.gpu is None):
            raise ValueError("prefill='colocated' needs prefill_rate, or model and gpu (from Clock or arguments)")
        if o.refresh == "auto":
            self.refresh = "replay" if o.policy == "layer_major" else "relocate"
        else:
            self.refresh = o.refresh
        self.writes_decode = o.policy in ("packed", "dedicated", "vllm_pool")
        self.hbm_out = o.policy in ("hbm_decode", "layer_major")
        self.parts, self.row, self.P = d.parts, d.segment_banks, d.segment_pages
        self.S = S = d.capacity_pages // self.P
        if S < 2 * self.parts:
            raise ValueError(f"The device holds {S} segments of {self.P} pages: need at least 2 per partition")
        self.read_mult = float(getattr(self.geo, "read_multiplier", 1.0))
        self.read_scale = o.share * o.hot * self.read_mult / self.row
        # segments
        self.fill, self.live, self.reads = np.zeros(S), np.zeros(S), np.zeros(S)
        self.pec = np.zeros(S, dtype=np.int64)
        self.remap_epoch = np.zeros(S, dtype=np.int64)
        self.state = np.zeros(S, dtype=np.int8)
        self.owner = np.full(S, -1, dtype=np.int64)
        self.busy = np.zeros(S, dtype=bool)
        self.part_of = np.arange(S) % self.parts
        self.erases = np.zeros((S, len(CAUSES)), dtype=np.int64)
        self.is_free = np.ones(S, dtype=bool)
        self.heaps = [[self._key(s) for s in range(p, S, self.parts)] for p in range(self.parts)]
        self.free_count = np.array([len(heap) for heap in self.heaps], dtype=np.int64)
        self.part_live = np.zeros(self.parts)
        self.part_pec = np.zeros(self.parts)  # sum of segment PECs per partition (tie-break for new entries)
        self.part_ver = np.zeros(self.parts, dtype=np.int64)
        self.reclaim_blocked = np.full(self.parts, -1, dtype=np.int64)
        self.frontier = [-1] * self.parts
        self.prefix_frontier = [-1] * self.parts
        # entries
        self.e_cap = 0
        self.free_entries = []
        self._grow_entries(256)
        self.e_rows, self.e_owned, self.e_key, self.e_ids = {}, {}, {}, {}
        self.prefix_entry, self.session_entry = {}, {}
        self.protected = set()
        # occupancy rows
        self.c_e = np.zeros(4096, dtype=np.int64)
        self.c_s = np.zeros(4096, dtype=np.int64)
        self.c_n = np.zeros(4096)
        self.c_t = np.zeros(4096)
        self.c_v = np.zeros(4096, dtype=bool)
        self.c_count = self.dead_rows = 0
        # running requests
        w = workload
        self.cap = cap = o.slots or (w.concurrency if w.arrival == "closed" else w.max_batch)
        self.sl_on = np.zeros(cap, dtype=bool)
        self.sl_req = [None] * cap
        self.sl_entry = np.full(cap, -1, dtype=np.int64)
        self.sl_prefix = np.full(cap, -1, dtype=np.int64)
        self.sl_ctx = np.zeros(cap, dtype=np.int64)
        self.sl_end = np.zeros(cap, dtype=np.int64)
        self.sl_gen = np.zeros(cap, dtype=np.int64)
        self.sl_part = np.zeros(cap, dtype=np.int64)
        self.sl_staged = np.zeros(cap)
        self.sl_dead = np.zeros(cap)
        self.free_slots = list(range(cap - 1, -1, -1))
        self.n_running = 0
        self.queue = []
        self.stream = Stream(workload, stagger=o.stagger)
        self.region, self.region_cache = [], {}
        # vLLM free-list tracker (runs alongside packed wear when policy == "vllm_pool")
        self.vllm = None
        self.block_tokens = 0
        self._vllm_residual = 0
        self.vllm_stranded = 0.0
        self.vllm_dirty = False
        if o.policy == "vllm_pool":
            if hasattr(self.geo, "is_uniform") and not self.geo.is_uniform:
                raise ValueError("vllm_pool requires uniform layers (not heterogeneous --kv-spec)")
            bt = o.block_tokens or math.lcm(*self.geo.tpp)
            if any(bt % tpp for tpp in self.geo.tpp):
                raise ValueError("vllm_pool block_tokens must be a multiple of every HBF tensor's tokens per page")
            self.block_tokens = bt
            windows = {spec.window for spec in self.geo.hbf}
            if len(windows) == 1:
                win = next(iter(windows))
                if win is not None and win[0] in (None, 0) and win[1] is not None and win[1] < 0:
                    self._vllm_residual = -win[1]
            per_id = self.geo.heads * sum(bt // tpp for tpp in self.geo.tpp)
            pool_pages = self.S * self.P // self.geo.layers
            n_ids = pool_pages // per_id
            if n_ids < 1:
                raise ValueError("vllm_pool capacity too small for even one block id")
            self.vllm = VllmTracker(n_ids, per_id, d.pages_per_block, d.banks, o.free_order, o.vllm_reuse)
        # clocks and counters
        self.t = 0.0
        self.page_cache = {}
        self.programs = np.zeros(len(CAUSES))
        self.seg_erases = np.zeros(len(CAUSES), dtype=np.int64)
        self.block_erases = 0
        self.m5_erases = 0
        self.acc = dict.fromkeys(("steps", "tokens", "batch_s", "ctx_s", "decode_s", "tpot_s", "m5_s",
                                  "stranded_s", "prefill_s"), 0.0)
        self.n = dict.fromkeys(("completed", "admitted", "refresh_rd", "refresh_ret", "compaction", "evicted_pages",
                                "recompute_tokens", "stalls", "reclaims", "waves", "quanta", "dead_pages"), 0.0)
        self.prefill_dt = 0.0
        self.guard = False
        self.pending_reads = False
        self.drops = 0  # bumps whenever live data dies or moves (retry point of a stalled admission)
        self.blocked = None
        self.m5_extra = self.max_step_extra = self.staging_peak = 0.0
        self.m5_mark = (0.0, 0)
        self.free_min = int(self.free_count.sum())
        self.warm = None
        self.timeseries = []
        self.low_wm = o.low_watermark or min(2 + math.ceil(self._max_request_pages() / self.P - 1e-9),
                                             max(1, S // self.parts // 2))
        self.wall_s = 0.0

    # ---- helpers ---------------------------------------------------------------------------

    # ---- one quantum ---------------------------------------------------------------------------
    def _next_retention(self):
        if not self.opt.retention_s:
            return math.inf
        n = self.c_count
        valid = self.c_v[:n]
        return float(self.c_t[:n][valid].min()) + self.opt.retention_s if valid.any() else math.inf

    def _next_ttl(self):
        idle = self.e_state == IDLE
        return float(self.e_ttl[idle].min()) if idle.any() else math.inf

    def _clock_step(self):
        on = np.flatnonzero(self.sl_on)
        parts = self.sl_part[on] if self.parts > 1 else None
        hbm = self.sl_gen[on] if self.hbm_out else None
        return self.clock.step(self.sl_ctx[on], parts, self.row if parts is not None else None, hbm)

    def _limit(self, n, T, horizon, warm):
        """Cut the quantum at the earliest predicted threshold crossing, retention deadline, warm-up end or horizon."""
        o, t = self.opt, self.t
        rates = self._rates()
        read = rates > 0
        if read.any():
            rate = rates[read]
            floor = max(1, int(o.rd_slack * o.x / rate.max()))
            first = float(((o.x - self.reads[read]) / rate).min())
            n = min(n, max(floor, math.ceil(first - 1e-9)))
        for deadline in (self._next_retention(), warm if self.warm is None else math.inf, horizon):
            if deadline < math.inf:
                n = min(n, max(1, math.ceil((deadline - t) / T - 1e-9)))
        return n

    def _advance(self, on, k, T):
        ctx0 = self.sl_ctx[on]
        ctx1 = ctx0 + k
        dt = k * T
        self._update_stranded(dt)
        if self.writes_decode:
            if hasattr(self.geo, "written_pages"):
                self.sl_staged[on] += self.geo.written_pages(ctx0, ctx1)
            else:
                self.sl_staged[on] += self.geo.stored_pages(ctx1) - self.geo.stored_pages(ctx0)
        if hasattr(self.geo, "dead_pages"):
            dead = self.geo.dead_pages(ctx0, ctx1)
            if dead.any():
                for idx in np.flatnonzero(dead).tolist():
                    slot = int(on[idx])
                    rem = self._retire_dead(int(self.sl_entry[slot]), float(dead[idx]))
                    if rem > EPS:
                        self.sl_dead[slot] += rem
        if self.vllm is not None:
            for idx, slot in enumerate(on.tolist()):
                self._vllm_write(int(self.sl_entry[slot]), int(ctx0[idx]), int(ctx1[idx]))
        np.add.at(self.e_steps, self.sl_entry[on], k)
        pre = self.sl_prefix[on]
        pre = pre[pre >= 0]
        if len(pre):
            if self.opt.prefix_reads == "per_request":
                np.add.at(self.e_steps, pre, k)
            else:
                self.e_steps[np.unique(pre)] += k
        self.pending_reads = True
        self.sl_ctx[on] = ctx1
        self.sl_gen[on] += k
        self.acc["batch_s"] += len(on) * dt
        self.acc["ctx_s"] += float(ctx0.sum() + ctx1.sum()) / 2 * dt
        self.acc["tokens"] += len(on) * k

    def _consume_prefill(self):
        """Advance clock by serial colocated-prefill time accumulated during admissions."""
        if self.prefill_dt > 0:
            dt = self.prefill_dt
            self.prefill_dt = 0.0
            self._update_stranded(dt)
            self.t += dt
            return dt
        return 0.0

    def _quantum(self, n, T, stop=math.inf):
        """Run n decode steps of length T; requests complete and arrive at their exact step. Returns steps run."""
        t0, done, prefill_q = self.t, 0, 0.0
        splits = self.opt.policy != "layer_major"
        while done < n:
            on = np.flatnonzero(self.sl_on)
            if not len(on):
                break
            k = max(1, min(n - done, int((self.sl_end[on] - self.sl_ctx[on]).min())))
            if splits and self.free_slots and self.blocked != self.drops:
                arrival = self.stream.next_time()
                if arrival < t0 + n * T + prefill_q:
                    k = min(k, max(1, math.ceil((arrival - t0 - prefill_q) / T - 1e-9) - done))
            self._advance(on, k, T)
            done += k
            self.t = t0 + done * T + prefill_q
            finished = on[self.sl_ctx[on] >= self.sl_end[on]]
            for slot in finished.tolist():
                self._complete(slot)
            if len(finished) or (self.free_slots and self.stream.next_time() <= self.t + 1e-12):
                self._admit_pending()
                prefill_q += self._consume_prefill()
                if self.t >= stop - 1e-9:
                    break
        return done

    def _flush_decode(self):
        """Write staged decode pages in whole rows (per partition for packed, per request for dedicated)."""
        if not self.writes_decode:
            return
        on = np.flatnonzero(self.sl_on & (self.sl_staged > EPS))
        if not len(on):
            return
        row = self.row
        if self.opt.policy == "dedicated":
            for slot in on.tolist():
                whole = math.floor(self.sl_staged[slot] / row + 1e-9) * row
                if whole:
                    self.sl_staged[slot] -= whole
                    e = int(self.sl_entry[slot])
                    self._write(e, whole, "decode")
                    if self.sl_dead[slot] > EPS:
                        self.sl_dead[slot] = self._retire_dead(e, float(self.sl_dead[slot]))
            return
        for part in np.unique(self.sl_part[on]).tolist():
            group = on[self.sl_part[on] == part]
            total = float(self.sl_staged[group].sum())
            whole = math.floor(total / row + 1e-9) * row
            if whole <= 0:
                continue
            share = whole / total
            for slot in group.tolist():
                pages = float(self.sl_staged[slot]) * share
                self.sl_staged[slot] -= pages
                e = int(self.sl_entry[slot])
                self._write(e, pages, "decode")
                if self.sl_dead[slot] > EPS:
                    self.sl_dead[slot] = self._retire_dead(e, float(self.sl_dead[slot]))

    def _housekeeping(self):
        """Idle expiry, read-disturb refresh, retention refresh and background reclaim."""
        o = self.opt
        for e in np.flatnonzero((self.e_state == IDLE) & (self.e_ttl <= self.t + 1e-9)).tolist():
            if self.e_state[e] == IDLE:
                self._kill(e)
        self._flush_reads()
        for s in np.flatnonzero(self.reads >= o.x * (1 - 1e-9)).tolist():
            if self.reads[s] < o.x * (1 - 1e-9):
                continue
            if self.live[s] > EPS:
                self._refresh(s, "refresh_rd")
            elif self.state[s] == OPEN and self.owner[s] == -1:
                self._close(s)  # blocks past X take no new data
        if o.retention_s:
            n = self.c_count
            due = self.c_v[:n] & (self.c_t[:n] <= self.t - o.retention_s + 1e-6)
            for s in np.unique(self.c_s[:n][due]).tolist():
                if self.live[s] > EPS:
                    self._refresh(s, "refresh_ret")
        if o.policy != "layer_major":
            need = [part for part in range(self.parts)
                    if self.free_count[part] < self.low_wm and self.reclaim_blocked[part] != self.part_ver[part]]
            if need:
                state = self._victim_state()
                for part in need:
                    if self.free_count[part] < self.low_wm:
                        if state is None:
                            state = self._victim_state()
                        if self._reclaim(part, state=state):
                            state = None

    def _m5(self, step, steps):
        programs, erases = float(self.programs.sum()), self.m5_erases
        if self.opt.m5 != "off":
            o = self.opt
            load = {"programs": (programs - self.m5_mark[0]) / steps, "erases": (erases - self.m5_mark[1]) / steps,
                    "wave_banks": self.row, "gaps_ns": step.gaps_ns, "step_ns": step.tpot_ns}
            result = interference(load, o.m5, o.t_prog, o.t_bers, o.suspend_ns, block_unit=o.block_unit)
            self.m5_extra = result["extra_step_ns"]
            if self.warm is not None:
                self.max_step_extra = max(self.max_step_extra, result["max_step_extra_ns"])
        self.m5_mark = (programs, erases)

    def _staging(self):
        on = np.flatnonzero(self.sl_on)
        pages = float(self.sl_staged[on].sum()) + len(on) * self.geo.partial_pages()
        held = float(self.sl_gen[on].sum()) * self.geo.hbf_token_bytes if self.hbm_out else 0.0
        self.staging_peak = max(self.staging_peak, pages * self.dev.page_bytes + held)

    def run(self):
        """Simulate ``hours``; returns the summary row (rates over the time after the warm-up)."""
        o = self.opt
        wall = time.perf_counter()
        horizon, warm = o.hours * 3600, o.warmup_h * 3600
        next_report, step, stuck = 0.0, None, 0
        if warm <= 0:
            self.warm = self._snapshot()
        self._admit_pending()
        self._consume_prefill()
        while self.t < horizon - 1e-9:
            if self.warm is None and self.t >= warm - 1e-9:
                self.warm = self._snapshot()
            if self.t >= next_report - 1e-9:
                self._record(step)
                next_report = (math.floor(self.t / o.report_s + 1e-9) + 1) * o.report_s
            if not self.n_running:
                if self.queue and self.blocked == self.drops and self._next_ttl() == math.inf:
                    raise CapacityError(f"The next request does not fit even after reclaim (t = {self.t:.1f} s, "
                                        f"{self.S} segments of {self.P} pages); raise --capacity-gib")
                events = [self.stream.next_time(), horizon, next_report, self._next_retention(), self._next_ttl()]
                if self.warm is None:
                    events.append(warm)
                before = self.t
                self.t = max(min(events), self.t)
                self._update_stranded(self.t - before)
                self._housekeeping()
                self.m5_mark = (float(self.programs.sum()), self.m5_erases)
                self._admit_pending()
                self._consume_prefill()
                stuck = stuck + 1 if self.t <= before and not self.n_running else 0
                if stuck > 1000:
                    raise RuntimeError(f"The simulation makes no progress at t = {self.t:.3f} s")
                continue
            step = self._clock_step()
            T = (step.tpot_ns + self.m5_extra) / 1e9
            n = self._limit(max(1, int(o.quantum_s / T + 1e-9)), T, horizon, warm)
            stop = min(horizon, warm) if self.warm is None else horizon
            steps = self._quantum(n, T, stop)
            self.n["quanta"] += 1
            self._staging()
            self._flush_reads()  # before the flush: staged pages were read from HBM, not HBF
            self._flush_decode()
            self._housekeeping()
            if steps:
                self.acc["steps"] += steps
                self.acc["decode_s"] += steps * T
                self.acc["tpot_s"] += steps * step.tpot_ns / 1e9
                self.acc["m5_s"] += steps * self.m5_extra / 1e9
                self._m5(step, steps)
            self._admit_pending()
            self._consume_prefill()
        if self.warm is None:
            self.warm = self._snapshot()
        self._record(step)
        self.wall_s = time.perf_counter() - wall
        return self.summary()

    # ---- outputs -------------------------------------------------------------------------------
    def _snapshot(self):
        return {"t": self.t, "programs": self.programs.copy(), "seg_erases": self.seg_erases.copy(),
                "block_erases": self.block_erases, "pec": self.pec.copy(), "acc": dict(self.acc), "n": dict(self.n),
                "vllm": dict(self.vllm.counts) if self.vllm is not None else {}}

    def _record(self, step):
        page, on = self.dev.page_bytes, np.flatnonzero(self.sl_on)
        idle = float(self.e_pages[self.e_state == IDLE].sum())
        stranded_frac = 0.0
        stranded_pages = 0.0
        if self.vllm is not None:
            if self.vllm_dirty:
                self.vllm_stranded = self.vllm.stranded()
                self.vllm_dirty = False
            stranded_frac = self.vllm_stranded
            stranded_pages = stranded_frac * self.vllm.ids * self.vllm.pages_per_id * self.geo.layers
        row = {"t_s": self.t, "running": self.n_running, "queued": len(self.queue),
               "sum_context": int(self.sl_ctx[on].sum()),
               "tpot_ms": step.tpot_ns / 1e6 if step else None,
               "tpot_m5_ms": (step.tpot_ns + self.m5_extra) / 1e6 if step else None}
        row |= {f"{cause}_gb": self.programs[i] * page / 1e9 for i, cause in enumerate(CAUSES)}
        row |= {f"erases_{cause}": int(self.seg_erases[i]) for i, cause in enumerate(CAUSES)}
        row |= {"block_erases": self.block_erases, "free_segments": int(self.free_count.sum()),
                "live_pages": float(self.live.sum()), "idle_pages": idle, "stranded_pages": stranded_pages,
                "stranded_frac": stranded_frac, "avg_pec": float(self.pec.mean()), "max_pec": int(self.pec.max()),
                "max_reads_over_x": float(self.reads.max() / self.opt.x)}
        self.timeseries.append(row)

    def summary(self):
        o, d, w0 = self.opt, self.dev, self.warm
        span = max(self.t - w0["t"], 1e-9)
        page, ppb = d.page_bytes, d.pages_per_block
        prog = self.programs - w0["programs"]
        host = prog[CAUSE["prefill"]] + prog[CAUSE["decode"]]
        total = float(prog.sum())
        erased = (self.block_erases - w0["block_erases"]) * ppb
        days = span / DAY
        dpec = self.pec - w0["pec"]
        avg_day = float(dpec.mean()) / days
        max_day = avg_day if o.wear == "ideal" else float(dpec.max()) / days
        acc = {key: self.acc[key] - w0["acc"][key] for key in self.acc}
        n = {key: self.n[key] - w0["n"][key] for key in self.n}
        steps, tokens = acc["steps"], acc["tokens"]
        mean_ctx = acc["ctx_s"] / acc["batch_s"] if acc["batch_s"] else 0.0
        tpot = acc["tpot_s"] / steps if steps else None
        refresh = prog[CAUSE["refresh_rd"]] + prog[CAUSE["refresh_ret"]] + prog[CAUSE["replay"]]
        vllm = ({key: self.vllm.counts[key] - w0["vllm"].get(key, 0) for key in self.vllm.counts}
                if self.vllm is not None else {})
        row = {"workload": self.workload.name, "policy": o.policy, "partition": d.partition, "pages_per_block": ppb,
               "x": o.x, "stacks": o.stacks, "mean_batch": acc["batch_s"] / span, "mean_context": mean_ctx,
               "mean_tpot_ms": tpot * 1e3 if steps else None,
               "tpot_m5_ms": (acc["tpot_s"] + acc["m5_s"]) / steps * 1e3 if steps else None,
               "max_step_extra_ms": self.max_step_extra / 1e6, "req_per_s": n["completed"] / span,
               "tokens_per_s": tokens / span, "pages_per_request": host / n["completed"] if n["completed"] else None,
               "dead_pages_per_request": n["dead_pages"] / n["completed"] if n["completed"] else None,
               "kv_bytes_per_token": self.geo.hbf_token_bytes,
               "hbm_bytes_per_request": float(self.geo.hbm_bytes(np.array([int(round(mean_ctx))]))[0]),
               "threshold_steps": threshold_steps(o.x, ppb, o.share, o.hot * self.read_mult),
               "refreshes_per_request": (prog[CAUSE["refresh_rd"]] + prog[CAUSE["replay"]]) / host if host else None,
               "kv_write_gb_s": host * page / span / 1e9}
        row |= {f"{cause}_gb_s": prog[i] * page / span / 1e9 for i, cause in enumerate(CAUSES)}
        row |= {"program_gb_s": total * page / span / 1e9,
                "program_bw_share": total * page / span / (d.banks * page / (o.t_prog / 1e9)),
                "wa": total / host if host else None, "erase_amp": erased / total if total else None,
                "erase_wa": erased / host if host else None, "refresh_to_kv": refresh / host if host else None,
                "erases_per_s": erased / ppb / span, "avg_pec_per_day": avg_day, "max_pec_per_day": max_day}
        for limit in o.pe_limits:
            row[f"days_to_limit_avg@{sci(limit)}"] = limit / avg_day if avg_day else math.inf
        for limit in o.pe_limits:
            row[f"days_to_limit_max@{sci(limit)}"] = limit / max_day if max_day else math.inf
        row["pe_per_mtok"] = avg_day / (tokens / span * DAY / 1e6) if tokens else None
        row["break_even_x"] = None
        row["staging_peak_mib"] = self.staging_peak / 2 ** 20
        row["pending_units_per_channel"] = row["mean_batch"] * self.geo.partial_pages() / 16
        row["unit_open_s"] = max(self.geo.tpp) * tpot if steps else None
        row["replay_buffer_mib"] = self.P * page / 2 ** 20 if self.refresh == "replay" else 0.0
        row["legal"] = o.policy != "vllm_pool"
        row |= {"hours": o.hours, "warmup_h": o.warmup_h, "measured_h": span / 3600, "segments": self.S,
                "segment_pages": self.P, "low_watermark": self.low_wm, "decode_steps": int(steps),
                "completed": int(n["completed"]),
                "refresh_events_rd": int(n["refresh_rd"]), "refresh_events_ret": int(n["refresh_ret"]),
                "compaction_events": int(n["compaction"]), "waves": int(n["waves"]),
                "segment_erases": int((self.seg_erases - w0["seg_erases"]).sum()),
                "free_segments_min": self.free_min, "evicted_gb": n["evicted_pages"] * page / 1e9,
                "recompute_tokens": n["recompute_tokens"], "stalls": int(n["stalls"]),
                "stranded_frac": acc["stranded_s"] / span if self.vllm is not None else 0.0,
                "violations_skip": int(vllm.get("skip", 0)),
                "violations_overlap": int(vllm.get("overlap", 0)),
                "violations_live_erase": int(vllm.get("live_erase", 0)),
                "legalize_gb": vllm.get("legalize_pages", 0) * page * self.geo.layers / 1e9,
                "prefill_s": acc["prefill_s"], "quanta": int(self.n["quanta"]), "wall_s": self.wall_s}
        return row

    def segment_rows(self):
        rows = []
        for s in range(self.S):
            row = {"segment": s, "partition": int(self.part_of[s]), "state": int(self.state[s]),
                   "pec": int(self.pec[s]), "live_pages": float(self.live[s]), "fill_pages": float(self.fill[s]),
                   "reads": float(self.reads[s])}
            row |= {f"erases_{cause}": int(self.erases[s, i]) for i, cause in enumerate(CAUSES)}
            rows.append(row)
        return rows


def simulate(geometry, clock, workload, device=None, options=None, model=None, gpu=None):
    """Run the event simulation of one stack; returns the Simulator with ``result`` (the summary row),
    ``timeseries`` and ``segment_rows()``. ``workload`` is one stack's share (``workload.per_stack``)."""
    sim = Simulator(geometry, clock, workload, device, options, model, gpu)
    sim.result = sim.run()
    return sim
