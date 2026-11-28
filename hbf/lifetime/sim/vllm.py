"""vLLM free-list write-legality tracker on HBF."""
import numpy as np

from .options import CapacityError

class VllmTracker:
    """Write legality of a vLLM-style per-layer pool on HBF (one layer's pool; all layers are alike).

    Block id i holds pool pages [i P_id, (i + 1) P_id); a page's NAND block is (lpa mod banks,
    (lpa // banks) // ppb) and its position (lpa // banks) mod ppb. A write is legal when it is
    the block's next position (fill), or position 0 of a block without live pages (auto-erase,
    section 5.4.1 item 5). Otherwise it is an overlap (an already-written position, section
    11.5.2.2), a skip (section 11.5.2.6) or a live erase (page 0 of a block that still holds live
    data). Freed ids return to the free queue (``reuse``: fifo as vLLM, or lifo); stranded ids
    are free ids below the fill position of a block that still holds live data.
    """

    def __init__(self, ids, pages_per_id, pages_per_block, banks=1024, order="fresh", reuse="fifo"):
        from collections import deque
        if ids < 1 or pages_per_id < 1 or pages_per_block < 1 or banks < 1:
            raise ValueError("ids, pages_per_id, pages_per_block and banks must be positive")
        self.ids, self.pages_per_id, self.ppb, self.banks = ids, pages_per_id, pages_per_block, banks
        # When pages_per_id divides banks, every id occupies one whole bank-group on a single row
        # and all banks of a group receive identical writes and frees, so tracking one bank per
        # group is exact and pages_per_id times faster.
        self._grouped = (banks % pages_per_id == 0)
        self._scale = pages_per_id if self._grouped else 1
        groups = banks // pages_per_id if self._grouped else banks
        rows = -(-ids * pages_per_id // banks)
        self.nblocks = -(-rows // pages_per_block)
        self.fill = np.zeros(groups * self.nblocks, dtype=np.int64)
        self.live = np.zeros(groups * self.nblocks, dtype=np.int64)
        start = np.arange(ids)
        if order.startswith("shuffled:"):
            try:
                seed = int(order[9:])
            except ValueError:
                raise ValueError(f"Invalid free_order {order!r}: use fresh or shuffled:<int>") from None
            np.random.Generator(np.random.PCG64(seed)).shuffle(start)
        elif order != "fresh":
            raise ValueError("free_order must be fresh or shuffled:<seed>")
        if reuse not in ("fifo", "lifo"):
            raise ValueError("vllm_reuse must be fifo or lifo")
        self.queue, self.reuse = deque(start.tolist()), reuse
        self.used = np.zeros(ids, dtype=bool)
        self.counts = {"skip": 0, "overlap": 0, "live_erase": 0, "erases": 0, "writes": 0, "legalize_pages": 0}

    def _pages(self, ids):
        arr = np.asarray(ids, dtype=np.int64)
        if self._grouped:
            lpa = arr * self.pages_per_id
            row = lpa // self.banks
            return ((lpa % self.banks) // self.pages_per_id) * self.nblocks + row // self.ppb, row % self.ppb
        lpa = (arr[:, None] * self.pages_per_id + np.arange(self.pages_per_id)).ravel()
        row = lpa // self.banks
        return (lpa % self.banks) * self.nblocks + row // self.ppb, row % self.ppb

    def allocate(self, n):
        if n <= 0:
            return []
        if n > len(self.queue):
            raise CapacityError(f"vLLM pool exhausted: need {n} ids, {len(self.queue)} free")
        ids = [self.queue.popleft() for _ in range(n)]
        self.used[ids] = True
        self.write(ids)
        return ids

    def write(self, ids):
        if not len(ids):
            return
        key, pos = self._pages(ids)
        owner = (np.arange(len(ids), dtype=np.int64) if self._grouped
                 else np.repeat(np.arange(len(ids)), self.pages_per_id))
        order = np.argsort(key, kind="stable")
        k, p, o = key[order], pos[order], owner[order]
        first = np.ones(len(k), dtype=bool)
        first[1:] = k[1:] != k[:-1]
        live_before = self.live[k]
        erase = (p == 0) & first & (live_before == 0)
        live_erase = (p == 0) & ~erase
        # Within each block, split erase epochs only on legal erases (live_before == 0), not on
        # live_erase, so fill never shrinks below still-live pages.
        epoch_start = first | erase
        epoch = np.cumsum(epoch_start) - 1
        scale = self.ppb + 2
        val = np.where(first & ~erase, np.maximum(self.fill[k] - 1, p), p)
        keyed = epoch * scale + (val + 1)
        after = np.maximum.accumulate(keyed)
        before = np.empty_like(keyed)
        before[:1] = 0
        before[1:] = after[:-1]
        prev_high = np.where(erase, -1, np.where(first, self.fill[k] - 1, before - epoch * scale - 1))
        legal = erase | ((p != 0) & (p == prev_high + 1))
        overlap = ~legal & ~live_erase & (p <= prev_high)
        skip = ~legal & ~live_erase & (p > prev_high + 1)
        for name, mask in (("overlap", overlap), ("skip", skip), ("live_erase", live_erase)):
            if mask.any():
                self.counts[name] += len(np.unique(o[mask]))
        bad_mask = overlap | live_erase
        if bad_mask.any():  # legalising = replaying the offending blocks up to their fill at violation time
            bad_k = k[bad_mask]
            eff_fill = np.zeros(len(self.fill), dtype=np.int64)
            np.maximum.at(eff_fill, bad_k, np.maximum(self.fill[bad_k], prev_high[bad_mask] + 1))
            self.counts["legalize_pages"] += int(eff_fill[np.unique(bad_k)].sum()) * self._scale
        self.counts["erases"] += int(erase.sum()) * self._scale
        self.counts["writes"] += len(ids)
        ends = np.flatnonzero(np.r_[first[1:], True])
        self.fill[k[ends]] = after[ends] - epoch[ends] * scale
        np.add.at(self.live, k, 1)

    def free(self, ids):
        if not len(ids):
            return
        key, _ = self._pages(ids)
        np.subtract.at(self.live, key, 1)
        self.used[list(ids)] = False
        if self.reuse == "fifo":
            self.queue.extend(ids)
        else:
            self.queue.extendleft(ids)

    def stranded(self):
        """Fraction of all ids that are free but cannot be rewritten legally now."""
        free = np.flatnonzero(~self.used)
        if not len(free):
            return 0.0
        key, pos = self._pages(free)
        stuck = (self.live[key] > 0) & (pos < self.fill[key])
        if self._grouped:
            return float(stuck.sum()) / self.ids
        per_id = stuck.reshape(len(free), self.pages_per_id).any(axis=1)
        return float(per_id.sum()) / self.ids

