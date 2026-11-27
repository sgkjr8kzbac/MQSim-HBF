"""Read-time estimates of one layer from its page placement, without running MQSim.

Two figures per layer (``analysis_pages(..., timing=True)``):

* ``bound_ns`` - ideal HBF: max(effective rounds x tR, busiest core-die channel's transfers,
  busiest host link). No command overhead; sensing and transfers overlap perfectly.
* ``est_ns`` - MQSim's timing of the same reads on a multi-plane topology (``ocp``, ``Cx1
  --planes P``), round by round per core-die (MQSim) channel. A die's k-th round reads n(d, k)
  pages: the planes holding at least k pages (independent planes), or the pages at each lock-step
  group's k-th offset. A channel's round takes t(k) = cmd(max n) + tR + t_RB; with cache read
  (T-CACHE) the transfers of round k-1 and the other dies' commands overlap the sense of round k:

      T_c = t(1) + sum_{k>=2} max(t(k), x * N(c, k-1) + (m(c, k) - 1) * cmd(1)) + x * N(c, R_c)

  and without it every round waits for its transfers: T_c = sum_k (t(k) + x * N(c, k)). N(c, k)
  is the pages of round k on channel c, m(c, k) its dies with a k-th round, x the page transfer
  time. est = W + max_c T_c + t_host, W being the T-CMD coalescing window.

The estimate was fitted to MQSim with the HBF command model (``--read-plane-ns 0
--coalesce-ns 1 --cache-read``) and tR = 4.096 us; it is unvalidated outside it (ONFI multi-plane
commands of about 10 us hold the shared channel in ways it does not model). Chip-as-plane
topologies (16x64 etc.) have no estimate: there every page is its own command.
"""
import numpy as np

from .config import DEFAULT_DBSY_NS

HBF_COMMAND_MODEL = "--read-plane-ns 0 --coalesce-ns 1 --cache-read"


def command_ns(cfg, planes):
    """``Config.multiplane_read_command_ns`` for an array of plane counts."""
    cs, wc, wb, rr = cfg.onfi or (20, 25, 100, 20)
    planes = np.asarray(planes, dtype=np.int64)
    if cfg.read_plane_ns is not None:
        return cs + 6 * wc + wb + rr + (planes - 1) * cfg.read_plane_ns
    dbsy = DEFAULT_DBSY_NS if cfg.dbsy_ns is None else cfg.dbsy_ns
    return cs + planes * 6 * wc + (planes - 1) * dbsy + wb + rr


def hbf_command_model(cfg):
    """Whether ``cfg`` is the command model the estimate was validated on."""
    return cfg.read_plane_ns == 0 and cfg.coalesce_ns is not None and cfg.cache_read


def round_pages(cfg, placed, group, part=slice(None)):
    """n(d, k): pages that die d reads in its k-th round, as a (dies, rounds) array."""
    from .analysis import plane_geometry
    dies, per_die = plane_geometry(cfg)
    die, plane, offset = placed.die[part], placed.plane[part], placed.offset[part]
    if not len(die):
        return np.zeros((dies, 0), dtype=np.int64)
    span = int(offset.max()) + 1
    owner = die * (per_die // group) + plane // group
    keys, counts = np.unique(owner * span + offset, return_counts=True)
    owners = keys // span
    first = np.searchsorted(owners, owners)  # keys are sorted, so each owner's offsets are contiguous
    rank = np.arange(len(keys)) - first
    pages = np.zeros((dies, int(rank.max()) + 1), dtype=np.int64)
    np.add.at(pages, (owners // (per_die // group), rank), counts)
    return pages


def estimate_ns(cfg, placed, group, part=slice(None)):
    """``est_ns`` of the pages ``part`` (see the module docstring), or None for chip-as-plane."""
    if cfg.dies == cfg.planes == 1:
        return None
    pages = round_pages(cfg, placed, group, part)
    if not pages.shape[1]:
        return 0
    per_channel = cfg.chips * cfg.dies
    rounds = pages.shape[1]
    by_channel = pages.reshape(cfg.channels, per_channel, rounds)
    total = by_channel.sum(axis=1)  # N(c, k)
    widest = by_channel.max(axis=1)  # max n(d, k) over the channel's dies
    active = (by_channel > 0).sum(axis=1)  # m(c, k)
    x, single = cfg.transfer_ns, int(command_ns(cfg, 1))
    step = command_ns(cfg, np.maximum(widest, 1)) + cfg.read_ns + cfg.rb_ns  # t(k)
    used = total > 0
    last = np.where(used.any(axis=1), rounds - np.argmax(used[:, ::-1], axis=1), 0)  # R_c
    worst = 0
    for channel in range(cfg.channels):
        count = int(last[channel])
        if not count:
            continue
        t, n, m = step[channel, :count], total[channel, :count], active[channel, :count]
        if cfg.cache_read:
            overlap = np.maximum(t[1:], x * n[:-1] + (m[1:] - 1) * single)
            span = int(t[0] + overlap.sum() + x * n[-1])
        else:
            span = int((t + x * n).sum())
        worst = max(worst, span)
    return (cfg.coalesce_ns or 0) + worst + cfg.host_ns


def block_ids(cfg, placed):
    """Erase-block id of each page: (MQSim unit, static offset // pages per block)."""
    return placed.unit + cfg.units * (placed.offset // cfg.pages_per_block)
