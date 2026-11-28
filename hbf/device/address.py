"""Static HBF address mapping (OCP HBF v0.7.0 section 4.6, Figure 3 & Figure 4) and CWDP mapping.

Single source of truth for LPA <-> (channel, chip, die, plane, block, page), used by Config,
rounds analysis, timing, placement, and MQSim scheduling.
"""
from dataclasses import dataclass

OCP_DIES_PER_HOST_CHANNEL = 4  # Figure 4: 2 core-die channels x 2 dies per host channel


@dataclass
class Placed:
    """Static physical coordinates of an array of host LPAs.

    * ``die``, ``plane`` - HBF die and plane/bank indices (matching ``analysis.locate``).
    * ``offset`` - static page offset inside the plane (``lpa // cfg.units``).
    * ``channel`` - core-die (MQSim) channel (``device_lpa % cfg.channels``).
    * ``host`` - host (UCIe) channel (``lpa % cfg.host_channels``; callers relying on it must
      call ``check_host_channels(cfg)``).
    * ``unit`` - MQSim (channel, chip, die, plane) unit index (``device_lpa % cfg.units``).
    """
    die: object
    plane: object
    offset: object
    channel: object
    host: object
    unit: object


def device_lpa(cfg, lpa):
    """Scalar host LPA -> MQSim device LPA so CWDP placement realises ``cfg.address_map``."""
    if cfg.address_map == "cwdp":
        return lpa
    hosts, banks = cfg.host_channels, cfg.planes
    host, local = lpa % hosts, lpa // hosts
    bank = local % banks
    die = local // banks % OCP_DIES_PER_HOST_CHANNEL
    offset = local // (banks * OCP_DIES_PER_HOST_CHANNEL)
    channel, chip = 2 * host + die % 2, die // 2
    return channel + cfg.channels * (chip + cfg.chips * (bank + banks * offset))


def host_lpa(cfg, device):
    """Scalar inverse of ``device_lpa``."""
    if cfg.address_map == "cwdp":
        return device
    hosts, banks = cfg.host_channels, cfg.planes
    channel, rest = device % cfg.channels, device // cfg.channels
    chip, rest = rest % cfg.chips, rest // cfg.chips
    bank, offset = rest % banks, rest // banks
    host, die = channel // 2, channel % 2 + 2 * chip
    local = bank + banks * (die + OCP_DIES_PER_HOST_CHANNEL * offset)
    return host + hosts * local


def check_host_channels(cfg):
    if cfg.address_map == "cwdp" and cfg.channels % cfg.host_channels:
        raise ValueError("Host channels need a channel count that is a multiple of 16")


def block_page(cfg, lpa):
    """Scalar ``(unit, block_in_plane, page_in_block)`` of a host LPA (section 4.6 & 5.4.1)."""
    unit = device_lpa(cfg, lpa) % cfg.units
    block, page = divmod(lpa // cfg.units, cfg.pages_per_block)
    return unit, block, page


def device_lpas(cfg, lpa):
    """Vectorised ``device_lpa`` for an array of host LPAs."""
    import numpy as np
    lpa = np.asarray(lpa, dtype=np.int64)
    if cfg.address_map == "cwdp":
        return lpa
    hosts, banks = cfg.host_channels, cfg.planes
    host, local = lpa % hosts, lpa // hosts
    bank = local % banks
    die = local // banks % OCP_DIES_PER_HOST_CHANNEL
    offset = local // (banks * OCP_DIES_PER_HOST_CHANNEL)
    channel, chip = 2 * host + die % 2, die // 2
    return channel + cfg.channels * (chip + cfg.chips * (bank + banks * offset))


def host_lpas(cfg, device):
    """Vectorised inverse of ``device_lpas``."""
    import numpy as np
    device = np.asarray(device, dtype=np.int64)
    if cfg.address_map == "cwdp":
        return device
    hosts, banks = cfg.host_channels, cfg.planes
    channel, rest = device % cfg.channels, device // cfg.channels
    chip, rest = rest % cfg.chips, rest // cfg.chips
    bank, offset = rest % banks, rest // banks
    host, die = channel // 2, channel % 2 + 2 * chip
    local = bank + banks * (die + OCP_DIES_PER_HOST_CHANNEL * offset)
    return host + hosts * local


def place(cfg, lpa):
    """Static physical locations of an array of host LPAs (offset ``lpa // units``)."""
    import numpy as np
    lpa = np.asarray(lpa, dtype=np.int64)
    if len(lpa) and (lpa.min() < 0 or lpa.max() >= cfg.capacity_sectors // cfg.spp):
        bad = lpa[(lpa < 0) | (lpa >= cfg.capacity_sectors // cfg.spp)][0]
        raise ValueError(f"LPA {bad} exceeds logical capacity")
    device = device_lpas(cfg, lpa)
    channels, chips, dies, planes = cfg.geometry
    channel, rest = device % channels, device // channels
    chip, rest = rest % chips, rest // chips
    die, rest = rest % dies, rest // dies
    plane = rest % planes
    if dies == planes == 1:
        hbf_die, hbf_plane = channel, chip
    else:
        hbf_die, hbf_plane = (channel * chips + chip) * dies + die, plane
    return Placed(hbf_die, hbf_plane, lpa // cfg.units, channel, lpa % cfg.host_channels, device % cfg.units)


def block_ids(cfg, placed):
    """Global erase-block ID of each page: ``unit + units * (offset // pages_per_block)``."""
    return placed.unit + cfg.units * (placed.offset // cfg.pages_per_block)


def block_and_page(cfg, placed):
    """Global erase-block ID and in-block page index arrays for ``placed``."""
    import numpy as np
    return block_ids(cfg, placed), (placed.offset % cfg.pages_per_block).astype(np.int64)
