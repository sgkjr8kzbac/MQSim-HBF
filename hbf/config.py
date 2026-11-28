"""MQSim geometry, address conversion and XML validation."""
import json
from dataclasses import dataclass, replace
from math import prod
from pathlib import Path
from xml.etree import ElementTree as ET

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parent
CONFIG = PACKAGE / "configs"
PARAMETERS = CONFIG / "parameters.json"
MAX_BATCH_PAGES = 8192
DEFAULT_READ_COMMAND_NS = 290
DEFAULT_DBSY_NS = 500
MAX_PLANES = 64
# None keeps upstream multi-plane grouping (same page index); see hbf/patches/T-PLANE.patch.
PLANE_RULES = (None, "independent", "lockstep")
ADDRESS_MAPS = ("cwdp", "ocp")
OCP_DIES_PER_HOST_CHANNEL = 4  # Figure 4: 2 core-die channels x 2 dies per host channel
OCP_HOST_CHANNELS = 16  # section 4.1: "Support 16 host channels"


@dataclass(frozen=True)
class Config:
    channels: int
    chips: int
    dies: int = 1
    planes: int = 1
    page_bytes: int = 4096
    sector_bytes: int = 512
    pages_per_block: int = 256
    blocks_per_plane: int = 512
    read_ns: int = 4096
    program_ns: int = 100000
    erase_ns: int = 1000000
    op_ratio: float = .07
    gc_threshold: float = .05
    aggregate_gb_s: int = 1024
    transfer_rate: int = 1000
    pcie_lane_gb_s: int = 256
    pcie_lanes: int = 16
    queue_depth: int = 65535
    queue_fetch: int = 65535
    batch_pages: int = MAX_BATCH_PAGES
    interval_ns: int = 100_000_000
    cmd_ns: int = DEFAULT_READ_COMMAND_NS
    rb_ns: int = 20
    host_ns: int = 1
    suspend: str = "NONE"
    suspend_ns: int = 100000
    suspend_threshold_ns: int = 100000
    scheduler: str = "PRIORITY_OUT_OF_ORDER"
    onfi: tuple[int, int, int, int] | None = None
    plane_rule: str | None = None
    dbsy_ns: int | None = None
    coalesce_ns: int | None = None
    read_plane_ns: int | None = None
    cache_read: bool = False
    address_map: str = "cwdp"

    def __post_init__(self):
        if any(type(n) is not int or not 1 <= n <= 65535 for n in self.geometry):
            raise ValueError("Invalid flash topology")
        if self.planes > MAX_PLANES:
            raise ValueError(f"MQSim multi-plane commands support at most {MAX_PLANES} planes per die (T-PLANE)")
        if self.plane_rule not in PLANE_RULES:
            raise ValueError(f"plane_rule must be one of {PLANE_RULES}")
        if self.dbsy_ns is not None and (self.plane_rule is None or type(self.dbsy_ns) is not int
                                         or not 0 <= self.dbsy_ns < 2**32):
            raise ValueError("A t_DBSY override needs a T-PLANE rule and a uint32 value")
        if self.read_plane_ns is not None and (self.plane_rule is None or type(self.read_plane_ns) is not int
                                               or not 0 <= self.read_plane_ns < 2**32):
            raise ValueError("A per-plane read command step needs a T-PLANE rule and a uint32 value")
        if self.coalesce_ns is not None and (type(self.coalesce_ns) is not int or self.coalesce_ns <= 0
                                             or self.scheduler != "PRIORITY_OUT_OF_ORDER"):
            raise ValueError("A coalescing window must be a positive integer with PRIORITY_OUT_OF_ORDER")
        if self.address_map not in ADDRESS_MAPS:
            raise ValueError(f"address_map must be one of {ADDRESS_MAPS}")
        if self.address_map == "ocp" and (self.chips != 2 or self.dies != 1 or self.channels % 2):
            raise ValueError("The OCP address map models Figure 4: 2 core-die channels per host channel, "
                             "2 dies (chips) per core-die channel and one MQSim die per chip")
        if type(self.cache_read) is not bool or self.cache_read and (
                self.scheduler != "PRIORITY_OUT_OF_ORDER" or self.suspend != "NONE"):
            raise ValueError("Cache read (T-CACHE) needs PRIORITY_OUT_OF_ORDER and no suspension")
        for name in ("page_bytes", "sector_bytes", "pages_per_block", "blocks_per_plane"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.page_bytes <= 0 or self.sector_bytes != 512 or self.page_bytes % self.sector_bytes:
            raise ValueError("MQSim requires 512 B sectors and sector-aligned pages")
        if (self.transfer_rate != 1000 or self.width <= 0
                or self.aggregate_gb_s % self.channels or self.page_bytes // self.width < 2):
            raise ValueError("Require rate=1000, a whole number of GB/s per channel and a page of at "
                             "least two bus cycles")
        if not 1 <= self.queue_depth <= 65535 or not 1 <= self.queue_fetch <= 65535:
            raise ValueError("Queue parameters must be nonzero uint16")
        if self.interval_ns <= 0:
            raise ValueError("Batch interval must be positive")
        if (type(self.batch_pages) is not int or not 1 <= self.batch_pages <= MAX_BATCH_PAGES
                or self.batch_pages >= self.queue_depth):
            raise ValueError(f"Batch size must be between 1 and {MAX_BATCH_PAGES} pages and below queue depth")
        if (min(self.read_ns, self.program_ns,
                self.erase_ns, self.pcie_lanes, self.pcie_lane_gb_s) <= 0
                or not 0 <= self.op_ratio < 1 or not 0 < self.gc_threshold < 1
                or self.suspend not in {"NONE", "PROGRAM"}
                or self.scheduler not in {"PRIORITY_OUT_OF_ORDER", "OUT_OF_ORDER"}
                or self.onfi is not None and (len(self.onfi) != 4 or
                    any(type(ns) is not int or not 0 <= ns < 2**32 for ns in self.onfi))):
            raise ValueError("Invalid flash timing, capacity, or controller settings")
        if self.onfi is None and self.cmd_ns != DEFAULT_READ_COMMAND_NS:
            raise ValueError("Non-default read command latency requires explicit ONFI XML timings")
        if not 0 < self.capacity_sectors * self.sector_bytes < 2**41:
            raise ValueError("Logical capacity must be positive and below 2 TiB")

    @property
    def geometry(self):
        return self.channels, self.chips, self.dies, self.planes

    @property
    def units(self):
        return prod(self.geometry)

    @property
    def spp(self):
        return self.page_bytes // self.sector_bytes

    @property
    def width(self):
        return self.aggregate_gb_s // self.channels

    @property
    def transfer_ns(self):
        """Page transfer time as MQSim computes it: (bytes / width / 2) two-unit cycles of 2 ns.

        NVDDR2DataOutTransferTime divides in integers, so a width that does not divide the page
        (e.g. 96 or 192 GB/s for the OCP grades 1.536 and 3.072 TB/s over 16 channels) is truncated:
        42 instead of 42.67 ns and 20 instead of 21.33 ns. ``transfer_error`` gives the deviation.
        """
        return self.page_bytes // self.width // 2 * 2

    @property
    def transfer_error(self):
        """Relative error of ``transfer_ns`` against the exact page_bytes / width."""
        return self.transfer_ns / (self.page_bytes / self.width) - 1

    @property
    def read_command_ns(self):
        if self.onfi is None:
            return self.cmd_ns
        cs, wc, wb, rr = self.onfi
        return cs + 6 * wc + wb + rr

    def multiplane_read_command_ns(self, planes):
        """Bus time of one read command for ``planes`` planes (ONFI_Channel_NVDDR2, T-PLANE)."""
        cs, wc, wb, rr = self.onfi or (20, 25, 100, 20)
        if self.read_plane_ns is not None:
            return cs + 6 * wc + wb + rr + (planes - 1) * self.read_plane_ns
        dbsy = DEFAULT_DBSY_NS if self.dbsy_ns is None else self.dbsy_ns
        return cs + planes * 6 * wc + (planes - 1) * dbsy + wb + rr

    @property
    def capacity_sectors(self):
        plane = self.blocks_per_plane * self.pages_per_block * self.spp
        if plane >= 2**32:
            raise ValueError("Physical sectors per plane exceed MQSim's unsigned 32-bit range")
        # MQSim multiplies in double precision and truncates each plane before summing.
        return self.units * int(plane * (1.0 - self.op_ratio))

    @property
    def gc_threshold_blocks(self):
        # GC_and_WL_Unit_Base::GC_and_WL_Unit_Base: max(floor(gc * B), 1), then at least the
        # 10 in-flight GC requests per plane that SSD_Device::SSD_Device passes in.
        return max(int(self.gc_threshold * self.blocks_per_plane), 1, 10)

    @property
    def gc_page_budget(self):
        """Fresh pages one plane may allocate while the free block pool stays at the GC threshold.

        Flash_Block_Manager_Base takes three write-frontier blocks (data, translation, GC) per
        flow at start, so the pool begins at B - 3 for the single flow used here. Filling the
        k-th data block takes a new one at once, so the pool falls below the threshold when page
        (B - 3 - threshold + 1) * pages_per_block is allocated. With invalid pages (rewrites)
        GC then erases; with only valid pages upstream MQSim later livelocks.
        """
        blocks = self.blocks_per_plane - 3 - self.gc_threshold_blocks + 1
        return max(blocks * self.pages_per_block - 1, 0)

    @property
    def host_channels(self):
        """Host (UCIe) channels: ``ocp`` has two core-die channels per host channel; ``cwdp`` takes
        the spec's 16 host channels (section 4.1) with 4 KiB interleave, or fewer MQSim channels."""
        if self.address_map == "ocp":
            return self.channels // 2
        return min(self.channels, OCP_HOST_CHANNELS)

    def device_lpa(self, lpa):
        """The LPA that MQSim sees, so that its CWDP placement realises ``address_map``.

        ``cwdp`` is MQSim's own order (channel, chip, die, plane). ``ocp`` follows the OCP HBF spec:
        host addresses are interleaved in 4 KiB units over the host channels (Figure 3, n = 6);
        inside a host channel the low 4 bits of the 4 KiB unit select the bank and the next 2 bits
        the die (section 4.6); Figure 4 puts dies 0 and 2 on core-die channel 0 and dies 1 and 3 on
        core-die channel 1. The page offset inside a bank is ``lpa // units`` in both maps.
        """
        if self.address_map == "cwdp":
            return lpa
        hosts, banks = self.host_channels, self.planes
        host, local = lpa % hosts, lpa // hosts
        bank, die, offset = local % banks, local // banks % OCP_DIES_PER_HOST_CHANNEL, local // (
            banks * OCP_DIES_PER_HOST_CHANNEL)
        channel, chip = 2 * host + die % 2, die // 2
        return channel + self.channels * (chip + self.chips * (bank + banks * offset))

    def host_lpa(self, device):
        """Inverse of ``device_lpa``."""
        if self.address_map == "cwdp":
            return device
        hosts, banks = self.host_channels, self.planes
        channel, rest = device % self.channels, device // self.channels
        chip, rest = rest % self.chips, rest // self.chips
        bank, offset = rest % banks, rest // banks
        host, die = channel // 2, channel % 2 + 2 * chip
        local = bank + banks * (die + OCP_DIES_PER_HOST_CHANNEL * offset)
        return host + hosts * local

    def host_channel(self, lpa):
        """Host (UCIe) channel of an LPA under the spec's 4 KiB interleave (Figure 3, n = 6).

        For ``cwdp`` with 16 or more MQSim channels the host channel is the MQSim channel modulo 16,
        so 64x16 and 256x4 keep the spec's 16 host channels (and 16 MOCS budgets)."""
        if self.address_map == "cwdp" and self.channels % self.host_channels:
            raise ValueError("Host channels need a channel count that is a multiple of 16")
        return lpa % self.host_channels

    def unit(self, lpa):
        """Index of the MQSim (channel, chip, die, plane) that holds ``lpa``."""
        return self.device_lpa(lpa) % self.units

    def lba(self, lpa):
        """MQSim LBA of a host LPA (through ``device_lpa``)."""
        if not 0 <= lpa < self.capacity_sectors // self.spp:
            raise ValueError(f"LPA {lpa} exceeds logical capacity")
        return self.device_lpa(lpa) * self.spp

    def physical(self, lpa):
        self.lba(lpa)
        lpa = self.device_lpa(lpa)
        address = []
        for length in self.geometry:
            lpa, index = divmod(lpa, length)
            address.append(index)
        return tuple(address)

    def unit_lpa(self, unit, k, base=0):
        if not 0 <= unit < self.units or k < 0 or base < 0:
            raise ValueError("Invalid unit or page index")
        if self.address_map != "cwdp" and base % self.units:
            raise ValueError("unit_lpa with the ocp address map needs a base that is a multiple of the unit count")
        lpa = self.host_lpa(base + (unit - base) % self.units + k * self.units)
        self.lba(lpa)
        return lpa

    def inverse(self, address, k=0, base=0):
        if len(address) != 4 or any(not 0 <= p < d for p, d in zip(address, self.geometry)):
            raise ValueError("Physical location outside topology")
        unit = 0
        for index, length in reversed(list(zip(address, self.geometry))):
            unit = unit * length + index
        return self.unit_lpa(unit, k, base)


def parse_topology(topology, named=None):
    """Config fields of a named topology or of any ``<channels>x<chips>``.

    A named topology is either ``[channels, chips]`` or a dict of Config fields.
    """
    if named and topology in named:
        value = named[topology]
        return dict(value) if isinstance(value, dict) else {"channels": value[0], "chips": value[1]}
    try:
        channels, chips = map(int, topology.split("x"))
    except (AttributeError, ValueError):
        raise ValueError(f"Topology {topology!r} is neither named nor <channels>x<chips>") from None
    return {"channels": channels, "chips": chips}


def load(topology="64x16", **overrides):
    params = json.loads(PARAMETERS.read_text())
    fields = parse_topology(topology, params.pop("topologies"))
    reference = params.pop("calibration_reference")
    if reference["width_gb_s"] * reference["xfer_ns"] != params["page_bytes"]:
        raise ValueError("Reference NAND transfer calibration disagrees with the page size")
    return replace(Config(**(params.pop("calibration") | params | fields)), **overrides)


def xml_settings(cfg):
    fields = {
        "PCIe_Lane_Bandwidth": cfg.pcie_lane_gb_s, "PCIe_Lane_Count": cfg.pcie_lanes,
        "Enabled_Preconditioning": "false", "IO_Queue_Depth": cfg.queue_depth,
        "Queue_Fetch_Size": cfg.queue_fetch, "Ideal_Mapping_Table": "true",
        "Plane_Allocation_Scheme": "CWDP", "HostInterface_Type": "NVME",
        "Transaction_Scheduling_Policy": cfg.scheduler, "Overprovisioning_Ratio": cfg.op_ratio,
        "GC_Exec_Threshold": cfg.gc_threshold, "Dynamic_Wearleveling_Enabled": "false",
        "Static_Wearleveling_Enabled": "false", "Flash_Channel_Count": cfg.channels,
        "Flash_Channel_Width": cfg.width, "Channel_Transfer_Rate": cfg.transfer_rate,
        "Chip_No_Per_Channel": cfg.chips, "Die_No_Per_Chip": cfg.dies, "Plane_No_Per_Die": cfg.planes,
        "Block_No_Per_Plane": cfg.blocks_per_plane, "Page_No_Per_Block": cfg.pages_per_block,
        "Page_Capacity": cfg.page_bytes, "Flash_Technology": "SLC",
        "Block_Erase_Latency": cfg.erase_ns, "CMD_Suspension_Support": cfg.suspend,
        "Suspend_Program_Time": cfg.suspend_ns,
        "Preferred_suspend_write_time_for_read": cfg.suspend_threshold_ns,
    }
    for page_type in ("LSB", "CSB", "MSB"):
        fields[f"Page_Read_Latency_{page_type}"] = cfg.read_ns
        fields[f"Page_Program_Latency_{page_type}"] = cfg.program_ns
    if cfg.onfi is not None:
        fields.update({f"ONFI_t_{key}": val for key, val in zip(("CS", "WC", "WB", "RR"), cfg.onfi)})
    if cfg.dbsy_ns is not None:
        fields["ONFI_t_DBSY"] = cfg.dbsy_ns
    return fields


def check_xml(path, expected):
    tree = ET.parse(path)
    for key, value in expected.items():
        matches = tree.findall(f".//{key}")
        if len(matches) != 1 or matches[0].text != str(value):
            raise ValueError(f"{path}: expected {key}={value}, got {[m.text for m in matches]}")
    return tree


def device_xml(cfg, path):
    tree = ET.parse(CONFIG / "template.xml")
    fields = xml_settings(cfg)
    for key, value in fields.items():
        element = tree.find(f".//{key}")
        if element is None:
            element = ET.SubElement(tree.find("Device_Parameter_Set"), key)
        element.text = str(value)
    ET.indent(tree)
    tree.write(path, encoding="utf-8", xml_declaration=True)
    check_xml(path, fields)


def workload_fields(cfg):
    fields = {
        "Priority_Class": "HIGH", "Device_Level_Data_Caching_Mode": "TURNED_OFF",
        "Initial_Occupancy_Percentage": 0, "File_Path": "trace.txt",
        "Percentage_To_Be_Executed": 100, "Relay_Count": 1, "Time_Unit": "NANOSECOND",
    }
    for name, count in zip(("Channel", "Chip", "Die", "Plane"), cfg.geometry):
        fields[f"{name}_IDs"] = ",".join(map(str, range(count)))
    return fields


def workload_xml(cfg, path):
    root = ET.Element("MQSim_IO_Scenarios")
    scenario = ET.SubElement(root, "IO_Scenario")
    flow = ET.SubElement(scenario, "IO_Flow_Parameter_Set_Trace_Based")
    fields = workload_fields(cfg)
    for key, value in fields.items():
        ET.SubElement(flow, key).text = str(value)
    tree = ET.ElementTree(root)
    ET.indent(tree)
    tree.write(path, encoding="utf-8", xml_declaration=True)
    check_xml(path, fields)
