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
        for name in ("page_bytes", "sector_bytes", "pages_per_block", "blocks_per_plane"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.page_bytes <= 0 or self.sector_bytes != 512 or self.page_bytes % self.sector_bytes:
            raise ValueError("MQSim requires 512 B sectors and sector-aligned pages")
        if (self.transfer_rate != 1000 or self.width <= 0
                or self.aggregate_gb_s % self.channels or self.page_bytes % (2 * self.width)):
            raise ValueError("Require rate=1000 and page divisible by 2*channel width")
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
        return self.page_bytes // self.width

    @property
    def read_command_ns(self):
        if self.onfi is None:
            return self.cmd_ns
        cs, wc, wb, rr = self.onfi
        return cs + 6 * wc + wb + rr

    def multiplane_read_command_ns(self, planes):
        """Bus time of one read command for ``planes`` planes (ONFI_Channel_NVDDR2, T-PLANE)."""
        cs, wc, wb, rr = self.onfi or (20, 25, 100, 20)
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

    def lba(self, lpa):
        if not 0 <= lpa < self.capacity_sectors // self.spp:
            raise ValueError(f"LPA {lpa} exceeds logical capacity")
        return lpa * self.spp

    def physical(self, lpa):
        self.lba(lpa)
        address = []
        for length in self.geometry:
            lpa, index = divmod(lpa, length)
            address.append(index)
        return tuple(address)

    def unit_lpa(self, unit, k, base=0):
        if not 0 <= unit < self.units or k < 0 or base < 0:
            raise ValueError("Invalid unit or page index")
        lpa = base + (unit - base) % self.units + k * self.units
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
    """Channels and chips per channel of a named topology or of any ``<channels>x<chips>``."""
    if named and topology in named:
        return tuple(named[topology])
    try:
        channels, chips = map(int, topology.split("x"))
    except (AttributeError, ValueError):
        raise ValueError(f"Topology {topology!r} is neither named nor <channels>x<chips>") from None
    return channels, chips


def load(topology="64x16", **overrides):
    params = json.loads(PARAMETERS.read_text())
    channels, chips = parse_topology(topology, params.pop("topologies"))
    reference = params.pop("calibration_reference")
    if reference["width_gb_s"] * reference["xfer_ns"] != params["page_bytes"]:
        raise ValueError("Reference NAND transfer calibration disagrees with the page size")
    return replace(Config(channels, chips, **params.pop("calibration"), **params), **overrides)


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
