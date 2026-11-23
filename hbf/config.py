"""MQSim geometry, address conversion and XML validation."""
import json
from dataclasses import dataclass, replace
from decimal import Decimal
from math import prod
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs"
PARAMETERS = CONFIG / "parameters.json"


@dataclass(frozen=True)
class Config:
    channels: int
    chips: int
    dies: int = 1
    planes: int = 1
    page_bytes: int = 4096
    sector_bytes: int = 512
    pages_per_block: int = 256
    blocks_per_plane: int = 64
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
    batch_pages: int = 8192
    interval_ns: int = 100_000_000
    cmd_ns: int = 290
    rb_ns: int = 20
    host_ns: int = 1
    suspend: str = "NONE"
    suspend_ns: int = 100000
    suspend_threshold_ns: int = 100000
    scheduler: str = "PRIORITY_OUT_OF_ORDER"
    onfi: tuple[int, int, int, int] | None = None

    def __post_init__(self):
        if any(type(n) is not int or not 1 <= n <= 65535 for n in self.geometry):
            raise ValueError("Invalid flash topology")
        if self.page_bytes <= 0 or self.sector_bytes != 512 or self.page_bytes % self.sector_bytes:
            raise ValueError("MQSim requires 512 B sectors and sector-aligned pages")
        if (self.transfer_rate != 1000 or self.width <= 0
                or self.aggregate_gb_s % self.channels or self.page_bytes % (2 * self.width)):
            raise ValueError("Require rate=1000 and page divisible by 2*channel width")
        if not 1 <= self.queue_depth <= 65535 or not 1 <= self.queue_fetch <= 65535:
            raise ValueError("Queue parameters must be nonzero uint16")
        if self.interval_ns <= 0 or not 1 <= self.batch_pages < self.queue_depth:
            raise ValueError("Invalid batch interval or size")
        if (min(self.pages_per_block, self.blocks_per_plane, self.read_ns, self.program_ns,
                self.erase_ns, self.pcie_lanes, self.pcie_lane_gb_s) <= 0
                or not 0 <= self.op_ratio < 1 or not 0 < self.gc_threshold < 1
                or self.suspend not in {"NONE", "PROGRAM"}
                or self.scheduler not in {"PRIORITY_OUT_OF_ORDER", "OUT_OF_ORDER"}
                or self.onfi is not None and (len(self.onfi) != 4 or min(self.onfi) < 0)):
            raise ValueError("Invalid flash timing, capacity, or controller settings")
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
    def capacity_sectors(self):
        plane = self.blocks_per_plane * self.pages_per_block * self.spp
        return self.units * int(plane * (1 - Decimal(str(self.op_ratio))))

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


def load(topology="64x16", **overrides):
    params = json.loads(PARAMETERS.read_text())
    channels, chips = params.pop("topologies")[topology]
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
