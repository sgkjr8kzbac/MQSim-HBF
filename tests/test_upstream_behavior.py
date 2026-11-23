"""Isolated raw-MQSim probes for settings that the normal pipeline rejects."""
import subprocess
from xml.etree import ElementTree as ET

from hbf.config import device_xml, workload_xml
from hbf.simulator import completions, dump, flash_stats, version
from test_validation import micro, phy


def raw(binary, path, cfg, trace, *, device=None, workload=None, enable_i3=True):
    path.mkdir()
    (path / "trace.txt").write_text(trace)
    for name, render, overrides in [("device.xml", device_xml, device),
                                    ("workload.xml", workload_xml, workload)]:
        render(cfg, path / name)
        if overrides:
            tree = ET.parse(path / name)
            for key, value in overrides.items():
                tree.find(f".//{key}").text = str(value)
            tree.write(path / name, encoding="utf-8", xml_declaration=True)
    env = {"HBF_I1": "completions.csv", "HBF_I2": "physical.csv"}
    if enable_i3:
        env["HBF_I3"] = "1"
    dump(path / "meta.json", {"source": "raw MQSim parser probe", "env": env})
    dump(path / "version.json", version([name[4:] for name in env]))
    with (path / "stdout.log").open("w") as log:
        child = subprocess.run([str(binary), "-i", "device.xml", "-w", "workload.xml"],
                               cwd=path, env=env, stdin=subprocess.DEVNULL,
                               stdout=log, stderr=log, timeout=30)
    return child.returncode


def test_upstream_integer_transfer_rate(binary, evidence):
    cfg = micro()
    observed = {}
    for rate in (500, 501, 750, 1000, 1001):
        path = evidence / f"rate-{rate}"
        assert raw(binary, path, cfg, "0 0 0 8 1\n",
                   device={"Channel_Transfer_Rate": rate}) == 0
        events = {kind: time for time, kind, *_ in phy(path)}
        observed[rate] = events["data_done"] - events["data_start"]
    dump(evidence / "rate-summary.json", observed)
    assert observed == {500: 128, 501: 64, 750: 64, 1000: 64, 1001: 0}


def test_upstream_queue_overflow(binary, evidence):
    cfg = micro()
    observed = {}
    for setting in ("Queue_Fetch_Size", "IO_Queue_Depth"):
        path = evidence / f"overflow-{setting}"
        assert raw(binary, path, cfg, "0 0 0 8 1\n", device={setting: 65536}) == 0
        observed[setting] = flash_stats(path / "workload_scenario_1.xml")["Issued_Flash_Read_CMD"]
    dump(evidence / "overflow-summary.json", observed)
    assert observed == {"Queue_Fetch_Size": 0, "IO_Queue_Depth": 0}


def test_upstream_id_buffer(binary, evidence):
    cfg = micro(channels=320, chips=1, aggregate_gb_s=1280)
    result = {}
    for enabled in (False, True):
        path = evidence / f"id-buffer-{enabled}"
        status = raw(binary, path, cfg, "0 0 2552 8 1\n", enable_i3=enabled)
        result[str(enabled)] = {"status": status, "channel": None}
        if status == 0 and (path / "physical.csv").exists():
            result[str(enabled)]["channel"] = phy(path)[0][3]
    dump(evidence / "id-buffer-summary.json", result)
    assert result["True"] == {"status": 0, "channel": 319}
    assert result["False"]["channel"] != 319


def test_upstream_trace_parser(binary, evidence):
    cfg = micro()
    path = evidence / "trace-malformed"
    assert raw(binary, path, cfg, "0 0 0 8 1\n0 0 8 8\n100 0 16 8 1\n") == 0
    assert len(list(completions(path / "completions.csv"))) == 1
    path = evidence / "trace-time-unit"
    assert raw(binary, path, cfg, "100 0 0 8 1\n123 0 8 8 1\n",
               workload={"Time_Unit": "MICROSECOND"}) == 0
    assert sorted(row.arrival_ns for row in completions(path / "completions.csv")) == [100, 123]


def test_upstream_first_read_allocates(binary, evidence):
    cfg = micro(chips=1)
    path = evidence / "first-read"
    assert raw(binary, path, cfg, "0 0 0 8 1\n10000 0 8 8 1\n20000 0 0 8 1\n") == 0
    locations = [(b, p, lpa) for _, event, _, _, _, _, _, b, p, lpa in phy(path) if event == "issue"]
    dump(evidence / "first-read-summary.json", locations)
    assert locations == [(0, 0, 0), (0, 1, 1), (0, 0, 0)]


def test_upstream_slc_uses_lsb(binary, evidence):
    cfg = micro()
    path = evidence / "slc-lsb"
    assert raw(binary, path, cfg, "0 0 0 8 1\n0 0 32 8 1\n",
               device={"Page_Read_Latency_CSB": 20000, "Page_Read_Latency_MSB": 30000}) == 0
    intervals = {}
    for time, event, _, _, _, _, _, _, _, lpa in phy(path):
        intervals.setdefault(lpa, {})[event] = time
    assert all(events["ready"] - events["cmd_done"] == 4116 for events in intervals.values())


def test_upstream_pcie_efficiency(binary, evidence):
    cfg = micro(channels=64, chips=16, aggregate_gb_s=1024, pcie_lane_gb_s=1)
    path = evidence / "pcie-low-bandwidth"
    trace = "".join(f"0 0 {cfg.lba(n)} 8 1\n" for n in range(8192))
    assert raw(binary, path, cfg, trace) == 0
    done = list(completions(path / "completions.csv"))
    service = max(row.complete_ns for row in done)
    efficiency = len(done) * cfg.page_bytes / service / (cfg.pcie_lanes * cfg.pcie_lane_gb_s)
    dump(path / "summary.json", {"ns": service, "efficiency": efficiency, "completed": len(done)})
    assert len(done) == 8192 and .80 <= efficiency <= .83
