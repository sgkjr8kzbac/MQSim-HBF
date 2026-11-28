"""Parse MQSim completions, XML statistics, and aggregate per-layer/step rows."""
import csv
from collections import defaultdict
from dataclasses import dataclass
from xml.etree import ElementTree as ET

from ...util import dump, write_table
from .schedule import metrics


@dataclass(frozen=True)
class Completion:
    arrival_ns: int
    enqueue_ns: int
    complete_ns: int
    start_lba: int
    lba_count: int
    op: str


def completions(path):
    with path.open(newline="") as stream:
        for arrival, enqueue, done, lba, size, op in csv.reader(stream):
            yield Completion(int(arrival), int(enqueue), int(done), int(lba), int(size), op)


def flash_stats(path):
    tree = ET.parse(path)
    ftl = tree.find(".//SSDDevice.FTL")
    if ftl is None:
        raise ValueError("MQSim output is missing FTL statistics")
    stats = {key: int(val) for key, val in ftl.attrib.items() if key.startswith("Issued_Flash")}
    stats["request_count"] = int(tree.findtext(".//Request_Count"))
    stats["gc_enqueued"] = sum(
        int(el.get("No_Of_Transactions_Enqueued"))
        for el in tree.iter() if ".GC_" in el.tag and el.get("No_Of_Transactions_Enqueued") is not None
    )
    return stats


def aggregate(cfg, group, keys):
    result = {**keys, "mqsim_ns": sum(row["mqsim_ns"] for _, row in group),
              **metrics(cfg, (req for batch, _ in group for req in batch.requests))}
    result["mq_est_unsplit_ns"] = result["mq_est_ns"]
    result["ideal_unsplit_ns"] = result["ideal_ns"]
    estimates = [row["mq_est_ns"] for _, row in group]
    result["mq_est_ns"] = None if None in estimates else sum(estimates)
    result["ideal_ns"] = sum(row["ideal_ns"] for _, row in group)
    result["estimate_gap_ns"] = None if result["mq_est_ns"] is None else result["mq_est_ns"] - result["ideal_ns"]
    result["mqsim_over_est"] = result["mqsim_ns"] / result["mq_est_ns"] if result["mq_est_ns"] else None
    if "layer" not in keys:
        # Each layer runs serially in the reconstructed step. Keep unsplit bounds additive.
        layers = defaultdict(list)
        for batch, row in group:
            layers[batch.layer].append((batch, row))
        per_layer = [metrics(cfg, (req for batch, _ in items for req in batch.requests)) for items in layers.values()]
        estimates = [layer["mq_est_ns"] for layer in per_layer]
        result["mq_est_unsplit_ns"] = None if None in estimates else sum(estimates)
        result["ideal_unsplit_ns"] = sum(layer["ideal_ns"] for layer in per_layer)
    return result
