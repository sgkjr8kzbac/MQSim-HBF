"""Parse HBFSim session receipts, live physical heat streams, wear snapshots, and heatmaps."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def parse_live_heatmap(path: Path) -> tuple[dict, list[dict]]:
    """Read an ``hbfsim.hbf_physical_live.v3`` JSONL stream into ``(header, events)``."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"Empty HBFSim heatmap stream: {path}")
    rows = [json.loads(line) for line in lines if line.strip()]
    header = rows[0]
    if header.get("schema") != "hbfsim.hbf_physical_live.v3" or header.get("kind") != "header":
        raise ValueError(f"Unexpected live heatmap header: {header}")
    return header, rows[1:]


def parse_heatmap_v1(summary_json: Path, scenario: str | None = None) -> dict:
    """Extract the ``hbfsim.address_heatmap.v1`` object from an ``hbfsim-reference`` summary JSON."""
    data = json.loads(Path(summary_json).read_text(encoding="utf-8"))
    scenarios = data.get("scenarios", [])
    if not scenarios:
        raise ValueError(f"No scenarios in HBFSim summary JSON: {summary_json}")
    chosen = scenarios[0] if scenario is None else next(
        (s for s in scenarios if s.get("name") == scenario), None
    )
    if chosen is None:
        raise ValueError(f"Scenario {scenario!r} not found in {summary_json}")
    heatmap = chosen.get("address_heatmap")
    if not heatmap or heatmap.get("schema") != "hbfsim.address_heatmap.v1":
        raise ValueError(f"Missing hbfsim.address_heatmap.v1 in scenario {chosen.get('name')!r}")
    return heatmap


def parse_wear_snapshot(receipt: dict) -> dict:
    """Normalize a ``WEAR_SNAPSHOT`` receipt from ``hbfsim``."""
    snap = receipt.get("wear_snapshot", receipt)
    counts = [int(x) for x in snap.get("block_erase_counts", [])]
    erase_sum = int(snap.get("block_erase_count_sum", sum(counts)))
    return {
        "writable_blocks": int(snap.get("writable_blocks", len(counts))),
        "block_erase_count_sum": erase_sum,
        "workload_erases": int(snap.get("workload_erases", erase_sum)),
        "block_erase_counts": counts,
    }


def extract_hbf_delta(batch_receipt: dict) -> dict:
    """Extract key per-batch HBF counters and timing from a ``SimulationSession`` batch receipt."""
    hbf = batch_receipt.get("devices", {}).get("hbf", {})
    if not hbf:
        hbf = batch_receipt.get("device_delta", {}).get("hbf", {})
    return {
        "batch_origin_ns": float(batch_receipt.get("batch_origin_ns", 0.0)),
        "first_issue_ns": float(batch_receipt.get("first_issue_ns", 0.0)),
        "blocking_finish_ns": float(batch_receipt.get("blocking_finish_ns", 0.0)),
        "finish_ns": float(batch_receipt.get("finish_ns", 0.0)),
        "elapsed_ns": float(batch_receipt.get("elapsed_ns", 0.0)),
        "total_elapsed_ns": float(batch_receipt.get("total_elapsed_ns", 0.0)),
        "page_reads": int(hbf.get("page_reads", 0)),
        "page_programs": int(hbf.get("page_programs", 0)),
        "raw_physical_programs": int(hbf.get("raw_physical_programs", 0)),
        "block_erases": int(hbf.get("block_erases", 0)),
        "read_buffer_hits": int(hbf.get("read_buffer_hits", 0)),
        "read_buffer_misses": int(hbf.get("read_buffer_misses", 0)),
        "physical_read_bytes": int(hbf.get("physical_read_bytes", 0)),
        "physical_write_bytes": int(hbf.get("physical_write_bytes", 0)),
        "hbf_delta": hbf,
    }


def aggregate_layer_rows(batch_rows: list[dict]) -> list[dict]:
    """Sum sub-batch durations and page counts into per-(step, layer) rows."""
    grouped = defaultdict(list)
    for row in batch_rows:
        grouped[(row["step"], row["layer"])].append(row)
    out = []
    for (step, layer), items in grouped.items():
        out.append({
            "step": step,
            "layer": layer,
            "subbatches": len(items),
            "read_pages": sum(r["read_pages"] for r in items),
            "write_pages": sum(r["write_pages"] for r in items),
            "hbfsim_ns": sum(r["hbfsim_ns"] for r in items),
            "media_page_reads": sum(r["media_page_reads"] for r in items),
            "read_buffer_hits": sum(r["read_buffer_hits"] for r in items),
            "read_buffer_misses": sum(r["read_buffer_misses"] for r in items),
            "page_programs": sum(r["page_programs"] for r in items),
            "block_erases": sum(r["block_erases"] for r in items),
        })
    return out


def aggregate_step_rows(batch_rows: list[dict]) -> list[dict]:
    """Sum sub-batch durations and page counts into per-step rows."""
    grouped = defaultdict(list)
    for row in batch_rows:
        grouped[row["step"]].append(row)
    out = []
    for step, items in grouped.items():
        out.append({
            "step": step,
            "read_pages": sum(r["read_pages"] for r in items),
            "write_pages": sum(r["write_pages"] for r in items),
            "hbfsim_ns": sum(r["hbfsim_ns"] for r in items),
            "media_page_reads": sum(r["media_page_reads"] for r in items),
            "read_buffer_hits": sum(r["read_buffer_hits"] for r in items),
            "read_buffer_misses": sum(r["read_buffer_misses"] for r in items),
            "page_programs": sum(r["page_programs"] for r in items),
            "block_erases": sum(r["block_erases"] for r in items),
        })
    return out
