"""Run HBFSim persistent sessions (`hbfsim`) and reference replay (`hbfsim-reference`)."""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

from ...device.config import CONFIG_DIR
from ...util import dump, write_table
from .build import build, sha256, version
from .convert import (
    ConvertedBatch,
    ConvertedTrace,
    HbfSimGeometry,
    decode_physical_address,
    hbfsim_geometry,
    write_replay_trace,
)
from .parse import (
    aggregate_layer_rows,
    aggregate_step_rows,
    extract_hbf_delta,
    parse_wear_snapshot,
)

HBFSIM_OCP_CFG = CONFIG_DIR / "hbfsim-ocp.cfg"
DEFAULT_OOB_BYTES = 224


def _channel_raw_bw_gb_s(cfg, geom: HbfSimGeometry, oob_bytes: int = DEFAULT_OOB_BYTES) -> float:
    """Internal per-channel raw codeword bandwidth (payload + OOB) matching ``cfg.aggregate_gb_s``."""
    payload_per_ch = cfg.aggregate_gb_s / geom.total_channels
    return payload_per_ch * (cfg.page_bytes + oob_bytes) / cfg.page_bytes


def write_config(
    cfg,
    path: Path,
    *,
    geom: HbfSimGeometry | None = None,
    speed_grade: int | None = None,
    die_blocking: bool = False,
    match_core_bw: bool = True,
    initial_free_blocks_erased: bool = False,
    overrides: dict[str, object] | None = None,
) -> HbfSimGeometry:
    """Generate a raw-physical HBFSim ``.cfg`` file corresponding to ``cfg``."""
    geom = hbfsim_geometry(cfg, speed_grade=speed_grade) if geom is None else geom
    oob = int((overrides or {}).get("hbf-oob-bytes", DEFAULT_OOB_BYTES))
    # Default HBFSim ECC raw bandwidth is 101.25 GB/s; ensure ecc_decode_latency_ns covers II.
    chan_bw = _channel_raw_bw_gb_s(cfg, geom, oob) if match_core_bw else 101.25
    ecc_bw = max(101.25, chan_bw)

    kv: dict[str, object] = {
        "hbf-standard": "OCP-HBF-0.7.0-2026-08-03",
        "hbm-standard": "JEDEC-JESD270-4-2025-04",
        "hbf-processor-interconnect": "ucie",
        "hbf-speed-grade": geom.speed_grade,
        "hbf-stacks": geom.stacks,
        "hbf-channels": geom.channels_per_stack,
        "hbf-dies-per-channel": geom.dies_per_channel,
        "hbf-planes-per-die": geom.planes_per_die,
        "hbf-blocks-per-plane": geom.blocks_per_plane,
        "hbf-pages-per-block": geom.pages_per_block,
        "hbf-page-size": geom.page_bytes,
        "hbf-oob-bytes": oob,
        "hbf-media-lanes-per-plane": 1,
        "hbf-page-buffer-banks-per-plane": 2,
        "hbf-read-ns": cfg.read_ns,
        "hbf-program-ns": cfg.program_ns,
        "hbf-erase-ns": cfg.erase_ns,
        "hbf-ecc-decode-latency-ns": 500,
        "hbf-ecc-encode-latency-ns": 500,
        "hbf-ecc-decode-raw-bw": f"{ecc_bw:.6g}",
        "hbf-ecc-encode-raw-bw": f"{ecc_bw:.6g}",
        "hbf-channel-bw": f"{chan_bw:.6g}",
        "hbf-tsv-bw": 1644,
        "hbf-media-lane-bw": 2048,
        "hbf-logic-sram-bw": 2048,
        "hbf-page-buffer-bw": 2048,
        "hbf-mapping-mode": "raw-physical",
        "hbf-write-coalescing": "false",
        "hbf-write-buffer-pages": 0,
        "hbf-write-buffer-flush-threshold-pages": 0,
        "hbf-auto-gc-enabled": "false",
        "hbf-gc-reserved-free-blocks-per-plane": 0,
        "hbf-zone-size-blocks": 1,
        "hbf-thermal-enable": "false",
        "hbf-initial-free-blocks-erased": "true" if initial_free_blocks_erased else "false",
        "hbf-die-blocking": "true" if die_blocking else "false",
    }
    if overrides:
        for k, v in overrides.items():
            if isinstance(v, bool):
                kv[k] = "true" if v else "false"
            else:
                kv[k] = v
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Auto-generated HBFSim raw-physical configuration\n"]
    lines.extend(f"{k}={v}\n" for k, v in kv.items())
    path.write_text("".join(lines), encoding="utf-8")
    return geom


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class HbfSimError(RuntimeError):
    """Raised when the HBFSim engine returns an error receipt or terminates abnormally."""


class HbfSimSession:
    """Interactive wrapper around the compiled ``hbfsim`` transaction engine."""

    def __init__(
        self,
        config_path: Path,
        work_dir: Path,
        *,
        binary: Path | None = None,
        enable_hbm: bool = False,
        enable_hbf: bool = True,
        static_blocks_per_plane: int = 0,
        published_blocks_per_plane: int = 0,
        heatmap_path: Path | None = None,
        heatmap_bins: int = 0,
        cli_overrides: dict[str, str] | None = None,
    ):
        self.binary = Path(binary or build(target="hbfsim")).resolve()
        self.config_path = Path(config_path).resolve()
        self.work_dir = Path(work_dir).resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.wear_prefix = self.work_dir / "wear"

        cmd = [
            str(self.binary),
            "--system-config", str(self.config_path),
            "--enable-hbm", "true" if enable_hbm else "false",
            "--enable-hbf", "true" if enable_hbf else "false",
            "--hbf-wear-output-prefix", str(self.wear_prefix),
        ]
        if static_blocks_per_plane:
            cmd.extend(["--static-hbf-blocks-per-plane", str(static_blocks_per_plane)])
        if published_blocks_per_plane:
            cmd.extend(["--published-hbf-blocks-per-plane", str(published_blocks_per_plane)])
        if heatmap_path is not None and heatmap_bins > 0:
            cmd.extend([
                "--hbf-physical-heatmap", str(Path(heatmap_path).resolve()),
                "--hbf-physical-heatmap-bins", str(heatmap_bins),
            ])
        if cli_overrides:
            for k, v in cli_overrides.items():
                cmd.extend([f"--{k}", str(v)])

        self.command = cmd
        self._proc = subprocess.Popen(
            cmd,
            cwd=str(self.work_dir),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.ready_receipt = self._read_json()
        if self.ready_receipt.get("result") != "ready":
            raise HbfSimError(f"HBFSim failed to start: {self.ready_receipt}")

    def _read_json(self) -> dict:
        line = self._proc.stdout.readline()
        if not line:
            stderr = self._proc.stderr.read() if self._proc.stderr else ""
            rc = self._proc.wait()
            raise HbfSimError(f"HBFSim closed stdout (rc={rc}): {stderr.strip()}")
        return json.loads(line)

    def _send(self, text: str) -> dict:
        assert self._proc.stdin is not None
        self._proc.stdin.write(text)
        self._proc.stdin.flush()
        return self._read_json()

    def run_batch(
        self,
        batch_id: str,
        transactions: list[tuple[str, str, str, int, int, float]],
        *,
        completions: bool = False,
    ) -> dict:
        """Submit a ``BEGIN ... TX ... END`` batch.

        Each tuple is ``(tx_id, target, op, addr, bytes, issue_ns)``.
        """
        lines = [
            f"TX id={tx_id} target={target} op={op} addr={addr} bytes={nbytes} "
            f"issue_ns={issue_ns:.6f} duration_ns=0 deps=- stack=-"
            for tx_id, target, op, addr, nbytes, issue_ns in transactions
        ]
        body = "".join(f"{line}\n" for line in lines)
        digest = _digest(body)
        comp_flag = 1 if completions else 0
        header = f"BEGIN {batch_id} {digest} {digest} completions={comp_flag}\n"
        msg = header + body + f"END {batch_id}\n"
        receipt = self._send(msg)
        if receipt.get("result") == "error":
            raise HbfSimError(receipt.get("message", str(receipt)))
        return receipt

    def zone_command(
        self,
        command: str,
        op_id: str,
        *,
        stack: int = 0,
        channel: int = 0,
        zone: int = 0,
        argument: int = 0,
    ) -> dict:
        """Execute a ``ZONE_*`` command (e.g. ``ZONE_POPULATE``, ``ZONE_INVALIDATE``, ``ZONE_RESET``)."""
        receipt = self._send(f"{command} {op_id} {stack} {channel} {zone} {argument}\n")
        if receipt.get("result") == "error":
            raise HbfSimError(receipt.get("message", str(receipt)))
        return receipt

    def zone_write_batch(
        self,
        batch_id: str,
        fragments: list[tuple[str, float, int, int]],
    ) -> dict:
        """Execute ``ZONE_WRITE_BATCH`` with ``(frag_id, arrival_offset_ns, byte_addr, bytes)``."""
        lines = [f"ZONE_WRITE_BATCH {batch_id} {len(fragments)}"]
        for frag_id, arr_ns, addr, nbytes in fragments:
            lines.append(f"{frag_id} {arr_ns:.6f} {addr} {nbytes}")
        receipt = self._send("\n".join(lines) + "\n")
        if receipt.get("result") == "error":
            raise HbfSimError(receipt.get("message", str(receipt)))
        return receipt

    def wear_snapshot(self, snap_id: str = "final") -> dict:
        receipt = self._send(f"WEAR_SNAPSHOT {snap_id}\n")
        if receipt.get("result") == "error":
            raise HbfSimError(receipt.get("message", str(receipt)))
        return parse_wear_snapshot(receipt)

    def close(self) -> dict | None:
        if self._proc.poll() is not None:
            return None
        try:
            receipt = self._send("QUIT\n")
        except Exception:
            receipt = None
        finally:
            if self._proc.stdin:
                self._proc.stdin.close()
            self._proc.wait(timeout=30)
            if self._proc.stdout:
                self._proc.stdout.close()
            if self._proc.stderr:
                self._proc.stderr.close()
        return receipt

    def __enter__(self) -> HbfSimSession:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def simulate_batches(
    cfg,
    converted: ConvertedTrace,
    directory: Path,
    *,
    binary: Path | None = None,
    die_blocking: bool = False,
    match_core_bw: bool = True,
    initial_free_blocks_erased: bool = False,
    static_blocks: int | None = None,
    published_blocks: int | None = None,
    heatmap_bins: int = 0,
    collect_completions: bool = False,
    write_replay: bool = False,
    overrides: dict[str, object] | None = None,
) -> tuple[list[dict], list[dict], list[dict], dict]:
    """Run a ``ConvertedTrace`` through ``hbfsim`` in per-subbatch mode (matching MQSim sub-batches).

    Returns ``(batch_rows, layer_rows, step_rows, stats)``.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    binary = Path(binary or build(target="hbfsim")).resolve()
    geom = converted.geometry

    cfg_path = directory / "hbfsim.cfg"
    write_config(
        cfg,
        cfg_path,
        geom=geom,
        die_blocking=die_blocking,
        match_core_bw=match_core_bw,
        initial_free_blocks_erased=initial_free_blocks_erased,
        overrides=overrides,
    )
    if write_replay:
        write_replay_trace(directory / "replay.trace", converted)

    has_writes = any(req.op == "W" for b in converted.batches for req in b.requests)
    if static_blocks is None:
        static_blocks = 0 if has_writes else (
            converted.resident_blocks_per_plane or geom.blocks_per_plane
        )
    if published_blocks is None:
        published_blocks = (geom.blocks_per_plane - static_blocks) if has_writes else 0

    heatmap_path = (directory / "heatmap.jsonl") if heatmap_bins > 0 else None

    t0 = time.perf_counter()
    batch_rows: list[dict] = []
    all_completions: list[dict] = []

    with HbfSimSession(
        cfg_path,
        directory,
        binary=binary,
        enable_hbm=False,
        enable_hbf=True,
        static_blocks_per_plane=static_blocks,
        published_blocks_per_plane=published_blocks,
        heatmap_path=heatmap_path,
        heatmap_bins=heatmap_bins,
    ) as session:
        written_zones: set[tuple[int, int, int]] = set()
        reset_seq = 0
        for b_idx, batch in enumerate(converted.batches):
            # HBFSim raw-physical requires host ZONE_INVALIDATE + ZONE_RESET before rewriting
            # page 0 of a previously-written block (triggering page-0 auto-erase on program).
            to_reset: list[tuple[int, int, int]] = []
            seen_reset: set[tuple[int, int, int]] = set()
            touched_zones: set[tuple[int, int, int]] = set()
            for req in batch.requests:
                if req.op == "W":
                    s, ch, d, p, b, pg = decode_physical_address(geom, req.addr)
                    zkey = (s, ch, ((d * geom.planes_per_die + p) * geom.blocks_per_plane) + b)
                    if pg == 0 and zkey in written_zones and zkey not in seen_reset:
                        seen_reset.add(zkey)
                        to_reset.append(zkey)
                    touched_zones.add(zkey)
            for s, ch, z in to_reset:
                session.zone_command("ZONE_INVALIDATE", f"zi_{reset_seq}", stack=s, channel=ch, zone=z)
                session.zone_command("ZONE_RESET", f"zr_{reset_seq}", stack=s, channel=ch, zone=z)
                reset_seq += 1
            txs = [
                (f"b{b_idx}/{i}", "HBF_PHYSICAL", req.op, req.addr, req.bytes, 0.0)
                for i, req in enumerate(batch.requests)
            ]
            receipt = session.run_batch(f"batch_{b_idx}", txs, completions=collect_completions)
            written_zones.update(touched_zones)
            delta = extract_hbf_delta(receipt)
            read_pages = sum(1 for r in batch.requests if r.op == "R")
            write_pages = sum(1 for r in batch.requests if r.op == "W")
            batch_rows.append({
                "step": batch.step,
                "layer": batch.layer,
                "subbatch": batch.subbatch,
                "arrival_ns": batch.arrival_ns,
                "hbfsim_ns": delta["elapsed_ns"],
                "read_pages": read_pages,
                "write_pages": write_pages,
                "media_page_reads": delta["page_reads"],
                "read_buffer_hits": delta["read_buffer_hits"],
                "read_buffer_misses": delta["read_buffer_misses"],
                "page_programs": delta["page_programs"],
                "block_erases": delta["block_erases"],
            })
            if collect_completions:
                all_completions.extend(receipt.get("transaction_completions", []))
        wear = session.wear_snapshot("final")
    wall_s = time.perf_counter() - t0

    layer_rows = aggregate_layer_rows(batch_rows)
    step_rows = aggregate_step_rows(batch_rows)
    stats = {
        "wall_seconds": wall_s,
        "geometry": asdict(geom),
        "die_blocking": die_blocking,
        "match_core_bw": match_core_bw,
        "static_blocks_per_plane": static_blocks,
        "published_blocks_per_plane": published_blocks,
        "dropped_releases": converted.dropped_releases,
        "cls_counts": dict(converted.cls_counts or {}),
        "total_read_pages": sum(r["read_pages"] for r in batch_rows),
        "total_write_pages": sum(r["write_pages"] for r in batch_rows),
        "total_media_page_reads": sum(r["media_page_reads"] for r in batch_rows),
        "total_read_buffer_hits": sum(r["read_buffer_hits"] for r in batch_rows),
        "total_page_programs": sum(r["page_programs"] for r in batch_rows),
        "total_block_erases": sum(r["block_erases"] for r in batch_rows),
        "wear_erases": wear["workload_erases"],
    }
    if collect_completions:
        stats["completions"] = all_completions

    ver = version(patches=["0001-die-blocking.patch"] if die_blocking else [])
    dump(directory / "meta.json", {
        "backend": "hbfsim",
        "hbfsim_commit": ver["hbfsim_commit"],
        "version": ver,
        "binary_sha256": sha256(binary),
        "geometry": asdict(geom),
        "die_blocking": die_blocking,
        "cls_counts": stats["cls_counts"],
    })
    dump(directory / "statistics.json", {k: v for k, v in stats.items() if k != "completions"})
    write_table(directory / "batches.csv", batch_rows)
    write_table(directory / "layers.csv", layer_rows)
    write_table(directory / "steps.csv", step_rows)
    return batch_rows, layer_rows, step_rows, stats


def simulate_trace(cfg, trace, directory: Path, **kwargs):
    """Convenience wrapper: convert an ``AddressTrace`` and run ``simulate_batches``."""
    from .convert import convert_trace

    speed_grade = kwargs.pop("speed_grade", None)
    geom = hbfsim_geometry(cfg, speed_grade=speed_grade)
    converted = convert_trace(cfg, trace, geom=geom)
    return simulate_batches(cfg, converted, directory, **kwargs)
