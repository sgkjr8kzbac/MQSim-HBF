#!/usr/bin/env python3
"""Convert Mooncake request traces (JSONL or unified requests.csv) to ServeLoop request_trace JSON.

Preserves exact arrival timestamps, prompt/output lengths, and (when ``--with-prefix`` is set)
maps Mooncake's 512-token ``hash_ids`` deterministically onto ServeLoop's ``token_ids`` so that
every shared 512-token Mooncake prefix block produces 32 identical 16-token content-addressed
blocks in ``serveloop.prefix.block_keys`` with zero false hits on unshared blocks or partial tails.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hbf.frontend.trace.requests import (
    RequestRecord,
    import_mooncake_jsonl,
    per_stack,
    read_requests_csv,
    workload_from_requests,
)

REQUEST_TRACE_SCHEMA = {"name": "serveloop.request_trace", "version": 1}
MOONCAKE_BLOCK_TOKENS = 512
DEFAULT_VOCAB_SIZE = 128256


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _block_token_ids(block_hash: int, count: int, vocab_size: int) -> list[int]:
    """Deterministic token IDs in ``[0, vocab_size)`` uniquely derived from a Mooncake block hash."""
    # SplitMix64-style mixing seeded by block_hash; 16 tokens per ServeLoop block -> collision prob ~0.
    base = (int(block_hash) * 0x9E3779B97F4A7C15 + 0x517CC1B727220A95) & 0xFFFFFFFFFFFFFFFF
    out: list[int] = []
    for i in range(count):
        z = (base + i * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        z = z ^ (z >> 31)
        out.append(int(z % vocab_size))
    return out


def _unique_tail_tokens(req_index: int, block_idx: int, count: int, vocab_size: int) -> list[int]:
    """Request-unique token IDs so partial or unshared blocks never collide across requests."""
    seed = ((req_index + 1) * 0xD6E8FEB86659FD93 + (block_idx + 1) * 0xA5A3564E27F8862D) & 0xFFFFFFFFFFFFFFFF
    return _block_token_ids(seed, count, vocab_size)


def mooncake_hash_ids_to_token_ids(
    hash_ids: list[int],
    prompt_tokens: int,
    output_tokens: int,
    req_index: int,
    *,
    block_tokens: int = MOONCAKE_BLOCK_TOKENS,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
) -> list[int]:
    """Expand Mooncake's 512-token ``hash_ids`` into ``prompt_tokens + output_tokens - 1`` token IDs.

    Full 512-token prompt blocks map deterministically from their Mooncake hash ID so shared leading
    ``hash_ids`` match across requests at ServeLoop's 16-token block granularity. A trailing
    partial prompt block (< 512 tokens) and the ``output_tokens - 1`` decode input tokens receive
    request-unique token IDs so they never create false prefix hits.
    """
    tokens: list[int] = []
    full_blocks = prompt_tokens // block_tokens
    rem = prompt_tokens % block_tokens
    for b_idx in range(full_blocks):
        if b_idx < len(hash_ids):
            tokens.extend(_block_token_ids(hash_ids[b_idx], block_tokens, vocab_size))
        else:
            tokens.extend(_unique_tail_tokens(req_index, b_idx, block_tokens, vocab_size))
    if rem > 0:
        tokens.extend(_unique_tail_tokens(req_index, full_blocks, rem, vocab_size))
    if output_tokens > 1:
        tokens.extend(_unique_tail_tokens(req_index, full_blocks + 1, output_tokens - 1, vocab_size))
    assert len(tokens) == prompt_tokens + output_tokens - 1
    return tokens


def select_mooncake_slice(
    jsonl_path: Path,
    *,
    stacks: int = 3,
    max_requests: int = 1200,
    retain_s: float = 300.0,
) -> tuple[list[dict[str, Any]], list[RequestRecord], dict[str, Any]]:
    """Select an aligned Mooncake slice (optionally prefix-preserving sharded to stack 0).

    Returns ``(raw_selected_rows, r7_records, summary)`` where both lists describe the exact same
    requests in identical order with arrivals shifted to start at ``0.0 s``.
    """
    raw_rows: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_rows.append(json.loads(line))
    recs = import_mooncake_jsonl(jsonl_path, prefix_mode="conversation")
    assert len(raw_rows) == len(recs)

    if stacks > 1:
        # Mirror hbf.frontend.trace.requests.per_stack sharding: conversation/branch groups route
        # by prefix_id % stacks (preserving multi-turn cache locality), while unprefixed requests
        # and cluster-wide universal prefixes (> N / (2 * stacks)) balance round-robin.
        from collections import Counter
        n_total = len(recs)
        pcounts = Counter(r.prefix_id for r in recs if r.prefix_id is not None and r.prefix_id >= 0)
        global_thresh = max(16.0, n_total / (2.0 * stacks))
        selected_indices = []
        rr = 0
        for idx, rec in enumerate(recs):
            pid = rec.prefix_id if rec.prefix_id is not None else -1
            if pid < 0 or pcounts[pid] > global_thresh:
                if rr % stacks == 0:
                    selected_indices.append(idx)
                rr += 1
            elif pid % stacks == 0:
                selected_indices.append(idx)
    else:
        selected_indices = list(range(len(recs)))

    selected_indices = selected_indices[:max_requests]
    if not selected_indices:
        raise ValueError("Empty Mooncake slice")

    t0_s = recs[selected_indices[0]].arrival_s
    sel_raw: list[dict[str, Any]] = []
    sel_recs: list[RequestRecord] = []
    for idx in selected_indices:
        rec = recs[idx]
        arr_s = round(rec.arrival_s - t0_s, 6)
        sel_raw.append({
            **raw_rows[idx],
            "arrival_s": arr_s,
            "orig_index": idx,
        })
        sel_recs.append(RequestRecord(
            arrival_s=arr_s,
            prompt_tokens=rec.prompt_tokens,
            output_tokens=rec.output_tokens,
            prefix_id=rec.prefix_id,
            prefix_tokens=rec.prefix_tokens,
        ))

    dur_s = sel_recs[-1].arrival_s if len(sel_recs) > 1 else 0.0
    summary = {
        "source": str(jsonl_path),
        "sha256": file_sha256(jsonl_path),
        "stacks_sharding": stacks,
        "count": len(sel_recs),
        "duration_s": dur_s,
        "rate_req_s": len(sel_recs) / dur_s if dur_s > 0 else 0.0,
        "mean_prompt": sum(r.prompt_tokens for r in sel_recs) / len(sel_recs),
        "max_prompt": max(r.prompt_tokens for r in sel_recs),
        "mean_output": sum(r.output_tokens for r in sel_recs) / len(sel_recs),
        "max_output": max(r.output_tokens for r in sel_recs),
        "retain_s": retain_s,
    }
    return sel_raw, sel_recs, summary


def build_serveloop_trace_dict(
    sel_raw: list[dict[str, Any]],
    *,
    source_sha256: str,
    source_label: str,
    model_id: str = "llama31_8b",
    with_prefix: bool = False,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
) -> dict[str, Any]:
    """Build a canonical ``serveloop.request_trace`` JSON document from selected Mooncake rows."""
    requests_out: list[dict[str, Any]] = []
    for i, row in enumerate(sel_raw):
        prompt = max(1, int(row["input_length"]))
        output = max(1, int(row["output_length"]))
        arrival_ns = float(row["arrival_s"]) * 1e9
        token_ids = (
            mooncake_hash_ids_to_token_ids(
                list(row.get("hash_ids", ())), prompt, output, i, vocab_size=vocab_size
            )
            if with_prefix
            else None
        )
        requests_out.append({
            "request_id": f"mc_{i:05d}",
            "arrival_ns": arrival_ns,
            "model_id": model_id,
            "prompt_tokens": prompt,
            "output_tokens": output,
            "token_ids": token_ids,
            "cache_salt": "",
        })
    return {
        "schema": dict(REQUEST_TRACE_SCHEMA),
        "provenance": {
            "kind": "production_trace",
            "source": source_label,
            "sha256": source_sha256,
        },
        "requests": requests_out,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert Mooncake JSONL trace to ServeLoop request_trace JSON")
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / ".build/traces/mooncake/conversation_trace.jsonl",
        help="Path to Mooncake conversation_trace.jsonl",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output serveloop.request_trace JSON path")
    parser.add_argument("--stacks", type=int, default=3, help="Prefix-preserving shard divisor (1 = unsharded)")
    parser.add_argument("--max-requests", type=int, default=1200, help="Number of requests to take")
    parser.add_argument("--with-prefix", action="store_true", help="Map Mooncake hash_ids to ServeLoop token_ids")
    parser.add_argument("--model-id", default="llama31_8b", help="ServeLoop model_id")
    args = parser.parse_args(argv)

    sel_raw, _, summary = select_mooncake_slice(
        args.input, stacks=args.stacks, max_requests=args.max_requests
    )
    doc = build_serveloop_trace_dict(
        sel_raw,
        source_sha256=summary["sha256"],
        source_label=f"mooncake_conversation_s{args.stacks}_n{summary['count']}_{'prefix' if args.with_prefix else 'noprefix'}",
        model_id=args.model_id,
        with_prefix=args.with_prefix,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(doc), encoding="utf-8")
    print(
        f"Wrote {args.output}: {summary['count']} reqs, dur={summary['duration_s']:.1f}s "
        f"({summary['rate_req_s']:.2f} req/s), prefix={'on' if args.with_prefix else 'off'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
