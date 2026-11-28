"""Closed-form steady-state endurance model (M4a)."""
import math

import numpy as np

from .geometry import (CAUSES, DAY, GIB, POLICIES, Device, erase_amp,
                       region_pages, sci, threshold_steps)
from .interference import MODES as M5_MODES, hbf_gaps, interference

def _pages(geometry, tokens):
    return float(geometry.stored_pages(np.array([int(round(tokens))]))[0])


def _written(geometry, c0, c1):
    c0_i, c1_i = int(round(c0)), int(round(c1))
    if hasattr(geometry, "written_pages"):
        return float(geometry.written_pages(np.array([c0_i]), np.array([c1_i]))[0])
    return max(0.0, _pages(geometry, c1_i) - _pages(geometry, c0_i))


def _dead(geometry, c0, c1):
    if not hasattr(geometry, "dead_pages"):
        return 0.0
    return float(geometry.dead_pages(np.array([int(round(c0))]), np.array([int(round(c1))]))[0])


def analytic(geometry, clock, workload, device=None, *, policy="packed", x_values=(1e6,), pe_limits=(1e4, 1e5, 1e6),
             share=1.0, hot=1.0, retention_s=86400.0, prefix_reads="per_request", t_prog=100000.0,
             t_bers=1000000.0, m5="gaps", suspend_ns=None, block_unit="die", stacks=1, align="row", region=None):
    """Steady-state rows per stack, one per X, for a closed-loop or Poisson workload (design-B section 3).

    * Rate: closed loop keeps ``concurrency / stacks`` requests (sessions for an agentic workload)
      in flight per stack; lambda = B / (out x T), T = Clock TPOT at the mean decode context.
    * Pages per request P_r = L H sum_t floor(w_t / tpp_t) (full pages; partial pages die in HBM).
    * Read disturb: rho = ppb x share x hot block reads per decode step; a page read for s steps is
      refreshed floor(s rho / X) times. For a request all its pages get its own s = out
      (conservative: generated pages are counted as if present from the start). Session data
      written in call k is read in every later call (out steps each, plus one prefill read).
    * Retention: data alive longer than T_ret is rewritten floor(life / T_ret) times.
    * P/E per block per day = programs x erase amplification x 86,400 / C (ideal wear levelling,
      the spec's AVGPEC, section 5.9.2.18).
    ``region`` (layer_major) is the wave's written pages; default phase A's layout via the clock's model.
    """
    device = device or Device()
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {', '.join(POLICIES)}")
    if m5 not in M5_MODES + ("off",):
        raise ValueError(f"m5 must be one of {', '.join(M5_MODES + ('off',))}")
    page, ppb = device.page_bytes, device.pages_per_block
    seg_banks, seg_pages, capacity = device.segment_banks, device.segment_pages, device.capacity_pages
    read_mult = float(getattr(geometry, "read_multiplier", 1.0))
    rho = ppb * share * hot * read_mult
    w = workload
    n_in, n_out = w.input.mean(), w.output.mean()
    sessions = w.sessions
    calls = w.calls if sessions else 1
    h0 = w.history.mean() if sessions and w.history is not None else 0.0
    prefix = w.prefix_tokens if w.prefix_share else 0
    hbm_out = policy in ("hbm_decode", "layer_major")
    if sessions:
        ctx = h0 + n_in + (calls - 1) / 2 * (n_in + n_out) + n_out / 2
    else:
        ctx = w.prefix_share * prefix + n_in + n_out / 2
    active = w.concurrency / stacks if w.arrival == "closed" else None
    b = max(1, math.ceil(active - 1e-9)) if active is not None else 1

    def step_of(batch):
        parts = (np.arange(batch) % device.parts) if device.parts > 1 else None
        return clock.step(np.full(batch, int(round(ctx))), parts, seg_banks if parts is not None else None,
                          np.full(batch, int(n_out / 2)) if hbm_out else None)

    out_wave = w.output.mean_max(b) if policy == "layer_major" else n_out
    for _ in range(40):
        step = step_of(b)
        tpot = step.tpot_ns / 1e9
        call_s = n_out * tpot
        if active is not None:
            new_b = max(1, round(active * call_s / (call_s + w.think_s))) if sessions else b
        else:
            new_b = max(1, round(w.rate / stacks * calls * call_s))
        if new_b == b:
            break
        b = new_b
        out_wave = w.output.mean_max(b) if policy == "layer_major" else n_out
    if policy == "layer_major":
        calls_per_s = b / (out_wave * tpot)
    elif active is not None:
        calls_per_s = active / (call_s + w.think_s) if sessions else b / call_s
    else:
        calls_per_s = w.rate / stacks * calls
    entry_rate = calls_per_s / calls
    tokens_per_s = calls_per_s * n_out

    # Chunks of one request (or session): (pages, cause, block reads over the chunk's life, life in s).
    chunks = []
    dead_total = 0.0
    if sessions:
        history = 0.0
        for k in range(calls):
            fresh = h0 + n_in if k == 0 else n_in
            pre = (min(_written(geometry, history, history + fresh), _pages(geometry, history + fresh))
                   if history > 0 else _written(geometry, 0, fresh))
            dec = _written(geometry, history + fresh, history + fresh + n_out)
            if history > 0:
                dead_total += _dead(geometry, history, history + fresh)
            dead_total += _dead(geometry, history + fresh, history + fresh + n_out)
            later = calls - 1 - k
            reads = (later + 1) * n_out * rho + later * ppb * read_mult
            life = (later + 1) * (call_s + w.think_s) + w.retain_s
            chunks.append((pre, "prefill", reads, life))
            if not hbm_out:
                chunks.append((dec, "decode", reads - n_out / 2 * rho, life))
            history += fresh + n_out
    else:
        hit = w.prefix_share
        pre_hit = (min(_written(geometry, prefix, prefix + n_in), _pages(geometry, prefix + n_in))
                   if hit else 0.0)
        own = hit * pre_hit + (1 - hit) * _written(geometry, 0, n_in)
        dec_hit = _written(geometry, prefix + n_in, prefix + n_in + n_out) if hit else 0.0
        dec = hit * dec_hit + (1 - hit) * _written(geometry, n_in, n_in + n_out)
        dead_hit = _dead(geometry, prefix + n_in, prefix + n_in + n_out) if hit else 0.0
        dead_total = hit * dead_hit + (1 - hit) * _dead(geometry, n_in, n_in + n_out)
        life = call_s + w.retain_s
        if policy == "layer_major":
            life = out_wave * tpot
        chunks.append((own, "prefill", n_out * rho, life))
        if not hbm_out and dec > 0:
            chunks.append((dec, "decode", n_out * rho, life))
        if hbm_out:
            dead_total = min(dead_total, own)
    host_pages = sum(chunk[0] for chunk in chunks)
    if policy == "layer_major":
        if region is None:
            if hasattr(geometry, "region_span"):
                region = geometry.region_span(b, max(1, round(w.input.mean_max(b))), ppb)
            else:
                model = getattr(clock, "model", None)
                if model is None:
                    raise ValueError("layer_major needs region= or a Clock with a model")
                region = region_pages(model, geometry.tensors, b, max(1, round(w.input.mean_max(b))), ppb)
        per_request = region / b  # each request's share of the wave's region

    rows = []
    for x in x_values:
        rd = ret = numerator = 0.0
        for pages, _, reads, life in chunks:
            rd += math.floor(reads / x + 1e-9) * pages
            numerator += reads * pages
            ret += math.floor(life / retention_s + 1e-9) * pages if retention_s else 0.0
        prefix_rd = prefix_ret = 0.0
        if prefix and w.prefix_groups:
            p_pages = _pages(geometry, prefix)
            weights = 1.0 / np.arange(1, w.prefix_groups + 1) ** w.zipf
            for p in weights / weights.sum():
                readers = b * w.prefix_share * p
                k = readers if prefix_reads == "per_request" else min(1.0, readers)
                per_s = k * rho / tpot + calls_per_s * w.prefix_share * p * ppb * read_mult  # decode + prefill reads
                prefix_rd += p_pages * per_s / x
                numerator += p_pages * per_s / entry_rate
            prefix_ret = w.prefix_groups * p_pages / retention_s if retention_s else 0.0
        rates = dict.fromkeys(CAUSES, 0.0)
        rates["prefill"] = entry_rate * sum(c[0] for c in chunks if c[1] == "prefill")
        rates["decode"] = entry_rate * sum(c[0] for c in chunks if c[1] == "decode")
        if policy == "layer_major":
            # The whole region is replayed in place (no skipping) once its blocks reach X reads.
            rates["padding"] = entry_rate * (per_request - host_pages)
            rates["replay"] = entry_rate * per_request * (math.floor(n_out * rho / x + 1e-9)
                                                          + (math.floor(out_wave * tpot / retention_s + 1e-9)
                                                             if retention_s else 0))
            amp = erase_amp(region, seg_pages)
            refreshes = rates["replay"] / (entry_rate * per_request)
        else:
            rates["refresh_rd"] = entry_rate * rd + prefix_rd
            rates["refresh_ret"] = entry_rate * ret + prefix_ret
            if align == "row" and policy in ("packed", "hbm_decode", "vllm_pool"):
                rates["padding"] = entry_rate * (seg_banks - 1) / 2  # prompts start on a row
            amp = erase_amp(host_pages, seg_pages) if policy == "dedicated" else 1.0
            refreshes = rd / host_pages if host_pages else 0.0
        programs = sum(rates.values())
        pe_day = programs * amp * DAY / capacity
        kv = rates["prefill"] + rates["decode"]
        m5_row = {"extra_step_ns": 0.0, "max_step_extra_ns": 0.0}
        if m5 != "off":
            m5_row = interference({"programs": programs * tpot, "erases": programs * amp / ppb * tpot,
                                   "wave_banks": seg_banks, "gaps_ns": step.gaps_ns, "step_ns": step.tpot_ns},
                                  m5, t_prog, t_bers, suspend_ns, block_unit=block_unit)
        row = {"workload": w.name, "policy": policy, "partition": device.partition, "pages_per_block": ppb,
               "x": x, "stacks": stacks, "mean_batch": b, "mean_context": ctx, "mean_tpot_ms": tpot * 1e3,
               "tpot_m5_ms": (step.tpot_ns + m5_row["extra_step_ns"]) / 1e6,
               "max_step_extra_ms": m5_row["max_step_extra_ns"] / 1e6, "req_per_s": calls_per_s,
               "tokens_per_s": tokens_per_s, "pages_per_request": host_pages / calls,
               "dead_pages_per_request": dead_total / calls,
               "kv_bytes_per_token": geometry.hbf_token_bytes,
               "hbm_bytes_per_request": float(geometry.hbm_bytes(np.array([int(round(ctx))]))[0]),
               "threshold_steps": threshold_steps(x, ppb, share, hot * read_mult),
               "refreshes_per_request": refreshes, "kv_write_gb_s": kv * page / 1e9}
        row |= {f"{cause}_gb_s": rates[cause] * page / 1e9 for cause in CAUSES}
        refresh = rates["refresh_rd"] + rates["refresh_ret"] + rates["replay"]
        row |= {"program_gb_s": programs * page / 1e9,
                "program_bw_share": programs * page / (device.banks * page / (t_prog / 1e9)),
                "wa": programs / kv if kv else None, "erase_amp": amp,
                "erase_wa": programs * amp / kv if kv else None,
                "refresh_to_kv": refresh / kv if kv else None, "erases_per_s": programs * amp / ppb,
                "avg_pec_per_day": pe_day, "max_pec_per_day": pe_day}
        for limit in pe_limits:
            row[f"days_to_limit_avg@{sci(limit)}"] = limit / pe_day if pe_day else math.inf
        for limit in pe_limits:
            row[f"days_to_limit_max@{sci(limit)}"] = limit / pe_day if pe_day else math.inf
        row["pe_per_mtok"] = pe_day / (tokens_per_s * DAY / 1e6) if tokens_per_s else None
        row["break_even_x"] = numerator * entry_rate / kv if kv and policy != "layer_major" else (
            n_out * rho if policy == "layer_major" else None)
        partial = geometry.partial_pages() * page
        row["staging_peak_mib"] = (b * partial + seg_banks * page
                                   + (b * n_out * geometry.token_bytes if hbm_out else 0)) / 2 ** 20
        row["pending_units_per_channel"] = b * geometry.partial_pages() / 16
        row["unit_open_s"] = max(geometry.tpp) * tpot
        row["replay_buffer_mib"] = seg_pages * page / 2 ** 20 if policy == "layer_major" else 0.0
        row["legal"] = policy != "vllm_pool"
        rows.append(row)
    return rows
