"""Address trace interface (hbf/frontend/addrtrace.py): serialization, validation, and backend consumption."""
import numpy as np
import pytest

from hbf.backends.mqsim.run import simulate
from hbf.backends.mqsim.schedule import batches_from_trace
from hbf.backends.rounds.analysis import Protocol, analyze_trace
from hbf.device.config import load
from hbf.frontend.addrtrace import (AddressTrace, LayerChunk, iter_trace, make_meta, read_trace,
                                    validate, write_trace)
from hbf.frontend.kv.layouts import LAYOUTS
from hbf.frontend.kv.tensors import Shape
from hbf.frontend.trace.synthetic import generate


def _chunk(step, layer, ops, lpas, *, sb=None, cls="kv"):
    n = len(lpas)
    return LayerChunk(
        step=step, layer=layer,
        subbatch=np.zeros(n, dtype=np.int64) if sb is None else np.asarray(sb, dtype=np.int64),
        request=np.zeros(n, dtype=np.int64),
        op=np.array(ops, dtype="U1"),
        lpa=np.asarray(lpas, dtype=np.int64),
        useful_bytes=np.full(n, 4096, dtype=np.int64),
        burst_bytes=np.full(n, 4096, dtype=np.int64),
        burst_runs=np.ones(n, dtype=np.int64),
        cls=np.full(n, cls, dtype="U8"),
    )


def test_addrtrace_roundtrip_and_both_backends(binary, tmp_path):
    """Write addr.npz/layers.npz/addr.json, read them back, and drive both MQSim and rounds backends."""
    cfg = load("ocp", channels=4, chips=2, planes=4, aggregate_gb_s=128,
               pages_per_block=8, blocks_per_plane=16)
    shape = Shape(2, 2, 128, 128)
    layout = LAYOUTS["striped_contiguous"](cfg, shape)
    accesses = list(generate(cfg, layout, "uniform_tokens", fraction=0.25, seed=9))
    proto = Protocol(256, host_order="trigger", ucie_grade=2)
    meta, chunks = iter_trace(cfg, layout, accesses, prefill=True, expected_steps=(0,),
                              protocol=proto, layout_name="striped_contiguous")
    out = tmp_path / "trace"
    out.mkdir()
    write_trace(out, meta, chunks)
    assert (out / "addr.npz").is_file() and (out / "layers.npz").is_file() and (out / "addr.json").is_file()

    loaded = read_trace(out)
    validate(cfg, loaded)
    assert loaded.meta["resident"] == [] and loaded.meta["protocol"]["mocs"] == 256

    layers_r, steps_r = analyze_trace(cfg, loaded.chunks, (1, 4), protocol=proto)
    assert len(layers_r) == 4 and len(steps_r) == 2

    batches = batches_from_trace(cfg, loaded.chunks)
    _, layers_m, _, stats = simulate(cfg, batches, tmp_path / "mqsim", binary)
    assert stats["unwritten_reads"] == 0 and any(r["step"] == 0 for r in layers_m)


def test_addrtrace_validate_rejects_violations_and_allows_release_then_rewrite():
    cfg = load("ocp", channels=2, chips=2, planes=2, aggregate_gb_s=64,
               pages_per_block=4, blocks_per_plane=8)
    cap = cfg.capacity_sectors // cfg.spp
    b0 = [cfg.unit_lpa(0, p) for p in range(4)]

    # 1. LPA out of range
    t_oob = AddressTrace(make_meta(cfg, resident=[(0, cap)]), [_chunk(0, 0, ["R"], [cap])])
    with pytest.raises(ValueError, match="out of range"):
        validate(cfg, t_oob)

    # 2. Read of unwritten page (section 5.4.3 / error 0x7)
    t_unwritten = AddressTrace(make_meta(cfg, resident=[]), [_chunk(0, 0, ["R"], [b0[0]])])
    with pytest.raises(ValueError, match="unwritten"):
        validate(cfg, t_unwritten)

    # 3. Skip page in block (section 5.4.1)
    t_skip = AddressTrace(make_meta(cfg, resident=[]), [_chunk(-1, 0, ["W", "W"], [b0[0], b0[2]], cls="prefill")])
    with pytest.raises(ValueError, match="Out-of-order"):
        validate(cfg, t_skip)

    # 4. Page-0 rewrite while block still has live pages (section 5.4.1)
    t_live_erase = AddressTrace(make_meta(cfg, resident=[]), [
        _chunk(-1, 0, ["W", "W"], [b0[0], b0[1]], cls="prefill"),
        _chunk(0, 0, ["D"], [b0[0]], cls="refresh"),
        _chunk(1, 0, ["W"], [b0[0]], cls="refresh"),
    ])
    with pytest.raises(ValueError, match="live pages"):
        validate(cfg, t_live_erase)

    # 5. Releasing all live pages with 'D' allows clean page-0 auto-erase and sequential rewrite,
    #    and reading a released page after rewrite of only page 0 fails as unwritten.
    t_ok = AddressTrace(make_meta(cfg, resident=[]), [
        _chunk(-1, 0, ["W", "W"], [b0[0], b0[1]], cls="prefill"),
        _chunk(0, 0, ["R", "R", "D", "D"], [b0[0], b0[1], b0[0], b0[1]]),
        _chunk(1, 0, ["W", "R"], [b0[0], b0[0]], cls="refresh"),
    ])
    validate(cfg, t_ok)

    t_read_after_erase = AddressTrace(make_meta(cfg, resident=[]), [
        _chunk(-1, 0, ["W", "W"], [b0[0], b0[1]], cls="prefill"),
        _chunk(0, 0, ["D", "D"], [b0[0], b0[1]]),
        _chunk(1, 0, ["W", "R"], [b0[0], b0[1]]),
    ])
    with pytest.raises(ValueError, match="unwritten"):
        validate(cfg, t_read_after_erase)
