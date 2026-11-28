"""Built-in extension ``hbf.ext.system``: GPU roofline, decode TPOT, TPOT clock, and GPU prefill rate."""
from .clock import Clock
from .evaluate import (KV_CONSUME, REQUIRED_ROW_KEYS, ROW_KEYS, SELECTIONS, WEIGHTS, SystemOptions,
                       SystemResult, evaluate, linear_ns, main, parse_overlap, read_layers, report_lines,
                       system_meta, timeline, write_outputs)
from .gpu import GPUS, GpuSpec, load_gpu, prefill_tok_s


class _SystemEvaluator:
    """Adapter exposing ``hbf.ext.system`` to core via ``registry.evaluators['system']``."""

    SystemOptions = SystemOptions
    load_gpu = staticmethod(load_gpu)
    evaluate = staticmethod(evaluate)
    report_lines = staticmethod(report_lines)
    main = staticmethod(main)

    @staticmethod
    def step_tpot_ns(layer_rows, model_spec, gpu_name, *, batch, mode, kv_format):
        gpu_spec = load_gpu(gpu_name)
        selection = "static" if mode in ("dense", "sink_recent") else "dynamic"
        meta = {"model": model_spec.name, "gpu": gpu_spec.name, "batch": batch, "stacks": 1,
                "analysis_batch": batch, "selection": selection, "kv_format": kv_format,
                "overlap": "serial", "kv_consume": "stream"}
        return evaluate(layer_rows, model_spec, gpu_spec, SystemOptions(), meta).steps[0]["tpot_hbf_est_ns"]


def register(registry):
    registry.register_clock("tpot", Clock)
    registry.register_prefill_rate("gpu_roofline", prefill_tok_s)
    registry.register_evaluator("system", _SystemEvaluator())
