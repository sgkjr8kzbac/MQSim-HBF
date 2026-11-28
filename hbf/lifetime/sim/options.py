"""Options, constants, and exceptions for the segment-level lifetime event simulator."""
from dataclasses import dataclass

from ..geometry import CAUSES
from ..interference import MODES as M5_MODES

SIM_POLICIES = ("packed", "dedicated", "hbm_decode", "layer_major", "vllm_pool")
WEAR = ("none", "dynamic", "ideal", "remap")
REFRESH = ("auto", "relocate", "replay")
VICTIMS = ("greedy", "cost_benefit")
CAUSE = {cause: index for index, cause in enumerate(CAUSES)}
EPS = 1e-6
OPEN, FULL = 1, 2  # segment states (0: never written)
ACTIVE, IDLE = 1, 2  # entry states (0: unused)
PRIVATE, PREFIX, SESSION = 0, 1, 2  # entry kinds


class CapacityError(RuntimeError):
    """No free segment for a write or an admission (raise --capacity-gib or lower the load)."""


@dataclass(frozen=True)
class SimOptions:
    """Settings of one event simulation (CLI names in ``build_parser``)."""
    policy: str = "packed"
    x: float = 1e6  # read-disturb threshold in block reads
    hot: float = 1.0  # persistent read skew (max over mean block reads)
    share: float = 1.0  # pages sensed per stored page per decode step
    retention_s: float = 86400.0  # 0: retention refresh off
    prefix_reads: str = "per_request"
    hours: float = 24.0
    warmup_h: float = 1.0
    quantum_s: float = 1.0
    report_s: float = 60.0
    rd_slack: float = 0.01  # shortest quantum as a share of the fastest segment's threshold in steps
    wear: str = "dynamic"
    refresh: str = "auto"  # auto: replay for layer_major, relocate otherwise
    refresh_idle: str = "relocate"  # or evict: idle entries in a refreshed segment are dropped
    reclaim: str = "evict"  # or compact: idle entries in a reclaim victim are copied too
    victim: str = "greedy"  # greedy (fewest pages to copy) or cost_benefit ((1 - u) * age / (1 + u))
    prefill: str = "disaggregated"  # disaggregated or colocated (adds input / prefill_tok_s per admission)
    prefill_rate: float | None = None  # optional explicit prefill tok/s (core, no GPU required)
    pre_open: bool = False  # pre-open free segments in idle time (takes non-replay erases off M5)
    separate_prefix: bool = False  # keep shared prefixes on their own write frontier per partition
    low_watermark: int | None = None  # free segments per partition kept by reclaim (default 2 + one request)
    align: str = "row"  # packed: a new entry starts on a row
    slots: int | None = None  # running requests at most (layer_major: slots per wave)
    stagger: bool = True
    stacks: int = 1  # for the summary only (the workload is already one stack's share)
    m5: str = "gaps"
    t_prog: float = 100000.0
    t_bers: float = 1000000.0
    suspend_ns: float | None = None
    block_unit: str = "die"
    pe_limits: tuple = (1e4, 1e5, 1e6)
    block_tokens: int | None = None  # vllm_pool: tokens per vLLM block (default: lcm of HBF tpp)
    free_order: str = "fresh"  # vllm_pool: fresh (ascending ids) or shuffled:<seed>
    vllm_reuse: str = "fifo"  # vllm_pool: fifo (vLLM default) or lifo

    def __post_init__(self):
        if self.policy not in SIM_POLICIES:
            raise ValueError(f"The simulation supports the policies {', '.join(SIM_POLICIES)}")
        if self.wear not in WEAR or self.refresh not in REFRESH:
            raise ValueError(f"wear must be one of {', '.join(WEAR)} and refresh one of {', '.join(REFRESH)}")
        if self.refresh_idle not in ("relocate", "evict") or self.reclaim not in ("evict", "compact"):
            raise ValueError("refresh_idle is relocate or evict; reclaim is evict or compact")
        if self.victim not in VICTIMS:
            raise ValueError(f"victim must be one of {', '.join(VICTIMS)}")
        if self.prefill not in ("disaggregated", "colocated"):
            raise ValueError("prefill must be disaggregated or colocated")
        if self.prefix_reads not in ("per_request", "once") or self.align not in ("row", "none"):
            raise ValueError("prefix_reads is per_request or once; align is row or none")
        if self.m5 not in M5_MODES + ("off",):
            raise ValueError(f"m5 must be one of {', '.join(M5_MODES + ('off',))}")
        if self.m5 == "suspend" and self.suspend_ns is None:
            raise ValueError("m5 suspend needs suspend_ns")
        if min(self.x, self.hot, self.hours, self.quantum_s, self.report_s) <= 0 or not 0 < self.share <= 1:
            raise ValueError("x, hot, hours, quantum and report interval must be positive and share in (0, 1]")
        if self.warmup_h < 0 or self.warmup_h >= self.hours:
            raise ValueError("The warm-up must be non-negative and shorter than the simulated time")
        if self.retention_s < 0 or not 0 <= self.rd_slack < 1:
            raise ValueError("retention_s must be >= 0 and rd_slack in [0, 1)")
        if self.low_watermark is not None and self.low_watermark < 1:
            raise ValueError("low_watermark must be at least 1 segment")
        if self.slots is not None and self.slots < 1:
            raise ValueError("slots must be positive")
        if self.policy == "layer_major" and self.refresh == "relocate":
            raise ValueError("layer_major regions are refreshed in place (replay, spec section 5.7)")
        if self.block_tokens is not None and self.block_tokens < 1:
            raise ValueError("block_tokens must be positive")
        if self.free_order != "fresh" and not self.free_order.startswith("shuffled:"):
            raise ValueError("free_order must be fresh or shuffled:<seed>")
        if self.free_order.startswith("shuffled:"):
            try:
                int(self.free_order[9:])
            except ValueError:
                raise ValueError(f"Invalid free_order {self.free_order!r}: use fresh or shuffled:<int>") from None
        if self.vllm_reuse not in ("fifo", "lifo"):
            raise ValueError("vllm_reuse must be fifo or lifo")
