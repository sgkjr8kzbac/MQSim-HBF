"""Serving workloads for the endurance model (M4): lengths, arrivals, shared prefixes, sessions.

Lengths are ``fixed:N`` (or ``N``), ``lognormal:median=M,sigma=S[,min=A,max=B]`` or
``uniform:lo=A,hi=B``. Arrivals are closed-loop (``concurrency`` requests, or sessions, in flight;
a finished one is replaced at once) or Poisson (``rate`` per second, at most ``max_batch``
running). A session (agentic workload) issues ``calls`` calls: the first writes the initial
history plus its new input, later calls reuse the history as a cached prefix and append their
new input and output to it; calls are ``think_s`` apart. Shared prefixes: a request uses one of
``prefix_groups`` system prompts of ``prefix_tokens`` with probability ``prefix_share``, the group
drawn from a Zipf law. Presets are in hbf/configs/workloads.json, each with its source tag.

The closed-form model (``endurance.analytic``) uses the length means (``Length.mean``, a
deterministic quantile grid) and the arrival parameters. The event simulation
(``endurance.simulate``) draws requests from a ``Stream``: seeded numpy generators (one each for
lengths, arrivals and prefix choices, spawned from ``seed``), so a run is reproducible.
``scripted(requests)`` builds a workload of explicit requests for tests.
"""
import csv
import heapq
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

import numpy as np

from ...device.config import WORKLOADS_JSON as PRESETS
ARRIVALS = ("closed", "poisson", "script")
KINDS = ("fixed", "lognormal", "uniform")


@dataclass(frozen=True)
class Length:
    kind: str
    params: tuple  # sorted (key, value) pairs

    def get(self, key, default=None):
        return dict(self.params).get(key, default)

    def _clip(self, value):
        lo, hi = self.get("min", 1), self.get("max")
        value = max(lo, int(round(value)))
        return value if hi is None else min(hi, value)

    def sample(self, rng):
        if self.kind == "fixed":
            return int(self.get("n"))
        if self.kind == "lognormal":
            return self._clip(self.get("median") * math.exp(self.get("sigma") * rng.standard_normal()))
        return self._clip(rng.uniform(self.get("lo"), self.get("hi")))

    def mean(self):
        """Mean of the sampled (rounded, clipped) values; deterministic (quantile grid)."""
        return self.mean_max(1)

    def mean_max(self, k):
        """Expected largest of k independent samples (k = 1: the mean); deterministic quantile grid."""
        if self.kind == "fixed":
            return float(self.get("n"))
        values = self._grid()
        weights = np.diff(np.linspace(0, 1, len(values) + 1) ** k)
        return float((values * weights).sum())

    def _grid(self):
        q = (np.arange(20000) + .5) / 20000
        if self.kind == "uniform":
            values = self.get("lo") + q * (self.get("hi") - self.get("lo"))
        else:
            values = self.get("median") * np.exp(self.get("sigma") * _probit(q))
        lo, hi = self.get("min", 1), self.get("max")
        values = np.maximum(lo, np.round(values))
        if hi is not None:
            values = np.minimum(hi, values)
        return values  # sorted, as the quantiles are

    def __str__(self):
        if self.kind == "fixed":
            return f"fixed:{self.get('n')}"
        return f"{self.kind}:" + ",".join(f"{key}={value:g}" for key, value in self.params)


def _probit(q):
    """Inverse standard normal CDF (Acklam's rational approximation, |error| < 1.2e-9)."""
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00)
    q = np.asarray(q, dtype=float)
    out = np.empty_like(q)
    low, high = q < .02425, q > 1 - .02425
    mid = ~(low | high)
    r = q[mid] - .5
    s = r * r
    out[mid] = ((((((a[0] * s + a[1]) * s + a[2]) * s + a[3]) * s + a[4]) * s + a[5]) * r /
                (((((b[0] * s + b[1]) * s + b[2]) * s + b[3]) * s + b[4]) * s + 1))
    for mask, sign, p in ((low, 1, q[low]), (high, -1, 1 - q[high])):
        t = np.sqrt(-2 * np.log(p))
        out[mask] = sign * ((((((c[0] * t + c[1]) * t + c[2]) * t + c[3]) * t + c[4]) * t + c[5]) /
                            ((((d[0] * t + d[1]) * t + d[2]) * t + d[3]) * t + 1))
    return out


def parse_length(text):
    """``fixed:N``, ``N``, ``lognormal:median=M,sigma=S[,min=,max=]`` or ``uniform:lo=,hi=[,min=,max=]``."""
    if isinstance(text, Length):
        return text
    if isinstance(text, int) and not isinstance(text, bool):
        text = f"fixed:{text}"
    text = str(text).strip()
    if text.isdigit():
        text = f"fixed:{text}"
    kind, _, rest = text.partition(":")
    if kind not in KINDS:
        raise ValueError(f"Unknown length {text!r}: use fixed:N, lognormal:median=,sigma= or uniform:lo=,hi=")
    if kind == "fixed":
        if not rest.isdigit() or int(rest) < 1:
            raise ValueError(f"fixed:N needs a positive integer, not {rest!r}")
        return Length("fixed", (("n", int(rest)),))
    params = {}
    for item in filter(None, rest.split(",")):
        key, sep, value = item.partition("=")
        try:
            params[key.strip()] = float(value)
        except ValueError:
            raise ValueError(f"Invalid length parameter {item!r} in {text!r}") from None
        if not sep:
            raise ValueError(f"Invalid length parameter {item!r} in {text!r}")
    need = {"lognormal": {"median", "sigma"}, "uniform": {"lo", "hi"}}[kind]
    if not need <= set(params) or set(params) - need - {"min", "max"}:
        raise ValueError(f"{kind} needs {sorted(need)} (optional min, max), got {sorted(params)}")
    if any(not math.isfinite(value) or value < 0 for value in params.values()) or params.get("min", 1) < 1:
        raise ValueError(f"Length parameters must be finite, non-negative and min >= 1: {text!r}")
    for key in ("min", "max"):
        if key in params:
            params[key] = int(params[key])
    if "max" in params and params["max"] < params.get("min", 1):
        raise ValueError(f"Length max must be >= min ({params.get('min', 1)}): {text!r}")
    if kind == "uniform" and params["lo"] > params["hi"]:
        raise ValueError(f"uniform needs lo <= hi: {text!r}")
    if kind == "lognormal" and params["median"] <= 0:
        raise ValueError(f"lognormal needs median > 0: {text!r}")
    return Length(kind, tuple(sorted(params.items())))


@dataclass(frozen=True)
class Workload:
    """Request stream; lengths in tokens, times in seconds."""
    name: str = "custom"
    input: Length = field(default_factory=lambda: parse_length("fixed:1024"))
    output: Length = field(default_factory=lambda: parse_length("fixed:128"))
    arrival: str = "closed"
    concurrency: int = 8
    rate: float | None = None
    max_batch: int = 256
    prefix_groups: int = 0
    prefix_tokens: int = 0
    prefix_share: float = 0.0
    zipf: float = 1.0
    calls: int = 0
    history: Length | None = None
    think_s: float = 0.0
    retain_s: float = 0.0
    seed: int = 1
    source: str = ""
    script: tuple = ()  # arrival "script": explicit requests (see ``scripted``)
    script_loop_s: float = 0.0  # arrival "script": optional loop period in seconds (0 = play once)

    def __post_init__(self):
        for name in ("input", "output", "history"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, Length):
                object.__setattr__(self, name, parse_length(value))
        if self.arrival not in ARRIVALS:
            raise ValueError(f"arrival must be one of {', '.join(ARRIVALS)}")
        if self.arrival == "closed" and (type(self.concurrency) is not int or self.concurrency < 1):
            raise ValueError("A closed-loop workload needs a positive integer concurrency")
        if self.arrival == "poisson" and (self.rate is None or self.rate <= 0):
            raise ValueError("A Poisson workload needs a positive rate")
        if (self.arrival == "script") != bool(self.script):
            raise ValueError("Scripted requests need arrival 'script' (use workload.scripted)")
        if self.script_loop_s < 0 or not math.isfinite(self.script_loop_s):
            raise ValueError("script_loop_s must be finite and non-negative")
        if type(self.max_batch) is not int or self.max_batch < 1:
            raise ValueError("max_batch must be a positive integer")
        if self.prefix_groups < 0 or self.prefix_tokens < 0 or not 0 <= self.prefix_share <= 1:
            raise ValueError("prefix_groups and prefix_tokens must be >= 0 and prefix_share in [0, 1]")
        if self.prefix_share and not (self.prefix_groups and self.prefix_tokens):
            raise ValueError("prefix_share needs prefix_groups and prefix_tokens")
        if self.calls < 0 or self.think_s < 0 or self.retain_s < 0:
            raise ValueError("calls, think_s and retain_s must be non-negative")
        if self.calls and self.prefix_share:
            raise ValueError("Sessions and shared prefixes cannot be combined")

    @property
    def sessions(self):
        return self.calls > 0

    def describe(self):
        text = f"`{self.name}`：输入 {self.input}，输出 {self.output}"
        if self.sessions:
            history = f"、初始历史 {self.history}" if self.history is not None else ""
            text += f"，会话 {self.calls} 次调用{history}、调用间隔 {self.think_s:g} s"
        if self.prefix_share:
            text += (f"，共享前缀 {self.prefix_groups} 组 × {self.prefix_tokens} token，"
                     f"命中比例 {self.prefix_share:g}，Zipf {self.zipf:g}")
        if self.retain_s:
            text += f"，空闲 KV 保留 {self.retain_s:g} s"
        if self.arrival == "closed":
            arrival = f"闭环并发 {self.concurrency}"
        elif self.arrival == "poisson":
            arrival = f"泊松到达 {self.rate:g}/s（最多 {self.max_batch} 个同时解码）"
        else:
            loop_note = f"（循环周期 {self.script_loop_s:g} s）" if self.script_loop_s > 0 else ""
            arrival = f"脚本 {len(self.script)} 个请求{loop_note}"
        return text + f"；{arrival}，种子 {self.seed}"


def presets():
    with open(PRESETS) as stream:
        return json.load(stream)


def load_workload(name, **overrides):
    """A preset (hbf/configs/workloads.json) with overrides; ``long`` is an alias of ``long32k``."""
    table = presets()
    name = "long32k" if name == "long" else name
    if name not in table:
        raise ValueError(f"Unknown workload {name!r}: presets are {', '.join(table)}")
    spec = dict(table[name])
    sessions = spec.pop("sessions", None) or {}
    values = {"name": name, "input": spec["input"], "output": spec["output"], "source": spec.get("source", "")}
    values |= {key: spec[key] for key in ("prefix_groups", "prefix_tokens", "prefix_share", "zipf", "retain_s")
               if key in spec}
    if sessions:
        values |= {"calls": sessions["calls"], "history": sessions.get("history"), "think_s": sessions.get("think_s", 0)}
    known = {item.name for item in fields(Workload)}
    unknown = set(overrides) - known
    if unknown:
        raise ValueError(f"Unknown workload fields: {', '.join(sorted(unknown))}")
    return Workload(**(values | {key: value for key, value in overrides.items() if value is not None}))


def with_overrides(workload, **overrides):
    return replace(workload, **{key: value for key, value in overrides.items() if value is not None})


def per_stack(workload, stacks):
    """The share of one HBF stack: concurrency / stacks (rounded, >= 1), rate / stacks, or sharded script.

    For scripted workloads with shared prefixes, conversation/branch groups are routed by
    ``prefix % stacks`` (preserving multi-turn cache locality), while unprefixed requests or
    cluster-wide universal prefixes (shared by > ``N / (2 * stacks)`` requests) are balanced
    round-robin across stacks.
    """
    if stacks == 1:
        return workload
    if workload.arrival == "closed":
        return replace(workload, concurrency=max(1, round(workload.concurrency / stacks)))
    if workload.arrival == "poisson":
        return replace(workload, rate=workload.rate / stacks)
    if workload.arrival == "script":
        n = len(workload.script)
        pcounts = Counter(int(item["prefix"]) for item in workload.script if int(item.get("prefix", -1)) >= 0)
        global_thresh = max(16.0, n / (2.0 * stacks))
        sliced = []
        rr = 0
        for item in workload.script:
            pid = int(item.get("prefix", -1))
            if pid < 0 or pcounts[pid] > global_thresh:
                if rr % stacks == 0:
                    sliced.append(item)
                rr += 1
            elif pid % stacks == 0:
                sliced.append(item)
        sliced = tuple(sliced) or workload.script[:1]
        scale = len(sliced) / n
        new_rate = workload.rate * scale if workload.rate else None
        return replace(workload, script=sliced, rate=new_rate)
    return workload


@dataclass
class Request:
    """One decode request: a standalone request or one call of a session (tokens, seconds)."""
    id: int
    arrival: float
    input: int  # uncached tokens written at prefill (a session's first call: its history plus input)
    output: int  # tokens to decode
    history: int = 0  # cached session tokens at the call's start (already on HBF)
    prefix: int = -1  # shared-prefix group, or -1
    session: int = -1
    call: int = 0
    last: bool = True  # the session's last call (standalone requests: True)
    retain_s: float = 0.0  # idle lifetime of the request's KV after it completes
    prefix_tokens: int = 0  # per-request shared prefix length (0 -> use workload.prefix_tokens)


class Stream:
    """Seeded request source for ``endurance.simulate``.

    * Closed loop, standalone requests: ``closed_new(now)`` draws a request for a freed slot.
      The first ``concurrency`` requests start staggered (``stagger``): the i-th decodes
      ``out - round(i out / C)`` tokens, so completions spread over one request lifetime as in
      steady state (a warm-up device; the simulation excludes its warm-up from rates).
    * Closed loop, sessions: ``concurrency`` sessions start at time 0 (staggered: the i-th starts
      at call floor(i calls / C), with the earlier calls' history written at once); a session's
      next call arrives ``think_s`` after the previous one completes (``call_done``), and a new
      session replaces a finished one at once.
    * Poisson: requests (or sessions) arrive with exponential gaps of mean 1 / rate.
    * Script: the workload's explicit requests at their arrival times.

    Timed arrivals sit in a heap: ``arrived(now)`` pops those due, ``next_time()`` peeks.
    """

    def __init__(self, workload, stagger=True):
        self.workload = workload
        lengths, arrivals, prefixes = np.random.SeedSequence(workload.seed).spawn(3)
        self.rng_len = np.random.Generator(np.random.PCG64(lengths))
        self.rng_arr = np.random.Generator(np.random.PCG64(arrivals))
        self.rng_pre = np.random.Generator(np.random.PCG64(prefixes))
        self.stagger = stagger
        self.count = 0  # requests created
        self.sessions = 0
        self.heap, self.order = [], 0
        self.next_poisson = None
        self.prefix_p = None
        self._script_items = ()
        self._script_loop_s = 0.0
        self._script_next_cycle = 0
        self._script_first_arrival = 0.0
        w = workload
        if w.prefix_groups:
            weights = 1.0 / np.arange(1, w.prefix_groups + 1) ** w.zipf
            self.prefix_p = weights / weights.sum()
        if w.arrival == "script":
            self._script_items = tuple(w.script)
            self._script_loop_s = float(w.script_loop_s)
            self._script_first_arrival = min(float(item.get("arrival", 0.0)) for item in self._script_items)
            self._push_script_cycle(0)
            self._script_next_cycle = 1
        elif w.arrival == "poisson":
            self.next_poisson = float(self.rng_arr.exponential(1 / w.rate))
        elif w.sessions:
            for i in range(w.concurrency):
                self._new_session(0.0, (i * w.calls) // w.concurrency if stagger else 0)

    def _push_script_cycle(self, cycle):
        w = self.workload
        offset = cycle * self._script_loop_s
        for item in self._script_items:
            self._push(Request(
                self._id(),
                float(item.get("arrival", 0.0)) + offset,
                int(item["input"]),
                int(item["output"]),
                prefix=int(item.get("prefix", -1)),
                retain_s=float(item.get("retain_s", w.retain_s)),
                prefix_tokens=int(item.get("prefix_tokens", 0)),
            ))

    def _fill_script(self, until):
        if self._script_loop_s <= 0:
            return
        while self._script_next_cycle * self._script_loop_s + self._script_first_arrival <= until + 1e-12:
            self._push_script_cycle(self._script_next_cycle)
            self._script_next_cycle += 1

    def _id(self):
        self.count += 1
        return self.count - 1

    def _push(self, request):
        heapq.heappush(self.heap, (request.arrival, self.order, request))
        self.order += 1

    def _prefix(self):
        w = self.workload
        if w.prefix_share and self.rng_pre.random() < w.prefix_share:
            return int(self.rng_pre.choice(w.prefix_groups, p=self.prefix_p))
        return -1

    def _standalone(self, now, shorten=0.0):
        w = self.workload
        n_in, n_out = w.input.sample(self.rng_len), w.output.sample(self.rng_len)
        n_out = max(1, n_out - round(shorten * n_out))
        return Request(self._id(), now, n_in, n_out, prefix=self._prefix(), retain_s=w.retain_s)

    def _new_session(self, now, start=0):
        """A session's first call; ``start`` > 0 begins mid-session with the earlier calls' history."""
        w = self.workload
        session, self.sessions = self.sessions, self.sessions + 1
        history = w.history.sample(self.rng_len) if w.history is not None else 0
        for _ in range(start):
            history += w.input.sample(self.rng_len) + w.output.sample(self.rng_len)
        self._push(Request(self._id(), now, history + w.input.sample(self.rng_len), w.output.sample(self.rng_len),
                           session=session, call=start, last=start == w.calls - 1, retain_s=w.retain_s))

    def closed_new(self, now):
        """A new standalone request for a freed closed-loop slot (None for other arrival kinds)."""
        w = self.workload
        if w.arrival != "closed" or w.sessions:
            return None
        i = self.count
        return self._standalone(now, i / w.concurrency if self.stagger and i < w.concurrency else 0.0)

    def _fill_poisson(self, until):
        w = self.workload
        while self.next_poisson is not None and self.next_poisson <= until:
            if w.sessions:
                self._new_session(self.next_poisson)
            else:
                self._push(self._standalone(self.next_poisson))
            self.next_poisson += float(self.rng_arr.exponential(1 / w.rate))

    def arrived(self, now):
        """Requests whose arrival time is <= now, in arrival order."""
        self._fill_poisson(now)
        self._fill_script(now)
        out = []
        while self.heap and self.heap[0][0] <= now + 1e-12:
            out.append(heapq.heappop(self.heap)[2])
        return out

    def next_time(self):
        """Time of the next timed arrival (inf when none is scheduled)."""
        times = [self.heap[0][0]] if self.heap else []
        if self.next_poisson is not None:
            times.append(self.next_poisson)
        if self._script_loop_s > 0:
            times.append(self._script_next_cycle * self._script_loop_s + self._script_first_arrival)
        return min(times) if times else math.inf

    def call_done(self, request, now):
        """Schedule a session's next call ``think_s`` later, or (closed loop) a replacement session."""
        w = self.workload
        if request.session < 0:
            return
        if request.last:
            if w.arrival == "closed":
                self._new_session(now)
            return
        call = request.call + 1
        history = request.history + request.input + request.output
        self._push(Request(self._id(), now + w.think_s, w.input.sample(self.rng_len), w.output.sample(self.rng_len),
                           history=history, session=request.session, call=call, last=call == w.calls - 1,
                           retain_s=w.retain_s))


def scripted(requests, **overrides):
    """A workload of explicit requests: dicts with ``arrival`` (s), ``input``, ``output`` and optional
    ``retain_s``, ``prefix`` (shared-prefix group), and ``prefix_tokens``."""
    script = tuple(dict(item) for item in requests)
    if not script:
        raise ValueError("scripted() needs at least one request")
    return Workload(**({"name": "script", "arrival": "script", "script": script} | overrides))


# ---- Unified request-level trace schema & public dataset importers (R7) ----------------------

REQUEST_CSV_REQUIRED = ("arrival_s", "prompt_tokens", "output_tokens")
REQUEST_CSV_OPTIONAL = ("prefix_id", "prefix_tokens")
REQUEST_CSV_ALLOWED = set(REQUEST_CSV_REQUIRED + REQUEST_CSV_OPTIONAL)
PREFIX_MODES = ("conversation", "longest", "none")


@dataclass(frozen=True)
class RequestRecord:
    """One request in the unified request-level trace format.

    Canonical CSV columns are ``arrival_s,prompt_tokens,output_tokens[,prefix_id]``.
    An optional fifth column ``prefix_tokens`` stores per-group shared-prefix length in tokens
    (populated automatically when importing block-hash traces such as Mooncake).
    """
    arrival_s: float
    prompt_tokens: int
    output_tokens: int
    prefix_id: int | None = None
    prefix_tokens: int | None = None

    def __post_init__(self):
        if not math.isfinite(self.arrival_s) or self.arrival_s < 0:
            raise ValueError(f"arrival_s must be finite and >= 0, got {self.arrival_s!r}")
        if type(self.prompt_tokens) is not int or self.prompt_tokens < 1:
            raise ValueError(f"prompt_tokens must be a positive int, got {self.prompt_tokens!r}")
        if type(self.output_tokens) is not int or self.output_tokens < 1:
            raise ValueError(f"output_tokens must be a positive int, got {self.output_tokens!r}")
        if self.prefix_id is not None and (type(self.prefix_id) is not int or self.prefix_id < 0):
            raise ValueError(f"prefix_id must be None or a non-negative int, got {self.prefix_id!r}")
        if self.prefix_tokens is not None:
            if type(self.prefix_tokens) is not int or not (0 <= self.prefix_tokens <= self.prompt_tokens):
                raise ValueError(
                    f"prefix_tokens must be in [0, prompt_tokens={self.prompt_tokens}], got {self.prefix_tokens!r}"
                )


def read_requests_csv(path):
    """Read unified request-level CSV ``arrival_s,prompt_tokens,output_tokens[,prefix_id[,prefix_tokens]]``."""
    path = Path(path)
    records = []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"Empty request trace CSV: {path}")
        names = [name.strip() for name in reader.fieldnames]
        if any(col not in names for col in REQUEST_CSV_REQUIRED):
            raise ValueError(
                f"Request trace CSV must include columns {list(REQUEST_CSV_REQUIRED)}, got {names}"
            )
        unknown = set(names) - REQUEST_CSV_ALLOWED
        if unknown:
            raise ValueError(f"Unknown columns in request trace CSV: {sorted(unknown)}")
        if "prefix_tokens" in names and "prefix_id" not in names:
            raise ValueError("Request trace CSV has 'prefix_tokens' without 'prefix_id'")
        has_pid = "prefix_id" in names
        has_ptok = "prefix_tokens" in names
        for idx, row in enumerate(reader, start=2):
            if None in row or any(row.get(col) in (None, "") for col in REQUEST_CSV_REQUIRED):
                raise ValueError(f"Invalid request trace CSV row {idx} in {path}: {row}")
            try:
                arrival_s = float(row["arrival_s"])
                prompt_tokens = int(row["prompt_tokens"])
                output_tokens = int(row["output_tokens"])
                pid_raw = row["prefix_id"].strip() if has_pid and row.get("prefix_id") is not None else ""
                if pid_raw in ("", "-1"):
                    prefix_id = None
                else:
                    prefix_id = int(pid_raw)
                ptok_raw = row["prefix_tokens"].strip() if has_ptok and row.get("prefix_tokens") is not None else ""
                if ptok_raw in ("", "0"):
                    prefix_tokens = None
                else:
                    prefix_tokens = int(ptok_raw)
            except ValueError as err:
                raise ValueError(f"Cannot parse row {idx} in {path}: {err}") from None
            if prefix_id is None:
                prefix_tokens = None
            records.append(RequestRecord(
                arrival_s=arrival_s,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                prefix_id=prefix_id,
                prefix_tokens=prefix_tokens,
            ))
    if not records:
        raise ValueError(f"Request trace CSV contains no requests: {path}")
    if any(records[i].arrival_s < records[i - 1].arrival_s for i in range(1, len(records))):
        records.sort(key=lambda r: r.arrival_s)
    return records


def write_requests_csv(path, records, *, include_prefix=None, include_prefix_tokens=None):
    """Write ``RequestRecord`` sequence to unified CSV ``arrival_s,prompt_tokens,output_tokens[,prefix_id[,prefix_tokens]]``."""
    path = Path(path)
    records = list(records)
    if not records:
        raise ValueError("write_requests_csv() requires at least one RequestRecord")
    if include_prefix_tokens is None:
        include_prefix_tokens = any(r.prefix_tokens is not None for r in records)
    if include_prefix is None:
        include_prefix = include_prefix_tokens or any(r.prefix_id is not None for r in records)
    if include_prefix_tokens and not include_prefix:
        include_prefix = True
    header = list(REQUEST_CSV_REQUIRED)
    if include_prefix:
        header.append("prefix_id")
    if include_prefix_tokens:
        header.append("prefix_tokens")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        for rec in records:
            row = [f"{rec.arrival_s:.6f}".rstrip("0").rstrip(".") or "0", rec.prompt_tokens, rec.output_tokens]
            if include_prefix:
                row.append("" if rec.prefix_id is None else rec.prefix_id)
            if include_prefix_tokens:
                row.append("" if rec.prefix_tokens is None else rec.prefix_tokens)
            writer.writerow(row)
    return path


def import_mooncake_jsonl(path, *, block_tokens=512, max_requests=None, start_s=None,
                          duration_s=None, min_prefix_count=2, prefix_mode="conversation",
                          shift_start=True):
    """Import Mooncake FAST'25 JSONL trace into unified ``RequestRecord`` objects.

    Each JSON line has ``timestamp`` (ms), ``input_length``, ``output_length``, and optional
    ``hash_ids`` (one hash per ``block_tokens``-token KV block, default 512).

    Prefix grouping modes (``prefix_mode``):
    - ``"conversation"`` (default): assigns each request to the maximal shared prefix chain of its
      conversation/branch plateau (skipping any single universal root block shared by all requests
      when a deeper shared prefix of ``>= min_prefix_count`` requests exists). Ensures every
      assigned group has ``>= min_prefix_count`` members and consistent ``prefix_tokens``.
    - ``"longest"``: assigns the longest leading ``hash_ids`` prefix appearing in
      ``>= min_prefix_count`` requests.
    - ``"none"``: ignores ``hash_ids`` (``prefix_id=None``, ``prefix_tokens=None``).
    """
    if type(block_tokens) is not int or block_tokens < 1:
        raise ValueError(f"block_tokens must be a positive int, got {block_tokens!r}")
    if type(min_prefix_count) is not int or min_prefix_count < 1:
        raise ValueError(f"min_prefix_count must be a positive int, got {min_prefix_count!r}")
    if prefix_mode not in PREFIX_MODES:
        raise ValueError(f"prefix_mode must be one of {PREFIX_MODES}, got {prefix_mode!r}")
    if max_requests is not None and max_requests < 1:
        raise ValueError("max_requests must be >= 1")
    if start_s is not None and start_s < 0:
        raise ValueError("start_s must be >= 0")
    if duration_s is not None and duration_s <= 0:
        raise ValueError("duration_s must be > 0")

    path = Path(path)
    raw = []
    t0 = None
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as err:
                raise ValueError(f"Invalid JSON at line {line_no} of {path}: {err}") from None
            for req_key in ("timestamp", "input_length", "output_length"):
                if req_key not in obj:
                    raise ValueError(f"Missing {req_key!r} at line {line_no} of {path}")
            ts_s = float(obj["timestamp"]) / 1000.0
            if t0 is None:
                t0 = ts_s
            rel_s = ts_s - t0
            if start_s is not None and rel_s < start_s - 1e-12:
                continue
            if duration_s is not None:
                win_start = start_s or 0.0
                if rel_s >= win_start + duration_s - 1e-12:
                    break
            prompt = max(1, int(obj["input_length"]))
            output = max(1, int(obj["output_length"]))
            hashes = tuple(int(h) for h in (obj.get("hash_ids") or ()))
            raw.append((rel_s, prompt, output, hashes))
            if max_requests is not None and len(raw) >= max_requests:
                break

    if not raw:
        raise ValueError(f"No Mooncake requests selected from {path}")
    if any(raw[i][0] < raw[i - 1][0] for i in range(1, len(raw))):
        raw.sort(key=lambda item: item[0])
    base_s = raw[0][0] if shift_start else 0.0
    n = len(raw)

    keys = [None] * n
    if prefix_mode != "none":
        prefix_counts = Counter()
        for _, _, _, hashes in raw:
            for k in range(1, len(hashes) + 1):
                prefix_counts[hashes[:k]] += 1
        for i, (_, _, _, hashes) in enumerate(raw):
            if not hashes or prefix_counts[(hashes[0],)] < min_prefix_count:
                continue
            if prefix_mode == "longest":
                for k in range(len(hashes), 0, -1):
                    key = hashes[:k]
                    if prefix_counts[key] >= min_prefix_count:
                        keys[i] = key
                        break
            else:  # "conversation"
                k = 1
                while k < len(hashes) and prefix_counts[hashes[:k]] == n and prefix_counts[hashes[:k + 1]] >= min_prefix_count:
                    k += 1
                base_c = prefix_counts[hashes[:k]]
                while k < len(hashes) and prefix_counts[hashes[:k + 1]] == base_c:
                    k += 1
                keys[i] = hashes[:k]

    group_prompts = defaultdict(list)
    for i, key in enumerate(keys):
        if key is not None:
            group_prompts[key].append(raw[i][1])

    key_to_id = {}
    key_to_ptok = {}
    for key in keys:
        if key is not None and key not in key_to_id:
            key_to_id[key] = len(key_to_id)
            key_to_ptok[key] = min(min(group_prompts[key]), len(key) * block_tokens)

    records = []
    for i, (rel_s, prompt, output, _) in enumerate(raw):
        key = keys[i]
        pid = key_to_id[key] if key is not None else None
        ptok = key_to_ptok[key] if key is not None else None
        records.append(RequestRecord(
            arrival_s=max(0.0, rel_s - base_s),
            prompt_tokens=prompt,
            output_tokens=output,
            prefix_id=pid,
            prefix_tokens=ptok,
        ))
    return records


def load_requests(path, *, fmt="auto", **kwargs):
    """Load request records from unified CSV or Mooncake JSONL (``fmt``: ``auto``, ``csv``, ``mooncake``)."""
    path = Path(path)
    if fmt == "auto":
        fmt = "mooncake" if path.suffix.lower() in (".jsonl", ".ndjson") else "csv"
    if fmt == "csv":
        records = read_requests_csv(path)
        start_s = kwargs.get("start_s")
        duration_s = kwargs.get("duration_s")
        max_requests = kwargs.get("max_requests")
        if start_s is not None or duration_s is not None:
            t0 = records[0].arrival_s
            lo = t0 + (start_s or 0.0)
            hi = (lo + duration_s) if duration_s is not None else math.inf
            records = [r for r in records if lo - 1e-12 <= r.arrival_s < hi - 1e-12]
        if max_requests is not None:
            records = records[:max_requests]
        if not records:
            raise ValueError(f"No CSV requests selected from {path}")
        if kwargs.get("shift_start", True) and records[0].arrival_s > 0:
            shift = records[0].arrival_s
            records = [replace(r, arrival_s=max(0.0, r.arrival_s - shift)) for r in records]
        return records
    if fmt == "mooncake":
        return import_mooncake_jsonl(path, **kwargs)
    raise ValueError(f"Unknown requests format {fmt!r}: choose 'auto', 'csv', or 'mooncake'")


def summarize_requests(records):
    """Return descriptive statistics (rates, length quantiles, prefix sharing) for ``RequestRecord``s."""
    records = list(records)
    if not records:
        raise ValueError("summarize_requests() requires at least one record")
    arrivals = np.array([r.arrival_s for r in records], dtype=np.float64)
    prompts = np.array([r.prompt_tokens for r in records], dtype=np.int64)
    outputs = np.array([r.output_tokens for r in records], dtype=np.int64)
    span_s = float(arrivals[-1] - arrivals[0])
    seen = set()
    hit_reqs = 0
    hit_tokens = 0
    shared_reqs = 0
    ptok_list = []
    groups = set()
    for r in records:
        if r.prefix_id is not None:
            groups.add(r.prefix_id)
            shared_reqs += 1
            pt = r.prefix_tokens or 0
            ptok_list.append(pt)
            if r.prefix_id in seen:
                hit_reqs += 1
                hit_tokens += pt
            seen.add(r.prefix_id)
    total_prompt = int(prompts.sum())
    return {
        "count": len(records),
        "duration_s": span_s,
        "rate_per_s": len(records) / span_s if span_s > 0 else 0.0,
        "prompt_min": int(prompts.min()),
        "prompt_p50": float(np.median(prompts)),
        "prompt_mean": float(prompts.mean()),
        "prompt_p90": float(np.percentile(prompts, 90)),
        "prompt_max": int(prompts.max()),
        "output_min": int(outputs.min()),
        "output_p50": float(np.median(outputs)),
        "output_mean": float(outputs.mean()),
        "output_p90": float(np.percentile(outputs, 90)),
        "output_max": int(outputs.max()),
        "prefix_groups": len(groups),
        "prefix_share": shared_reqs / len(records),
        "prefix_tokens_mean": float(np.mean(ptok_list)) if ptok_list else 0.0,
        "causal_hit_rate": hit_reqs / len(records),
        "causal_cached_token_share": hit_tokens / total_prompt if total_prompt else 0.0,
    }


def workload_from_requests(records, *, name="requests", prefix_tokens=None, retain_s=0.0,
                           request_retain_s=0.0, max_batch=256, loop=True, loop_s=None,
                           source="", seed=1):
    """Build a scripted ``Workload`` from ``RequestRecord`` objects.

    Each record with ``prefix_id is not None`` splits its ``prompt_tokens`` into a shared prefix
    of ``pt = min(record.prefix_tokens or prefix_tokens or 0, prompt_tokens)`` tokens plus
    ``input = prompt_tokens - pt`` uncached prompt tokens. Shared prefix entries are kept idle for
    ``retain_s`` seconds after their last active reader completes; private suffixes use
    ``request_retain_s`` (default 0.0).

    When ``loop=True`` and the trace has a positive time span (or ``loop_s`` is specified), the
    script replays periodically so multi-hour lifetime simulations sustain steady-state traffic.
    """
    records = list(records)
    if not records:
        raise ValueError("workload_from_requests() requires at least one record")
    if retain_s < 0 or request_retain_s < 0:
        raise ValueError("retain_s and request_retain_s must be non-negative")
    if prefix_tokens is not None and prefix_tokens < 0:
        raise ValueError("prefix_tokens must be non-negative")

    script = []
    uncached_inputs = []
    outputs = []
    used_prefix_ids = set()
    used_prefix_tokens = []
    for rec in records:
        pt = 0
        pid = -1
        if rec.prefix_id is not None:
            raw_pt = rec.prefix_tokens if rec.prefix_tokens is not None else (prefix_tokens or 0)
            pt = min(int(raw_pt), rec.prompt_tokens)
            if pt > 0:
                pid = rec.prefix_id
                used_prefix_ids.add(pid)
                used_prefix_tokens.append(pt)
        own_in = rec.prompt_tokens - pt
        item = {
            "arrival": float(rec.arrival_s),
            "input": int(own_in),
            "output": int(rec.output_tokens),
            "retain_s": float(request_retain_s),
        }
        if pid >= 0:
            item["prefix"] = int(pid)
            item["prefix_tokens"] = int(pt)
        script.append(item)
        uncached_inputs.append(own_in)
        outputs.append(rec.output_tokens)

    span_s = max(r.arrival_s for r in records) - min(r.arrival_s for r in records)
    if loop:
        if loop_s is not None:
            cycle_s = float(loop_s)
        elif span_s > 0:
            cycle_s = span_s + span_s / max(1, len(records) - 1)
        else:
            cycle_s = 0.0
    else:
        cycle_s = 0.0

    rate_span = cycle_s if cycle_s > 0 else span_s
    mean_rate = len(records) / rate_span if rate_span > 0 else None
    mean_in = max(1, round(float(np.mean(uncached_inputs))))
    mean_out = max(1, round(float(np.mean(outputs))))
    n_groups = len(used_prefix_ids)
    mean_pt = max(1, round(float(np.mean(used_prefix_tokens)))) if used_prefix_tokens else 0
    share = len(used_prefix_tokens) / len(records) if used_prefix_tokens else 0.0

    return Workload(
        name=name,
        input=parse_length(f"fixed:{mean_in}"),
        output=parse_length(f"fixed:{mean_out}"),
        arrival="script",
        rate=mean_rate,
        max_batch=int(max_batch),
        prefix_groups=n_groups,
        prefix_tokens=mean_pt,
        prefix_share=share,
        retain_s=float(retain_s),
        seed=int(seed),
        source=source or f"Imported request trace ({len(records)} requests)",
        script=tuple(script),
        script_loop_s=cycle_s,
    )
