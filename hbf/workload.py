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
import heapq
import json
import math
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

import numpy as np

PRESETS = Path(__file__).resolve().parent / "configs" / "workloads.json"
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
    if any(value < 0 for value in params.values()) or params.get("min", 1) < 1:
        raise ValueError(f"Length parameters must be non-negative and min >= 1: {text!r}")
    for key in ("min", "max"):
        if key in params:
            params[key] = int(params[key])
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
            text += f"，会话 {self.calls} 次调用、初始历史 {self.history}、调用间隔 {self.think_s:g} s"
        if self.prefix_share:
            text += (f"，共享前缀 {self.prefix_groups} 组 × {self.prefix_tokens} token，"
                     f"命中比例 {self.prefix_share:g}，Zipf {self.zipf:g}")
        if self.retain_s:
            text += f"，空闲 KV 保留 {self.retain_s:g} s"
        arrival = (f"闭环并发 {self.concurrency}" if self.arrival == "closed"
                   else f"泊松到达 {self.rate:g}/s（最多 {self.max_batch} 个同时解码）" if self.arrival == "poisson"
                   else f"脚本 {len(self.script)} 个请求")
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
    """The share of one HBF stack: concurrency / stacks (rounded, >= 1) or rate / stacks."""
    if stacks == 1:
        return workload
    if workload.arrival == "closed":
        return replace(workload, concurrency=max(1, round(workload.concurrency / stacks)))
    if workload.arrival == "poisson":
        return replace(workload, rate=workload.rate / stacks)
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
        w = workload
        if w.prefix_groups:
            weights = 1.0 / np.arange(1, w.prefix_groups + 1) ** w.zipf
            self.prefix_p = weights / weights.sum()
        if w.arrival == "script":
            for item in w.script:
                self._push(Request(self._id(), float(item.get("arrival", 0.0)), int(item["input"]),
                                   int(item["output"]), prefix=int(item.get("prefix", -1)),
                                   retain_s=float(item.get("retain_s", w.retain_s))))
        elif w.arrival == "poisson":
            self.next_poisson = float(self.rng_arr.exponential(1 / w.rate))
        elif w.sessions:
            for i in range(w.concurrency):
                self._new_session(0.0, (i * w.calls) // w.concurrency if stagger else 0)

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
        out = []
        while self.heap and self.heap[0][0] <= now + 1e-12:
            out.append(heapq.heappop(self.heap)[2])
        return out

    def next_time(self):
        """Time of the next timed arrival (inf when none is scheduled)."""
        times = [self.heap[0][0]] if self.heap else []
        if self.next_poisson is not None:
            times.append(self.next_poisson)
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
    ``retain_s`` and ``prefix`` (a shared-prefix group; then give prefix_groups and prefix_tokens)."""
    script = tuple(dict(item) for item in requests)
    if not script:
        raise ValueError("scripted() needs at least one request")
    return Workload(**({"name": "script", "arrival": "script", "script": script} | overrides))
