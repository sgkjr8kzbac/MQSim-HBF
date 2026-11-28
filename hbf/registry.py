"""Unified plugin registry for both built-in methods and optional ``hbf.ext.*`` extensions.

Core modules never import ``hbf.ext`` directly. Built-in methods (formats, layouts, synthetic
modes, clocks) and optional extensions under ``hbf/ext/`` register into the exact same registry
tables. Deleting ``hbf/ext/`` (or setting ``HBF_NO_EXT=1``) leaves the core simulator working;
referencing an unregistered extension name raises ``ValueError`` naming the extension that owns it.
"""
import importlib
import os
import pkgutil

# Maps optional extension feature names -> owning extension package for actionable error messages.
KNOWN_EXT_OWNERS = {
    # hbf.ext.system
    "system": "hbf.ext.system",
    "gpu": "hbf.ext.system",
    "tpot": "hbf.ext.system",
    "gpu_roofline": "hbf.ext.system",
    "overlap": "hbf.ext.system",
    "kv_consume": "hbf.ext.system",
    # hbf.ext.kv_terms
    "mla": "hbf.ext.kv_terms",
    "mla_fp8": "hbf.ext.kv_terms",
    "loki": "hbf.ext.kv_terms",
    "sparq": "hbf.ext.kv_terms",
    "channel_select": "hbf.ext.kv_terms",
    "kv_spec": "hbf.ext.kv_terms",
    "kvspec": "hbf.ext.kv_terms",
    "hybrid": "hbf.ext.kv_terms",
    "ssm": "hbf.ext.kv_terms",
    "cla": "hbf.ext.kv_terms",
    "yoco": "hbf.ext.kv_terms",
    "duo": "hbf.ext.kv_terms",
    "pyramid": "hbf.ext.kv_terms",
    "density": "hbf.ext.kv_terms",
    "select_static": "hbf.ext.kv_terms",
    "reuse": "hbf.ext.kv_terms",
    "query_groups": "hbf.ext.kv_terms",
    "full_tensors": "hbf.ext.kv_terms",
    "frontiers": "hbf.ext.kv_terms",
    "prefix_tokens": "hbf.ext.kv_terms",
    "placement_k7": "hbf.ext.kv_terms",
    "selection_k5": "hbf.ext.kv_terms",
    "geometry_kvspec": "hbf.ext.kv_terms",
    "format_resolver": "hbf.ext.kv_terms",
}


class Registry:
    """Single registry shared by built-in plugins and ``hbf.ext.*`` extensions."""

    def __init__(self):
        self.formats = {}          # name -> callable(name, params, head_dim, q_heads) -> KVFormat
        self.format_params = {}    # name -> (defaults_dict, extra_allowed_set)
        self.layouts = {}          # name -> Layout subclass
        self.modes = {}            # name -> generator handler (None means built-in inline in synthetic.py)
        self.clocks = {}           # name -> clock class/factory
        self.evaluators = {}       # name -> evaluator hook object (e.g. "system")
        self.prefill_rates = {}    # name -> callable(model, gpu, tokens) -> tok_s
        self.hooks = {}            # name -> hook object/callable (kvspec, selection, placement, geometry, ...)
        self.loaded_extensions = []
        self._builtins_loaded = False
        self._extensions_loaded = False

    def register_format(self, name, builder, *, defaults=None, extra_params=()):
        self.formats[name] = builder
        self.format_params[name] = (dict(defaults or {}), set(extra_params))

    def register_layout(self, name, cls):
        self.layouts[name] = cls

    def register_mode(self, name, handler=None):
        self.modes[name] = handler

    def register_clock(self, name, factory):
        self.clocks[name] = factory

    def register_evaluator(self, name, evaluator):
        self.evaluators[name] = evaluator

    def register_prefill_rate(self, name, fn):
        self.prefill_rates[name] = fn

    def register_hook(self, name, hook):
        self.hooks[name] = hook

    def missing_error(self, kind, name):
        owner = KNOWN_EXT_OWNERS.get(name)
        if owner:
            return ValueError(
                f"{kind} {name!r} is not registered; it is provided by extension {owner!r} "
                f"(restore {owner.replace('.', '/')} and unset HBF_NO_EXT)"
            )
        return ValueError(f"Unknown {kind} {name!r}")

    def get_format(self, name):
        ensure_loaded()
        if name not in self.formats:
            raise self.missing_error("KV format", name)
        return self.formats[name], self.format_params[name]

    def get_layout(self, name):
        ensure_loaded()
        if name not in self.layouts:
            raise self.missing_error("layout", name)
        return self.layouts[name]

    def get_clock(self, name):
        ensure_loaded()
        if name not in self.clocks:
            raise self.missing_error("clock", name)
        return self.clocks[name]

    def get_evaluator(self, name, *, required=False):
        ensure_loaded()
        ev = self.evaluators.get(name)
        if ev is None and required:
            raise self.missing_error("evaluator", name)
        return ev

    def get_prefill_rate(self, name="gpu_roofline", *, required=False):
        ensure_loaded()
        fn = self.prefill_rates.get(name)
        if fn is None and required:
            raise self.missing_error("prefill rate", name)
        return fn

    def get_hook(self, name, *, required=False):
        ensure_loaded()
        hook = self.hooks.get(name)
        if hook is None and required:
            raise self.missing_error("feature", name)
        return hook


REGISTRY = Registry()


def register_builtins():
    """Register all built-in formats, layouts, synthetic modes, and core clocks."""
    if REGISTRY._builtins_loaded:
        return
    REGISTRY._builtins_loaded = True
    from .frontend.kv.formats import register_builtins as reg_formats
    from .frontend.kv.layouts import LAYOUTS
    from .frontend.trace.synthetic import CORE_MODES
    from .lifetime.clock import ConstantClock, ReadTimeClock

    reg_formats(REGISTRY)
    for name, cls in LAYOUTS.items():
        REGISTRY.register_layout(name, cls)
    for name in CORE_MODES:
        REGISTRY.register_mode(name, None)
    REGISTRY.register_clock("read_time", ReadTimeClock)
    REGISTRY.register_clock("constant", ConstantClock)


def load_extensions():
    """Discover and import every ``hbf.ext.*`` subpackage and invoke its ``register(REGISTRY)``.

    Skips silently ONLY when ``hbf.ext`` itself does not exist (``ModuleNotFoundError`` with
    ``name == 'hbf.ext'``) or when ``HBF_NO_EXT=1``. Any import error inside an existing
    extension propagates normally.
    """
    register_builtins()
    if REGISTRY._extensions_loaded:
        return list(REGISTRY.loaded_extensions)
    REGISTRY._extensions_loaded = True
    if os.environ.get("HBF_NO_EXT") == "1":
        return []
    try:
        ext_pkg = importlib.import_module("hbf.ext")
    except ModuleNotFoundError as exc:
        if exc.name == "hbf.ext":
            return []
        raise
    ext_paths = getattr(ext_pkg, "__path__", None)
    if not ext_paths:
        return []
    for info in sorted(pkgutil.iter_modules(ext_paths), key=lambda m: m.name):
        if info.name.startswith("_"):
            continue
        mod_name = f"hbf.ext.{info.name}"
        mod = importlib.import_module(mod_name)
        reg = getattr(mod, "register", None)
        if callable(reg):
            reg(REGISTRY)
        REGISTRY.loaded_extensions.append(mod_name)
    return list(REGISTRY.loaded_extensions)


def ensure_loaded():
    """Ensure built-ins and any installed ``hbf.ext.*`` extensions are registered."""
    if not REGISTRY._builtins_loaded:
        register_builtins()
    if not REGISTRY._extensions_loaded:
        load_extensions()
    return REGISTRY
