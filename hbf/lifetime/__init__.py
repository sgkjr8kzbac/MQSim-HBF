"""HBF lifetime modelling: geometry, clocks, closed-form curves, event simulation, and CLI."""
from .analytic import analytic
from .clock import ConstantClock, ReadTimeClock, StepClock, StepTime, calibrate, layout_rounds, phase_a
from .geometry import (CAUSES, DAY, F1, GIB, HBF_FLAGS, PARTITIONS, POLICIES, Device, Geometry,
                       erase_amp, geometry_for, region_pages, resolve_prefill_tok_s, sci, threshold_steps)
from .sim import CAUSE, CapacityError, SimOptions, Simulator, VllmTracker, simulate
