"""Working-cycle generator: boom lift / hold / lower with a patterned random load.

One cycle = hold at bottom (digging) -> lift (0.13 m/s) -> hold at top (dumping)
-> lower (0.1 m/s, no regeneration). Speeds are the same in every cycle; only the
load mass changes. The load is periodic with strong cycle-to-cycle variation:
m_k = 600 + 200 sin(2 pi k / P + phi) + U(-100, 100) kg, P = 3-5 cycles, clipped to [400, 800] kg.
"""
from dataclasses import dataclass

import numpy as np

from . import params as p

PHASE_HOLD_BOTTOM, PHASE_LIFT, PHASE_HOLD_TOP, PHASE_LOWER = 0, 1, 2, 3


def _trapezoid(n_steps, v):
    """Cylinder speed and acceleration (step midpoints) for a trapezoidal profile."""
    t = (np.arange(n_steps) + 0.5) * p.DT
    t_end = n_steps * p.DT
    vel = v * np.clip(np.minimum(t, t_end - t) / p.T_RAMP, 0.0, 1.0)
    acc = np.where(t < p.T_RAMP, v / p.T_RAMP, np.where(t > t_end - p.T_RAMP, -v / p.T_RAMP, 0.0))
    return vel, acc


N_LIFT = int(round((p.STROKE / p.V_UP + p.T_RAMP) / p.DT))
N_LOWER = int(round((p.STROKE / p.V_DOWN + p.T_RAMP) / p.DT))
N_HOLD_TOP = int(round(p.T_HOLD_TOP / p.DT))
N_HOLD_BOTTOM = int(round(p.T_HOLD_BOTTOM / p.DT))
N_CYCLE = N_HOLD_BOTTOM + N_LIFT + N_HOLD_TOP + N_LOWER


def cycle_demand(load):
    """Power demand at the power-coupler output (W) over one cycle, and the phase index."""
    to_coupler = 1.0 / (p.ETA_CVT * p.ETA_CPL)

    # holds: pump at standby pressure / leakage flow
    p_hold = (p.P_STANDBY * p.Q_STANDBY / p.ETA_PUMP + p.P_PUMP_NOLOAD) * to_coupler

    # lift: cap side fed by the pump, rod side to tank
    v, a = _trapezoid(N_LIFT, p.V_UP)
    force = (p.M_STRUCT + p.R_LOAD * load) * (p.G + a) + p.F_FRIC
    p_cap = (force + p.P_TANK * p.A_ROD) / p.A_CAP + p.DP_VALVE
    p_lift = (p_cap * p.A_CAP * v / p.ETA_PUMP + p.P_PUMP_NOLOAD) * to_coupler

    # lower: gravity-driven, meter-out throttled; pump only fills the rod side
    v, _ = _trapezoid(N_LOWER, p.V_DOWN)
    p_lower = (p.P_ROD_LOWER * p.A_ROD * v / p.ETA_PUMP + p.P_PUMP_NOLOAD) * to_coupler

    demand = np.concatenate([
        np.full(N_HOLD_BOTTOM, p_hold), p_lift, np.full(N_HOLD_TOP, p_hold), p_lower])
    phase = np.concatenate([
        np.full(N_HOLD_BOTTOM, PHASE_HOLD_BOTTOM), np.full(N_LIFT, PHASE_LIFT),
        np.full(N_HOLD_TOP, PHASE_HOLD_TOP), np.full(N_LOWER, PHASE_LOWER)])
    return demand, phase


@dataclass
class Scenario:
    loads: np.ndarray       # kg, one per cycle
    p_dem: np.ndarray       # W, per time step
    cycle: np.ndarray       # cycle index per time step
    phase: np.ndarray       # phase index per time step
    p_dem_avg: np.ndarray   # W, moving average of P_dem over the previous cycle length

    @property
    def n_steps(self):
        return len(self.p_dem)

    @property
    def time(self):
        return np.arange(self.n_steps) * p.DT


def moving_average(x, window):
    """Trailing moving average over `window` samples (expanding at the start)."""
    c = np.cumsum(np.insert(x, 0, 0.0))
    idx = np.arange(1, len(x) + 1)
    lo = np.maximum(idx - window, 0)
    return (c[idx] - c[lo]) / (idx - lo)


def make_scenario(loads):
    loads = np.asarray(loads, dtype=float)
    parts = [cycle_demand(m) for m in loads]
    p_dem = np.concatenate([d for d, _ in parts])
    phase = np.concatenate([ph for _, ph in parts])
    cycle = np.repeat(np.arange(len(loads)), N_CYCLE)
    return Scenario(loads, p_dem, cycle, phase, moving_average(p_dem, N_CYCLE))


def random_loads(n_cycles, rng):
    """Independent load per cycle (used in the first study, kept for comparison)."""
    return rng.uniform(p.LOAD_MIN, p.LOAD_MAX, size=n_cycles)


def block_loads(n_cycles, rng):
    """Block-wise persistent load: level ~ U(400, 800) held for 8-12 consecutive cycles."""
    loads, level = [], None
    while len(loads) < n_cycles:
        new = rng.uniform(p.LOAD_MIN, p.LOAD_MAX)
        while level is not None and abs(new - level) < p.LOAD_MIN_JUMP:
            new = rng.uniform(p.LOAD_MIN, p.LOAD_MAX)
        level = new
        n = int(rng.integers(p.LOAD_BLOCK_MIN, p.LOAD_BLOCK_MAX + 1))
        jitter = 1.0 + p.LOAD_JITTER * rng.uniform(-1.0, 1.0, size=n)
        loads.extend(np.clip(level * jitter, p.LOAD_MIN, p.LOAD_MAX))
    return np.array(loads[:n_cycles])


def periodic_loads(n_cycles, rng):
    """Periodic trend (period 3-5 cycles, random phase) plus strong random variation per cycle."""
    period = int(rng.integers(p.LOAD_PERIOD_MIN, p.LOAD_PERIOD_MAX + 1))
    phase = rng.uniform(0.0, 2 * np.pi)
    k = np.arange(n_cycles)
    m = (p.LOAD_MEAN + p.LOAD_AMPL * np.sin(2 * np.pi * k / period + phase)
         + rng.uniform(-p.LOAD_NOISE, p.LOAD_NOISE, n_cycles))
    return np.clip(m, p.LOAD_MIN, p.LOAD_MAX)


patterned_loads = periodic_loads      # load pattern used for training, validation and test


def test_scenario():
    return make_scenario(patterned_loads(p.N_TEST_CYCLES, np.random.default_rng(p.TEST_SEED)))


def constant_scenario(load, n_cycles=p.N_TUNE_CYCLES):
    return make_scenario(np.full(n_cycles, load))
