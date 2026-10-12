"""Working-cycle generator: boom lift / hold / lower under a continuously varying load.

One cycle = hold at bottom (digging) -> lift (0.13 m/s) -> hold at top (dumping)
-> lower (0.1 m/s, no regeneration). Speeds are the same in every cycle; only the
load mass changes. The load is periodic with strong cycle-to-cycle variation:
The load m(t) changes continuously in time, also within a cycle (periodic trend of 3-5 cycles
plus smooth random variation with periods of 2-15 s, softly saturated into 400-800 kg). The
load stays on the boom during the whole cycle. The test scenario repeats one 50-cycle set
4 times (200 cycles).
"""
from dataclasses import dataclass

import numpy as np

from . import params as p

PHASE_HOLD_BOTTOM, PHASE_LIFT, PHASE_HOLD_TOP, PHASE_LOWER = 0, 1, 2, 3


def _trapezoid(n_steps, v):
    """Cylinder speed and acceleration (step midpoints): constant speed with raised-cosine ramps."""
    t = (np.arange(n_steps) + 0.5) * p.DT
    tau = np.clip(np.minimum(t, n_steps * p.DT - t) / p.T_RAMP, 0.0, 1.0)
    vel = v * 0.5 * (1.0 - np.cos(np.pi * tau))
    acc = v * np.pi / (2 * p.T_RAMP) * np.sin(np.pi * tau) * np.where(t < n_steps * p.DT / 2, 1.0, -1.0)
    return vel, acc


N_LIFT = int(round((p.STROKE / p.V_UP + p.T_RAMP) / p.DT))
N_LOWER = int(round((p.STROKE / p.V_DOWN + p.T_RAMP) / p.DT))
N_HOLD_TOP = int(round(p.T_HOLD_TOP / p.DT))
N_HOLD_BOTTOM = int(round(p.T_HOLD_BOTTOM / p.DT))
N_CYCLE = N_HOLD_BOTTOM + N_LIFT + N_HOLD_TOP + N_LOWER


def cycle_demand(load):
    """Power demand at the power-coupler output (W) over one cycle, and the phase index.

    `load` is the load mass (kg), either one value for the cycle or one value per time step.
    """
    load = np.broadcast_to(np.asarray(load, dtype=float), (N_CYCLE,))
    to_coupler = 1.0 / (p.ETA_CVT * p.ETA_CPL)
    i1, i2, i3 = N_HOLD_BOTTOM, N_HOLD_BOTTOM + N_LIFT, N_HOLD_BOTTOM + N_LIFT + N_HOLD_TOP
    demand = np.empty(N_CYCLE)

    # holds: load kept on the boom; the pump makes up the leakage at the holding pressure
    hold = np.r_[0:i1, i2:i3]
    p_holding = (p.M_STRUCT + p.R_LOAD * load[hold]) * p.G / p.A_CAP
    demand[hold] = (p_holding * p.Q_LEAK_HOLD / p.ETA_PUMP + p.P_PUMP_NOLOAD) * to_coupler

    # lift: cap side fed by the pump, rod side to tank
    v, a = _trapezoid(N_LIFT, p.V_UP)
    force = (p.M_STRUCT + p.R_LOAD * load[i1:i2]) * (p.G + a) + p.F_FRIC
    p_cap = (force + p.P_TANK * p.A_ROD) / p.A_CAP + p.DP_VALVE
    demand[i1:i2] = (p_cap * p.A_CAP * v / p.ETA_PUMP + p.P_PUMP_NOLOAD) * to_coupler

    # lower: gravity-driven, meter-out throttled (no regeneration); pump only fills the rod side
    v, _ = _trapezoid(N_LOWER, p.V_DOWN)
    demand[i3:] = (p.P_ROD_LOWER * p.A_ROD * v / p.ETA_PUMP + p.P_PUMP_NOLOAD) * to_coupler

    phase = np.concatenate([
        np.full(N_HOLD_BOTTOM, PHASE_HOLD_BOTTOM), np.full(N_LIFT, PHASE_LIFT),
        np.full(N_HOLD_TOP, PHASE_HOLD_TOP), np.full(N_LOWER, PHASE_LOWER)])
    return demand, phase


@dataclass
class Scenario:
    loads: np.ndarray       # kg, mean load of each cycle
    p_dem: np.ndarray       # W, per time step
    cycle: np.ndarray       # cycle index per time step
    phase: np.ndarray       # phase index per time step
    p_dem_avg: np.ndarray   # W, moving average of P_dem over the previous cycle length
    load_t: np.ndarray = None   # kg, load per time step

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


def make_scenario(loads, per_step=False):
    """Scenario from one load per cycle, or (per_step=True) one load per time step."""
    loads = np.asarray(loads, dtype=float)
    load_t = loads if per_step else np.repeat(loads, N_CYCLE)
    per_cycle = load_t.reshape(-1, N_CYCLE)
    parts = [cycle_demand(m) for m in per_cycle]
    p_dem = np.concatenate([d for d, _ in parts])
    phase = np.concatenate([ph for _, ph in parts])
    cycle = np.repeat(np.arange(len(per_cycle)), N_CYCLE)
    return Scenario(per_cycle.mean(axis=1), p_dem, cycle, phase, moving_average(p_dem, N_CYCLE), load_t)


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


def repeated_set_loads(n_cycles, rng):
    """One random, strongly varying set of LOAD_SET_CYCLES cycles, repeated up to n_cycles."""
    return np.resize(periodic_loads(p.LOAD_SET_CYCLES, rng), n_cycles)


def continuous_loads(n_cycles, rng):
    """Load per time step: a periodic 50-cycle set (trend + smooth noise, soft-saturated), repeated."""
    n_set = p.LOAD_SET_CYCLES * N_CYCLE
    t = np.arange(n_set) * p.DT
    T = n_set * p.DT
    n_per = int(rng.integers(p.LOAD_SET_CYCLES // p.LOAD_PERIOD_MAX, p.LOAD_SET_CYCLES // p.LOAD_PERIOD_MIN + 1))
    trend = p.LOAD_TREND_AMPL * np.sin(2 * np.pi * n_per * t / T + rng.uniform(0, 2 * np.pi))
    j = np.arange(int(np.ceil(T / p.LOAD_NOISE_TMAX)), int(T / p.LOAD_NOISE_TMIN) + 1)   # harmonics of T
    amp, ph = rng.standard_normal(len(j)), rng.uniform(0, 2 * np.pi, len(j))
    noise = np.zeros(n_set)
    for a_j, j_j, ph_j in zip(amp, j, ph):
        noise += a_j * np.sin(2 * np.pi * j_j * t / T + ph_j)
    noise *= p.LOAD_NOISE_STD / noise.std()
    half = (p.LOAD_MAX - p.LOAD_MIN) / 2
    m = p.LOAD_MEAN + half * np.tanh((trend + noise) / half)
    return np.resize(m, n_cycles * N_CYCLE)


def pattern_scenario(n_cycles, rng):
    """Scenario with the current load pattern (used for training, validation and test)."""
    return make_scenario(continuous_loads(n_cycles, rng), per_step=True)


def test_scenario():
    return pattern_scenario(p.N_TEST_CYCLES, np.random.default_rng(p.TEST_SEED))


def constant_scenario(load, n_cycles=p.N_TUNE_CYCLES):
    return make_scenario(np.full(n_cycles, load))
