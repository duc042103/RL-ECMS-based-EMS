"""Power-split candidates of the hybrid powertrain (engine + EMG -> coupler -> CVT -> pump).

The decision variable of every strategy (ECMS family and DP) is the engine power
P_ice. Candidate set per time step:
  u = 0      EMG drive       (engine clutch open, engine idling)
  u = 1      engine-only     (EMG clutch open, P_ice = P_dem)
  u >= 2     hybrid          (P_ice on a fixed grid, P_emg = P_dem - P_ice)
Engine speed follows its optimal operating line; in hybrid mode the EMG speed is
I_MOTOR times the engine speed, in electric-only mode the EMG runs at its most
efficient speed (the CVT adapts the pump speed in all cases).
"""
import numpy as np

from . import params as p
from . import components as c

P_ICE_GRID = np.linspace(100.0, c.P_ICE_MAX, p.N_PICE - 2)
N_U = p.N_PICE
MODE_ELEC, MODE_ENG, MODE_HYB = 0, 1, 2

_ool_n, _ool_t, _ool_mdot = c.eng_ool(P_ICE_GRID)
IDLE_FUEL = float(c.eng_fuel_rate(0.0, p.ENG_N_IDLE))


def candidates(p_dem):
    """Candidate table for demand array p_dem (T,). All outputs have shape (T, N_U)."""
    p_dem = np.atleast_1d(np.asarray(p_dem, dtype=float))
    T = len(p_dem)
    shape = (T, N_U)
    p_ice = np.zeros(shape)
    n_e = np.full(shape, np.nan)
    t_e = np.zeros(shape)
    mdot = np.zeros(shape)
    n_m = np.full(shape, np.nan)
    ok = np.ones(shape, dtype=bool)

    # u = 0: EMG drive, engine declutched but kept running at idle
    n_m[:, 0] = c.emg_best_speed(p_dem)
    ok[:, 0] = np.isfinite(n_m[:, 0])
    mdot[:, 0] = IDLE_FUEL

    # u = 1: engine-only
    n1, t1, m1 = c.eng_ool(p_dem)
    p_ice[:, 1] = p_dem
    n_e[:, 1], t_e[:, 1], mdot[:, 1] = n1, t1, m1
    ok[:, 1] = np.isfinite(n1)

    # u >= 2: hybrid
    p_ice[:, 2:] = P_ICE_GRID[None, :]
    n_e[:, 2:] = _ool_n[None, :]
    t_e[:, 2:] = _ool_t[None, :]
    mdot[:, 2:] = _ool_mdot[None, :]
    n_m[:, 2:] = p.I_MOTOR * _ool_n[None, :]

    p_emg = p_dem[:, None] - p_ice
    p_emg[:, 1] = 0.0

    w_m = c.rpm2rad(np.nan_to_num(n_m, nan=1.0))
    t_m = np.where(np.isfinite(n_m), p_emg / w_m, 0.0)
    ok[:, 2:] &= np.abs(t_m[:, 2:]) <= c.emg_torque_max(n_m[:, 2:]) + 1e-9

    p_bat = np.where(np.isfinite(n_m), c.emg_elec_power(p_emg, np.nan_to_num(n_m, nan=1.0)), 0.0)
    p_bat[:, 1] = 0.0
    mode = np.empty(shape, dtype=np.int8)
    mode[:, 0], mode[:, 1], mode[:, 2:] = MODE_ELEC, MODE_ENG, MODE_HYB

    return dict(p_ice=p_ice, p_emg=p_emg, p_bat=p_bat, mdot=mdot, n_e=n_e, t_e=t_e,
                n_m=n_m, t_m=t_m, ok=ok, mode=mode)
