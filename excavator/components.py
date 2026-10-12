"""Component models: diesel engine (Willans line), EMG (loss map), battery (Rint)."""
import numpy as np

from . import params as p


def rpm2rad(n):
    return np.asarray(n) * np.pi / 30.0


# ------------------------------------------------------------------ engine
def eng_fmep(n):
    c0, c1, c2 = p.ENG_FMEP
    x = np.asarray(n) / 1000.0
    return c0 + c1 * x + c2 * x ** 2


def eng_bmep_max(n):
    a0, a1 = p.ENG_BMEP_MAX
    return a0 - a1 * ((np.asarray(n) - 2000.0) / 1000.0) ** 2


def eng_torque_max(n):
    return eng_bmep_max(n) * p.ENG_VD / (4 * np.pi)


def eng_power_max(n):
    return eng_torque_max(n) * rpm2rad(n)


def eng_fuel_rate(torque, n):
    """Fuel mass flow (kg/s) from the Willans line: P_fuel = (P_b + P_fr) / eta_ind."""
    p_b = np.asarray(torque) * rpm2rad(n)
    p_fr = eng_fmep(n) * p.ENG_VD * np.asarray(n) / 120.0
    return (p_b + p_fr) / (p.ENG_ETA_IND * p.Q_LHV)


def eng_efficiency(torque, n):
    mdot = eng_fuel_rate(torque, n)
    return np.asarray(torque) * rpm2rad(n) / (mdot * p.Q_LHV)


def eng_ool(power):
    """Optimal operating line: lowest feasible speed (min. friction) for a brake power.

    Returns (speed_rpm, torque_Nm, fuel_kg_s); NaN speed where the power is infeasible.
    """
    power = np.atleast_1d(np.asarray(power, dtype=float))
    n_grid = np.linspace(p.ENG_N_MIN, p.ENG_N_MAX, 1801)
    p_max = eng_power_max(n_grid)            # monotonic increasing on the speed range
    n = np.interp(power, p_max, n_grid, left=p.ENG_N_MIN, right=np.nan)
    n = np.where(power <= p_max[-1] + 1e-9, n, np.nan)
    torque = power / rpm2rad(n)
    mdot = eng_fuel_rate(torque, n)
    return n, torque, mdot


P_ICE_MAX = float(eng_power_max(p.ENG_N_MAX))


# ------------------------------------------------------------------ EMG
def emg_torque_max(n):
    w = np.maximum(rpm2rad(n), 1e-6)
    t = np.minimum(p.EMG_T_MAX, p.EMG_P_MAX / w)
    return np.where(np.asarray(n) <= p.EMG_N_MAX, t, 0.0)


def emg_loss(torque, n):
    kc, ki, kw, c = p.EMG_LOSS
    w = rpm2rad(n)
    return kc * np.asarray(torque) ** 2 + ki * w + kw * w ** 3 + c


def emg_elec_power(p_mech, n):
    """Electrical power (W, + = drawn from battery) for a mechanical power at speed n."""
    t = np.asarray(p_mech) / rpm2rad(n)
    return np.asarray(p_mech) + emg_loss(t, n)


def emg_efficiency(torque, n):
    pm = np.asarray(torque) * rpm2rad(n)
    pe = pm + emg_loss(torque, n)
    return np.where(pm >= 0, pm / np.maximum(pe, 1e-9), pe / np.minimum(pm, -1e-9))


def emg_best_speed(p_mech):
    """Most efficient feasible speed for electric-only drive (engine declutched)."""
    p_mech = np.atleast_1d(np.asarray(p_mech, dtype=float))
    n_grid = np.linspace(p.EMG_N_MIN, p.EMG_N_MAX, 541)
    t = p_mech[:, None] / rpm2rad(n_grid)[None, :]
    feas = np.abs(t) <= emg_torque_max(n_grid)[None, :] + 1e-9
    loss = np.where(feas, emg_loss(t, n_grid[None, :]), np.inf)
    k = np.argmin(loss, axis=1)
    n = n_grid[k]
    ok = np.isfinite(loss[np.arange(len(k)), k])
    return np.where(ok, n, np.nan)


# ------------------------------------------------------------------ battery
def bat_current(p_bat, soc):
    """Rint model: P_bat = V_oc I - R I^2. Returns (current A, feasible mask)."""
    voc = p.ocv(soc)
    disc = voc ** 2 - 4 * p.BAT_R * p_bat
    ok = disc >= 0
    i = (voc - np.sqrt(np.maximum(disc, 0.0))) / (2 * p.BAT_R)
    ok &= np.abs(i) <= p.BAT_I_MAX
    return i, ok


def soc_step(soc, current, dt=p.DT):
    return soc - current * dt / (p.BAT_Q_AH * 3600.0)
