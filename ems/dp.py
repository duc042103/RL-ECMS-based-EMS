"""Dynamic programming benchmark (full knowledge of the load sequence).

State: SoC on a uniform grid. Control: the same P_ice candidate set as ECMS.
Cost: fuel mass. Terminal condition: SoC(T) = SoC_0 (charge sustaining).
"""
import numpy as np

from excavator import params as p
from excavator import components as c
from excavator.powertrain import candidates
from ems.ecms import LOG_KEYS

N_SOC = 801
TERMINAL_PENALTY = 1.0e3      # kg of fuel per unit of terminal SoC error


def _interp(v, soc_next, grid):
    d = grid[1] - grid[0]
    x = (soc_next - grid[0]) / d
    i = np.clip(np.floor(x).astype(int), 0, len(grid) - 2)
    f = np.clip(x - i, 0.0, 1.0)
    return v[i] * (1 - f) + v[i + 1] * f


def solve(scenario, soc0=p.SOC_0, soc_final=None):
    soc_final = soc0 if soc_final is None else soc_final
    cand = candidates(scenario.p_dem)
    grid = np.linspace(p.SOC_MIN, p.SOC_MAX, N_SOC)
    T = scenario.n_steps
    voc = p.ocv(grid)[:, None]

    V = np.empty((T + 1, N_SOC))
    V[T] = TERMINAL_PENALTY * np.abs(grid - soc_final)
    for t in range(T - 1, -1, -1):
        p_bat = cand["p_bat"][t][None, :]
        disc = voc ** 2 - 4 * p.BAT_R * p_bat
        i = (voc - np.sqrt(np.maximum(disc, 0))) / (2 * p.BAT_R)
        soc_next = grid[:, None] - i * p.DT / (p.BAT_Q_AH * 3600.0)
        feas = (cand["ok"][t][None, :] & (disc >= 0) & (np.abs(i) <= p.BAT_I_MAX)
                & (soc_next >= p.SOC_MIN) & (soc_next <= p.SOC_MAX))
        cost = cand["mdot"][t][None, :] * p.DT + _interp(V[t + 1], soc_next, grid)
        V[t] = np.where(feas, cost, np.inf).min(axis=1)
        V[t] = np.where(np.isfinite(V[t]), V[t], 1e9)

    # forward pass with exact SoC (re-optimised against V[t+1])
    log = {k: np.zeros(T) for k in LOG_KEYS}
    log.update(soc=np.zeros(T + 1), u=np.zeros(T, int))
    soc = soc0
    log["soc"][0] = soc
    for t in range(T):
        i, okb = c.bat_current(cand["p_bat"][t], soc)
        soc_next = c.soc_step(soc, i)
        feas = cand["ok"][t] & okb & (soc_next >= p.SOC_MIN) & (soc_next <= p.SOC_MAX)
        cost = cand["mdot"][t] * p.DT + _interp(V[t + 1], soc_next, grid)
        u = int(np.argmin(np.where(feas, cost, np.inf)))
        for k in LOG_KEYS:
            log[k][t] = cand[k][t][u]
        log["u"][t] = u
        soc = float(soc_next[u])
        log["soc"][t + 1] = soc
    return log
