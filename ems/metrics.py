"""Energy metrics.

E_k   = Q_lhv m_fuel,k + s_bar dSoC_k C_bat V_oc   (dSoC_k = SoC at cycle start - SoC at cycle end)
E_tot = sum_k E_k
eps_k = (E_k - E_k^DP) / E_k^DP * 100 %

s_bar converts battery energy into the equivalent fuel energy. It is the marginal fuel
cost of battery energy obtained from DP on the same scenario (dE_fuel^DP / dSoC_final),
so a strategy cannot lower E_tot by simply draining the battery.
"""
import numpy as np

from excavator import params as p

C_BAT_VOC = p.BAT_Q_AH * 3600.0 * p.BAT_V_NOM      # J of battery energy per unit SoC
DSOC_PROBE = 0.02


def soc_value_factor(scenario):
    """s_bar from DP: central difference of the optimal fuel energy w.r.t. the final SoC."""
    from ems import dp
    f = [dp.solve(scenario, soc_final=p.SOC_0 + d)["mdot"].sum() * p.DT * p.Q_LHV
         for d in (-DSOC_PROBE, DSOC_PROBE)]
    return (f[1] - f[0]) / (2 * DSOC_PROBE * C_BAT_VOC)


def cycle_energy(log, scenario, s_bar):
    n = len(scenario.loads)
    fuel = np.bincount(scenario.cycle, weights=log["mdot"] * p.DT, minlength=n)   # kg
    idx = np.searchsorted(scenario.cycle, np.arange(n))
    soc_start = log["soc"][idx]
    soc_end = log["soc"][np.append(idx[1:], scenario.n_steps)]
    e_fuel = p.Q_LHV * fuel
    e_bat = s_bar * (soc_start - soc_end) * C_BAT_VOC
    return dict(E=e_fuel + e_bat, E_fuel=e_fuel, E_bat=e_bat, fuel_kg=fuel,
                soc_start=soc_start, soc_end=soc_end)


def total_energy(log, scenario, s_bar):
    return float(cycle_energy(log, scenario, s_bar)["E"].sum())
