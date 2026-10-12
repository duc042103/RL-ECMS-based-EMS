"""ECMS core shared by ECMS, A-ECMS and RL-ECMS, plus the closed-loop simulator.

Eq. (2): m_tot = m_ice(T_ice, w_ice) + s(t) P_emg / Q_lhv
Eq. (4): s(t) = s0 + kp (SoC_ref(t) - SoC(t)),  s0 = s_d (P_emg >= 0) or s_ch (P_emg < 0)
  ECMS:    kp = 0
  A-ECMS:  SoC_ref fixed
  RL-ECMS: SoC_ref(t) from the DDPG actor
"""
import numpy as np

from excavator import params as p
from excavator import components as c
from excavator.powertrain import candidates, N_U


class ECMSCore:
    def __init__(self, eta_ice, kp=0.0):
        self.eta_ice = float(eta_ice)
        self.kp = float(kp)
        self.s_d = 1.0 / (eta_ice * p.ETA_MDR * p.ETA_D)
        self.s_ch = p.ETA_MGE * p.ETA_CH / eta_ice

    def equivalence(self, soc, soc_ref):
        e = self.kp * (soc_ref - soc)
        return self.s_d + e, self.s_ch + e

    def choose(self, row, soc, soc_ref):
        """Minimise Eq. (2) over the candidate row subject to Eq. (3). Returns (u, soc_next, s)."""
        s_dis, s_chg = self.equivalence(soc, soc_ref)
        s = np.where(row["p_emg"] >= 0, s_dis, s_chg)
        cost = row["mdot"] + s * row["p_emg"] / p.Q_LHV
        i, okb = c.bat_current(row["p_bat"], soc)
        soc_next = c.soc_step(soc, i)
        feas = row["ok"] & okb & (soc_next >= p.SOC_MIN) & (soc_next <= p.SOC_MAX)
        if not feas.any():                                  # SoC at a bound: fall back
            feas = row["ok"] & okb
        u = int(np.argmin(np.where(feas, cost, np.inf)))
        return u, float(soc_next[u]), float(s[u])


LOG_KEYS = ("p_ice", "p_emg", "p_bat", "mdot", "n_e", "t_e", "n_m", "t_m", "mode")


class ExcavatorEnv:
    """Hybrid excavator + ECMS layer. One `step` = one agent interval (AGENT_DT)."""

    def __init__(self, scenario, core):
        self.sc = scenario
        self.core = core
        self.cand = candidates(scenario.p_dem)
        self.sub = int(round(p.AGENT_DT / p.DT))

    def reset(self, soc0=p.SOC_0):
        self.t = 0
        self.soc = float(soc0)
        T = self.sc.n_steps
        self.log = {k: np.zeros(T) for k in LOG_KEYS}
        self.log.update(soc=np.zeros(T + 1), soc_ref=np.zeros(T), s=np.zeros(T), u=np.zeros(T, int))
        self.log["soc"][0] = self.soc
        return self.obs()

    @property
    def done(self):
        return self.t >= self.sc.n_steps

    def obs(self):
        t = min(self.t, self.sc.n_steps - 1)
        return np.array([self.sc.p_dem[t], self.sc.p_dem_avg[t], self.soc])

    def step(self, soc_ref):
        """Apply SoC_ref for one agent interval. Returns (obs, fuel_J, done)."""
        fuel = 0.0
        for _ in range(self.sub):
            if self.done:
                break
            t = self.t
            row = {k: v[t] for k, v in self.cand.items()}
            u, soc_next, s = self.core.choose(row, self.soc, soc_ref)
            for k in LOG_KEYS:
                self.log[k][t] = row[k][u]
            self.log["u"][t], self.log["s"][t], self.log["soc_ref"][t] = u, s, soc_ref
            fuel += row["mdot"][u] * p.DT * p.Q_LHV
            self.soc = soc_next
            self.t += 1
            self.log["soc"][self.t] = self.soc
        return self.obs(), fuel, self.done


def run_fixed_ref(scenario, core, soc_ref=p.SOC_A, soc0=p.SOC_0):
    """ECMS (kp = 0) or A-ECMS (fixed SoC_ref) over a whole scenario."""
    env = ExcavatorEnv(scenario, core)
    env.reset(soc0)
    while not env.done:
        env.step(soc_ref)
    return env.log


def run_policy(scenario, core, policy, soc0=p.SOC_0):
    """RL-ECMS: SoC_ref = policy(obs) at every agent interval."""
    env = ExcavatorEnv(scenario, core)
    obs = env.reset(soc0)
    while not env.done:
        obs, _, _ = env.step(float(policy(obs)))
    return env.log
