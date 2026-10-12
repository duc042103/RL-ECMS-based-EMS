"""Plot the test load sequence and the power demand of the first cycles (load_profile_test.png)."""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from excavator import params as p, scenario as S
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8,
                     "svg.fonttype": "none", "savefig.dpi": 300})
CM = 1 / 2.54
rng = np.random.default_rng(p.TEST_SEED)
period = int(rng.integers(p.LOAD_PERIOD_MIN, p.LOAD_PERIOD_MAX + 1)); phase = rng.uniform(0, 2 * np.pi)
sc = S.test_scenario(); k = np.arange(1, len(sc.loads) + 1); n = len(k)
trend = p.LOAD_MEAN + p.LOAD_AMPL * np.sin(2 * np.pi * (k - 1) / period + phase)
fig, (a0, a2, a1) = plt.subplots(3, 1, figsize=(17 * CM, 13 * CM), gridspec_kw=dict(hspace=0.8))
a0.bar(k, sc.loads, width=0.8, color="#8f8e86")
for r in range(1, p.LOAD_SET_REPEATS):
    a0.axvline(r * p.LOAD_SET_CYCLES + 0.5, color="#2a78d6", lw=0.9)
for r in range(p.LOAD_SET_REPEATS):
    a0.text((r + 0.5) * p.LOAD_SET_CYCLES + 0.5, 860, f"set {r + 1}", ha="center", fontsize=7, color="#2a78d6")
a0.axhline(600, color="#52514e", lw=0.6, ls=(0, (2, 2)))
a0.set_ylim(300, 900); a0.set_xlim(0.3, k[-1] + 0.7); a0.set_xticks(np.arange(0, n + 1, 25))
a0.set_xlabel("Working cycle $k$"); a0.set_ylabel("Load (kg)")
a0.set_title(f"(a) Test load: a random {p.LOAD_SET_CYCLES}-cycle set (600 + {p.LOAD_AMPL:.0f} sin(2πk/{period} + φ) ± {p.LOAD_NOISE:.0f} kg) "
              f"repeated {p.LOAD_SET_REPEATS} times = {n} cycles", fontsize=8, pad=12)
lift = sc.phase == S.PHASE_LIFT
peak = np.array([sc.p_dem[(sc.cycle == c) & lift].max() for c in range(n)]) / 1e3
a2.plot(k, peak, color="#eb6834", lw=0.9, marker="o", ms=1.8)
for r in range(1, p.LOAD_SET_REPEATS):
    a2.axvline(r * p.LOAD_SET_CYCLES + 0.5, color="#2a78d6", lw=0.9)
a2.set_xlim(0.3, k[-1] + 0.7); a2.set_xticks(np.arange(0, n + 1, 25)); a2.set_ylim(3, 8)
a2.set_xlabel("Working cycle $k$"); a2.set_ylabel("Peak $P_{dem}$ (kW)")
a2.set_title("(b) Peak power demand while lifting, every cycle", fontsize=8)
m = sc.cycle < 10
a1.plot(sc.time[m], sc.p_dem[m] / 1e3, color="#eb6834", lw=0.9)
for c in range(10):
    t0 = c * S.N_CYCLE * p.DT
    a1.text(t0 + 5.5, 7.6, f"{sc.loads[c]:.0f} kg", ha="center", fontsize=6.5, color="#52514e")
a1.set_ylim(0, 8.2); a1.set_xlim(0, sc.time[m][-1])
a1.set_xlabel("Time (s)"); a1.set_ylabel("$P_{dem}$ (kW)")
a1.set_title("(c) Power demand at the coupler over time, first 10 cycles (lift 0.13 m/s, lower 0.10 m/s)", fontsize=8)
for ax in (a0, a1): ax.spines[["top", "right"]].set_visible(False)
fig.savefig(str(Path(__file__).resolve().parents[1] / "results" / "figures" / "load_profile_test.png"), bbox_inches="tight")
d = np.abs(np.diff(sc.loads))
print(f"period {period}, min {sc.loads.min():.0f}, max {sc.loads.max():.0f}, mean {sc.loads.mean():.0f} kg, "
      f"mean |change| {d.mean():.0f} kg, max change {d.max():.0f} kg")
