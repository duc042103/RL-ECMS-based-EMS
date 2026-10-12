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
sc = S.test_scenario(); k = np.arange(1, len(sc.loads) + 1)
trend = p.LOAD_MEAN + p.LOAD_AMPL * np.sin(2 * np.pi * (k - 1) / period + phase)
fig, (a0, a1) = plt.subplots(2, 1, figsize=(17 * CM, 9 * CM), gridspec_kw=dict(hspace=0.55))
a0.bar(k, sc.loads, width=0.8, color="#c3c2b7", label="Load per cycle")
kk = np.linspace(1, k[-1], 800)
a0.plot(kk, p.LOAD_MEAN + p.LOAD_AMPL * np.sin(2 * np.pi * (kk - 1) / period + phase), color="#2a78d6", lw=1.1,
        label=f"Periodic trend (period {period} cycles)")
a0.axhline(600, color="#52514e", lw=0.6, ls=(0, (2, 2)))
a0.set_ylim(300, 900); a0.set_xlim(0.3, k[-1] + 0.7); a0.set_xticks(np.arange(10, 81, 10))
a0.set_xlabel("Working cycle $k$"); a0.set_ylabel("Load (kg)")
a0.legend(ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.2))
a0.set_title(f"(a) Test load: 80 cycles, 600 + {p.LOAD_AMPL:.0f} sin(2πk/{period} + φ) ± {p.LOAD_NOISE:.0f} kg, clipped to 400–800 kg", fontsize=8, pad=16)
m = sc.cycle < 10
a1.plot(sc.time[m], sc.p_dem[m] / 1e3, color="#eb6834", lw=0.9)
for c in range(10):
    t0 = c * S.N_CYCLE * p.DT
    a1.text(t0 + 5.5, 7.6, f"{sc.loads[c]:.0f} kg", ha="center", fontsize=6.5, color="#52514e")
a1.set_ylim(0, 8.2); a1.set_xlim(0, sc.time[m][-1])
a1.set_xlabel("Time (s)"); a1.set_ylabel("$P_{dem}$ (kW)")
a1.set_title("(b) Power demand at the coupler, first 10 cycles (lift 0.13 m/s, lower 0.10 m/s)", fontsize=8)
for ax in (a0, a1): ax.spines[["top", "right"]].set_visible(False)
fig.savefig(str(Path(__file__).resolve().parents[1] / "results" / "figures" / "load_profile_test.png"), bbox_inches="tight")
d = np.abs(np.diff(sc.loads))
print(f"period {period}, min {sc.loads.min():.0f}, max {sc.loads.max():.0f}, mean {sc.loads.mean():.0f} kg, "
      f"mean |change| {d.mean():.0f} kg, max change {d.max():.0f} kg")
