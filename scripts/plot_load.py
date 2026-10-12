"""Preview of the test load m(t) and the power demand at the coupler (load_profile_test.png)."""
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from excavator import params as p, scenario as S     # noqa: E402

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8,
                     "svg.fonttype": "none", "savefig.dpi": 300})
CM = 1 / 2.54
INK2 = "#52514e"


def main(n_zoom=10):
    sc = S.test_scenario()
    t_cyc = S.N_CYCLE * p.DT
    k = sc.time / t_cyc                                   # time in working cycles
    fig, (a0, a1, a2) = plt.subplots(3, 1, figsize=(17 * CM, 13 * CM), gridspec_kw=dict(hspace=0.75))

    a0.plot(k, sc.load_t, color="#8f8e86", lw=0.5)
    for r in range(1, p.LOAD_SET_REPEATS):
        a0.axvline(r * p.LOAD_SET_CYCLES, color="#2a78d6", lw=0.9)
    for r in range(p.LOAD_SET_REPEATS):
        a0.text((r + 0.5) * p.LOAD_SET_CYCLES, 850, f"set {r + 1}", ha="center", fontsize=7, color="#2a78d6")
    a0.set_xlim(0, k[-1]); a0.set_ylim(350, 900); a0.set_xticks(np.arange(0, len(sc.loads) + 1, 25))
    a0.set_xlabel("Working cycle"); a0.set_ylabel("Load (kg)")
    a0.set_title(f"(a) Test load m(t): a {p.LOAD_SET_CYCLES}-cycle set repeated {p.LOAD_SET_REPEATS} times "
                 f"= {len(sc.loads)} cycles ({sc.n_steps * p.DT / 60:.0f} min)", fontsize=8, pad=12)

    m = sc.cycle < n_zoom
    a1.plot(sc.time[m], sc.load_t[m], color="#8f8e86", lw=1.0)
    a1.set_xlim(0, sc.time[m][-1]); a1.set_ylim(350, 850)
    a1.set_xlabel("Time (s)"); a1.set_ylabel("Load (kg)")
    a1.set_title(f"(b) Load m(t), first {n_zoom} cycles", fontsize=8)

    a2.plot(sc.time[m], sc.p_dem[m] / 1e3, color="#eb6834", lw=1.0)
    lift = sc.phase == S.PHASE_LIFT
    for c in range(n_zoom):
        sel = (sc.cycle == c) & lift
        a2.text(sc.time[sel].mean(), sc.p_dem[sel].max() / 1e3 + 0.25, f"{sc.load_t[sel].mean():.0f} kg",
                ha="center", fontsize=6.5, color=INK2)
    a2.set_xlim(0, sc.time[m][-1]); a2.set_ylim(0, 8.5)
    a2.set_xlabel("Time (s)"); a2.set_ylabel("$P_{dem}$ (kW)")
    a2.set_title("(c) Power demand at the coupler (labels: mean load while lifting)", fontsize=8)
    for ax in (a0, a1, a2):
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(ROOT / "results" / "figures" / "load_profile_test.png", bbox_inches="tight")

    lifts = np.array([sc.p_dem[(sc.cycle == c) & lift] for c in range(len(sc.loads))])
    swing = lifts.max(axis=1) - lifts[:, lifts.shape[1] // 4: 3 * lifts.shape[1] // 4].min(axis=1)
    dm = np.abs(np.diff(sc.load_t)) / p.DT
    print(f"load {sc.load_t.min():.0f}-{sc.load_t.max():.0f} kg, mean {sc.load_t.mean():.0f} kg, "
          f"mean |dm/dt| {dm.mean():.0f} kg/s; P_dem {sc.p_dem.min() / 1e3:.2f}-{sc.p_dem.max() / 1e3:.2f} kW; "
          f"lift peaks {lifts.max(axis=1).min() / 1e3:.2f}-{lifts.max(axis=1).max() / 1e3:.2f} kW; "
          f"mean variation inside one lift {swing.mean() / 1e3:.2f} kW")


if __name__ == "__main__":
    main()
