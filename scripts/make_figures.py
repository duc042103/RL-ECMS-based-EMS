"""Fig. 3 (per-cycle deviation from DP) and Fig. 4 (operating points on efficiency maps).

Both figures are written as PNG (600 dpi) and SVG (editable text) to results/figures/.
"""
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from excavator import params as p, components as c                # noqa: E402
from excavator.powertrain import MODE_ENG, MODE_HYB, MODE_ELEC     # noqa: E402
from ems.strategies import RESULTS                                 # noqa: E402

FIG = RESULTS / "figures"
STRATS = ("ECMS", "A-ECMS", "RL-ECMS")
COLOR = {"ECMS": "#1baf7a", "A-ECMS": "#eb6834", "RL-ECMS": "#2a78d6"}
MARKER = {"ECMS": "s", "A-ECMS": "^", "RL-ECMS": "o"}
LINESTYLE = {"ECMS": (0, (4, 2)), "A-ECMS": (0, (6, 2, 1.5, 2)), "RL-ECMS": "-"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
CM = 1 / 2.54
FIG4_ASPECT = 1125 / 491      # width / height of the slot reserved for Fig. 4 in the paper

plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8,
    "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.edgecolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK, "text.color": INK,
    "svg.fonttype": "none", "savefig.dpi": 600,
})


def save(fig, name, tight=True):
    """PNG + SVG. tight=False keeps the exact figure size (fixed aspect ratio for the paper layout)."""
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        if tight:
            fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02)
        else:
            fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)


def load_logs():
    z = np.load(RESULTS / "test_logs.npz")
    logs = {}
    for key in z.files:
        if "__" in key:
            n, k = key.split("__")
            logs.setdefault(n, {})[k] = z[key]
    return logs, z


def fig3():
    with open(RESULTS / "per_cycle.csv") as f:
        rows = list(csv.DictReader(f))
    k = np.array([int(r["cycle"]) for r in rows])
    load = np.array([float(r["load_kg"]) for r in rows])
    long = len(k) > 20                                  # long scenario: full-width figure, no markers
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=((17.0 if long else 8.5) * CM, 6.6 * CM), sharex=True,
                                   gridspec_kw=dict(height_ratios=[1, 2.6], hspace=0.08))
    ax0.bar(k, load, width=0.8 if long else 0.62, color="#c3c2b7", edgecolor="none")
    for r in range(1, len(k) // p.LOAD_SET_CYCLES if len(k) % p.LOAD_SET_CYCLES == 0 else 0):
        for ax in (ax0, ax1):
            ax.axvline(r * p.LOAD_SET_CYCLES + 0.5, color=GRID, lw=0.8, zorder=0)
    ax0.axhline(p.LOAD_TUNE, color=INK2, lw=0.6, ls=(0, (2, 2)))
    ax0.text(k[-1] + 0.8, p.LOAD_TUNE, "tuning\n600 kg", va="center", ha="left", fontsize=6.5, color=INK2)
    ax0.set_ylim(0, 900)
    ax0.set_yticks([0, 400, 800])
    ax0.set_ylabel("Load (kg)")

    for n in STRATS:
        eps = np.array([float(r[f"eps_{n}_pct"]) for r in rows])
        style = dict(lw=1.0) if long else dict(lw=1.2, marker=MARKER[n], ms=4, mfc="white", mew=1.0)
        ax1.plot(k, eps, color=COLOR[n], ls=LINESTYLE[n], label=n,
                 zorder=3 if n == "RL-ECMS" else 2, **style)
    ax1.axhline(0, color=INK2, lw=0.6)
    ax1.text(k[-1] + 0.5, 0.03, "DP", va="bottom", ha="right", fontsize=6.5, color=INK2)
    top = max(float(r[f"eps_{n}_pct"]) for r in rows for n in STRATS)
    ax1.set_ylim(min(-0.3, ax1.get_ylim()[0]), top * 1.25)
    ax1.set_xlabel("Working cycle $k$")
    ax1.set_ylabel(r"$\varepsilon_k$ (%)")
    ax1.set_xticks(np.arange(0, k[-1] + 1, 25 if len(k) > 100 else 10) if long else k)
    ax1.set_xlim(0.4, k[-1] + 0.6)
    ax1.grid(axis="y", color=GRID, lw=0.5)
    ax1.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.02),
               handlelength=2.6, columnspacing=1.0)
    for ax in (ax0, ax1):
        ax.spines[["top", "right"]].set_visible(False)
    save(fig, "Fig3_per_cycle_deviation")


def bubbles(ax, x, y, w, color, marker, rx, ry):
    """Time-weighted operating points: identical (rounded) points merged, area ~ time share."""
    key = np.stack([np.round(x / rx) * rx, np.round(y / ry) * ry], axis=1)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    share = np.bincount(inv.ravel(), weights=w) / w.sum()
    ax.scatter(uniq[:, 0], uniq[:, 1], s=4 + 260 * share, facecolor=color, edgecolor="white",
               linewidth=0.4, alpha=0.85, marker=marker, zorder=3)


def ice_map(ax):
    n = np.linspace(p.ENG_N_MIN, p.ENG_N_MAX, 200)
    t = np.linspace(0.5, 32, 200)
    N, T = np.meshgrid(n, t)
    eta = np.where(T <= c.eng_torque_max(N), c.eng_efficiency(T, N) * 100, np.nan)
    cs = ax.contour(N, T, eta, levels=[15, 20, 24, 27, 29, 30], colors="#9a998f", linewidths=0.5)
    ax.clabel(cs, fmt="%g", fontsize=5.5, inline_spacing=1)
    ax.plot(n, c.eng_torque_max(n), color=INK2, lw=0.9)
    ax.set_xlim(1100, 3100)
    ax.set_ylim(0, 33)
    ax.set_yticks([0, 10, 20, 30])


def emg_map(ax):
    n = np.linspace(300, p.EMG_N_MAX, 200)
    t = np.linspace(-17, 17, 240)
    N, T = np.meshgrid(n, t)
    eta = np.where(np.abs(T) <= c.emg_torque_max(N), c.emg_efficiency(T, N) * 100, np.nan)
    eta = np.where(np.abs(T) < 0.3, np.nan, eta)
    cs = ax.contour(N, T, eta, levels=[70, 80, 85, 88, 90, 92], colors="#9a998f", linewidths=0.5)
    ax.clabel(cs, fmt="%g", fontsize=5.5, inline_spacing=1)
    ax.plot(n, c.emg_torque_max(n), color=INK2, lw=0.9)
    ax.plot(n, -c.emg_torque_max(n), color=INK2, lw=0.9)
    ax.axhline(0, color=GRID, lw=0.5)
    ax.set_xlim(0, 6200)
    ax.set_ylim(-23, 18)
    ax.set_yticks([-20, -10, 0, 10])


def fig4():
    logs, _ = load_logs()
    width = 17.0 * CM
    fig, axes = plt.subplots(2, 3, figsize=(width, width / FIG4_ASPECT), sharex="row", sharey="row")
    fig.subplots_adjust(left=0.065, right=0.985, top=0.915, bottom=0.185, hspace=0.86, wspace=0.2)
    for j, name in enumerate(STRATS):
        lg = logs[name]
        mode = lg["mode"].astype(int)
        # (a) engine: coupled operation only (declutched idle excluded)
        on = (mode == MODE_ENG) | (mode == MODE_HYB)
        ax = axes[0, j]
        ice_map(ax)
        pe = lg["t_e"][on] * c.rpm2rad(lg["n_e"][on])
        fuel = lg["mdot"][on]
        bubbles(ax, lg["n_e"][on], lg["t_e"][on], np.ones(on.sum()), COLOR[name], MARKER[name], 25, 0.4)
        eta_mean = pe.sum() / (fuel.sum() * p.Q_LHV) * 100
        ax.text(0.97, 0.05, rf"$\bar\eta_{{ice}}$ = {eta_mean:.1f} %", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7, bbox=dict(fc="white", ec="none", pad=1))
        ax.set_title(name, color=INK)
        # (b) EMG: engaged operation only
        eng = ((mode == MODE_ELEC) | (mode == MODE_HYB)) & (np.abs(lg["p_emg"]) > 1.0)
        ax = axes[1, j]
        emg_map(ax)
        bubbles(ax, lg["n_m"][eng], lg["t_m"][eng], np.ones(eng.sum()), COLOR[name], MARKER[name], 50, 0.3)
        pm, pb = lg["p_emg"][eng], lg["p_bat"][eng]
        mot, gen = pm > 0, pm < 0
        eta_m = f"{pm[mot].sum() / pb[mot].sum() * 100:.1f} %" if mot.any() else "–"
        eta_g = f"{pb[gen].sum() / pm[gen].sum() * 100:.1f} %" if gen.any() else "–"
        ax.text(0.97, 0.02, rf"$\bar\eta_{{mot}}$ = {eta_m},  $\bar\eta_{{gen}}$ = {eta_g}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5,
                bbox=dict(fc="white", ec="none", pad=1))
    axes[0, 0].set_ylabel("ICE torque (Nm)")
    axes[1, 0].set_ylabel("EMG torque (Nm)")
    for j in range(3):
        axes[0, j].set_xlabel("ICE speed (rpm)")
        axes[1, j].set_xlabel("EMG speed (rpm)")
    for row, label in ((0, "(a) ICE"), (1, "(b) EMG")):
        axes[row, 1].text(0.5, -0.47, label, transform=axes[row, 1].transAxes, ha="center", va="top",
                          fontsize=8)
    for ax in axes.ravel():
        ax.spines[["top", "right"]].set_visible(False)
    save(fig, "Fig4_operating_points", tight=False)


if __name__ == "__main__":
    fig3()
    fig4()
