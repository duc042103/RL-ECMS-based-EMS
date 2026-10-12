"""Run ECMS, A-ECMS, RL-ECMS and DP on the test scenario; write Table 1 and per-cycle results."""
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from excavator import params as p, scenario as S                      # noqa: E402
from ems import dp, metrics                                            # noqa: E402
from ems.strategies import RESULTS, load_agent, load_tuning, run_all   # noqa: E402

NAMES = ("ECMS", "A-ECMS", "RL-ECMS", "DP")


def main():
    sc = S.test_scenario()
    s_bar = metrics.soc_value_factor(sc)
    logs = run_all(sc, load_tuning(), load_agent())
    logs["DP"] = dp.solve(sc)
    np.savez_compressed(RESULTS / "test_logs.npz",
                        **{f"{n}__{k}": v for n, lg in logs.items() for k, v in lg.items()},
                        loads=sc.loads, p_dem=sc.p_dem, cycle=sc.cycle, phase=sc.phase)

    ce = {n: metrics.cycle_energy(logs[n], sc, s_bar) for n in NAMES}
    E = {n: float(ce[n]["E"].sum()) for n in NAMES}
    eps = {n: (ce[n]["E"] - ce["DP"]["E"]) / ce["DP"]["E"] * 100 for n in NAMES}

    summary = dict(
        test_loads_kg=sc.loads.round(1).tolist(), s_bar=s_bar,
        E_tot_kJ={n: E[n] / 1e3 for n in NAMES},
        fuel_g={n: float(ce[n]["fuel_kg"].sum() * 1e3) for n in NAMES},
        soc_end={n: float(logs[n]["soc"][-1]) for n in NAMES},
        reduction_by_rl_pct={n: (E[n] - E["RL-ECMS"]) / E[n] * 100 for n in ("ECMS", "A-ECMS")},
        ratio_to_dp_pct={n: E["DP"] / E[n] * 100 for n in NAMES},
        max_eps_pct={n: float(np.max(eps[n])) for n in NAMES[:3]},
        mean_abs_eps_pct={n: float(np.mean(np.abs(eps[n]))) for n in NAMES[:3]},
    )
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=1))

    with open(RESULTS / "per_cycle.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cycle", "load_kg"] + [f"E_{n}_kJ" for n in NAMES] + [f"eps_{n}_pct" for n in NAMES[:3]]
                   + [f"SoC_end_{n}" for n in NAMES])
        for k in range(len(sc.loads)):
            w.writerow([k + 1, round(sc.loads[k], 1)] + [round(ce[n]["E"][k] / 1e3, 3) for n in NAMES]
                       + [round(eps[n][k], 3) for n in NAMES[:3]] + [round(ce[n]["soc_end"][k], 4) for n in NAMES])

    r, q = summary["reduction_by_rl_pct"], summary["ratio_to_dp_pct"]
    table = [
        "| Criteria | ECMS | A-ECMS [2] | RL-ECMS | DP |",
        "|---|---|---|---|---|",
        "| Total consumed energy (kJ) | " + " | ".join(f"{E[n] / 1e3:.1f}" for n in NAMES) + " |",
        f"| Reduction by RL-ECMS (%) | {r['ECMS']:.2f} | {r['A-ECMS']:.2f} | – | – |",
        "| Ratio to DP (%) | " + " | ".join(f"{q[n]:.2f}" for n in NAMES[:3]) + " | 100 |",
    ]
    (RESULTS / "table1.md").write_text("\n".join(table) + "\n")
    print("\n".join(table))
    print(json.dumps({k: summary[k] for k in ("fuel_g", "soc_end", "max_eps_pct", "mean_abs_eps_pct")}, indent=1))


if __name__ == "__main__":
    main()
