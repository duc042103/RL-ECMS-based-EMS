"""Grid search of ECMS (s0) and A-ECMS (s0, kp) on the 600 kg tuning cycle."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from excavator import params as p, scenario as S          # noqa: E402
from ems.ecms import ECMSCore, run_fixed_ref               # noqa: E402
from ems import metrics                                    # noqa: E402
from ems.strategies import TUNING_FILE                     # noqa: E402

ETA_GRID = np.round(np.arange(0.20, 0.5001, 0.0025), 4)
KP_GRID = [1, 2, 3, 5, 10, 20, 40]
SOC_TOL = 0.02     # charge-sustaining requirement on the tuning cycle: |SoC_end - SoC_0| <= SOC_TOL


def select(rows, e_idx, soc_idx):
    """Lowest energy among charge-sustaining settings; else the one closest to sustaining."""
    ok = [r for r in rows if abs(r[soc_idx] - p.SOC_0) <= SOC_TOL]
    if ok:
        return min(ok, key=lambda r: r[e_idx])
    return min(rows, key=lambda r: (round(abs(r[soc_idx] - p.SOC_0), 4), r[e_idx]))


def main():
    sc = S.constant_scenario(p.LOAD_TUNE)
    s_bar = metrics.soc_value_factor(sc)
    print(f"tuning cycle {p.LOAD_TUNE:.0f} kg, s_bar = {s_bar:.3f}")

    def energy(core):
        log = run_fixed_ref(sc, core, soc_ref=p.SOC_A)
        return metrics.total_energy(log, sc, s_bar), float(log["soc"][-1])

    ecms = [(eta, *energy(ECMSCore(eta, 0.0))) for eta in ETA_GRID]
    e_best = select(ecms, 1, 2)
    print(f"ECMS   eta_ice = {e_best[0]:.4f}  E = {e_best[1] / 1e3:.2f} kJ  SoC_end = {e_best[2]:.3f}")

    aecms = [(eta, kp, *energy(ECMSCore(eta, kp))) for kp in KP_GRID for eta in ETA_GRID]
    a_best = select(aecms, 2, 3)
    print(f"A-ECMS eta_ice = {a_best[0]:.4f}  kp = {a_best[1]}  E = {a_best[2] / 1e3:.2f} kJ  "
          f"SoC_end = {a_best[3]:.3f}")

    out = dict(
        tuning_load=p.LOAD_TUNE, s_bar=s_bar, soc_tol=SOC_TOL,
        ecms=dict(eta_ice=float(e_best[0]), E_kJ=e_best[1] / 1e3, soc_end=e_best[2]),
        aecms=dict(eta_ice=float(a_best[0]), kp=float(a_best[1]), E_kJ=a_best[2] / 1e3,
                   soc_end=a_best[3], soc_ref=p.SOC_A),
        grid_ecms=[[float(a), b / 1e3, c] for a, b, c in ecms],
        grid_aecms=[[float(a), float(k), b / 1e3, c] for a, k, b, c in aecms],
    )
    TUNING_FILE.parent.mkdir(exist_ok=True)
    TUNING_FILE.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
