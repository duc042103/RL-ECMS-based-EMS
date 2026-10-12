# RL-ECMS for a hybrid excavator under varying loads

Python-only simulation for the conference paper *"Reinforcement Learning-Based Adaptive
Energy Management of Hybrid Hydraulic Excavators under Varying Loads"*. A DDPG agent
generates the SoC reference of an ECMS (not hierarchical); ECMS, A-ECMS and DP are the
baselines.

## Run

```bash
pip install -r requirements.txt
python run_all.py            # tune -> train -> evaluate -> figures (~15 min on a laptop CPU)
```

Individual steps: `scripts/tune_baselines.py`, `scripts/train_rl.py`, `scripts/evaluate.py`,
`scripts/make_figures.py`. DDPG is implemented in numpy (`ems/ddpg.py`), no deep-learning
framework is needed.

## Model (all components in Python)

| Block | Model | File |
|---|---|---|
| Diesel engine (~7.5 kW) | Willans line, FMEP(N), BMEP_max(N); optimal operating line through the CVT; keeps running at idle when declutched | `excavator/components.py` |
| EMG (4 kW PMSM + inverter) | loss map: copper + iron + windage + constant | `excavator/components.py` |
| Battery (48 V, 3 Ah) | Rint model, OCV(SoC) | `excavator/components.py` |
| Power coupler, CVT, pump (30 cc/rev) | constant efficiencies, pump no-load loss | `excavator/scenario.py` |
| Boom cylinder (50/28 mm) | force = (M_struct + R m_L)(g + a) + friction, valve drop, no regeneration | `excavator/scenario.py` |
| Clutches | candidates: EMG drive (engine idling), engine only, hybrid | `excavator/powertrain.py` |

Every parameter is in `excavator/params.py`, tagged **GIVEN** (from earlier work) or
**ASSUMED** (chosen for this study).

## Scenario and metric

* Test: 12 cycles (seed 2026), load per cycle U(400, 800) kg, lift 0.13 m/s, lower 0.10 m/s.
* ECMS and A-ECMS tuned on 600 kg cycles (charge-sustaining requirement |dSoC| <= 0.02);
  RL trained on random loads (other seeds); nothing is retuned for the test.
* `E_k = Q_lhv m_fuel,k + s_bar dSoC_k C_bat V_oc`; `s_bar` is the marginal fuel cost of
  battery energy obtained from DP on the same scenario (see `ems/metrics.py`).

## Outputs (`results/`)

`tuning.json`, `ddpg_actor.npz`, `summary.json`, `table1.md`, `per_cycle.csv`,
`figures/Fig3_per_cycle_deviation.{png,svg}`, `figures/Fig4_operating_points.{png,svg}`.
