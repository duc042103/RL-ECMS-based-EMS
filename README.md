# RL-ECMS for a hybrid excavator under varying loads

Python-only simulation for the conference paper *"Reinforcement Learning-Based Adaptive
Energy Management of Hybrid Hydraulic Excavators under Varying Loads"*. A DDPG agent
generates the SoC reference of an ECMS (not hierarchical); ECMS, A-ECMS and DP are the
baselines.

## Run

```bash
pip install -r requirements.txt
python run_all.py            # tune -> train -> evaluate -> figures (~10 min, single-threaded BLAS)
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

`tuning.json`, `ddpg_actor.npz` (+ `ddpg_actor_log.csv`), `summary.json`, `table1.md`,
`per_cycle.csv`, `test_logs.npz`, `figures/Fig3_per_cycle_deviation.{png,svg}`,
`figures/Fig4_operating_points.{png,svg}`. `results/variants/` keeps the three DDPG
training configurations that were compared on the validation scenarios.

## Results (test scenario)

| Criteria | ECMS | A-ECMS [2] | RL-ECMS | DP |
|---|---|---|---|---|
| Total consumed energy (kJ) | 1197.3 | 1197.2 | 1195.2 | 1188.7 |
| Reduction by RL-ECMS (%) | 0.18 | 0.17 | – | – |
| Ratio to DP (%) | 99.28 | 99.29 | 99.46 | 100 |

Max / mean |eps_k|: ECMS 1.78 / 0.74 %, A-ECMS 1.17 / 0.73 %, RL-ECMS 1.24 / 0.55 %.
Final SoC: ECMS 0.518, A-ECMS 0.580, RL-ECMS 0.583, DP 0.600 (s_bar = 2.91).
Paper text with these numbers: `paper/section3_results.md`.

## Findings that shaped the setup

1. **SoC term of the metric.** With the SoC change counted 1:1 as electrical energy, a
   strategy that drains the battery scores *better than DP* (one early run: ECMS 528 kJ vs
   DP 782 kJ), because 1 J of battery energy costs about 2.9-3.7 J of fuel. The SoC term
   is therefore weighted by `s_bar`, the DP marginal fuel cost of battery energy.
2. **Engine stop/start.** If the diesel may stop whenever it is declutched (every ~13 s
   cycle), every ECMS variant is within 0.2 % of DP and there is nothing to improve. The
   engine is kept running at idle when declutched, which is how excavator diesels work.
3. **Fixed-s ECMS is a knife edge.** With a Willans-line engine the marginal fuel cost at
   idle speed is constant (1/eta_ind = 2.5), so for s_ch slightly above 2.5 the engine charges the
   battery in every low-demand phase (SoC -> 0.8) and slightly below it never does
   (SoC drifts down). No fixed s sustains the SoC at 600 kg; the tuned ECMS ends at 0.535.
4. **Small headroom.** On the test scenario ECMS and A-ECMS already reach ~99.3 % of DP.
   An A-ECMS re-tuned on random loads reaches 99.44 % on the validation scenarios, close to
   RL-ECMS (99.43 % validation, 99.46 % test). The engine efficiency along its optimal line is
   almost flat (29.1-30.7 % for the 3.9-7.1 kW lifting demand of 400-800 kg), and the 3 Ah battery never
   reaches its SoC limits in 12 cycles, so (by Pontryagin's principle) a nearly constant
   equivalence factor is already close to optimal. Load variation therefore changes the
   optimal power split only slightly.
5. **DDPG training.** With the raw Eq. (6) reward the critic is dominated by the
   uncontrollable fuel of the duty cycle and the policy drifts below A-ECMS. Adding the
   engine-only fuel of the current demand back as a baseline (policy-invariant, exogenous)
   gives stable learning. Three configurations were compared on validation scenarios
   (seeds 101-103) and the best (lambda = 5e5, baseline) was used for the test.
