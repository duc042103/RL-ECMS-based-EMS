# Section 3 – Results and Discussion (numbers from the simulation)

Source: `results/summary.json`, `results/table1.md`, `results/per_cycle.csv`
(test scenario: 12 cycles, seed 2026, loads 472 / 656 / 587 / 548 / 542 / 716 / 762 / 471 / 661 / 519 / 787 / 768 kg).

## Setup paragraph

> All components of the hybrid excavator are modeled in Python. The test scenario concatenates 12 working cycles, in which the load of each cycle is randomly selected between 400 and 800 kg while the lifting and lowering speeds are kept constant. ECMS and A-ECMS are tuned at 600 kg. The consumed energy includes the fuel energy and the equivalent energy of the SoC change, converted with the marginal fuel cost of battery energy obtained from DP.

## Table 1. Comparison of energy consumption under randomly varying loads

| Criteria | ECMS | A-ECMS [2] | RL-ECMS | DP |
|---|---|---|---|---|
| Total consumed energy (kJ) | 1197.3 | 1197.2 | 1195.2 | 1188.7 |
| Reduction by RL-ECMS (%) | 0.18 | 0.17 | – | – |
| Ratio to DP (%) | 99.28 | 99.29 | 99.46 | 100 |

## Discussion (2 short paragraphs, rewritten to match the data)

> Table 1 and Fig. 3 compare the energy consumption of the strategies over this scenario. ECMS cannot sustain the SoC with a fixed equivalence factor, which falls from 0.60 to 0.52, and its deviation from DP fluctuates between 0.26 % and 1.78 %. A-ECMS [2] holds the SoC near its reference through feedback, but its deviation rises to about 1 % in the heavier cycles from cycle 6 onward. RL-ECMS stays within 1.24 % of DP in every cycle and lowers the mean deviation from 0.74 % and 0.73 % to 0.55 %, reducing total energy consumption by 0.18 % and 0.17 % compared with ECMS and A-ECMS and reaching 99.46 % of the DP optimum.
>
> Fig. 4 explains this improvement. RL-ECMS raises the mean ICE efficiency from 22.8 % (ECMS) and 24.7 % (A-ECMS) to 25.0 % and the mean EMG motoring efficiency from 87.9 % and 89.6 % to 90.6 %, as the agent lowers the SoC reference in heavier cycles and raises it in lighter ones, instead of correcting the equivalence factor only after the SoC has deviated, as A-ECMS does [2].

## Values to fill in the handoff placeholders

| Placeholder | Value |
|---|---|
| c (RL-ECMS max per-cycle deviation from DP) | 1.24 % |
| x (reduction vs ECMS) | 0.18 % |
| y (reduction vs A-ECMS) | 0.17 % |
| z (RL-ECMS ratio to DP) | 99.46 % |

## Sentences of the previous draft that the data do NOT support

* "ECMS performs close to DP only in cycles near its 600 kg tuning load, and its deviation grows as the load departs from it": the ECMS deviation does not correlate with |load − 600 kg| (e.g. 1.78 % at 519 kg, 0.46 % at 787 kg).
* "RL-ECMS stays within c % of DP in every cycle" as a distinguishing claim: true for c = 1.24 %, but A-ECMS has a lower maximum (1.17 %); the advantage of RL-ECMS is in the mean deviation, not the maximum.
* "A-ECMS shows peaks after large load changes because it corrects only after the SoC has deviated": A-ECMS rises to ~1 % in cycles 6–12, but RL-ECMS also peaks in cycle 8 (1.24 %), so this causal sentence was removed.
* Supporting evidence for the Fig. 4 sentence: the per-cycle mean SoC_ref of the agent correlates with the cycle load at r = −0.74 (amplitude about ±0.01).
* The margins (0.17–0.18 %) are small; see `README.md` → *Findings* for why.
