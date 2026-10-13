# Section 3 – Results and Discussion (numbers from the simulation)

Source: `results/summary.json`, `results/table1.md`, `results/per_cycle.csv`.
Test scenario: 200 working cycles (~47 min), seed 2026. The load varies continuously in time,
also within a cycle (periodic trend of 3–5 cycles plus smooth random variation with periods of
2–15 s, 407–795 kg); one 50-cycle set is repeated 4 times. Preview:
`results/figures/load_profile_test.png`.

## Setup paragraph

> All components of the hybrid excavator are modeled in Python. The test scenario consists of 200 working cycles (about 47 min) in which the load varies continuously between 400 and 800 kg, also during lifting, while the lifting and lowering speeds are kept constant. ECMS and A-ECMS are tuned at 600 kg. The consumed energy includes the fuel energy and the equivalent energy of the SoC change, converted with the marginal fuel cost of battery energy obtained from DP.

## Table 1. Energy consumption over 200 working cycles under varying loads

| Criteria | ECMS | A-ECMS [2] | RL-ECMS | DP |
|---|---|---|---|---|
| Total consumed energy (MJ) | 21.16 | 21.04 | 20.99 | 20.89 |
| Reduction by RL-ECMS (%) | 0.80 | 0.26 | – | – |
| Ratio to DP (%) | 98.73 | 99.27 | 99.53 | 100 |

## Discussion (2 paragraphs; same text as `captions_and_paragraphs.md`)

> Table 1 compares the total consumed energy, which includes the equivalent energy of the SoC change. ECMS burns the least fuel only because it depletes the battery from 0.60 to its lower limit, while A-ECMS settles at a SoC of 0.58, below its reference. RL-ECMS keeps the battery charge-sustaining and achieves the lowest consumption among the online strategies.
>
> Figs. 3 and 4 explain these differences. Once ECMS depletes the battery at cycle 36, the EMG can no longer assist lifting, so its deviation from DP grows with the load and the engine alone covers a wide speed range. A-ECMS and RL-ECMS instead charge the battery near the efficient region of the engine during holding and lowering and use this energy for lifting. The advantage of RL-ECMS comes mainly from light-load cycles, where the equivalence factor of A-ECMS tuned at 600 kg restricts EMG assistance: the agent lowers the SoC reference so that the EMG supplies 1.00 kW on average during lifting, close to the 0.93 kW of DP and compared with 0.15 kW for A-ECMS. This reduces the mean deviation from DP from 0.75 % to 0.49 % and moves the operating points of both the ICE and the EMG toward higher efficiency. The negative deviations in the first cycles arise because DP, knowing the whole load sequence, charges the battery in advance.

## Values for the placeholders

| Placeholder | Value |
|---|---|
| c (RL-ECMS max per-cycle deviation from DP) | 1.47 % (A-ECMS 1.37 %, ECMS 3.43 %) – not used in the text, A-ECMS has the lower maximum |
| x (reduction vs ECMS) | 0.80 % |
| y (reduction vs A-ECMS) | 0.26 % |
| z (RL-ECMS ratio to DP) | 99.53 % |

## Other sentences of the paper that must follow the new scenario

* Section 2: "The agent is trained with the load mass randomized in [400, 800] kg at each episode." → "The agent is trained on load profiles that vary continuously between 400 and 800 kg."
* Abstract: "over a working cycle with randomly varying loads" → "over 200 working cycles with continuously varying loads".

## Checks behind the claims

* Mean deviation from DP: ECMS 1.22 %, A-ECMS 0.75 %, RL-ECMS 0.49 %. Maximum: 3.43 / 1.37 / 1.47 %.
* Light-load cycles (< 550 kg): mean deviation A-ECMS 0.80 %, RL-ECMS 0.18 %. Heavy-load cycles (> 650 kg): A-ECMS 0.68 %, RL-ECMS 0.70 %. The gain of RL-ECMS comes from the light-load cycles.
* Lifting under light loads (< 500 kg): mean SoC_ref 0.606, s = 2.92 (A-ECMS 3.22); mean P_emg RL 1.00 kW, A-ECMS 0.15 kW, DP 0.93 kW. Hold and lowering phases: charging RL 0.67 kW, A-ECMS 0.42 kW, DP 0.75 kW.
* The gain is not a constant SoC_ref offset: A-ECMS with any fixed SoC_ref in 0.60–0.63 gives 99.27 % of DP on the test.
* After cycle 36 the ECMS deviation correlates with the cycle load (r = 0.92).
* Final SoC: ECMS 0.400 (limit), A-ECMS 0.580, RL-ECMS 0.608, DP 0.600.
* DDPG selection (validation, 3 scenarios of 200 cycles): lambda 5e5 seed 1 99.51 %, lambda 5e5 seed 0 99.38 %, lambda 1e6 99.37 %; A-ECMS 99.27 %. Seed-to-seed spread ~0.13 percentage point.
