# Section 3 – Results and Discussion (numbers from the simulation)

Source: `results/summary.json`, `results/table1.md`, `results/per_cycle.csv`.
Test scenario: 80 cycles (~18 min), seed 2026; the load is held for 8–12 consecutive cycles,
then jumps to a new level: 472 / 787 / 405 / 779 / 476 / 718 / 567 / 405 / 618 kg.

## Setup paragraph

> All components of the hybrid excavator are modeled in Python. The test scenario concatenates 80 working cycles (about 18 min), in which the load is kept constant for 8–12 consecutive cycles and then changed to a new random level between 400 and 800 kg, while the lifting and lowering speeds are kept constant. ECMS and A-ECMS are tuned at 600 kg. The consumed energy includes the fuel energy and the equivalent energy of the SoC change, converted with the marginal fuel cost of battery energy obtained from DP.

## Table 1. Comparison of energy consumption under varying loads

| Criteria | ECMS | A-ECMS [2] | RL-ECMS | DP |
|---|---|---|---|---|
| Total consumed energy (kJ) | 7757.3 | 7745.6 | 7720.3 | 7677.3 |
| Reduction by RL-ECMS (%) | 0.48 | 0.33 | – | – |
| Ratio to DP (%) | 98.97 | 99.12 | 99.44 | 100 |

## Discussion (2 short paragraphs)

> Table 1 and Fig. 3 compare the energy consumption of the strategies over this scenario. With a fixed equivalence factor, ECMS cannot sustain the SoC: the battery reaches its lower limit at cycle 32, after which the EMG can no longer assist heavy lifts and the deviation from DP rises to 2.3–3.4 % in the heavy-load blocks. A-ECMS [2] keeps the SoC near its reference, but its deviation reaches up to 1.40 % per block and is generally larger in the light-load blocks, where the equivalence factor tuned at 600 kg restricts EMG assistance. RL-ECMS deviates less than A-ECMS in 74 of the 80 cycles and stays within 1.24 % of DP, reducing total energy consumption by 0.48 % and 0.33 % compared with ECMS and A-ECMS and reaching 99.44 % of the DP optimum.
>
> Fig. 4 explains this improvement. When lifting light loads, the agent lowers the SoC reference, so the EMG supplies 0.99 kW on average, close to the 1.00 kW of DP, compared with 0.26 kW for A-ECMS, and the battery is recharged in the following hold and lowering phases. RL-ECMS therefore raises the mean ICE efficiency from 22.9 % (ECMS) and 24.5 % (A-ECMS) to 25.1 % and the mean EMG motoring efficiency of A-ECMS from 87.9 % to 90.7 %, while returning the SoC to 0.60 at the end of every load block.

## Values for the placeholders

| Placeholder | Value |
|---|---|
| c (RL-ECMS max per-cycle deviation from DP) | 1.24 % (A-ECMS 1.58 %, ECMS 3.36 %) |
| x (reduction vs ECMS) | 0.48 % |
| y (reduction vs A-ECMS) | 0.33 % |
| z (RL-ECMS ratio to DP) | 99.44 % |

## Other sentences of the paper that must follow the new scenario

* Section 2: "The agent is trained with the load mass randomized in [400, 800] kg at each episode." → "The agent is trained on load sequences in which a random load in [400, 800] kg is held for 8–12 consecutive cycles."
* Abstract: "over a working cycle with randomly varying loads" → "over 80 working cycles with step-wise varying loads" (or similar).

## Checks behind the claims

* The gain is not a constant SoC_ref offset: A-ECMS with any fixed SoC_ref in 0.60–0.63 stays at 99.10–99.15 % of DP (test and three validation scenarios), RL-ECMS reaches 99.42–99.44 % in all four.
* Lifting under light loads (< 550 kg): mean SoC_ref 0.606, s = 3.07 (A-ECMS 3.25); mean P_emg RL 0.99 kW, A-ECMS 0.26 kW, DP 1.00 kW.
* Final SoC: ECMS 0.400 (limit), A-ECMS 0.580, RL-ECMS 0.601, DP 0.600. Because ECMS ends at the SoC limit, the linear SoC correction slightly favours ECMS.
* Remaining gap to DP (0.56 %): DP keeps the SoC near 0.79 for most of the scenario and releases it only in the last blocks, which requires knowing the whole load sequence; it also charges during the hold before lifting, which neither A-ECMS nor RL-ECMS does.
* A-ECMS mean deviation per block: 0.23 / 0.65 / 1.06 / 0.80 / 1.02 / 0.82 / 1.40 / 1.26 / 0.88 % (blocks at 472 / 787 / 405 / 779 / 476 / 718 / 567 / 405 / 618 kg).
