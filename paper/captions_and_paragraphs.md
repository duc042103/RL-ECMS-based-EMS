# Captions and main paragraphs (paper style)

Numbers: continuous-load test, 200 working cycles (`results/summary.json`, `results/per_cycle.csv`).

---

**Fig. 1. Powertrain configuration of the hybrid excavator.**

The studied hybrid excavator is shown in Fig. 1. A 7.5 kW diesel engine and a 4 kW EMG are connected through clutches to a power coupler, which drives a 30 cc/rev variable-displacement pump through a CVT, and the pump supplies the boom cylinder (bore 50 mm, rod 28 mm). The EMG is powered by a 48 V, 3 Ah lithium-ion battery. The clutches allow EMG-only, engine-only and hybrid operation, while the CVT lets the engine operate on its optimal operating line. All components are modeled in Python.

---

**Fig. 2. Framework of the proposed RL-ECMS.**

Fig. 2 illustrates the proposed RL-ECMS framework. At each time step, the DDPG actor receives the state $x_t$, consisting of the power demand, its moving average and the SoC, and outputs the SoC reference $SoC_{ref}(t)$. This reference updates the equivalence factor through Eq. (4), and the ECMS solves Eqs. (2)–(3) to determine the ICE and EMG power commands. The resulting fuel consumption and SoC form the reward in Eq. (6), which the critic uses to evaluate and improve the actor. The agent is trained on load profiles that vary continuously between 400 and 800 kg and shares $s_0$ and $k_p$ with A-ECMS, so it only adjusts the SoC reference while the power split remains determined by the ECMS.

---

## Results (2 paragraphs)

**Table 1. Energy consumption over 200 working cycles under varying loads.**

| Criteria | ECMS | A-ECMS [2] | RL-ECMS | DP |
|---|---|---|---|---|
| Total consumed energy (MJ) | 21.16 | 21.04 | 20.99 | 20.89 |
| Reduction by RL-ECMS (%) | 0.80 | 0.26 | – | – |
| Ratio to DP (%) | 98.73 | 99.27 | 99.53 | 100 |

**Fig. 3. Load of each cycle (top) and per-cycle energy deviation from DP (bottom).**

**Fig. 4. Operating points on the efficiency maps of (a) the ICE and (b) the EMG; marker area is proportional to the time spent.**

> Table 1 compares the total consumed energy, which includes the equivalent energy of the SoC change. ECMS burns the least fuel only because it depletes the battery from 0.60 to its lower limit, while A-ECMS settles at a SoC of 0.58, below its reference. RL-ECMS keeps the battery charge-sustaining and achieves the lowest consumption among the online strategies.
>
> Figs. 3 and 4 explain these differences. Once ECMS depletes the battery at cycle 36, the EMG can no longer assist lifting, so its deviation from DP grows with the load and the engine alone covers a wide speed range. A-ECMS and RL-ECMS instead charge the battery near the efficient region of the engine during holding and lowering and use this energy for lifting. The advantage of RL-ECMS comes mainly from light-load cycles, where the equivalence factor of A-ECMS tuned at 600 kg restricts EMG assistance: the agent lowers the SoC reference so that the EMG supplies 1.00 kW on average during lifting, close to the 0.93 kW of DP and compared with 0.15 kW for A-ECMS. This reduces the mean deviation from DP from 0.75 % to 0.49 % and moves the operating points of both the ICE and the EMG toward higher efficiency. The negative deviations in the first cycles arise because DP, knowing the whole load sequence, charges the battery in advance.
