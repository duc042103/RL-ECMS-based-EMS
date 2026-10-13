# Captions and main paragraphs (paper style)

Numbers: continuous-load test, 200 working cycles (`results/summary.json`, `results/per_cycle.csv`).

---

**Fig. 1. Powertrain configuration of the hybrid excavator.**

The studied hybrid excavator is shown in Fig. 1. A 7.5 kW diesel engine and a 4 kW EMG are connected through clutches to a power coupler, which drives a 30 cc/rev variable-displacement pump through a CVT, and the pump supplies the boom cylinder (bore 50 mm, rod 28 mm). The EMG is powered by a 48 V, 3 Ah lithium-ion battery. The clutches allow EMG-only, engine-only and hybrid operation, while the CVT lets the engine operate on its optimal operating line. All components are modeled in Python.

---

**Fig. 2. Framework of the proposed RL-ECMS.**

Fig. 2 illustrates the proposed RL-ECMS framework. At each time step, the DDPG actor receives the state $x_t$, consisting of the power demand, its moving average and the SoC, and outputs the SoC reference $SoC_{ref}(t)$. This reference updates the equivalence factor through Eq. (4), and the ECMS solves Eqs. (2)–(3) to determine the ICE and EMG power commands. The resulting fuel consumption and SoC form the reward in Eq. (6), which the critic uses to evaluate and improve the actor. The agent is trained on load profiles that vary continuously between 400 and 800 kg and shares $s_0$ and $k_p$ with A-ECMS, so it only adjusts the SoC reference while the power split remains determined by the ECMS.

---

**Table 1. Comparison of energy consumption over 200 working cycles under continuously varying loads.**

| Criteria | ECMS | A-ECMS [2] | RL-ECMS | DP |
|---|---|---|---|---|
| Total consumed energy (MJ) | 21.16 | 21.04 | 20.99 | 20.89 |
| Reduction by RL-ECMS (%) | 0.80 | 0.26 | – | – |
| Ratio to DP (%) | 98.73 | 99.27 | 99.53 | 100 |

Table 1 summarizes the total energy consumption, which includes the fuel energy and the equivalent energy of the SoC change. ECMS burns the least fuel only because it depletes the battery from 0.60 to 0.40; once this deficit is included, it has the highest consumption of 21.16 MJ. A-ECMS sustains the SoC near 0.58 and consumes 21.04 MJ. RL-ECMS consumes 20.99 MJ, which is 0.80 % and 0.26 % less than ECMS and A-ECMS and corresponds to 99.53 % of the DP optimum.

---

**Fig. 3. Per-cycle energy deviation from DP over 200 working cycles: mean load of each cycle (top) and deviation $\varepsilon_k$ of ECMS, A-ECMS and RL-ECMS (bottom).**

Fig. 3 shows the energy deviation of each cycle from DP. The negative values in the first cycles occur because DP, knowing the whole load sequence, charges the battery during these cycles. With a fixed equivalence factor, ECMS depletes the battery to its lower limit at cycle 36; afterwards, its deviation follows the load and reaches 2.3 % on average and up to 3.4 % in heavy-load cycles, where the EMG can no longer assist lifting. A-ECMS sustains the SoC with a mean deviation of 0.75 %, whereas RL-ECMS deviates less than A-ECMS in 144 of the 200 cycles, with a mean deviation of 0.49 %. The improvement is concentrated in light-load cycles (0.18 % versus 0.80 %), where the equivalence factor of A-ECMS tuned at 600 kg restricts EMG assistance.

---

**Fig. 4. Operating points of ECMS, A-ECMS and RL-ECMS on the efficiency maps of (a) the ICE and (b) the EMG. The marker area is proportional to the time spent at each point.**

Fig. 4 explains these differences. ECMS never charges the battery, so during holding and lowering the engine runs alone at low torque, where its efficiency is below 15 %, and once the battery is depleted it supplies heavy lifts alone at speeds up to 2300 rpm. A-ECMS and RL-ECMS instead use the low-demand phases to charge the battery near the efficient region of the engine at 1200 rpm and use this energy to assist lifting. RL-ECMS does so more effectively: when lifting light loads, the agent lowers the SoC reference so that the EMG supplies 1.00 kW on average, close to the 0.93 kW of DP and compared with 0.15 kW for A-ECMS. As a result, the mean ICE efficiency increases from 23.8 % (ECMS) and 25.2 % (A-ECMS) to 25.7 %, and the mean EMG motoring efficiency from 88.7 % and 88.2 % to 90.6 %.
