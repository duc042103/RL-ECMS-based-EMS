"""Model parameters of the hybrid excavator (boom circuit) used in all simulations.

Values tagged GIVEN come from the earlier work (KSFC 2026 Spring paper, handoff notes).
Values tagged ASSUMED were chosen for this study to be physically consistent with a
small hybrid excavator test platform (~7.5 kW diesel); they are not measured data.
"""
import numpy as np

G = 9.81
Q_LHV = 42.5e6  # J/kg, diesel lower heating value (ASSUMED, typical)

# ---------------------------------------------------------------- simulation
DT = 0.1                      # s, ECMS / plant time step (ASSUMED)
AGENT_DT = 1.0                # s, DDPG decision interval (ASSUMED, as Li et al. control cycle)

# ---------------------------------------------------------------- scenario
V_UP = 0.13                   # m/s, lifting speed (GIVEN)
V_DOWN = 0.10                 # m/s, lowering speed (GIVEN)
STROKE = 0.45                 # m, boom cylinder working stroke per cycle (ASSUMED)
T_RAMP = 0.3                  # s, acceleration / deceleration ramp (ASSUMED)
T_HOLD_TOP = 2.0              # s, hold at top (dumping) (ASSUMED)
T_HOLD_BOTTOM = 3.0           # s, hold at bottom (digging / loading) (ASSUMED)
LOAD_MIN, LOAD_MAX = 400.0, 800.0   # kg, load mass per cycle (GIVEN)
LOAD_TUNE = 600.0             # kg, tuning load of ECMS / A-ECMS (GIVEN)
N_TUNE_CYCLES = 12            # cycles of the constant-load tuning scenario
N_TEST_CYCLES = 80            # cycles in the test scenario, ~18 min of operation (user decision)
N_TRAIN_CYCLES = 30           # cycles per training episode
TEST_SEED = 2026              # seed of the test load sequence (GIVEN, user decision)
TRAIN_SEED = 1                # seed of training load sequences (differs from test)
# block load (second study): the same load for 8-12 consecutive cycles, then a new level
LOAD_BLOCK_MIN, LOAD_BLOCK_MAX = 8, 12  # consecutive cycles with the same load
LOAD_JITTER = 0.0             # -, cycle-to-cycle variation inside a block (same load: none)
LOAD_MIN_JUMP = 150.0         # kg, minimum change of level between blocks
# periodic load (current study, user decision): periodic trend + strong cycle-to-cycle variation
#   m_k = 600 + A sin(2 pi k / P + phi) + U(-noise, noise), clipped to [400, 800] kg
LOAD_MEAN = 600.0             # kg, centre of the periodic trend
LOAD_AMPL = 170.0             # kg, amplitude of the periodic trend (ASSUMED)
LOAD_NOISE = 60.0             # kg, random variation added to every cycle (ASSUMED)
LOAD_PERIOD_MIN, LOAD_PERIOD_MAX = 5, 8   # cycles per load period, drawn per scenario (ASSUMED)

# ---------------------------------------------------------------- boom cylinder / load
BORE = 0.050                  # m (GIVEN)
ROD = 0.028                   # m (GIVEN)
A_CAP = np.pi / 4 * BORE ** 2
A_ROD = np.pi / 4 * (BORE ** 2 - ROD ** 2)
M_STRUCT = 500.0              # kg, boom/arm/bucket mass referred to the cylinder (ASSUMED)
R_LOAD = 4.4                  # -, lever ratio load -> cylinder force (ASSUMED; 13-21.5 MPa over 400-800 kg)
F_FRIC = 500.0                # N, cylinder seal friction (ASSUMED)
P_TANK = 0.3e6                # Pa, return-line back pressure (ASSUMED)
DP_VALVE = 1.0e6              # Pa, control-valve pressure drop when lifting (ASSUMED)
P_ROD_LOWER = 1.5e6           # Pa, rod-side supply pressure when lowering (no regeneration) (ASSUMED)
P_STANDBY = 1.5e6             # Pa, standby pressure when holding (ASSUMED)
Q_STANDBY = 1.0 / 60000       # m^3/s, standby / leakage flow when holding (ASSUMED)

# ---------------------------------------------------------------- pump, CVT, power coupler
PUMP_DISP = 30e-6             # m^3/rev, variable-displacement pump (GIVEN)
PUMP_SPEED = 1500.0           # rpm, pump speed held by the CVT (ASSUMED)
ETA_PUMP = 0.88               # -, pump overall efficiency at load (ASSUMED)
P_PUMP_NOLOAD = 150.0         # W, pump no-load loss at 1500 rpm (ASSUMED)
ETA_CVT = 0.93                # -, CVT efficiency (ASSUMED)
ETA_CPL = 0.97                # -, power coupler efficiency (ASSUMED)
I_MOTOR = 1.5                 # -, coupler ratio motor speed / engine speed (ASSUMED)

# ---------------------------------------------------------------- engine (Willans line)
ENG_VD = 0.479e-3             # m^3, displacement, 2-cyl small diesel (ASSUMED)
ENG_ETA_IND = 0.40            # -, indicated efficiency of the Willans line (ASSUMED)
ENG_FMEP = (1.5e5, 0.4e5, 0.08e5)   # Pa, FMEP = c0 + c1 (N/1000) + c2 (N/1000)^2 (ASSUMED)
ENG_BMEP_MAX = (7.8e5, 1.5e5)        # Pa, BMEPmax = a0 - a1 ((N-2000)/1000)^2 (ASSUMED)
ENG_N_MIN, ENG_N_MAX = 1200.0, 3000.0  # rpm (ASSUMED); rated ~7.5 kW at 3000 rpm (GIVEN)
ENG_N_IDLE = 1200.0           # rpm, engine keeps running at idle when declutched (ASSUMED)

# ---------------------------------------------------------------- EMG (PMSM + inverter)
EMG_P_MAX = 4.0e3             # W, peak mechanical power (ASSUMED)
EMG_T_MAX = 16.0              # Nm, peak torque (ASSUMED)
EMG_N_MAX = 6000.0            # rpm (ASSUMED)
EMG_N_MIN = 600.0             # rpm, lowest speed used in electric-only mode (ASSUMED)
EMG_LOSS = (0.8, 0.30, 1.0e-6, 40.0)  # copper kc*T^2, iron ki*w, windage kw*w^3, constant C (ASSUMED)

# ---------------------------------------------------------------- battery (Rint)
BAT_NS = 13                   # series cells, 48 V class pack (ASSUMED)
BAT_Q_AH = 3.0                # Ah, capacity, high-power cells (ASSUMED; ~27 s of rated engine power usable)
BAT_R = 0.04                  # Ohm, pack internal resistance (ASSUMED)
BAT_I_MAX = 100.0             # A, current limit (ASSUMED)
BAT_V_NOM = BAT_NS * 3.7      # V, nominal open-circuit voltage used in the energy metric
SOC_MIN, SOC_MAX = 0.4, 0.8   # -, SoC window (ASSUMED)
SOC_0 = 0.6                   # -, initial SoC, same for every strategy (ASSUMED)
SOC_A = 0.6                   # -, target SoC in the reward / A-ECMS reference (ASSUMED)

# ---------------------------------------------------------------- ECMS
ETA_MDR = 0.90                # -, mean EMG efficiency, motoring (ASSUMED)
ETA_MGE = 0.90                # -, mean EMG efficiency, generating (ASSUMED)
ETA_D = 0.95                  # -, mean battery discharge efficiency (ASSUMED)
ETA_CH = 0.95                 # -, mean battery charge efficiency (ASSUMED)
N_PICE = 80                   # -, size of the engine-power grid searched by ECMS and DP


def ocv(soc):
    """Pack open-circuit voltage (V) for a 13s NMC pack."""
    cell = 3.4 + 0.75 * soc - 0.3 * soc ** 2 + 0.25 * soc ** 3
    return BAT_NS * cell
