"""Construction of the three ECMS-based strategies from the tuned parameters."""
import json
from pathlib import Path

import numpy as np

from excavator import params as p
from ems.ecms import ECMSCore, run_fixed_ref, run_policy
from ems.ddpg import DDPG

RESULTS = Path(__file__).resolve().parents[1] / "results"
TUNING_FILE = RESULTS / "tuning.json"
ACTOR_FILE = RESULTS / "ddpg_actor.npz"

# observation scaling and action range of the agent
OBS_SCALE = np.array([4000.0, 3000.0, 0.1])
OBS_SHIFT = np.array([0.0, 0.0, p.SOC_A])
SOC_REF_MIN, SOC_REF_MAX = 0.5, 0.7


def normalise(obs):
    return (np.asarray(obs) - OBS_SHIFT) / OBS_SCALE


def action_to_soc_ref(a):
    return SOC_REF_MIN + (a + 1.0) * 0.5 * (SOC_REF_MAX - SOC_REF_MIN)


def load_tuning():
    return json.loads(TUNING_FILE.read_text())


def ecms_core(tuning):
    return ECMSCore(tuning["ecms"]["eta_ice"], 0.0)


def aecms_core(tuning):
    return ECMSCore(tuning["aecms"]["eta_ice"], tuning["aecms"]["kp"])


def rl_core(tuning):
    """RL-ECMS uses the same s0 and kp as A-ECMS; only SoC_ref differs."""
    return aecms_core(tuning)


def load_agent():
    agent = DDPG(obs_dim=3)
    agent.load_actor(dict(np.load(ACTOR_FILE)))
    return agent


def rl_policy(agent):
    return lambda obs: action_to_soc_ref(agent.act(normalise(obs)))


def run_all(scenario, tuning, agent):
    return {
        "ECMS": run_fixed_ref(scenario, ecms_core(tuning)),
        "A-ECMS": run_fixed_ref(scenario, aecms_core(tuning), soc_ref=p.SOC_A),
        "RL-ECMS": run_policy(scenario, rl_core(tuning), rl_policy(agent)),
    }
