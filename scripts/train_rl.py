"""Train the DDPG agent of RL-ECMS on patterned loads (same load for 8-12 cycles, levels in [400, 800] kg).

State  x_t = [P_dem, P_dem_avg, SoC]                                  (Eq. 5)
Action a_t = SoC_ref(t)
Reward r_t = -[Q_lhv m_ice dt + lambda (SoC(t+1) - SoC_a)^2]           (Eq. 6)
The best actor is selected on validation scenarios (seeds differ from training and test).

For the critic, the engine-only fuel energy of the current demand (an exogenous signal,
independent of the agent's actions) is added back to the reward as a baseline. This only
shifts Q by a policy-independent term, so the optimal policy of Eq. (6) is unchanged,
but it removes most of the uncontrollable variance from the critic target.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from excavator import params as p, scenario as S          # noqa: E402
from ems import dp, metrics                                # noqa: E402
from ems.ddpg import DDPG                                  # noqa: E402
from ems.ecms import ExcavatorEnv                         # noqa: E402
from excavator.powertrain import candidates               # noqa: E402
from ems.strategies import (ACTOR_FILE, RESULTS, action_to_soc_ref, load_tuning,  # noqa: E402
                            normalise, rl_core)

LAMBDA = 5.0e5            # J, SoC penalty weight in Eq. (6)
GAMMA = 0.995             # discount per 1 s agent step (horizon ~200 s, longer than one load block)
REWARD_SCALE = 1.0e-4     # reward normalisation for the critic
EPISODES = 400
CYCLES_PER_EPISODE = p.N_TRAIN_CYCLES
UPDATE_EVERY = 2          # agent steps per gradient update
CRITIC_WARMUP = 10_000    # agent steps with critic-only updates (actor frozen at its initial policy)
NOISE_START, NOISE_END = 0.15, 0.02
VAL_SEEDS = (101, 102, 103)
VAL_EVERY = 20
VAL_CACHE = RESULTS / "val_cache"     # one JSON file per validation scenario: [s_bar, E_DP]


def validation_set(seeds=VAL_SEEDS):
    """Validation scenarios (same structure as the test); DP references are cached on disk."""
    VAL_CACHE.mkdir(exist_ok=True)
    out = []
    for seed in seeds:
        sc = S.make_scenario(S.patterned_loads(p.N_TEST_CYCLES, np.random.default_rng(seed)))
        f = VAL_CACHE / f"{seed}_{p.N_TEST_CYCLES}_{sc.loads.sum():.3f}.json"
        if not f.exists():
            s_bar = metrics.soc_value_factor(sc)
            f.write_text(json.dumps([s_bar, metrics.total_energy(dp.solve(sc), sc, s_bar)]))
        out.append((sc, *json.loads(f.read_text())))
    return out


def validate(agent, envs, val):
    """Mean ratio to DP over the validation scenarios (environments are built once and reused)."""
    ratios = []
    for env, (sc, s_bar, e_dp) in zip(envs, val):
        obs = env.reset()
        while not env.done:
            obs, _, _ = env.step(action_to_soc_ref(agent.act(normalise(obs))))
        ratios.append(e_dp / metrics.total_energy(env.log, sc, s_bar))
    return float(np.mean(ratios))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, default=LAMBDA)
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gamma", type=float, default=GAMMA)
    ap.add_argument("--out", default=str(ACTOR_FILE))
    ap.add_argument("--episodes", type=int, default=EPISODES)
    ap.add_argument("--prepare-val", type=int, nargs="*", help="only compute the DP cache of these seeds")
    args = ap.parse_args()
    if args.prepare_val is not None:
        validation_set(args.prepare_val or VAL_SEEDS)
        return
    actor_file = Path(args.out)
    tuning = load_tuning()
    core = rl_core(tuning)
    agent = DDPG(obs_dim=3, seed=args.seed, gamma=args.gamma, lr_actor=3e-5, lr_critic=3e-4, batch=256)
    rng = np.random.default_rng(p.TRAIN_SEED)
    val = validation_set()
    val_envs = [ExcavatorEnv(sc, core) for sc, _, _ in val]
    best = validate(agent, val_envs, val)
    print(f"initial actor (SoC_ref = {action_to_soc_ref(0.0):.2f}, i.e. A-ECMS): val ratio {best * 100:.3f} %")
    np.savez(actor_file, **agent.state_dict())

    rows, steps, t0 = [], 0, time.time()
    log_file = actor_file.with_name(actor_file.stem + "_log.csv")
    for ep in range(args.episodes):
        sc = S.make_scenario(S.patterned_loads(CYCLES_PER_EPISODE, rng))
        env = ExcavatorEnv(sc, core)
        base = np.zeros(sc.n_steps) if args.no_baseline else \
            candidates(sc.p_dem)["mdot"][:, 1] * p.DT * p.Q_LHV       # engine-only fuel energy
        obs = env.reset(rng.uniform(0.55, 0.65))
        s = normalise(obs)
        noise = NOISE_START + (NOISE_END - NOISE_START) * ep / (args.episodes - 1)
        ret, losses = 0.0, []
        while not env.done:
            a = agent.act(s, noise)
            t_prev = env.t
            obs2, fuel, done = env.step(action_to_soc_ref(a))
            r = -(fuel + args.lam * (env.soc - p.SOC_A) ** 2 - base[t_prev:env.t].sum()) * REWARD_SCALE
            s2 = normalise(obs2)
            agent.buf.add(s, a, r, s2, 0.0)        # time-limit end: bootstrap (not terminal)
            s, ret, steps = s2, ret + r, steps + 1
            if steps >= agent.batch and steps % UPDATE_EVERY == 0:
                losses.append(agent.update(train_actor=steps >= CRITIC_WARMUP))
        row = dict(episode=ep + 1, ret=ret, soc_end=env.soc, noise=noise,
                   critic_loss=float(np.mean(losses)) if losses else np.nan, val_ratio=np.nan)
        if (ep + 1) % VAL_EVERY == 0:
            vr = validate(agent, val_envs, val)
            row["val_ratio"] = vr
            if vr > best:
                best = vr
                np.savez(actor_file, **agent.state_dict())
            print(f"ep {ep + 1:4d}  return {ret:8.2f}  SoC_end {env.soc:.3f}  "
                  f"val {vr * 100:.3f} %  best {best * 100:.3f} %  ({time.time() - t0:.0f} s)")
        rows.append(row)
        if (ep + 1) % VAL_EVERY == 0 or ep + 1 == args.episodes:
            with open(log_file, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    print(f"best validation ratio to DP: {best * 100:.3f} %  -> {actor_file.name}")


if __name__ == "__main__":
    main()
