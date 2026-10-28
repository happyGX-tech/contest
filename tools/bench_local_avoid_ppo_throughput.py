"""Measure LocalAvoidPPO training throughput (collection vs update) on a real device.

Run from the repo root; no engine required. torch and numpy are seeded from
--seed, so identical arguments reproduce the same numbers.

    python tools/bench_local_avoid_ppo_throughput.py --device cuda:7
"""
import json, sys, time, argparse
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policies.red.skills.evasion_ppo.local_ppo import LocalAvoidPPO, collect_rollout
from policies.red.skills.evasion_ppo.local_env import VectorLocalAvoidEnv

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="models/local_avoid_ppo_v13_gate_aligned_s3_best.pt")
ap.add_argument("--num-envs", type=int, default=256)
ap.add_argument("--rollout-steps", type=int, default=128)
ap.add_argument("--minibatch-size", type=int, default=2048)
ap.add_argument("--ppo-epochs", type=int, default=5)
ap.add_argument("--iters", type=int, default=5)
ap.add_argument("--warmup", type=int, default=2)
ap.add_argument("--device", default="cuda:7")
ap.add_argument("--stage", type=int, default=3)
ap.add_argument("--stage-dist", default="0.0,0.0,0.2,0.8")
ap.add_argument("--seed", type=int, default=701)
ap.add_argument("--lr", type=float, default=1e-5)
args = ap.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

policy, env_cfg, payload = LocalAvoidPPO.load(args.model, device=args.device)
policy.config.num_parallel_envs = args.num_envs
policy.config.rollout_steps = args.rollout_steps
policy.config.minibatch_size = args.minibatch_size
policy.config.ppo_epochs = args.ppo_epochs
policy.config.learning_rate = args.lr
policy.optimizer = torch.optim.Adam(policy.network.parameters(), lr=args.lr)
policy.reward_rms = type(policy.reward_rms)()
policy.discounted_reward = np.zeros(args.num_envs, dtype=np.float64)
policy.update_count = 0
policy.total_steps = 0

dist = tuple(float(x) for x in args.stage_dist.split(","))
env = VectorLocalAvoidEnv(args.num_envs, env_cfg, stage=args.stage, seed=args.seed, stage_distribution=dist)
obs, crit = env.reset()

def one_iter():
    t0 = time.perf_counter()
    rollout, obs_, crit_, completed = collect_rollout(policy, env, obs, crit)
    t1 = time.perf_counter()
    m = policy.update(rollout)
    t2 = time.perf_counter()
    return t1 - t0, t2 - t1, len(completed), rollout["observations"].shape[0], m

for i in range(args.warmup):
    c, u, ep, n, m = one_iter()
    print(f"warmup {i}: collect={c:.2f}s update={u:.2f}s episodes={ep} samples={n}", flush=True)

cs, us, eps = [], [], []
for i in range(args.iters):
    c, u, ep, n, m = one_iter()
    cs.append(c); us.append(u); eps.append(ep)
    print(f"iter {i}: collect={c:.2f}s update={u:.2f}s episodes={ep} samples={n} "
          f"loss={m.get('policy_loss'):.4f} kl={m.get('approx_kl'):.5f}", flush=True)

import statistics as st
tot = st.mean(cs) + st.mean(us)
print(json.dumps({
    "device": args.device, "num_envs": args.num_envs, "rollout_steps": args.rollout_steps,
    "samples_per_update": args.num_envs * args.rollout_steps,
    "collect_s_mean": round(st.mean(cs), 3), "update_s_mean": round(st.mean(us), 3),
    "iter_s_mean": round(tot, 3),
    "env_steps_per_s_collect": round(args.num_envs * args.rollout_steps / st.mean(cs), 1),
    "samples_per_s_total": round(args.num_envs * args.rollout_steps / tot, 1),
    "update_share_pct": round(100 * st.mean(us) / tot, 1),
    "episodes_per_iter": round(st.mean(eps), 1),
    "episodes_per_s": round(st.mean(eps) / tot, 2),
}, indent=2))
