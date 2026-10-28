"""Phase-1 reproduction of the V13 training config, saving a checkpoint at EVERY PPO update.

Purpose: produce the intermediate checkpoints needed to calibrate `selection_score`
(the local proxy) against the whole-game score. The stock trainer only keeps `_best.pt`,
which erases exactly the trajectory we need to look at.

Config is derived from `tools/train_local_avoid_ppo_v10_mission_s3.py` so it stays in
lockstep with the real trainer: base_config == v10's base_config (+ --strict-mission),
phase == v10's PHASES[0] "single_recovery".

No engine required. Does not touch core/.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from policies.red.skills.evasion_ppo.local_env import LocalAvoidConfig, VectorLocalAvoidEnv
from policies.red.skills.evasion_ppo.local_ppo import (
    LocalAvoidPPO,
    RunningMeanStd,
    collect_rollout,
    summarize_episodes,
)


def base_config(strict_mission: bool) -> LocalAvoidConfig:
    """v10 base_config, verbatim from tools/train_local_avoid_ppo_v10_mission_s3.py."""
    config = LocalAvoidConfig(
        horizon_steps=900,
        unit_speeds_mps=(200.0, 120.0, 100.0),
        bug_speed_mps=250.0,
        unit_turn_rates_deg_s=(56.1, 70.2, 90.0),
        integrated_turn_actions=True,
        straight_recovery_deg=90.0,
        goal_distance_min_m=40_000.0,
        goal_distance_max_m=80_000.0,
        bug_spawn_min_m=3_000.0,
        bug_spawn_max_m=10_000.0,
        threat_distance_scale_m=10_000.0,
        relative_speed_scale_mps=500.0,
        risk_t_cpa_s=45.0,
        risk_d_cpa_m=3_000.0,
        risk_distance_m=1_000.0,
        dangerous_spawn_only=True,
        dangerous_max_cpa_m=500.0,
        actor_risk_gate=True,
        terminate_on_evade=False,
        retire_threats_on_evade=True,
        maneuver_burst_limit_steps=2,
        progress_scale=1.5,
        hit_penalty=150.0,
        evade_bonus=25.0,
        arrival_bonus=150.0,
        timeout_penalty=100.0,
        deviation_penalty=0.02,
        oscillation_penalty=0.50,
        recovery_penalty=0.75,
        heading_error_penalty=1.0,
        excess_maneuver_penalty=3.0,
        recovery_bonus=20.0,
        recovery_heading_tolerance_deg=5.0,
        safety_improvement_scale=25.0,
        unsafe_straight_penalty=0.50,
        reward_clip=200.0,
        stage3_same_side_probability=0.15,
    )
    if strict_mission:
        config = replace(
            config,
            progress_scale=2.0,
            hit_penalty=200.0,
            evade_bonus=10.0,
            arrival_bonus=250.0,
            timeout_penalty=200.0,
            recovery_penalty=1.5,
            heading_error_penalty=2.0,
            excess_maneuver_penalty=5.0,
            recovery_bonus=30.0,
        )
    return config


# v10 PHASES[0]
PHASE = {
    "name": "single_recovery",
    "bug_spawn_min_m": 8_000.0,
    "bug_spawn_max_m": 16_000.0,
    "bug_speed_mps": 220.0,
    "stage_distribution": (0.0, 0.25, 0.75, 0.0),
    "hard_replay_fraction": 0.0,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=262_144)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--minibatch-size", type=int, default=2048)
    parser.add_argument("--ppo-epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--save-every-update", action="store_true",
                        help="write a checkpoint after every PPO update (calibration only; "
                             "the default keeps just the final one)")
    parser.add_argument("--strict-mission", action="store_true")
    parser.add_argument("--preserve-normalizers", action="store_true")
    parser.add_argument("--peer-track-probability", type=float, default=0.0)
    parser.add_argument("--satellite-track-probability", type=float, default=0.0)
    parser.add_argument("--eval-episodes", type=int, default=300)
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    history_path = args.out_dir / "training_history.jsonl"
    history = history_path.open("a", encoding="utf-8")

    policy, previous_config, _payload = LocalAvoidPPO.load(args.resume, device=args.device)
    policy.config.permutation_actor_coef = 0.02
    policy.config.permutation_critic_coef = 0.01
    policy.config.freeze_observation_normalizer = False
    policy.optimizer = torch.optim.Adam(policy.network.parameters(), lr=args.learning_rate)
    if not args.preserve_normalizers:
        policy.actor_rms = RunningMeanStd((policy.config.actor_obs_dim,))
        policy.critic_rms = RunningMeanStd((policy.config.critic_state_dim,))
    policy.reward_rms = RunningMeanStd(())
    policy.discounted_reward = np.zeros(args.num_envs, dtype=np.float64)
    policy.update_count = 0
    policy.total_steps = 0
    policy.stage = 3

    base = replace(
        base_config(args.strict_mission),
        encode_track_source=False,
        peer_track_probability=args.peer_track_probability,
        satellite_track_probability=args.satellite_track_probability,
    )
    full_config = replace(
        base,
        stage3_simultaneous_probability=0.34,
        stage3_pincer_probability=0.33,
    )
    phase_config = replace(
        base,
        bug_spawn_min_m=PHASE["bug_spawn_min_m"],
        bug_spawn_max_m=PHASE["bug_spawn_max_m"],
        bug_speed_mps=PHASE["bug_speed_mps"],
        threat_distance_scale_m=PHASE["bug_spawn_max_m"],
        stage3_simultaneous_probability=0.34,
        stage3_pincer_probability=0.33,
    )

    hard_count = round(args.num_envs * PHASE["hard_replay_fraction"])
    environment = VectorLocalAvoidEnv(
        args.num_envs,
        [full_config] * hard_count + [phase_config] * (args.num_envs - hard_count),
        stage=3,
        seed=args.seed,
        stage_distribution=PHASE["stage_distribution"],
    )
    actor_observation, critic_state = environment.reset()

    per_update = args.num_envs * args.rollout_steps
    total_updates = max(1, args.steps // per_update)
    print(
        f"LOCAL_CALIB_START steps={args.steps} per_update={per_update} "
        f"updates={total_updates} device={args.device} out={args.out_dir}",
        flush=True,
    )

    completed: list[dict] = []
    for update_index in range(1, total_updates + 1):
        rollout, actor_observation, critic_state, done = collect_rollout(
            policy, environment, actor_observation, critic_state
        )
        completed.extend(done)
        train_metrics = policy.update(rollout)

        skill = summarize_episodes(completed[-5000:])
        record = {
            "phase": PHASE["name"],
            "update": policy.update_count,
            "total_steps": policy.total_steps,
            "train": train_metrics,
            "skill": skill,
        }
        history.write(json.dumps(record) + "\n")
        history.flush()
        print(
            "LOCAL_CALIB_TRAIN " + json.dumps({
                "update": policy.update_count,
                "total_steps": policy.total_steps,
                "policy_loss": train_metrics.get("policy_loss"),
                "value_loss": train_metrics.get("value_loss"),
                "entropy": train_metrics.get("entropy"),
                "explained_variance": train_metrics.get("explained_variance"),
                "approx_kl": train_metrics.get("approx_kl"),
                "reward_mean": train_metrics.get("reward_mean"),
                "episodes_recent": skill.get("episodes"),
                "arrival_rate_recent": skill.get("arrival_rate"),
                "evade_rate_recent": skill.get("evade_success_rate"),
            }),
            flush=True,
        )
        if args.save_every_update:
            ckpt = args.out_dir / f"ckpt_update{update_index:03d}_steps{policy.total_steps}.pt"
            policy.save(ckpt, environment_config=full_config, metrics=record)
            print(f"LOCAL_CALIB_CKPT {ckpt}", flush=True)

    final_ckpt = args.out_dir / f"ckpt_final_steps{policy.total_steps}.pt"
    policy.save(final_ckpt, environment_config=full_config, metrics=record)
    print(f"LOCAL_CALIB_CKPT {final_ckpt}", flush=True)

    history.close()
    print(f"LOCAL_CALIB_DONE updates={policy.update_count} steps={policy.total_steps}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
